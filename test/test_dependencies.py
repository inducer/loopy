import sys
from contextlib import redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

import pyopencl as cl
from pyopencl.tools import (
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,  # noqa
)

import loopy as lp
from loopy.check import check_precise_variable_access_ordered
from loopy.diagnostic import VariableAccessNotOrdered
from loopy.kernel.dependency import (
    add_lexicographic_happens_after,
    reduce_strict_ordering,
)


def test_no_dependency():
    t_unit = lp.make_kernel(
        "{ [i,j] : 0 <= i, j < n}",
        """
        a[i,j] = 2*i        {id=S}
        b[i,j] = a[i+1,j+1] {id=T}
        """,
    )

    t_unit = add_lexicographic_happens_after(t_unit)
    t_unit = reduce_strict_ordering(t_unit)
    knl = t_unit.default_entrypoint

    assert len(knl.id_to_insn["S"].happens_after) == 0
    assert len(knl.id_to_insn["T"].happens_after) == 0


def test_odd_even_dependencies():
    t_unit = lp.make_kernel(
        "{ [i] : 0 <= i < np }",
        """
        u[2*i+1] = i {id=S}
        u[2*i] = i   {id=T}
        u[i] = i     {id=V}
        """
    )

    t_unit = add_lexicographic_happens_after(t_unit)
    t_unit = reduce_strict_ordering(t_unit)

    knl = t_unit.default_entrypoint
    assert "S" in knl.id_to_insn["V"].happens_after
    assert "T" in knl.id_to_insn["V"].happens_after
    for insn in knl.instructions:
        print(f"{insn.id}:")
        for insn_after, instances_rel in insn.happens_after.items():
            print(f"    {insn_after}: {instances_rel}")


def test_barrier_preserved_as_all_memory_access():
    t_unit = lp.make_kernel(
        "{ [i]: 0 <= i < n }",
        """
        tmp[i] = a[i]       {id=load}
        ... lbarrier        {id=barrier}
        out[i] = tmp[i]     {id=store}
        """,
        [
            lp.GlobalArg("a", shape=("n",), dtype=np.float64),
            lp.GlobalArg("out", shape=("n",), dtype=np.float64),
            lp.TemporaryVariable(
                "tmp", shape=("n",), dtype=np.float64,
                address_space=lp.AddressSpace.LOCAL),
        ],
    )

    t_unit = add_lexicographic_happens_after(t_unit)
    t_unit = reduce_strict_ordering(t_unit)
    knl = t_unit.default_entrypoint

    assert "load" in knl.id_to_insn["barrier"].happens_after
    assert "barrier" in knl.id_to_insn["store"].happens_after
    check_precise_variable_access_ordered(knl)


def test_precise_dependency_validation_catches_missing_dependency():
    t_unit = lp.make_kernel(
        "{ [i]: 0 <= i < n }",
        """
        tmp[i] = a[i]       {id=load}
        ... lbarrier        {id=barrier}
        out[i] = tmp[i]     {id=store}
        """,
        [
            lp.GlobalArg("a", shape=("n",), dtype=np.float64),
            lp.GlobalArg("out", shape=("n",), dtype=np.float64),
            lp.TemporaryVariable(
                "tmp", shape=("n",), dtype=np.float64,
                address_space=lp.AddressSpace.LOCAL),
        ],
    )

    t_unit = add_lexicographic_happens_after(t_unit)
    t_unit = reduce_strict_ordering(t_unit)
    knl = t_unit.default_entrypoint
    instructions = [
        insn.copy(happens_after={})
        if insn.id == "store" else insn
        for insn in knl.instructions
    ]
    knl = knl.copy(instructions=instructions)

    with pytest.raises(VariableAccessNotOrdered):
        check_precise_variable_access_ordered(knl)


def test_finite_difference_diamond_compute_example_codegen():
    example_path = (
        Path(__file__).parent.parent
        / "examples/python/compute-examples/finite-difference-diamond.py"
    )
    spec = spec_from_file_location("finite_difference_diamond", example_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    with redirect_stdout(StringIO()):
        module.main(
            ntime=24,
            nspace=128,
            stencil_width=5,
            time_block_size=4,
            space_block_size=32,
            use_compute=True,
            print_device_code=True,
            run_kernel=False,
        )


@pytest.mark.parametrize("img_size", [(512, 512), (1920, 1080)])
def test_3x3_blur(ctx_factory, img_size):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    hx, hy = img_size
    img = np.random.default_rng(seed=42).random(size=(hx, hy))

    knl = lp.make_kernel(
        "{ [x, y]: 0 <= x < hx and 0 <= y < hy }",
        """
        img_(i, j)  := img[i+1, j+1]
        blurx(i, j) := img_(i-1, j) + img_(i, j) + img_(i+1, j)

        out[x, y] = blurx(x, y-1) + blurx(x, y) + blurx(x, y+1)
        """,
        [
            lp.GlobalArg("out",
                         dtype=np.float64,
                         shape=(hx, hy),
                         is_output=True),
            lp.GlobalArg("img",
                         dtype=np.float64,
                         shape=(hx, hy))
        ]
    )

    knl = lp.fix_parameters(knl, hx=hx-2, hy=hy-2)

    knl = add_lexicographic_happens_after(knl)
    knl = reduce_strict_ordering(knl)

    bsize = 4
    knl = lp.split_iname(knl, "x", bsize, inner_tag="vec", outer_tag="for")
    knl = lp.split_iname(knl, "y", bsize, inner_tag="for", outer_tag="g.0")
    knl = lp.precompute(
        knl,
        "blurx",
        sweep_inames="x_inner, y_inner",
        precompute_outer_inames="x_outer, y_outer",
        precompute_inames="bx, by"
    )

    knl = lp.prioritize_loops(knl, "y_outer, x_outer, y_inner, x_inner")
    knl = lp.expand_subst(knl)

    _, out = knl(queue, img=img)
    blurx = np.zeros_like(img)
    out_np = np.zeros_like(img)
    for x in range(hx-2):
        blurx[x, :] = img[x, :] + img[x+1, :] + img[x+2, :]
    for y in range(hy-2):
        out_np[:, y] = blurx[:, y] + blurx[:, y+1] + blurx[:, y+2]

    import numpy.linalg as la
    assert (la.norm(out[0] - out_np) / la.norm(out_np)) <= 1e-14


def test_self_dependence():
    t_unit = lp.make_kernel(
        "[nt, nx] -> { [t, x]: 0 <= t < nt and 0 <= x < nx }",
        """
        u[t+2,x+1] = 2*u[t+1,x+1] {id=self}
        """
    )

    t_unit = add_lexicographic_happens_after(t_unit)
    t_unit = reduce_strict_ordering(t_unit)

    knl = t_unit.default_entrypoint
    assert "self" in knl.instructions[0].happens_after.keys()
    print(knl.id_to_insn["self"].happens_after["self"].instances_rel)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])

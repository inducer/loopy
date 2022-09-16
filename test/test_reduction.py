__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import sys
import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.clmath  # noqa
import pyopencl.clrandom  # noqa
import pytest

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

__all__ = [
        "pytest_generate_tests",
        "cl"  # 'cl.create_some_context'
        ]


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


def test_nonsense_reduction(ctx_factory):

    knl = lp.make_kernel(
            "{[i]: 0<=i<100}",
            """
                a[i] = sum(i, 2)
                """,
            [lp.GlobalArg("a", np.float32, shape=(100,))]
            )

    import pytest
    with pytest.raises(RuntimeError):
        knl = lp.preprocess_kernel(knl)


def test_empty_reduction(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            [
                "{[i]: 0<=i<20}",
                "[i] -> {[j]: 0<=j<0}"
                ],
            "a[i] = sum(j, j)",
            )

    knl = lp.preprocess_kernel(knl)
    print(knl)

    knl = lp.set_options(knl, write_code=True)
    evt, (a,) = knl(queue)

    assert (a.get() == 0).all()


def test_nested_dependent_reduction(ctx_factory):
    dtype = np.dtype(np.int32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            [
                "{[i]: 0<=i<n}",
                "{[j]: 0<=j<i+sumlen}"
                ],
            [
                "<> sumlen = ell[i]",
                "a[i] = sum(j, j)",
                ],
            [
                lp.ValueArg("n", np.int32),
                lp.GlobalArg("a", dtype, ("n",)),
                lp.GlobalArg("ell", np.int32, ("n",)),
                ])

    n = 330
    ell = np.arange(n, dtype=np.int32)
    evt, (a,) = knl(queue, ell=ell, n=n, out_host=True)

    tgt_result = (2*ell-1)*2*ell/2
    assert (a == tgt_result).all()


def test_multi_nested_dependent_reduction():
    dtype = np.dtype(np.int32)

    knl = lp.make_kernel(
            [
                "{[itgt]: 0 <= itgt < ntgts}",
                "{[isrc_box]: 0 <= isrc_box < nboxes}",
                "{[isrc]: 0 <= isrc < npart}"
                ],
            """
            for itgt
                for isrc_box
                    <> npart = nparticles_per_box[isrc_box]
                end
                a[itgt] = sum((isrc_box, isrc), 1)
            end
            """,
            [
                lp.ValueArg("n", np.int32),
                lp.GlobalArg("a", dtype, ("n",)),
                lp.GlobalArg("nparticles_per_box", np.int32, ("nboxes",)),
                lp.ValueArg("ntgts", np.int32),
                lp.ValueArg("nboxes", np.int32),
                ],
            assumptions="ntgts>=1",
            target=lp.PyOpenCLTarget())

    print(lp.generate_code_v2(knl).device_code())
    # FIXME: Actually test functionality.


def test_recursive_nested_dependent_reduction():
    dtype = np.dtype(np.int32)

    knl = lp.make_kernel(
            [
                "{[itgt]: 0 <= itgt < ntgts}",
                "{[isrc_box]: 0 <= isrc_box < nboxes}",
                "{[isrc]: 0 <= isrc < npart}"
                ],
            """
            for itgt
                for isrc_box
                    <> npart = nparticles_per_box[isrc_box]
                    <> boxsum = sum(isrc, isrc+isrc_box+itgt)
                end
                a[itgt] = sum(isrc_box, boxsum)
            end
            """,
            [
                lp.ValueArg("n", np.int32),
                lp.GlobalArg("a", dtype, ("n",)),
                lp.GlobalArg("nparticles_per_box", np.int32, ("nboxes",)),
                lp.ValueArg("ntgts", np.int32),
                lp.ValueArg("nboxes", np.int32),
                ],
            assumptions="ntgts>=1",
            target=lp.PyOpenCLTarget())

    print(lp.generate_code_v2(knl).device_code())
    # FIXME: Actually test functionality.


@pytest.mark.parametrize("size", [128, 5, 113, 67, 1])
def test_local_parallel_reduction(ctx_factory, size):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i, j]: 0 <= i < n and 0 <= j < 5}",
            """
            z[j] = sum(i, i+j)
            """)

    knl = lp.fix_parameters(knl, n=size)

    ref_knl = knl

    def variant0(knl):
        return lp.tag_inames(knl, "i:l.0")

    def variant1(knl):
        return lp.tag_inames(knl, "i:l.0,j:l.1")

    def variant2(knl):
        return lp.tag_inames(knl, "i:l.0,j:g.0")

    for variant in [
            variant0,
            variant1,
            variant2
            ]:
        knl = variant(ref_knl)

        lp.auto_test_vs_ref(ref_knl, ctx, knl)


@pytest.mark.parametrize("size", [1000])
def test_global_parallel_reduction(ctx_factory, size):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i]: 0 <= i < n }",
            """
            # Using z[0] instead of z works around a bug in ancient PyOpenCL.
            z[0] = sum(i, a[i])
            """)

    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})
    ref_knl = knl

    gsize = 128
    knl = lp.split_iname(knl, "i", gsize * 20)
    knl = lp.split_iname(knl, "i_inner", gsize, inner_tag="l.0")
    knl = lp.split_reduction_outward(knl, "i_outer")
    knl = lp.split_reduction_inward(knl, "i_inner_outer")
    from loopy.transform.data import reduction_arg_to_subst_rule
    knl = reduction_arg_to_subst_rule(knl, "i_outer")

    knl = lp.precompute(knl, "red_i_outer_arg", "i_outer",
            temporary_address_space=lp.AddressSpace.GLOBAL,
            default_tag="l.auto")
    knl = lp.realize_reduction(knl)
    knl = lp.tag_inames(knl, "i_outer_0:g.0")

    # Keep the i_outer accumulator on the  correct (lower) side of the barrier,
    # otherwise there will be useless save/reload code generated.
    knl = lp.add_dependency(
            knl, "writes:acc_i_outer",
            "id:red_i_outer_arg_barrier")

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl, parameters={"n": size},
            print_ref_code=True)


@pytest.mark.parametrize("size", [1000])
def test_global_mc_parallel_reduction(ctx_factory, size):
    ctx = ctx_factory()

    import pyopencl.version  # noqa
    if cl.version.VERSION < (2016, 2):
        pytest.skip("Random123 RNG not supported in PyOpenCL < 2016.2")

    knl = lp.make_kernel(
            "{[i]: 0 <= i < n }",
            """
            for i
                <> key = make_uint2(i, 324830944)  {inames=i}
                <> ctr = make_uint4(0, 1, 2, 3)  {inames=i,id=init_ctr}
                <> vals, ctr = philox4x32_f32(ctr, key)  {dep=init_ctr}
            end
            z = sum(i, vals.s0 + vals.s1 + vals.s2 + vals.s3)
            """)

    ref_knl = knl
    ref_knl = lp.add_dtypes(ref_knl, {"n": np.int32})

    gsize = 128
    knl = lp.split_iname(knl, "i", gsize * 20)
    knl = lp.split_iname(knl, "i_inner", gsize, outer_tag="l.0")
    knl = lp.split_reduction_inward(knl, "i_inner_inner")
    knl = lp.split_reduction_inward(knl, "i_inner_outer")
    from loopy.transform.data import reduction_arg_to_subst_rule
    knl = reduction_arg_to_subst_rule(knl, "i_outer")
    knl = lp.precompute(knl, "red_i_outer_arg", "i_outer",
            temporary_address_space=lp.AddressSpace.GLOBAL,
            default_tag="l.auto")
    knl = lp.preprocess_kernel(knl)
    knl = lp.add_dependency(
            knl, "writes:acc_i_outer",
            "id:red_i_outer_arg_barrier")

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl, parameters={"n": size})


def test_argmax(ctx_factory):
    logging.basicConfig(level=logging.INFO)

    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 10000

    knl = lp.make_kernel(
            "{[i]: 0<=i<%d}" % n,
            """
            max_val, max_idx = argmax(i, abs(a[i]), i)
            """)

    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})
    print(lp.preprocess_kernel(knl))
    knl = lp.set_options(knl, write_code=True, allow_terminal_colors=True)

    a = np.random.randn(10000).astype(dtype)
    evt, (max_idx, max_val) = knl(queue, a=a, out_host=True)
    assert max_val == np.max(np.abs(a))
    assert max_idx == np.where(np.abs(a) == max_val)[-1]


def test_simul_reduce(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 20

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n }",
            [
                "a = simul_reduce(sum, (i,j), i*j)",
                "b = simul_reduce(sum, i, simul_reduce(sum, j, i*j))",
                ],
            assumptions="n>=1")

    evt, (a, b) = knl(queue, n=n)

    ref = sum(i*j for i in range(n) for j in range(n))
    assert a.get() == ref
    assert b.get() == ref


@pytest.mark.parametrize(("op_name", "np_op"), [
    ("sum", np.sum),
    ("product", np.prod),
    ("min", np.min),
    ("max", np.max),
    ])
def test_reduction_library(ctx_factory, op_name, np_op):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{[i,j]: 0<=i<n and 0<=j<m }",
            [
                "res[i] = reduce(%s, j, a[i,j])" % op_name,
                ],
            assumptions="n>=1")

    a = np.random.randn(20, 10)
    evt, (res,) = knl(queue, a=a)

    assert np.allclose(res, np_op(a, axis=1))


def test_split_reduction(ctx_factory):
    knl = lp.make_kernel(
            "{[i,j,k]: 0<=i,j,k<n}",
            """
                b = sum((i,j,k), a[i,j,k])
                """,
            [
                lp.GlobalArg("box_source_starts,box_source_counts_nonchild,a",
                    None, shape=None),
                "..."])

    knl = lp.split_reduction_outward(knl, "j,k")
    # FIXME: finish test


def test_double_sum_made_unique(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 20

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n }",
            [
                "a = sum((i,j), i*j)",
                "b = sum(i, sum(j, i*j))",
                ],
            assumptions="n>=1")

    knl = lp.make_reduction_inames_unique(knl)
    print(knl)

    evt, (a, b) = knl(queue, n=n)

    ref = sum(i*j for i in range(n) for j in range(n))
    assert a.get() == ref
    assert b.get() == ref


def test_parallel_multi_output_reduction(ctx_factory):
    knl = lp.make_kernel(
                "{[i]: 0<=i<128}",
                """
                max_val, max_indices = argmax(i, abs(a[i]), i)
                """)
    knl = lp.tag_inames(knl, dict(i="l.0"))
    knl = lp.add_dtypes(knl, dict(a=np.float64))

    ctx = ctx_factory()

    with cl.CommandQueue(ctx) as queue:
        a = np.random.rand(128)
        out, (max_index, max_val) = knl(queue, a=a)

        assert max_val == np.max(a)
        assert max_index == np.argmax(np.abs(a))


def test_reduction_with_conditional():
    # The purpose of the 'l' iname is to force the entire kernel (including the
    # predicate) into device code.

    knl = lp.make_kernel(
                "{ [l,i] : 0<=l,i<42 }",
                """
                if l > 0
                    b[l] = sum(i, l*a[i])
                end
                """,
                [lp.ValueArg("n", dtype=np.int32), "..."])

    knl = lp.tag_inames(knl, "l:g.0")
    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})
    code = lp.generate_code_v2(knl).device_code()
    print(code)

    # Check that the if appears before the loop that realizes the reduction.
    assert code.index("if") < code.index("for")


def test_any_all(ctx_factory):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{[i, j]: 0<=i,j<10}",
        """
        out1 = reduce(any, [i], i == 4)
        out2 = reduce(all, [j], j == 5)
        """)
    knl = lp.set_options(knl, return_dict=True)

    _, out_dict = knl(cq)

    assert out_dict["out1"].get()
    assert not out_dict["out2"].get()


def test_reduction_without_inames(ctx_factory):
    """Ensure that reductions with no inames get rewritten to the element
    being reduced over. This was sometimes erroneously eliminated because
    reduction realization used the generation of new statements as a criterion
    for whether work was done.
    """
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{:}",
            """
            out = reduce(any, [], 5)
            """)
    knl = lp.set_options(knl, return_dict=True)

    _, out_dict = knl(cq)

    assert out_dict["out"].get() == 5


def test_reduction_in_conditional(ctx_factory):
    # https://github.com/inducer/loopy/issues/533#issuecomment-1028472366
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{[i, j, k]: 0<=i,j,k<10}",
        """
        y[i] = 1729 if (sum(j, j) == 0) else sum(k, k)
        """)

    knl = lp.set_options(knl, write_code=True)

    knl = lp.preprocess_program(knl)

    evt, (out,) = knl(cq)

    assert (out == 45).all()


def test_realize_reduction_insn_id_filter_list(ctx_factory):
    ctx = ctx_factory()

    t_unit = lp.make_kernel(
        "{[i, j, k]: 0<=i,j,k<10}",
        """
        a = sum(i, 8*i)    {id=w_a}
        b = sum(j, j*j)    {id=w_b}
        c = sum(k, sin(3.14*k)) {id=w_c}
        """)
    ref_t_unit = t_unit

    knl = t_unit.default_entrypoint
    assert knl.id_to_insn["w_a"].reduction_inames() == frozenset({"i"})
    assert knl.id_to_insn["w_b"].reduction_inames() == frozenset({"j"})
    assert knl.id_to_insn["w_c"].reduction_inames() == frozenset({"k"})

    t_unit = lp.realize_reduction(t_unit, insn_id_filter=["w_a", "w_b"])

    knl = t_unit.default_entrypoint
    assert knl.id_to_insn["w_a"].reduction_inames() == frozenset()
    assert knl.id_to_insn["w_b"].reduction_inames() == frozenset()
    assert knl.id_to_insn["w_c"].reduction_inames() == frozenset({"k"})

    lp.auto_test_vs_ref(t_unit, ctx, ref_t_unit)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

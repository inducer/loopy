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
from pytools.tag import Tag

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
        "cl"  # "cl.create_some_context"
        ]


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


@pytest.mark.parametrize("fix_parameters", (True, False))
def test_chunk_iname(ctx_factory, fix_parameters):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = 2*a[i]",
            [
                lp.GlobalArg("out,a", np.float32, shape=lp.auto),
                "..."
                ],
            assumptions="n>0")

    ref_knl = knl
    knl = lp.chunk_iname(knl, "i", 3, inner_tag="l.0")
    knl = lp.prioritize_loops(knl, "i_outer, i_inner")

    if fix_parameters:
        ref_knl = lp.fix_parameters(ref_knl, n=130)
        knl = lp.fix_parameters(knl, n=130)
        lp.auto_test_vs_ref(ref_knl, ctx, knl)
    else:
        lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters={"n": 130})


def test_collect_common_factors(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n}",
            """
            <float32> out_tmp = 0 {id=out_init,inames=i}
            out_tmp = out_tmp + alpha[i]*a[i,j]*b1[j] {id=out_up1,dep=out_init}
            out_tmp = out_tmp + alpha[i]*a[j,i]*b2[j] \
                    {id=out_up2,dep=out_init,nosync=out_up1}
            out[i] = out_tmp {dep=out_up1:out_up2}
            """)
    knl = lp.add_and_infer_dtypes(knl,
            dict(a=np.float32, alpha=np.float32, b1=np.float32, b2=np.float32))

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 256, outer_tag="g.0", inner_tag="l.0")
    knl = lp.collect_common_factors_on_increment(knl, "out_tmp")

    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=13))


def test_to_batched(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
         """ { [i,j]: 0<=i,j<n } """,
         """ out[i] = sum(j, a[i,j]*x[j])""")
    knl = lp.add_and_infer_dtypes(knl, dict(out=np.float32,
                                            x=np.float32,
                                            a=np.float32))

    bknl = lp.to_batched(knl, "nbatches", "out,x")

    ref_knl = lp.make_kernel(
         """ { [i,j,k]: 0<=i,j<n and 0<=k<nbatches} """,
         """out[k, i] = sum(j, a[i,j]*x[k, j])""")
    ref_knl = lp.add_and_infer_dtypes(ref_knl, dict(out=np.float32,
                                                    x=np.float32,
                                                    a=np.float32))

    a = np.random.randn(5, 5).astype(np.float32)
    x = np.random.randn(7, 5).astype(np.float32)

    # Running both the kernels
    evt, (out1, ) = bknl(queue, a=a, x=x, n=5, nbatches=7)
    evt, (out2, ) = ref_knl(queue, a=a, x=x, n=5, nbatches=7)

    # checking that the outputs are same
    assert np.linalg.norm(out1-out2) < 1e-15


def test_to_batched_temp(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
         """ { [i,j]: 0<=i,j<n } """,
         """ cnst = 2.0
         out[i] = sum(j, cnst*a[i,j]*x[j])""",
         [lp.TemporaryVariable(
             "cnst",
             dtype=np.float32,
             shape=(),
             address_space=lp.AddressSpace.PRIVATE), "..."])
    knl = lp.add_and_infer_dtypes(knl, dict(out=np.float32,
                                            x=np.float32,
                                            a=np.float32))
    ref_knl = lp.make_kernel(
         """ { [i,j]: 0<=i,j<n } """,
         """out[i] = sum(j, 2.0*a[i,j]*x[j])""")
    ref_knl = lp.add_and_infer_dtypes(ref_knl, dict(out=np.float32,
                                                    x=np.float32,
                                                    a=np.float32))

    bknl = lp.to_batched(knl, "nbatches", "out,x")
    bref_knl = lp.to_batched(ref_knl, "nbatches", "out,x")

    # checking that cnst is not being bathced
    assert bknl["loopy_kernel"].temporary_variables["cnst"].shape == ()

    a = np.random.randn(5, 5)
    x = np.random.randn(7, 5)

    # Checking that the program compiles and the logic is correct
    lp.auto_test_vs_ref(
            bref_knl, ctx, bknl,
            parameters=dict(a=a, x=x, n=5, nbatches=7))


def test_add_barrier(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    knl = lp.make_kernel(
            "{[i, j, ii, jj]: 0<=i,j, ii, jj<n}",
            """
            out[j, i] = a[i, j]{id=transpose}
            out[ii, jj] = 2*out[ii, jj]{id=double}
            """,
            [
                lp.GlobalArg("out", is_input=False, shape=lp.auto),
                ...
            ]
    )
    a = np.random.randn(16, 16)
    knl = lp.add_barrier(knl, "id:transpose", "id:double", "gb1")

    knl = lp.split_iname(knl, "i", 2, outer_tag="g.0", inner_tag="l.0")
    knl = lp.split_iname(knl, "j", 2, outer_tag="g.1", inner_tag="l.1")
    knl = lp.split_iname(knl, "ii", 2, outer_tag="g.0", inner_tag="l.0")
    knl = lp.split_iname(knl, "jj", 2, outer_tag="g.1", inner_tag="l.1")

    evt, (out,) = knl(queue, a=a)
    assert (np.linalg.norm(out-2*a.T) < 1e-16)


def test_rename_argument(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    kernel = lp.make_kernel(
         """{ [i]: 0<=i<n }""",
         """out[i] = a + 2""")

    kernel = lp.rename_argument(kernel, "a", "b")

    evt, (out,) = kernel(queue, b=np.float32(12), n=20)

    assert (np.abs(out.get() - 14) < 1e-8).all()


def test_fusion():
    exp_kernel = lp.make_kernel(
         """ { [i]: 0<=i<n } """,
         """ exp[i] = pow(E, z[i])""",
         assumptions="n>0")

    sum_kernel = lp.make_kernel(
        "{ [j]: 0<=j<n }",
        "out2 = sum(j, exp[j])",
        assumptions="n>0")

    knl = lp.fuse_kernels([exp_kernel, sum_kernel])

    print(knl)


def test_alias_temporaries(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        """
        times2(i) := 2*a[i]
        times3(i) := 3*a[i]
        times4(i) := 4*a[i]

        x[i] = times2(i)
        y[i] = times3(i)
        z[i] = times4(i)
        """)

    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 16, outer_tag="g.0", inner_tag="l.0")

    knl = lp.precompute(knl, "times2", "i_inner", default_tag="l.auto")
    knl = lp.precompute(knl, "times3", "i_inner", default_tag="l.auto")
    knl = lp.precompute(knl, "times4", "i_inner", default_tag="l.auto")

    knl = lp.alias_temporaries(knl, ["times2_0", "times3_0", "times4_0"])

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
            parameters=dict(n=30))


def test_vectorize(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        """
        <> temp = 2*b[i]
        a[i] = temp
        """)
    knl = lp.add_and_infer_dtypes(knl, dict(b=np.float32))
    knl = lp.set_array_axis_names(knl, "a,b", "i")
    knl = lp.split_array_dim(knl, [("a", 0), ("b", 0)], 4,
            split_kwargs=dict(slabs=(0, 1)))

    knl = lp.tag_array_axes(knl, "a,b", "c,vec")
    ref_knl = knl
    ref_knl = lp.tag_inames(ref_knl, {"i_inner": "unr"})

    knl = lp.tag_inames(knl, {"i_inner": "vec"})

    knl = lp.preprocess_kernel(knl)
    code, inf = lp.generate_code(knl)

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
            parameters=dict(n=30))


def test_extract_subst(ctx_factory):
    prog = lp.make_kernel(
            "{[i]: 0<=i<n}",
            """
                a[i] = 23*b[i]**2 + 25*b[i]**2
                """, name="extract_subst")

    prog = lp.extract_subst(prog, "bsquare", "alpha*b[i]**2", "alpha")

    print(prog)

    from loopy.symbolic import parse

    insn, = prog["extract_subst"].instructions
    assert insn.expression == parse("bsquare(23) + bsquare(25)")


def test_join_inames(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<16}",
            [
                "b[i,j] = 2*a[i,j]"
                ],
            [
                lp.GlobalArg("a", np.float32, shape=(16, 16,)),
                lp.GlobalArg("b", np.float32, shape=(16, 16,))
                ],
            )

    ref_knl = knl

    knl = lp.add_prefetch(knl, "a", sweep_inames=["i", "j"], default_tag="l.auto")
    knl = lp.join_inames(knl, ["a_dim_0", "a_dim_1"])

    lp.auto_test_vs_ref(ref_knl, ctx, knl, print_ref_code=True)


def test_tag_data_axes(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{ [i,j,k]: 0<=i,j,k<n }",
            "out[i,j,k] = 15")

    ref_knl = knl

    with pytest.raises(lp.LoopyError):
        lp.tag_array_axes(knl, "out", "N1,N0,N5")

    with pytest.raises(lp.LoopyError):
        lp.tag_array_axes(knl, "out", "N1,N0,c")

    knl = lp.tag_array_axes(knl, "out", "N1,N0,N2")
    knl = lp.tag_inames(knl, dict(j="g.0", i="g.1"))

    lp.auto_test_vs_ref(ref_knl, ctx, knl,
            parameters=dict(n=20))


def test_set_arg_order():
    knl = lp.make_kernel(
            "{ [i,j]: 0<=i,j<n }",
            "out[i,j] = a[i]*b[j]")

    knl = lp.set_argument_order(knl, "out,a,n,b")


def test_affine_map_inames():
    knl = lp.make_kernel(
        "{[e, i,j,n]: 0<=e<E and 0<=i,j,n<N}",
        "rhsQ[e, n+i, j] = rhsQ[e, n+i, j] - D[i, n]*x[i,j]")

    knl = lp.affine_map_inames(knl,
            "i", "i0",
            "i0 = n+i")

    print(knl)


def test_precompute_confusing_subst_arguments(ctx_factory):
    ctx = ctx_factory()

    prog = lp.make_kernel(
        "{[i,j]: 0<=i<n and 0<=j<5}",
        """
        D(i):=a[i+1]-a[i]
        b[i,j] = D(j)
        """, name="precomputer")

    prog = lp.add_and_infer_dtypes(prog, dict(a=np.float32))

    ref_prog = prog

    prog = lp.tag_inames(prog, dict(j="g.1"))
    prog = lp.split_iname(prog, "i", 128, outer_tag="g.0", inner_tag="l.0")

    from loopy.symbolic import get_dependencies
    assert "i_inner" not in get_dependencies(
            prog["precomputer"].substitutions["D"].expression)
    prog = lp.precompute(prog, "D", sweep_inames="j",
            precompute_outer_inames="j, i_inner, i_outer")

    lp.auto_test_vs_ref(
            ref_prog, ctx, prog,
            parameters=dict(n=12345))


def test_precompute_nested_subst(ctx_factory):
    ctx = ctx_factory()

    prog = lp.make_kernel(
        "{[i]: 0<=i<n}",
        """
        E:=a[i]
        D:=E*E
        b[i] = D
        """, name="precomputer")

    prog = lp.add_and_infer_dtypes(prog, dict(a=np.float32))

    ref_prog = prog

    prog = lp.split_iname(prog, "i", 128, outer_tag="g.0", inner_tag="l.0")

    from loopy.symbolic import get_dependencies
    assert "i_inner" not in get_dependencies(
            prog["precomputer"].substitutions["D"].expression)
    prog = lp.precompute(prog, "D", "i_inner", default_tag="l.auto")

    # There's only one surviving 'E' rule.
    assert len([
        rule_name
        for rule_name in prog["precomputer"].substitutions
        if rule_name.startswith("E")]) == 1

    # That rule should use the newly created prefetch inames,
    # not the prior 'i_inner'
    assert "i_inner" not in get_dependencies(
            prog["precomputer"].substitutions["E"].expression)

    lp.auto_test_vs_ref(
            ref_prog, ctx, prog,
            parameters=dict(n=12345))


def test_precompute_with_preexisting_inames(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[e,i,j,k]: 0<=e<E and 0<=i,j,k<n}",
        """
        result[e,i] = sum(j, D1[i,j]*u[e,j])
        result2[e,i] = sum(k, D2[i,k]*u[e,k])
        """)

    knl = lp.add_and_infer_dtypes(knl, {
        "u": np.float32,
        "D1": np.float32,
        "D2": np.float32,
        })

    knl = lp.fix_parameters(knl, n=13)

    ref_knl = knl

    knl = lp.extract_subst(knl, "D1_subst", "D1[ii,jj]", parameters="ii,jj")
    knl = lp.extract_subst(knl, "D2_subst", "D2[ii,jj]", parameters="ii,jj")

    knl = lp.precompute(knl, "D1_subst", "i,j", default_tag="for",
            precompute_inames="ii,jj")
    knl = lp.precompute(knl, "D2_subst", "i,k", default_tag="for",
            precompute_inames="ii,jj")

    knl = lp.prioritize_loops(knl, "ii,jj,e,j,k")

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
            parameters=dict(E=200))


def test_precompute_with_preexisting_inames_fail():
    knl = lp.make_kernel(
        "{[e,i,j,k]: 0<=e<E and 0<=i,j<n and 0<=k<2*n}",
        """
        result[e,i] = sum(j, D1[i,j]*u[e,j])
        result2[e,i] = sum(k, D2[i,k]*u[e,k])
        """)

    knl = lp.add_and_infer_dtypes(knl, {
        "u": np.float32,
        "D1": np.float32,
        "D2": np.float32,
        })

    knl = lp.fix_parameters(knl, n=13)

    knl = lp.extract_subst(knl, "D1_subst", "D1[ii,jj]", parameters="ii,jj")
    knl = lp.extract_subst(knl, "D2_subst", "D2[ii,jj]", parameters="ii,jj")

    knl = lp.precompute(knl, "D1_subst", "i,j", default_tag="for",
            precompute_inames="ii,jj")
    with pytest.raises(lp.LoopyError):
        lp.precompute(knl, "D2_subst", "i,k", default_tag="for",
                precompute_inames="ii,jj")


def test_add_nosync():
    orig_prog = lp.make_kernel("{[i]: 0<=i<10}",
        """
        <>tmp[i] = 10 {id=insn1}
        <>tmp2[i] = 10 {id=insn2}

        <>tmp3[2*i] = 0 {id=insn3}
        <>tmp4 = 1 + tmp3[2*i] {id=insn4}

        <>tmp5[i] = 0 {id=insn5,groups=g1}
        tmp5[i] = 1 {id=insn6,conflicts=g1}
        """, name="nosync")

    orig_prog = lp.set_temporary_scope(orig_prog, "tmp3", "local")
    orig_prog = lp.set_temporary_scope(orig_prog, "tmp5", "local")

    # No dependency present - don't add nosync
    prog = lp.add_nosync(orig_prog, "any", "writes:tmp", "writes:tmp2",
            empty_ok=True)
    assert frozenset() == (
            prog["nosync"].id_to_insn["insn2"].no_sync_with)

    # Dependency present
    prog = lp.add_nosync(orig_prog, "local", "writes:tmp3", "reads:tmp3")
    assert frozenset() == (
            prog["nosync"].id_to_insn["insn3"].no_sync_with)
    assert frozenset([("insn3", "local")]) == (
            prog["nosync"].id_to_insn["insn4"].no_sync_with)

    # Bidirectional
    prog = lp.add_nosync(
            orig_prog, "local", "writes:tmp3", "reads:tmp3", bidirectional=True)
    assert frozenset([("insn4", "local")]) == (
            prog["nosync"].id_to_insn["insn3"].no_sync_with)
    assert frozenset([("insn3", "local")]) == (
            prog["nosync"].id_to_insn["insn4"].no_sync_with)

    # Groups
    prog = lp.add_nosync(orig_prog, "local", "insn5", "insn6")
    assert frozenset([("insn5", "local")]) == (
            prog["nosync"].id_to_insn["insn6"].no_sync_with)


def test_uniquify_instruction_ids():
    i1 = lp.Assignment("b", 1, id=None)
    i2 = lp.Assignment("b", 1, id=None)
    i3 = lp.Assignment("b", 1, id=lp.UniqueName("b"))
    i4 = lp.Assignment("b", 1, id=lp.UniqueName("b"))

    prog = lp.make_kernel("{[i]: i = 1}", [], name="lpy_knl")
    new_root_kernel = prog["lpy_knl"].copy(instructions=[i1, i2, i3, i4])
    prog = prog.with_kernel(new_root_kernel)

    from loopy.transform.instruction import uniquify_instruction_ids
    prog = uniquify_instruction_ids(prog)

    insn_ids = {insn.id for insn in prog["lpy_knl"].instructions}

    assert len(insn_ids) == 4
    assert all(isinstance(id, str) for id in insn_ids)


def test_split_iname_only_if_in_within():
    prog = lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            c[i] = 3*d[i] {id=to_split}
            a[i] = 2*b[i] {id=not_to_split}
            """, name="splitter")

    prog = lp.split_iname(prog, "i", 4, within="id:to_split")

    for insn in prog["splitter"].instructions:
        if insn.id == "to_split":
            assert insn.within_inames == frozenset({"i_outer", "i_inner"})
        if insn.id == "not_to_split":
            assert insn.within_inames == frozenset({"i"})


def test_nested_substs_in_insns(ctx_factory):
    ctx = ctx_factory()

    ref_prg = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
        a(x) := 2 * x
        b(x) := x**2
        c(x) := 7 * x
        f[i] = c(b(a(i)))
        """
    )

    t_unit = lp.expand_subst(ref_prg)
    assert not any(
            cknl.subkernel.substitutions
            for cknl in t_unit.callables_table.values())

    lp.auto_test_vs_ref(ref_prg, ctx, t_unit)


# {{{ test_map_domain_vs_split_iname

def test_map_domain_vs_split_iname():

    # {{{ Make kernel

    knl = lp.make_kernel(
        [
            "[nx,nt] -> {[x, t]: 0 <= x < nx and 0 <= t < nt}",
            "[ni] -> {[i]: 0 <= i < ni}",
        ],
        """
        a[x,t] = b[x,t]  {id=stmta}
        c[x,t] = d[x,t]  {id=stmtc}
        e[i] = f[i]
        """,
        lang_version=(2018, 2),
        )
    knl = lp.add_and_infer_dtypes(knl, {"b,d,f": np.float32})
    ref_knl = knl

    # }}}

    # {{{ Apply domain change mapping

    knl_map_dom = ref_knl  # loop priority goes away, deps stay

    # Create map_domain mapping:
    import islpy as isl
    transform_map = isl.BasicMap(
        "[nt] -> {[t] -> [t_outer, t_inner]: "
        "0 <= t_inner < 32 and "
        "32*t_outer + t_inner = t and "
        "0 <= 32*t_outer + t_inner < nt}")

    # Call map_domain to transform kernel
    knl_map_dom = lp.map_domain(knl_map_dom, transform_map)

    # Prioritize loops (prio should eventually be updated in map_domain?)
    knl_map_dom = lp.prioritize_loops(knl_map_dom, "x, t_outer, t_inner")

    # Get a linearization
    proc_knl_map_dom = lp.preprocess_kernel(knl_map_dom)
    lin_knl_map_dom = lp.get_one_linearized_kernel(
        proc_knl_map_dom["loopy_kernel"], proc_knl_map_dom.callables_table)

    # }}}

    # {{{ Split iname and see if we get the same result

    knl_split_iname = ref_knl
    knl_split_iname = lp.split_iname(knl_split_iname, "t", 32)
    knl_split_iname = lp.prioritize_loops(knl_split_iname, "x, t_outer, t_inner")
    proc_knl_split_iname = lp.preprocess_kernel(knl_split_iname)
    lin_knl_split_iname = lp.get_one_linearized_kernel(
        proc_knl_split_iname["loopy_kernel"], proc_knl_split_iname.callables_table)

    from loopy.schedule.checker.utils import (
        ensure_dim_names_match_and_align,
    )
    for d_map_domain, d_split_iname in zip(
            knl_map_dom["loopy_kernel"].domains,
            knl_split_iname["loopy_kernel"].domains):
        d_map_domain_aligned = ensure_dim_names_match_and_align(
            d_map_domain, d_split_iname)
        assert d_map_domain_aligned == d_split_iname

    for litem_map_domain, litem_split_iname in zip(
            lin_knl_map_dom.linearization, lin_knl_split_iname.linearization):
        assert litem_map_domain == litem_split_iname

    # Can't easily compare instructions because equivalent subscript
    # expressions may have different orders

    # }}}

# }}}


# {{{ test_map_domain_with_transform_map_missing_dims

def test_map_domain_with_transform_map_missing_dims():
    # Make sure map_domain works correctly when the mapping doesn't include
    # all the dims in the domain.

    # {{{ Make kernel

    knl = lp.make_kernel(
        [
            "[nx,nt] -> {[x, y, z, t]: 0 <= x,y,z < nx and 0 <= t < nt}",
        ],
        """
        a[y,x,t,z] = b[y,x,t,z]  {id=stmta}
        """,
        lang_version=(2018, 2),
        )
    knl = lp.add_and_infer_dtypes(knl, {"b": np.float32})
    ref_knl = knl

    # }}}

    # {{{ Apply domain change mapping

    knl_map_dom = ref_knl  # loop priority goes away, deps stay

    # Create map_domain mapping that only includes t and y
    # (x and z should be unaffected)
    import islpy as isl
    transform_map = isl.BasicMap(
        "[nx,nt] -> {[t, y] -> [t_outer, t_inner, y_new]: "
        "0 <= t_inner < 32 and "
        "32*t_outer + t_inner = t and "
        "0 <= 32*t_outer + t_inner < nt and "
        "y = y_new"
        "}")

    # Call map_domain to transform kernel
    knl_map_dom = lp.map_domain(knl_map_dom, transform_map)

    # Prioritize loops (prio should eventually be updated in map_domain?)
    try:
        # Use constrain_loop_nesting if it's available
        desired_prio = "x, t_outer, t_inner, z, y_new"
        knl_map_dom = lp.constrain_loop_nesting(knl_map_dom, desired_prio)
    except AttributeError:
        # For some reason, prioritize_loops can't handle the ordering above
        # when linearizing knl_split_iname below
        desired_prio = "z, y_new, x, t_outer, t_inner"
        knl_map_dom = lp.prioritize_loops(knl_map_dom, desired_prio)

    # Get a linearization
    proc_knl_map_dom = lp.preprocess_kernel(knl_map_dom)
    lin_knl_map_dom = lp.get_one_linearized_kernel(
        proc_knl_map_dom["loopy_kernel"], proc_knl_map_dom.callables_table)

    # }}}

    # {{{ Split iname and see if we get the same result

    knl_split_iname = ref_knl
    knl_split_iname = lp.split_iname(knl_split_iname, "t", 32)
    knl_split_iname = lp.rename_iname(knl_split_iname, "y", "y_new")
    try:
        # Use constrain_loop_nesting if it's available
        knl_split_iname = lp.constrain_loop_nesting(knl_split_iname, desired_prio)
    except AttributeError:
        knl_split_iname = lp.prioritize_loops(knl_split_iname, desired_prio)
    proc_knl_split_iname = lp.preprocess_kernel(knl_split_iname)
    lin_knl_split_iname = lp.get_one_linearized_kernel(
        proc_knl_split_iname["loopy_kernel"], proc_knl_split_iname.callables_table)

    from loopy.schedule.checker.utils import (
        ensure_dim_names_match_and_align,
    )
    for d_map_domain, d_split_iname in zip(
            knl_map_dom["loopy_kernel"].domains,
            knl_split_iname["loopy_kernel"].domains):
        d_map_domain_aligned = ensure_dim_names_match_and_align(
            d_map_domain, d_split_iname)
        assert d_map_domain_aligned == d_split_iname

    for litem_map_domain, litem_split_iname in zip(
            lin_knl_map_dom.linearization, lin_knl_split_iname.linearization):
        assert litem_map_domain == litem_split_iname

    # Can't easily compare instructions because equivalent subscript
    # expressions may have different orders

    # }}}

# }}}


def test_diamond_tiling(ctx_factory, interactive=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    ref_knl = lp.make_kernel(
        "[nx,nt] -> {[ix, it]: 1<=ix<nx-1 and 0<=it<nt}",
        """
        u[ix, it+2] = (
            2*u[ix, it+1]
            + dt**2/dx**2 * (u[ix+1, it+1] - 2*u[ix, it+1] + u[ix-1, it+1])
            - u[ix, it])
        """)

    # FIXME: Handle priorities in map_domain
    knl_for_transform = ref_knl

    ref_knl = lp.prioritize_loops(ref_knl, "it, ix")

    import islpy as isl
    m = isl.BasicMap(
        "[nx,nt] -> {[ix, it] -> [tx, tt, tparity, itt, itx]: "
        "16*(tx - tt) + itx - itt = ix - it and "
        "16*(tx + tt + tparity) + itt + itx = ix + it and "
        "0<=tparity<2 and 0 <= itx - itt < 16 and 0 <= itt+itx < 16}")
    knl = lp.map_domain(knl_for_transform, m)
    knl = lp.prioritize_loops(knl, "tt,tparity,tx,itt,itx")

    if interactive:
        nx = 43
        u = np.zeros((nx, 200))
        x = np.linspace(-1, 1, nx)
        dx = x[1] - x[0]
        u[:, 0] = u[:, 1] = np.exp(-100*x**2)

        u_dev = cl.array.to_device(queue, u)
        knl(queue, u=u_dev, dx=dx, dt=dx)

        u = u_dev.get()
        import matplotlib.pyplot as plt
        plt.imshow(u.T)
        plt.show()
    else:
        types = {"dt,dx,u": np.float64}
        knl = lp.add_and_infer_dtypes(knl, types)
        ref_knl = lp.add_and_infer_dtypes(ref_knl, types)

        lp.auto_test_vs_ref(ref_knl, ctx, knl,
                parameters={
                    "nx": 200, "nt": 300,
                    "dx": 1, "dt": 1
                    })


def test_extract_subst_with_iname_deps_in_templ(ctx_factory):
    knl = lp.make_kernel(
            "{[i, j, k]: 0<=i<100 and 0<=j,k<5}",
            """
            y[i, j, k] = x[i, j, k]
            """,
            [lp.GlobalArg("x,y", shape=lp.auto, dtype=float)],
            lang_version=(2018, 2))

    knl = lp.extract_subst(knl, "rule1", "x[i, arg1, arg2]",
            parameters=("arg1", "arg2"))

    lp.auto_test_vs_ref(knl, ctx_factory(), knl)


def test_prefetch_local_into_private():
    # https://gitlab.tiker.net/inducer/loopy/-/issues/210
    n = 32
    m = 32
    n_vecs = 32

    knl = lp.make_kernel(
        """{[k,i,j]:
            0<=k<n_vecs and
            0<=i<m and
            0<=j<n}""",
        """
        result[i,k] = sum(j, mat[i, j] * vec[j, k])
        """,
        kernel_data=[
            lp.GlobalArg("result", np.float32, shape=(m, n_vecs), order="C"),
            lp.GlobalArg("mat", np.float32, shape=(m, n), order="C"),
            lp.GlobalArg("vec", np.float32, shape=(n, n_vecs), order="C")
        ],
        assumptions="n > 0 \
                     and m > 0 \
                     and n_vecs > 0",
        name="mxm"
    )

    knl = lp.fix_parameters(knl, m=m, n=n, n_vecs=n_vecs)
    knl = lp.prioritize_loops(knl, "i,k,j")

    knl = lp.add_prefetch(
            knl, "mat", "i, j", temporary_name="s_mat", default_tag="for")
    knl = lp.add_prefetch(
            knl, "s_mat", "j", temporary_name="p_mat", default_tag="for")


def test_add_inames_for_unused_hw_axes(ctx_factory):
    ctx = ctx_factory()
    dtype = np.float32
    order = "F"

    n = 16**3

    knl = lp.make_kernel(
            "[n] -> {[i,j]: 0<=i,j<n}",
            [
                """
                <> alpha = 2.0 {id=init_alpha}
                for i
                  for j
                    c[i, j] = alpha*a[i]*b[j] {id=outerproduct}
                  end
                end
                """
                ],
            [
                lp.GlobalArg("a", dtype, shape=("n",), order=order),
                lp.GlobalArg("b", dtype, shape=("n",), order=order),
                lp.GlobalArg("c", dtype, shape=("n, n"), order=order),
                lp.ValueArg("n", np.int32, approximately=n),
                ],
            name="rank_one",
            assumptions="n >= 16",
            lang_version=(2018, 2))

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 16,
            outer_tag="g.0", inner_tag="l.0")
    knl = lp.split_iname(knl, "j", 16,
            outer_tag="g.1", inner_tag="l.1")

    knl = lp.add_prefetch(knl, "a")
    knl = lp.add_prefetch(knl, "b")

    knl = lp.add_inames_for_unused_hw_axes(knl)

    assert (knl["rank_one"].id_to_insn["init_alpha"].within_inames
            == frozenset(["i_inner", "i_outer", "j_outer", "j_inner"]))
    assert (knl["rank_one"].id_to_insn["a_fetch_rule"].within_inames
            == frozenset(["i_inner", "i_outer", "j_outer", "j_inner"]))
    assert (knl["rank_one"].id_to_insn["b_fetch_rule"].within_inames
            == frozenset(["i_inner", "i_outer", "j_outer", "j_inner"]))

    lp.auto_test_vs_ref(ref_knl, ctx, knl,
            op_count=[np.dtype(dtype).itemsize*n**2/1e9], op_label=["GBytes"],
            parameters={"n": n})


def test_rename_argument_of_domain_params(ctx_factory):
    knl = lp.make_kernel(
            "{[i, j]: 0<=i<n and 0<=j<m}",
            """
            y[i, j] = 2.0f
            """)

    knl = lp.rename_argument(knl, "n", "N")
    knl = lp.rename_argument(knl, "m", "M")

    # renamed variables should not appear in the code
    code_str = lp.generate_code_v2(knl).device_code()
    assert code_str.find("int const n") == -1
    assert code_str.find("int const m") == -1
    assert code_str.find("int const N") != -1
    assert code_str.find("int const M") != -1

    lp.auto_test_vs_ref(knl, ctx_factory(), knl, parameters={"M": 10, "N": 4})


def test_rename_argument_with_auto_stride(ctx_factory):
    from loopy.kernel.array import FixedStrideArrayDimTag

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            y[i] = x[i]
            """, [lp.GlobalArg("x", dtype=float,
                               shape=lp.auto,
                               dim_tags=[FixedStrideArrayDimTag(lp.auto)]), ...])

    knl = lp.rename_argument(knl, "x", "x_new")

    code_str = lp.generate_code_v2(knl).device_code()
    assert code_str.find("double const *__restrict__ x_new,") != -1
    assert code_str.find("double const *__restrict__ x,") == -1

    evt, (out, ) = knl(queue, x_new=np.random.rand(10))


def test_rename_argument_with_assumptions():
    import islpy as isl
    knl = lp.make_kernel(
            "{[i]: 0<=i<n_old}",
            """
            y[i] = 2.0f
            """)
    knl = lp.assume(knl, "n_old=10")

    knl = lp.rename_argument(knl, "n_old", "n_new")
    assumptions = knl["loopy_kernel"].assumptions

    assert "n_old" not in assumptions.get_var_dict()
    assert "n_new" in assumptions.get_var_dict()
    assert (
            (assumptions & isl.BasicSet("[n_new]->{: n_new=10}"))
            == assumptions)


def test_tag_iname_with_match_pattern():
    knl = lp.make_kernel(
            "{[i0, i1]: 0<=i0, i1<n}",
            """
            x[i0] = 2.0f
            y[i1] = 2.0f
            """)

    knl = lp.tag_inames(knl, "i*:unr")
    knl = knl["loopy_kernel"]
    i0_tag, = knl.inames["i0"].tags
    i1_tag, = knl.inames["i1"].tags

    assert str(i0_tag) == "unr"
    assert str(i1_tag) == "unr"


# {{{ custom iname tags

class ElementLoopTag(Tag):
    def __str__(self):
        return "iel"


class DOFLoopTag(Tag):
    def __str__(self):
        return "idof"


def test_custom_iname_tag():
    t_unit = lp.make_kernel(
            "{[ifuzz0, ifuzz1]: 0<=ifuzz0<100 and 0<=ifuzz1<32}",
            """
            out_dofs[ifuzz0, ifuzz1] = 2*in_dofs[ifuzz0, ifuzz1]
            """)
    t_unit = lp.add_and_infer_dtypes(t_unit, {"in_dofs": np.float64})
    t_unit = lp.tag_inames(t_unit,
            {"ifuzz0": ElementLoopTag(), "ifuzz1": DOFLoopTag()})

    knl = t_unit.default_entrypoint
    ifuzz0_tag, = knl.inames["ifuzz0"].tags
    ifuzz1_tag, = knl.inames["ifuzz1"].tags

    assert str(ifuzz0_tag) == "iel"
    assert str(ifuzz1_tag) == "idof"

    lp.generate_code_v2(t_unit)

    t_unit = lp.tag_inames(t_unit, {"ifuzz0": "g.0", "ifuzz1": "l.0"})
    assert len(t_unit.default_entrypoint.inames["ifuzz0"].tags) == 2

    lp.generate_code_v2(t_unit)

# }}}


def test_remove_instructions_with_recursive_deps():
    t_unit = lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            y[i] = 0 {id=insn0}
            a[i] = 2*b[i] {id=insn1}
            c[i] = 2*b[i] {id=insn2}
            y[i] = y[i] + x[i] {id=insn3}
            """, seq_dependencies=True, name="myknl")

    knl = lp.remove_instructions(t_unit, {"insn1", "insn2"})["myknl"]

    assert knl.id_to_insn["insn3"].depends_on == frozenset(["insn0"])
    assert knl.id_to_insn["insn0"].depends_on == frozenset()


def test_prefetch_with_within(ctx_factory):
    t_unit = lp.make_kernel(
            ["{[j]: 0<=j<256}",
             "{[i, k]: 0<=i<100 and 0<=k<128}"],
            """
            f[j] = 3.14 * j {id=set_f}
            f[j] = 2 * f[j] {id=update_f, nosync=set_f}
            ... gbarrier {id=insn_gbar}
            y[i, k] = f[k] * x[i, k] {id=set_y}
            """, [lp.GlobalArg("x", shape=lp.auto, dtype=float), ...],
            seq_dependencies=True,
            name="myknl")

    ref_t_unit = t_unit

    t_unit = lp.split_iname(t_unit, "j", 32, inner_tag="l.0", outer_tag="g.0")
    t_unit = lp.split_iname(t_unit, "i", 32, inner_tag="l.0", outer_tag="g.0")

    t_unit = lp.add_prefetch(t_unit, "f", prefetch_insn_id="f_prftch",
                             within="id:set_y", sweep_inames="k",
                             dim_arg_names="iprftch", default_tag=None,
                             temporary_address_space=lp.AddressSpace.LOCAL,
                             temporary_name="foo",
                             fetch_outer_inames=frozenset({"i_outer"}))
    t_unit = lp.add_dependency(t_unit, "id:f_prftch", "id:insn_gbar")
    t_unit = lp.split_iname(t_unit, "iprftch", 32, inner_tag="l.0")

    # test that 'f' is only prefetched in set_y
    assert t_unit["myknl"].temporary_variables["foo"].shape == (128,)

    lp.auto_test_vs_ref(ref_t_unit, ctx_factory(), t_unit)


def test_privatize_with_nonzero_lbound(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{[j]:10<=j<14}",
        """
        for j
            <> tmp = j
            out[j] = tmp
        end
        """,
        name="arange_10_to_14",
        seq_dependencies=True)

    knl = lp.privatize_temporaries_with_inames(knl, {"j"})
    assert knl["arange_10_to_14"].temporary_variables["tmp"].shape == (4,)
    _, (out, ) = knl(queue)
    np.testing.assert_allclose(out.get()[10:14], np.arange(10, 14))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

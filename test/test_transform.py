from __future__ import division, absolute_import, print_function

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
        "cl"  # "cl.create_some_context"
        ]


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


def test_chunk_iname(ctx_factory):
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
    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=130))


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
    assert bknl.temporary_variables["cnst"].shape == ()

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
            """)
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
    knl = lp.set_array_dim_names(knl, "a,b", "i")
    knl = lp.split_array_dim(knl, [("a", 0), ("b", 0)], 4,
            split_kwargs=dict(slabs=(0, 1)))

    knl = lp.tag_data_axes(knl, "a,b", "c,vec")
    ref_knl = knl
    ref_knl = lp.tag_inames(ref_knl, {"i_inner": "unr"})

    knl = lp.tag_inames(knl, {"i_inner": "vec"})

    knl = lp.preprocess_kernel(knl)
    knl = lp.get_one_scheduled_kernel(knl)
    code, inf = lp.generate_code(knl)

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
            parameters=dict(n=30))


def test_extract_subst(ctx_factory):
    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            """
                a[i] = 23*b[i]**2 + 25*b[i]**2
                """)

    knl = lp.extract_subst(knl, "bsquare", "alpha*b[i]**2", "alpha")

    print(knl)

    from loopy.symbolic import parse

    insn, = knl.instructions
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
        lp.tag_data_axes(knl, "out", "N1,N0,N5")

    with pytest.raises(lp.LoopyError):
        lp.tag_data_axes(knl, "out", "N1,N0,c")

    knl = lp.tag_data_axes(knl, "out", "N1,N0,N2")
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

    knl = lp.make_kernel(
        "{[i,j]: 0<=i<n and 0<=j<5}",
        """
        D(i):=a[i+1]-a[i]
        b[i,j] = D(j)
        """)

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32))

    ref_knl = knl

    knl = lp.tag_inames(knl, dict(j="g.1"))
    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    from loopy.symbolic import get_dependencies
    assert "i_inner" not in get_dependencies(knl.substitutions["D"].expression)
    knl = lp.precompute(knl, "D", sweep_inames="j",
            precompute_outer_inames="j, i_inner, i_outer")

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
            parameters=dict(n=12345))


def test_precompute_nested_subst(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        """
        E:=a[i]
        D:=E*E
        b[i] = D
        """)

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32))

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    from loopy.symbolic import get_dependencies
    assert "i_inner" not in get_dependencies(knl.substitutions["D"].expression)
    knl = lp.precompute(knl, "D", "i_inner", default_tag="l.auto")

    # There's only one surviving 'E' rule.
    assert len([
        rule_name
        for rule_name in knl.substitutions
        if rule_name.startswith("E")]) == 1

    # That rule should use the newly created prefetch inames,
    # not the prior 'i_inner'
    assert "i_inner" not in get_dependencies(knl.substitutions["E"].expression)

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
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
    orig_knl = lp.make_kernel("{[i]: 0<=i<10}",
        """
        <>tmp[i] = 10 {id=insn1}
        <>tmp2[i] = 10 {id=insn2}

        <>tmp3[2*i] = 0 {id=insn3}
        <>tmp4 = 1 + tmp3[2*i] {id=insn4}

        <>tmp5[i] = 0 {id=insn5,groups=g1}
        tmp5[i] = 1 {id=insn6,conflicts=g1}
        """)

    orig_knl = lp.set_temporary_scope(orig_knl, "tmp3", "local")
    orig_knl = lp.set_temporary_scope(orig_knl, "tmp5", "local")

    # No dependency present - don't add nosync
    knl = lp.add_nosync(orig_knl, "any", "writes:tmp", "writes:tmp2",
            empty_ok=True)
    assert frozenset() == knl.id_to_insn["insn2"].no_sync_with

    # Dependency present
    knl = lp.add_nosync(orig_knl, "local", "writes:tmp3", "reads:tmp3")
    assert frozenset() == knl.id_to_insn["insn3"].no_sync_with
    assert frozenset([("insn3", "local")]) == knl.id_to_insn["insn4"].no_sync_with

    # Bidirectional
    knl = lp.add_nosync(
            orig_knl, "local", "writes:tmp3", "reads:tmp3", bidirectional=True)
    assert frozenset([("insn4", "local")]) == knl.id_to_insn["insn3"].no_sync_with
    assert frozenset([("insn3", "local")]) == knl.id_to_insn["insn4"].no_sync_with

    # Groups
    knl = lp.add_nosync(orig_knl, "local", "insn5", "insn6")
    assert frozenset([("insn5", "local")]) == knl.id_to_insn["insn6"].no_sync_with


def test_uniquify_instruction_ids():
    i1 = lp.Assignment("b", 1, id=None)
    i2 = lp.Assignment("b", 1, id=None)
    i3 = lp.Assignment("b", 1, id=lp.UniqueName("b"))
    i4 = lp.Assignment("b", 1, id=lp.UniqueName("b"))

    knl = lp.make_kernel("{[i]: i = 1}", []).copy(instructions=[i1, i2, i3, i4])

    from loopy.transform.instruction import uniquify_instruction_ids
    knl = uniquify_instruction_ids(knl)

    insn_ids = set(insn.id for insn in knl.instructions)

    assert len(insn_ids) == 4
    assert all(isinstance(id, str) for id in insn_ids)


def test_split_iname_only_if_in_within():
    knl = lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            c[i] = 3*d[i] {id=to_split}
            a[i] = 2*b[i] {id=not_to_split}
            """)

    knl = lp.split_iname(knl, "i", 4, within="id:to_split")

    for insn in knl.instructions:
        if insn.id == "to_split":
            assert insn.within_inames == frozenset({"i_outer", "i_inner"})
        if insn.id == "not_to_split":
            assert insn.within_inames == frozenset({"i"})


def test_nested_substs_in_insns(ctx_factory):
    ctx = ctx_factory()
    import loopy as lp

    ref_knl = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
        a(x) := 2 * x
        b(x) := x**2
        c(x) := 7 * x
        f[i] = c(b(a(i)))
        """
    )

    knl = lp.expand_subst(ref_knl)
    assert not knl.substitutions

    lp.auto_test_vs_ref(ref_knl, ctx, knl)


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

    assert knl.id_to_insn["init_alpha"].within_inames == frozenset(["i_inner",
        "i_outer", "j_outer", "j_inner"])
    assert knl.id_to_insn["a_fetch_rule"].within_inames == frozenset(["i_inner",
        "i_outer", "j_outer", "j_inner"])
    assert knl.id_to_insn["b_fetch_rule"].within_inames == frozenset(["i_inner",
        "i_outer", "j_outer", "j_inner"])

    lp.auto_test_vs_ref(ref_knl, ctx, knl,
            op_count=[np.dtype(dtype).itemsize*n**2/1e9], op_label=["GBytes"],
            parameters={"n": n})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

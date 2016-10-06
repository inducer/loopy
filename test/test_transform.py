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
        "cl"  # 'cl.create_some_context'
        ]


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
    knl = lp.set_loop_priority(knl, "i_outer, i_inner")
    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=130))


def test_collect_common_factors(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j,k]: 0<=i,j<n}",
            """
            <float32> out_tmp = 0 {id=out_init,inames=i}
            out_tmp = out_tmp + alpha[i]*a[i,j]*b1[j] {id=out_up1,dep=out_init}
            out_tmp = out_tmp + alpha[i]*a[j,i]*b2[j] {id=out_up2,dep=out_init}
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
         ''' { [i,j]: 0<=i,j<n } ''',
         ''' out[i] = sum(j, a[i,j]*x[j])''')

    bknl = lp.to_batched(knl, "nbatches", "out,x")

    a = np.random.randn(5, 5)
    x = np.random.randn(7, 5)

    bknl(queue, a=a, x=x)


def test_rename_argument(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    kernel = lp.make_kernel(
         '''{ [i]: 0<=i<n }''',
         '''out[i] = a + 2''')

    kernel = lp.rename_argument(kernel, "a", "b")

    evt, (out,) = kernel(queue, b=np.float32(12), n=20)

    assert (np.abs(out.get() - 14) < 1e-8).all()


def test_fusion():
    exp_kernel = lp.make_kernel(
         ''' { [i]: 0<=i<n } ''',
         ''' exp[i] = pow(E, z[i])''',
         assumptions="n>0")

    sum_kernel = lp.make_kernel(
        '{ [j]: 0<=j<n }',
        'out2 = sum(j, exp[j])',
        assumptions='n>0')

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

    knl = lp.precompute(knl, "times2", "i_inner")
    knl = lp.precompute(knl, "times3", "i_inner")
    knl = lp.precompute(knl, "times4", "i_inner")

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

    knl = lp.add_prefetch(knl, "a", sweep_inames=["i", "j"])
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
    knl = lp.precompute(knl, "D")

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
            parameters=dict(n=12345))


def test_precompute_nested_subst(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[i,j]: 0<=i<n and 0<=j<5}",
        """
        E:=a[i]
        D:=E*E
        b[i] = D
        """)

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32))

    ref_knl = knl

    knl = lp.tag_inames(knl, dict(j="g.1"))
    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    from loopy.symbolic import get_dependencies
    assert "i_inner" not in get_dependencies(knl.substitutions["D"].expression)
    knl = lp.precompute(knl, "D", "i_inner")

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

    knl = lp.set_loop_priority(knl, "ii,jj,e,j,k")

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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: foldmethod=marker

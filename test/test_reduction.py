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


def test_nonsense_reduction(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i]: 0<=i<100}",
            """
                a[i] = sum(i, 2)
                """,
            [lp.GlobalArg("a", np.float32, shape=(100,))]
            )

    import pytest
    with pytest.raises(RuntimeError):
        knl = lp.preprocess_kernel(knl, ctx.devices[0])


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

    knl = lp.realize_reduction(knl)
    print(knl)

    knl = lp.set_options(knl, write_cl=True)
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
                "<> sumlen = l[i]",
                "a[i] = sum(j, j)",
                ],
            [
                lp.ValueArg("n", np.int32),
                lp.GlobalArg("a", dtype, ("n",)),
                lp.GlobalArg("l", np.int32, ("n",)),
                ])

    cknl = lp.CompiledKernel(ctx, knl)

    n = 330
    l = np.arange(n, dtype=np.int32)
    evt, (a,) = cknl(queue, l=l, n=n, out_host=True)

    tgt_result = (2*l-1)*2*l/2
    assert (a == tgt_result).all()


def test_multi_nested_dependent_reduction(ctx_factory):
    dtype = np.dtype(np.int32)
    ctx = ctx_factory()

    knl = lp.make_kernel(
            [
                "{[itgt]: 0 <= itgt < ntgts}",
                "{[isrc_box]: 0 <= isrc_box < nboxes}",
                "{[isrc]: 0 <= isrc < npart}"
                ],
            [
                "<> npart = nparticles_per_box[isrc_box]",
                "a[itgt] = sum((isrc_box, isrc), 1)",
                ],
            [
                lp.ValueArg("n", np.int32),
                lp.GlobalArg("a", dtype, ("n",)),
                lp.GlobalArg("nparticles_per_box", np.int32, ("nboxes",)),
                lp.ValueArg("ntgts", np.int32),
                lp.ValueArg("nboxes", np.int32),
                ],
            assumptions="ntgts>=1")

    cknl = lp.CompiledKernel(ctx, knl)
    print(cknl.get_code())
    # FIXME: Actually test functionality.


def test_recursive_nested_dependent_reduction(ctx_factory):
    dtype = np.dtype(np.int32)
    ctx = ctx_factory()

    knl = lp.make_kernel(
            [
                "{[itgt]: 0 <= itgt < ntgts}",
                "{[isrc_box]: 0 <= isrc_box < nboxes}",
                "{[isrc]: 0 <= isrc < npart}"
                ],
            [
                "<> npart = nparticles_per_box[isrc_box]",
                "<> boxsum = sum(isrc, isrc+isrc_box+itgt)",
                "a[itgt] = sum(isrc_box, boxsum)",
                ],
            [
                lp.ValueArg("n", np.int32),
                lp.GlobalArg("a", dtype, ("n",)),
                lp.GlobalArg("nparticles_per_box", np.int32, ("nboxes",)),
                lp.ValueArg("ntgts", np.int32),
                lp.ValueArg("nboxes", np.int32),
                ],
            assumptions="ntgts>=1")

    cknl = lp.CompiledKernel(ctx, knl)
    print(cknl.get_code())
    # FIXME: Actually test functionality.


@pytest.mark.parametrize("size", [128, 5, 113, 67])
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


# FIXME: Make me a test
@pytest.mark.parametrize("size", [10000])
def no_test_global_parallel_reduction(ctx_factory, size):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{[i]: 0 <= i < n }",
            """
            <> key = make_uint2(i, 324830944)  {inames=i}
            <> ctr = make_uint4(0, 1, 2, 3)  {inames=i,id=init_ctr}
            <> vals, ctr = philox4x32_f32(ctr, key)  {dep=init_ctr}
            z = sum(i, vals.s0 + vals.s1 + vals.s2 + vals.s3)
            """)

    # ref_knl = knl

    gsize = 128
    knl = lp.split_iname(knl, "i", gsize * 20)
    knl = lp.split_iname(knl, "i_inner", gsize, outer_tag="l.0")
    knl = lp.split_reduction_inward(knl, "i_inner_inner")
    knl = lp.split_reduction_inward(knl, "i_inner_outer")
    from loopy.transform.data import reduction_arg_to_subst_rule
    knl = reduction_arg_to_subst_rule(knl, "i_outer")
    knl = lp.precompute(knl, "red_i_outer_arg", "i_outer")
    print(knl)
    1/0
    knl = lp.realize_reduction(knl)

    evt, (z,) = knl(queue, n=size)

    #lp.auto_test_vs_ref(ref_knl, ctx, knl)


@pytest.mark.parametrize("size", [10000])
def test_global_parallel_reduction_simpler(ctx_factory, size):
    ctx = ctx_factory()

    pytest.xfail("very sensitive to kernel ordering, fails unused hw-axis check")

    knl = lp.make_kernel(
            "{[l,g,j]: 0 <= l < nl and 0 <= g,j < ng}",
            """
            <> key = make_uint2(l+nl*g, 1234)  {inames=l:g}
            <> ctr = make_uint4(0, 1, 2, 3)  {inames=l:g,id=init_ctr}
            <> vals, ctr = philox4x32_f32(ctr, key)  {dep=init_ctr}

            <> tmp[g] = sum(l, vals.s0 + 1j*vals.s1 + vals.s2 + 1j*vals.s3)

            result = sum(j, tmp[j])
            """)

    ng = 50
    knl = lp.fix_parameters(knl, ng=ng)

    knl = lp.set_options(knl, write_cl=True)

    ref_knl = knl

    knl = lp.split_iname(knl, "l", 128, inner_tag="l.0")
    knl = lp.split_reduction_outward(knl, "l_inner")
    knl = lp.tag_inames(knl, "g:g.0,j:l.0")

    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters={"nl": size})


def test_argmax(ctx_factory):
    logging.basicConfig(level=logging.INFO)

    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 10000

    knl = lp.make_kernel(
            "{[i]: 0<=i<%d}" % n,
            """
            max_val, max_idx = argmax(i, fabs(a[i]))
            """)

    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})
    print(lp.preprocess_kernel(knl))
    knl = lp.set_options(knl, write_cl=True, highlight_cl=True)

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


def test_fd_demo():
    knl = lp.make_kernel(
        "{[i,j]: 0<=i,j<n}",
        "result[i+1,j+1] = u[i + 1, j + 1]**2 + -1 + (-4)*u[i + 1, j + 1] \
                + u[i + 1 + 1, j + 1] + u[i + 1 + -1, j + 1] \
                + u[i + 1, j + 1 + 1] + u[i + 1, j + 1 + -1]")
    #assumptions="n mod 16=0")
    knl = lp.split_iname(knl,
            "i", 16, outer_tag="g.1", inner_tag="l.1")
    knl = lp.split_iname(knl,
            "j", 16, outer_tag="g.0", inner_tag="l.0")
    knl = lp.add_prefetch(knl, "u",
            ["i_inner", "j_inner"],
            fetch_bounding_box=True)

    #n = 1000
    #u = cl.clrandom.rand(queue, (n+2, n+2), dtype=np.float32)

    knl = lp.set_options(knl, write_cl=True)
    knl = lp.add_and_infer_dtypes(knl, dict(u=np.float32))
    code, inf = lp.generate_code(knl)
    print(code)

    assert "double" not in code


def test_fd_1d(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        "result[i] = u[i+1]-u[i]")

    knl = lp.add_and_infer_dtypes(knl, {"u": np.float32})
    ref_knl = knl

    knl = lp.split_iname(knl, "i", 16)
    knl = lp.extract_subst(knl, "u_acc", "u[j]", parameters="j")
    knl = lp.precompute(knl, "u_acc", "i_inner", default_tag="for")
    knl = lp.assume(knl, "n mod 16 = 0")

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
            parameters=dict(n=2048))


def test_poisson_fem(ctx_factory):
    # Stolen from Peter Coogan and Rob Kirby for FEM assembly
    ctx = ctx_factory()

    nbf = 5
    nqp = 5
    sdim = 3

    knl = lp.make_kernel(
            "{ [c,i,j,k,ell,ell2,ell3]: \
            0 <= c < nels and \
            0 <= i < nbf and \
            0 <= j < nbf and \
            0 <= k < nqp and \
            0 <= ell,ell2 < sdim}",
            """
            dpsi(bf,k0,dir) := \
                    simul_reduce(sum, ell2, DFinv[c,ell2,dir] * DPsi[bf,k0,ell2] )
            Ael[c,i,j] = \
                    J[c] * w[k] * sum(ell, dpsi(i,k,ell) * dpsi(j,k,ell))
            """,
            assumptions="nels>=1 and nbf >= 1 and nels mod 4 = 0")

    print(knl)

    knl = lp.fix_parameters(knl, nbf=nbf, sdim=sdim, nqp=nqp)

    ref_knl = knl

    knl = lp.set_loop_priority(knl, ["c", "j", "i", "k"])

    def variant_1(knl):
        knl = lp.precompute(knl, "dpsi", "i,k,ell", default_tag='for')
        knl = lp.set_loop_priority(knl, "c,i,j")
        return knl

    def variant_2(knl):
        knl = lp.precompute(knl, "dpsi", "i,ell", default_tag='for')
        knl = lp.set_loop_priority(knl, "c,i,j")
        return knl

    def add_types(knl):
        return lp.add_and_infer_dtypes(knl, dict(
            w=np.float32,
            J=np.float32,
            DPsi=np.float32,
            DFinv=np.float32,
            ))

    for variant in [
            #variant_1,
            variant_2
            ]:
        knl = variant(knl)

        lp.auto_test_vs_ref(
                add_types(ref_knl), ctx, add_types(knl),
                parameters=dict(n=5, nels=15, nbf=5, sdim=2, nqp=7))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: foldmethod=marker

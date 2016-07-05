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


# {{{ convolutions

def test_convolution(ctx_factory):
    ctx = ctx_factory()

    dtype = np.float32

    knl = lp.make_kernel(
        "{ [iimg, ifeat, icolor, im_x, im_y, f_x, f_y]: \
                -f_w <= f_x,f_y <= f_w \
                and 0 <= im_x < im_w and 0 <= im_y < im_h \
                and 0<=iimg<=nimgs and 0<=ifeat<nfeats and 0<=icolor<ncolors \
                }",
        """
        out[iimg, ifeat, im_x, im_y] = sum((f_x, f_y, icolor), \
            img[iimg, f_w+im_x-f_x, f_w+im_y-f_y, icolor] \
            * f[ifeat, f_w+f_x, f_w+f_y, icolor])
        """,
        [
            lp.GlobalArg("f", dtype, shape=lp.auto),
            lp.GlobalArg("img", dtype, shape=lp.auto),
            lp.GlobalArg("out", dtype, shape=lp.auto),
            "..."
            ],
        assumptions="f_w>=1 and im_w, im_h >= 2*f_w+1 and nfeats>=1 and nimgs>=0",
        options="annotate_inames")

    f_w = 3

    knl = lp.fix_parameters(knl, f_w=f_w, ncolors=3)

    ref_knl = knl

    def variant_0(knl):
        #knl = lp.split_iname(knl, "im_x", 16, inner_tag="l.0")
        knl = lp.set_loop_priority(knl, "iimg,im_x,im_y,ifeat,f_x,f_y")
        return knl

    def variant_1(knl):
        knl = lp.split_iname(knl, "im_x", 16, inner_tag="l.0")
        knl = lp.set_loop_priority(knl, "iimg,im_x_outer,im_y,ifeat,f_x,f_y")
        return knl

    def variant_2(knl):
        knl = lp.split_iname(knl, "im_x", 16, outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_iname(knl, "im_y", 16, outer_tag="g.1", inner_tag="l.1")
        knl = lp.tag_inames(knl, dict(ifeat="g.2"))
        knl = lp.add_prefetch(knl, "f[ifeat,:,:,:]")
        knl = lp.add_prefetch(knl, "img", "im_x_inner, im_y_inner, f_x, f_y")
        return knl

    for variant in [
            #variant_0,
            #variant_1,
            variant_2
            ]:
        lp.auto_test_vs_ref(ref_knl, ctx, variant(knl),
                parameters=dict(
                    im_w=128, im_h=128, f_w=f_w,
                    nfeats=3, nimgs=3
                    ))


def test_convolution_with_nonzero_base(ctx_factory):
    # This is kept alive as a test for domains that don't start at zero.
    # These are a bad idea for split_iname, which places its origin at zero
    # and therefore produces a first block that is odd-sized.
    #
    # Therefore, for real tests, check test_convolution further up.

    ctx = ctx_factory()

    dtype = np.float32

    knl = lp.make_kernel(
        "{ [iimg, ifeat, icolor, im_x, im_y, f_x, f_y]: \
                -f_w <= f_x,f_y <= f_w \
                and f_w <= im_x < im_w-f_w and f_w <= im_y < im_h-f_w \
                and 0<=iimg<=nimgs and 0<=ifeat<nfeats and 0<=icolor<ncolors \
                }",
        """
        out[iimg, ifeat, im_x-f_w, im_y-f_w] = sum((f_x, f_y, icolor), \
            img[iimg, im_x-f_x, im_y-f_y, icolor] \
            * f[ifeat, f_w+f_x, f_w+f_y, icolor])
        """,
        [
            lp.GlobalArg("f", dtype, shape=lp.auto),
            lp.GlobalArg("img", dtype, shape=lp.auto),
            lp.GlobalArg("out", dtype, shape=lp.auto),
            "..."
            ],
        assumptions="f_w>=1 and im_w, im_h >= 2*f_w+1 and nfeats>=1 and nimgs>=0",
        options="annotate_inames")

    knl = lp.fix_parameters(knl, ncolors=3)

    ref_knl = knl

    f_w = 3

    def variant_0(knl):
        #knl = lp.split_iname(knl, "im_x", 16, inner_tag="l.0")
        knl = lp.set_loop_priority(knl, "iimg,im_x,im_y,ifeat,f_x,f_y")
        return knl

    def variant_1(knl):
        knl = lp.split_iname(knl, "im_x", 16, inner_tag="l.0")
        knl = lp.set_loop_priority(knl, "iimg,im_x_outer,im_y,ifeat,f_x,f_y")
        return knl

    for variant in [
            variant_0,
            variant_1,
            ]:
        lp.auto_test_vs_ref(ref_knl, ctx, variant(knl),
                parameters=dict(
                    im_w=128, im_h=128, f_w=f_w,
                    nfeats=12, nimgs=17
                    ))

# }}}


def test_rob_stroud_bernstein(ctx_factory):
    ctx = ctx_factory()

    # NOTE: tmp would have to be zero-filled beforehand

    knl = lp.make_kernel(
            "{[el, i2, alpha1,alpha2]: \
                    0 <= el < nels and \
                    0 <= i2 < nqp1d and \
                    0 <= alpha1 <= deg and 0 <= alpha2 <= deg-alpha1 }",
            """
            for el,i2
                <> xi = qpts[1, i2]
                <> s = 1-xi
                <> r = xi/s
                <> aind = 0 {id=aind_init}

                for alpha1
                    <> w = s**(deg-alpha1) {id=init_w}

                    for alpha2
                        tmp[el,alpha1,i2] = tmp[el,alpha1,i2] + w * coeffs[aind] \
                                {id=write_tmp}
                        w = w * r * ( deg - alpha1 - alpha2 ) / (1 + alpha2) \
                                {id=update_w,dep=init_w:write_tmp}
                        aind = aind + 1 \
                                {id=aind_incr,dep=aind_init:write_tmp:update_w}
                    end
                end
            end
            """,
            [
                # Must declare coeffs to have "no" shape, to keep loopy
                # from trying to figure it out the shape automatically.

                lp.GlobalArg("coeffs", None, shape=None),
                "..."
                ],
            assumptions="deg>=0 and nels>=1"
            )

    knl = lp.fix_parameters(knl, nqp1d=7, deg=4)
    knl = lp.split_iname(knl, "el", 16, inner_tag="l.0")
    knl = lp.split_iname(knl, "el_outer", 2, outer_tag="g.0", inner_tag="ilp",
            slabs=(0, 1))
    knl = lp.tag_inames(knl, dict(i2="l.1", alpha1="unr", alpha2="unr"))

    print(lp.CompiledKernel(ctx, knl).get_highlighted_code(
            dict(
                qpts=np.float32,
                coeffs=np.float32,
                tmp=np.float32,
                )))


def test_rob_stroud_bernstein_full(ctx_factory):
    #logging.basicConfig(level=logging.DEBUG)
    ctx = ctx_factory()

    # NOTE: result would have to be zero-filled beforehand

    knl = lp.make_kernel(
        "{[el, i2, alpha1,alpha2, i1_2, alpha1_2, i2_2]: \
                0 <= el < nels and \
                0 <= i2 < nqp1d and \
                0 <= alpha1 <= deg and 0 <= alpha2 <= deg-alpha1 and\
                \
                0 <= i1_2 < nqp1d and \
                0 <= alpha1_2 <= deg and \
                0 <= i2_2 < nqp1d \
                }",
        """
        for el
            for i2
                <> xi = qpts[1, i2]
                <> s = 1-xi
                <> r = xi/s
                <> aind = 0 {id=aind_init}

                for alpha1
                    <> w = s**(deg-alpha1) {id=init_w}

                    <> tmp[alpha1,i2] = tmp[alpha1,i2] + w * coeffs[aind] \
                            {id=write_tmp}
                    for alpha2
                        w = w * r * ( deg - alpha1 - alpha2 ) / (1 + alpha2) \
                            {id=update_w,dep=init_w:write_tmp}
                        aind = aind + 1 \
                            {id=aind_incr,dep=aind_init:write_tmp:update_w}
                    end
                end
            end

            for i1_2
                <> xi2 = qpts[0, i1_2] {dep=aind_incr}
                <> s2 = 1-xi2
                <> r2 = xi2/s2
                <> w2 = s2**deg

                for alpha1_2
                    for i2_2
                        result[el, i1_2, i2_2] = result[el, i1_2, i2_2] + \
                                w2 * tmp[alpha1_2, i2_2]
                    end

                    w2 = w2 * r2 * (deg-alpha1_2) / (1+alpha1_2)
                end
            end
        end
        """,
        [
            # Must declare coeffs to have "no" shape, to keep loopy
            # from trying to figure it out the shape automatically.

            lp.GlobalArg("coeffs", None, shape=None),
            "..."
            ],
        assumptions="deg>=0 and nels>=1"
        )

    knl = lp.fix_parameters(knl, nqp1d=7, deg=4)

    if 0:
        knl = lp.split_iname(knl, "el", 16, inner_tag="l.0")
        knl = lp.split_iname(knl, "el_outer", 2, outer_tag="g.0", inner_tag="ilp",
                slabs=(0, 1))
        knl = lp.tag_inames(knl, dict(i2="l.1", alpha1="unr", alpha2="unr"))

    from pickle import dumps, loads
    knl = loads(dumps(knl))

    knl = lp.CompiledKernel(ctx, knl).get_highlighted_code(
            dict(
                qpts=np.float32,
                tmp=np.float32,
                coeffs=np.float32,
                result=np.float32,
                ))
    print(knl)


def test_stencil(ctx_factory):
    ctx = ctx_factory()

    # n=32 causes corner case behavior in size calculations for temprorary (a
    # non-unifiable, two-constant-segments PwAff as the base index)

    n = 256
    knl = lp.make_kernel(
            "{[i,j]: 0<= i,j < %d}" % n,
            [
                "a_offset(ii, jj) := a[ii+1, jj+1]",
                "z[i,j] = -2*a_offset(i,j)"
                " + a_offset(i,j-1)"
                " + a_offset(i,j+1)"
                " + a_offset(i-1,j)"
                " + a_offset(i+1,j)"
                ],
            [
                lp.GlobalArg("a", np.float32, shape=(n+2, n+2,)),
                lp.GlobalArg("z", np.float32, shape=(n+2, n+2,))
                ])

    ref_knl = knl

    def variant_1(knl):
        knl = lp.split_iname(knl, "i", 16, outer_tag="g.1", inner_tag="l.1")
        knl = lp.split_iname(knl, "j", 16, outer_tag="g.0", inner_tag="l.0")
        knl = lp.add_prefetch(knl, "a", ["i_inner", "j_inner"])
        knl = lp.set_loop_priority(knl, ["a_dim_0_outer", "a_dim_1_outer"])
        return knl

    def variant_2(knl):
        knl = lp.split_iname(knl, "i", 16, outer_tag="g.1", inner_tag="l.1")
        knl = lp.split_iname(knl, "j", 16, outer_tag="g.0", inner_tag="l.0")
        knl = lp.add_prefetch(knl, "a", ["i_inner", "j_inner"],
                fetch_bounding_box=True)
        knl = lp.set_loop_priority(knl, ["a_dim_0_outer", "a_dim_1_outer"])
        return knl

    for variant in [
            #variant_1,
            variant_2,
            ]:
        lp.auto_test_vs_ref(ref_knl, ctx, variant(knl),
                print_ref_code=False,
                op_count=[n*n], op_label=["cells"])


def test_stencil_with_overfetch(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<= i,j < n}",
            [
                "a_offset(ii, jj) := a[ii+2, jj+2]",
                "z[i,j] = -2*a_offset(i,j)"
                " + a_offset(i,j-1)"
                " + a_offset(i,j+1)"
                " + a_offset(i-1,j)"
                " + a_offset(i+1,j)"

                " + a_offset(i,j-2)"
                " + a_offset(i,j+2)"
                " + a_offset(i-2,j)"
                " + a_offset(i+2,j)"
                ],
            assumptions="n>=1")

    if ctx.devices[0].platform.name == "Portable Computing Language":
        # https://github.com/pocl/pocl/issues/205
        pytest.skip("takes very long to compile on pocl")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32))

    ref_knl = knl

    def variant_overfetch(knl):
        knl = lp.split_iname(knl, "i", 16, outer_tag="g.1", inner_tag="l.1",
                slabs=(1, 1))
        knl = lp.split_iname(knl, "j", 16, outer_tag="g.0", inner_tag="l.0",
               slabs=(1, 1))
        knl = lp.add_prefetch(knl, "a", ["i_inner", "j_inner"],
                fetch_bounding_box=True)
        knl = lp.set_loop_priority(knl, ["a_dim_0_outer", "a_dim_1_outer"])
        return knl

    for variant in [variant_overfetch]:
        n = 200
        lp.auto_test_vs_ref(ref_knl, ctx, variant(knl),
                print_ref_code=False,
                op_count=[n*n], parameters=dict(n=n), op_label=["cells"])


def test_sum_factorization():
    knl = lp.make_kernel(
        "{[i,j,ip,jp,k,l]: "
        "0<=i<I and 0<=j<J and 0<=ip<IP and 0<=jp<JP and 0<=k,l<Q}",
        """
        phi1(i, x) := x**i
        phi2(i, x) := x**i
        psi1(i, x) := x**i
        psi2(i, x) := x**i
        a(x, y) := 1

        A[i,j,ip,jp] = sum(k,sum(l,
            phi1(i,x[0,k]) * phi2(j,x[1,l])
            * psi1(ip, x[0,k]) * psi2(jp, x[1, l])
            * w[0,k] * w[1,l]
            * a(x[0,k], x[1,l])
        ))
        """)

    pytest.xfail("extract_subst is currently too stupid for sum factorization")

    knl = lp.extract_subst(knl, "temp_array",
            "phi1(i,x[0,k]) *psi1(ip, x[0,k]) * w[0,k]")
    knl = lp.extract_subst(knl, "temp_array",
            "sum(k, phi1(i,x[0,k]) *psi1(ip, x[0,k]) * w[0,k])")

    print(knl)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: foldmethod=marker

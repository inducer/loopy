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

from loopy.diagnostic import LoopyError

__all__ = [
        "pytest_generate_tests",
        "cl"  # "cl.create_some_context"
        ]


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


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
        knl = lp.prioritize_loops(knl, "iimg,im_x,im_y,ifeat,f_x,f_y")
        return knl

    def variant_1(knl):
        knl = lp.split_iname(knl, "im_x", 16, inner_tag="l.0")
        knl = lp.prioritize_loops(knl, "iimg,im_x_outer,im_y,ifeat,f_x,f_y")
        return knl

    def variant_2(knl):
        knl = lp.split_iname(knl, "im_x", 16, outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_iname(knl, "im_y", 16, outer_tag="g.1", inner_tag="l.1")
        knl = lp.tag_inames(knl, dict(ifeat="g.2"))
        knl = lp.add_prefetch(knl, "f[ifeat,:,:,:]",
                fetch_outer_inames="im_x_outer, im_y_outer, ifeat",
                default_tag="l.auto")
        knl = lp.add_prefetch(knl, "img", "im_x_inner, im_y_inner, f_x, f_y",
                fetch_outer_inames="iimg, im_x_outer, im_y_outer, ifeat, icolor",
                default_tag="l.auto")
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
        knl = lp.prioritize_loops(knl, "iimg,im_x,im_y,ifeat,f_x,f_y")
        return knl

    def variant_1(knl):
        knl = lp.split_iname(knl, "im_x", 16, inner_tag="l.0")
        knl = lp.prioritize_loops(knl, "iimg,im_x_outer,im_y,ifeat,f_x,f_y")
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


def test_rob_stroud_bernstein():
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
                                {id=write_tmp,dep=init_w:aind_init}
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
            assumptions="deg>=0 and nels>=1",
            target=lp.PyOpenCLTarget()
            )

    knl = lp.fix_parameters(knl, nqp1d=7, deg=4)
    knl = lp.split_iname(knl, "el", 16, inner_tag="l.0")
    knl = lp.split_iname(knl, "el_outer", 2, outer_tag="g.0", inner_tag="ilp",
            slabs=(0, 1))
    knl = lp.tag_inames(knl, dict(i2="l.1", alpha1="unr", alpha2="unr"))
    knl = lp.add_dtypes(knl, dict(
                qpts=np.float32,
                coeffs=np.float32,
                tmp=np.float32,
                ))
    print(lp.generate_code_v2(knl))


def test_rob_stroud_bernstein_full():
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
                            {id=write_tmp,dep=init_w:aind_init}
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
                <> w2 = s2**deg  {id=w2_init}

                for alpha1_2
                    for i2_2
                        result[el, i1_2, i2_2] = result[el, i1_2, i2_2] + \
                                w2 * tmp[alpha1_2, i2_2]  {id=res2,dep=w2_init}
                    end

                    w2 = w2 * r2 * (deg-alpha1_2) / (1+alpha1_2)  \
                            {id=w2_update, dep=res2}
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
        assumptions="deg>=0 and nels>=1",
        target=lp.PyOpenCLTarget()
        )

    knl = lp.fix_parameters(knl, nqp1d=7, deg=4)

    if 0:
        knl = lp.split_iname(knl, "el", 16, inner_tag="l.0")
        knl = lp.split_iname(knl, "el_outer", 2, outer_tag="g.0", inner_tag="ilp",
                slabs=(0, 1))
        knl = lp.tag_inames(knl, dict(i2="l.1", alpha1="unr", alpha2="unr"))

    from pickle import dumps, loads
    knl = loads(dumps(knl))

    knl = lp.add_dtypes(knl,
            dict(
                qpts=np.float32,
                tmp=np.float32,
                coeffs=np.float32,
                result=np.float32,
                ))
    print(lp.generate_code_v2(knl))


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
        knl = lp.add_prefetch(knl, "a", ["i_inner", "j_inner"], default_tag="l.auto")
        knl = lp.prioritize_loops(knl, ["a_dim_0_outer", "a_dim_1_outer"])
        return knl

    def variant_2(knl):
        knl = lp.split_iname(knl, "i", 16, outer_tag="g.1", inner_tag="l.1")
        knl = lp.split_iname(knl, "j", 16, outer_tag="g.0", inner_tag="l.0")
        knl = lp.add_prefetch(knl, "a", ["i_inner", "j_inner"],
                fetch_bounding_box=True, default_tag="l.auto")
        knl = lp.prioritize_loops(knl, ["a_dim_0_outer", "a_dim_1_outer"])
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
                fetch_bounding_box=True, default_tag="l.auto")
        knl = lp.prioritize_loops(knl, ["a_dim_0_outer", "a_dim_1_outer"])
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


def test_lbm(ctx_factory):
    ctx = ctx_factory()

    # D2Q4Q4Q4 lattice Boltzmann scheme for the shallow water equations
    # Example by Loic Gouarin <loic.gouarin@math.u-psud.fr>
    knl = lp.make_kernel(
        "{[ii,jj]:0<=ii<nx-2 and 0<=jj<ny-2}",
        """  # noqa (silences flake8 line length warning)
        i := ii + 1
        j := jj + 1
        for ii, jj
            with {id_prefix=init_m}
                <> m[0] =   +    f[i-1, j, 0] +    f[i, j-1, 1] + f[i+1, j, 2] +  f[i, j+1, 3]
                m[1] =   + 4.*f[i-1, j, 0] - 4.*f[i+1, j, 2]
                m[2] =   + 4.*f[i, j-1, 1] - 4.*f[i, j+1, 3]
                m[3] =   +    f[i-1, j, 0] -    f[i, j-1, 1] + f[i+1, j, 2] -  f[i, j+1, 3]
                m[4] =   +    f[i-1, j, 4] +    f[i, j-1, 5] + f[i+1, j, 6] +  f[i, j+1, 7]
                m[5] =   + 4.*f[i-1, j, 4] - 4.*f[i+1, j, 6]
                m[6] =   + 4.*f[i, j-1, 5] - 4.*f[i, j+1, 7]
                m[7] =   +    f[i-1, j, 4] -    f[i, j-1, 5] + f[i+1, j, 6] -  f[i, j+1, 7]
                m[8] =   +    f[i-1, j, 8] +    f[i, j-1, 9] + f[i+1, j, 10] + f[i, j+1, 11]
                m[9] =   + 4.*f[i-1, j, 8] - 4.*f[i+1, j, 10]
                m[10] =  + 4.*f[i, j-1, 9] - 4.*f[i, j+1, 11]
                m[11] =  +    f[i-1, j, 8] -    f[i, j-1, 9] + f[i+1, j, 10] - f[i, j+1, 11]
            end

            with {id_prefix=update_m,dep=init_m*}
                m[1] = m[1] + 2.*(m[4] - m[1])
                m[2] = m[2] + 2.*(m[8] - m[2])
                m[3] = m[3]*(1. - 1.5)
                m[5] = m[5] + 1.5*(0.5*(m[0]*m[0]) + (m[4]*m[4])/m[0] - m[5])
                m[6] = m[6] + 1.5*(m[4]*m[8]/m[0] - m[6])
                m[7] = m[7]*(1. - 1.2000000000000000)
                m[9] = m[9] + 1.5*(m[4]*m[8]/m[0] - m[9])
                m[10] = m[10] + 1.5*(0.5*(m[0]*m[0]) + (m[8]*m[8])/m[0] - m[10])
                m[11] = m[11]*(1. - 1.2)
            end

            with {dep=update_m*}
                f_new[i, j, 0] =  + 0.25*m[0] + 0.125*m[1] + 0.25*m[3]
                f_new[i, j, 1] =  + 0.25*m[0] + 0.125*m[2] - 0.25*m[3]
                f_new[i, j, 2] =  + 0.25*m[0] - 0.125*m[1] + 0.25*m[3]
                f_new[i, j, 3] =  + 0.25*m[0] - 0.125*m[2] - 0.25*m[3]
                f_new[i, j, 4] =  + 0.25*m[4] + 0.125*m[5] + 0.25*m[7]
                f_new[i, j, 5] =  + 0.25*m[4] + 0.125*m[6] - 0.25*m[7]
                f_new[i, j, 6] =  + 0.25*m[4] - 0.125*m[5] + 0.25*m[7]
                f_new[i, j, 7] =  + 0.25*m[4] - 0.125*m[6] - 0.25*m[7]
                f_new[i, j, 8] =  + 0.25*m[8] + 0.125*m[9] + 0.25*m[11]
                f_new[i, j, 9] =  + 0.25*m[8] + 0.125*m[10] - 0.25*m[11]
                f_new[i, j, 10] =  + 0.25*m[8] - 0.125*m[9] + 0.25*m[11]
                f_new[i, j, 11] =  + 0.25*m[8] - 0.125*m[10] - 0.25*m[11]
           end
        end
        """)

    knl = lp.add_and_infer_dtypes(knl, {"f": np.float32})

    ref_knl = knl

    knl = lp.split_iname(knl, "ii", 16, outer_tag="g.1", inner_tag="l.1")
    knl = lp.split_iname(knl, "jj", 16, outer_tag="g.0", inner_tag="l.0")
    knl = lp.expand_subst(knl)
    knl = lp.add_prefetch(knl, "f", "ii_inner,jj_inner", fetch_bounding_box=True,
            default_tag="l.auto")

    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters={"nx": 20, "ny": 20})


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
            fetch_bounding_box=True,
            default_tag="l.auto")

    #n = 1000
    #u = cl.clrandom.rand(queue, (n+2, n+2), dtype=np.float32)

    knl = lp.set_options(knl, write_code=True)
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
            "{ [c,i,j,k,ell,ell2]: \
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

    knl = lp.prioritize_loops(knl, ["c", "j", "i", "k"])

    def variant_1(knl):
        knl = lp.precompute(knl, "dpsi", "i,k,ell", default_tag="for")
        knl = lp.prioritize_loops(knl, "c,i,j")
        return knl

    def variant_2(knl):
        knl = lp.precompute(knl, "dpsi", "i,ell", default_tag="for")
        knl = lp.prioritize_loops(knl, "c,i,j")
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


def test_domain_tree_nesting():
    # From https://github.com/inducer/loopy/issues/78

    AS = lp.AddressSpace        # noqa

    out_map = np.array([1, 2], dtype=np.int32)
    if_val = np.array([-1, 0], dtype=np.int32)
    vals = np.array([2, 3], dtype=np.int32)
    num_vals = np.array([2, 4], dtype=np.int32)
    num_vals_offset = np.array(np.cumsum(num_vals) - num_vals, dtype=np.int32)

    TV = lp.TemporaryVariable  # noqa

    knl = lp.make_kernel(["{[i]: 0 <= i < 12}",
                    "{[j]: 0 <= j < 100}",
                    "{[a_count]: 0 <= a_count < a_end}",
                    "{[b_count]: 0 <= b_count < b_end}"],
    """
    for j
        for i
            <> a_end = abs(if_val[i])

            <>b_end = num_vals[i]
            <>offset = num_vals_offset[i] {id=offset}
            <>b_sum = 0 {id=b_init}
            for b_count
                <>val = vals[offset + b_count] {dep=offset}
            end
            b_sum = exp(b_sum) {id=b_final}

            out[j,i] =  b_sum {dep=b_final}
        end
    end
    """,
    [
        TV("out_map", initializer=out_map, read_only=True, address_space=AS.PRIVATE),
        TV("if_val", initializer=if_val, read_only=True, address_space=AS.PRIVATE),
        TV("vals", initializer=vals, read_only=True, address_space=AS.PRIVATE),
        TV("num_vals", initializer=num_vals, read_only=True,
           address_space=AS.PRIVATE),
        TV("num_vals_offset", initializer=num_vals_offset, read_only=True,
           address_space=AS.PRIVATE),
        lp.GlobalArg("B", shape=(100, 31), dtype=np.float64),
        lp.GlobalArg("out", shape=(100, 12), dtype=np.float64)],
        name="nested_domain")

    parents_per_domain = knl["nested_domain"].parents_per_domain()

    def depth(i):
        if parents_per_domain[i] is None:
            return 0
        else:
            return 1 + depth(parents_per_domain[i])

    for i in range(len(parents_per_domain)):
        assert depth(i) < 2


def test_prefetch_through_indirect_access():
    knl = lp.make_kernel("{[i, j, k]: 0 <= i,k < 10 and 0<=j<2}",
        """
        for i, j, k
            a[map1[indirect[i], j], k] = 2
        end
        """,
        [
            lp.GlobalArg("a", strides=(2, 1), dtype=int),
            lp.GlobalArg("map1", shape=(10, 10), dtype=int),
            "..."
            ],
        target=lp.CTarget())

    knl = lp.prioritize_loops(knl, "i,j,k")

    with pytest.raises(LoopyError):
        knl = lp.add_prefetch(knl, "map1[:, j]", default_tag="l.auto")


def test_unsigned_types_to_mod():
    knl = lp.make_kernel("{[i]: 0<=i<10}",
        """
            <> c = b[i] {id=init,dup=i}
            a[i] = i % c {dep=init}
        """,
        [lp.GlobalArg("a", shape=(10,), dtype=np.uint32),
         lp.GlobalArg("b", shape=(10,), dtype=np.uint32)]
    )
    assert "loopy_mod" not in lp.generate_code_v2(knl).device_code()


def test_abs_as_index():
    knl = lp.make_kernel(
        ["{[i]: 0<=i<10}"],
        """
        b[i] = a[abs(5-i)]
        """,
        [
            lp.GlobalArg("a", np.float32),
            ...
            ])
    print(lp.generate_code_v2(knl).device_code())


def test_sumpy_p2p_reduced():
    knl = lp.make_kernel(
        [
            "{[itgt_box]: 0<=itgt_box<5 }",
            "{[isrc_box]: 0<=isrc_box<isrc_box_end }",
            "{[inner]: 0 <=inner<=31 }",
            "{[itgt_offset_outer]: itgt_offset_outer=0 }",
            "{[isrc_prefetch_inner]: isrc_prefetch_inner=0 and 0<=inner<=25 }",
        ],
        """
        <> isrc_box_end = source_box_starts[itgt_box + 1] {inames=inner:itgt_box}
        <> itgt_offset = inner {inames=inner:itgt_box}
        <> isrc_prefetch = isrc_prefetch_inner*32 + inner \
            {inames=isrc_prefetch_inner:inner:isrc_box:itgt_box}
        """,
        [
            lp.GlobalArg("box_target_starts", dtype=np.int32),
            lp.GlobalArg("box_target_counts_nonchild", dtype=np.int32),
            lp.GlobalArg("box_source_starts", dtype=np.int32),
            lp.GlobalArg("box_source_counts_nonchild", dtype=np.int32),
            lp.GlobalArg("source_box_starts", dtype=np.int32),
            lp.GlobalArg("source_box_lists", dtype=np.int32),
        ],
        silenced_warnings="unused_inames",
    )
    lp.generate_code_v2(knl).device_code()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

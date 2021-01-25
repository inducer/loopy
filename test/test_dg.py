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


import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa
import loopy as lp

import logging  # noqa

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


def test_dg_volume(ctx_factory):
    #logging.basicConfig(level=logging.DEBUG)

    dtype = np.float32
    dtype4 = cl.array.vec.float4
    ctx = ctx_factory()

    order = "F"

    N = 3  # noqa
    Np = (N+1)*(N+2)*(N+3)//6  # noqa

    K = 10000  # noqa

    knl = lp.make_kernel([
            "{[n,m,k]: 0<= n,m < Np and 0<= k < K}",
            ],
            """
                <> du_drst = simul_reduce(sum, m, DrDsDt[n,m]*u[k,m])
                <> dv_drst = simul_reduce(sum, m, DrDsDt[n,m]*v[k,m])
                <> dw_drst = simul_reduce(sum, m, DrDsDt[n,m]*w[k,m])
                <> dp_drst = simul_reduce(sum, m, DrDsDt[n,m]*p[k,m])

                # volume flux
                rhsu[k,n] = dot(drst_dx[k],dp_drst)
                rhsv[k,n] = dot(drst_dy[k],dp_drst)
                rhsw[k,n] = dot(drst_dz[k],dp_drst)
                rhsp[k,n] = dot(drst_dx[k], du_drst) + dot(drst_dy[k], dv_drst) \
                    + dot(drst_dz[k], dw_drst)
                """,
            [
                lp.GlobalArg("u,v,w,p,rhsu,rhsv,rhsw,rhsp",
                    dtype, shape="K, Np", order="C"),
                lp.GlobalArg("DrDsDt", dtype4, shape="Np, Np", order="C"),
                lp.GlobalArg("drst_dx,drst_dy,drst_dz", dtype4, shape="K",
                    order=order),
                lp.ValueArg("K", np.int32, approximately=1000),
                ],
            name="dg_volume", assumptions="K>=1")

    knl = lp.fix_parameters(knl, Np=Np)

    seq_knl = knl

    def variant_basic(knl):
        knl = lp.tag_inames(knl, dict(k="g.0", n="l.0"))
        return knl

    def variant_more_per_work_group(knl):
        knl = lp.tag_inames(knl, dict(n="l.0"))
        knl = lp.split_iname(knl, "k", 3, outer_tag="g.0", inner_tag="l.1")
        return knl

    def variant_image_d(knl):
        knl = lp.tag_inames(knl, dict(n="l.0"))
        knl = lp.split_iname(knl, "k", 3, outer_tag="g.0", inner_tag="l.1")
        knl = lp.change_arg_to_image(knl, "DrDsDt")
        return knl

    def variant_prefetch_d(knl):
        knl = lp.tag_inames(knl, dict(n="l.0"))
        knl = lp.split_iname(knl, "k", 3, outer_tag="g.0", inner_tag="l.1")
        knl = lp.add_prefetch(knl, "DrDsDt[:,:]",
                fetch_outer_inames="k_outer",
                default_tag="l.auto")
        return knl

    def variant_prefetch_fields(knl):
        knl = lp.tag_inames(knl, dict(n="l.0"))
        knl = lp.split_iname(knl, "k", 3, outer_tag="g.0", inner_tag="l.1")
        for name in ["u", "v", "w", "p"]:
            knl = lp.add_prefetch(knl, "%s[k,:]" % name, ["k_inner"],
                    default_tag="l.auto")

        return knl

    def variant_k_ilp(knl):
        knl = lp.tag_inames(knl, dict(n="l.0"))

        knl = lp.split_iname(knl, "k", 3, outer_tag="g.0", inner_tag="ilp")
        knl = lp.tag_inames(knl, dict(m="unr"))
        return knl

    def variant_simple_padding(knl):
        knl = lp.tag_inames(knl, dict(n="l.0"))

        knl = lp.split_iname(knl, "k", 3, outer_tag="g.0", inner_tag="l.1")

        arg_names = [
                prefix+name
                for name in ["u", "v", "w", "p"]
                for prefix in ["", "rhs"]]

        for name in arg_names:
            knl = lp.add_padding(knl, name, axis=0, align_bytes=32)

        knl = lp.tag_inames(knl, dict(m="unr"))

        return knl

    def variant_fancy_padding(knl):
        knl = lp.tag_inames(knl, dict(n="l.0"))

        pad_mult = lp.find_padding_multiple(knl, "u", 1, 32)

        arg_names = [
                prefix+name
                for name in ["u", "v", "w", "p"]
                for prefix in ["", "rhs"]]

        knl = lp.split_array_dim(knl, [(nm, 0) for nm in arg_names], pad_mult)

        return knl

    parameters_dict = dict(K=K)

    variants = [
            variant_basic,
            variant_more_per_work_group,
            variant_prefetch_d,
            variant_prefetch_fields,
            variant_k_ilp,
            variant_simple_padding,
            variant_fancy_padding
            ]

    if (ctx.devices[0].image_support
            and ctx.devices[0].platform.name != "Portable Computing Language"):
        variants.append(variant_image_d)

    for variant in variants:
        lp.auto_test_vs_ref(
                seq_knl, ctx, variant(knl), parameters=parameters_dict,
                #codegen_kwargs=dict(with_annotation=True)
                )


def no_test_dg_surface(ctx_factory):
    # tough to test, would need the right index info
    dtype = np.float32
    ctx = ctx_factory()

    order = "F"

    N = 3  # noqa
    Np = (N+1)*(N+2)*(N+3)//6  # noqa
    Nfp = (N+1)*(N+2)//2  # noqa
    Nfaces = 4  # noqa

    K = 10000  # noqa

    knl = lp.make_kernel(
            [
                "{[m,n,k]: 0<= m < NfpNfaces and 0<= n < Np and 0<= k < K }"
                ],
            """
                <> idP = vmapP[m,k]
                <> idM = vmapM[m,k]

                <> du = u[[idP]]-u[[idM]]
                <> dv = v[[idP]]-v[[idM]]
                <> dw = w[[idP]]-w[[idM]]
                <> dp = bc[m,k]*p[[idP]] - p[[idM]]

                <> dQ = 0.5*Fscale[m,k]* \
                        (dp - nx[m,k]*du - ny[m,k]*dv - nz[m,k]*dw)

                <> fluxu = -nx[m,k]*dQ
                <> fluxv = -ny[m,k]*dQ
                <> fluxw = -nz[m,k]*dQ
                <> fluxp =          dQ

                # reduction here
                rhsu[n,k] = sum(m, LIFT[n,m]*fluxu)
                rhsv[n,k] = sum(m, LIFT[n,m]*fluxv)
                rhsw[n,k] = sum(m, LIFT[n,m]*fluxw)
                rhsp[n,k] = sum(m, LIFT[n,m]*fluxp)
                """,
            [
                lp.GlobalArg("vmapP,vmapM",
                    np.int32, shape="NfpNfaces, K", order=order),
                lp.GlobalArg("u,v,w,p,rhsu,rhsv,rhsw,rhsp",
                    dtype, shape="Np, K", order=order),
                lp.GlobalArg("nx,ny,nz,Fscale,bc",
                    dtype, shape="NfpNfaces, K", order=order),
                lp.GlobalArg("LIFT", dtype, shape="Np, NfpNfaces", order="C"),
                lp.ValueArg("K", np.int32, approximately=1000),
                ],
            name="dg_surface", assumptions="K>=1")

    knl = lp.fix_parameters(knl,
            Np=Np, Nfp=Nfp, NfpNfaces=Nfaces*Nfp, nsurf_dofs=K*Nfp)

    seq_knl = knl

    def variant_basic(knl):
        return knl

    parameters_dict = dict(K=K)

    for variant in [
            variant_basic,
            ]:

        lp.auto_test_vs_ref(seq_knl, ctx, variant(knl), parameters=parameters_dict)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

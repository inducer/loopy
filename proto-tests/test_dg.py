from __future__ import division

import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as cl_random
import loopy as lp

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests





def dot_mangler(name, arg_dtypes):
    scalar_dtype, offset, field_name = arg_dtypes[0].fields["s0"]
    return scalar_dtype, name

def test_dg_volume(ctx_factory):
    dtype = np.float32
    dtype4 = cl.array.vec.float4
    ctx = ctx_factory()

    order = "F"

    N = 3
    Np = (N+1)*(N+2)*(N+3)//6

    K = 10000

    knl = lp.make_kernel(ctx.devices[0], [
            "{[n,m,k]: 0<= n,m < Np and 0<= k < K}",
            ],
            [
                "<> dudR = sum(m, DrDsDt[n,m]*u[m,k])",
                "<> dvdR = sum(m, DrDsDt[n,m]*v[m,k])",
                "<> dwdR = sum(m, DrDsDt[n,m]*w[m,k])",
                "<> dpdR = sum(m, DrDsDt[n,m]*p[m,k])",
                # volume flux
                "rhsu[n,k] = dot(dRdx[k],dpdR)",
                "rhsv[n,k] = dot(dRdy[k],dpdR)",
                "rhsw[n,k] = dot(dRdz[k],dpdR)",
                "rhsp[n,k] = dot(dRdx[k], dudR) + dot(dRdy[k], dvdR)"
                "+ dot(dRdz[k], dwdR)",
                ],
            [
                lp.GlobalArg("u,v,w,p,rhsu,rhsv,rhsw,rhsp",
                    dtype, shape="Np, K", order=order),
                lp.GlobalArg("DrDsDt", dtype4, shape="Np, Np", order="C"),
                lp.GlobalArg("dRdx,dRdy,dRdz", dtype4, shape="K", order=order),
                lp.ValueArg("K", np.int32, approximately=1000),
                ],
            name="dg_volume", assumptions="K>=1",
            defines=dict(Np=Np),
            function_manglers=[
                lp.default_function_mangler,
                lp.opencl_function_mangler,
                lp.single_arg_function_mangler,
                dot_mangler]
            )

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
        knl = lp.add_prefetch(knl, "DrDsDt[:,:]")
        return knl

    def variant_prefetch_fields(knl):
        knl = lp.tag_inames(knl, dict(n="l.0"))
        knl = lp.split_iname(knl, "k", 3, outer_tag="g.0", inner_tag="l.1")
        for name in ["u", "v", "w", "p"]:
            # FIXME
            knl = lp.add_prefetch(knl, "%s[:,k]" % name)

        return knl

    def variant_k_ilp(knl):
        knl = lp.tag_inames(knl, dict(n="l.0"))

        # FIXME
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
            knl = lp.add_padding(knl, name, axis=1, align_bytes=32)

        knl = lp.tag_inames(knl, dict(m="unr"))

        return knl

    def variant_fancy_padding(knl):
        knl = lp.tag_inames(knl, dict(n="l.0"))

        pad_mult = lp.find_padding_multiple(knl, "u", 1, 32)

        knl = lp.split_iname(knl, "k", pad_mult, outer_tag="g.0", inner_tag="l.1")

        arg_names = [
                prefix+name
                for name in ["u", "v", "w", "p"]
                for prefix in ["", "rhs"]]

        # FIXME
        knl = lp.split_arg_axis(knl, [(nm, 1) for nm in arg_names], pad_mult)


        return knl

    parameters_dict = dict(K=K)

    for variant in [
            #variant_basic,
            #variant_more_per_work_group,
            variant_image_d,
            #variant_prefetch_d,
            #variant_prefetch_fields,
            #variant_k_ilp,
            #variant_simple_padding,
            #variant_fancy_padding
            ]:
        kernel_gen = lp.generate_loop_schedules(variant(knl))
        kernel_gen = lp.check_kernels(kernel_gen, parameters_dict)

        lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen, parameters=parameters_dict)


def test_dg_surface(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()

    order = "F"

    N = 3
    Np = (N+1)*(N+2)*(N+3)//6
    Nfp = (N+1)*(N+2)//2
    Nfaces = 4

    K = 10000

    knl = lp.make_kernel(ctx.devices[0],
            ["{[m,n,k]: 0<= m < NfpNfaces and 0<= n < Np and 0<= k < K }"
                ],
            """
                <> idP = vmapP[m,k]
                <> idM = vmapM[m,k]

                # can we bounce to single index (row/column major is important)
                # can we use this indexing here for clarity ?
                <> du = u[[idP]]-u[idM]
                <> dv = v[idP]-v[idM]
                <> dw = w[idP]-w[idM]
                <> dp = bc[m,k]*p[idP] - p[idM]

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
                    dtype, shape="nsurf_dofs", order=order),
                lp.GlobalArg("LIFT", dtype, shape="Np, NfpNfaces", order="C"),
                lp.ValueArg("K", np.int32, approximately=1000),
                ],
            name="dg_surface", assumptions="K>=1",
            defines=dict(Np=Np, Nfp=Nfp, NfpNfaces=Nfaces*Nfp, nsurf_dofs=K*Nfp),
            )

    seq_knl = knl

    def variant_basic(knl):
        return knl

    parameters_dict = dict(K=K)

    for variant in [
            variant_basic,
            ]:
        kernel_gen = lp.generate_loop_schedules(variant(knl))
        kernel_gen = lp.check_kernels(kernel_gen, parameters_dict)

        lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen, parameters=parameters_dict)




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

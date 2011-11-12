from __future__ import division

import numpy as np
import pyopencl as cl
import loopy as lp

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests




def test_laplacian_stiffness(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    dim = 2

    Nq = 40 # num. quadrature points
    Nc = 100 # num. cells
    Nb = 20 # num. basis functions

    # K - run-time symbolic

    from pymbolic import var
    Nc_sym = var("Nc")

    knl = lp.make_kernel(ctx.devices[0],
            "[Nc] -> {[K,i,j,q, ax_a, ax_b]: 0<=K<Nc and 0<=i,j<%(Nb)d and 0<=q<%(Nq)d "
            "and 0<= ax_a, ax_b < %(dim)d}"
            % dict(Nb=Nb, Nq=Nq, dim=dim),
            [
                "dPsi(a, dxi) := sum_float32(@ax_b,"
                    "  jacInv[ax_b,dxi,K,q] * DPsi[ax_b,a,q])",
                "A[K, i, j] = sum_float32(q, w[q] * jacDet[K,q] * ("
                    "sum_float32(ax_a, dPsi(0,ax_a)*dPsi(1,ax_a))))"
                ],
            [
            lp.ArrayArg("jacInv", dtype, shape=(dim, dim, Nc_sym, Nq), order=order),
            lp.ConstantArrayArg("DPsi", dtype, shape=(dim, Nb, Nq), order=order),
            lp.ArrayArg("jacDet", dtype, shape=(Nc_sym, Nq), order=order),
            lp.ConstantArrayArg("w", dtype, shape=(Nq, dim), order=order),
            lp.ArrayArg("A", dtype, shape=(Nc_sym, Nb, Nb), order=order),
            lp.ScalarArg("Nc",  np.int32, approximately=1000),
            ],
            name="lapquad", assumptions="Nc>=1")

    knl = lp.tag_dimensions(knl, dict(ax_b="unr"))
    seq_knl = knl

    def variant_1(knl):
        # no ILP across elements
        knl = lp.split_dimension(knl, "K", 16, outer_tag="g.0", slabs=(0,1))
        knl = lp.tag_dimensions(knl, {"i": "l.1", "j": "l.0"})
        knl = lp.add_prefetch(knl, 'jacInv',
                ["jacInv_dim_0", "jacInv_dim_1", "K_inner", "q"])
        return knl

    def variant_2(knl):
        # with ILP across elements
        knl = lp.split_dimension(knl, "K", 16, outer_tag="g.0", slabs=(0,1))
        knl = lp.split_dimension(knl, "K_inner", 4, inner_tag="ilp")
        knl = lp.tag_dimensions(knl, {"i": "l.1", "j": "l.0"})
        knl = lp.add_prefetch(knl, "jacInv",
                ["jacInv_dim_0", "jacInv_dim_1", "K_inner_inner", "K_inner_outer", "q"])
        return knl

    def variant_3(knl):
        # no ILP across elements, precompute dPsiTransf

        # generates correct code--but suboptimal in a bunch of ways.

        knl = lp.split_dimension(knl, "K", 16, outer_tag="g.0", slabs=(0,1))
        knl = lp.add_prefetch(knl, "jacInv",
                ["jacInv_dim_0", "jacInv_dim_1", "q"])
        knl = lp.tag_dimensions(knl, {"i": "l.1", "j": "l.0"})
        knl = lp.precompute(knl, "dPsi", np.float32,
                sweep_axes=["K_inner"])
        return knl

    for variant in [variant_1, variant_2, variant_3]:
    #for variant in [variant_3]:
        kernel_gen = lp.generate_loop_schedules(variant(knl),
                loop_priority=["jacInv_dim_0", "jacInv_dim_1"])
        kernel_gen = lp.check_kernels(kernel_gen, dict(Nc=Nc))

        lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
                op_count=0, op_label="GFlops",
                parameters={"Nc": Nc}, print_ref_code=True,
                timing_rounds=30)




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])


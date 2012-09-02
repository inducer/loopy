from __future__ import division

import numpy as np
import pyopencl as cl
import loopy as lp

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests




def test_tim2d(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    n = 8

    from pymbolic import var
    K_sym = var("K")

    field_shape = (K_sym, n, n)

    # K - run-time symbolic
    knl = lp.make_kernel(ctx.devices[0],
            "[K] -> {[i,j,e,m,o,gi]: 0<=i,j,m,o<%d and 0<=e<K and 0<=gi<3}" % n,
           [
            "ur(a,b) := sum(@o, D[a,o]*u[e,o,b])",
            "us(a,b) := sum(@o, D[b,o]*u[e,a,o])",

            #"Gu(mat_entry,a,b) := G[mat_entry,e,m,j]*ur(m,j)",

            "Gux(a,b) := G$x[0,e,a,b]*ur(a,b)+G$x[1,e,a,b]*us(a,b)",
            "Guy(a,b) := G$y[1,e,a,b]*ur(a,b)+G$y[2,e,a,b]*us(a,b)",
            "lap[e,i,j]  = "
            "  sum(m, D[m,i]*Gux(m,j))"
            "+ sum(m, D[m,j]*Guy(i,m))"

            ],
            [
            lp.GlobalArg("u", dtype, shape=field_shape, order=order),
            lp.GlobalArg("lap", dtype, shape=field_shape, order=order),
            lp.GlobalArg("G", dtype, shape=(3,)+field_shape, order=order),
            # lp.ConstantArrayArg("D", dtype, shape=(n, n), order=order),
            lp.GlobalArg("D", dtype, shape=(n, n), order=order),
            # lp.ImageArg("D", dtype, shape=(n, n)),
            lp.ValueArg("K", np.int32, approximately=1000),
            ],
            name="semlap2D", assumptions="K>=1")

    seq_knl = knl

    def variant_orig(knl):
        knl = lp.tag_inames(knl, dict(i="l.0", j="l.1", e="g.0"))

        knl = lp.add_prefetch(knl, "D[:,:]")
        knl = lp.add_prefetch(knl, "u[e, :, :]")

        knl = lp.precompute(knl, "ur(m,j)", np.float32, ["m", "j"])
        knl = lp.precompute(knl, "us(i,m)", np.float32, ["i", "m"])

        knl = lp.precompute(knl, "Gux(m,j)", np.float32, ["m", "j"])
        knl = lp.precompute(knl, "Guy(i,m)", np.float32, ["i", "m"])

        knl = lp.add_prefetch(knl, "G$x")
        knl = lp.add_prefetch(knl, "G$y")

        knl = lp.tag_inames(knl, dict(o="unr"))
        knl = lp.tag_inames(knl, dict(m="unr"))

        knl = lp.set_instruction_priority(knl, "D_fetch", 5)

        return knl

    for variant in [variant_orig]:
        kernel_gen = lp.generate_loop_schedules(variant(knl))
        kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000))

        K = 1000
        lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
                op_count=[K*(n*n*n*2*2 + n*n*2*3 + n**3 * 2*2)/1e9],
                op_label=["GFlops"],
                parameters={"K": K}, print_ref_code=True)




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

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
            "ur(a,b) := sum_float32(@o, D[a,o]*u[e,o,b])",
            "us(a,b) := sum_float32(@o, D[b,o]*u[e,a,o])",

            #"Gu(mat_entry,a,b) := G[mat_entry,e,m,j]*ur(m,j)",

            "Gux(a,b) := G[0,e,a,b]*ur(a,b)+G[1,e,a,b]*us(a,b)",
            "Guy(a,b) := G[1,e,a,b]*ur(a,b)+G[2,e,a,b]*us(a,b)",
            "lap[e,i,j]  = "
            "  sum_float32(m, D[m,i]*Gux(m,j))"
            "+ sum_float32(m, D[m,j]*Guy(i,m))"

            ],
            [
            lp.ArrayArg("u", dtype, shape=field_shape, order=order),
            lp.ArrayArg("lap", dtype, shape=field_shape, order=order),
            lp.ArrayArg("G", dtype, shape=(3,)+field_shape, order=order),
            # lp.ConstantArrayArg("D", dtype, shape=(n, n), order=order),
            lp.ArrayArg("D", dtype, shape=(n, n), order=order),
            # lp.ImageArg("D", dtype, shape=(n, n)),
            lp.ScalarArg("K", np.int32, approximately=1000),
            ],
            name="semlap2D", assumptions="K>=1")

    seq_knl = knl

    def variant_orig(knl):
        knl = lp.tag_dimensions(knl, dict(i="l.0", j="l.1", e="g.0"))

        knl = lp.add_prefetch(knl, "D", ["m", "j", "i","o"])
        knl = lp.add_prefetch(knl, "u", ["i", "j",  "o"])

        knl = lp.precompute(knl, "ur", np.float32, ["m", "j"], "ur(m,j)")
        knl = lp.precompute(knl, "us", np.float32, ["i", "m"], "us(i,m)")

        knl = lp.add_prefetch(knl, "G")

        knl = lp.precompute(knl, "Gux", np.float32, ["m", "j"], "Gux(m,j)")
        knl = lp.precompute(knl, "Guy", np.float32, ["i", "m"], "Gux(i,m)")

        knl = lp.tag_dimensions(knl, dict(o="unr"))
        knl = lp.tag_dimensions(knl, dict(m="unr"))

        return knl

    def variant_prefetch(knl):
        knl = lp.precompute(knl, "ur", np.float32, ["a", "b"])
        knl = lp.precompute(knl, "us", np.float32, ["a", "b"])
        return knl

    def variant_1(knl):
        # BUG? why can't the prefetch be in the j loop??!
        print knl
        from pudb import set_trace; set_trace()
        knl = lp.precompute(knl, "ur", np.float32, ["a"])
        print knl
        1/0
        #knl = lp.precompute(knl, "us", np.float32, ["a"])
        return knl

    def variant_g_prefetch(knl):
        knl = lp.precompute(knl, "ur", np.float32, ["a"])
        knl = lp.precompute(knl, "us", np.float32, ["a"])
        knl = lp.add_prefetch(knl, "G", per_access=True) # IMPLEMENT!
        return knl

    def variant_gu_precomp(knl):
        knl = lp.precompute(knl, "ur", np.float32, ["a"])
        knl = lp.precompute(knl, "us", np.float32, ["a"])
        knl = lp.precompute(knl, "Gux", np.float32, ["a", "b"])
        knl = lp.precompute(knl, "Guy", np.float32, ["a", "b"])
        return knl

    for variant in [variant_orig]:
    #for variant in [variant_1]:
        kernel_gen = lp.generate_loop_schedules(variant(knl))
        kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000))

        K = 1000
        lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
                op_count=K*(n*n*n*2*2 + n*n*2*3 + n**3 * 2*2)/1e9,
                op_label="GFlops",
                parameters={"K": K}, print_ref_code=True)




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

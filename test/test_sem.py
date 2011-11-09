from __future__ import division

import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests




def test_sem_3d(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    n = 8

    from pymbolic import var
    K_sym = var("K")

    field_shape = (K_sym, n, n, n)

    # load:     1+6 fields + 1/N D entry
    # store:    1   fields
    # perform:  N*2*6 + 3*5 flops
    # ratio:   (12*N+15)/8  flops per 4 bytes on bus 
    #          ~ 14 FLOPS per 4 bytes at N=8
    #          ~ 525 GFLOPS max on a 150GB/s device at N=8 if done perfectly

    # K - run-time symbolic
    knl = lp.make_kernel(ctx.devices[0],
            "[K] -> {[i,j,k,e,m,o,gi]: 0<=i,j,k,m,o<%d and 0<=e<K and 0<=gi<6}" % n,
            [
                "CSE: ur(i,j,k) = sum_float32(@o, D[i,o]*u[e,o,j,k])",
                "CSE: us(i,j,k) = sum_float32(@o, D[j,o]*u[e,i,o,k])",
                "CSE: ut(i,j,k) = sum_float32(@o, D[k,o]*u[e,i,j,o])",

                "lap[e,i,j,k]  = "
                "  sum_float32(m, D[m,i]*(G[0,e,m,j,k]*ur(m,j,k) + G[1,e,m,j,k]*us(m,j,k) + G[2,e,m,j,k]*ut(m,j,k)))"
                "+ sum_float32(m, D[m,j]*(G[1,e,i,m,k]*ur(i,m,k) + G[3,e,i,m,k]*us(i,m,k) + G[4,e,i,m,k]*ut(i,m,k)))"
                "+ sum_float32(m, D[m,k]*(G[2,e,i,j,m]*ur(i,j,m) + G[4,e,i,j,m]*us(i,j,m) + G[5,e,i,j,m]*ut(i,j,m)))"
                ],
            [
            lp.ArrayArg("u", dtype, shape=field_shape, order=order),
            lp.ArrayArg("lap", dtype, shape=field_shape, order=order),
            lp.ArrayArg("G", dtype, shape=(6,)+field_shape, order=order),
            lp.ArrayArg("D", dtype, shape=(n, n), order=order),
            lp.ScalarArg("K", np.int32, approximately=1000),
            ],
            name="semlap", assumptions="K>=1")


    def add_pf(knl):
        knl = lp.add_prefetch(knl, "G", ["gi", "m", "j", "k"], "G[gi,e,m,j,k]")
        knl = lp.add_prefetch(knl, "D", ["m", "j"])
        knl = lp.add_prefetch(knl, "u", ["i", "j", "k"], "u[*,i,j,k]")
        knl = lp.realize_cse(knl, "ur", np.float32, ["k", "j", "m"])
        knl = lp.realize_cse(knl, "us", np.float32, ["i", "m", "k"])
        knl = lp.realize_cse(knl, "ut", np.float32, ["i", "j", "m"])

    seq_knl = add_pf(knl)

    knl = lp.split_dimension(knl, "e", 16, outer_tag="g.0")#, slabs=(0, 1))
    #knl = lp.split_dimension(knl, "e_inner", 4, inner_tag="ilp")

    knl = add_pf(knl)
    #print seq_knl
    #print lp.preprocess_kernel(seq_knl)
    #1/0


    knl = lp.tag_dimensions(knl, dict(i="l.0", j="l.1"))

    kernel_gen = lp.generate_loop_schedules(knl,
            loop_priority=["j_dr", "j_ds",  "i_dt"])
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000))

    K = 1000
    lp.auto_test_vs_seq(seq_knl, ctx, kernel_gen,
            op_count=K*(n*n*n*n*2*3 + n*n*n*5*3 + n**4 * 2*3)/1e9,
            op_label="GFlops",
            parameters={"K": K}, print_seq_code=True)



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

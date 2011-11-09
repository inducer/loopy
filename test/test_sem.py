from __future__ import division

import numpy as np
import pyopencl as cl
import loopy as lp

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests




def test_laplacian(ctx_factory):
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
            "[K] -> {[i,j,k,e,m,o1,o2,o3,gi]: 0<=i,j,k,m,o1,o2,o3<%d and 0<=e<K and 0<=gi<6}" % n,
            [
                "CSE: ur(i,j,k) = sum_float32(o1, D[i,o1]*cse(u[e,o1,j,k], urf))",
                "CSE: us(i,j,k) = sum_float32(o2, D[j,o2]*cse(u[e,i,o2,k], usf))",
                "CSE: ut(i,j,k) = sum_float32(o3, D[k,o3]*cse(u[e,i,j,o3], utf))",

                # define function
                "CSE: Gu(i,j,k) = G[0,e,i,j,k]*ur(i,j,k) + G[1,e,i,j,k]*us(i,j,k) + G[2,e,i,j,k]*ut(i,j,k)",
                "CSE: Gv(i,j,k) = G[1,e,i,j,k]*ur(i,j,k) + G[3,e,i,j,k]*us(i,j,k) + G[4,e,i,j,k]*ut(i,j,k)",
                "CSE: Gw(i,j,k) = G[2,e,i,j,k]*ur(i,j,k) + G[4,e,i,j,k]*us(i,j,k) + G[5,e,i,j,k]*ut(i,j,k)",

                "lap[e,i,j,k]  = "
                "  sum_float32(m, D[m,i]*Gu(m,j,k))"
                "+ sum_float32(m, D[m,j]*Gv(i,m,k))"
                "+ sum_float32(m, D[m,k]*Gw(i,j,m))"
                ],
            [
            lp.ArrayArg("u", dtype, shape=field_shape, order=order),
            lp.ArrayArg("lap", dtype, shape=field_shape, order=order),
            lp.ArrayArg("G", dtype, shape=(6,)+field_shape, order=order),
            lp.ArrayArg("D", dtype, shape=(n, n), order=order),
            lp.ScalarArg("K", np.int32, approximately=1000),
            ],
            name="semlap", assumptions="K>=1")

    #print lp.preprocess_kernel(knl, cse_ok=True)
    #1/0
    #
    #print knl
    #1/0
    knl = lp.realize_cse(knl, "urf", np.float32, ["o1"])
    knl = lp.realize_cse(knl, "usf", np.float32, ["o2"])
    knl = lp.realize_cse(knl, "utf", np.float32, ["o3"])

    knl = lp.realize_cse(knl, "Gu", np.float32, ["m", "j", "k"])
    knl = lp.realize_cse(knl, "Gv", np.float32, ["i", "m", "k"])
    knl = lp.realize_cse(knl, "Gw", np.float32, ["i", "j", "m"])

    knl = lp.realize_cse(knl, "ur", np.float32, ["k", "j", "m"])
    knl = lp.realize_cse(knl, "us", np.float32, ["i", "m", "k"])
    knl = lp.realize_cse(knl, "ut", np.float32, ["i", "j", "m"])

    if 0:
        pass
        #seq_knl = lp.add_prefetch(knl, "G", ["gi", "m", "j", "k"], "G[gi,e,m,j,k]")
        #seq_knl = lp.add_prefetch(seq_knl, "D", ["m", "j"])
        #seq_knl = lp.add_prefetch(seq_knl, "u", ["i", "j", "k"], "u[*,i,j,k]")
    else:
        seq_knl = knl

    knl = lp.split_dimension(knl, "e", 16, outer_tag="g.0")#, slabs=(0, 1))

    knl = lp.add_prefetch(knl, "G", ["gi", "m", "j", "k"], "G[gi,e,m,j,k]")
    knl = lp.add_prefetch(knl, "D", ["m", "j"])
    #knl = lp.add_prefetch(knl, "u", ["i", "j", "k"], "u[*,i,j,k]")

    #knl = lp.split_dimension(knl, "e_inner", 4, inner_tag="ilp")

    #print seq_knl
    #print lp.preprocess_kernel(knl)
    #1/0

    knl = lp.tag_dimensions(knl, dict(i="l.0", j="l.1"))

    kernel_gen = lp.generate_loop_schedules(knl,
            loop_priority=["m_fetch_G", "i_fetch_u"])
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000))

    K = 1000
    lp.auto_test_vs_seq(seq_knl, ctx, kernel_gen,
            op_count=K*(n*n*n*n*2*3 + n*n*n*5*3 + n**4 * 2*3)/1e9,
            op_label="GFlops",
            parameters={"K": K}, print_seq_code=True)




def test_advect(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()

    order = "C"

    N = 8

    from pymbolic import var
    K_sym = var("K")

    field_shape = (K_sym, N, N, N)

    # 1. direction-by-direction similarity transform on u
    # 2. invert diagonal 
    # 3. transform back (direction-by-direction)

    # K - run-time symbolic

    # A. updated for CSE: notation. 
    # B. fixed temp indexing and C ordering
    # load:     3+9 fields + 1/N D entry
    # store:    3   fields
    # perform:  N*2*6 + 3*5 + 3*5 flops
    # ratio:   (12*N+30)/15  flops per 4 bytes on bus 
    #          ~ 8.4 FLOPS per 4 bytes at N=8
    #          ~ 300 GFLOPS max on a 150GB/s device at N=8 if done perfectly
    knl = lp.make_kernel(ctx.devices[0],
            "[K] -> {[i,j,k,m,e]: 0<=i,j,k,m<%d AND 0<=e<K}" % N,
            [
                # differentiate u
                "CSE:  ur(i,j,k) = sum_float32(@m, D[i,m]*u[e,m,j,k])",
                "CSE:  us(i,j,k) = sum_float32(@m, D[j,m]*u[e,i,m,k])",
                "CSE:  ut(i,j,k) = sum_float32(@m, D[k,m]*u[e,i,j,m])",

                # differentiate v
                "CSE:  vr(i,j,k) = sum_float32(@m, D[i,m]*v[e,m,j,k])",
                "CSE:  vs(i,j,k) = sum_float32(@m, D[j,m]*v[e,i,m,k])",
                "CSE:  vt(i,j,k) = sum_float32(@m, D[k,m]*v[e,i,j,m])",

                # differentiate w
                "CSE:  wr(i,j,k) = sum_float32(@m, D[i,m]*w[e,m,j,k])",
                "CSE:  ws(i,j,k) = sum_float32(@m, D[j,m]*w[e,i,m,k])",
                "CSE:  wt(i,j,k) = sum_float32(@m, D[k,m]*w[e,i,j,m])",

                # find velocity in (r,s,t) coordinates
                # CSE?
                "CSE: Vr(i,j,k) = G[0,e,i,j,k]*u[e,i,j,k] + G[1,e,i,j,k]*v[e,i,j,k] + G[2,e,i,j,k]*w[e,i,j,k]",
                "CSE: Vs(i,j,k) = G[3,e,i,j,k]*u[e,i,j,k] + G[4,e,i,j,k]*v[e,i,j,k] + G[5,e,i,j,k]*w[e,i,j,k]",
                "CSE: Vt(i,j,k) = G[6,e,i,j,k]*u[e,i,j,k] + G[7,e,i,j,k]*v[e,i,j,k] + G[8,e,i,j,k]*w[e,i,j,k]",

                # form nonlinear term on integration nodes
                "Nu[e,i,j,k] = Vr(i,j,k)*ur(i,j,k)+Vs(i,j,k)*us(i,j,k)+Vt(i,j,k)*ut(i,j,k)",
                "Nv[e,i,j,k] = Vr(i,j,k)*vr(i,j,k)+Vs(i,j,k)*vs(i,j,k)+Vt(i,j,k)*vt(i,j,k)",
                "Nw[e,i,j,k] = Vr(i,j,k)*wr(i,j,k)+Vs(i,j,k)*ws(i,j,k)+Vt(i,j,k)*wt(i,j,k)",
                ],
            [
            lp.ArrayArg("u",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("v",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("w",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("Nu",  dtype, shape=field_shape, order=order),
            lp.ArrayArg("Nv",  dtype, shape=field_shape, order=order),
            lp.ArrayArg("Nw",  dtype, shape=field_shape, order=order),
            lp.ArrayArg("G",   dtype, shape=(9,)+field_shape, order=order),
            lp.ArrayArg("D",   dtype, shape=(N, N),  order=order),
            lp.ScalarArg("K",  np.int32, approximately=1000),
            ],
            name="sem_advect", assumptions="K>=1")

    print knl
    1/0

    seq_knl = knl

    knl = lp.split_dimension(knl, "e", 16, outer_tag="g.0")#, slabs=(0, 1))

    knl = lp.tag_dimensions(knl, dict(i="l.0", j="l.1"))


    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000), kill_level_min=5)


    K = 1000
    lp.auto_test_vs_seq(seq_knl, ctx, kernel_gen,
            op_count=0,
            op_label="GFlops",
            parameters={"K": K}, print_seq_code=True,)




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

import numpy as np

import pyopencl as cl  # noqa
from pyopencl.tools import (
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,  # noqa
)

import loopy as lp


1/0  # inspect me


def test_laplacian(ctx_factory):
    1/0  # not adapted to new language

    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    n = 8

    from pymbolic import var
    K_sym = var("K")  # noqa

    field_shape = (K_sym, n, n, n)

    # load:     1+6 fields + 1/N D entry
    # store:    1   fields
    # perform:  N*2*6 + 3*5 flops
    # ratio:   (12*N+15)/8  flops per 4 bytes on bus
    #          ~ 14 FLOPS per 4 bytes at N=8
    #          ~ 525 GFLOPS max on a 150GB/s device at N=8 if done perfectly

    # K - run-time symbolic
    knl = lp.make_kernel(ctx.devices[0],
            "[K] -> {[i,j,k,e,m,o1,o2,o3,gi]: 0<=i,j,k,m,o1,o2,o3<%d and 0<=e<K and 0<=gi<6}" % n,  # noqa
            [
                "CSE: ur(i,j,k) = sum_float32(o1, D[i,o1]*cse(u[e,o1,j,k], urf))",
                "CSE: us(i,j,k) = sum_float32(o2, D[j,o2]*cse(u[e,i,o2,k], usf))",
                "CSE: ut(i,j,k) = sum_float32(o3, D[k,o3]*cse(u[e,i,j,o3], utf))",

                # define function
                "CSE: Gu(i,j,k) = G[0,e,i,j,k]*ur(i,j,k) + G[1,e,i,j,k]*us(i,j,k) + G[2,e,i,j,k]*ut(i,j,k)",  # noqa
                "CSE: Gv(i,j,k) = G[1,e,i,j,k]*ur(i,j,k) + G[3,e,i,j,k]*us(i,j,k) + G[4,e,i,j,k]*ut(i,j,k)",  # noqa
                "CSE: Gw(i,j,k) = G[2,e,i,j,k]*ur(i,j,k) + G[4,e,i,j,k]*us(i,j,k) + G[5,e,i,j,k]*ut(i,j,k)",  # noqa

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
            lp.ValueArg("K", np.int32, approximately=1000),
            ],
            name="semlap", assumptions="K>=1")

    # print(lp.preprocess_kernel(knl, cse_ok=True))
    # 1/0
    #
    # print(knl)
    # 1/0
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
        #seq_knl = lp.add_prefetch(knl, "G", ["gi", "m", "j", "k"], "G[gi,e,m,j,k]", default_tag="l.auto")  # noqa
        # seq_knl = lp.add_prefetch(seq_knl, "D", ["m", "j"], default_tag="l.auto")
        #seq_knl = lp.add_prefetch(seq_knl, "u", ["i", "j", "k"], "u[*,i,j,k]", default_tag="l.auto")  # noqa
    else:
        seq_knl = knl

    knl = lp.split_iname(knl, "e", 16, outer_tag="g.0")  # , slabs=(0, 1))

    knl = lp.add_prefetch(knl, "G", ["gi", "m", "j", "k"], "G[gi,e,m,j,k]",
            default_tag="l.auto")
    knl = lp.add_prefetch(knl, "D", ["m", "j"],
            default_tag="l.auto")
    #knl = lp.add_prefetch(knl, "u", ["i", "j", "k"], "u[*,i,j,k]", default_tag="l.auto")  # noqa

    # knl = lp.split_iname(knl, "e_inner", 4, inner_tag="ilp")

    # print(seq_knl)
    # print(lp.preprocess_kernel(knl))
    # 1/0

    knl = lp.tag_inames(knl, dict(i="l.0", j="l.1"))

    kernel_gen = lp.generate_loop_schedules(knl,
            loop_priority=["m_fetch_G", "i_fetch_u"])
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000))

    K = 1000  # noqa
    lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
            op_count=K*(n*n*n*n*2*3 + n*n*n*5*3 + n**4 * 2*3)/1e9,
            op_label="GFlops",
            parameters={"K": K}, print_seq_code=True)


# TW: start here
def test_laplacian_lmem(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    n = 4

    from pymbolic import var
    K_sym = var("K")  # noqa

    field_shape = (K_sym, n, n, n)

    # K - run-time symbolic
    knl = lp.make_kernel(ctx.devices[0],
            "[K] -> {[i,j,k,e,m,o,gi]: 0<=i,j,k,m,o<%d and 0<=e<K and 0<=gi<6}" % n,
            [
                "ur(a,b,c) := sum_float32(@o, D[a,o]*u[e,o,b,c])",
                "us(a,b,c) := sum_float32(@o, D[b,o]*u[e,a,o,c])",
                "ut(a,b,c) := sum_float32(@o, D[c,o]*u[e,a,b,o])",

                "lap[e,i,j,k]  = "
                "  sum_float32(m, D[m,i]*(G[0,e,m,j,k]*ur(m,j,k) + G[1,e,m,j,k]*us(m,j,k) + G[2,e,m,j,k]*ut(m,j,k)))"  # noqa
                "+ sum_float32(m, D[m,j]*(G[1,e,i,m,k]*ur(i,m,k) + G[3,e,i,m,k]*us(i,m,k) + G[4,e,i,m,k]*ut(i,m,k)))"  # noqa
                "+ sum_float32(m, D[m,k]*(G[2,e,i,j,m]*ur(i,j,m) + G[4,e,i,j,m]*us(i,j,m) + G[5,e,i,j,m]*ut(i,j,m)))"  # noqa
                ],
            [
            lp.ArrayArg("u", dtype, shape=field_shape, order=order),
            lp.ArrayArg("lap", dtype, shape=field_shape, order=order),
            lp.ArrayArg("G", dtype, shape=(6,)+field_shape, order=order),
            lp.ArrayArg("D", dtype, shape=(n, n), order=order),
            lp.ValueArg("K", np.int32, approximately=1000),
            ],
            name="semlap", assumptions="K>=1")

    seq_knl = knl

    if 1:
        # original
        knl = lp.add_prefetch(knl, "u", ["i", "j", "k", "o"],
                default_tag="l.auto")
        knl = lp.precompute(knl, "ur", np.float32, ["a", "b", "c"],
                default_tag="l.auto")
        knl = lp.precompute(knl, "us", np.float32, ["a", "b", "c"],
                default_tag="l.auto")
        knl = lp.precompute(knl, "ut", np.float32, ["a", "b", "c"],
                default_tag="l.auto")
        knl = lp.split_iname(knl, "e", 16, outer_tag="g.0")  # , slabs=(0, 1))
        knl = lp.add_prefetch(knl, "D", ["m", "j", "k", "i"],
                default_tag="l.auto")
    else:
        # experiment
        # knl = lp.add_prefetch(knl, "u", ["i", "j", "k", "o"], default_tag="l.auto")
        knl = lp.precompute(knl, "eu", np.float32, ["b", "c"], default_tag="l.auto")
        knl = lp.precompute(knl, "ur", np.float32, ["b", "c"], default_tag="l.auto")
        knl = lp.precompute(knl, "us", np.float32, ["b", "c"], default_tag="l.auto")
        knl = lp.precompute(knl, "ut", np.float32, ["b", "c"], default_tag="l.auto")
        knl = lp.split_iname(knl, "e", 1, outer_tag="g.0")  # , slabs=(0, 1))
        knl = lp.add_prefetch(knl, "D", ["m", "j", "k", "i"], default_tag="l.auto")

    #knl = lp.add_prefetch(knl, "G", [2,3,4], default_tag="l.auto") # axis/argument indices on G  # noqa
    #knl = lp.add_prefetch(knl, "G", ["i", "j", "m", "k"], default_tag="l.auto") # axis/argument indices on G  # noqa
    # print(knl)
    # 1/0

    # knl = lp.split_iname(knl, "e_inner", 4, inner_tag="ilp")
#    knl = lp.join_dimensions(knl, ["i", "j"], "i_and_j")

    # print(seq_knl)
    # print(lp.preprocess_kernel(knl))
    # 1/0

# TW: turned this off since it generated:
# ValueError: cannot tag 'i_and_j'--not known
#    knl = lp.tag_inames(knl, dict(i_and_j="l.0", k="l.1"))

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000))

    K = 1000  # noqa
    lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
            op_count=K*(n*n*n*n*2*3 + n*n*n*5*3 + n**4 * 2*3)/1e9,
            op_label="GFlops",
            parameters={"K": K})

#TW:   ^^^^^^^^^^^^^^^ TypeError: auto_test_vs_ref() got an unexpected keyword argument 'print_seq_code'  # noqa


def test_laplacian_lmem_ilp(ctx_factory):
    # This does not lead to practical/runnable code (out of lmem), but it's an
    # excellent stress test for the code generator. :)

    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    n = 8

    from pymbolic import var
    K_sym = var("K")  # noqa

    field_shape = (K_sym, n, n, n)

    # K - run-time symbolic
    knl = lp.make_kernel(ctx.devices[0],
            "[K] -> {[i,j,k,e,m,o,gi]: 0<=i,j,k,m,o<%d and 0<=e<K }" % n,
            [
                "ur(i,j,k) := sum_float32(@o, D[i,o]*u[e,o,j,k])",
                "us(i,j,k) := sum_float32(@o, D[j,o]*u[e,i,o,k])",
                "ut(i,j,k) := sum_float32(@o, D[k,o]*u[e,i,j,o])",

                "lap[e,i,j,k]  = "
                "  sum_float32(m, D[m,i]*(G[0,e,m,j,k]*ur(m,j,k) + G[1,e,m,j,k]*us(m,j,k) + G[2,e,m,j,k]*ut(m,j,k)))"  # noqa
                "+ sum_float32(m, D[m,j]*(G[1,e,i,m,k]*ur(i,m,k) + G[3,e,i,m,k]*us(i,m,k) + G[4,e,i,m,k]*ut(i,m,k)))"  # noqa
                "+ sum_float32(m, D[m,k]*(G[2,e,i,j,m]*ur(i,j,m) + G[4,e,i,j,m]*us(i,j,m) + G[5,e,i,j,m]*ut(i,j,m)))"  # noqa
                ],
            [
            lp.ArrayArg("u", dtype, shape=field_shape, order=order),
            lp.ArrayArg("lap", dtype, shape=field_shape, order=order),
            lp.ArrayArg("G", dtype, shape=(6,)+field_shape, order=order),
            lp.ArrayArg("D", dtype, shape=(n, n), order=order),
            lp.ValueArg("K", np.int32, approximately=1000),
            ],
            name="semlap", assumptions="K>=1")

    # Must act on u first, otherwise stencil becomes crooked and
    # footprint becomes non-convex.

    knl = lp.split_iname(knl, "e", 16, outer_tag="g.0")  # , slabs=(0, 1))
    knl = lp.split_iname(knl, "e_inner", 4, inner_tag="ilp")

    knl = lp.add_prefetch(knl, "u", [1, 2, 3, "e_inner_inner"], default_tag="l.auto")

    knl = lp.precompute(knl, "ur", np.float32, [0, 1, 2, "e_inner_inner"], default_tag="l.auto")  # noqa
    knl = lp.precompute(knl, "us", np.float32, [0, 1, 2, "e_inner_inner"], default_tag="l.auto")  # noqa
    knl = lp.precompute(knl, "ut", np.float32, [0, 1, 2, "e_inner_inner"], default_tag="l.auto")  # noqa

    knl = lp.add_prefetch(knl, "G", ["m", "i", "j", "k", "e_inner_inner"], default_tag="l.auto")  # noqa
    knl = lp.add_prefetch(knl, "D", ["m", "j"], default_tag="l.auto")

    # print(seq_knl)
    # 1/0

    knl = lp.tag_inames(knl, dict(i="l.0", j="l.1"))

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000))

    for knl in kernel_gen:
        print(lp.generate_code(knl))


def test_advect(ctx_factory):
    1/0  # not ready

    dtype = np.float32
    ctx = ctx_factory()

    order = "C"

    N = 8  # noqa

    from pymbolic import var
    K_sym = var("K")  # noqa

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
                "CSE: Vr(i,j,k) = G[0,e,i,j,k]*u[e,i,j,k] + G[1,e,i,j,k]*v[e,i,j,k] + G[2,e,i,j,k]*w[e,i,j,k]",  # noqa
                "CSE: Vs(i,j,k) = G[3,e,i,j,k]*u[e,i,j,k] + G[4,e,i,j,k]*v[e,i,j,k] + G[5,e,i,j,k]*w[e,i,j,k]",  # noqa
                "CSE: Vt(i,j,k) = G[6,e,i,j,k]*u[e,i,j,k] + G[7,e,i,j,k]*v[e,i,j,k] + G[8,e,i,j,k]*w[e,i,j,k]",  # noqa

                # form nonlinear term on integration nodes
                "Nu[e,i,j,k] = Vr(i,j,k)*ur(i,j,k)+Vs(i,j,k)*us(i,j,k)+Vt(i,j,k)*ut(i,j,k)",  # noqa
                "Nv[e,i,j,k] = Vr(i,j,k)*vr(i,j,k)+Vs(i,j,k)*vs(i,j,k)+Vt(i,j,k)*vt(i,j,k)",  # noqa
                "Nw[e,i,j,k] = Vr(i,j,k)*wr(i,j,k)+Vs(i,j,k)*ws(i,j,k)+Vt(i,j,k)*wt(i,j,k)",  # noqa
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
            lp.ValueArg("K",  np.int32, approximately=1000),
            ],
            name="sem_advect", assumptions="K>=1")

    print(knl)
    1/0

    seq_knl = knl

    knl = lp.split_iname(knl, "e", 16, outer_tag="g.0")  # , slabs=(0, 1))

    knl = lp.tag_inames(knl, dict(i="l.0", j="l.1"))

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000), kill_level_min=5)

    K = 1000  # noqa
    lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
            op_count=0,
            op_label="GFlops",
            parameters={"K": K}, print_seq_code=True,)


def test_advect_dealias(ctx_factory):
    1/0  # not ready

    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    N = 8  # noqa
    M = 8  # noqa

    from pymbolic import var
    K_sym = var("K")  # noqa

    field_shape = (N, N, N, K_sym)
    interim_field_shape = (M, M, M, K_sym)  # noqa

    # 1. direction-by-direction similarity transform on u
    # 2. invert diagonal
    # 3. transform back (direction-by-direction)

    # K - run-time symbolic
    knl = lp.make_kernel(ctx.devices[0],
            "[K] -> {[i,ip,j,jp,k,kp,m,e]: 0<=i,j,k,m<%d AND 0<=o,ip,jp,kp<%d 0<=e<K}" %M %N  # noqa
            [

                # interpolate u to integration nodes
                "CSE:  u0[i,jp,kp,e] = sum_float32(@o, I[i,o]*u[o,jp,kp,e])",
                "CSE:  u1[i,j,kp,e]  = sum_float32(@o, I[j,o]*u0[i,o,kp,e])",
                "CSE:  Iu[i,j,k,e]   = sum_float32(@o, I[k,o]*u1[i,j,o,e])",

                # differentiate u on integration nodes
                "CSE:  Iur[i,j,k,e]  = sum_float32(@m, D[i,m]*Iu[m,j,k,e])",
                "CSE:  Ius[i,j,k,e]  = sum_float32(@m, D[j,m]*Iu[i,m,k,e])",
                "CSE:  Iut[i,j,k,e]  = sum_float32(@m, D[k,m]*Iu[i,j,m,e])",

                # interpolate v to integration nodes
                "CSE:  v0[i,jp,kp,e] = sum_float32(@o, I[i,o]*v[o,jp,kp,e])",
                "CSE:  v1[i,j,kp,e]  = sum_float32(@o, I[j,o]*v0[i,o,kp,e])",
                "CSE:  Iv[i,j,k,e]   = sum_float32(@o, I[k,o]*v1[i,j,o,e])",

                # differentiate v on integration nodes
                "CSE:  Ivr[i,j,k,e]  = sum_float32(@m, D[i,m]*Iv[m,j,k,e])",
                "CSE:  Ivs[i,j,k,e]  = sum_float32(@m, D[j,m]*Iv[i,m,k,e])",
                "CSE:  Ivt[i,j,k,e]  = sum_float32(@m, D[k,m]*Iv[i,j,m,e])",

                # interpolate w to integration nodes
                "CSE:  w0[i,jp,kp,e] = sum_float32(@o, I[i,o]*w[o,jp,kp,e])",
                "CSE:  w1[i,j,kp,e]  = sum_float32(@o, I[j,o]*w0[i,o,kp,e])",
                "CSE:  Iw[i,j,k,e]   = sum_float32(@o, I[k,o]*w1[i,j,o,e])",

                # differentiate v on integration nodes
                "CSE:  Iwr[i,j,k,e]  = sum_float32(@m, D[i,m]*Iw[m,j,k,e])",
                "CSE:  Iws[i,j,k,e]  = sum_float32(@m, D[j,m]*Iw[i,m,k,e])",
                "CSE:  Iwt[i,j,k,e]  = sum_float32(@m, D[k,m]*Iw[i,j,m,e])",

                # find velocity in (r,s,t) coordinates
                # QUESTION: should I use CSE here ?
                "CSE: Vr[i,j,k,e] = G[i,j,k,0,e]*Iu[i,j,k,e] + G[i,j,k,1,e]*Iv[i,j,k,e] + G[i,j,k,2,e]*Iw[i,j,k,e]",  # noqa
                "CSE: Vs[i,j,k,e] = G[i,j,k,3,e]*Iu[i,j,k,e] + G[i,j,k,4,e]*Iv[i,j,k,e] + G[i,j,k,5,e]*Iw[i,j,k,e]",  # noqa
                "CSE: Vt[i,j,k,e] = G[i,j,k,6,e]*Iu[i,j,k,e] + G[i,j,k,7,e]*Iv[i,j,k,e] + G[i,j,k,8,e]*Iw[i,j,k,e]",  # noqa

                # form nonlinear term on integration nodes
                # QUESTION: should I use CSE here ?
                "<SE: Nu[i,j,k,e] = Vr[i,j,k,e]*Iur[i,j,k,e]+Vs[i,j,k,e]*Ius[i,j,k,e]+Vt[i,j,k,e]*Iut[i,j,k,e]",  # noqa
                "<SE: Nv[i,j,k,e] = Vr[i,j,k,e]*Ivr[i,j,k,e]+Vs[i,j,k,e]*Ivs[i,j,k,e]+Vt[i,j,k,e]*Ivt[i,j,k,e]",  # noqa
                "<SE: Nw[i,j,k,e] = Vr[i,j,k,e]*Iwr[i,j,k,e]+Vs[i,j,k,e]*Iws[i,j,k,e]+Vt[i,j,k,e]*Iwt[i,j,k,e]",  # noqa

                # L2 project Nu back to Lagrange basis
                "CSE: Nu2[ip,j,k,e]   = sum_float32(@m, V[ip,m]*Nu[m,j,k,e])",
                "CSE: Nu1[ip,jp,k,e]  = sum_float32(@m, V[jp,m]*Nu2[ip,m,k,e])",
                "INu[ip,jp,kp,e] = sum_float32(@m, V[kp,m]*Nu1[ip,jp,m,e])",

                # L2 project Nv back to Lagrange basis
                "CSE: Nv2[ip,j,k,e]   = sum_float32(@m, V[ip,m]*Nv[m,j,k,e])",
                "CSE: Nv1[ip,jp,k,e]  = sum_float32(@m, V[jp,m]*Nv2[ip,m,k,e])",
                "INv[ip,jp,kp,e] = sum_float32(@m, V[kp,m]*Nv1[ip,jp,m,e])",

                # L2 project Nw back to Lagrange basis
                "CSE: Nw2[ip,j,k,e]   = sum_float32(@m, V[ip,m]*Nw[m,j,k,e])",
                "CSE: Nw1[ip,jp,k,e]  = sum_float32(@m, V[jp,m]*Nw2[ip,m,k,e])",
                "INw[ip,jp,kp,e] = sum_float32(@m, V[kp,m]*Nw1[ip,jp,m,e])",

                ],
            [
            lp.ArrayArg("u",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("v",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("w",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("INu",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("INv",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("INw",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("D",   dtype, shape=(M, M), order=order),
            lp.ArrayArg("I",   dtype, shape=(M, N), order=order),
            lp.ArrayArg("V",   dtype, shape=(N, M), order=order),
            lp.ValueArg("K",  np.int32, approximately=1000),
            ],
            name="sem_advect", assumptions="K>=1")

    print(knl)
    1/0

    knl = lp.split_iname(knl, "e", 16, outer_tag="g.0")  # , slabs=(0, 1))

    knl = lp.tag_inames(knl, dict(i="l.0", j="l.1"))

    print(knl)
    # 1/0

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000), kill_level_min=5)

    K = 1000  # noqa
    lp.auto_test_vs_ref(knl, ctx, kernel_gen,
            op_count=0,
            op_label="GFlops",
            parameters={"K": K}, print_seq_code=True,)


def test_interp_diff(ctx_factory):
    1/0  # not ready
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    N = 8  # noqa
    M = 8  # noqa

    from pymbolic import var
    K_sym = var("K")  # noqa

    field_shape = (N, N, N, K_sym)
    interim_field_shape = (M, M, M, K_sym)

    # 1. direction-by-direction similarity transform on u
    # 2. invert diagonal
    # 3. transform back (direction-by-direction)

    # K - run-time symbolic
    knl = lp.make_kernel(ctx.devices[0],
            "[K] -> {[i,ip,j,jp,k,kp,e]: 0<=i,j,k<%d AND 0<=ip,jp,kp<%d 0<=e<K}" %M %N  # noqa
            [
                "[|i,jp,kp] <float32>  u1[i ,jp,kp,e] = sum_float32(ip, I[i,ip]*u [ip,jp,kp,e])",  # noqa
                "[|i,j ,kp] <float32>  u2[i ,j ,kp,e] = sum_float32(jp, I[j,jp]*u1[i ,jp,kp,e])",  # noqa
                "[|i,j ,k ] <float32>  u3[i ,j ,k ,e] = sum_float32(kp, I[k,kp]*u2[i ,j ,kp,e])",  # noqa
                "[|i,j ,k ] <float32>  Pu[i ,j ,k ,e] = P[i,j,k,e]*u3[i,j,k,e]",
                "[|i,j ,kp] <float32> Pu3[i ,j ,kp,e] = sum_float32(k, V[kp,k]*Pu[i ,j , k,e])",  # noqa
                "[|i,jp,kp] <float32> Pu2[i ,jp,kp,e] = sum_float32(j, V[jp,j]*Pu[i ,j ,kp,e])",  # noqa
                "Pu[ip,jp,kp,e] = sum_float32(i, V[ip,i]*Pu[i ,jp,kp,e])",
                ],
            [
            lp.ArrayArg("u",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("P",   dtype, shape=interim_field_shape, order=order),
            lp.ArrayArg("I",   dtype, shape=(M, N), order=order),
            lp.ArrayArg("V",   dtype, shape=(N, M), order=order),
            lp.ArrayArg("Pu",  dtype, shape=field_shape, order=order),
            lp.ValueArg("K",  np.int32, approximately=1000),
            ],
            name="sem_lap_precon", assumptions="K>=1")

    print(knl)
    1/0

    knl = lp.split_iname(knl, "e", 16, outer_tag="g.0")  # , slabs=(0, 1))

    knl = lp.tag_inames(knl, dict(i="l.0", j="l.1"))

    print(knl)
    # 1/0

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000), kill_level_min=5)

    K = 1000  # noqa

    lp.auto_test_vs_ref(knl, ctx, kernel_gen,
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

import numpy as np
import pyopencl as cl  # noqa
import loopy as lp

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests  # noqa

1/0  # see sem_reagan?


def test_tim2d(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    n = 8

    from pymbolic import var
    K_sym = var("K")  # noqa

    field_shape = (K_sym, n, n)

    # K - run-time symbolic
    knl = lp.make_kernel(ctx.devices[0],
            "[K] -> {[i,j,e,m,o,gi]: 0<=i,j,m,o<%d and 0<=e<K and 0<=gi<3}" % n,
           [
            "ur(a,b) := sum_float32(@o, D[a,o]*u[e,o,b])",
            "us(a,b) := sum_float32(@o, D[b,o]*u[e,a,o])",

            "lap[e,i,j]  = "
            "  sum_float32(m, D[m,i]*(G[0,e,m,j]*ur(m,j) + G[1,e,m,j]*us(m,j)))"
            "+ sum_float32(m, D[m,j]*(G[1,e,i,m]*ur(i,m) + G[2,e,i,m]*us(i,m)))"

            ],
            [
            lp.ArrayArg("u", dtype, shape=field_shape, order=order),
            lp.ArrayArg("lap", dtype, shape=field_shape, order=order),
            lp.ArrayArg("G", dtype, shape=(3,)+field_shape, order=order),
            # lp.ConstantArrayArg("D", dtype, shape=(n, n), order=order),
            lp.ArrayArg("D", dtype, shape=(n, n), order=order),
            # lp.ImageArg("D", dtype, shape=(n, n)),
            lp.ValueArg("K", np.int32, approximately=1000),
            ],
            name="semlap2D", assumptions="K>=1")

    unroll = 32  # noqa

    seq_knl = knl
    knl = lp.add_prefetch(knl, "D", ["m", "j", "i", "o"], default_tag="l.auto")
    knl = lp.add_prefetch(knl, "u", ["i", "j",  "o"], default_tag="l.auto")
    knl = lp.precompute(knl, "ur", np.float32, ["a", "b"], default_tag="l.auto")
    knl = lp.precompute(knl, "us", np.float32, ["a", "b"], default_tag="l.auto")
    knl = lp.split_iname(knl, "e", 1, outer_tag="g.0")  # , slabs=(0, 1))

    knl = lp.tag_inames(knl, dict(i="l.0", j="l.1"))
    knl = lp.tag_inames(knl, dict(o="unr"))
    knl = lp.tag_inames(knl, dict(m="unr"))


#    knl = lp.add_prefetch(knl, "G", [2,3], default_tag=None) # axis/argument indices on G  # noqa
    knl = lp.add_prefetch(knl, "G", [2, 3], default_tag="l.auto")  # axis/argument indices on G  # noqa

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000))

    K = 1000  # noqa
    lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
            op_count=K*(n*n*n*2*2 + n*n*2*3 + n**3 * 2*2)/1e9,
            op_label="GFlops",
            parameters={"K": K})

#TW:   ^^^^^^^^^^^^^^^ TypeError: auto_test_vs_ref() got an unexpected keyword argument 'print_seq_code'  # noqa


def test_red2d(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    n = 16

    from pymbolic import var
    K_sym = var("K")  # noqa

    field_shape = (K_sym, n, n)

    # K - run-time symbolic
    knl = lp.make_kernel(ctx.devices[0],
            "[K] -> {[i,j,e,m,o,gi]: 0<=i,j,m,o<%d and 0<=e<K and 0<=gi<3}" % n,
           [
            "ue(a,b) := u[e,a,b]",
            "ur(a,b) := sum_float32(@o, D[a,o]*ue(o,b))",
            "us(a,b) := sum_float32(@o, D[b,o]*ue(a,o))",
            "lap[e,i,j]  = "
            "  sum_float32(m, D[m,i]*(G[0,e,m,j]*ur(m,j)+G[1,e,m,j]*us(m,j)))"
            "+ sum_float32(m, D[m,j]*(G[1,e,i,m]*ur(i,m)+G[2,e,i,m]*us(i,m)))"
            ],
            [
            lp.ArrayArg("u", dtype, shape=field_shape, order=order),
            lp.ArrayArg("lap", dtype, shape=field_shape, order=order),
            lp.ArrayArg("G", dtype, shape=(3,)+field_shape, order=order),
            lp.ArrayArg("D", dtype, shape=(n, n), order=order),
            lp.ValueArg("K", np.int32, approximately=1000),
            ],
            name="semlap2D", assumptions="K>=1")

    unroll = 32  # noqa

    seq_knl = knl
    knl = lp.add_prefetch(knl, "D", ["m", "j", "i", "o"], default_tag="l.auto")
    knl = lp.add_prefetch(knl, "u", ["i", "j",  "o"], default_tag="l.auto")
    knl = lp.precompute(knl, "ue", np.float32, ["a", "b", "m"],
            default_tag="l.auto")
    knl = lp.precompute(knl, "ur", np.float32, ["a", "b"],
            default_tag="l.auto")
    knl = lp.precompute(knl, "us", np.float32, ["a", "b"],
            default_tag="l.auto")
    knl = lp.split_iname(knl, "e", 2, outer_tag="g.0")
    knl = lp.split_iname(knl, "j", n, inner_tag="l.0")  # , slabs=(0, 1))
    knl = lp.split_iname(knl, "i", n, inner_tag="l.1")  # , slabs=(0, 1))

    knl = lp.tag_inames(knl, dict(o="unr"))
    knl = lp.tag_inames(knl, dict(m="unr"))

    knl = lp.add_prefetch(knl, "G", [2, 3], default_tag="l.auto")  # axis/argument indices on G  # noqa

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000))

    K = 1000  # noqa
    lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
            op_count=K*((n**3)*2*2 + n*n*2*3 + (n**3)*2*2)/1e9,
            op_label="GFlops",
            parameters={"K": K})

#TW:   ^^^^^^^^^^^^^^^ TypeError: auto_test_vs_ref() got an unexpected keyword argument 'print_seq_code'  # noqa


def test_tim3d(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    n = 8

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
            "   sum_float32(m, D[m,i]*(G[0,e,m,j,k]*ur(m,j,k) + G[1,e,m,j,k]*us(m,j,k) + G[2,e,m,j,k]*ut(m,j,k)))"  # noqa
            " + sum_float32(m, D[m,j]*(G[1,e,i,m,k]*ur(i,m,k) + G[3,e,i,m,k]*us(i,m,k) + G[4,e,i,m,k]*ut(i,m,k)))"  # noqa
            " + sum_float32(m, D[m,k]*(G[2,e,i,j,m]*ur(i,j,m) + G[4,e,i,j,m]*us(i,j,m) + G[5,e,i,j,m]*ut(i,j,m)))"  # noqa
             ],
            [
            lp.ArrayArg("u", dtype, shape=field_shape, order=order),
            lp.ArrayArg("lap", dtype, shape=field_shape, order=order),

            lp.ArrayArg("G", dtype, shape=(6,)+field_shape, order=order),
            # lp.ConstantArrayArg("D", dtype, shape=(n, n), order=order),
            lp.ArrayArg("D", dtype, shape=(n, n), order=order),
            # lp.ImageArg("D", dtype, shape=(n, n)),
            lp.ValueArg("K", np.int32, approximately=1000),
            ],
             name="semlap3D", assumptions="K>=1")

    seq_knl = knl
    knl = lp.add_prefetch(knl, "D", ["m", "j", "i", "k", "o"], default_tag="l.auto")
    knl = lp.add_prefetch(knl, "u", ["i", "j",  "o", "k"], default_tag="l.auto")
    knl = lp.precompute(knl, "ur", np.float32, ["a", "b", "c"],
            default_tag="l.auto")
    knl = lp.precompute(knl, "us", np.float32, ["a", "b", "c"],
            default_tag="l.auto")
    knl = lp.precompute(knl, "ut", np.float32, ["a", "b", "c"],
            default_tag="l.auto")
    knl = lp.split_iname(knl, "e", 1, outer_tag="g.0")  # , slabs=(0, 1))
    knl = lp.split_iname(knl, "j", n, inner_tag="l.1")  # , slabs=(0, 1))
    knl = lp.split_iname(knl, "k", n, inner_tag="l.2")  # , slabs=(0, 1))
    knl = lp.split_iname(knl, "i", n, inner_tag="l.0")  # , slabs=(0, 1))

#    knl = lp.tag_inames(knl, dict(k_nner="unr"))

    knl = lp.tag_inames(knl, dict(o="unr"))
    knl = lp.tag_inames(knl, dict(m="unr"))
#    knl = lp.tag_inames(knl, dict(i="unr"))

    knl = lp.add_prefetch(knl, "G", [2, 3, 4], default_tag="l.auto")  # axis/argument indices on G  # noqa

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000))

    K = 4000  # noqa
    lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
            op_count=K*((n**4)*3*2 + (n**3)*5*3 + (n**4)*3*2)/1e9,
            op_label="GFlops",
            parameters={"K": K})

#TW:   ^^^^^^^^^^^^^^^ TypeError: auto_test_vs_ref() got an unexpected keyword argument 'print_seq_code'  # noqa


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

from __future__ import division

import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as cl_random
import loopy as lp

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests




def make_well_conditioned_dev_matrix(queue, shape, dtype=np.float32, 
        order="C", ran_factor=1, id_factor=5, inc_factor=0, od=0):
    if isinstance(shape, int):
        shape = (shape, shape)
    l = max(shape)
    eye_ish = id_factor*np.eye(l, k=od)
    if inc_factor:
        eye_ish[np.arange(l), np.arange(l)] = inc_factor*np.arange(l)
    ary = np.asarray(
        ran_factor*np.random.randn(*shape)
        + eye_ish[:shape[0], :shape[1]],
        dtype=dtype, order=order)

    return cl_array.to_device(queue, ary)




DO_CHECK = True

DEBUG_PREAMBLE = r"""
    #pragma OPENCL EXTENSION cl_amd_printf: enable
    #define MY_J (j_outer*64+j_inner_outer*16+j_inner_inner)
    #define MY_I (i_outer*16+i_inner)
    #define IFDIAG if (MY_I == MY_J)
    #define TST(S) if (MY_J == 144 && MY_I == 16-48) \
            for (int aa = 0; aa < 16: ++ab) \
                for (int bb = 0; bb < 16: ++bb) 
    """




def check_error(refsol, sol):
    if not DO_CHECK:
        return

    if sol.shape == 2:
        norm_order = "fro"
    else:
        norm_order = 2

    rel_err = la.norm(refsol-sol, norm_order)/la.norm(refsol, norm_order)
    if rel_err > 1e-5 or np.isinf(rel_err) or np.isnan(rel_err):
        if 1:
            import matplotlib.pyplot as pt
            pt.imshow(refsol-sol)
            pt.colorbar()
            pt.show()
        elif 0:
            print "---------------------------"
            print "ACTUAL"
            print "---------------------------"
            np.set_printoptions(threshold=1000000, linewidth=200)
            print sol[:16,:16]
            print "---------------------------"
            print "CORRECT"
            print "---------------------------"
            print refsol[:16,:16]
        raise RuntimeError("check failed, rel err=%g" % rel_err)



def test_sem(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    # n = get_suitable_size(ctx)

    # 0<=i,j,k,m<=N AND 0<=k<K
    
    # ur(i,j,k,e) = D(i,m)*u(m,j,k,e)
    # us(i,j,k,e) = D(j,m)*u(i,m,k,e)
    # ut(i,j,k,e) = D(k,m)*u(i,j,m,e)

    # (grad phi, grad u) = (Dr' Ds' Dt')*(G)*(Dr; Ds; Dt)
    # lap(i,j,k,e)  =  D(m,i)*(G(0,m,j,k,e)*ur(m,j,k,e) + G(1,m,j,k,e)*us(m,j,k,e) + G(2,m,j,k,e)*ut(m,j,k,e));
    # lap(i,j,k,e) +=  D(m,j)*(G(1,i,m,k,e)*ur(i,m,k,e) + G(3,i,m,k,e)*us(i,m,k,e) + G(4,i,m,k,e)*ut(i,m,k,e));
    # lap(i,j,k,e) +=  D(m,k)*(G(2,i,j,m,e)*ur(i,j,m,e) + G(4,i,j,m,e)*us(i,j,m,e) + G(5,i,j,m,e)*ut(i,j,m,e));


    # K - run-time symbolic
    from pymbolic import var
    K = var("K")
    n = 8
    knl = lp.make_kernel(ctx.devices[0],
            #"[K] -> {[i,j,k,ip,jp,kp,e,m,mp]: 0<=i,j,k,m,ip,jp,kp,mp<%d AND 0<=e<K}" % n,
            #[
                #"ur[e, k, j, i] = sum_float32(m, D[i, m]*u[e, k, j, m])",
                #"lap[e, kp, jp, ip] = sum_float32(mp, D[ip, mp]*(G[e, 0, kp, jp, mp]*ur[e, kp, jp, mp]))"
                #],
            "[K] -> {[i,j,k,e,m,mp]: 0<=i,j,k,m,mp<%d AND 0<=e<K}" % n,
            [
                "lap[e, k, j, i] = "
                    "sum_float32(mp, D[i, mp]*(G[e, 0, k, j, mp]*"
                    "cse(sum_float32(m, D[mp, m]*u[e, k, j, m]), build_ur)))"
                ],
            [
            lp.ArrayArg("u",   dtype, shape=(K, n, n, n), order=order),
            lp.ArrayArg("ur",  dtype, shape=(K, n, n, n), order=order),
            lp.ArrayArg("lap", dtype, shape=(K, n, n, n), order=order),
            lp.ArrayArg("G",   dtype, shape=(K, 6, n, n, n), order=order),
            lp.ArrayArg("D",   dtype, shape=(n, n), order=order),
            lp.ScalarArg("K",  np.int32, approximately=1000),
            ],
            name="semlap", assumptions="K>=1")

    knl = lp.split_dimension(knl, "e", 16, outer_tag="g.0")#, slabs=(0, 1))
    #knl = lp.split_dimension(knl, "e_inner", 4, inner_tag="ilp")
    knl = lp.tag_dimensions(knl, dict(i="l.0", j="l.1"))
    #knl = lp.realize_cse(knl, "build_ur", np.float32, ["j", "k"])
    knl = lp.realize_cse(knl, "build_ur", np.float32, ["j", "k", "mp"])
    print knl
    #1/0

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000), kill_level_min=5)

    a = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    b = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    c = cl_array.empty_like(a)
    refsol = np.dot(a.get(), b.get())

    def launcher(kernel, gsize, lsize, check):
        evt = kernel(queue, gsize(), lsize(), a.data, b.data, c.data,
                g_times_l=True)

        if check:
            check_error(refsol, c.get())

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3)




def test_sem_nd(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    # n = get_suitable_size(ctx)

    # 0<=i,j,k,m<=N AND 0<=k<K

    # ur(i,j,k,e) = D(i,m)*u(m,j,k,e)
    # us(i,j,k,e) = D(j,m)*u(i,m,k,e)
    # ut(i,j,k,e) = D(k,m)*u(i,j,m,e)

    # (grad phi, grad u) = (Dr' Ds' Dt')*(G)*(Dr; Ds; Dt)
    # lap(i,j,k,e)  =  D(m,i)*(G(0,m,j,k,e)*ur(m,j,k,e) + G(1,m,j,k,e)*us(m,j,k,e) + G(2,m,j,k,e)*ut(m,j,k,e));
    # lap(i,j,k,e) +=  D(m,j)*(G(1,i,m,k,e)*ur(i,m,k,e) + G(3,i,m,k,e)*us(i,m,k,e) + G(4,i,m,k,e)*ut(i,m,k,e));
    # lap(i,j,k,e) +=  D(m,k)*(G(2,i,j,m,e)*ur(i,j,m,e) + G(4,i,j,m,e)*us(i,j,m,e) + G(5,i,j,m,e)*ut(i,j,m,e));

    from pymbolic import var
    K_sym, G_sym, u_sym, D_sym, m_sym, i_sym, j_sym, k_sym, e_sym = [
            var(i) for i in "KGuDmijke"]

    sym_lookup = {
            (0,0): 0,
            (0,1): 1,
            (0,2): 2,
            (1,1): 3,
            (1,2): 4,
            (2,2): 5,
            }

    for i, j in sym_lookup.keys():
        sym_lookup[j, i] = sym_lookup[i, j]

    dim = 3
    local_derivatives = []

    ijk = [i_sym, j_sym, k_sym]
    from loopy.symbolic import Reduction
    from loopy.kernel import parse_reduction_op
    for axis in range(dim):
        u_index = [i_sym, j_sym, k_sym, e_sym]
        u_index[axis] = m_sym
        local_derivatives.append(
                Reduction(
                    parse_reduction_op("sum_float32"),
                    ("m",),
                    D_sym[ijk[axis], m_sym]
                    * u_sym[tuple(u_index)]))

    #for axis in range(dim):
    #div 
    for ld in local_derivatives:
        print ld

    1/0




    field_shape = (K_sym,) + dim*(n,)
    # K - run-time symbolic
    n = 8
    knl = lp.make_kernel(ctx.devices[0],
            #"[K] -> {[i,j,k,ip,jp,kp,e,m,mp]: 0<=i,j,k,m,ip,jp,kp,mp<%d AND 0<=e<K}" % n,
            #[
                #"ur[e, k, j, i] = sum_float32(m, D[i, m]*u[e, k, j, m])",
                #"lap[e, kp, jp, ip] = sum_float32(mp, D[ip, mp]*(G[e, 0, kp, jp, mp]*ur[e, kp, jp, mp]))"
                #],
            "[K] -> {[i,j,k,e,m,mp]: 0<=i,j,k,m,mp<%d AND 0<=e<K}" % n,
            [
                "lap[e, k, j, i] = "
                    "sum_float32(mp, D[i, mp]*(G[e, 0, k, j, mp]*"
                    "cse(sum_float32(m, D[mp, m]*u[e, k, j, m]), build_ur)))"
                ],
            [
            lp.ArrayArg("u",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("ur",  dtype, shape=field_shape, order=order),
            lp.ArrayArg("lap", dtype, shape=field_shape, order=order),
            lp.ArrayArg("G",   dtype, shape=field_shape + (6,), order=order),
            lp.ArrayArg("D",   dtype, shape=(n, n), order=order),
            lp.ScalarArg("K",  np.int32, approximately=1000),
            ],
            name="semlap", assumptions="K>=1")

    knl = lp.split_dimension(knl, "e", 16, outer_tag="g.0")#, slabs=(0, 1))
    #knl = lp.split_dimension(knl, "e_inner", 4, inner_tag="ilp")
    knl = lp.tag_dimensions(knl, dict(i="l.0", j="l.1"))
    #knl = lp.realize_cse(knl, "build_ur", np.float32, ["j", "k"])
    knl = lp.realize_cse(knl, "build_ur", np.float32, ["j", "k", "mp"])
    print knl
    #1/0

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000), kill_level_min=5)

    a = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    b = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    c = cl_array.empty_like(a)
    refsol = np.dot(a.get(), b.get())

    def launcher(kernel, gsize, lsize, check):
        evt = kernel(queue, gsize(), lsize(), a.data, b.data, c.data,
                g_times_l=True)

        if check:
            check_error(refsol, c.get())

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3)




def test_sem_3d(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = 8

    from pymbolic import var
    K_sym = var("K")

    field_shape = (n, n, n, K_sym)

    # K - run-time symbolic
    n = 8
    knl = lp.make_kernel(ctx.devices[0],
            "[K] -> {[i,j,k,e,m,mp]: 0<=i,j,k,m<%d AND 0<=e<K}" % n,
            [
                "[|i,j,k] <float32> ur[i,j,k,e] = sum_float32(m, D[i,m]*u[m,j,k,e])",
                "[|i,j,k] <float32> us[i,j,k,e] = sum_float32(m, D[j,m]*u[i,m,k,e])",
                "[|i,j,k] <float32> ut[i,j,k,e] = sum_float32(m, D[k,m]*u[i,j,m,e])",

                "lap[i,j,k,e]  = "
                "  sum_float32(m, D[m,i]*(G[0,m,j,k,e]*ur[m,j,k,e] + G[1,m,j,k,e]*us[m,j,k,e] + G[2,m,j,k,e]*ut[m,j,k,e]))"
                "+ sum_float32(m, D[m,j]*(G[1,i,m,k,e]*ur[i,m,k,e] + G[3,i,m,k,e]*us[i,m,k,e] + G[4,i,m,k,e]*ut[i,m,k,e]))"
                "+ sum_float32(m, D[m,k]*(G[2,i,j,m,e]*ur[i,j,m,e] + G[4,i,j,m,e]*us[i,j,m,e] + G[5,i,j,m,e]*ut[i,j,m,e]))"
                ],
            [
            lp.ArrayArg("u",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("lap", dtype, shape=field_shape, order=order),
            lp.ArrayArg("G",   dtype, shape=(6,) + field_shape, order=order),
            lp.ArrayArg("D",   dtype, shape=(n, n), order=order),
            lp.ScalarArg("K",  np.int32, approximately=1000),
            ],
            name="semlap", assumptions="K>=1")

    print knl
    1/0

    knl = lp.split_dimension(knl, "e", 16, outer_tag="g.0")#, slabs=(0, 1))
    #knl = lp.split_dimension(knl, "e_inner", 4, inner_tag="ilp")
    knl = lp.tag_dimensions(knl, dict(i="l.0", j="l.1"))
    #knl = lp.realize_cse(knl, "build_ur", np.float32, ["j", "k"])
    knl = lp.realize_cse(knl, "build_ur", np.float32, ["j", "k", "mp"])
    print knl
    #1/0

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(K=1000), kill_level_min=5)

    a = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    b = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    c = cl_array.empty_like(a)
    refsol = np.dot(a.get(), b.get())

    def launcher(kernel, gsize, lsize, check):
        evt = kernel(queue, gsize(), lsize(), a.data, b.data, c.data,
                g_times_l=True)

        if check:
            check_error(refsol, c.get())

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3)




if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the
    # tests.
    import pyopencl as cl

    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

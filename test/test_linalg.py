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




def get_suitable_size(ctx):
    dev, = ctx.devices
    if dev.type == cl.device_type.CPU:
        return 160
    else:
        return 1600




def test_axpy(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = 20*1024**2

    knl = lp.make_kernel(ctx.devices[0],
            "[n] -> {[i]: 0<=i<n}",
            [
                "z[i] = a*x[i]+b*y[i]"
                ],
            [
                lp.ScalarArg("a", dtype),
                lp.ArrayArg("x", dtype, shape="n,"),
                lp.ScalarArg("b", dtype),
                lp.ArrayArg("y", dtype, shape="n,"),
                lp.ArrayArg("z", dtype, shape="n,"),
                lp.ScalarArg("n", np.int32, approximately=n),
                ],
            name="matmul")

    def variant_cpu(knl):
        unroll = 16
        block_size = unroll*4096
        knl = lp.split_dimension(knl, "i", block_size, outer_tag="g.0", slabs=(0, 1))
        knl = lp.split_dimension(knl, "i_inner", unroll, inner_tag="unr")
        return knl

    def variant_gpu(knl):
        unroll = 4
        block_size = 256
        knl = lp.split_dimension(knl, "i", unroll*block_size, outer_tag="g.0", slabs=(0, 1))
        knl = lp.split_dimension(knl, "i_inner", block_size, outer_tag="unr", inner_tag="l.0")
        return knl

    a = cl_random.rand(queue, n, dtype=dtype, luxury=2)
    b = cl_random.rand(queue, n, dtype=dtype, luxury=2)
    c = cl_array.zeros_like(a)
    refsol = (2*a+3*b).get()

    for variant in [variant_cpu, variant_gpu]:
        kernel_gen = lp.generate_loop_schedules(variant(knl),
                loop_priority=["i_inner_outer"])
        kernel_gen = lp.check_kernels(kernel_gen, dict(n=n))

        def launcher(kernel, gsize, lsize, check):
            evt = kernel(queue, gsize(n), lsize(n), 2, a.data, 3, b.data, c.data, n,
                    g_times_l=True)

            if check:
                check_error(refsol, c.get())

            return evt

        lp.drive_timing_run(kernel_gen, queue, launcher, 5*n)





def test_transpose(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = get_suitable_size(ctx)

    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j]: 0<=i,j<%d}" % n,
            [
                "b[i, j] = a[j, i]"
                ],
            [
                lp.ArrayArg("a", dtype, shape=(n, n), order=order),
                lp.ArrayArg("b", dtype, shape=(n, n), order=order),
                ],
            name="transpose")

    seq_knl = knl

    knl = lp.split_dimension(knl, "i", 16,
            outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_dimension(knl, "j", 16,
            outer_tag="g.1", inner_tag="l.0")
    knl = lp.add_prefetch(knl, 'a', ["i_inner", "j_inner"])

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, {})

    a = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    b = cl_array.empty_like(a)
    refsol = a.get().T.copy()

    def launcher(kernel, gsize, lsize, check):
        evt = kernel(queue, gsize(), lsize(), a.data, b.data,
                g_times_l=True)

        if check:
            check_error(refsol, b.get())

        return evt

    #lp.drive_timing_run(kernel_gen, queue, launcher, 0)
    lp.auto_test_vs_seq(seq_knl, ctx, kernel_gen,
            op_count=dtype.itemsize*n**2*2/1e9, op_label="GByte",
            parameters={}, print_seq_code=True)




def test_plain_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = get_suitable_size(ctx)

    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = sum_float32(k, a[i, k]*b[k, j])"
                ],
            [
                lp.ArrayArg("a", dtype, shape=(n, n), order=order),
                lp.ArrayArg("b", dtype, shape=(n, n), order=order),
                lp.ArrayArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    knl = lp.split_dimension(knl, "i", 16,
            outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_dimension(knl, "j", 16,
            outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_dimension(knl, "k", 16)
    knl = lp.add_prefetch(knl, 'a', ["k_inner", "i_inner"])
    knl = lp.add_prefetch(knl, 'b', ["j_inner", "k_inner", ])

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, {})

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





def test_variable_size_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = get_suitable_size(ctx)

    knl = lp.make_kernel(ctx.devices[0],
            "[n] -> {[i,j,k]: 0<=i,j,k<n}",
            [
                "label: c[i, j] = sum_float32(k, cse(a[i, k], lhsmat)*cse(b[k, j], rhsmat))"
                ],
            [
                lp.ArrayArg("a", dtype, shape=(n, n), order=order),
                lp.ArrayArg("b", dtype, shape=(n, n), order=order),
                lp.ArrayArg("c", dtype, shape=(n, n), order=order),
                lp.ScalarArg("n", np.int32, approximately=n),
                ],
            name="matmul", assumptions="n >= 16")

    knl = lp.split_dimension(knl, "i", 16,
            outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_dimension(knl, "j", 8,
            outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_dimension(knl, "k", 32)

    knl = lp.realize_cse(knl, "lhsmat", dtype, ["k_inner", "i_inner"])
    knl = lp.realize_cse(knl, "rhsmat", dtype, ["j_inner", "k_inner"])

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(n=n))

    a = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    b = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    c = cl_array.empty_like(a)
    refsol = np.dot(a.get(), b.get())

    def launcher(kernel, gsize, lsize, check):
        evt = kernel(queue, gsize(n), lsize(n), a.data, b.data, c.data, n,
                g_times_l=True)

        if check:
            check_error(refsol, c.get())

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3)




def test_rank_one(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = int(get_suitable_size(ctx)**(2.7/2))

    knl = lp.make_kernel(ctx.devices[0],
            "[n] -> {[i,j]: 0<=i,j<n}",
            [
                "label: c[i, j] = a[i]*b[j]"
                ],
            [
                lp.ArrayArg("a", dtype, shape=(n,), order=order),
                lp.ArrayArg("b", dtype, shape=(n,), order=order),
                lp.ArrayArg("c", dtype, shape=(n, n), order=order),
                lp.ScalarArg("n", np.int32, approximately=n),
                ],
            name="rank_one", assumptions="n >= 16")

    def variant_1(knl):
        knl = lp.add_prefetch(knl, "a")
        knl = lp.add_prefetch(knl, "b")
        return knl

    def variant_2(knl):
        knl = lp.split_dimension(knl, "i", 16,
                outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_dimension(knl, "j", 16,
                outer_tag="g.1", inner_tag="l.1")

        knl = lp.add_prefetch(knl, "a")
        knl = lp.add_prefetch(knl, "b")
        return knl

    def variant_3(knl):
        # Throws an error--doesn't use all hardware axis.
        # Probably the right thing to do.

        knl = lp.split_dimension(knl, "i", 16,
                outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_dimension(knl, "j", 16,
                outer_tag="g.1", inner_tag="l.1")

        knl = lp.add_prefetch(knl, "a", ["i_inner"])
        knl = lp.add_prefetch(knl, "b", ["j_inner"])
        return knl

    def variant_4(knl):
        knl = lp.split_dimension(knl, "i", 256,
                outer_tag="g.0", slabs=(0, 1))
        knl = lp.split_dimension(knl, "j", 256,
                outer_tag="g.1", slabs=(0, 1))

        knl = lp.add_prefetch(knl, "a", ["i_inner"])
        knl = lp.add_prefetch(knl, "b", ["j_inner"])

        knl = lp.split_dimension(knl, "i_inner", 16,
                inner_tag="l.0")
        knl = lp.split_dimension(knl, "j_inner", 16,
                inner_tag="l.1")

        knl = lp.split_dimension(knl, "j_inner_0", 16,
                outer_tag="l.1", inner_tag="l.0")
        knl = lp.split_dimension(knl, "i_inner_0", 16,
                outer_tag="l.1", inner_tag="l.0")
        return knl

    for variant in [variant_1, variant_2, variant_4]:

        kernel_gen = lp.generate_loop_schedules(variant(knl))
        kernel_gen = lp.check_kernels(kernel_gen, dict(n=n))

        a = cl_random.rand(queue, n, dtype=dtype)
        b = cl_random.rand(queue, n, dtype=dtype)
        refsol = a.get()[:, np.newaxis] * b.get()
        c = cl_array.empty(queue, refsol.shape, refsol.dtype)

        def launcher(kernel, gsize, lsize, check):
            evt = kernel(queue, gsize(n), lsize(n), a.data, b.data, c.data, n,
                    g_times_l=True)

            if check:
                check_error(refsol, c.get())

            return evt

        lp.drive_timing_run(kernel_gen, queue, launcher, n**2)




def test_troublesome_premagma_fermi_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = 6*16*2

    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = sum_float32(k, a[i, k]*b[k, j])"
                ],
            [
                lp.ArrayArg("a", dtype, shape=(n, n), order=order),
                lp.ArrayArg("b", dtype, shape=(n, n), order=order),
                lp.ArrayArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    i_reg = 2
    j_reg = 2
    i_chunks = 16
    j_chunks = 16
    knl = lp.split_dimension(knl, "i", i_reg*i_chunks, outer_tag="g.0")
    knl = lp.split_dimension(knl, "i_inner", i_reg, outer_tag="l.0", inner_tag="ilp")
    knl = lp.split_dimension(knl, "j", j_reg*j_chunks, outer_tag="g.1")
    knl = lp.split_dimension(knl, "j_inner", j_reg, outer_tag="l.1", inner_tag="ilp")
    knl = lp.split_dimension(knl, "k", 16)
    knl = lp.add_prefetch(knl, 'a', ["k_inner", "i_inner_inner", "i_inner_outer"])

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(n=n))

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




def test_intel_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = 6*16

    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = sum_float32(k, a[i, k]*b[k, j])"
                ],
            [
                lp.ArrayArg("a", dtype, shape=(n, n), order=order),
                lp.ArrayArg("b", dtype, shape=(n, n), order=order),
                lp.ArrayArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    i_reg = 4
    j_reg = 4
    i_chunks = 16
    j_chunks = 16
    knl = lp.split_dimension(knl, "i", i_reg*i_chunks, outer_tag="g.0")
    knl = lp.split_dimension(knl, "i_inner", i_reg, outer_tag="l.0", inner_tag="ilp")
    knl = lp.split_dimension(knl, "j", j_reg*j_chunks, outer_tag="g.1")
    knl = lp.split_dimension(knl, "j_inner", j_reg, outer_tag="l.1", inner_tag="ilp")
    knl = lp.split_dimension(knl, "k", 16)
    #knl = lp.split_dimension(knl, "k_inner", 8, outer_tag="unr")

    knl = lp.add_prefetch(knl, 'a', ["i_inner_inner", "k_inner", "i_inner_outer"])
    knl = lp.add_prefetch(knl, 'b', ["j_inner_inner", "k_inner", "j_inner_outer"])

    # FIXME: Grouped prefetch
    #knl = lp.add_prefetch(knl, 'a', ["k_inner", ("i_inner_inner", "i_inner_outer")])
    #knl = lp.add_prefetch(knl, 'b', ["k_inner", ("j_inner_inner", "j_inner_outer"),])

    kernel_gen = lp.generate_loop_schedules(knl)
    #hints=["k_outer", "k_inner_outer", "k_inner_inner"]
    kernel_gen = lp.check_kernels(kernel_gen, dict(n=n))

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





def test_magma_fermi_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = 6*16*16

    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = sum_float32(k, a[i, k]*b[k, j])"
                ],
            [
                lp.ImageArg("a", dtype, 2),
                lp.ImageArg("b", dtype, 2),
                lp.ArrayArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    i_reg = 4
    j_reg = 4
    i_chunks = 16
    j_chunks = 16
    knl = lp.split_dimension(knl, "i", i_reg*i_chunks, outer_tag="g.0")
    knl = lp.split_dimension(knl, "i_inner", i_reg, outer_tag="l.0", inner_tag="ilp")
    knl = lp.split_dimension(knl, "j", j_reg*j_chunks, outer_tag="g.1")
    knl = lp.split_dimension(knl, "j_inner", j_reg, outer_tag="l.1", inner_tag="ilp")
    knl = lp.split_dimension(knl, "k", 16)
    #knl = lp.split_dimension(knl, "k_inner", 8, outer_tag="unr")
    knl = lp.add_prefetch(knl, 'a', ["k_inner", ("i_inner_inner", "i_inner_outer")])
    knl = lp.add_prefetch(knl, 'b', ["k_inner", ("j_inner_inner", "j_inner_outer"),])

    kernel_gen = lp.generate_loop_schedules(knl)
    #hints=["k_outer", "k_inner_outer", "k_inner_inner"]
    kernel_gen = lp.check_kernels(kernel_gen, dict(n=n))

    a = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    b = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    a_img = cl.image_from_array(ctx, a.get(), 1)
    b_img = cl.image_from_array(ctx, b.get(), 1)
    c = cl_array.empty_like(a)
    refsol = np.dot(a.get(), b.get())

    def launcher(kernel, gsize, lsize, check):
        evt = kernel(queue, gsize(), lsize(), a_img, b_img, c.data,
                g_times_l=True)

        if check:
            check_error(refsol, c.get())

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3)





def test_image_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = get_suitable_size(ctx)

    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = sum_float32(k, a[i, k]*b[k, j])"
                ],
            [
                lp.ImageArg("a", dtype, 2),
                lp.ImageArg("b", dtype, 2),
                lp.ArrayArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    knl = lp.split_dimension(knl, "i", 16, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_dimension(knl, "j", 16, outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_dimension(knl, "k", 32)
    # conflict-free
    knl = lp.add_prefetch(knl, 'a', ["i_inner", "k_inner"])
    knl = lp.add_prefetch(knl, 'b', ["j_inner", "k_inner"])

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(n=n))

    a = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    b = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    c = cl_array.empty_like(a)
    refsol = np.dot(a.get(), b.get())
    a_img = cl.image_from_array(ctx, a.get(), 1)
    b_img = cl.image_from_array(ctx, b.get(), 1)

    def launcher(kernel, gsize, lsize, check):
        evt = kernel(queue, gsize(), lsize(), a_img, b_img, c.data,
                g_times_l=True)

        if check:
            check_error(refsol, c.get())

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3)




def test_image_matrix_mul_ilp(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = 2*get_suitable_size(ctx)

    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = sum_float32(k, a[i, k]*b[k, j])"
                ],
            [
                lp.ImageArg("a", dtype, 2),
                lp.ImageArg("b", dtype, 2),
                lp.ArrayArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    ilp = 4
    knl = lp.split_dimension(knl, "i", 2, outer_tag="g.0", inner_tag="l.1")
    j_inner_split = 16
    knl = lp.split_dimension(knl, "j", ilp*j_inner_split, outer_tag="g.1")
    knl = lp.split_dimension(knl, "j_inner", j_inner_split, outer_tag="ilp", inner_tag="l.0")
    knl = lp.split_dimension(knl, "k", 2)
    # conflict-free
    knl = lp.add_prefetch(knl, 'a', ["i_inner", "k_inner"])
    knl = lp.add_prefetch(knl, 'b', ["j_inner_outer", "j_inner_inner", "k_inner"],
            ["b_j_io", "b_j_ii", "b_k_i"])
    knl = lp.join_dimensions(knl, ["b_j_io", "b_j_ii"])

    #print lp.preprocess_kernel(knl)
    #1/0

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(n=n))

    a = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    b = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order)
    c = cl_array.empty_like(a)

    refsol = np.dot(a.get(), b.get())
    a_img = cl.image_from_array(ctx, a.get(), 1)
    b_img = cl.image_from_array(ctx, b.get(), 1)

    def launcher(kernel, gsize, lsize, check):
        evt = kernel(queue, gsize(), lsize(), a_img, b_img, c.data,
                g_times_l=True)

        if check:
            check_error(refsol, c.get())

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3)





def test_fancy_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    order = "C"

    n = get_suitable_size(ctx)

    knl = lp.make_kernel(ctx.devices[0],
            "[n] -> {[i,j,k]: 0<=i,j,k<n }",
            [
                "c[i, j] = sum_float32(k, a[i, k]*b[k, j])"
                ],
            [
                lp.ArrayArg("a", dtype, shape="(n, n)", order=order),
                lp.ArrayArg("b", dtype, shape="(n, n)", order=order),
                lp.ArrayArg("c", dtype, shape="(n, n)", order=order),
                lp.ScalarArg("n", np.int32, approximately=1000),
                ], name="fancy_matmul", assumptions="n>=1")

    knl = lp.split_dimension(knl, "i", 16, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_dimension(knl, "j", 16, outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_dimension(knl, "k", 16, slabs=(0,1))
    knl = lp.add_prefetch(knl, 'a', ["i_inner", "k_inner"])
    knl = lp.add_prefetch(knl, 'b', ["k_inner", "j_inner"])

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(n=n))

    a = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order, 
            ran_factor=0)
    b = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order,
            ran_factor=0)
    c = cl_array.empty_like(a)
    refsol = np.dot(a.get(), b.get())

    def launcher(kernel, gsize, lsize, check):
        evt = kernel(queue, gsize(n), lsize(n), a.data, b.data, c.data, n,
                g_times_l=True)

        if check:
            check_error(refsol, c.get())

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3)




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

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

    n = get_suitable_size(ctx)**2

    knl = lp.LoopKernel(ctx.devices[0],
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

    unroll = 16
    block_size = 256
    knl = lp.split_dimension(knl, "i", unroll*block_size, outer_tag="g.0")
    knl = lp.split_dimension(knl, "i_inner", block_size, outer_tag="unr", inner_tag="l.0")
    assert knl.get_problems({"n": n})[0] <= 2

    kernel_gen = (lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl))

    a = cl_random.rand(queue, n, dtype=dtype)
    b = cl_random.rand(queue, n, dtype=dtype)
    c = cl_array.empty_like(a)
    refsol = (2*a+3*b).get()

    def launcher(kernel, gsize, lsize, check):
        evt = kernel(queue, gsize(n), lsize(n), 2, a.data, 3, b.data, c.data, n,
                g_times_l=True)

        if check:
            check_error(refsol, c.get())

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, 5*n)





def test_plain_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = get_suitable_size(ctx)

    knl = lp.LoopKernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = a[i, k]*b[k, j]"
                ],
            [
                lp.ArrayArg("a", dtype, shape=(n, n), order=order),
                lp.ArrayArg("b", dtype, shape=(n, n), order=order),
                lp.ArrayArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    knl = lp.split_dimension(knl, "i", 16,
            outer_tag="g.0", inner_tag="l.1", no_slabs=True)
    knl = lp.split_dimension(knl, "j", 16,
            outer_tag="g.1", inner_tag="l.0", no_slabs=True)
    knl = lp.split_dimension(knl, "k", 16, no_slabs=True)
    knl = lp.add_prefetch(knl, 'a', ["k_inner", "i_inner"])
    knl = lp.add_prefetch(knl, 'b', ["j_inner", "k_inner", ])
    assert knl.get_problems({})[0] <= 2

    kernel_gen = (lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl))

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





def test_plain_matrix_mul_new_ui(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = get_suitable_size(ctx)

    knl = lp.LoopKernel(ctx.devices[0],
            "[n] -> {[i,j,k]: 0<=i,j,k<n}",
            [
                "{label} c[i, j] = sum_float32(k, cse(a[i, k], lhsmat)*cse(b[k, j], rhsmat))"
                ],
            [
                lp.ArrayArg("a", dtype, shape=(n, n), order=order),
                lp.ArrayArg("b", dtype, shape=(n, n), order=order),
                lp.ArrayArg("c", dtype, shape=(n, n), order=order),
                lp.ScalarArg("n", np.int32, approximately=n),
                ],
            name="matmul", assumptions="n >= 1")

    knl = lp.split_dimension(knl, "i", 16,
            outer_tag="g.0", inner_tag="l.1", no_slabs=True)
    knl = lp.split_dimension(knl, "j", 16,
            outer_tag="g.1", inner_tag="l.0", no_slabs=True)
    knl = lp.split_dimension(knl, "k", 16, no_slabs=True)

    knl = lp.realize_cse(knl, "lhsmat", dtype, ["k_inner", "i_inner"])
    knl = lp.realize_cse(knl, "rhsmat", dtype, ["j_inner", "k_inner"])

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, dict(n=n), kill_level_min=6)

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




def test_troublesome_premagma_fermi_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = 6*16*2

    knl = lp.LoopKernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = a[i, k]*b[k, j]"
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
    knl = lp.split_dimension(knl, "i", i_reg*i_chunks, outer_tag="g.0", no_slabs=True)
    knl = lp.split_dimension(knl, "i_inner", i_reg, outer_tag="l.0", inner_tag="ilp", no_slabs=True)
    knl = lp.split_dimension(knl, "j", j_reg*j_chunks, outer_tag="g.1", no_slabs=True)
    knl = lp.split_dimension(knl, "j_inner", j_reg, outer_tag="l.1", inner_tag="ilp", no_slabs=True)
    knl = lp.split_dimension(knl, "k", 16, no_slabs=True)
    knl = lp.add_prefetch(knl, 'a', ["k_inner", "i_inner_inner"])
    assert knl.get_problems({})[0] <= 2

    kernel_gen = (lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl))

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

    n = 6*16*16

    knl = lp.LoopKernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = a[i, k]*b[k, j]"
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
    knl = lp.split_dimension(knl, "i", i_reg*i_chunks, outer_tag="g.0", no_slabs=True)
    knl = lp.split_dimension(knl, "i_inner", i_reg, outer_tag="l.0", inner_tag="ilp", no_slabs=True)
    knl = lp.split_dimension(knl, "j", j_reg*j_chunks, outer_tag="g.1", no_slabs=True)
    knl = lp.split_dimension(knl, "j_inner", j_reg, outer_tag="l.1", inner_tag="ilp", no_slabs=True)
    knl = lp.split_dimension(knl, "k", 16, no_slabs=True)
    #knl = lp.split_dimension(knl, "k_inner", 8, outer_tag="unr")
    knl = lp.add_prefetch(knl, 'a', ["k_inner", ("i_inner_inner", "i_inner_outer")])
    knl = lp.add_prefetch(knl, 'b', ["k_inner", ("j_inner_inner", "j_inner_outer"),])
    assert knl.get_problems({})[0] <= 2

    kernel_gen = (lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl,
                hints=["k_outer", "k_inner_outer", "k_inner_inner"]
                ))

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

    knl = lp.LoopKernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = a[i, k]*b[k, j]"
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
    knl = lp.split_dimension(knl, "i", i_reg*i_chunks, outer_tag="g.0", no_slabs=True)
    knl = lp.split_dimension(knl, "i_inner", i_reg, outer_tag="l.0", inner_tag="ilp", no_slabs=True)
    knl = lp.split_dimension(knl, "j", j_reg*j_chunks, outer_tag="g.1", no_slabs=True)
    knl = lp.split_dimension(knl, "j_inner", j_reg, outer_tag="l.1", inner_tag="ilp", no_slabs=True)
    knl = lp.split_dimension(knl, "k", 16, no_slabs=True)
    #knl = lp.split_dimension(knl, "k_inner", 8, outer_tag="unr")
    knl = lp.add_prefetch(knl, 'a', ["k_inner", ("i_inner_inner", "i_inner_outer")])
    knl = lp.add_prefetch(knl, 'b', ["k_inner", ("j_inner_inner", "j_inner_outer"),])
    assert knl.get_problems({})[0] <= 2

    kernel_gen = (lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl,
                hints=["k_outer", "k_inner_outer", "k_inner_inner"]
                ))

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

    knl = lp.LoopKernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = a[i, k]*b[k, j]"
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
    assert knl.get_problems({})[0] <= 2

    kernel_gen = (lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl))

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

    n = get_suitable_size(ctx)

    knl = lp.LoopKernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = a[i, k]*b[k, j]"
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
    knl = lp.add_prefetch(knl, 'b', [("j_inner_outer", "j_inner_inner"), "k_inner"])
    assert knl.get_problems({})[0] <= 2

    kernel_gen = (lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl))

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

    knl = lp.LoopKernel(ctx.devices[0],
            "[n] -> {[i,j,k]: 0<=i,j,k<n }",
            [
                "c[i, j] = a[i, k]*b[k, j]"
                ],
            [
                lp.ArrayArg("a", dtype, shape="(n, n)", order=order),
                lp.ArrayArg("b", dtype, shape="(n, n)", order=order),
                lp.ArrayArg("c", dtype, shape="(n, n)", order=order),
                lp.ScalarArg("n", np.int32, approximately=1000),
                ], name="fancy_matmul")

    knl = lp.split_dimension(knl, "i", 16, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_dimension(knl, "j", 16, outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_dimension(knl, "k", 16)
    knl = lp.add_prefetch(knl, 'a', ["i_inner", "k_inner"])
    knl = lp.add_prefetch(knl, 'b', ["k_inner", "j_inner"])
    assert knl.get_problems(dict(n=n))[0] <= 2

    kernel_gen = (lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl))

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




def test_dg_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    Np = 84
    Np_padded = 96
    K = get_suitable_size(ctx)*4
    dim = 3
    num_flds = 2
    use_images = False

    from pymbolic import var
    fld = var("fld")
    matrix_names = ["d%d" % i for i in range(dim)]
    i, j, k = [var(s) for s in "i j k".split()]

    fld_strides = (1, Np_padded)

    knl = lp.LoopKernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j< %d and 0<=k<%d}" % (Np, K),
            [
                (var(mn+"fld%d" % ifld)[i, k], 
                    var(mn)[i, j]*var("fld%d" % ifld)[j, k])
                for mn in matrix_names
                for ifld in range(num_flds)
                ],
            ([lp.ImageArg(mn, dtype, 2) for mn in matrix_names]
            if use_images else
            [lp.ArrayArg(mn, dtype, shape=(Np, Np), order="C") for mn in matrix_names])
            + [lp.ArrayArg("fld%d" % ifld, dtype,
                strides=fld_strides)
                for ifld in range(num_flds)
                ]
            + [lp.ArrayArg(mn+"fld%d" % ifld, dtype,
                strides=fld_strides)
                for ifld in range(num_flds)
                for mn in matrix_names
                ],
            name="dg_matmul")

    #ilp = 4
    knl = lp.split_dimension(knl, "i", 30, 32, outer_tag="g.0", inner_tag="l.0")
    knl = lp.split_dimension(knl, "k", 16, outer_tag="g.1", inner_tag="l.1")
    #knl = lp.split_dimension(knl, "k_inner", 16, outer_tag="ilp", inner_tag="l.1")

    assert Np % 2 == 0
    #knl = lp.split_dimension(knl, "j", Np//2)
    #knl = lp.split_dimension(knl, "k", 32)

    #for mn in matrix_names:
        #knl = lp.add_prefetch(knl, mn, ["j", "i_inner"])
    for ifld in range(num_flds):
        knl = lp.add_prefetch(knl, 'fld%d' % ifld,
                #["k_inner_outer", "k_inner_inner", "j"])
                ["k_inner", "j"])
    assert knl.get_problems({})[0] <= 2

    kernel_gen = list(lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl))[:1]

    matrices = [
            make_well_conditioned_dev_matrix(queue, Np, dtype=dtype, order="C",
                ran_factor=0)
            for mn in matrix_names]
    flds = [
            make_well_conditioned_dev_matrix(queue, (Np_padded, K), dtype=dtype, order="F")
            for ifld in range(num_flds)]
    outputs = [cl_array.empty_like(flds[0])
            for ifld in range(num_flds)
            for mn in matrix_names]

    ref_soln = [np.dot(mat.get(), fld.get()[:Np]) 
            for fld in flds
            for mat in matrices]

    if use_images:
        mat_images = [
                cl.image_from_array(ctx, mat.get(), 1) for mat in matrices]

    def launcher(kernel, gsize, lsize, check):
        if use_images:
            args = mat_images
        else:
            args = [mat.data for mat in matrices]

        args = args + [fld.data for fld in flds] + [out.data for out in outputs]
        kwargs = dict(g_times_l=True)
        evt = kernel(queue, gsize(), lsize(), *args, g_times_l=True)

        if check:
            for out, ref in zip(outputs, ref_soln):
                check_error(ref, out.get()[:Np])

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, num_flds*dim*2*(Np**2)*K)





def main_elwise_scaled_matrix_mul():
    Np = 64
    K = 2000
    from pymbolic import var
    m, u, v, g, i, j, k = [var(s) for s in "muvgijk"]

    knl = make_loop_kernel([
        LoopDimension("i", Np),
        LoopDimension("j", Np),
        LoopDimension("k", K),
        ], [
        (v[i+Np*k], m[i+Np*j]*u[j+Np*k]*g[k])
        ])

    gen_kwargs = {
            "min_threads": 128,
            "min_blocks": 32,
            "prefetch_hints": {"g": False, "m":True},
            }

    if True and HAVE_CUDA:
        if HAVE_CUDA:
            g = curandom.rand((K))
            u = curandom.rand((Np, K))
            m = curandom.rand((Np, Np))
            v = gpuarray.empty_like(u)

        def launcher(grid, kernel, texref_lookup):
            g.bind_to_texref_ext(texref_lookup["g"])
            u.bind_to_texref_ext(texref_lookup["u"])
            m.bind_to_texref_ext(texref_lookup["m"])
            kernel.prepared_call(grid, v.gpudata)

        drive_timing_run(
                generate_all_kernels(knl, **gen_kwargs),
                launcher, 2*Np**2*K)
    else:
        show_kernel_codes(generate_all_kernels(knl, **gen_kwargs))




def main_transpose():
    n = 16*48
    from pymbolic import var
    a, b, i, j = [var(s) for s in "abij"]

    k = make_loop_kernel([
        LoopDimension("i", n),
        LoopDimension("j", n),
        ], [
        (b[i+n*j], a[j+n*i])
        ])

    gen_kwargs = {
            "min_threads": 128,
            "min_blocks": 32,
            }

    if True and HAVE_CUDA:
        if HAVE_CUDA:
            a = curandom.rand((n, n))
            b = gpuarray.empty_like(a)

        def launcher(grid, kernel, texref_lookup):
            a.bind_to_texref_ext(texref_lookup["a"])
            kernel.prepared_call(grid, b.gpudata)

        drive_timing_run(
                generate_all_kernels(k, **gen_kwargs),
                launcher, 0)
    else:
        show_kernel_codes(generate_all_kernels(k, **gen_kwargs))





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

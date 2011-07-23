import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp




FAST_OPTIONS = ["-cl-mad-enable", "-cl-fast-relaxed-math", 
        "-cl-no-signed-zeros", "-cl-strict-aliasing"]

def make_well_conditioned_dev_matrix(queue, shape, dtype=np.float32, order="C", ran_factor=1, od=0):
    if isinstance(shape, int):
        shape = (shape, shape)
    ary = np.asarray(
        ran_factor*np.random.randn(*shape)
        + 5*np.eye(max(shape), k=od)[:shape[0], :shape[1]],
        dtype=dtype, order=order)

    return cl_array.to_device(queue, ary)




DO_CHECK = True

DEBUG_PREAMBLE = r"""
    #pragma OPENCL EXTENSION cl_amd_printf: enable
    #define IFDIAG if (i_outer*16+i_inner == j_outer*16+j_inner)
    #define TST(S) IFDIAG if (i_outer*16+i_inner == 0) \
            printf("ko=%d ki=%d" #S "\n", k_outer, k_inner);
    """




def check_error(refsol, sol):
    rel_err = la.norm(refsol-sol, "fro")/la.norm(refsol, "fro")
    if DO_CHECK and rel_err > 1e-5:
        if 1:
            import matplotlib.pyplot as pt
            pt.imshow(refsol-sol)
            pt.colorbar()
            pt.show()
        elif 0:
            print "---------------------------"
            print "ACTUAL"
            print "---------------------------"
            print sol
            print "---------------------------"
            print "CORRECT"
            print "---------------------------"
            print refsol
        raise RuntimeError("check failed, rel err=%g" % rel_err)




def plain_matrix_mul(ctx_factory=cl.create_some_context):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = 16*100
    from pymbolic import var
    a, b, c, i, j, k, n_sym = [var(s) for s in "abcijkn"]

    knl = lp.LoopKernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                (c[i, j], a[i, k]*b[k, j])
                ],
            [
                lp.ArrayArg("a", dtype, shape=(n, n), order=order),
                lp.ArrayArg("b", dtype, shape=(n, n), order=order),
                lp.ArrayArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    knl = lp.split_dimension(knl, "i", 16, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_dimension(knl, "j", 16, outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_dimension(knl, "k", 4)
    knl = lp.add_prefetch(knl, 'a', ["k_inner", "i_inner"])
    knl = lp.add_prefetch(knl, 'b', ["j_inner", "k_inner", ])
    assert knl.get_invalid_reason() is None

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

    lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3,
            options=FAST_OPTIONS)




def dg_matrix_mul(ctx_factory=cl.create_some_context):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    Np = 84
    Np_padded = 96
    K = 20000
    dim = 3
    num_flds = 6

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
            [lp.ImageArg(mn, dtype, 2) for mn in matrix_names]
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

    knl = lp.split_dimension(knl, "i", 30, 32, outer_tag="g.0", inner_tag="l.0")
    knl = lp.split_dimension(knl, "k", 16, outer_tag="g.1", inner_tag="l.1")
    assert Np % 2 == 0
    #knl = lp.split_dimension(knl, "j", Np//2)
    #knl = lp.split_dimension(knl, "k", 32)

    #for mn in matrix_names:
        #knl = lp.add_prefetch(knl, mn, ["j", "i_inner"])
    for ifld in range(num_flds):
        knl = lp.add_prefetch(knl, 'fld%d' % ifld, ["k_inner", "j"])
    assert knl.get_invalid_reason() is None

    kernel_gen = list(lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl))[:1]

    matrices = [
            make_well_conditioned_dev_matrix(queue, Np, dtype=dtype, order="C")
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

    mat_images = [
            cl.image_from_array(ctx, mat.get(), 1) for mat in matrices]

    def launcher(kernel, gsize, lsize, check):
        args = mat_images + [fld.data for fld in flds] + [out.data for out in outputs]
        kwargs = dict(g_times_l=True)
        evt = kernel(queue, gsize(), lsize(), *args, g_times_l=True)

        if check:
            for out, ref in zip(outputs, ref_soln):
                check_error(ref, out.get()[:Np])

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, num_flds*dim*2*(Np**2)*K,
            options=FAST_OPTIONS + ["-cl-nv-verbose"],
            force_rebuild=True, edit=True
            )





def fancy_matrix_mul(ctx_factory=cl.create_some_context):
    dtype = np.float32
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    order = "C"

    n = 16*40
    from pymbolic import var
    a, b, c, i, j, k, n_sym = [var(s) for s in "abcijkn"]

    knl = lp.LoopKernel(ctx.devices[0],
            "[n] -> {[i,j,k]: 0<=i,j,k<n }",
            [
                (c[i, j], a[i, k]*b[k, j])
                ],
            [
                lp.ArrayArg("a", dtype, shape=(n_sym, n_sym), order=order),
                lp.ArrayArg("b", dtype, shape=(n_sym, n_sym), order=order),
                lp.ArrayArg("c", dtype, shape=(n_sym, n_sym), order=order),
                lp.ScalarArg("n", np.int32, approximately=1000),
                ], name="fancy_matmul")

    knl = lp.split_dimension(knl, "i", 16, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_dimension(knl, "j", 16, outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_dimension(knl, "k", 16)
    knl = lp.add_prefetch(knl, 'a', ["i_inner", "k_inner"])
    knl = lp.add_prefetch(knl, 'b', ["k_inner", "j_inner"])
    knl = lp.add_prefetch(knl, 'a', ["i_inner", "k_inner"])
    knl = lp.add_prefetch(knl, 'b', ["k_inner", "j_inner"])
    assert knl.get_invalid_reason() is None

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

    lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3,
            options=FAST_OPTIONS + ["-cl-nv-verbose"],
            force_rebuild=True)




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
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        image_matrix_mul()

import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp




def make_well_condition_dev_matrix(queue, n, dtype=np.float32, order="C"):
    return cl_array.to_device(queue,
            np.asarray(np.random.randn(n, n) + 5*np.eye(n),
                dtype=dtype, order=order))




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
    knl = lp.split_dimension(knl, "k", 16)
    knl = lp.add_prefetch(knl, 'a', ["k_inner", "i_inner"])
    knl = lp.add_prefetch(knl, 'b', ["j_inner", "k_inner", ])
    assert knl.get_invalid_reason() is None

    kernel_gen = (lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl))

    a = make_well_condition_dev_matrix(queue, n, dtype=dtype, order=order)
    b = make_well_condition_dev_matrix(queue, n, dtype=dtype, order=order)
    c = cl_array.empty_like(a)
    refsol = np.dot(a.get(), b.get())

    def launcher(kernel, gsize, lsize, check):
        evt = kernel(queue, gsize(), lsize(), a.data, b.data, c.data,
                g_times_l=True)

        if check:
            sol = c.get()
            rel_err = la.norm(refsol-sol, "fro")/la.norm(refsol, "fro")
            if rel_err > 1e-5:
                import matplotlib.pyplot as pt
                pt.imshow(refsol-sol)
                pt.colorbar()
                pt.show()
                raise RuntimeError("check failed")

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3)




def fancy_matrix_mul(ctx_factory=cl.create_some_context):
    dtype = np.float32
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    order = "F"

    n = 16*30
    from pymbolic import var
    a, b, c, i, j, k, n_sym = [var(s) for s in "abcijkn"]

    knl = lp.LoopKernel(ctx.devices[0],
            "[n] -> {[i,j,k]: 0<=i,j,k<n }",
            [
                (c[i, j], a[i, k]*b[k, j])
                ],
            [
                lp.ArrayArg("a", dtype, shape=(n_sym, n_sym)),
                lp.ArrayArg("b", dtype, shape=(n_sym, n_sym)),
                lp.ArrayArg("c", dtype, shape=(n_sym, n_sym)),
                lp.ScalarArg("n", np.int32, approximately=1000),
                ], name="fancy_matmul")

    knl = lp.split_dimension(knl, "i", 16, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_dimension(knl, "j", 16, outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_dimension(knl, "k", 16, inner_tag="unr1")
    knl = lp.add_prefetch(knl, 'a', ["i_inner", "k_inner"])
    knl = lp.add_prefetch(knl, 'b', ["k_inner", "j_inner"])
    assert knl.get_invalid_reason() is None

    kernel_gen = (lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl))

    a = make_well_condition_dev_matrix(queue, n, dtype=dtype, order=order)
    b = make_well_condition_dev_matrix(queue, n, dtype=dtype, order=order)
    c = cl_array.empty_like(a)
    refsol = np.dot(a.get(), b.get())

    def launcher(kernel, gsize, lsize, check):
        evt = kernel(queue, gsize(n), lsize(n), a.data, b.data, c.data, n,
                g_times_l=True)

        if check:
            sol = c.get()
            rel_err = la.norm(refsol-sol, "fro")/la.norm(refsol, "fro")
            if rel_err > 1e-5:
                import matplotlib.pyplot as pt
                pt.imshow(refsol-sol)
                pt.colorbar()
                pt.show()
                raise RuntimeError("check failed")

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3)




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
        plain_matrix_mul()

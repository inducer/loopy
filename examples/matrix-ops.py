import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as clrandom
import loopy as lp




def main_matrix_mul():
    ctx = cl.create_some_context(answers=[2])
    queue = cl.CommandQueue(ctx, 
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    #n = 16*34
    n = 16*34
    from pymbolic import var
    a, b, c, i, j, k = [var(s) for s in "abcijk"]

    k = lp.make_loop_kernel(ctx.devices[0], [
        lp.LoopDimension("i", n),
        lp.LoopDimension("j", n),
        lp.LoopDimension("k", n),
        ], [ 
        (c[i+n*j], a[i+n*k]*b[k+n*j]) 
        ],
        hints={
            lp.HINTS.PREFETCH: {"a": True, "b":True},
            lp.HINTS.MIN_GROUP_SIZE: 128,
            lp.HINTS.MIN_GROUP_COUNT: 30,
            })


    kernel_gen = lp.generate_all_kernels(k)

    if 1:
        a = clrandom.rand(queue, (n, n), dtype=np.float32)
        b = clrandom.rand(queue, (n, n), dtype=np.float32)
        c = cl_array.empty_like(a)

        def launcher(gsize, lsize, kernel):
            kernel(queue, gsize, lsize, a.data, b.data, c.data,
                    g_times_l=True)
            refsol = np.dot(a.astype(np.float64).get(), b.astype(np.float64).get())
            sol = c.astype(np.float64).get()

            print la.norm(refsol-sol)/la.norm(refsol)

        lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3)
    else:
        lp.show_kernel_codes(kernel_gen)




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
        main_matrix_mul()

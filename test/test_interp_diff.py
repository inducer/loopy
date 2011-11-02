
def test_interp_diff(ctx_factory):

    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    N = 8
    M = 8

    from pymbolic import var
    K_sym = var("K")

    field_shape = (N, N, N, K_sym)
    interim_field_shape = (M, M, M, K_sym)

    # 1. direction-by-direction similarity transform on u
    # 2. invert diagonal 
    # 3. transform back (direction-by-direction)

    # K - run-time symbolic
    knl = lp.make_kernel(ctx.devices[0],
            "[K] -> {[i,ip,j,jp,k,kp,e]: 0<=i,j,k<%d AND 0<=ip,jp,kp<%d 0<=e<K}" %M %N
            [
                "[|i,jp,kp] <float32>  u1[i ,jp,kp,e] = sum_float32(ip, I[i,ip]*u [ip,jp,kp,e])",
                "[|i,j ,kp] <float32>  u2[i ,j ,kp,e] = sum_float32(jp, I[j,jp]*u1[i ,jp,kp,e])",
                "[|i,j ,k ] <float32>  u3[i ,j ,k ,e] = sum_float32(kp, I[k,kp]*u2[i ,j ,kp,e])",
                "[|i,j ,k ] <float32>  Pu[i ,j ,k ,e] = P[i,j,k,e]*u3[i,j,k,e]",
                "[|i,j ,kp] <float32> Pu3[i ,j ,kp,e] = sum_float32(k, V[kp,k]*Pu[i ,j , k,e])",
                "[|i,jp,kp] <float32> Pu2[i ,jp,kp,e] = sum_float32(j, V[jp,j]*Pu[i ,j ,kp,e])",
                "Pu[ip,jp,kp,e] = sum_float32(i, V[ip,i]*Pu[i ,jp,kp,e])",
                ],
            [
            lp.ArrayArg("u",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("P",   dtype, shape=interim_field_shape, order=order),
            lp.ArrayArg("I",   dtype, shape=(M, N), order=order),
            lp.ArrayArg("V",   dtype, shape=(N, M), order=order),
            lp.ArrayArg("Pu",  dtype, shape=field_shape, order=order),
            lp.ScalarArg("K",  np.int32, approximately=1000),
            ],
            name="sem_lap_precon", assumptions="K>=1")

    print knl
    1/0

    knl = lp.split_dimension(knl, "e", 16, outer_tag="g.0")#, slabs=(0, 1))

    knl = lp.tag_dimensions(knl, dict(i="l.0", j="l.1"))

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

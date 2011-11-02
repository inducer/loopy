
def test_advect(ctx_factory):

    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    N = 8

    from pymbolic import var
    K_sym = var("K")

    field_shape = (N, N, N, K_sym)

    # 1. direction-by-direction similarity transform on u
    # 2. invert diagonal 
    # 3. transform back (direction-by-direction)

    # K - run-time symbolic
    knl = lp.make_kernel(ctx.devices[0],
            "[K] -> {[i,ip,j,jp,k,kp,m,e]: 0<=i,j,k,m<%d AND 0<=e<K}" %N
            [
                # differentiate u
                "[|i,j,k] <float32>  ur[i,j,k,e] = sum_float32(m, D[i,m]*u[m,j,k,e])",
                "[|i,j,k] <float32>  us[i,j,k,e] = sum_float32(m, D[j,m]*u[i,m,k,e])",
                "[|i,j,k] <float32>  ut[i,j,k,e] = sum_float32(m, D[k,m]*u[i,j,m,e])",

                # differentiate v
                "[|i,j,k] <float32>  vr[i,j,k,e] = sum_float32(m, D[i,m]*v[m,j,k,e])",
                "[|i,j,k] <float32>  vs[i,j,k,e] = sum_float32(m, D[j,m]*v[i,m,k,e])",
                "[|i,j,k] <float32>  vt[i,j,k,e] = sum_float32(m, D[k,m]*v[i,j,m,e])",

                # differentiate w
                "[|i,j,k] <float32>  wr[i,j,k,e] = sum_float32(m, D[i,m]*w[m,j,k,e])",
                "[|i,j,k] <float32>  ws[i,j,k,e] = sum_float32(m, D[j,m]*w[i,m,k,e])",
                "[|i,j,k] <float32>  wt[i,j,k,e] = sum_float32(m, D[k,m]*w[i,j,m,e])",

                # find velocity in (r,s,t) coordinates
                "<float32> Vr[i,j,k,e] = "
                "G[i,j,k,0,e]*u[i,j,k,e] + G[i,j,k,1,e]*v[i,j,k,e] + G[i,j,k,2,e]*w[i,j,k,e]",
                "<float32> Vs[i,j,k,e] = "
                "G[i,j,k,3,e]*u[i,j,k,e] + G[i,j,k,4,e]*v[i,j,k,e] + G[i,j,k,5,e]*w[i,j,k,e]",
                "<float32> Vt[i,j,k,e] = "
                "G[i,j,k,6,e]*u[i,j,k,e] + G[i,j,k,7,e]*v[i,j,k,e] + G[i,j,k,8,e]*w[i,j,k,e]",

                # form nonlinear term on integration nodes
                "Nu[i,j,k,e] = Vr[i,j,k,e]*ur[i,j,k,e]+Vs[i,j,k,e]*us[i,j,k,e]+Vt[i,j,k,e]*ut[i,j,k,e]",
                "Nv[i,j,k,e] = Vr[i,j,k,e]*vr[i,j,k,e]+Vs[i,j,k,e]*vs[i,j,k,e]+Vt[i,j,k,e]*vt[i,j,k,e]",
                "Nw[i,j,k,e] = Vr[i,j,k,e]*wr[i,j,k,e]+Vs[i,j,k,e]*ws[i,j,k,e]+Vt[i,j,k,e]*wt[i,j,k,e]",
                ],
            [
            lp.ArrayArg("u",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("v",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("w",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("Nu",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("Nv",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("Nw",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("D",   dtype, shape=(N, N),  order=order),
            lp.ScalarArg("K",  np.int32, approximately=1000),
            ],
            name="sem_advect", assumptions="K>=1")

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

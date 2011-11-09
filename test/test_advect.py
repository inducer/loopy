
def test_advect(ctx_factory):

    dtype = np.float32
    ctx = ctx_factory()

    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    N = 8

    from pymbolic import var
    K_sym = var("K")

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
            "[K] -> {[i,j,k,m,e]: 0<=i,j,k,m<%d AND 0<=e<K}" %N
            [
                # differentiate u
                "CSE:  ur[i,j,k] = sum_float32(@m, D[i,m]*u[e,m,j,k])",
                "CSE:  us[i,j,k] = sum_float32(@m, D[j,m]*u[e,i,m,k])",
                "CSE:  ut[i,j,k] = sum_float32(@m, D[k,m]*u[e,i,j,m])",

                # differentiate v
                "CSE:  vr[i,j,k] = sum_float32(@m, D[i,m]*v[e,m,j,k])",
                "CSE:  vs[i,j,k] = sum_float32(@m, D[j,m]*v[e,i,m,k])",
                "CSE:  vt[i,j,k] = sum_float32(@m, D[k,m]*v[e,i,j,m])",

                # differentiate w
                "CSE:  wr[i,j,k] = sum_float32(@m, D[i,m]*w[e,m,j,k])",
                "CSE:  ws[i,j,k] = sum_float32(@m, D[j,m]*w[e,i,m,k])",
                "CSE:  wt[i,j,k] = sum_float32(@m, D[k,m]*w[e,i,j,m])",

                # find velocity in (r,s,t) coordinates
                # CSE?
                "Vr[i,j,k] = G[0,e,i,j,k]*u[e,i,j,k] + G[1,e,i,j,k]*v[e,i,j,k] + G[2,e,i,j,k]*w[e,i,j,k]",
                "Vs[i,j,k] = G[3,e,i,j,k]*u[e,i,j,k] + G[4,e,i,j,k]*v[e,i,j,k] + G[5,e,i,j,k]*w[e,i,j,k]",
                "Vt[i,j,k] = G[6,e,i,j,k]*u[e,i,j,k] + G[7,e,i,j,k]*v[e,i,j,k] + G[8,e,i,j,k]*w[e,i,j,k]",

                # form nonlinear term on integration nodes
                "Nu[e,i,j,k] = Vr[i,j,k]*ur[i,j,k]+Vs[i,j,k]*us[i,j,k]+Vt[i,j,k]*ut[i,j,k]",
                "Nv[e,i,j,k] = Vr[i,j,k]*vr[i,j,k]+Vs[i,j,k]*vs[i,j,k]+Vt[i,j,k]*vt[i,j,k]",
                "Nw[e,i,j,k] = Vr[i,j,k]*wr[i,j,k]+Vs[i,j,k]*ws[i,j,k]+Vt[i,j,k]*wt[i,j,k]",
                ],
            [
            lp.ArrayArg("u",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("v",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("w",   dtype, shape=field_shape, order=order),
            lp.ArrayArg("Nu",  dtype, shape=field_shape, order=order),
            lp.ArrayArg("Nv",  dtype, shape=field_shape, order=order),
            lp.ArrayArg("Nw",  dtype, shape=field_shape, order=order),
            lp.ArrayArg("G",   dtype, shape=(6,)+field_shape, order=order),
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

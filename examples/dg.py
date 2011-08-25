# FIXME NOT UPDATED YET FOR NEW-STYLE LOOPY!




def main_volume_d_dx():
    Np =  128
    K = 22000
    from pymbolic import var

    D, g, u, p, i, j, k = [var(s) for s in "Dgupijk"]

    dim = 3

    ker = make_loop_kernel([
        LoopDimension("k", K),
        LoopDimension("j", Np),
        LoopDimension("i", Np),
        ], [ 
            (p[j+Np*k], 
                sum(g[dim*(j+Np*k)+i] * D[dim*(j+Np*i)+i] for i in range(3)) 
                * u[i+Np*k])
            ])

    gen_kwargs = {
            "min_threads": 128,
            "min_blocks": 32,
            }

    if 0:
        u = curandom.rand((Np, K))
        p = curandom.rand((Np, K))
        g = curandom.rand((Np*3, K))
        D = curandom.rand((Np*3, Np))

        def launcher(grid, kernel, texref_lookup):
            u.bind_to_texref_ext(texref_lookup["u"])
            g.bind_to_texref_ext(texref_lookup["g"])
            D.bind_to_texref_ext(texref_lookup["D"])
            kernel.prepared_call(grid, p.gpudata)

        drive_timing_run(
                generate_all_kernels(ker, **gen_kwargs),
                launcher, Np*Np*K)
    else:
        show_kernel_codes(generate_all_kernels(ker, **gen_kwargs))




def main_hex_volume_d_dx():
    d = 3
    N = 4
    Np1 = N+1
    Np = Np1**d

    Np_padded = 128
    K = 20000
    from pymbolic import var

    D, Du, u, g, i0, i1, i2, j, k = [var(s) for s in [
        "D", "Du", "u", "g", "i0", "i1", "i2", "j", "k"]]

    axis_indices = [i0,i1,i2]

    ker = make_loop_kernel([
        LoopDimension("k", K),
        LoopDimension("j", Np1),
        ] + [LoopDimension(ai.name, Np1) for ai in axis_indices], 
        [ 
            (Du[k*Np_padded + sum(axis_indices[i_dim]*Np1**i_dim for i_dim in range(d))],
                D[j*Np1+i0] * u[k*Np_padded + sum(
                    (axis_indices[i_dim] if i_dim != d_out else j)*Np1**i_dim 
                    for i_dim in range(d))]
                ) 
            for d_out in [0]
            ]
        )

    gen_kwargs = {
            "min_threads": 64,
            "min_blocks": 32,
            }

    if True and HAVE_CUDA:
        if HAVE_CUDA:
            u = curandom.rand((Np_padded, K))
            #g = curandom.rand((Np*3, K))
            D = curandom.rand((Np1, Np1))
            Du = curandom.rand((Np_padded, K))

        def launcher(grid, kernel, texref_lookup):
            u.bind_to_texref_ext(texref_lookup["u"])
            #g.bind_to_texref_ext(texref_lookup["g"])
            D.bind_to_texref_ext(texref_lookup["D"])
            kernel.prepared_call(grid, Du.gpudata)

        drive_timing_run(
                generate_all_kernels(ker, **gen_kwargs),
                launcher, 2*(Np1**3)*K)
    else:
        show_kernel_codes(generate_all_kernels(ker, **gen_kwargs))



# FIXME: requires precise depedendency checking
if 0:
    import time

    import namedisl as nisl
    import numpy as np
    import numpy.linalg as la

    import pyopencl as cl

    import loopy as lp
    # from loopy.kernel.dependency import (
    #     add_lexicographic_happens_after,
    #     reduce_strict_ordering,
    # )
    from loopy.transform.compute import compute
    from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2


    def centered_second_derivative_coefficients(radius: int, dtype) -> np.ndarray:
        offsets = np.arange(-radius, radius + 1, dtype=dtype)
        powers = np.arange(2 * radius + 1)
        vandermonde = offsets[np.newaxis, :] ** powers[:, np.newaxis]
        rhs = np.zeros(2 * radius + 1, dtype=dtype)
        rhs[2] = 2

        return np.linalg.solve(vandermonde, rhs).astype(dtype)


    def benchmark_executor(ex, queue, args, warmup: int, iterations: int) -> float:
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        evt = None
        for _ in range(warmup):
            evt, _ = ex(queue, **args)
        if evt is not None:
            evt.wait()

        start = time.perf_counter()
        for _ in range(iterations):
            evt, _ = ex(queue, **args)
        if evt is not None:
            evt.wait()
        end = time.perf_counter()

        return (end - start) / iterations


    def fd_flop_count(
            ntime: int,
            nx: int,
            ny: int,
            nz: int,
            stencil_width: int
        ) -> int:
        radius = stencil_width // 2
        output_points = (
            (ntime - 1)
            * (nx - 2 * radius)
            * (ny - 2 * radius)
            * (nz - 2 * radius)
        )
        return 6 * stencil_width * output_points


    def offset_name(ell: int) -> str:
        return f"p{ell}" if ell >= 0 else f"m{-ell}"


    def offset_expr(name: str, ell: int) -> str:
        if ell == 0:
            return name
        return f"{name} {'+' if -ell >= 0 else '-'} {abs(ell)}"


    def make_initial_condition(nx: int, ny: int, nz: int, dtype) -> np.ndarray:
        x = np.linspace(-1, 1, num=nx, endpoint=True, dtype=dtype)
        y = np.linspace(-1, 1, num=ny, endpoint=True, dtype=dtype)
        z = np.linspace(-1, 1, num=nz, endpoint=True, dtype=dtype)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        return np.sin(2 * np.pi * xx) * np.cos(np.pi * yy) * np.cos(np.pi * zz)


    def derive_precise_dependencies(knl):
        from loopy.kernel import KernelState
        from loopy.preprocess import preprocess_program
        from loopy.type_inference import infer_unknown_types

        if knl.state < KernelState.PREPROCESSED:
            knl = preprocess_program(knl)
        knl = infer_unknown_types(knl, expect_completion=True)
        if knl.state < KernelState.LINEARIZED:
            knl = lp.linearize(knl)

        knl = add_lexicographic_happens_after(knl)
        knl = reduce_strict_ordering(knl)
        kernel = knl.default_entrypoint
        return knl.with_kernel(
            kernel.copy(instructions=[
                insn.copy(depends_on_is_final=True)
                for insn in kernel.instructions
            ])
        )


    def finalize_compute_schedule_dependencies(knl, compute_insn_ids, use_ring):
        first_compute_insn_id = compute_insn_ids[0]
        compute_insn_ids = frozenset(compute_insn_ids)
        staging_insn_ids = compute_insn_ids | (frozenset(["z_ring_load"])
                                               if use_ring else frozenset())
        no_sync_with_staging = frozenset(
            (insn_id, "global") for insn_id in staging_insn_ids
        )
        kernel = knl.default_entrypoint
        instructions = []
        for insn in kernel.instructions:
            if insn.id in compute_insn_ids:
                insn = insn.copy(
                    depends_on=frozenset(),
                    depends_on_is_final=True,
                    no_sync_with=insn.no_sync_with | frozenset([("step", "global")]),
                )
            elif use_ring and insn.id == "z_ring_load":
                insn = insn.copy(
                    depends_on=frozenset([first_compute_insn_id]),
                    depends_on_is_final=True,
                    no_sync_with=insn.no_sync_with | frozenset([("step", "global")]),
                )
            elif insn.id == "step":
                insn = insn.copy(
                    depends_on_is_final=True,
                    no_sync_with=insn.no_sync_with | no_sync_with_staging,
                )
            instructions.append(insn)

        return knl.with_kernel(kernel.copy(instructions=instructions))


    def reference_time_stepper(
            u0: np.ndarray,
            coeffs: np.ndarray,
            ntime: int,
            radius: int
        ) -> np.ndarray:
        result = np.empty((2, *u0.shape), dtype=u0.dtype)
        result[0] = u0
        result[1] = u0

        sl = (
            slice(radius, u0.shape[0] - radius),
            slice(radius, u0.shape[1] - radius),
            slice(radius, u0.shape[2] - radius),
        )

        for t in range(ntime - 1):
            src = t % 2
            dst = (t + 1) % 2
            result[dst] = u0
            for i in range(radius, u0.shape[0] - radius):
                for j in range(radius, u0.shape[1] - radius):
                    for k in range(radius, u0.shape[2] - radius):
                        result[dst, i, j, k] = sum(
                            coeffs[ell + radius] * (
                                result[src, i - ell, j, k]
                                + result[src, i, j - ell, k]
                                + result[src, i, j, k - ell]
                            )
                            for ell in range(-radius, radius + 1)
                        )

        return result[(ntime - 1) % 2][sl]


    def main(
            ntime: int = 64,
            nx: int = 512,
            ny: int = 64,
            nz: int = 64,
            stencil_width: int = 5,
            time_block_size: int = 4,
            x_block_size: int = 16,
            y_block_size: int = 2,
            z_block_size: int = 2,
            use_compute: bool = False,
            compute_mode: str = "diamond",
            print_device_code: bool = False,
            print_kernel: bool = False,
            run_kernel: bool = False,
            warmup: int = 3,
            iterations: int = 10,
        ) -> float | None:
        if stencil_width <= 0 or stencil_width % 2 == 0:
            raise ValueError("stencil_width must be a positive odd integer")
        if ntime <= stencil_width:
            raise ValueError("ntime must be larger than stencil_width")
        if min(nx, ny, nz) <= 2 * stencil_width:
            raise ValueError("spatial sizes must be larger than twice stencil_width")
        if compute_mode not in {"diamond", "diamond-ring"}:
            raise ValueError("compute_mode must be 'diamond' or 'diamond-ring'")

        ctx = cl.create_some_context() if run_kernel else None

        dtype = np.float64
        r = stencil_width // 2
        use_ring = use_compute and compute_mode == "diamond-ring"

        u0 = make_initial_condition(nx, ny, nz, dtype)
        u_buf = np.empty((2, nx, ny, nz), dtype=dtype)
        u_buf[0] = u0
        u_buf[1] = u0

        h = dtype(2 / (max(nx, ny, nz) - 1))
        dt = dtype(0.01 * h**2)
        lap_coeffs = centered_second_derivative_coefficients(r, dtype) / h**2
        coeffs = (dt * lap_coeffs).astype(dtype)

        bt = time_block_size
        bx = x_block_size
        by = y_block_size
        bz = z_block_size

        subst_rules = []
        stencil_terms = []
        z_ring_terms = []
        for ell in range(-r, r + 1):
            suffix = f"x_{offset_name(ell)}"
            subst_rules.append(
                f"{suffix}(ts, is, js, ks) := "
                f"u_buf[ts % 2, {offset_expr('is', ell)}, js, ks]"
            )
            x_term = f"{suffix}(t, i, j, k)"
            if use_ring:
                stencil_terms.append(
                    f"c[{ell + r}] * ("
                    f"{x_term}"
                    f" + u_buf[t % 2, i, {offset_expr('j', ell)}, k])"
                )
                z_ring_terms.append(f"c[{ell + r}] * z_ring_3d[{ell + r}]")
            elif ell == 0:
                stencil_terms.append(f"c[{ell + r}] * (3 * {x_term})")
            else:
                stencil_terms.append(
                    f"c[{ell + r}] * ("
                    f"{x_term}"
                    f" + u_buf[t % 2, i, {offset_expr('j', ell)}, k]"
                    f" + u_buf[t % 2, i, j, {offset_expr('k', ell)}])"
                )

        if use_ring:
            stencil_terms.extend(z_ring_terms)

        domain = (
            "{ [t, i, j, k"
            + (", lb" if use_ring else "")
            + "] : "
            "0 <= t < ntime - 1 and "
            "r <= i < nx - r and "
            "r <= j < ny - r and "
            "r <= k < nz - r"
            + (" and -r <= lb < r + 1" if use_ring else "")
            + " }"
        )
        instructions = "\n".join(subst_rules) + "\n\n"
        if use_ring:
            instructions += (
                "z_ring_3d[lb + r] = u_buf[t % 2, i, j, k - lb] "
                "{id=z_ring_load}\n"
            )
        instructions += (
            f"u_buf[(t + 1) % 2, i, j, k] = {' + '.join(stencil_terms)} "
            "{id=step" + (",dep=z_ring_load" if use_ring else "")
            + "}"
        )

        args = [
            lp.GlobalArg(
                "u_buf",
                dtype=dtype,
                shape=(2, nx, ny, nz),
                is_input=True,
                is_output=True,
            ),
            lp.GlobalArg("c", dtype=dtype, shape=(stencil_width,)),
        ]
        if use_ring:
            args.append(
                lp.TemporaryVariable(
                    "z_ring_3d",
                    dtype=dtype,
                    shape=(stencil_width,),
                    address_space=lp.AddressSpace.PRIVATE,
                )
            )

        knl = lp.make_kernel(
            domain,
            instructions,
            args,
            lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2,
        )

        knl = lp.fix_parameters(knl, ntime=ntime, nx=nx, ny=ny, nz=nz, r=r)
        knl = lp.split_iname(knl, "t", bt, inner_iname="ti", outer_iname="to")
        knl = lp.split_iname(knl, "i", bx, inner_iname="xi", outer_iname="xo")
        knl = lp.split_iname(knl, "j", by, inner_iname="yi", outer_iname="yo")
        knl = lp.split_iname(knl, "k", bz, inner_iname="zi", outer_iname="zo")

        if use_compute:
            compute_insn_ids = []
            for ell in range(-r, r + 1):
                suffix = f"x_{offset_name(ell)}"
                compute_insn_id = f"{suffix}_diamond_3d_compute"
                compute_insn_ids.append(compute_insn_id)

                x_storage = f"xs_{suffix}"
                y_storage = f"ys_{suffix}"
                z_storage = f"zs_{suffix}"
                diamond_map = nisl.make_map(f"""{{
                    [ts, is, js, ks] -> [
                        to, xo, ti, {x_storage}, yo, {y_storage}, zo, {z_storage}
                    ] :
                    ts = to * {bt} + ti and
                    is = xo * {bx} + {x_storage} + ti - {bt - 1} and
                    js = yo * {by} + {y_storage} and
                    ks = zo * {bz} + {z_storage}
                }}""")

                knl = compute(
                    knl,
                    suffix,
                    compute_map=diamond_map,
                    storage_indices=[x_storage, y_storage, z_storage],
                    temporary_name=f"{suffix}_diamond_3d",
                    temporary_address_space=lp.AddressSpace.LOCAL,
                    temporary_dtype=dtype,
                    compute_insn_id=compute_insn_id,
                )

                x_tile = f"{x_storage}_tile"
                x_local = f"{x_storage}_local"
                y_tile = f"{y_storage}_tile"
                y_local = f"{y_storage}_local"
                z_tile = f"{z_storage}_tile"
                z_local = f"{z_storage}_local"

                knl = lp.split_iname(
                    knl, x_storage, bx, outer_iname=x_tile, inner_iname=x_local)
                knl = lp.split_iname(
                    knl, y_storage, by, outer_iname=y_tile, inner_iname=y_local)
                knl = lp.split_iname(
                    knl, z_storage, bz, outer_iname=z_tile, inner_iname=z_local)
                knl = lp.tag_inames(knl, {
                    x_local: "l.2",
                    y_local: "l.1",
                    z_local: "l.0",
                })

            knl = finalize_compute_schedule_dependencies(
                knl, compute_insn_ids, use_ring)

        knl = lp.tag_inames(knl, {
            "xo": "g.2",
            "yo": "g.1",
            "zo": "g.0",
            "xi": "l.2",
            "yi": "l.1",
            "zi": "l.0",
        })
        if use_ring:
            knl = lp.tag_inames(knl, {"lb": "unr"})
        knl = lp.prioritize_loops(knl, "to,ti")
        knl = lp.set_options(knl, insert_gbarriers=True)

        if print_kernel:
            print(knl)

        if use_compute:
            knl = derive_precise_dependencies(knl)

        if print_device_code:
            print(lp.generate_code_v2(knl).device_code())

        if not run_kernel:
            return None

        assert ctx is not None
        queue = cl.CommandQueue(ctx)
        ex = knl.executor(queue)

        import pyopencl.array as cl_array

        u_buf_cl = cl_array.to_device(queue, u_buf)
        coeffs_cl = cl_array.to_device(queue, coeffs)

        args = {"c": coeffs_cl, "u_buf": u_buf_cl}
        avg_time_per_iter = benchmark_executor(
            ex, queue, args, warmup=warmup, iterations=iterations
        )
        avg_gflops = (
            fd_flop_count(ntime, nx, ny, nz, stencil_width)
            / avg_time_per_iter
            / 1e9
        )

        u_buf_cl = cl_array.to_device(queue, u_buf)
        args["u_buf"] = u_buf_cl
        _, out = ex(queue, **args)
        reference = reference_time_stepper(u0, coeffs, ntime, r)
        sl = (
            slice(r, nx - r),
            slice(r, ny - r),
            slice(r, nz - r),
        )
        result = out[0].get()[(ntime - 1) % 2][sl]
        rel_err = la.norm(reference - result) / la.norm(reference)

        variant = "baseline" if not use_compute else compute_mode

        print(20 * "=", "3D diamond finite difference report", 20 * "=")
        print(f"Variant      : {variant}")
        print(f"Time steps   : {ntime}")
        print(f"Grid         : {nx} x {ny} x {nz}")
        print(f"Stencil width: {stencil_width}")
        print(f"Tile shape   : bt = {bt}, bx = {bx}, by = {by}, bz = {bz}")
        print(f"Iterations   : warmup = {warmup}, timed = {iterations}")
        print(f"Average time per iteration: {avg_time_per_iter:.6e} s")
        print(f"Average throughput: {avg_gflops:.3f} GFLOP/s")
        print(f"Relative error: {rel_err:.3e}")
        print((40 + len(" 3D diamond finite difference report ")) * "=")

        return avg_time_per_iter


    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser()

        _ = parser.add_argument("--ntime", action="store", type=int, default=64)
        _ = parser.add_argument("--nx", action="store", type=int, default=512)
        _ = parser.add_argument("--ny", action="store", type=int, default=64)
        _ = parser.add_argument("--nz", action="store", type=int, default=64)
        _ = parser.add_argument("--stencil-width", action="store", type=int, default=5)
        _ = parser.add_argument("--time-block-size", action="store", type=int, default=4)
        _ = parser.add_argument("--x-block-size", action="store", type=int, default=16)
        _ = parser.add_argument("--y-block-size", action="store", type=int, default=2)
        _ = parser.add_argument("--z-block-size", action="store", type=int, default=2)

        _ = parser.add_argument("--compare", action="store_true")
        _ = parser.add_argument("--compute", action="store_true")
        _ = parser.add_argument(
            "--compute-mode",
            choices=("diamond", "diamond-ring"),
            default="diamond",
        )
        _ = parser.add_argument("--run-kernel", action="store_true")
        _ = parser.add_argument("--no-run-kernel", action="store_false", dest="run_kernel")
        _ = parser.add_argument("--print-device-code", action="store_true")
        _ = parser.add_argument("--print-kernel", action="store_true")
        _ = parser.add_argument("--warmup", action="store", type=int, default=3)
        _ = parser.add_argument("--iterations", action="store", type=int, default=10)

        args = parser.parse_args()

        if args.compare:
            if not args.run_kernel:
                raise ValueError("--compare requires --run-kernel")

            print("Running example without compute...")
            no_compute_time = main(
                ntime=args.ntime,
                nx=args.nx,
                ny=args.ny,
                nz=args.nz,
                stencil_width=args.stencil_width,
                time_block_size=args.time_block_size,
                x_block_size=args.x_block_size,
                y_block_size=args.y_block_size,
                z_block_size=args.z_block_size,
                use_compute=False,
                compute_mode=args.compute_mode,
                print_device_code=args.print_device_code,
                print_kernel=args.print_kernel,
                run_kernel=args.run_kernel,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            print(50 * "=", "\n")

            print("Running example with compute...")
            compute_time = main(
                ntime=args.ntime,
                nx=args.nx,
                ny=args.ny,
                nz=args.nz,
                stencil_width=args.stencil_width,
                time_block_size=args.time_block_size,
                x_block_size=args.x_block_size,
                y_block_size=args.y_block_size,
                z_block_size=args.z_block_size,
                use_compute=True,
                compute_mode=args.compute_mode,
                print_device_code=args.print_device_code,
                print_kernel=args.print_kernel,
                run_kernel=args.run_kernel,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            print(50 * "=", "\n")

            assert no_compute_time is not None
            assert compute_time is not None
            speedup = no_compute_time / compute_time
            print(f"Speedup: {speedup:.3f}x")
            time_reduction = (1 - compute_time / no_compute_time) * 100
            print(f"Relative time reduction: {time_reduction:.2f}%")
        else:
            _ = main(
                ntime=args.ntime,
                nx=args.nx,
                ny=args.ny,
                nz=args.nz,
                stencil_width=args.stencil_width,
                time_block_size=args.time_block_size,
                x_block_size=args.x_block_size,
                y_block_size=args.y_block_size,
                z_block_size=args.z_block_size,
                use_compute=args.compute,
                compute_mode=args.compute_mode,
                print_device_code=args.print_device_code,
                print_kernel=args.print_kernel,
                run_kernel=args.run_kernel,
                warmup=args.warmup,
                iterations=args.iterations,
            )

import time

import namedisl as nisl
import numpy as np
import numpy.linalg as la

import pyopencl as cl

import loopy as lp
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


def fd_flop_count(ntime: int, nspace: int, stencil_width: int) -> int:
    radius = stencil_width // 2
    output_points = (ntime - 1) * (nspace - 2 * radius)
    return 2 * stencil_width * output_points


def make_initial_condition(nspace: int, dtype) -> np.ndarray:
    x = np.linspace(-1, 1, num=nspace, endpoint=True, dtype=dtype)
    wave_number = dtype(2 * np.pi)
    return np.sin(wave_number * x).astype(dtype)


def reference_time_stepper(
    u0: np.ndarray,
    coeffs: np.ndarray,
    ntime: int,
    radius: int,
) -> np.ndarray:
    result = np.zeros((ntime, u0.size), dtype=u0.dtype)
    result[0] = u0
    result[1:, :radius] = u0[:radius]
    result[1:, u0.size - radius :] = u0[u0.size - radius :]

    for t in range(ntime - 1):
        for i in range(radius, u0.size - radius):
            result[t + 1, i] = sum(
                coeffs[ell + radius] * result[t, i - ell]
                for ell in range(-radius, radius + 1)
            )

    return result


def offset_name(ell: int) -> str:
    return f"u_p{ell}" if ell >= 0 else f"u_m{-ell}"


def main(
    ntime: int = 128,
    nspace: int = 4096,
    stencil_width: int = 9,
    time_block_size: int = 8,
    space_block_size: int = 128,
    use_compute: bool = False,
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
    if nspace <= 2 * stencil_width:
        raise ValueError("nspace must be larger than twice stencil_width")

    dtype = np.float64
    r = stencil_width // 2

    u0 = make_initial_condition(nspace, dtype)
    u_hist = np.zeros((ntime, nspace), dtype=dtype)
    u_hist[0] = u0
    u_hist[1:, :r] = u0[:r]
    u_hist[1:, nspace - r :] = u0[nspace - r :]
    h = dtype(2 / (nspace - 1))
    dt = dtype(0.05 * h**2)
    lap_coeffs = centered_second_derivative_coefficients(r, dtype) / h**2
    coeffs = (dt * lap_coeffs).astype(dtype)
    coeffs[r] += 1

    bt = time_block_size
    bx = space_block_size
    subst_rules = "\n".join(
        f"{offset_name(ell)}(ts, is) := u_hist[ts, is "
        f"{'+' if -ell >= 0 else '-'} {abs(ell)}]"
        for ell in range(-r, r + 1)
    )
    stencil_expr = " + ".join(
        f"c[{ell + r}] * {offset_name(ell)}(t, i)" for ell in range(-r, r + 1)
    )

    knl = lp.make_kernel(
        "{ [t, i] : 0 <= t < ntime - 1 and r <= i < nspace - r }",
        f"""
        {subst_rules}

        u_hist[t + 1, i] = {stencil_expr} {{id=step}}
        """,
        [
            lp.GlobalArg(
                "u_hist",
                dtype=dtype,
                shape=(ntime, nspace),
                is_input=True,
                is_output=True,
            ),
            lp.GlobalArg("c", dtype=dtype, shape=(stencil_width,)),
        ],
        lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2,
    )

    knl = lp.fix_parameters(knl, ntime=ntime, nspace=nspace, r=r)
    knl = lp.split_iname(knl, "t", bt, inner_iname="ti", outer_iname="to")
    knl = lp.split_iname(knl, "i", bx, inner_iname="xi", outer_iname="xo")

    if use_compute:
        raise NotImplementedError(
            "The recurrent diamond time-stepper cannot currently be lowered "
            "through compute(): Loopy represents the instance-wise dependence "
            "between compute loads from u_hist[t] and writes to u_hist[t+1] as "
            "an instruction-level dependency cycle."
        )
        compute_insn_ids = []
        for ell in range(-r, r + 1):
            suffix = offset_name(ell)
            compute_insn_id = f"{suffix}_diamond_compute"
            compute_insn_ids.append(compute_insn_id)
            storage_axis = f"xi_s_{suffix}"
            diamond_map = nisl.make_map(f"""{{
                [ts, is] -> [to, xo, ti, {storage_axis}] :
                ts = to * {bt} + ti and
                is = xo * {bx} + {storage_axis} + ti - {bt - 1}
            }}""")

            knl = compute(
                knl,
                suffix,
                compute_map=diamond_map,
                storage_indices=[storage_axis],
                temporal_inames=["to", "xo", "ti"],
                temporary_name=f"{suffix}_diamond",
                temporary_address_space=lp.AddressSpace.LOCAL,
                temporary_dtype=dtype,
                compute_insn_id=compute_insn_id,
            )
            knl = knl.with_kernel(
                lp.map_instructions(
                    knl.default_entrypoint,
                    f"id:{suffix}_diamond_compute",
                    lambda insn: insn.copy(depends_on=frozenset()),
                )
            )

            knl = lp.split_iname(
                knl,
                storage_axis,
                128,
                outer_iname=f"{storage_axis}_tile",
                inner_iname=f"{storage_axis}_local",
            )
            knl = lp.tag_inames(knl, {f"{storage_axis}_local": "l.0"})

        no_sync_with_computes = frozenset(
            (compute_insn_id, "global") for compute_insn_id in compute_insn_ids
        )
        knl = knl.with_kernel(
            lp.map_instructions(
                knl.default_entrypoint,
                "id:step",
                lambda insn: insn.copy(
                    no_sync_with=insn.no_sync_with | no_sync_with_computes
                ),
            )
        )
        for compute_insn_id in compute_insn_ids:
            knl = knl.with_kernel(
                lp.map_instructions(
                    knl.default_entrypoint,
                    f"id:{compute_insn_id}",
                    lambda insn: insn.copy(
                        no_sync_with=insn.no_sync_with | frozenset([("step", "global")])
                    ),
                )
            )

    knl = lp.tag_inames(knl, {"xi": "l.0"})
    knl = lp.prioritize_loops(knl, "to,ti,xo,xi")
    knl = lp.set_options(knl, insert_gbarriers=True)

    if print_device_code:
        print(lp.generate_code_v2(knl).device_code())

    if print_kernel:
        print(knl)

    if not run_kernel:
        return None

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    ex = knl.executor(queue)

    import pyopencl.array as cl_array

    u_hist_cl = cl_array.to_device(queue, u_hist)
    coeffs_cl = cl_array.to_device(queue, coeffs)

    args = {"c": coeffs_cl, "u_hist": u_hist_cl}
    avg_time_per_iter = benchmark_executor(
        ex, queue, args, warmup=warmup, iterations=iterations
    )
    avg_gflops = fd_flop_count(ntime, nspace, stencil_width) / avg_time_per_iter / 1e9

    _, out = ex(queue, **args)
    reference = reference_time_stepper(u0, coeffs, ntime, r)
    sl = (slice(None), slice(r, nspace - r))
    rel_err = la.norm(reference[sl] - out[0].get()[sl]) / la.norm(reference[sl])

    print(20 * "=", "Diamond finite difference report", 20 * "=")
    print(f"Variant      : {'compute' if use_compute else 'baseline'}")
    print(f"Time steps   : {ntime}")
    print(f"Space points : {nspace}")
    print(f"Stencil width: {stencil_width}")
    print(f"Tile shape   : bt = {bt}, bx = {bx}")
    print(f"Iterations   : warmup = {warmup}, timed = {iterations}")
    print(f"Average time per iteration: {avg_time_per_iter:.6e} s")
    print(f"Average throughput: {avg_gflops:.3f} GFLOP/s")
    print(f"Relative error: {rel_err:.3e}")
    print((40 + len(" Diamond finite difference report ")) * "=")

    return avg_time_per_iter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    _ = parser.add_argument("--ntime", action="store", type=int, default=128)
    _ = parser.add_argument("--nspace", action="store", type=int, default=4096)
    _ = parser.add_argument("--stencil-width", action="store", type=int, default=9)
    _ = parser.add_argument("--time-block-size", action="store", type=int, default=8)
    _ = parser.add_argument("--space-block-size", action="store", type=int, default=128)

    _ = parser.add_argument("--compare", action="store_true")
    _ = parser.add_argument("--compute", action="store_true")
    _ = parser.add_argument("--run-kernel", action="store_true")
    _ = parser.add_argument("--no-run-kernel", action="store_false", dest="run_kernel")
    _ = parser.add_argument("--print-device-code", action="store_true")
    _ = parser.add_argument("--print-kernel", action="store_true")
    _ = parser.add_argument("--warmup", action="store", type=int, default=3)
    _ = parser.add_argument("--iterations", action="store", type=int, default=10)

    args = parser.parse_args()

    if args.compare:
        print("Running example without compute...")
        no_compute_time = main(
            ntime=args.ntime,
            nspace=args.nspace,
            stencil_width=args.stencil_width,
            time_block_size=args.time_block_size,
            space_block_size=args.space_block_size,
            use_compute=False,
            print_device_code=args.print_device_code,
            print_kernel=args.print_kernel,
            run_kernel=True,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(50 * "=", "\n")

        print("Running example with compute...")
        compute_time = main(
            ntime=args.ntime,
            nspace=args.nspace,
            stencil_width=args.stencil_width,
            time_block_size=args.time_block_size,
            space_block_size=args.space_block_size,
            use_compute=True,
            print_device_code=args.print_device_code,
            print_kernel=args.print_kernel,
            run_kernel=True,
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
            nspace=args.nspace,
            stencil_width=args.stencil_width,
            time_block_size=args.time_block_size,
            space_block_size=args.space_block_size,
            use_compute=args.compute,
            print_device_code=args.print_device_code,
            print_kernel=args.print_kernel,
            run_kernel=args.run_kernel,
            warmup=args.warmup,
            iterations=args.iterations,
        )

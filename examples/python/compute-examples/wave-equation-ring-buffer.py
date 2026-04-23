import time

import namedisl as nisl
import numpy as np
import numpy.linalg as la

import pyopencl as cl

import loopy as lp
from loopy.transform.compute import compute
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2


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


def wave_flop_count(ntime: int) -> int:
    return 5 * (ntime - 2)


def main(
        ntime: int = 128,
        use_compute: bool = False,
        print_device_code: bool = False,
        print_kernel: bool = False,
        run_kernel: bool = False,
        warmup: int = 3,
        iterations: int = 10
    ) -> float | None:
    dtype = np.float64

    dt = dtype(1 / 512)
    omega = dtype(2 * np.pi)
    omega2 = dtype(omega**2)

    t = dt * np.arange(ntime, dtype=dtype)
    u = np.cos(omega * t).astype(dtype)

    bt = 32

    knl = lp.make_kernel(
        "{ [t] : 1 <= t < ntime - 1 }",
        """
        u_hist(ts) := u[ts]

        u_next[t + 1] = (
            2 * u_hist(t)
            - u_hist(t - 1)
            - dt2 * omega2 * u_hist(t)
        )
        """,
        [
            lp.GlobalArg("u", dtype=dtype, shape=(ntime,)),
            lp.GlobalArg("u_next", dtype=dtype, shape=(ntime,),
                         is_output=True),
            lp.ValueArg("dt2", dtype=dtype),
            lp.ValueArg("omega2", dtype=dtype),
        ],
        lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2,
    )

    knl = lp.fix_parameters(knl, ntime=ntime)
    knl = lp.split_iname(knl, "t", bt, inner_iname="ti", outer_iname="to")

    if use_compute:
        ring_buffer_map = nisl.make_map(f"""{{
                [ts] -> [to, ti, tb] :
                tb = ts - (to * {bt} + ti) + 1
        }}""")

        knl = compute(
            knl,
            "u_hist",
            compute_map=ring_buffer_map,
            storage_indices=["tb"],
            temporal_inames=["to", "ti"],

            temporary_name="u_time_buf",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_dtype=dtype,

            compute_insn_id="u_time_buf_compute",
            inames_to_advance=["ti"],
        )

        knl = lp.tag_inames(knl, {"tb": "unr"})

    knl = lp.tag_inames(knl, {"to": "g.0"})

    if print_device_code:
        print(lp.generate_code_v2(knl).device_code())

    if print_kernel:
        print(knl)

    if not run_kernel:
        return None

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    ex = knl.executor(queue)

    dt2 = dtype(dt**2)
    avg_time_per_iter = benchmark_executor(
        ex, queue, {"u": u, "dt2": dt2, "omega2": omega2},
        warmup=warmup, iterations=iterations)
    avg_gflops = wave_flop_count(ntime) / avg_time_per_iter / 1e9

    _, out = ex(queue, u=u, dt2=dt2, omega2=omega2)

    ref = np.zeros_like(u)
    for time_idx in range(1, ntime - 1):
        ref[time_idx + 1] = (
            2 * u[time_idx]
            - u[time_idx - 1]
            - dt2 * omega2 * u[time_idx]
        )

    sl = slice(2, ntime)
    rel_err = la.norm(ref[sl] - out[0][sl]) / la.norm(ref[sl])

    print(20 * "=", "Wave recurrence report", 20 * "=")
    print(f"Variant    : {'compute' if use_compute else 'baseline'}")
    print(f"Time steps : {ntime}")
    print(f"Iterations : warmup = {warmup}, timed = {iterations}")
    print(f"Average time per iteration: {avg_time_per_iter:.6e} s")
    print(f"Average throughput: {avg_gflops:.3f} GFLOP/s")
    print(f"Relative error: {rel_err:.3e}")
    print((40 + len(" Wave recurrence report ")) * "=")

    return avg_time_per_iter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    _ = parser.add_argument("--ntime", action="store", type=int, default=128)

    _ = parser.add_argument("--compare", action="store_true")
    _ = parser.add_argument("--compute", action="store_true")
    _ = parser.add_argument("--run-kernel", action="store_true")
    _ = parser.add_argument("--no-run-kernel", action="store_false",
                            dest="run_kernel")
    _ = parser.add_argument("--print-device-code", action="store_true")
    _ = parser.add_argument("--print-kernel", action="store_true")
    _ = parser.add_argument("--warmup", action="store", type=int, default=3)
    _ = parser.add_argument("--iterations", action="store", type=int, default=10)

    args = parser.parse_args()

    if args.compare:
        print("Running example without compute...")
        no_compute_time = main(
            ntime=args.ntime,
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
            use_compute=args.compute,
            print_device_code=args.print_device_code,
            print_kernel=args.print_kernel,
            run_kernel=args.run_kernel,
            warmup=args.warmup,
            iterations=args.iterations,
        )

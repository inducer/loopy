import time

import namedisl as nisl
import numpy as np
import numpy.linalg as la

import pyopencl as cl

import loopy as lp
from benchmark_output import (
    add_json_report_argument,
    benchmark_report,
    variant_result,
    write_json_report,
)
from loopy.transform.compute import compute
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2


def centered_second_derivative_coefficients(radius: int, dtype) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=dtype)
    powers = np.arange(2 * radius + 1)
    vandermonde = offsets[np.newaxis, :] ** powers[:, np.newaxis]
    rhs = np.zeros(2 * radius + 1, dtype=dtype)
    rhs[2] = 2

    return np.linalg.solve(vandermonde, rhs).astype(dtype)


# FIXME: choose a better test case
def f(x, y, z):
    return x**2 + y**2 + z**2


def laplacian_f(x, y, z):
    return 6 * np.ones_like(x)


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


def laplacian_flop_count(npts: int, stencil_width: int) -> int:
    radius = stencil_width // 2
    output_points = (npts - 2 * radius) ** 3
    return 4 * stencil_width * output_points


def laplacian_byte_count(npts: int, stencil_width: int, dtype) -> int:
    radius = stencil_width // 2
    output_points = (npts - 2 * radius) ** 3
    itemsize = np.dtype(dtype).itemsize
    return itemsize * (output_points * (3 * stencil_width + 1) + stencil_width)


def main(
        npts: int = 64,
        stencil_width: int = 5,
        use_compute: bool = False,
        print_device_code: bool = False,
        print_kernel: bool = False,
        run_kernel: bool = False,
        warmup: int = 3,
        iterations: int = 10
    ) -> dict | None:
    if stencil_width <= 0 or stencil_width % 2 == 0:
        raise ValueError("stencil_width must be a positive odd integer")

    pts = np.linspace(-1, 1, num=npts, endpoint=True)
    h = pts[1] - pts[0]

    x, y, z = np.meshgrid(*(pts,)*3)

    dtype = np.float64
    x = x.reshape(*(npts,)*3).astype(dtype)
    y = y.reshape(*(npts,)*3).astype(dtype)
    z = z.reshape(*(npts,)*3).astype(dtype)

    m = stencil_width
    r = m // 2
    c = (centered_second_derivative_coefficients(r, dtype) / h**2).astype(dtype)

    bm = bn = 16
    bk = 32

    knl = lp.make_kernel(
        "{ [i, j, k, l] : r <= i, j, k < npts - r and -r <= l < r + 1 }",
        """
        u_(is, js, ks) := u[is, js, ks]

        lap_u[i,j,k] = sum(
            [l],
            c[l+r] * (u_(i-l,j,k) + u_(i,j-l,k) + u_(i,j,k-l))
        )
        """,
        [
            lp.GlobalArg("u", dtype=dtype, shape=(npts, npts, npts)),
            lp.GlobalArg("lap_u", dtype=dtype, shape=(npts, npts, npts),
                         is_output=True),
            lp.GlobalArg("c", dtype=dtype, shape=(m,))
        ],
        lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2
    )

    knl = lp.fix_parameters(knl, npts=npts, r=r)

    knl = lp.split_iname(knl, "i", bm, inner_iname="ii", outer_iname="io")
    knl = lp.split_iname(knl, "j", bn, inner_iname="ji", outer_iname="jo")
    knl = lp.split_iname(knl, "k", bk, inner_iname="ki", outer_iname="ko")

    if use_compute:
        plane_map = nisl.make_map(f"""{{
                [is, js, ks] -> [io, ii_s, jo, ji_s, ko, ki] :
                is = io * {bm} + ii_s - {r} and
                js = jo * {bn} + ji_s - {r} and
                ks = ko * {bk} + ki
        }}""")

        knl = compute(
            knl,
            "u_",
            compute_map=plane_map,
            storage_indices=["ii_s", "ji_s"],

            temporary_name="u_ij_plane",
            temporary_address_space=lp.AddressSpace.LOCAL,
            temporary_dtype=dtype,

            compute_insn_id="u_plane_compute"
        )

        ring_buffer_map = nisl.make_map(f"""{{
                [is, js, ks] -> [io, ii, jo, ji, ko, ki, kb] :
                is = io * {bm} + ii and
                js = jo * {bn} + ji and
                kb = ks - (ko * {bk} + ki) + {r}
        }}""")

        knl = compute(
            knl,
            "u_",
            compute_map=ring_buffer_map,
            storage_indices=["kb"],

            temporary_name="u_k_buf",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_dtype=dtype,

            compute_insn_id="u_ring_buf_compute",
            inames_to_advance=["ki"]
        )

        nt = 16
        knl = lp.split_iname(
            knl, "ii_s", nt, outer_iname="ii_s_tile", inner_iname="ii_s_local"
        )

        knl = lp.split_iname(
            knl, "ji_s", nt, outer_iname="ji_s_tile", inner_iname="ji_s_local"
        )

        knl = lp.tag_inames(knl, {
            # 2D plane compute storage loops
            "ii_s_local": "l.1",
            "ji_s_local": "l.0",

            # force the use of registers by unrolling
            "kb": "unr"
        })

    knl = lp.tag_inames(knl, {
        # outer block loops
        "io": "g.2",
        "jo": "g.1",
        "ko": "g.0",

        # inner tile loops
        "ii": "l.1",
        "ji": "l.0",
    })

    if print_device_code:
        print(lp.generate_code_v2(knl).device_code())

    if print_kernel:
        print(knl)

    if not run_kernel:
        return None

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    ex = knl.executor(queue)

    f_vals = f(x, y, z)

    import pyopencl.array as cl_array
    f_vals_cl = cl_array.to_device(queue, f_vals)
    c_cl = cl_array.to_device(queue, c)
    lap_u_cl = cl_array.zeros(queue, (npts,)*3, dtype=f_vals_cl.dtype)
    avg_time_per_iter = benchmark_executor(
        ex, queue, {"u": f_vals_cl, "c": c_cl, "lap_u": lap_u_cl},
        warmup=warmup, iterations=iterations)
    modeled_flops = laplacian_flop_count(npts, stencil_width)
    avg_gflops = modeled_flops / avg_time_per_iter / 1e9

    _, lap_fd = ex(queue, u=f_vals_cl, c=c_cl, lap_u=lap_u_cl)
    lap_true = laplacian_f(x, y, z)
    sl = (slice(r, npts - r),)*3

    rel_err = la.norm(lap_true[sl] - lap_fd[0].get()[sl]) / la.norm(lap_true[sl])

    print(20 * "=", "Finite difference report", 20 * "=")
    print(f"Variant      : {'compute' if use_compute else 'baseline'}")
    print(f"Grid points  : {npts}^3")
    print(f"Stencil width: {stencil_width}")
    print(f"Iterations   : warmup = {warmup}, timed = {iterations}")
    print(f"Average time per iteration: {avg_time_per_iter:.6e} s")
    print(f"Average throughput: {avg_gflops:.3f} GFLOP/s")
    print(f"Relative error: {rel_err:.3e}")
    print((40 + len(" Finite difference report ")) * "=")

    return variant_result(
        "compute" if use_compute else "baseline",
        "optimized" if use_compute else "baseline",
        time_s=avg_time_per_iter,
        flop_count=modeled_flops,
        bytes_moved=laplacian_byte_count(npts, stencil_width, dtype),
        dtype=np.dtype(dtype).name,
        relative_error=rel_err,
        metadata={
            "npoints": npts,
            "stencil_width": stencil_width,
            "byte_model": "3 stencil input streams plus one output store",
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    _ = parser.add_argument("--npoints", action="store", type=int, default=64)
    _ = parser.add_argument("--stencil-width", action="store", type=int, default=5)

    _ = parser.add_argument("--compare", action="store_true")
    _ = parser.add_argument("--compute", action="store_true")
    _ = parser.add_argument("--run-kernel", action="store_true")
    _ = parser.add_argument("--no-run-kernel", action="store_false",
                            dest="run_kernel")
    _ = parser.add_argument("--print-device-code", action="store_true")
    _ = parser.add_argument("--print-kernel", action="store_true")
    _ = parser.add_argument("--warmup", action="store", type=int, default=3)
    _ = parser.add_argument("--iterations", action="store", type=int, default=10)
    add_json_report_argument(parser, "finite-difference-2-5d.json")

    args = parser.parse_args()

    variants = []
    if args.compare:
        print("Running example without compute...")
        no_compute_time = main(
            npts=args.npoints,
            stencil_width=args.stencil_width,
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
            npts=args.npoints,
            stencil_width=args.stencil_width,
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
        variants = [no_compute_time, compute_time]
        speedup = no_compute_time["time_s"] / compute_time["time_s"]
        print(f"Speedup: {speedup:.3f}x")
        time_reduction = (
            1 - compute_time["time_s"] / no_compute_time["time_s"]) * 100
        print(f"Relative time reduction: {time_reduction:.2f}%")
    else:
        result = main(
            npts=args.npoints,
            stencil_width=args.stencil_width,
            use_compute=args.compute,
            print_device_code=args.print_device_code,
            print_kernel=args.print_kernel,
            run_kernel=args.run_kernel,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        if result is not None:
            variants = [result]

    write_json_report(
        args.json_report,
        benchmark_report(
            example="finite-difference-2-5D",
            description="3D finite-difference Laplacian with compute staging",
            parameters={
                "npoints": args.npoints,
                "stencil_width": args.stencil_width,
                "warmup": args.warmup,
                "iterations": args.iterations,
            },
            baseline_name="baseline",
            variants=variants,
        ),
    )

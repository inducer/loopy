"""Benchmark a 2D Cartesian Taylor P2M microkernel with Loopy compute.

FMM kernel class: kernel-independent Cartesian Taylor/asymptotic multipole
moment formation.  This script does not evaluate a particular Green's function
such as the 2D Laplace or Helmholtz kernel.  It builds the source monomial
moments that a Taylor FMM would later pair with kernel derivatives or translated
coefficients.  In a Laplace FMM, the Laplace-specific derivative recurrence or
compressed representation lives outside this benchmark.

The kernel forms tensor-product source moments from particle strengths:

    beta[q0, q1] = sum_{isrc} strength[isrc]
        * x[isrc]**q0 / q0!
        * y[isrc]**q1 / q1!

The inline variant is a GPU-parallel reduction over sources for every output
coefficient.  The compute variant splits the source and q1 loops and uses
:func:`loopy.transform.compute.compute` to precompute reusable x and y monomial
basis values in private temporaries.  This tests whether compute can expose
source-tile and coefficient-tile reuse in a reduction-heavy P2M-like kernel.

Use ``--compare`` to run both GPU-parallel variants, compare with the NumPy
reference result, and print timing, modeled GFLOP/s, speedup, and relative
error.
"""

import os
import time

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import namedisl as nisl
import numpy as np
import numpy.linalg as la

import loopy as lp
from loopy.transform.compute import compute
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2


def inv_factorials(order: int, dtype) -> np.ndarray:
    result = np.empty(order + 1, dtype=dtype)
    result[0] = 1
    for i in range(1, order + 1):
        result[i] = result[i - 1] / i
    return result


def reference_p2m(
        strength: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        inv_fact: np.ndarray
    ) -> np.ndarray:
    order = inv_fact.size - 1
    beta = np.empty((order + 1, order + 1), dtype=x.dtype)

    for q0 in range(order + 1):
        for q1 in range(order + 1):
            acc = 0
            for isrc in range(x.size):
                x_monom = x[isrc]**q0 * inv_fact[q0]
                y_monom = y[isrc]**q1 * inv_fact[q1]
                acc += strength[isrc] * x_monom * y_monom
            beta[q0, q1] = acc

    return beta


def make_kernel(
        nsources: int,
        order: int,
        q1_tile_size: int,
        source_tile_size: int,
        dtype,
        use_compute: bool = False
    ) -> lp.TranslationUnit:
    if (order + 1) % q1_tile_size:
        raise ValueError("order + 1 must be divisible by q1_tile_size")
    if nsources % source_tile_size:
        raise ValueError("nsources must be divisible by source_tile_size")

    knl = lp.make_kernel(
        "{ [isrc, q0, q1] : 0 <= isrc < nsources and 0 <= q0, q1 <= p }",
        """
        x_monom_(isrc_arg, q0_arg) := (
            x[isrc_arg] ** q0_arg * inv_fact[q0_arg]
        )

        y_monom_(isrc_arg, q1_arg) := (
            y[isrc_arg] ** q1_arg * inv_fact[q1_arg]
        )

        beta[q0, q1] = sum(
            [isrc],
            strength[isrc] * x_monom_(isrc, q0) * y_monom_(isrc, q1)
        )
        """,
        [
            lp.GlobalArg("x", dtype=dtype, shape=(nsources,)),
            lp.GlobalArg("y", dtype=dtype, shape=(nsources,)),
            lp.GlobalArg("strength", dtype=dtype, shape=(nsources,)),
            lp.GlobalArg("inv_fact", dtype=dtype, shape=(order + 1,)),
            lp.GlobalArg("beta", dtype=dtype, shape=(order + 1, order + 1),
                         is_output=True),
        ],
        lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2,
    )

    knl = lp.fix_parameters(knl, nsources=nsources, p=order)
    knl = lp.split_iname(
        knl,
        "q1",
        q1_tile_size,
        inner_iname="q1_inner",
        outer_iname="q1_outer",
    )
    knl = lp.split_iname(
        knl,
        "isrc",
        source_tile_size,
        inner_iname="isrc_inner",
        outer_iname="isrc_outer",
    )

    if use_compute:
        x_monom_map = nisl.make_map(f"""{{
            [isrc_arg, q0_arg] -> [q0, q1_outer, isrc_outer, isrc_s] :
                isrc_arg = isrc_outer * {source_tile_size} + isrc_s and
                q0_arg = q0
        }}""")

        knl = compute(
            knl,
            "x_monom_",
            compute_map=x_monom_map,
            storage_indices=["isrc_s"],
            temporal_inames=["q0", "q1_outer", "isrc_outer"],
            temporary_name="x_monom_for_q1_tile",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_dtype=dtype,
            compute_insn_id="x_monom_compute",
        )

        y_monom_map = nisl.make_map(f"""{{
            [isrc_arg, q1_arg] -> [q0, q1_outer, q1_inner, isrc_outer, isrc_s] :
                isrc_arg = isrc_outer * {source_tile_size} + isrc_s and
                q1_arg = q1_outer * {q1_tile_size} + q1_inner
        }}""")

        knl = compute(
            knl,
            "y_monom_",
            compute_map=y_monom_map,
            storage_indices=["isrc_s"],
            temporal_inames=["q0", "q1_outer", "q1_inner", "isrc_outer"],
            temporary_name="y_monom_for_coeff",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_dtype=dtype,
            compute_insn_id="y_monom_compute",
        )

    return lp.tag_inames(knl, {
        "q0": "g.1",
        "q1_outer": "g.0",
        "q1_inner": "ilp",
    })


def operation_model(
        nsources: int,
        order: int,
        q1_tile_size: int
    ) -> tuple[int, int]:
    ncoeff = order + 1
    inline_monomial_evals = 2 * nsources * ncoeff**2
    compute_monomial_evals = (
        nsources * ncoeff**2
        + nsources * ncoeff**2 // q1_tile_size
    )
    return inline_monomial_evals, compute_monomial_evals


def p2m_flop_count(
        nsources: int,
        order: int,
        q1_tile_size: int,
        use_compute: bool
    ) -> int:
    ncoeff = order + 1

    contraction_flops = 3 * nsources * ncoeff**2
    if use_compute:
        monomial_scale_flops = (
            nsources * ncoeff**2
            + nsources * ncoeff**2 // q1_tile_size
        )
    else:
        monomial_scale_flops = 2 * nsources * ncoeff**2

    return contraction_flops + monomial_scale_flops


def benchmark_executor(ex, queue, args, warmup: int, iterations: int) -> float:
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


def run_kernel(
        knl: lp.TranslationUnit,
        x: np.ndarray,
        y: np.ndarray,
        strength: np.ndarray,
        inv_fact: np.ndarray,
        warmup: int,
        iterations: int
    ) -> tuple[np.ndarray, float]:
    import pyopencl as cl
    import pyopencl.array as cl_array

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    ex = knl.executor(queue)

    x_cl = cl_array.to_device(queue, x)
    y_cl = cl_array.to_device(queue, y)
    strength_cl = cl_array.to_device(queue, strength)
    inv_fact_cl = cl_array.to_device(queue, inv_fact)
    beta_cl = cl_array.zeros(queue, (inv_fact.size, inv_fact.size),
                             dtype=x.dtype)

    elapsed = benchmark_executor(
        ex,
        queue,
        {
            "x": x_cl,
            "y": y_cl,
            "strength": strength_cl,
            "inv_fact": inv_fact_cl,
            "beta": beta_cl,
        },
        warmup=warmup,
        iterations=iterations,
    )

    _, out = ex(
        queue, x=x_cl, y=y_cl, strength=strength_cl,
        inv_fact=inv_fact_cl, beta=beta_cl)
    return out[0].get(), elapsed


def main(
        nsources: int = 256,
        order: int = 12,
        q1_tile_size: int = 13,
        source_tile_size: int = 128,
        use_compute: bool = False,
        compare: bool = False,
        print_kernel: bool = False,
        print_device_code: bool = False,
        run: bool = False,
        warmup: int = 3,
        iterations: int = 10
    ) -> None:
    dtype = np.float64
    rng = np.random.default_rng(18)
    x = rng.uniform(-0.25, 0.25, size=nsources).astype(dtype)
    y = rng.uniform(-0.25, 0.25, size=nsources).astype(dtype)
    strength = rng.normal(size=nsources).astype(dtype)
    inv_fact = inv_factorials(order, dtype)
    reference = reference_p2m(strength, x, y, inv_fact)

    inline_evals, compute_evals = operation_model(
        nsources, order, q1_tile_size)

    variants = [False, True] if compare else [use_compute]
    timings: dict[bool, float] = {}
    for variant_uses_compute in variants:
        knl = make_kernel(
            nsources, order, q1_tile_size, source_tile_size, dtype,
            use_compute=variant_uses_compute)
        modeled_flops = p2m_flop_count(
            nsources, order, q1_tile_size,
            use_compute=variant_uses_compute)

        print(20 * "=", "P2M basis report", 20 * "=")
        print(f"Variant: {'compute' if variant_uses_compute else 'inline'}")
        print(f"Sources: {nsources}")
        print(f"Order  : {order}")
        print(f"q1 tile: {q1_tile_size}")
        print(f"Source tile: {source_tile_size}")
        print(f"Inline monomial evaluations: {inline_evals}")
        print(f"Compute monomial evaluations: {compute_evals}")
        print(f"Modeled flop count: {modeled_flops}")

        if print_kernel:
            print(knl)

        if print_device_code:
            print(lp.generate_code_v2(knl).device_code())

        if run or compare:
            try:
                result, elapsed = run_kernel(
                    knl, x, y, strength, inv_fact,
                    warmup=warmup, iterations=iterations)
            except Exception as exc:
                print(f"Runtime execution unavailable: {exc}")
            else:
                rel_err = la.norm(result - reference) / la.norm(reference)
                gflops = modeled_flops / elapsed * 1e-9
                timings[variant_uses_compute] = elapsed
                print(f"Average time per iteration: {elapsed:.6e} s")
                print(f"Modeled throughput: {gflops:.3f} GFLOP/s")
                print(f"Relative error: {rel_err:.3e}")

        print((40 + len(" P2M basis report ")) * "=")

    if compare and False in timings and True in timings:
        speedup = timings[False] / timings[True]
        time_reduction = (1 - timings[True] / timings[False]) * 100
        print(f"Speedup: {speedup:.3f}x")
        print(f"Relative time reduction: {time_reduction:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--nsources", action="store", type=int, default=256)
    _ = parser.add_argument("--order", action="store", type=int, default=12)
    _ = parser.add_argument("--q1-tile-size", action="store", type=int, default=13)
    _ = parser.add_argument("--source-tile-size", action="store",
                            type=int, default=128)
    _ = parser.add_argument("--compute", action="store_true")
    _ = parser.add_argument("--compare", action="store_true")
    _ = parser.add_argument("--run-kernel", action="store_true")
    _ = parser.add_argument("--print-kernel", action="store_true")
    _ = parser.add_argument("--print-device-code", action="store_true")
    _ = parser.add_argument("--warmup", action="store", type=int, default=3)
    _ = parser.add_argument("--iterations", action="store", type=int, default=10)

    args = parser.parse_args()

    main(
        nsources=args.nsources,
        order=args.order,
        q1_tile_size=args.q1_tile_size,
        source_tile_size=args.source_tile_size,
        use_compute=args.compute,
        compare=args.compare,
        print_kernel=args.print_kernel,
        print_device_code=args.print_device_code,
        run=args.run_kernel,
        warmup=args.warmup,
        iterations=args.iterations,
    )

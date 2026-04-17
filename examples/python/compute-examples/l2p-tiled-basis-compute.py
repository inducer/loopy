"""Benchmark a 2D Cartesian Taylor L2P microkernel with Loopy compute.

FMM kernel class: kernel-independent Cartesian Taylor/asymptotic local
expansion evaluation.  This script does not evaluate a particular Green's
function such as the 2D Laplace or Helmholtz kernel.  The local coefficients
``gamma`` are treated as already available; if they came from a Laplace FMM,
this kernel is the L2P monomial-contraction stage after the Laplace-specific
coefficient/derivative work has already happened.

The kernel evaluates a tensor-product Taylor-like local expansion at many
target points:

    phi[itgt] = sum_{q0,q1} gamma[q0, q1]
        * x[itgt]**q0 / q0!
        * y[itgt]**q1 / q1!

The inline variant is a parallel GPU kernel over target blocks that expands the
two basis substitutions at every use inside the coefficient sum.  The compute
variant uses :func:`loopy.transform.compute.compute` to materialize the x and y
basis values into private temporaries for each target, so the powers/factorial
scalings are reused across the inner coefficient loops instead of recomputed.

Use ``--compare`` to run both GPU-parallel variants, check against the NumPy
reference implementation, and report timing, modeled GFLOP/s, speedup, and
relative error.
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


def reference_l2p(
        gamma: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        inv_fact: np.ndarray
    ) -> np.ndarray:
    order = gamma.shape[0] - 1
    result = np.empty_like(x)

    for itgt in range(x.size):
        acc = 0
        for q0 in range(order + 1):
            x_basis = x[itgt]**q0 * inv_fact[q0]
            for q1 in range(order + 1):
                y_basis = y[itgt]**q1 * inv_fact[q1]
                acc += gamma[q0, q1] * x_basis * y_basis
        result[itgt] = acc

    return result


def make_kernel(
        ntargets: int,
        order: int,
        target_block_size: int,
        dtype,
        use_compute: bool = False
    ) -> lp.TranslationUnit:
    if ntargets % target_block_size:
        raise ValueError("ntargets must be divisible by target_block_size")

    knl = lp.make_kernel(
        "{ [itgt, q0, q1] : 0 <= itgt < ntargets and 0 <= q0, q1 <= p }",
        """
        x_basis_(itgt_arg, q0_arg) := (
            x[itgt_arg] ** q0_arg * inv_fact[q0_arg]
        )

        y_basis_(itgt_arg, q1_arg) := (
            y[itgt_arg] ** q1_arg * inv_fact[q1_arg]
        )

        phi[itgt] = sum(
            [q0, q1],
            gamma[q0, q1] * x_basis_(itgt, q0) * y_basis_(itgt, q1)
        )
        """,
        [
            lp.GlobalArg("x", dtype=dtype, shape=(ntargets,)),
            lp.GlobalArg("y", dtype=dtype, shape=(ntargets,)),
            lp.GlobalArg("inv_fact", dtype=dtype, shape=(order + 1,)),
            lp.GlobalArg("gamma", dtype=dtype, shape=(order + 1, order + 1)),
            lp.GlobalArg("phi", dtype=dtype, shape=(ntargets,), is_output=True),
        ],
        lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2,
    )

    knl = lp.fix_parameters(knl, ntargets=ntargets, p=order)
    knl = lp.split_iname(
        knl,
        "itgt",
        target_block_size,
        inner_iname="itgt_inner",
        outer_iname="itgt_block",
    )

    if use_compute:
        x_basis_map = nisl.make_map(f"""{{
            [itgt_arg, q0_arg] -> [itgt_block, itgt_inner, q0_s] :
                itgt_arg = itgt_block * {target_block_size} + itgt_inner and
                q0_arg = q0_s
        }}""")

        knl = compute(
            knl,
            "x_basis_",
            compute_map=x_basis_map,
            storage_indices=["q0_s"],
            temporal_inames=["itgt_block", "itgt_inner"],
            temporary_name="x_basis_reg",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_dtype=dtype,
            compute_insn_id="x_basis_compute",
        )

        y_basis_map = nisl.make_map(f"""{{
            [itgt_arg, q1_arg] -> [itgt_block, itgt_inner, q1_s] :
                itgt_arg = itgt_block * {target_block_size} + itgt_inner and
                q1_arg = q1_s
        }}""")

        knl = compute(
            knl,
            "y_basis_",
            compute_map=y_basis_map,
            storage_indices=["q1_s"],
            temporal_inames=["itgt_block", "itgt_inner"],
            temporary_name="y_basis_reg",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_dtype=dtype,
            compute_insn_id="y_basis_compute",
        )

    iname_tags = {
        "itgt_block": "g.0",
        "itgt_inner": "l.0",
        "q0": "unr",
        "q1": "unr",
    }
    if use_compute:
        iname_tags.update({
            "q0_s": "unr",
            "q1_s": "unr",
        })

    knl = lp.tag_inames(knl, iname_tags)
    return knl


def operation_model(
        ntargets: int,
        order: int,
        target_block_size: int
    ) -> tuple[int, int]:
    ncoeff = order + 1
    inline_basis_evals = 2 * ntargets * ncoeff**2
    tiled_compute_basis_evals = 2 * ntargets * ncoeff
    return inline_basis_evals, tiled_compute_basis_evals


def l2p_flop_count(ntargets: int, order: int, use_compute: bool) -> int:
    ncoeff = order + 1

    contraction_flops = 3 * ntargets * ncoeff**2
    if use_compute:
        basis_scale_flops = 2 * ntargets * ncoeff
    else:
        basis_scale_flops = 2 * ntargets * ncoeff**2

    return contraction_flops + basis_scale_flops


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
        inv_fact: np.ndarray,
        gamma: np.ndarray,
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
    inv_fact_cl = cl_array.to_device(queue, inv_fact)
    gamma_cl = cl_array.to_device(queue, gamma)
    phi_cl = cl_array.zeros(queue, x.shape, dtype=x.dtype)

    elapsed = benchmark_executor(
        ex,
        queue,
        {
            "x": x_cl,
            "y": y_cl,
            "inv_fact": inv_fact_cl,
            "gamma": gamma_cl,
            "phi": phi_cl,
        },
        warmup=warmup,
        iterations=iterations,
    )

    _, out = ex(
        queue, x=x_cl, y=y_cl, inv_fact=inv_fact_cl,
        gamma=gamma_cl, phi=phi_cl)
    return out[0].get(), elapsed


def main(
        ntargets: int = 256,
        order: int = 12,
        target_block_size: int = 32,
        use_compute: bool = False,
        compare: bool = False,
        print_kernel: bool = False,
        print_device_code: bool = False,
        run: bool = False,
        warmup: int = 3,
        iterations: int = 10
    ) -> None:
    dtype = np.float64
    rng = np.random.default_rng(14)
    x = rng.uniform(-0.25, 0.25, size=ntargets).astype(dtype)
    y = rng.uniform(-0.25, 0.25, size=ntargets).astype(dtype)
    inv_fact = inv_factorials(order, dtype)
    gamma = rng.normal(size=(order + 1, order + 1)).astype(dtype)
    reference = reference_l2p(gamma, x, y, inv_fact)

    inline_evals, compute_evals = operation_model(
        ntargets, order, target_block_size)

    variants = [False, True] if compare else [use_compute]
    timings: dict[bool, float] = {}
    for variant_uses_compute in variants:
        knl = make_kernel(
            ntargets, order, target_block_size, dtype,
            use_compute=variant_uses_compute)
        modeled_flops = l2p_flop_count(
            ntargets, order, use_compute=variant_uses_compute)

        print(20 * "=", "L2P basis report", 20 * "=")
        print(f"Variant     : {'tiled compute' if variant_uses_compute else 'inline'}")
        print(f"Targets     : {ntargets}")
        print(f"Order       : {order}")
        print(f"Target block: {target_block_size}")
        print(f"Inline basis evaluations: {inline_evals}")
        print(f"Tiled compute evaluations: {compute_evals}")
        print(f"Modeled flop count: {modeled_flops}")

        if print_kernel:
            print(knl)

        if print_device_code:
            print(lp.generate_code_v2(knl).device_code())

        if run or compare:
            try:
                result, elapsed = run_kernel(
                    knl, x, y, inv_fact, gamma,
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

        print((40 + len(" L2P basis report ")) * "=")

    if compare and False in timings and True in timings:
        speedup = timings[False] / timings[True]
        time_reduction = (1 - timings[True] / timings[False]) * 100
        print(f"Speedup: {speedup:.3f}x")
        print(f"Relative time reduction: {time_reduction:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--ntargets", action="store", type=int, default=256)
    _ = parser.add_argument("--order", action="store", type=int, default=12)
    _ = parser.add_argument("--target-block-size", action="store",
                            type=int, default=32)
    _ = parser.add_argument("--compute", action="store_true")
    _ = parser.add_argument("--compare", action="store_true")
    _ = parser.add_argument("--run-kernel", action="store_true")
    _ = parser.add_argument("--print-kernel", action="store_true")
    _ = parser.add_argument("--print-device-code", action="store_true")
    _ = parser.add_argument("--warmup", action="store", type=int, default=3)
    _ = parser.add_argument("--iterations", action="store", type=int, default=10)

    args = parser.parse_args()

    main(
        ntargets=args.ntargets,
        order=args.order,
        target_block_size=args.target_block_size,
        use_compute=args.compute,
        compare=args.compare,
        print_kernel=args.print_kernel,
        print_device_code=args.print_device_code,
        run=args.run_kernel,
        warmup=args.warmup,
        iterations=args.iterations,
    )

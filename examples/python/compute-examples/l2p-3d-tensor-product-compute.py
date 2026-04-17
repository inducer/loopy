"""Benchmark a 3D Cartesian Taylor L2P microkernel with Loopy compute.

FMM kernel class: kernel-independent Cartesian Taylor/asymptotic local
expansion evaluation in three spatial dimensions.  This script does not
directly evaluate the 3D Laplace, Helmholtz, or biharmonic Green's function.
The dense local coefficient tensor ``gamma`` is assumed to have already been
formed by the relevant FMM translation machinery; this benchmark isolates the
target-side monomial contraction.

The kernel evaluates a dense 3D tensor-product local expansion at many target
points:

    phi[itgt] = sum_{q0,q1,q2} gamma[q0, q1, q2]
        * x[itgt]**q0 / q0!
        * y[itgt]**q1 / q1!
        * z[itgt]**q2 / q2!

The baseline variants are GPU-parallel kernels over target blocks that expand
the basis substitutions inline.  The compute variants use
:func:`loopy.transform.compute.compute` to materialize the x, y, and z basis
values into private temporaries.  The script includes both a direct tiled
compute schedule and an optimized register-tiled schedule, following the style
of Loopy's compute matmul example.

Use ``--compare`` to run the naive parallel baseline and the optimized compute
kernel, validate both against the NumPy reference, and report timing, modeled
GFLOP/s, speedup, and relative error.
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


def reference_l2p_3d(
        gamma: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
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
                for q2 in range(order + 1):
                    z_basis = z[itgt]**q2 * inv_fact[q2]
                    acc += gamma[q0, q1, q2] * x_basis * y_basis * z_basis
        result[itgt] = acc

    return result


def make_kernel(
        ntargets: int,
        order: int,
        dtype
    ) -> lp.TranslationUnit:
    knl = lp.make_kernel(
        "{ [itgt, q0, q1, q2] : "
        "0 <= itgt < ntargets and 0 <= q0, q1, q2 <= p }",
        """
        x_basis_(itgt_arg, q0_arg) := (
            x[itgt_arg] ** q0_arg * inv_fact[q0_arg]
        )

        y_basis_(itgt_arg, q1_arg) := (
            y[itgt_arg] ** q1_arg * inv_fact[q1_arg]
        )

        z_basis_(itgt_arg, q2_arg) := (
            z[itgt_arg] ** q2_arg * inv_fact[q2_arg]
        )

        phi[itgt] = sum(
            [q0, q1, q2],
            gamma[q0, q1, q2]
            * x_basis_(itgt, q0)
            * y_basis_(itgt, q1)
            * z_basis_(itgt, q2)
        )
        """,
        [
            lp.GlobalArg("x", dtype=dtype, shape=(ntargets,)),
            lp.GlobalArg("y", dtype=dtype, shape=(ntargets,)),
            lp.GlobalArg("z", dtype=dtype, shape=(ntargets,)),
            lp.GlobalArg("inv_fact", dtype=dtype, shape=(order + 1,)),
            lp.GlobalArg(
                "gamma",
                dtype=dtype,
                shape=(order + 1, order + 1, order + 1),
            ),
            lp.GlobalArg("phi", dtype=dtype, shape=(ntargets,), is_output=True),
        ],
        lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2,
    )
    return lp.fix_parameters(knl, ntargets=ntargets, p=order)


def split_targets(
        knl: lp.TranslationUnit,
        target_block_size: int
    ) -> lp.TranslationUnit:
    knl = lp.split_iname(
        knl,
        "itgt",
        target_block_size,
        inner_iname="itgt_inner",
        outer_iname="itgt_block",
    )
    return lp.tag_inames(knl, {"itgt_block": "g.0"})


def block_private_l2p_3d(
        knl: lp.TranslationUnit,
        target_block_size: int,
        dtype
    ) -> lp.TranslationUnit:
    knl = split_targets(knl, target_block_size)

    x_basis_map = nisl.make_map(f"""{{
        [itgt_arg, q0_arg] -> [itgt_block, itgt_s, q0_s] :
            itgt_arg = itgt_block * {target_block_size} + itgt_s and
            q0_arg = q0_s
    }}""")

    knl = compute(
        knl,
        "x_basis_",
        compute_map=x_basis_map,
        storage_indices=["itgt_s", "q0_s"],
        temporal_inames=["itgt_block"],
        temporary_name="x_basis_tile",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        temporary_dtype=dtype,
        compute_insn_id="x_basis_compute",
    )

    y_basis_map = nisl.make_map(f"""{{
        [itgt_arg, q1_arg] -> [itgt_block, itgt_s, q1_s] :
            itgt_arg = itgt_block * {target_block_size} + itgt_s and
            q1_arg = q1_s
    }}""")

    knl = compute(
        knl,
        "y_basis_",
        compute_map=y_basis_map,
        storage_indices=["itgt_s", "q1_s"],
        temporal_inames=["itgt_block"],
        temporary_name="y_basis_tile",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        temporary_dtype=dtype,
        compute_insn_id="y_basis_compute",
    )

    z_basis_map = nisl.make_map(f"""{{
        [itgt_arg, q2_arg] -> [itgt_block, itgt_s, q2_s] :
            itgt_arg = itgt_block * {target_block_size} + itgt_s and
            q2_arg = q2_s
    }}""")

    return compute(
        knl,
        "z_basis_",
        compute_map=z_basis_map,
        storage_indices=["itgt_s", "q2_s"],
        temporal_inames=["itgt_block"],
        temporary_name="z_basis_tile",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        temporary_dtype=dtype,
        compute_insn_id="z_basis_compute",
    )


def register_tiled_l2p_3d(
        knl: lp.TranslationUnit,
        target_block_size: int,
        dtype
    ) -> lp.TranslationUnit:
    knl = split_targets(knl, target_block_size)

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

    z_basis_map = nisl.make_map(f"""{{
        [itgt_arg, q2_arg] -> [itgt_block, itgt_inner, q2_s] :
            itgt_arg = itgt_block * {target_block_size} + itgt_inner and
            q2_arg = q2_s
    }}""")

    knl = compute(
        knl,
        "z_basis_",
        compute_map=z_basis_map,
        storage_indices=["q2_s"],
        temporal_inames=["itgt_block", "itgt_inner"],
        temporary_name="z_basis_reg",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        temporary_dtype=dtype,
        compute_insn_id="z_basis_compute",
    )

    return lp.tag_inames(knl, {
        "itgt_inner": "l.0",
        "q0_s": "unr",
        "q1_s": "unr",
        "q2_s": "unr",
        "q0": "unr",
        "q1": "unr",
        "q2": "unr",
    })


def operation_model(
        ntargets: int,
        order: int,
        target_block_size: int
    ) -> tuple[int, int]:
    ncoeff = order + 1
    inline_basis_evals = 3 * ntargets * ncoeff**3
    tiled_compute_basis_evals = 3 * ntargets * ncoeff
    return inline_basis_evals, tiled_compute_basis_evals


def l2p_3d_flop_count(ntargets: int, order: int, use_compute: bool) -> int:
    ncoeff = order + 1

    contraction_flops = 4 * ntargets * ncoeff**3
    if use_compute:
        basis_scale_flops = 3 * ntargets * ncoeff
    else:
        basis_scale_flops = 3 * ntargets * ncoeff**3

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
        z: np.ndarray,
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
    z_cl = cl_array.to_device(queue, z)
    inv_fact_cl = cl_array.to_device(queue, inv_fact)
    gamma_cl = cl_array.to_device(queue, gamma)
    phi_cl = cl_array.zeros(queue, x.shape, dtype=x.dtype)

    elapsed = benchmark_executor(
        ex,
        queue,
        {
            "x": x_cl,
            "y": y_cl,
            "z": z_cl,
            "inv_fact": inv_fact_cl,
            "gamma": gamma_cl,
            "phi": phi_cl,
        },
        warmup=warmup,
        iterations=iterations,
    )

    _, out = ex(
        queue, x=x_cl, y=y_cl, z=z_cl, inv_fact=inv_fact_cl,
        gamma=gamma_cl, phi=phi_cl)
    return out[0].get(), elapsed


def main(
        ntargets: int = 256,
        order: int = 8,
        target_block_size: int = 32,
        use_compute: bool = False,
        use_block_private_compute: bool = False,
        compare: bool = False,
        print_kernel: bool = False,
        print_device_code: bool = False,
        run: bool = False,
        warmup: int = 3,
        iterations: int = 10
    ) -> None:
    if ntargets % target_block_size:
        raise ValueError("ntargets must be divisible by target_block_size")

    dtype = np.float64
    rng = np.random.default_rng(22)
    x = rng.uniform(-0.25, 0.25, size=ntargets).astype(dtype)
    y = rng.uniform(-0.25, 0.25, size=ntargets).astype(dtype)
    z = rng.uniform(-0.25, 0.25, size=ntargets).astype(dtype)
    inv_fact = inv_factorials(order, dtype)
    gamma = rng.normal(size=(order + 1, order + 1, order + 1)).astype(dtype)
    reference = reference_l2p_3d(gamma, x, y, z, inv_fact)

    inline_evals, compute_evals = operation_model(
        ntargets, order, target_block_size)

    if compare:
        variants = ["inline", "register-tiled compute"]
    elif use_block_private_compute:
        variants = ["block-private compute"]
    elif use_compute:
        variants = ["register-tiled compute"]
    else:
        variants = ["inline"]

    timings: dict[str, float] = {}
    for variant in variants:
        knl = make_kernel(ntargets, order, dtype)

        if variant == "inline":
            knl = split_targets(knl, target_block_size)
            knl = lp.tag_inames(knl, {
                "itgt_inner": "l.0",
                "q0": "unr",
                "q1": "unr",
                "q2": "unr",
            })
        elif variant == "block-private compute":
            knl = block_private_l2p_3d(knl, target_block_size, dtype)
        elif variant == "register-tiled compute":
            knl = register_tiled_l2p_3d(knl, target_block_size, dtype)
        else:
            raise ValueError(f"unknown variant '{variant}'")

        variant_uses_compute = variant != "inline"
        modeled_flops = l2p_3d_flop_count(
            ntargets, order, use_compute=variant_uses_compute)

        print(20 * "=", "3D L2P basis report", 20 * "=")
        print(f"Variant     : {variant}")
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
                    knl, x, y, z, inv_fact, gamma,
                    warmup=warmup, iterations=iterations)
            except Exception as exc:
                print(f"Runtime execution unavailable: {exc}")
            else:
                rel_err = la.norm(result - reference) / la.norm(reference)
                gflops = modeled_flops / elapsed * 1e-9
                timings[variant] = elapsed
                print(f"Average time per iteration: {elapsed:.6e} s")
                print(f"Modeled throughput: {gflops:.3f} GFLOP/s")
                print(f"Relative error: {rel_err:.3e}")

        print((40 + len(" 3D L2P basis report ")) * "=")

    if (
            compare
            and "inline" in timings
            and "register-tiled compute" in timings):
        speedup = timings["inline"] / timings["register-tiled compute"]
        time_reduction = (
            1 - timings["register-tiled compute"] / timings["inline"]) * 100
        print(f"Speedup: {speedup:.3f}x")
        print(f"Relative time reduction: {time_reduction:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--ntargets", action="store", type=int, default=256)
    _ = parser.add_argument("--order", action="store", type=int, default=8)
    _ = parser.add_argument("--target-block-size", action="store",
                            type=int, default=32)
    _ = parser.add_argument("--compute", action="store_true")
    _ = parser.add_argument("--block-private-compute", action="store_true")
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
        use_block_private_compute=args.block_private_compute,
        compare=args.compare,
        print_kernel=args.print_kernel,
        print_device_code=args.print_device_code,
        run=args.run_kernel,
        warmup=args.warmup,
        iterations=args.iterations,
    )

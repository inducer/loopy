"""Benchmark compressed Cartesian Taylor M2M sum factorization with compute.

FMM kernel class: compressed Cartesian Taylor/asymptotic multipole-to-multipole
translation.  This is the binomial center-shift part of an FMM translation, not
a direct evaluation of a Green's function such as Laplace, Helmholtz, or
biharmonic.  The translation weights are powers of the center displacement
divided by factorials; PDE-specific derivative generation, compression
matrices, and recompression are outside this microbenchmark.

The stored-index pattern here is intentionally simple.  The 2D mode stores the
two coordinate axes, and the 3D mode stores the three coordinate axes.  That
captures the sum-factorized structure from Section 4.2.3, but it is not the
full compressed 3D Laplace stored set, which would retain PDE-derived
hyperplane layers with O(p**2) stored coefficients rather than only O(p) axis
coefficients.

This script models a multipole-to-multipole-like translation in 2D or 3D where
the input expansion is stored only on the coordinate axes.  For 2D, the stored
coefficients are ``beta[zeta0, 0]`` and ``beta[0, zeta1]``.  For 3D, they are
``beta[zeta0, 0, 0]``, ``beta[0, zeta1, 0]``, and ``beta[0, 0, zeta2]``.  The
output still fills the full tensor-product coefficient grid.

The inline variant is a GPU-parallel kernel over output coefficient indices
that expands the one-dimensional translation sums at each output.  The compute
variant uses :func:`loopy.transform.compute.compute` to materialize those axis
sums into private temporaries and reuses them across an ILP tile of the last
output axis.  In 2D it tiles ``eta1``; in 3D it tiles ``eta2`` while reusing the
``eta0`` and ``eta1`` axis sums across that register tile.

Use ``--dimension 2`` or ``--dimension 3`` to choose the kernel.  Use
``--compare`` to run both GPU-parallel variants, check against the NumPy
reference implementation, and report timing, modeled GFLOP/s, speedup, and
relative error.
"""

import os
import time

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import namedisl as nisl
import numpy as np
import numpy.linalg as la
import pymbolic.primitives as p

import loopy as lp
import loopy.transform.compute as compute_mod
from loopy.symbolic import DependencyMapper
from loopy.transform.compute import compute
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2


def enable_compute_for_reduction_substitutions() -> None:
    """Let compute inspect substitution rules whose expressions are reductions."""

    def gather_vars(expr):
        deps = DependencyMapper()(expr)
        var_names = set()
        for dep in deps:
            if isinstance(dep, p.Variable):
                var_names.add(dep.name)
            elif (
                    isinstance(dep, p.Subscript)
                    and isinstance(dep.aggregate, p.Variable)):
                var_names.add(dep.aggregate.name)

        return var_names

    compute_mod._gather_vars = gather_vars


def translation_weights(h: np.ndarray, order: int) -> np.ndarray:
    weights = np.empty((len(h), order + 1), dtype=h.dtype)
    weights[:, 0] = 1

    for axis in range(len(h)):
        for n in range(1, order + 1):
            weights[axis, n] = weights[axis, n - 1] * h[axis] / n

    return weights


def make_axis_compressed_coefficients(
        order: int,
        dimension: int,
        dtype
    ) -> np.ndarray:
    rng = np.random.default_rng(12)
    beta = np.zeros(dimension * (order + 1,), dtype=dtype)

    for axis in range(dimension):
        axis_slice = [0] * dimension
        axis_slice[axis] = slice(None)
        beta[tuple(axis_slice)] = rng.normal(size=order + 1)

    beta[(0,) * dimension] = rng.normal()

    return beta


def reference_axis_m2m_2d(beta: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = beta.shape[0] - 1
    sigma = np.empty_like(beta)

    for eta0 in range(order + 1):
        for eta1 in range(order + 1):
            acc = 0

            for zeta1 in range(eta1 + 1):
                acc += (
                    weights[0, eta0]
                    * weights[1, eta1 - zeta1]
                    * beta[0, zeta1]
                )

            for zeta0 in range(eta0 + 1):
                acc += (
                    weights[0, eta0 - zeta0]
                    * weights[1, eta1]
                    * beta[zeta0, 0]
                )

            acc -= weights[0, eta0] * weights[1, eta1] * beta[0, 0]
            sigma[eta0, eta1] = acc

    return sigma


def reference_axis_m2m_3d(beta: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = beta.shape[0] - 1
    sigma = np.empty_like(beta)

    for eta0 in range(order + 1):
        for eta1 in range(order + 1):
            for eta2 in range(order + 1):
                acc = 0

                for zeta0 in range(eta0 + 1):
                    acc += (
                        weights[0, eta0 - zeta0]
                        * weights[1, eta1]
                        * weights[2, eta2]
                        * beta[zeta0, 0, 0]
                    )

                for zeta1 in range(eta1 + 1):
                    acc += (
                        weights[0, eta0]
                        * weights[1, eta1 - zeta1]
                        * weights[2, eta2]
                        * beta[0, zeta1, 0]
                    )

                for zeta2 in range(eta2 + 1):
                    acc += (
                        weights[0, eta0]
                        * weights[1, eta1]
                        * weights[2, eta2 - zeta2]
                        * beta[0, 0, zeta2]
                    )

                acc -= (
                    2
                    * weights[0, eta0]
                    * weights[1, eta1]
                    * weights[2, eta2]
                    * beta[0, 0, 0]
                )
                sigma[eta0, eta1, eta2] = acc

    return sigma


def make_kernel_2d(
        order: int,
        eta_tile_size: int,
        dtype,
        use_compute: bool = False
    ) -> lp.TranslationUnit:
    if (order + 1) % eta_tile_size:
        raise ValueError("order + 1 must be divisible by eta_tile_size")

    knl = lp.make_kernel(
        "{ [eta0, eta1, zeta0, zeta1] : 0 <= eta0, eta1, zeta0, zeta1 <= p }",
        """
        x_axis_sum_(eta0_arg) := sum(
            [zeta0],
            if(
                zeta0 <= eta0_arg,
                w[0, eta0_arg - zeta0] * beta[zeta0, 0],
                0
            )
        )

        y_axis_sum_(eta1_arg) := sum(
            [zeta1],
            if(
                zeta1 <= eta1_arg,
                w[1, eta1_arg - zeta1] * beta[0, zeta1],
                0
            )
        )

        sigma[eta0, eta1] = (
            w[0, eta0] * y_axis_sum_(eta1)
            + w[1, eta1] * x_axis_sum_(eta0)
            - w[0, eta0] * w[1, eta1] * beta[0, 0]
        )
        """,
        [
            lp.GlobalArg("beta", dtype=dtype, shape=(order + 1, order + 1)),
            lp.GlobalArg("w", dtype=dtype, shape=(2, order + 1)),
            lp.GlobalArg("sigma", dtype=dtype, shape=(order + 1, order + 1),
                         is_output=True),
        ],
        lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2,
    )

    knl = lp.fix_parameters(knl, p=order)
    knl = lp.split_iname(
        knl,
        "eta1",
        eta_tile_size,
        inner_iname="eta1_inner",
        outer_iname="eta1_block",
    )

    if use_compute:
        x_axis_sum_map = nisl.make_map(f"""{{
            [eta0_arg] -> [eta0, eta1_block, x_slot] :
                eta0_arg = eta0 and x_slot = 0
        }}""")
        knl = compute(
            knl,
            "x_axis_sum_",
            compute_map=x_axis_sum_map,
            storage_indices=["x_slot"],
            temporal_inames=["eta0", "eta1_block"],
            temporary_name="x_axis_sum_vec",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_dtype=dtype,
            compute_insn_id="x_axis_sum_compute",
        )

        y_axis_sum_map = nisl.make_map(f"""{{
            [eta1_arg] -> [eta0, eta1_block, y_slot] :
                eta1_arg = eta1_block * {eta_tile_size} + y_slot
        }}""")
        knl = compute(
            knl,
            "y_axis_sum_",
            compute_map=y_axis_sum_map,
            storage_indices=["y_slot"],
            temporal_inames=["eta0", "eta1_block"],
            temporary_name="y_axis_sum_vec",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_dtype=dtype,
            compute_insn_id="y_axis_sum_compute",
        )
        knl = lp.tag_inames(knl, {
            "x_slot": "unr",
            "y_slot": "unr",
        })

    knl = lp.tag_inames(knl, {
        "eta0": "g.1",
        "eta1_block": "g.0",
        "eta1_inner": "ilp",
    })
    return knl


def make_kernel_3d(
        order: int,
        eta_tile_size: int,
        dtype,
        use_compute: bool = False
    ) -> lp.TranslationUnit:
    if (order + 1) % eta_tile_size:
        raise ValueError("order + 1 must be divisible by eta_tile_size")

    knl = lp.make_kernel(
        """
        { [eta0, eta1, eta2, zeta0, zeta1, zeta2] :
            0 <= eta0, eta1, eta2, zeta0, zeta1, zeta2 <= p }
        """,
        """
        x_axis_sum_(eta0_arg) := sum(
            [zeta0],
            if(
                zeta0 <= eta0_arg,
                w[0, eta0_arg - zeta0] * beta[zeta0, 0, 0],
                0
            )
        )

        y_axis_sum_(eta1_arg) := sum(
            [zeta1],
            if(
                zeta1 <= eta1_arg,
                w[1, eta1_arg - zeta1] * beta[0, zeta1, 0],
                0
            )
        )

        z_axis_sum_(eta2_arg) := sum(
            [zeta2],
            if(
                zeta2 <= eta2_arg,
                w[2, eta2_arg - zeta2] * beta[0, 0, zeta2],
                0
            )
        )

        sigma[eta0, eta1, eta2] = (
            w[1, eta1] * w[2, eta2] * x_axis_sum_(eta0)
            + w[0, eta0] * w[2, eta2] * y_axis_sum_(eta1)
            + w[0, eta0] * w[1, eta1] * z_axis_sum_(eta2)
            - 2 * w[0, eta0] * w[1, eta1] * w[2, eta2] * beta[0, 0, 0]
        )
        """,
        [
            lp.GlobalArg(
                "beta", dtype=dtype, shape=(order + 1, order + 1, order + 1)),
            lp.GlobalArg("w", dtype=dtype, shape=(3, order + 1)),
            lp.GlobalArg(
                "sigma", dtype=dtype,
                shape=(order + 1, order + 1, order + 1),
                is_output=True),
        ],
        lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2,
    )

    knl = lp.fix_parameters(knl, p=order)
    knl = lp.split_iname(
        knl,
        "eta2",
        eta_tile_size,
        inner_iname="eta2_inner",
        outer_iname="eta2_block",
    )

    if use_compute:
        x_axis_sum_map = nisl.make_map("""
            {
                [eta0_arg] -> [eta0, eta1, eta2_block, x_slot] :
                    eta0_arg = eta0 and x_slot = 0
            }
        """)
        knl = compute(
            knl,
            "x_axis_sum_",
            compute_map=x_axis_sum_map,
            storage_indices=["x_slot"],
            temporal_inames=["eta0", "eta1", "eta2_block"],
            temporary_name="x_axis_sum_vec",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_dtype=dtype,
            compute_insn_id="x_axis_sum_compute",
        )

        y_axis_sum_map = nisl.make_map("""
            {
                [eta1_arg] -> [eta0, eta1, eta2_block, y_slot] :
                    eta1_arg = eta1 and y_slot = 0
            }
        """)
        knl = compute(
            knl,
            "y_axis_sum_",
            compute_map=y_axis_sum_map,
            storage_indices=["y_slot"],
            temporal_inames=["eta0", "eta1", "eta2_block"],
            temporary_name="y_axis_sum_vec",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_dtype=dtype,
            compute_insn_id="y_axis_sum_compute",
        )

        z_axis_sum_map = nisl.make_map(f"""{{
            [eta2_arg] -> [eta0, eta1, eta2_block, z_slot] :
                eta2_arg = eta2_block * {eta_tile_size} + z_slot
        }}""")
        knl = compute(
            knl,
            "z_axis_sum_",
            compute_map=z_axis_sum_map,
            storage_indices=["z_slot"],
            temporal_inames=["eta0", "eta1", "eta2_block"],
            temporary_name="z_axis_sum_vec",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_dtype=dtype,
            compute_insn_id="z_axis_sum_compute",
        )
        knl = lp.tag_inames(knl, {
            "x_slot": "unr",
            "y_slot": "unr",
            "z_slot": "unr",
        })

    knl = lp.tag_inames(knl, {
        "eta0": "g.2",
        "eta1": "g.1",
        "eta2_block": "g.0",
        "eta2_inner": "ilp",
    })
    return knl


def make_kernel(
        order: int,
        eta_tile_size: int,
        dimension: int,
        dtype,
        use_compute: bool = False
    ) -> lp.TranslationUnit:
    if dimension == 2:
        return make_kernel_2d(order, eta_tile_size, dtype, use_compute)
    if dimension == 3:
        return make_kernel_3d(order, eta_tile_size, dtype, use_compute)
    raise ValueError("dimension must be 2 or 3")


def reference_axis_m2m(beta: np.ndarray, weights: np.ndarray) -> np.ndarray:
    dimension = beta.ndim
    if dimension == 2:
        return reference_axis_m2m_2d(beta, weights)
    if dimension == 3:
        return reference_axis_m2m_3d(beta, weights)
    raise ValueError("dimension must be 2 or 3")


def operation_model(
        order: int,
        eta_tile_size: int,
        dimension: int
    ) -> tuple[int, int]:
    ncoeff = order + 1
    if dimension == 2:
        inline_sum_terms = 2 * ncoeff**3
        tiled_compute_sum_terms = ncoeff**3 + ncoeff**3 // eta_tile_size
    elif dimension == 3:
        inline_sum_terms = 3 * ncoeff**4
        tiled_compute_sum_terms = ncoeff**4 + 2 * ncoeff**4 // eta_tile_size
    else:
        raise ValueError("dimension must be 2 or 3")
    return inline_sum_terms, tiled_compute_sum_terms


def m2m_flop_count(
        order: int,
        eta_tile_size: int,
        dimension: int,
        use_compute: bool
    ) -> int:
    ncoeff = order + 1

    if dimension == 2:
        if use_compute:
            sum_flops = 2 * ncoeff**3 + 2 * ncoeff**3 // eta_tile_size
        else:
            sum_flops = 4 * ncoeff**3
        correction_flops = 3 * ncoeff**2
    elif dimension == 3:
        if use_compute:
            sum_flops = 2 * ncoeff**4 + 4 * ncoeff**4 // eta_tile_size
        else:
            sum_flops = 6 * ncoeff**4
        correction_flops = 8 * ncoeff**3
    else:
        raise ValueError("dimension must be 2 or 3")

    return sum_flops + correction_flops


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


def run_kernel(
        knl: lp.TranslationUnit,
        beta: np.ndarray,
        weights: np.ndarray,
        warmup: int,
        iterations: int
    ) -> tuple[np.ndarray, float]:
    import pyopencl as cl
    import pyopencl.array as cl_array

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    ex = knl.executor(queue)

    beta_cl = cl_array.to_device(queue, beta)
    weights_cl = cl_array.to_device(queue, weights)
    sigma_cl = cl_array.zeros(queue, beta.shape, dtype=beta.dtype)

    elapsed = benchmark_executor(
        ex,
        queue,
        {"beta": beta_cl, "w": weights_cl, "sigma": sigma_cl},
        warmup=warmup,
        iterations=iterations,
    )

    _, out = ex(queue, beta=beta_cl, w=weights_cl, sigma=sigma_cl)
    return out[0].get(), elapsed


def main(
        order: int = 16,
        eta_tile_size: int = 8,
        dimension: int = 2,
        use_compute: bool = False,
        compare: bool = False,
        print_kernel: bool = False,
        print_device_code: bool = False,
        run: bool = False,
        warmup: int = 3,
        iterations: int = 10
    ) -> None:
    if order < 0:
        raise ValueError("order must be nonnegative")
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")

    dtype = np.float64
    h = np.array([0.25, -0.2, 0.15][:dimension], dtype=dtype)
    weights = translation_weights(h, order)
    beta = make_axis_compressed_coefficients(order, dimension, dtype)
    reference = reference_axis_m2m(beta, weights)

    inline_terms, compute_terms = operation_model(
        order, eta_tile_size, dimension)

    variants = [False, True] if compare else [use_compute]
    timings: dict[bool, float] = {}
    for variant_uses_compute in variants:
        knl = make_kernel(
            order, eta_tile_size, dimension, dtype,
            use_compute=variant_uses_compute)
        modeled_flops = m2m_flop_count(
            order, eta_tile_size, dimension,
            use_compute=variant_uses_compute)

        print(20 * "=", "Compressed M2M report", 20 * "=")
        print(f"Variant: {'compute sum-factorized' if variant_uses_compute else 'inline'}")
        print(f"Dimension: {dimension}D")
        print(f"Order  : {order}")
        print(f"Eta tile: {eta_tile_size}")
        if dimension == 2:
            print("Stored compressed set: zeta0 = 0 or zeta1 = 0")
        else:
            print("Stored compressed set: exactly one zeta axis may be nonzero")
        print(f"Inline sum terms      : {inline_terms}")
        print(f"Tiled compute sum terms: {compute_terms}")
        print(f"Modeled flop count    : {modeled_flops}")

        if print_kernel:
            print(knl)

        if print_device_code:
            print(lp.generate_code_v2(knl).device_code())

        if run or compare:
            try:
                result, elapsed = run_kernel(
                    knl, beta, weights, warmup=warmup, iterations=iterations)
            except Exception as exc:
                print(f"Runtime execution unavailable: {exc}")
            else:
                rel_err = la.norm(result - reference) / la.norm(reference)
                gflops = modeled_flops / elapsed * 1e-9
                timings[variant_uses_compute] = elapsed
                print(f"Average time per iteration: {elapsed:.6e} s")
                print(f"Modeled throughput: {gflops:.3f} GFLOP/s")
                print(f"Relative error: {rel_err:.3e}")

        print((40 + len(" Compressed M2M report ")) * "=")

    if compare and False in timings and True in timings:
        speedup = timings[False] / timings[True]
        time_reduction = (1 - timings[True] / timings[False]) * 100
        print(f"Speedup: {speedup:.3f}x")
        print(f"Relative time reduction: {time_reduction:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    _ = parser.add_argument("--order", action="store", type=int, default=16)
    _ = parser.add_argument("--eta-tile-size", action="store", type=int, default=8)
    _ = parser.add_argument("--dimension", action="store", type=int, choices=(2, 3),
                            default=2)
    _ = parser.add_argument("--dim", action="store", type=int, choices=(2, 3),
                            dest="dimension")
    _ = parser.add_argument("--compute", action="store_true")
    _ = parser.add_argument("--compare", action="store_true")
    _ = parser.add_argument("--run-kernel", action="store_true")
    _ = parser.add_argument("--print-kernel", action="store_true")
    _ = parser.add_argument("--print-device-code", action="store_true")
    _ = parser.add_argument("--warmup", action="store", type=int, default=3)
    _ = parser.add_argument("--iterations", action="store", type=int, default=10)

    args = parser.parse_args()

    main(
        order=args.order,
        eta_tile_size=args.eta_tile_size,
        dimension=args.dimension,
        use_compute=args.compute,
        compare=args.compare,
        print_kernel=args.print_kernel,
        print_device_code=args.print_device_code,
        run=args.run_kernel,
        warmup=args.warmup,
        iterations=args.iterations,
    )

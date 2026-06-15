
import namedisl as nisl
import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pyopencl.array as cl_array

import loopy as lp
from benchmark_output import (
    add_json_report_argument,
    benchmark_report,
    variant_result,
    write_json_report,
)
from loopy.transform.compute import compute
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2


def matmul_flop_count(m: int, n: int, k: int) -> int:
    return 2 * m * n * k


def matmul_byte_count(m: int, n: int, k: int, dtype) -> int:
    itemsize = np.dtype(dtype).itemsize
    return itemsize * (m * k + k * n + m * n)


def benchmark_kernel(
    knl: lp.TranslationUnit,
    queue: cl.CommandQueue,
    a: np.ndarray,
    b: np.ndarray,
    variant_name: str,
    role: str,
    tile_metadata: dict,
    nwarmup: int = 5,
    niterations: int = 20
) -> dict:
    ex = knl.executor(queue)

    a_cl = cl_array.to_device(queue, a)
    b_cl = cl_array.to_device(queue, b)
    c_cl = cl_array.zeros(queue, (a.shape[0], b.shape[1]), dtype=a_cl.dtype)

    start = cl.enqueue_marker(queue)
    for _ in range(nwarmup):
        ex(queue, a=a_cl, b=b_cl, c=c_cl)
    end = cl.enqueue_marker(queue)
    end.wait()
    start.wait()

    start = cl.enqueue_marker(queue)
    for _ in range(niterations):
        ex(queue, a=a_cl, b=b_cl, c=c_cl)
    end = cl.enqueue_marker(queue)
    end.wait()
    start.wait()

    total_ns = end.profile.end - start.profile.end
    total_elapsed_s = total_ns * 1e-9
    s_per_iter = total_elapsed_s / niterations

    total_flops = matmul_flop_count(a.shape[0], b.shape[1], a.shape[1])
    gflops = (total_flops / s_per_iter) * 1e-9

    c_ref = a @ b
    _, c_res = ex(queue, a=a_cl, b=b_cl, c=c_cl)

    error = la.norm(c_res[0].get() - c_ref) / la.norm(c_ref)

    m, k = a.shape
    _, n = b.shape
    print("================= Results =================")
    print(f"M = {m}, N = {n}, K = {k}")
    print(f"           Error = {error:.4}")
    print(f"   Total time (s): {total_elapsed_s:.4}")
    print(f"Time per iter (s): {s_per_iter:.4}")
    print(f"          GFLOP/s: {gflops}")
    print("===========================================")

    return variant_result(
        variant_name,
        role,
        time_s=s_per_iter,
        flop_count=total_flops,
        bytes_moved=matmul_byte_count(m, n, k, a.dtype),
        dtype=np.dtype(a.dtype).name,
        relative_error=error,
        metadata={
            **tile_metadata,
            "byte_model": "a, b, and c matrix arrays",
        },
    )


def naive_matmul(
        knl: lp.TranslationUnit,
        bm: int,
        bn: int,
        bk: int
    ) -> lp.TranslationUnit:
    knl = lp.split_iname(knl, "i", bm, inner_iname="ii", outer_iname="io")
    knl = lp.split_iname(knl, "j", bn, inner_iname="ji", outer_iname="jo")
    knl = lp.split_iname(knl, "k", bk, inner_iname="ki", outer_iname="ko")

    iname_tags = {
        "io": "g.1",
        "jo": "g.0",

        "ii": "l.1",
        "ji": "l.0"
    }

    return lp.tag_inames(knl, iname_tags)


def shared_memory_tiled_matmul(
        knl: lp.TranslationUnit,
        bm: int,
        bn: int,
        bk: int
    ) -> lp.TranslationUnit:
    knl = lp.split_iname(knl, "i", bm, inner_iname="ii", outer_iname="io")
    knl = lp.split_iname(knl, "j", bn, inner_iname="ji", outer_iname="jo")
    knl = lp.split_iname(knl, "k", bk, inner_iname="ki", outer_iname="ko")

    compute_map_a = nisl.make_map(f"""{{
        [is, ks] -> [a_ii, io, a_ki, ko] :
            is = io * {bm} + a_ii and
            ks = ko * {bk} + a_ki
    }}""")

    compute_map_b = nisl.make_map(f"""{{
        [ks, js] -> [b_ki, ko, b_ji, jo] :
            js = jo * {bn} + b_ji and
            ks = ko * {bk} + b_ki
    }}""")

    knl = compute(
        knl,
        "a_",
        compute_map=compute_map_a,
        storage_indices=["a_ii", "a_ki"],
        placement_inames=["jo"],
        temporary_name="a_tile",
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id="a_load"
    )

    knl = compute(
        knl,
        "b_",
        compute_map=compute_map_b,
        storage_indices=["b_ki", "b_ji"],
        placement_inames=["io"],
        temporary_name="b_tile",
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id="b_load"
    )

    iname_tags = {
        "io": "g.1",
        "ii": "l.1",

        "jo": "g.0",
        "ji": "l.0",

        "a_ii": "l.1",
        "a_ki": "l.0",

        "b_ki": "l.1",
        "b_ji": "l.0"
    }

    return lp.tag_inames(knl, iname_tags)


def register_tiled_matmul(
        knl: lp.TranslationUnit,
        bm: int,
        bn: int,
        bk: int,
        tm: int,
        tn: int
    ) -> lp.TranslationUnit:

    # {{{ shared-memory-level split / compute

    knl = lp.split_iname(knl, "i", bm, inner_iname="ii", outer_iname="io")
    knl = lp.split_iname(knl, "j", bn, inner_iname="ji", outer_iname="jo")
    knl = lp.split_iname(knl, "k", bk, inner_iname="ki", outer_iname="ko")

    compute_map_a = nisl.make_map(f"""{{
        [is, ks] -> [a_ii, io, a_ki, ko] :
            is = io * {bm} + a_ii and
            ks = ko * {bk} + a_ki
    }}""")

    compute_map_b = nisl.make_map(f"""{{
        [ks, js] -> [b_ki, ko, b_ji, jo] :
            js = jo * {bn} + b_ji and
            ks = ko * {bk} + b_ki
    }}""")

    knl = compute(
        knl,
        "a_",
        compute_map=compute_map_a,
        storage_indices=["a_ii", "a_ki"],
        placement_inames=["jo"],
        temporary_name="a_smem",
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id="a_load"
    )

    knl = compute(
        knl,
        "b_",
        compute_map=compute_map_b,
        storage_indices=["b_ki", "b_ji"],
        placement_inames=["io"],
        temporary_name="b_smem",
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id="b_load"
    )

    wg_size_i = bm // tm
    wg_size_j = bn // tn
    knl = lp.split_iname(
        knl,
        "a_ii",
        wg_size_i,
        inner_iname="a_local",
        outer_iname="a_tile"
    )

    knl = lp.split_iname(
        knl,
        "b_ji",
        wg_size_j,
        inner_iname="b_local",
        outer_iname="b_tile"
    )

    # }}}

    # {{{ register-level split / compute

    knl = lp.extract_subst(
        knl,
        "a_smem_",
        "a_smem[is, ks]",
        parameters="is, ks"
    )

    knl = lp.extract_subst(
        knl,
        "b_smem_",
        "b_smem[ks, js]",
        parameters="ks, js"
    )

    knl = lp.split_iname(knl, "ii", tm,
                         inner_iname="ii_reg",
                         outer_iname="ii_thr")

    knl = lp.split_iname(knl, "ji", tn,
                         inner_iname="ji_reg",
                         outer_iname="ji_thr")

    knl = lp.split_iname(knl, "ki", 8,
                         inner_iname="dot",
                         outer_iname="ki_outer")

    a_reg_tile = nisl.make_map(f"""{{
        [is, ks] -> [a_reg_i, ii_thr, ki_outer, dot] :
        is = ii_thr * {tm} + a_reg_i and
        ks = ki_outer * 8 + dot
    }}""")

    b_reg_tile = nisl.make_map(f"""{{
        [ks, js] -> [b_reg_j, ki_outer, dot, ji_thr] :
        ks = ki_outer * 8 + dot and
        js = ji_thr * {tn} + b_reg_j
    }}""")

    knl = compute(
        knl,
        "a_smem_",
        compute_map=a_reg_tile,
        storage_indices=["a_reg_i"],
        placement_inames=["ji_thr", "io", "jo", "ko"],
        temporary_name="a_reg",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        compute_insn_id="a_reg_load"
    )

    knl = compute(
        knl,
        "b_smem_",
        compute_map=b_reg_tile,
        storage_indices=["b_reg_j"],
        placement_inames=["ii_thr", "io", "jo", "ko"],
        temporary_name="b_reg",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        compute_insn_id="b_reg_load"
    )

    # }}}

    iname_tags = {
        # global tiles
        "io": "g.1",
        "jo": "g.0",

        # a local storage axes
        "a_local": "l.1",
        "a_ki": "l.0",

        # b local storage axes
        "b_local": "l.0",
        "b_ki": "l.1",

        # register tiles
        "ii_thr": "l.1",
        "ji_thr": "l.0",

        # register storage axes
        "a_reg_i": "ilp",
        "b_reg_j": "ilp",

        # compute axes
        "ii_reg": "ilp",
        "ji_reg": "ilp"
    }

    return lp.tag_inames(knl, iname_tags)


def make_base_kernel(
        m: int,
        n: int,
        k: int,
        dtype: lp.ToLoopyTypeConvertible
    ) -> lp.TranslationUnit:
    knl = lp.make_kernel(
        "{ [i, j, k] : 0 <= i < M and 0 <= j < N and 0 <= k < K }",
        """
        a_(is, ks) := a[is, ks]
        b_(ks, js) := b[ks, js]

        c[i, j] = sum([k], a_(i, k) * b_(k, j))
        """,
        [
            lp.GlobalArg("a", shape=(m, k), dtype=dtype),
            lp.GlobalArg("b", shape=(k, n), dtype=dtype),
            lp.GlobalArg("c", shape=(m, n), is_output=True)
        ],
        lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2,
    )

    return lp.fix_parameters(knl, M=m, N=n, K=k)


def main(
        m: int = 1024,
        n: int = 1024,
        k: int = 1024,
        bm: int = 64,
        bn: int = 64,
        bk: int = 32,
        tm: int = 4,
        tn: int = 4,
        shared_memory_tiled: bool = False,
        register_tiled: bool = False,
        compare: bool = False,
        dtype: lp.ToLoopyTypeConvertible = np.float32,
        print_kernel: bool = False,
        print_device_code: bool = False,
        run_kernel: bool = True,
        warmup: int = 5,
        iterations: int = 20
    ) -> list[dict]:

    queue = None
    a = None
    b = None
    if run_kernel:
        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(
            ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE
        )

        rng = np.random.default_rng()
        a = rng.standard_normal((m, k)).astype(dtype)
        b = rng.standard_normal((k, n)).astype(dtype)

    if compare:
        base_bm = min(bm, 32)
        base_bn = min(bn, 32)
        variant_specs = [
            ("naive", "baseline", {"bm": base_bm, "bn": base_bn, "bk": bk}),
            (
                "shared-memory tiled",
                "optimized",
                {"bm": base_bm, "bn": base_bn, "bk": bk},
            ),
            (
                "register tiled",
                "optimized",
                {"bm": bm, "bn": bn, "bk": bk, "tm": tm, "tn": tn},
            ),
        ]
    elif shared_memory_tiled:
        variant_specs = [
            ("shared-memory tiled", "optimized", {"bm": bm, "bn": bn, "bk": bk})
        ]
    elif register_tiled:
        variant_specs = [
            (
                "register tiled",
                "optimized",
                {"bm": bm, "bn": bn, "bk": bk, "tm": tm, "tn": tn},
            )
        ]
    else:
        variant_specs = [
            ("naive", "baseline", {"bm": bm, "bn": bn, "bk": bk})
        ]

    variant_records = []
    for variant_name, role, tile_metadata in variant_specs:
        try:
            knl = make_base_kernel(m, n, k, dtype)
            if variant_name == "shared-memory tiled":
                knl = shared_memory_tiled_matmul(
                    knl,
                    tile_metadata["bm"],
                    tile_metadata["bn"],
                    tile_metadata["bk"],
                )
            elif variant_name == "register tiled":
                knl = register_tiled_matmul(
                    knl,
                    tile_metadata["bm"],
                    tile_metadata["bn"],
                    tile_metadata["bk"],
                    tile_metadata["tm"],
                    tile_metadata["tn"],
                )
            else:
                knl = naive_matmul(
                    knl,
                    tile_metadata["bm"],
                    tile_metadata["bn"],
                    tile_metadata["bk"],
                )

            if print_kernel:
                print(knl)

            if print_device_code:
                print(lp.generate_code_v2(knl).device_code())

            if run_kernel:
                assert queue is not None
                assert a is not None
                assert b is not None
                record = benchmark_kernel(
                    knl,
                    queue,
                    a,
                    b,
                    variant_name,
                    role,
                    tile_metadata,
                    nwarmup=warmup,
                    niterations=iterations,
                )
            else:
                record = variant_result(
                    variant_name,
                    role,
                    flop_count=matmul_flop_count(m, n, k),
                    bytes_moved=matmul_byte_count(m, n, k, dtype),
                    dtype=np.dtype(dtype).name,
                    metadata={
                        **tile_metadata,
                        "byte_model": "a, b, and c matrix arrays",
                    },
                )
        except Exception as exc:
            print(f"Runtime execution unavailable for {variant_name}: {exc}")
            record = variant_result(
                variant_name,
                role,
                flop_count=matmul_flop_count(m, n, k),
                bytes_moved=matmul_byte_count(m, n, k, dtype),
                dtype=np.dtype(dtype).name,
                metadata={
                    **tile_metadata,
                    "byte_model": "a, b, and c matrix arrays",
                },
                error=str(exc),
            )
        variant_records.append(record)

    if compare:
        timings = {
            record["name"]: record["time_s"]
            for record in variant_records
            if "time_s" in record and "error" not in record
        }
        if "naive" in timings:
            for variant_name in ("shared-memory tiled", "register tiled"):
                if variant_name in timings:
                    speedup = timings["naive"] / timings[variant_name]
                    time_reduction = (
                        1 - timings[variant_name] / timings["naive"]) * 100
                    print(
                        f"{variant_name} speedup: {speedup:.3f}x; "
                        f"relative time reduction: {time_reduction:.2f}%"
                    )

    return variant_records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    _ = parser.add_argument("--m", action="store", type=int, default=1024)
    _ = parser.add_argument("--n", action="store", type=int, default=1024)
    _ = parser.add_argument("--k", action="store", type=int, default=1024)

    _ = parser.add_argument("--bm", action="store", type=int, default=64)
    _ = parser.add_argument("--bn", action="store", type=int, default=64)
    _ = parser.add_argument("--bk", action="store", type=int, default=16)

    _ = parser.add_argument("--tm", action="store", type=int, default=4)
    _ = parser.add_argument("--tn", action="store", type=int, default=4)

    _ = parser.add_argument("--shared-memory-tiled", action="store_true")
    _ = parser.add_argument("--register-tiled", action="store_true")
    _ = parser.add_argument("--compare", action="store_true")

    _ = parser.add_argument("--print-kernel", action="store_true")
    _ = parser.add_argument("--print-device-code", action="store_true")
    _ = parser.add_argument("--run-kernel", action="store_true", default=True)
    _ = parser.add_argument("--no-run-kernel", action="store_false",
                            dest="run_kernel")
    _ = parser.add_argument("--warmup", action="store", type=int, default=5)
    _ = parser.add_argument("--iterations", action="store", type=int, default=20)
    add_json_report_argument(parser, "matmul.json")

    args = parser.parse_args()

    variants = main(
        m=args.m, n=args.n, k=args.k,
        bm=args.bm, bn=args.bn, bk=args.bk,
        tm=args.tm, tn=args.tn,
        shared_memory_tiled=args.shared_memory_tiled,
        register_tiled=args.register_tiled,
        compare=args.compare,
        print_kernel=args.print_kernel,
        print_device_code=args.print_device_code,
        run_kernel=args.run_kernel,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    write_json_report(
        args.json_report,
        benchmark_report(
            example="matmul",
            description="Dense matrix multiplication schedule comparison",
            parameters={
                "m": args.m,
                "n": args.n,
                "k": args.k,
                "bm": args.bm,
                "bn": args.bn,
                "bk": args.bk,
                "tm": args.tm,
                "tn": args.tn,
                "warmup": args.warmup,
                "iterations": args.iterations,
            },
            baseline_name="naive",
            variants=variants,
        ),
    )

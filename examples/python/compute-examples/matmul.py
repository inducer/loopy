import time

import namedisl as nisl
import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pyopencl.array as cl_array

import loopy as lp
from loopy.transform.compute import compute
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2


def benchmark_kernel(
    knl: lp.TranslationUnit,
    queue: cl.CommandQueue,
    a: np.ndarray,
    b: np.ndarray,
    nwarmup: int = 5,
    niterations: int = 20
):
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

    total_flops = 2 * a.shape[0] * a.shape[1] * b.shape[1]
    gflops = (total_flops / s_per_iter) * 1e-9

    c_ref = a @ b
    _, c_res = ex(queue, a=a_cl, b=b_cl, c=c_cl)

    error = la.norm(c_res[0].get() - c_ref) / la.norm(c_ref)

    m, k = a.shape
    _, n = b.shape
    print(f"================= Results =================")
    print(f"M = {m}, N = {n}, K = {k}")
    print(f"           Error = {error:.4}")
    print(f"   Total time (s): {total_elapsed_s:.4}")
    print(f"Time per iter (s): {s_per_iter:.4}")
    print(f"          GFLOP/s: {gflops}")
    print(f"===========================================")


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
        [is, ks] -> [a_ii, io, a_ki, ko, jo] :
            is = io * {bm} + a_ii and
            ks = ko * {bk} + a_ki
    }}""")

    compute_map_b = nisl.make_map(f"""{{
        [ks, js] -> [b_ki, ko, b_ji, jo, io] :
            js = jo * {bn} + b_ji and
            ks = ko * {bk} + b_ki
    }}""")

    knl = compute(
        knl,
        "a_",
        compute_map=compute_map_a,
        storage_indices=["a_ii", "a_ki"],
        temporal_inames=["io", "ko"],
        temporary_name="a_tile",
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id="a_load"
    )

    knl = compute(
        knl,
        "b_",
        compute_map=compute_map_b,
        storage_indices=["b_ki", "b_ji"],
        temporal_inames=["ko", "jo", "io"],
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
        [is, ks] -> [a_ii, io, a_ki, ko, jo] :
            is = io * {bm} + a_ii and
            ks = ko * {bk} + a_ki
    }}""")

    compute_map_b = nisl.make_map(f"""{{
        [ks, js] -> [b_ki, ko, b_ji, jo, io] :
            js = jo * {bn} + b_ji and
            ks = ko * {bk} + b_ki
    }}""")

    knl = compute(
        knl,
        "a_",
        compute_map=compute_map_a,
        storage_indices=["a_ii", "a_ki"],
        temporal_inames=["io", "ko"],
        temporary_name="a_smem",
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id="a_load"
    )

    knl = compute(
        knl,
        "b_",
        compute_map=compute_map_b,
        storage_indices=["b_ki", "b_ji"],
        temporal_inames=["ko", "jo"],
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
        [is, ks] -> [a_reg_i, ii_thr, ji_thr, ki_outer, dot, io, jo, ko] :
        is = ii_thr * {tm} + a_reg_i and
        ks = ki_outer * 8 + dot
    }}""")

    b_reg_tile = nisl.make_map(f"""{{
        [ks, js] -> [b_reg_j, ki_outer, dot, ii_thr, ji_thr, io, jo, ko] :
        ks = ki_outer * 8 + dot and
        js = ji_thr * {tn} + b_reg_j
    }}""")

    knl = compute(
        knl,
        "a_smem_",
        compute_map=a_reg_tile,
        storage_indices=["a_reg_i"],
        temporal_inames=["ii_thr", "ji_thr", "ki_outer", "dot", "io", "jo", "ko"],
        temporary_name="a_reg",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        compute_insn_id="a_reg_load"
    )

    knl = compute(
        knl,
        "b_smem_",
        compute_map=b_reg_tile,
        storage_indices=["b_reg_j"],
        temporal_inames=["ii_thr", "ji_thr", "ki_outer", "dot", "io", "jo", "ko"],
        temporary_name="b_reg",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        compute_insn_id="b_reg_load"
    )

    # }}}

    iname_tags = {
        # global tiles
        "io" : "g.1",
        "jo" : "g.0",

        # a local storage axes
        "a_local": "l.1",
        "a_ki"   : "l.0",

        # b local storage axes
        "b_local": "l.0",
        "b_ki"   : "l.1",

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
        dtype=np.float32,
        print_kernel: bool = False,
        print_device_code: bool = False
    ) -> None:

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
        ]
    )

    knl = lp.fix_parameters(knl, M=m, N=n, K=k)

    if shared_memory_tiled:
        knl = shared_memory_tiled_matmul(knl, bm, bn, bk)
    elif register_tiled:
        knl = register_tiled_matmul(knl, bm, bn, bk, tm, tn)
    else:
        knl = naive_matmul(knl, bm, bn, bk)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE
    )

    a = np.random.randn(m, k).astype(dtype)
    b = np.random.randn(k, n).astype(dtype)

    benchmark_kernel(knl, queue, a, b)

    if print_kernel:
        print(knl)

    if print_device_code:
        print(lp.generate_code_v2(knl).device_code())


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

    _ = parser.add_argument("--print-kernel", action="store_true")
    _ = parser.add_argument("--print-device-code", action="store_true")

    args = parser.parse_args()

    main(
        m=args.m, n=args.n, k=args.k,
        bm=args.bm, bn=args.bn, bk=args.bk,
        tm=args.tm, tn=args.tn,
        shared_memory_tiled=args.shared_memory_tiled,
        register_tiled=args.register_tiled,
        print_kernel=args.print_kernel,
        print_device_code=args.print_device_code
    )

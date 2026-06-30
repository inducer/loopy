import loopy as lp
from loopy.transform.compute import compute
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

import namedisl as nisl

import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as cl_random


def _cuda_launch_dims(
    m: int,
    n: int,
    bm: int,
    bn: int,
    tm: int,
    tn: int,
):
    # j -> blockIdx.x, i -> blockIdx.y
    #
    # Each CUDA block computes a bm x bn output tile.
    # Each thread computes a tm x tn register tile.
    grid_dim = (
        (n + bn - 1) // bn,  # N / j dimension
        (m + bm - 1) // bm,  # M / i dimension
    )

    block_dim = (
        (bn + tn - 1) // tn,  # j threads
        (bm + tm - 1) // tm,  # i threads
        1,
    )

    return grid_dim, block_dim


def _build_cuda_kernel(knl: lp.TranslationUnit):
    import cupy as cu

    cuda_src = lp.generate_code_v2(knl).device_code()

    module = cu.RawModule(
        code=cuda_src,
        options=("--std=c++17",),
        backend="nvrtc",
    )

    return module.get_function("loopy_kernel")


def _compute_relative_error_cl(
    knl: lp.TranslationUnit,
    m: int,
    n: int,
    k: int,
    dtype: lp.ToLoopyTypeConvertible = np.float32,
) -> float:
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    a_cl = cl_random.rand(queue, (m, k), dtype=dtype)
    b_cl = cl_random.rand(queue, (k, n), dtype=dtype)
    c_cl = cl_array.zeros(queue, (m, n), dtype=dtype)

    _, out = knl(queue, A=a_cl, B=b_cl, C=c_cl)
    c = out[0].get()

    a = a_cl.get()
    b = b_cl.get()
    c_ref = a @ b

    return float(la.norm(c - c_ref) / la.norm(c_ref))


def _compute_relative_error_cuda(
    knl: lp.TranslationUnit,
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    tm: int,
    tn: int,
    dtype=np.float32,
) -> float:
    import cupy as cu

    cuda_knl = _build_cuda_kernel(knl)

    grid_dim, block_dim = _cuda_launch_dims(m, n, bm, bn, tm, tn)

    a = cu.random.randn(m, k, dtype=dtype)
    b = cu.random.randn(k, n, dtype=dtype)
    c = cu.zeros((m, n), dtype=dtype)

    cuda_knl(grid_dim, block_dim, (a, b, c))
    cu.cuda.Stream.null.synchronize()

    c_ref = a @ b

    err = cu.linalg.norm(c - c_ref)
    ref = cu.linalg.norm(c_ref)

    return float((err / ref).get())


def verify_correctness(
    knl: lp.TranslationUnit,
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    bk: int,
    tm: int,
    tn: int,
    dtype: lp.ToLoopyTypeConvertible = np.float32,
    use_cuda: bool = False,
):
    if use_cuda:
        rel_err = _compute_relative_error_cuda(knl, m, n, k, bm, bn, tm, tn, dtype)
    else:
        rel_err = _compute_relative_error_cl(knl, m, n, k, dtype)

    tol = 10 * np.finfo(np.dtype(dtype)).eps

    if rel_err > tol:
        raise RuntimeError(f"Correctness check failed: {rel_err:.4e} > {tol:.4e}")

    backend = "CUDA" if use_cuda else "OpenCL"
    print(f"{backend} success! (rel_err < tol) {rel_err:.4e} < {tol:.4e}")


def _benchmark_kernel_with_cl(
    knl: lp.TranslationUnit,
    m: int,
    n: int,
    k: int,
    niterations: int = 10,
    nwarmup: int = 3,
    dtype: lp.ToLoopyTypeConvertible = np.float32,
):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )

    a_cl = cl_random.rand(queue, (m, k), dtype=dtype)
    b_cl = cl_random.rand(queue, (k, n), dtype=dtype)
    c_cl = cl_array.zeros(queue, (m, n), dtype=dtype)

    knl_exec = knl.executor(queue)

    start = cl.enqueue_marker(queue)
    for _ in range(nwarmup):
        knl_exec(queue, A=a_cl, B=b_cl, C=c_cl)
    stop = cl.enqueue_marker(queue)
    stop.wait()

    start = cl.enqueue_marker(queue)
    for _ in range(niterations):
        knl_exec(queue, A=a_cl, B=b_cl, C=c_cl)
    stop = cl.enqueue_marker(queue)
    stop.wait()

    total_s = (stop.profile.end - start.profile.start) * 1e-9
    s_per_iter = total_s / niterations

    flop_count = 2 * m * n * k
    gflops = (flop_count / s_per_iter) * 1e-9

    print(f"M = {m}, N = {n}, K = {k}")
    print(f"Total time (s)  : {total_s}")
    print(f"Average time (s): {s_per_iter}")
    print(f"GFLOP/s         : {gflops}")


def _benchmark_kernel_with_cuda(
    knl: lp.TranslationUnit,
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    tm: int,
    tn: int,
    niterations: int = 10,
    nwarmup: int = 3,
    dtype: lp.ToLoopyTypeConvertible = np.float32,
):
    import cupy as cu

    cuda_knl = _build_cuda_kernel(knl)

    grid_dim, block_dim = _cuda_launch_dims(m, n, bm, bn, tm, tn)

    print(grid_dim)
    print(block_dim)

    a_cu = cu.random.randn(m, k, dtype=dtype)
    b_cu = cu.random.randn(k, n, dtype=dtype)
    c_cu = cu.zeros((m, n), dtype=dtype)

    args = (a_cu, b_cu, c_cu)

    stream = cu.cuda.Stream.null

    for _ in range(nwarmup):
        cuda_knl(grid_dim, block_dim, args)
    stream.synchronize()

    start = cu.cuda.Event()
    stop = cu.cuda.Event()

    start.record(stream)
    for _ in range(niterations):
        cuda_knl(grid_dim, block_dim, args)
    stop.record(stream)
    stop.synchronize()

    elapsed_ms = cu.cuda.get_elapsed_time(start, stop)

    total_s = elapsed_ms * 1e-3
    s_per_iter = total_s / niterations

    flop_count = 2 * m * n * k
    gflops = (flop_count / s_per_iter) * 1e-9

    print(f"M = {m}, N = {n}, K = {k}")
    print(f"Total time (s)  : {total_s}")
    print(f"Average time (s): {s_per_iter}")
    print(f"GFLOP/s         : {gflops}")


def benchmark_kernel(
    knl: lp.TranslationUnit,
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    tm: int,
    tn: int,
    niterations: int = 10,
    nwarmup: int = 3,
    dtype: lp.ToLoopyTypeConvertible = np.float32,
    backend: str = "cl",
):
    if backend == "cl":
        _benchmark_kernel_with_cl(knl, m, n, k, niterations, nwarmup, dtype)
    elif backend == "cuda":
        _benchmark_kernel_with_cuda(
            knl, m, n, k, bm, bn, tm, tn, niterations, nwarmup, dtype
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def schedule_kernel(
    knl: lp.TranslationUnit,
    bm: int,
    bn: int,
    bk: int,
    tm: int,
    tn: int,
) -> lp.TranslationUnit:
    # {{{ cta / workgroup output block level split

    knl = lp.split_iname(
        knl,
        "i",
        outer_iname="i_o",
        inner_iname="i_i",
        inner_length=bm,
    )

    knl = lp.split_iname(
        knl,
        "j",
        outer_iname="j_o",
        inner_iname="j_i",
        inner_length=bn,
    )

    # }}}

    # {{{ thread / workitem output block level split

    knl = lp.split_iname(
        knl,
        "i_i",
        outer_iname="i_i_o",
        inner_iname="i_i_i",
        inner_length=tm,
    )

    knl = lp.split_iname(
        knl,
        "j_i",
        outer_iname="j_i_o",
        inner_iname="j_i_i",
        inner_length=tn,
    )

    # }}}

    # {{{ k-dimension split

    knl = lp.split_iname(
        knl,
        "k",
        outer_iname="k_o",
        inner_iname="k_i",
        inner_length=bk,
    )

    # }}}

    return knl


def add_materialization_points(
    knl: lp.TranslationUnit,
    bm: int,
    bn: int,
    bk: int,
    tm: int,
    tn: int,
) -> lp.TranslationUnit:
    a_shared_reln = nisl.make_map(f"""{{
        [is, ks] -> [i_o, a_i, k_o, a_k] :
        is = i_o * {bm} + a_i and
        ks = k_o * {bk} + a_k
    }}""")

    knl = compute(
        knl,
        substitution="a",
        storage_indices=["a_i", "a_k"],
        compute_map=a_shared_reln,
        placement_inames=["j_o"],
        temporary_name="a_smem",
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id="a_shared_fetch",
    )

    b_shared_reln = nisl.make_map(f"""{{
        [ks, js] -> [k_o, b_k, j_o, b_j] :
        ks = k_o * {bk} + b_k and
        js = j_o * {bn} + b_j
    }}""")

    knl = compute(
        knl,
        substitution="b",
        storage_indices=["b_k", "b_j"],
        compute_map=b_shared_reln,
        placement_inames=["i_o"],
        temporary_name="b_smem",
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id="b_shared_fetch",
    )

    # FIXME: probably not the best way of doing this
    knl = lp.extract_subst(
        knl, "a_smem_subst", "a_smem[is, ks]", parameters=["is", "ks"]
    )
    knl = lp.extract_subst(
        knl, "b_smem_subst", "b_smem[ks, js]", parameters=["ks", "js"]
    )

    knl = lp.split_iname(
        knl,
        "a_i",
        outer_iname="a_i_o",
        inner_iname="a_i_i",
        inner_length=tm,
    )

    knl = lp.split_iname(
        knl,
        "b_j",
        outer_iname="b_j_o",
        inner_iname="b_j_i",
        inner_length=tn,
    )

    wg_i = bm // tm
    wg_j = bn // tn
    knl = lp.split_iname(
        knl, "a_k", outer_iname="a_k_o", inner_iname="a_k_i", inner_length=wg_j
    )

    knl = lp.split_iname(
        knl, "b_k", outer_iname="b_k_o", inner_iname="b_k_i", inner_length=wg_i
    )

    a_reg_reln = nisl.make_map(f"""{{
        [is, ks] -> [i_i_o, a_reg_i, k_i] :
            is = i_i_o * {tm} + a_reg_i and
            ks = k_i
    }}""")

    knl = compute(
        knl,
        substitution="a_smem_subst",
        compute_map=a_reg_reln,
        storage_indices=["a_reg_i"],
        placement_inames=["i_o", "j_o", "k_o", "j_i_o"],
        temporary_name="a_reg_tile",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        compute_insn_id="a_reg_tile_fetch",
    )

    b_reg_reln = nisl.make_map(f"""{{
        [ks, js] -> [k_i, j_i_o, b_reg_j] :
            ks = k_i and
            js = j_i_o * {tn} + b_reg_j
    }}""")

    knl = compute(
        knl,
        substitution="b_smem_subst",
        compute_map=b_reg_reln,
        storage_indices=["b_reg_j"],
        placement_inames=["i_o", "j_o", "k_o", "i_i_o"],
        temporary_name="b_reg_tile",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        compute_insn_id="b_reg_tile_fetch",
    )

    return knl


def parallelize_kernel(
    knl: lp.TranslationUnit,
    use_compute: bool = False,
) -> lp.TranslationUnit:
    knl = lp.tag_inames(
        knl,
        {
            "i_o": "g.1",
            "j_o": "g.0",
            "i_i_o": "l.1",
            "j_i_o": "l.0",
            "i_i_i": "ilp.unr",
            "j_i_i": "ilp.unr",
        },
    )

    if use_compute:
        knl = lp.tag_inames(
            knl,
            {
                "a_i_i": "l.1",
                "a_k_i": "l.0",
                "a_reg_i": "ilp.unr",
                "b_k_i": "l.1",
                "b_j_i": "l.0",
                "b_reg_j": "ilp.unr",
            },
        )

    return knl


def make_matmul_kernel(
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    bk: int,
    tm: int,
    tn: int,
    use_compute: bool,
    use_cuda: bool,
    dtype: lp.ToLoopyTypeConvertible = np.float32,
) -> lp.TranslationUnit:
    knl = lp.make_kernel(
        f"{{ [i, j, k] : 0 <= i < {m} and 0 <= j < {n} and 0 <= k < {k} }}",
        """
        a(is, ks) := A[is, ks]
        b(ks, js) := B[ks, js]

        C[i, j] = sum([k], a(i, k) * b(k, j))
        """,
        [
            lp.GlobalArg("A", shape=(m, k), dtype=dtype),
            lp.GlobalArg("B", shape=(k, n), dtype=dtype),
            lp.GlobalArg("C", shape=(m, n), dtype=dtype, is_output=True),
        ],
        silenced_warnings=["v1_scheduler_fallback"],
    )

    knl = schedule_kernel(knl, bm, bn, bk, tm, tn)

    if use_compute:
        knl = add_materialization_points(knl, bm, bn, bk, tm, tn)

    knl = parallelize_kernel(knl, use_compute=use_compute)

    if use_cuda:
        knl = knl.copy(target=lp.CudaTarget())

    return knl


def main(
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    bk: int,
    tm: int,
    tn: int,
    use_compute: bool,
    use_cuda: bool,
    print_device_code: bool,
    benchmark: bool,
    dtype: lp.ToLoopyTypeConvertible = np.float32,
):
    knl = make_matmul_kernel(
        m=m,
        n=n,
        k=k,
        bm=bm,
        bn=bn,
        bk=bk,
        tm=tm,
        tn=tn,
        use_compute=use_compute,
        use_cuda=use_cuda,
        dtype=dtype,
    )

    verify_correctness(
        knl,
        m=m,
        n=n,
        k=k,
        bm=bm,
        bn=bn,
        bk=bk,
        tm=tm,
        tn=tn,
        dtype=dtype,
        use_cuda=use_cuda,
    )

    if benchmark:
        benchmark_kernel(
            knl,
            m=m,
            n=n,
            k=k,
            bm=bm,
            bn=bn,
            tm=tm,
            tn=tn,
            backend="cuda" if use_cuda else "cl",
            dtype=dtype,
        )

    if print_device_code:
        print(lp.generate_code_v2(knl).device_code())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--m", action="store", type=int, default=4096)
    parser.add_argument("--n", action="store", type=int, default=4096)
    parser.add_argument("--k", action="store", type=int, default=4096)

    parser.add_argument("--bm", action="store", type=int, default=32)
    parser.add_argument("--bn", action="store", type=int, default=32)
    parser.add_argument("--bk", action="store", type=int, default=16)

    parser.add_argument("--tm", action="store", type=int, default=1)
    parser.add_argument("--tn", action="store", type=int, default=1)

    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--use-compute", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--print-device-code", action="store_true")

    args = parser.parse_args()

    if args.compare and args.use_compute:
        print("--compare and --use-compute set. Ignoring --use-compute.")

    if args.compare:
        print(25 * "=", "Running without compute", 25 * "=")

        main(
            m=args.m,
            n=args.n,
            k=args.k,
            bm=args.bm,
            bn=args.bn,
            bk=args.bk,
            tm=args.tm,
            tn=args.tn,
            use_compute=False,
            use_cuda=args.use_cuda,
            benchmark=True,
            print_device_code=args.print_device_code,
        )

        print(25 * "=", "Running with compute", 25 * "=")

        main(
            m=args.m,
            n=args.n,
            k=args.k,
            bm=args.bm,
            bn=args.bn,
            bk=args.bk,
            tm=args.tm,
            tn=args.tn,
            use_compute=True,
            use_cuda=args.use_cuda,
            benchmark=True,
            print_device_code=args.print_device_code,
        )

    else:
        main(
            m=args.m,
            n=args.n,
            k=args.k,
            bm=args.bm,
            bn=args.bn,
            bk=args.bk,
            tm=args.tm,
            tn=args.tn,
            use_compute=args.use_compute,
            use_cuda=args.use_cuda,
            benchmark=args.benchmark,
            print_device_code=args.print_device_code,
        )

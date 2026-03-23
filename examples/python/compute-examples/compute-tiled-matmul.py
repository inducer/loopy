import namedisl as nisl

import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from loopy.transform.compute import compute

import numpy as np
import numpy.linalg as la
import pyopencl as cl


def main(
        M: int = 128,
        N: int = 128,
        K: int = 128,
        bm: int = 32,
        bn: int = 32,
        bk: int = 16,
        run_sequentially: bool = False,
        use_precompute: bool = False,
        use_compute: bool = False,
        run_kernel: bool = False,
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
            lp.GlobalArg("a", shape=(M, K), dtype=np.float64),
            lp.GlobalArg("b", shape=(K, N), dtype=np.float64),
            lp.GlobalArg("c", shape=(M, N), dtype=np.float64,
                         is_output=True)
        ]
    )

    # FIXME: without this, there are complaints about in-bounds access guarantees
    # knl = lp.fix_parameters(knl, M=M, N=N, K=K)

    knl = lp.split_iname(knl, "i", bm, inner_iname="ii", outer_iname="io")
    knl = lp.split_iname(knl, "j", bn, inner_iname="ji", outer_iname="jo")
    knl = lp.split_iname(knl, "k", bk, inner_iname="ki", outer_iname="ko")

    # FIXME: Given the input is already tiled, we shouldn't have to supply compute bounds here.
    compute_map_a = nisl.make_map(f"""{{
        [is, ks] -> [io, ii_s, ko, ki_s] :
            is = io * {bm} + ii_s and
            ks = ko * {bk} + ki_s
    }}""")

    compute_map_b = nisl.make_map(f"""{{
        [ks, js] -> [ko, ki_s, jo, ji_s] :
            js = jo * {bn} + ji_s and
            ks = ko * {bk} + ki_s
    }}""")

    if use_compute:
        knl = compute(
            knl,
            "a_",
            compute_map=compute_map_a,
            storage_indices=["ii_s", "ki_s"],
            temporal_inames=["io", "ko", "jo"],
            temporary_address_space=lp.AddressSpace.LOCAL
        )

        knl = compute(
            knl,
            "b_",
            compute_map=compute_map_b,
            storage_indices=["ki_s", "ji_s"],
            temporal_inames=["io", "ko", "jo"],
            temporary_address_space=lp.AddressSpace.LOCAL
        )

    if use_precompute:
        knl = lp.precompute(
            knl,
            "a_",
            sweep_inames=["ii", "ki"],
        )

    if not run_sequentially:
        knl = lp.tag_inames(
            knl, {
                "io"   : "g.0", # outer block loop over block rows
                "jo"   : "g.1", # outer block loop over block cols

                "ii"   : "l.0", # inner block loop over rows
                "ji"   : "l.1", # inner block loop over cols

                "ii_s" : "l.0", # inner storage loop over a rows
                "ji_s" : "l.0", # inner storage loop over b cols
                "ki_s" : "l.1"  # inner storage loop over a cols / b rows
            }
        )

    knl = lp.add_inames_for_unused_hw_axes(knl)

    if run_kernel:
        a = np.random.randn(M, K)
        b = np.random.randn(K, N)

        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        ex = knl.executor(ctx)
        _, out = ex(queue, a=a, b=b)

        print(20*"=", "Tiled matmul report", 20*"=")
        print(f"Problem size:  M = {M:-4},  N = {N:-4},  K = {K:-4}")
        print(f"Tile size   : BM = {bm:-4}, BN = {bn:-4}, BK = {bk:-4}")
        print(f"Relative error = {la.norm((a @ b) - out) / la.norm(out)}")
        print((40 + len(" Tiled matmul report "))*"=")

    if print_device_code:
        print(lp.generate_code_v2(knl).device_code())

    if print_kernel:
        print(knl)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    _ = parser.add_argument("--precompute", action="store_true")
    _ = parser.add_argument("--compute", action="store_true")
    _ = parser.add_argument("--run-kernel", action="store_true")
    _ = parser.add_argument("--print-kernel", action="store_true")
    _ = parser.add_argument("--print-device-code", action="store_true")
    _ = parser.add_argument("--run-sequentially", action="store_true")

    _ = parser.add_argument("--m", action="store", type=int, default=128)
    _ = parser.add_argument("--n", action="store", type=int, default=128)
    _ = parser.add_argument("--k", action="store", type=int, default=128)

    _ = parser.add_argument("--bm", action="store", type=int, default=32)
    _ = parser.add_argument("--bn", action="store", type=int, default=32)
    _ = parser.add_argument("--bk", action="store", type=int, default=16)

    args = parser.parse_args()

    main(
        M=args.m,
        N=args.n,
        K=args.k,
        bm=args.bm,
        bn=args.bn,
        bk=args.bk,
        use_precompute=args.precompute,
        use_compute=args.compute,
        run_kernel=args.run_kernel,
        print_kernel=args.print_kernel,
        print_device_code=args.print_device_code,
        run_sequentially=args.run_sequentially
    )

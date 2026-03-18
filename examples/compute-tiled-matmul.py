import namedisl as nisl

import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from loopy.transform.compute import compute

import numpy as np
import numpy.linalg as la
import pyopencl as cl


def main(
        use_precompute: bool = False,
        use_compute: bool = False,
        run_kernel: bool = False
    ) -> None:

    knl = lp.make_kernel(
        "{ [i, j, k] : 0 <= i, j, k < 128 }",
        """
        a_(is, ks) := a[is, ks]
        b_(ks, js) := b[ks, js]
        out[i, j] = sum([k], a_(i, k) * b_(k, j))
        """,
        [
            lp.GlobalArg("a", shape=(128, 128), dtype=np.float64),
            lp.GlobalArg("b", shape=(128, 128), dtype=np.float64),
            lp.GlobalArg("out", shape=(128, 128), dtype=np.float64,
                         is_output=True)
        ]
    )

    bm = bn = 32
    bk = 16

    knl = lp.split_iname(knl, "i", bm, inner_iname="ii", outer_iname="io")
    knl = lp.split_iname(knl, "j", bn, inner_iname="ji", outer_iname="jo")
    knl = lp.split_iname(knl, "k", bk, inner_iname="ki", outer_iname="ko")

    compute_map_a = nisl.make_map(f"""{{
        [is, ks] -> [io, ii_s, ko, ki_s] :
            0 <= ii_s < {bm} and 0 <= ki_s < {bk} and
            is = io * {bm} + ii_s and
            ks = ko * {bk} + ki_s
    }}""")

    compute_map_b = nisl.make_map(f"""{{
        [ks, js] -> [ko, ki_s, jo, ji_s] :
            0 <= ji_s < {bn} and 0 <= ki_s < {bk} and
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

        knl = lp.tag_inames(
            knl, {
                "io" : "g.0",
                "jo" : "g.1",
                "ii" : "l.0",
                "ji" : "l.1",
            }
        )

        knl = lp.add_inames_for_unused_hw_axes(knl)

    if use_precompute:
        knl = lp.precompute(
            knl,
            "a_",
            sweep_inames=["ii", "ki"],
        )

    if run_kernel:
        a = np.random.randn(128, 128)
        b = np.random.randn(128, 128)

        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        ex = knl.executor(ctx)
        _, out = ex(queue, a=a, b=b)

        print(la.norm((a @ b) - out) / la.norm(out))

        knl = lp.generate_code_v2(knl).device_code()

    print(knl)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    _ = parser.add_argument("--precompute", action="store_true")
    _ = parser.add_argument("--compute", action="store_true")
    _ = parser.add_argument("--run-kernel", action="store_true")

    args = parser.parse_args()

    main(args.precompute, args.compute, args.run_kernel)

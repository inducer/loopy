import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from loopy.transform.compute import compute

import namedisl as nisl

import numpy as np
import numpy.linalg as la

import pyopencl as cl


# FIXME: more complicated function, or better yet define a set of functions
# with sympy and get the exact laplacian symbolically
def f(x, y, z):
    return x**2 + y**2 + z**2


def laplacian_f(x, y, z):
    return 6 * np.ones_like(x)


def main(
        use_compute: bool = False,
        print_device_code: bool = False,
        print_kernel: bool = False
    ) -> None:
    npts = 64
    pts = np.linspace(-1, 1, num=npts, endpoint=True)
    h = pts[1] - pts[0]

    x, y, z = np.meshgrid(*(pts,)*3)

    dtype = np.float32
    x = x.reshape(*(npts,)*3).astype(np.float32)
    y = y.reshape(*(npts,)*3).astype(np.float32)
    z = z.reshape(*(npts,)*3).astype(np.float32)

    f_ = f(x, y, z)
    lap_fd = np.zeros_like(f_)
    c = (np.array([-1/12, 4/3, -5/2, 4/3, -1/12]) / h**2).astype(dtype)

    m = 5
    r = m // 2

    bm = bn = m

    # FIXME: the usage on the k dimension is incorrect since we are only testing
    # tiling (i, j) planes
    knl = lp.make_kernel(
        "{ [i, j, k, l] : r <= i, j, k < npts - r and -r <= l < r + 1 }",
        """
        u_(is, js, ks) := u[is, js, ks]

        lap_u[i,j,k] = sum(
            [l],
            c[l+r] * (u_(i-l,j,k) + u_(i,j-l,k) + u[i,j,k-l])
        )
        """,
        [
            lp.GlobalArg("u", dtype=dtype, shape=(npts,npts,npts)),
            lp.GlobalArg("lap_u", dtype=dtype, shape=(npts,npts,npts),
                         is_output=True),
            lp.GlobalArg("c", dtype=dtype, shape=(m))
        ]
    )

    knl = lp.fix_parameters(knl, npts=npts, r=r)

    knl = lp.split_iname(knl, "i", bm, inner_iname="ii", outer_iname="io")
    knl = lp.split_iname(knl, "j", bn, inner_iname="ji", outer_iname="jo")

    # FIXME: need to split k dimension

    if use_compute:
        compute_map = nisl.make_map(
            f"""
            {{
                [is, js, ks] -> [io, ii_s, jo, ji_s, k] :
                is = io * {bm} + ii_s - {r} and
                js = jo * {bn} + ji_s - {r} and
                ks = k
            }}
            """
        )

        knl = compute(
            knl,
            "u_",
            compute_map=compute_map,
            storage_indices=["ii_s", "ji_s"],
            temporal_inames=["io", "jo", "k"],
            temporary_name="u_compute",
            temporary_address_space=lp.AddressSpace.LOCAL,
            temporary_dtype=np.float32
        )

    if print_device_code:
        print(lp.generate_code_v2(knl).device_code())

    if print_kernel:
        print(knl)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    ex = knl.executor(queue)
    _, lap_fd = ex(queue, u=f(x, y, z), c=c)

    lap_true = laplacian_f(x, y, z)
    sl = (slice(r, npts - r),)*3

    print(la.norm(lap_true[sl] - lap_fd[0][sl]) / la.norm(lap_true[sl]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    _ = parser.add_argument("--compute", action="store_true")
    _ = parser.add_argument("--print-device-code", action="store_true")
    _ = parser.add_argument("--print-kernel", action="store_true")

    args = parser.parse_args()

    main(
        use_compute=args.compute,
        print_device_code=args.print_device_code,
        print_kernel=args.print_kernel
    )

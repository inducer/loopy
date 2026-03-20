import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
import numpy as np
import numpy.linalg as la
import pyopencl as cl


# FIXME: more complicated function, or better yet define a set of functions
# with sympy and get the exact laplacian symbolically
def f(x, y, z):
    return x**2 + y**2 + z**2


def laplacian_f(x, y, z):
    return 6 * np.ones_like(x)


def main(use_compute: bool = False) -> None:
    knl = lp.make_kernel(
        "{ [i, j, k, l] : r <= i, j, k < npts - r and -r <= l < r + 1 }",
        """
        u_(i, j, k) := u[i, j, k]

        lap_u[i,j,k] = sum([l], c[l+2] * (u[i-l,j,k] + u[i,j-l,k] + u[i,j,k-l]))
        """
    )

    if use_compute:
        raise NotImplementedError("WIP")

    npts = 50
    pts = np.linspace(-1, 1, num=npts, endpoint=True)
    h = pts[1] - pts[0]

    x, y, z = np.meshgrid(*(pts,)*3)

    x = x.reshape(*(npts,)*3)
    y = y.reshape(*(npts,)*3)
    z = z.reshape(*(npts,)*3)

    f_ = f(x, y, z)
    lap_fd = np.zeros_like(f_)
    c = np.array([-1/12, 4/3, -5/2, 4/3, -1/12]) / h**2

    m = 5
    r = m // 2

    knl = lp.fix_parameters(knl, npts=npts, r=r)

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

    args = parser.parse_args()

    main(use_compute=args.compute)

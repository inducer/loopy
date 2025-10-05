import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pyopencl.array as cla
import pyopencl.clrandom as clrandom

import loopy as lp


def main():
    import pathlib
    fn = pathlib.Path(__file__).parent / "matmul.floopy"

    with open(fn) as inf:
        source = inf.read()

    dgemm = lp.parse_transformed_fortran(source, filename=fn)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    n = 2048
    a = cla.empty(queue, (n, n), dtype=np.float64, order="F")
    b = cla.empty(queue, (n, n), dtype=np.float64, order="F")
    c = cla.zeros(queue, (n, n), dtype=np.float64, order="F")
    clrandom.fill_rand(a)
    clrandom.fill_rand(b)

    dgemm = lp.set_options(dgemm, write_code=True)

    dgemm(queue, a=a, b=b, alpha=1, c=c)

    c_ref = (a.get() @ b.get())
    assert la.norm(c_ref - c.get())/la.norm(c_ref) < 1e-10


if __name__ == "__main__":
    main()

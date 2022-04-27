__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import sys
import numpy as np  # noqa
import numpy.linalg as la
import loopy as lp
import pyopencl as cl
import pyopencl.clrandom  # noqa

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

__all__ = [
        "pytest_generate_tests",
        "cl"  # 'cl.create_some_context'
        ]


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


def test_diff(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
         """{ [i,j]: 0<=i,j<n }""",
         """
         <> a = 1/(1+sinh(x[i] + y[j])**2)
         z[i] = sum(j, exp(a * x[j]))
         """, name="diff")

    knl = lp.fix_parameters(knl, n=50)

    from loopy.transform.diff import diff_kernel
    #FIXME Is this the correct interface. Does it make sense to take the entire
    #translation unit?
    dknl, diff_map = diff_kernel(knl["diff"], "z", "x")
    dknl = knl.with_kernel(dknl)
    dknl = lp.remove_unused_arguments(dknl)

    dknl = lp.add_inames_to_insn(dknl, "diff_i0", "writes:a_dx or writes:a")

    print(dknl)

    n = 50
    x = np.random.randn(n)
    y = np.random.randn(n)

    dx = np.random.randn(n)

    fac = 1e-1
    h1 = 1e-4
    h2 = h1 * fac

    evt, (z0,) = knl(queue, x=x, y=y)
    evt, (z1,) = knl(queue, x=(x + h1*dx), y=y)
    evt, (z2,) = knl(queue, x=(x + h2*dx), y=y)

    dknl = lp.set_options(dknl, write_code=True)
    evt, (df,) = dknl(queue, x=x, y=y)

    diff1 = (z1-z0)
    diff2 = (z2-z0)

    diff1_predicted = df.dot(h1*dx)
    diff2_predicted = df.dot(h2*dx)

    err1 = la.norm(diff1 - diff1_predicted) / la.norm(diff1)
    err2 = la.norm(diff2 - diff2_predicted) / la.norm(diff2)
    print(err1, err2)

    assert (err2 < err1 * fac * 1.1).all()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

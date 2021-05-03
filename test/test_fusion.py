__copyright__ = "Copyright (C) 2021 University of Illinois Board of Trustees"

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


import loopy as lp
import numpy as np
import pyopencl as cl

from pyopencl.tools import \
    pytest_generate_tests_for_pyopencl as pytest_generate_tests  # noqa


def test_two_kernel_fusion(ctx_factory):
    """
    A simple fusion test of two sets of instructions.
    """

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knla = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
            out[i] = i
        """
    )
    knlb = lp.make_kernel(
        "{[j]: 0<=j<10}",
        """
            out[j] = j+100
        """
    )
    knl = lp.fuse_kernels([knla, knlb], data_flow=[("out", 0, 1)])
    evt, (out,) = knl(queue)
    np.testing.assert_allclose(out.get(), np.arange(100, 110))

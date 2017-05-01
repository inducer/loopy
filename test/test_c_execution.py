from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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

import numpy as np
import loopy as lp
import pyopencl.clmath  # noqa
import pyopencl.clrandom  # noqa

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


def test_c_target():
    from loopy.target.c import CTarget

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = 2*a[i]",
            [
                lp.GlobalArg("out", np.float32, shape=lp.auto),
                lp.GlobalArg("a", np.float32, shape=lp.auto),
                "..."
                ],
            target=CTarget())

    assert np.allclose(knl(a=np.arange(16, dtype=np.float32))[1],
                2 * np.arange(16, dtype=np.float32))


def test_c_target_strides():
    from loopy.target.c import CTarget

    def __get_kernel(order='C'):
        return lp.make_kernel(
                "{ [i,j]: 0<=i,j<n }",
                "out[i, j] = 2*a[i, j]",
                [
                    lp.GlobalArg("out", np.float32, shape=('n', 'n'), order=order),
                    lp.GlobalArg("a", np.float32, shape=('n', 'n'), order=order),
                    "..."
                    ],
                target=CTarget())

    # test with C-order
    knl = __get_kernel('C')
    a_np = np.reshape(np.arange(16 * 16, dtype=np.float32), (16, -1),
                      order='C')

    assert np.allclose(knl(a=a_np)[1],
                2 * a_np)

    # test with F-order
    knl = __get_kernel('F')
    a_np = np.reshape(np.arange(16 * 16, dtype=np.float32), (16, -1),
                      order='F')

    assert np.allclose(knl(a=a_np)[1],
                2 * a_np)

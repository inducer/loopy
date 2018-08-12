from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2018 Kaushik Kulkarni"

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
import pyopencl as cl
import pyopencl.clrandom  # noqa: F401
import loopy as lp
import sys


from pyopencl.tools import (  # noqa: F401
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


def test_register_function_lookup(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from testlib import register_log2_lookup

    x = np.random.rand(10)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    prog = lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            y[i] = log2(x[i])
            """)
    prog = lp.register_function_id_to_in_knl_callable_mapper(prog,
            register_log2_lookup)

    evt, (out, ) = prog(queue, x=x)

    assert np.linalg.norm(np.log2(x)-out)/np.linalg.norm(np.log2(x)) < 1e-15


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

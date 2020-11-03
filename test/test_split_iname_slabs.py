__copyright__ = "Copyright (C) 2020 Lawrence Mitchell"

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
import pyopencl as cl  # noqa
import pyopencl.array as clarray
import pytest
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa
from pyopencl.tools import \
    pytest_generate_tests_for_pyopencl as pytest_generate_tests  # noqa


@pytest.fixture
def vanilla():
    k = lp.make_kernel("{ [i] : k <= i < n}",
                       """
                       a[i] = a[i] + 1
                       """,
                       [lp.ValueArg("k", dtype="int32"),
                        lp.ValueArg("n", dtype="int32"),
                        lp.GlobalArg("a", shape=(None, ),
                                        dtype="int32")])
    k = lp.assume(k, "k >= 0 and n >= k")
    return k


@pytest.fixture
def split(vanilla):
    k = lp.split_iname(vanilla, "i", 4, slabs=(1, 1))
    k = lp.prioritize_loops(k, "i_outer,i_inner")
    return k


@pytest.fixture(params=[(1, 4), (1, 5), (4, 8)],
                ids=lambda x: "{k=%s, n=%s}" % x)
def parameters(request):
    return dict(zip("kn", request.param))


def test_split_slabs(ctx_factory, vanilla, split, parameters):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    expect = clarray.zeros(queue, 8, dtype=np.int32)
    actual = clarray.zeros(queue, 8, dtype=np.int32)
    _, (expect, ) = vanilla(queue, a=expect, **parameters)
    _, (actual, ) = split(queue, a=actual, **parameters)
    assert np.array_equal(expect.get(), actual.get())

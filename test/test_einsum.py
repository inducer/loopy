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


import sys
import pytest
import loopy as lp
import numpy as np
import pyopencl as cl
import pyopencl.array

from pyopencl.tools import \
    pytest_generate_tests_for_pyopencl as pytest_generate_tests  # noqa


def test_make_einsum_error_handling():
    with pytest.raises(ValueError):
        lp.make_einsum("ij,j->j", ("a",))

    with pytest.raises(ValueError):
        lp.make_einsum("ij,j->jj", ("a", "b"))


@pytest.mark.parametrize("spec", [
    "ij->ij",  # identity
    "ij->ji",  # transpose
    "ii->i",   # extract diagonal
])
def test_einsum_array_manipulation(ctx_factory, spec):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 4
    a = np.random.rand(n, n)
    arg_names = ("a",)

    knl = lp.make_einsum(spec, arg_names)
    evt, (out,) = knl(queue, a=a)
    ans = np.einsum(spec, a)

    assert np.linalg.norm(out - ans) <= 1e-15


@pytest.mark.parametrize("spec", [
    "ij,j->j",
])
def test_einsum_array_matvec(ctx_factory, spec):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 4
    a = np.random.rand(n, n)
    b = np.random.rand(n)
    arg_names = ("a", "b")

    knl = lp.make_einsum(spec, arg_names)
    evt, (out,) = knl(queue, a=a, b=b)
    ans = np.einsum(spec, a, b)

    assert np.linalg.norm(out - ans) <= 1e-15


@pytest.mark.parametrize("spec", [
    "ij,ij->ij",  # A * B
    "ij,ji->ij",  # A * B.T
    "ij,kj->ik",  # inner(A, B)
])
def test_einsum_array_ops_same_dims(ctx_factory, spec):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 4
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    arg_names = ("a", "b")

    knl = lp.make_einsum(spec, arg_names)
    evt, (out,) = knl(queue, a=a, b=b)
    ans = np.einsum(spec, a, b)

    assert np.linalg.norm(out - ans) <= 1e-15


@pytest.mark.parametrize("spec", [
    "ik,kj->ij",  # A @ B
])
def test_einsum_array_ops_diff_dims(ctx_factory, spec):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 4
    m = 3
    o = 5
    a = np.random.rand(n, m)
    b = np.random.rand(m, o)
    arg_names = ("a", "b")

    knl = lp.make_einsum(spec, arg_names)
    evt, (out,) = knl(queue, a=a, b=b)
    ans = np.einsum(spec, a, b)

    assert np.linalg.norm(out - ans) <= 1e-15


@pytest.mark.parametrize("spec", [
    "im,mj,km->ijk",
])
def test_einsum_array_ops_triple_prod(ctx_factory, spec):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 3
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    c = np.random.rand(n, n)
    arg_names = ("a", "b", "c")

    knl = lp.make_einsum(spec, arg_names)
    evt, (out,) = knl(queue, a=a, b=b, c=c)
    ans = np.einsum(spec, a, b, c)

    assert np.linalg.norm(out - ans) <= 1e-15


def test_einsum_with_variable_strides(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    spec = "ijk,jl->il"
    knl = lp.make_einsum(spec, ("a", "b"),
                         default_order=lp.auto, default_offset=lp.auto)

    a_untransposed = np.random.randn(3, 5, 4)
    b = np.random.randn(4, 5)

    a = a_untransposed.transpose((0, 2, 1))
    a_dev = cl.array.to_device(queue, a_untransposed).transpose((0, 2, 1))
    assert a_dev.strides == a.strides

    _evt, (result,) = knl(queue, a=a_dev, b=b)

    ref = np.einsum(spec, a, b)

    assert np.allclose(result.get(), ref)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

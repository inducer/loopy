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


import pytest
import loopy as lp
import numpy as np
import pyopencl as cl

from pyopencl.tools import \
    pytest_generate_tests_for_pyopencl as pytest_generate_tests


@pytest.mark.parametrize("spec", [
    "ij->ij",  # identity
    "ij->ji",  # transpose
    "ii->i",   # diagonalization
])
def test_einsum_array_manipulation(ctx_factory, spec):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 4
    A = np.random.rand(n, n)
    arg_names = ("A",)

    knl = lp.make_einsum(spec, arg_names)
    evt, (out,) = knl(queue, A=A)
    ans = np.einsum(spec, A)

    assert np.linalg.norm(out - ans) <= 1e-15


@pytest.mark.parametrize("spec", [
    "ij,j->j",
])
def test_einsum_array_matvec(ctx_factory, spec):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 4
    A = np.random.rand(n, n)
    B = np.random.rand(n)
    arg_names = ("A", "B")

    knl = lp.make_einsum(spec, arg_names)
    evt, (out,) = knl(queue, A=A, B=B)
    ans = np.einsum(spec, A, B)

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
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    arg_names = ("A", "B")

    knl = lp.make_einsum(spec, arg_names)
    evt, (out,) = knl(queue, A=A, B=B)
    ans = np.einsum(spec, A, B)

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
    A = np.random.rand(n, m)
    B = np.random.rand(m, o)
    arg_names = ("A", "B")

    knl = lp.make_einsum(spec, arg_names)
    evt, (out,) = knl(queue, A=A, B=B)
    ans = np.einsum(spec, A, B)

    assert np.linalg.norm(out - ans) <= 1e-15


@pytest.mark.parametrize("spec", [
    "ia,aj,ka->ijk",  # X[i,j,k] = \sum_a A[i,a] B[a,j] C[k,a]
])
def test_einsum_array_ops_triple_prod(ctx_factory, spec):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 3
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    C = np.random.rand(n, n)
    arg_names = ("A", "B", "C")

    knl = lp.make_einsum(spec, arg_names)
    evt, (out,) = knl(queue, A=A, B=B, C=C)
    ans = np.einsum(spec, A, B, C)

    assert np.linalg.norm(out - ans) <= 1e-15


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

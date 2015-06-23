from __future__ import division, print_function

__copyright__ = "Copyright (C) 2015 James Stevens"

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
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)
import loopy as lp
from loopy.statistics import get_op_poly  # noqa
import numpy as np


def test_op_counter_basic():

    knl = lp.make_kernel(
            "[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
                c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
                e[i, k] = g[i,k]*h[i,k+1]
                """
            ],
            name="weird", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl,
                        dict(a=np.float32, b=np.float32, g=np.float64, h=np.float64))
    poly = get_op_poly(knl)
    n = 512
    m = 256
    l = 128
    f32 = poly.dict[np.dtype(np.float32)].eval_with_dict({'n': n, 'm': m, 'l': l})
    f64 = poly.dict[np.dtype(np.float64)].eval_with_dict({'n': n, 'm': m, 'l': l})
    i32 = poly.dict[np.dtype(np.int32)].eval_with_dict({'n': n, 'm': m, 'l': l})
    assert f32 == 3*n*m*l
    assert f64 == n*m
    assert i32 == n*m


def test_op_counter_reduction():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],
            name="matmul", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
    poly = get_op_poly(knl)
    n = 512
    m = 256
    l = 128
    f32 = poly.dict[np.dtype(np.float32)].eval_with_dict({'n': n, 'm': m, 'l': l})
    assert f32 == 2*n*m*l


def test_op_counter_logic():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
                e[i,k] = if(not(k<l-2) and k>6 or k/2==l, g[i,k]*2, g[i,k]+h[i,k]/2)
                """
            ],
            name="logic", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(g=np.float32, h=np.float64))
    poly = get_op_poly(knl)
    n = 512
    m = 256
    l = 128
    f32 = poly.dict[np.dtype(np.float32)].eval_with_dict({'n': n, 'm': m, 'l': l})
    f64 = poly.dict[np.dtype(np.float64)].eval_with_dict({'n': n, 'm': m, 'l': l})
    i32 = poly.dict[np.dtype(np.int32)].eval_with_dict({'n': n, 'm': m, 'l': l})
    assert f32 == n*m
    assert f64 == 3*n*m
    assert i32 == n*m


def test_op_counter_specialops():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
                c[i, j, k] = (2*a[i,j,k])%(2+b[i,j,k]/3.0)
                e[i, k] = (1+g[i,k])**(1+h[i,k+1])
                """
            ],
            name="specialops", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl,
                        dict(a=np.float32, b=np.float32, g=np.float64, h=np.float64))
    poly = get_op_poly(knl)
    n = 512
    m = 256
    l = 128
    f32 = poly.dict[np.dtype(np.float32)].eval_with_dict({'n': n, 'm': m, 'l': l})
    f64 = poly.dict[np.dtype(np.float64)].eval_with_dict({'n': n, 'm': m, 'l': l})
    i32 = poly.dict[np.dtype(np.int32)].eval_with_dict({'n': n, 'm': m, 'l': l})
    assert f32 == 4*n*m*l
    assert f64 == 3*n*m
    assert i32 == n*m


def test_op_counter_bitwise():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
                c[i, j, k] = (a[i,j,k] | 1) + (b[i,j,k] & 1)
                e[i, k] = (g[i,k] ^ k)*(~h[i,k+1]) + (g[i, k] << (h[i,k] >> k))
                """
            ],
            name="bitwise", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl,
                        dict(
                            a=np.int32, b=np.int32,
                            g=np.int64, h=np.int64))
    poly = get_op_poly(knl)
    n = 10
    m = 10
    l = 10
    i32 = poly.dict[np.dtype(np.int32)].eval_with_dict({'n': n, 'm': m, 'l': l})
    print(poly.dict)
    not_there = poly[np.dtype(np.float64)].eval_with_dict({'n': n, 'm': m, 'l': l})
    assert i32 == 3*n*m + n*m*l
    assert not_there == 0


def test_op_counter_triangular_domain():

    knl = lp.make_kernel(
            "{[i,j]: 0<=i<n and 0<=j<m and i<j}",
            """
            a[i, j] = b[i,j] * 2
            """,
            name="bitwise", assumptions="n,m >= 1")

    knl = lp.add_and_infer_dtypes(knl,
            dict(b=np.float64))

    expect_fallback = False
    import islpy as isl
    try:
        isl.BasicSet.card
    except AttributeError:
        expect_fallback = True
    else:
        expect_fallback = False

    poly = get_op_poly(knl)[np.dtype(np.float64)]
    value_dict = dict(m=13, n=200)
    flops = poly.eval_with_dict(value_dict)

    if expect_fallback:
        assert flops == 144
    else:
        assert flops == 78


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

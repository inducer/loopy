from __future__ import division

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
from pyopencl.tools import (
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)
from loopy.statistics import *


def test_op_counter_basic(ctx_factory):

    knl = lp.make_kernel(
            "[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
            """
            c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
            e[i, k] = g[i,k]*h[i,k]
            """
            ],
            name="weird", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32, g=np.float32, h=np.float32))
    poly = get_op_poly(knl)
    n=512
    m=256
    l=128
    flops = poly.eval_with_dict({'n':n, 'm':m, 'l':l})
    assert flops == n*m+3*n*m*l

def test_op_counter_reduction(ctx_factory):

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
            "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],
            name="matmul", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
    poly = get_op_poly(knl)
    n=512
    m=256
    l=128
    flops = poly.eval_with_dict({'n':n, 'm':m, 'l':l})
    assert flops == 2*n*m*l

def test_op_counter_logic(ctx_factory):

    knl = lp.make_kernel(
            "[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
            """
            e[i,k] = if(not(k < l-2) and k > l+6 or k/2 == l, g[i,k]*h[i,k], g[i,k]+h[i,k]/2.0)
            """
            ],
            name="logic", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(g=np.float32, h=np.float32))
    poly = get_op_poly(knl)
    n=512
    m=256
    l=128
    flops = poly.eval_with_dict({'n':n, 'm':m, 'l':l})
    assert flops == 5*n*m

def test_op_counter_remainder(ctx_factory):

    knl = lp.make_kernel(
            "[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
            """
            c[i, j, k] = (2*a[i,j,k])%(2+b[i,j,k]/3.0)
            """
            ],
            name="logic", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
    poly = get_op_poly(knl)
    n=512
    m=256
    l=128
    flops = poly.eval_with_dict({'n':n, 'm':m, 'l':l})
    assert flops == 4*n*m*l

def test_op_counter_power(ctx_factory):

    knl = lp.make_kernel(
            "[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
            """
            c[i, j, k] = a[i,j,k]**3.0
            e[i, k] = (1+g[i,k])**(1+h[i,k+1])
            """
            ],
            name="weird", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, g=np.float32, h=np.float32))
    poly = get_op_poly(knl)
    n=512
    m=256
    l=128
    flops = poly.eval_with_dict({'n':n, 'm':m, 'l':l})
    assert flops == 4*n*m+n*m*l

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

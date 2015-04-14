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
from pymbolic.mapper.flop_counter import FlopCounter
from loopy.statistics import *


def test_flop_counter_basic(ctx_factory):

	knl = lp.make_kernel(
			"[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
			[
			"""
			c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
			e[i, k] = g[i,k]*h[i,k]
			"""
			],
			name="weird", assumptions="n,m,l >= 1")

	poly = get_flop_poly(knl)
	n=512
	m=256
	l=128
	flops = poly.eval_with_dict({'n':n, 'm':m, 'l':l})
	assert flops == n*m+3*n*m*l


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

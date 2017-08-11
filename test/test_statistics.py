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

import six
import sys
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)
import loopy as lp
from loopy.types import to_loopy_type
import numpy as np

from pymbolic.primitives import Variable


def test_op_counter_basic():

    knl = lp.make_kernel(
            "[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
                c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
                e[i, k+1] = -g[i,k]*h[i,k+1]
                """
            ],
            name="basic", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl,
                                  dict(a=np.float32, b=np.float32,
                                       g=np.float64, h=np.float64))
    op_map = lp.get_op_map(knl, count_redundant_work=True)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32add = op_map[lp.Op(np.float32, 'add')].eval_with_dict(params)
    f32mul = op_map[lp.Op(np.float32, 'mul')].eval_with_dict(params)
    f32div = op_map[lp.Op(np.float32, 'div')].eval_with_dict(params)
    f64mul = op_map[lp.Op(np.dtype(np.float64), 'mul')].eval_with_dict(params)
    i32add = op_map[lp.Op(np.dtype(np.int32), 'add')].eval_with_dict(params)
    assert f32add == f32mul == f32div == n*m*l
    assert f64mul == n*m
    assert i32add == n*m*2


def test_op_counter_reduction():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],
            name="matmul_serial", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
    op_map = lp.get_op_map(knl, count_redundant_work=True)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32add = op_map[lp.Op(np.float32, 'add')].eval_with_dict(params)
    f32mul = op_map[lp.Op(np.dtype(np.float32), 'mul')].eval_with_dict(params)
    assert f32add == f32mul == n*m*l

    op_map_dtype = op_map.group_by('dtype')
    f32 = op_map_dtype[lp.Op(dtype=np.float32)].eval_with_dict(params)
    assert f32 == f32add + f32mul


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
    op_map = lp.get_op_map(knl, count_redundant_work=True)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32mul = op_map[lp.Op(np.float32, 'mul')].eval_with_dict(params)
    f64add = op_map[lp.Op(np.float64, 'add')].eval_with_dict(params)
    f64div = op_map[lp.Op(np.dtype(np.float64), 'div')].eval_with_dict(params)
    i32add = op_map[lp.Op(np.dtype(np.int32), 'add')].eval_with_dict(params)
    assert f32mul == n*m
    assert f64div == 2*n*m  # TODO why?
    assert f64add == n*m
    assert i32add == n*m


def test_op_counter_specialops():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
                c[i, j, k] = (2*a[i,j,k])%(2+b[i,j,k]/3.0)
                e[i, k] = (1+g[i,k])**(1+h[i,k+1])+rsqrt(g[i,k])*sin(g[i,k])
                """
            ],
            name="specialops", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl,
                                  dict(a=np.float32, b=np.float32,
                                       g=np.float64, h=np.float64))
    op_map = lp.get_op_map(knl, count_redundant_work=True)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32mul = op_map[lp.Op(np.float32, 'mul')].eval_with_dict(params)
    f32div = op_map[lp.Op(np.float32, 'div')].eval_with_dict(params)
    f32add = op_map[lp.Op(np.float32, 'add')].eval_with_dict(params)
    f64pow = op_map[lp.Op(np.float64, 'pow')].eval_with_dict(params)
    f64add = op_map[lp.Op(np.dtype(np.float64), 'add')].eval_with_dict(params)
    i32add = op_map[lp.Op(np.dtype(np.int32), 'add')].eval_with_dict(params)
    f64rsq = op_map[lp.Op(np.dtype(np.float64), 'func:rsqrt')].eval_with_dict(params)
    f64sin = op_map[lp.Op(np.dtype(np.float64), 'func:sin')].eval_with_dict(params)
    assert f32div == 2*n*m*l
    assert f32mul == f32add == n*m*l
    assert f64add == 3*n*m
    assert f64pow == i32add == f64rsq == f64sin == n*m


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

    knl = lp.add_and_infer_dtypes(
            knl, dict(
                a=np.int32, b=np.int32,
                g=np.int64, h=np.int64))

    op_map = lp.get_op_map(knl, count_redundant_work=True)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    i32add = op_map[lp.Op(np.int32, 'add')].eval_with_dict(params)
    i32bw = op_map[lp.Op(np.int32, 'bw')].eval_with_dict(params)
    i64bw = op_map[lp.Op(np.dtype(np.int64), 'bw')].eval_with_dict(params)
    i64mul = op_map[lp.Op(np.dtype(np.int64), 'mul')].eval_with_dict(params)
    i64add = op_map[lp.Op(np.dtype(np.int64), 'add')].eval_with_dict(params)
    i64shift = op_map[lp.Op(np.dtype(np.int64), 'shift')].eval_with_dict(params)
    assert i32add == n*m+n*m*l
    assert i32bw == 2*n*m*l
    assert i64bw == 2*n*m
    assert i64add == i64mul == n*m
    assert i64shift == 2*n*m


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

    op_map = lp.get_op_map(knl, count_redundant_work=True)[lp.Op(np.float64, 'mul')]
    value_dict = dict(m=13, n=200)
    flops = op_map.eval_with_dict(value_dict)

    if expect_fallback:
        assert flops == 144
    else:
        assert flops == 78


def test_mem_access_counter_basic():

    knl = lp.make_kernel(
            "[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
                c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
                e[i, k] = g[i,k]*h[i,k+1]
                """
            ],
            name="basic", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl,
                        dict(a=np.float32, b=np.float32, g=np.float64, h=np.float64))
    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32l = mem_map[lp.MemAccess('global', np.float32,
                         stride=0, direction='load', variable='a')
                   ].eval_with_dict(params)
    f32l += mem_map[lp.MemAccess('global', np.float32,
                          stride=0, direction='load', variable='b')
                    ].eval_with_dict(params)
    f64l = mem_map[lp.MemAccess('global', np.float64,
                         stride=0, direction='load', variable='g')
                   ].eval_with_dict(params)
    f64l += mem_map[lp.MemAccess('global', np.float64,
                          stride=0, direction='load', variable='h')
                    ].eval_with_dict(params)
    assert f32l == 3*n*m*l
    assert f64l == 2*n*m

    f32s = mem_map[lp.MemAccess('global', np.dtype(np.float32),
                         stride=0, direction='store', variable='c')
                   ].eval_with_dict(params)
    f64s = mem_map[lp.MemAccess('global', np.dtype(np.float64),
                         stride=0, direction='store', variable='e')
                   ].eval_with_dict(params)
    assert f32s == n*m*l
    assert f64s == n*m


def test_mem_access_counter_reduction():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],
            name="matmul", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32l = mem_map[lp.MemAccess('global', np.float32,
                         stride=0, direction='load', variable='a')
                   ].eval_with_dict(params)
    f32l += mem_map[lp.MemAccess('global', np.float32,
                          stride=0, direction='load', variable='b')
                    ].eval_with_dict(params)
    assert f32l == 2*n*m*l

    f32s = mem_map[lp.MemAccess('global', np.dtype(np.float32),
                         stride=0, direction='store', variable='c')
                   ].eval_with_dict(params)
    assert f32s == n*l

    ld_bytes = mem_map.filter_by(mtype=['global'], direction=['load']
                                 ).to_bytes().eval_and_sum(params)
    st_bytes = mem_map.filter_by(mtype=['global'], direction=['store']
                                 ).to_bytes().eval_and_sum(params)
    assert ld_bytes == 4*f32l
    assert st_bytes == 4*f32s


def test_mem_access_counter_logic():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
                e[i,k] = if(not(k<l-2) and k>6 or k/2==l, g[i,k]*2, g[i,k]+h[i,k]/2)
                """
            ],
            name="logic", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(g=np.float32, h=np.float64))
    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}

    reduced_map = mem_map.group_by('mtype', 'dtype', 'direction')

    f32_g_l = reduced_map[lp.MemAccess('global', to_loopy_type(np.float32),
                                       direction='load')
                          ].eval_with_dict(params)
    f64_g_l = reduced_map[lp.MemAccess('global', to_loopy_type(np.float64),
                                       direction='load')
                          ].eval_with_dict(params)
    f64_g_s = reduced_map[lp.MemAccess('global', to_loopy_type(np.float64),
                                       direction='store')
                          ].eval_with_dict(params)
    assert f32_g_l == 2*n*m
    assert f64_g_l == n*m
    assert f64_g_s == n*m


def test_mem_access_counter_specialops():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
                c[i, j, k] = (2*a[i,j,k])%(2+b[i,j,k]/3.0)
                e[i, k] = (1+g[i,k])**(1+h[i,k+1])
                """
            ],
            name="specialops", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32,
                                            g=np.float64, h=np.float64))
    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32 = mem_map[lp.MemAccess('global', np.float32,
                         stride=0, direction='load', variable='a')
                  ].eval_with_dict(params)
    f32 += mem_map[lp.MemAccess('global', np.float32,
                          stride=0, direction='load', variable='b')
                   ].eval_with_dict(params)
    f64 = mem_map[lp.MemAccess('global', np.dtype(np.float64),
                         stride=0, direction='load', variable='g')
                  ].eval_with_dict(params)
    f64 += mem_map[lp.MemAccess('global', np.dtype(np.float64),
                          stride=0, direction='load', variable='h')
                   ].eval_with_dict(params)
    assert f32 == 2*n*m*l
    assert f64 == 2*n*m

    f32 = mem_map[lp.MemAccess('global', np.float32,
                         stride=0, direction='store', variable='c')
                  ].eval_with_dict(params)
    f64 = mem_map[lp.MemAccess('global', np.float64,
                         stride=0, direction='store', variable='e')
                  ].eval_with_dict(params)
    assert f32 == n*m*l
    assert f64 == n*m

    filtered_map = mem_map.filter_by(direction=['load'], variable=['a', 'g'])
    #tot = lp.eval_and_sum_polys(filtered_map, params)
    tot = filtered_map.eval_and_sum(params)
    assert tot == n*m*l + n*m


def test_mem_access_counter_bitwise():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
                c[i, j, k] = (a[i,j,k] | 1) + (b[i,j,k] & 1)
                e[i, k] = (g[i,k] ^ k)*(~h[i,k+1]) + (g[i, k] << (h[i,k] >> k))
                """
            ],
            name="bitwise", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(
            knl, dict(
                a=np.int32, b=np.int32,
                g=np.int32, h=np.int32))

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    i32 = mem_map[lp.MemAccess('global', np.int32,
                         stride=0, direction='load', variable='a')
                  ].eval_with_dict(params)
    i32 += mem_map[lp.MemAccess('global', np.int32,
                          stride=0, direction='load', variable='b')
                   ].eval_with_dict(params)
    i32 += mem_map[lp.MemAccess('global', np.int32,
                          stride=0, direction='load', variable='g')
                   ].eval_with_dict(params)
    i32 += mem_map[lp.MemAccess('global', np.dtype(np.int32),
                          stride=0, direction='load', variable='h')
                   ].eval_with_dict(params)
    assert i32 == 4*n*m+2*n*m*l

    i32 = mem_map[lp.MemAccess('global', np.int32,
                         stride=0, direction='store', variable='c')
                  ].eval_with_dict(params)
    i32 += mem_map[lp.MemAccess('global', np.int32,
                          stride=0, direction='store', variable='e')
                   ].eval_with_dict(params)
    assert i32 == n*m+n*m*l


def test_mem_access_counter_mixed():

    knl = lp.make_kernel(
            "[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
            c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]+x[i,k]
            e[i, k] = g[i,k]*(2+h[i,k])
            """
            ],
            name="mixed", assumptions="n,m,l >= 1")
    knl = lp.add_and_infer_dtypes(knl, dict(
                a=np.float32, b=np.float32, g=np.float64, h=np.float64,
                x=np.float32))
    bsize = 16
    knl = lp.split_iname(knl, "j", bsize)
    knl = lp.tag_inames(knl, {"j_inner": "l.0", "j_outer": "g.0"})

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True)  # noqa
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f64uniform = mem_map[lp.MemAccess('global', np.float64,
                                stride=0, direction='load', variable='g')
                         ].eval_with_dict(params)
    f64uniform += mem_map[lp.MemAccess('global', np.float64,
                                 stride=0, direction='load', variable='h')
                          ].eval_with_dict(params)
    f32uniform = mem_map[lp.MemAccess('global', np.float32,
                                stride=0, direction='load', variable='x')
                         ].eval_with_dict(params)
    f32nonconsec = mem_map[lp.MemAccess('global', np.dtype(np.float32),
                                  stride=Variable('m'), direction='load',
                                  variable='a')
                           ].eval_with_dict(params)
    f32nonconsec += mem_map[lp.MemAccess('global', np.dtype(np.float32),
                                   stride=Variable('m'), direction='load',
                                   variable='b')
                            ].eval_with_dict(params)
    assert f64uniform == 2*n*m*l/bsize
    assert f32uniform == n*m*l/bsize
    assert f32nonconsec == 3*n*m*l

    f64uniform = mem_map[lp.MemAccess('global', np.float64,
                                stride=0, direction='store', variable='e')
                         ].eval_with_dict(params)
    f32nonconsec = mem_map[lp.MemAccess('global', np.float32,
                                  stride=Variable('m'), direction='store',
                                  variable='c')
                           ].eval_with_dict(params)
    assert f64uniform == n*m*l/bsize
    assert f32nonconsec == n*m*l


def test_mem_access_counter_nonconsec():

    knl = lp.make_kernel(
            "[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
            c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
            e[i, k] = g[i,k]*(2+h[i,k])
            """
            ],
            name="nonconsec", assumptions="n,m,l >= 1")
    knl = lp.add_and_infer_dtypes(knl, dict(
                a=np.float32, b=np.float32, g=np.float64, h=np.float64))
    knl = lp.split_iname(knl, "i", 16)
    knl = lp.tag_inames(knl, {"i_inner": "l.0", "i_outer": "g.0"})

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True)  # noqa
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f64nonconsec = mem_map[lp.MemAccess('global', np.float64,
                                  stride=Variable('m'), direction='load',
                                  variable='g')
                           ].eval_with_dict(params)
    f64nonconsec += mem_map[lp.MemAccess('global', np.float64,
                                   stride=Variable('m'), direction='load',
                                   variable='h')
                            ].eval_with_dict(params)
    f32nonconsec = mem_map[lp.MemAccess('global', np.dtype(np.float32),
                                  stride=Variable('m')*Variable('l'),
                                  direction='load', variable='a')
                           ].eval_with_dict(params)
    f32nonconsec += mem_map[lp.MemAccess('global', np.dtype(np.float32),
                                   stride=Variable('m')*Variable('l'),
                                   direction='load', variable='b')
                            ].eval_with_dict(params)
    assert f64nonconsec == 2*n*m
    assert f32nonconsec == 3*n*m*l

    f64nonconsec = mem_map[lp.MemAccess('global', np.float64,
                                  stride=Variable('m'), direction='store',
                                  variable='e')
                           ].eval_with_dict(params)
    f32nonconsec = mem_map[lp.MemAccess('global', np.float32,
                                  stride=Variable('m')*Variable('l'),
                                  direction='store', variable='c')
                           ].eval_with_dict(params)
    assert f64nonconsec == n*m
    assert f32nonconsec == n*m*l


def test_mem_access_counter_consec():

    knl = lp.make_kernel(
            "[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
            c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
            e[i, k] = g[i,k]*(2+h[i,k])
            """
            ],
            name="consec", assumptions="n,m,l >= 1")
    knl = lp.add_and_infer_dtypes(knl, dict(
                a=np.float32, b=np.float32, g=np.float64, h=np.float64))
    knl = lp.tag_inames(knl, {"k": "l.0", "i": "g.0", "j": "g.1"})

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}

    f64consec = mem_map[lp.MemAccess('global', np.float64,
                        stride=1, direction='load', variable='g')
                        ].eval_with_dict(params)
    f64consec += mem_map[lp.MemAccess('global', np.float64,
                        stride=1, direction='load', variable='h')
                         ].eval_with_dict(params)
    f32consec = mem_map[lp.MemAccess('global', np.float32,
                        stride=1, direction='load', variable='a')
                        ].eval_with_dict(params)
    f32consec += mem_map[lp.MemAccess('global', np.dtype(np.float32),
                        stride=1, direction='load', variable='b')
                         ].eval_with_dict(params)
    assert f64consec == 2*n*m*l
    assert f32consec == 3*n*m*l

    f64consec = mem_map[lp.MemAccess('global', np.float64,
                        stride=1, direction='store', variable='e')
                        ].eval_with_dict(params)
    f32consec = mem_map[lp.MemAccess('global', np.float32,
                        stride=1, direction='store', variable='c')
                        ].eval_with_dict(params)
    assert f64consec == n*m*l
    assert f32consec == n*m*l


def test_barrier_counter_nobarriers():

    knl = lp.make_kernel(
            "[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
                c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
                e[i, k] = g[i,k]*h[i,k+1]
                """
            ],
            name="basic", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32,
                                            g=np.float64, h=np.float64))
    sync_map = lp.get_synchronization_map(knl)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    assert len(sync_map) == 1
    assert sync_map["kernel_launch"].eval_with_dict(params) == 1


def test_barrier_counter_barriers():

    knl = lp.make_kernel(
            "[n,m,l] -> {[i,k,j]: 0<=i<50 and 1<=k<98 and 0<=j<10}",
            [
                """
            c[i,j,k] = 2*a[i,j,k] {id=first}
            e[i,j,k] = c[i,j,k+1]+c[i,j,k-1] {dep=first}
            """
            ], [
                lp.TemporaryVariable("c", lp.auto, shape=(50, 10, 99)),
                "..."
            ],
            name="weird2",
            )
    knl = lp.add_and_infer_dtypes(knl, dict(a=np.int32))
    knl = lp.split_iname(knl, "k", 128, inner_tag="l.0")
    sync_map = lp.get_synchronization_map(knl)
    print(sync_map)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    barrier_count = sync_map["barrier_local"].eval_with_dict(params)
    assert barrier_count == 50*10*2


def test_all_counters_parallel_matmul():

    bsize = 16
    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],
            name="matmul", assumptions="n,m,l >= 1")
    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
    knl = lp.split_iname(knl, "i", bsize, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_iname(knl, "j", bsize, outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_iname(knl, "k", bsize)
    knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"])
    knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner"])

    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}

    sync_map = lp.get_synchronization_map(knl)
    assert len(sync_map) == 2
    assert sync_map["kernel_launch"].eval_with_dict(params) == 1
    assert sync_map["barrier_local"].eval_with_dict(params) == 2*m/bsize

    op_map = lp.get_op_map(knl, count_redundant_work=True)
    f32mul = op_map[
                        lp.Op(np.float32, 'mul')
                        ].eval_with_dict(params)
    f32add = op_map[
                        lp.Op(np.float32, 'add')
                        ].eval_with_dict(params)
    i32ops = op_map[
                        lp.Op(np.int32, 'add')
                        ].eval_with_dict(params)
    i32ops += op_map[
                        lp.Op(np.dtype(np.int32), 'mul')
                        ].eval_with_dict(params)

    assert f32mul+f32add == n*m*l*2

    op_map = lp.get_mem_access_map(knl, count_redundant_work=True)

    f32s1lb = op_map[lp.MemAccess('global', np.float32,
                     stride=1, direction='load', variable='b')
                     ].eval_with_dict(params)
    f32s1la = op_map[lp.MemAccess('global', np.float32,
                     stride=1, direction='load', variable='a')
                     ].eval_with_dict(params)

    assert f32s1lb == n*m*l/bsize
    assert f32s1la == n*m*l/bsize

    f32coal = op_map[lp.MemAccess('global', np.float32,
                     stride=1, direction='store', variable='c')
                     ].eval_with_dict(params)

    assert f32coal == n*l

    local_mem_map = lp.get_mem_access_map(knl,
                        count_redundant_work=True).filter_by(mtype=['local'])
    local_mem_l = local_mem_map[lp.MemAccess('local', np.dtype(np.float32),
                                             direction='load')
                                ].eval_with_dict(params)
    assert local_mem_l == n*m*l*2


def test_gather_access_footprint():
    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i,j,k<n}",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j]) + a[i,j]"
            ],
            name="matmul", assumptions="n >= 1")
    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))

    from loopy.statistics import gather_access_footprints, count
    fp = gather_access_footprints(knl)

    for key, footprint in six.iteritems(fp):
        print(key, count(knl, footprint))


def test_gather_access_footprint_2():
    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            "c[2*i] = a[i]",
            name="matmul", assumptions="n >= 1")
    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32))

    from loopy.statistics import gather_access_footprints, count
    fp = gather_access_footprints(knl)

    params = {"n": 200}
    for key, footprint in six.iteritems(fp):
        assert count(knl, footprint).eval_with_dict(params) == 200
        print(key, count(knl, footprint))


def test_summations_and_filters():

    knl = lp.make_kernel(
            "[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
                c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
                e[i, k+1] = -g[i,k]*h[i,k+1]
                """
            ],
            name="basic", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl,
                        dict(a=np.float32, b=np.float32, g=np.float64, h=np.float64))
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True)

    loads_a = mem_map.filter_by(direction=['load'], variable=['a']
                                ).eval_and_sum(params)
    assert loads_a == 2*n*m*l

    global_stores = mem_map.filter_by(mtype=['global'], direction=['store']
                                      ).eval_and_sum(params)
    assert global_stores == n*m*l + n*m

    ld_bytes = mem_map.filter_by(mtype=['global'], direction=['load']
                                 ).to_bytes().eval_and_sum(params)
    st_bytes = mem_map.filter_by(mtype=['global'], direction=['store']
                                 ).to_bytes().eval_and_sum(params)
    assert ld_bytes == 4*n*m*l*3 + 8*n*m*2
    assert st_bytes == 4*n*m*l + 8*n*m

    # ignore stride and variable names in this map
    reduced_map = mem_map.group_by('mtype', 'dtype', 'direction')
    f32lall = reduced_map[lp.MemAccess('global', np.float32, direction='load')
                          ].eval_with_dict(params)
    f64lall = reduced_map[lp.MemAccess('global', np.float64, direction='load')
                          ].eval_with_dict(params)
    assert f32lall == 3*n*m*l
    assert f64lall == 2*n*m

    op_map = lp.get_op_map(knl, count_redundant_work=True)
    #for k, v in op_map.items():
    #    print(type(k), "\n", k.name, k.dtype, type(k.dtype), " :\n", v)

    op_map_dtype = op_map.group_by('dtype')
    f32 = op_map_dtype[lp.Op(dtype=np.float32)].eval_with_dict(params)
    f64 = op_map_dtype[lp.Op(dtype=np.float64)].eval_with_dict(params)
    i32 = op_map_dtype[lp.Op(dtype=np.int32)].eval_with_dict(params)
    assert f32 == n*m*l*3
    assert f64 == n*m
    assert i32 == n*m*2

    addsub_all = op_map.filter_by(name=['add', 'sub']).eval_and_sum(params)
    f32ops_all = op_map.filter_by(dtype=[np.float32]).eval_and_sum(params)
    assert addsub_all == n*m*l + n*m*2
    assert f32ops_all == n*m*l*3

    non_field = op_map.filter_by(xxx=[np.float32]).eval_and_sum(params)
    assert non_field == 0

    ops_nodtype = op_map.group_by('name')
    ops_noname = op_map.group_by('dtype')
    mul_all = ops_nodtype[lp.Op(name='mul')].eval_with_dict(params)
    f64ops_all = ops_noname[lp.Op(dtype=np.float64)].eval_with_dict(params)
    assert mul_all == n*m*l + n*m
    assert f64ops_all == n*m

    def func_filter(key):
        return key.stride < 1 and key.dtype == to_loopy_type(np.float64) and \
               key.direction == 'load'
    s1f64l = mem_map.filter_by_func(func_filter).eval_and_sum(params)
    assert s1f64l == 2*n*m


def test_strided_footprint():
    param_dict = dict(n=2**20)
    knl = lp.make_kernel(
        "[n] -> {[i]: 0<=i<n}",
        [
            "z[i] = x[3*i]"
        ], name="s3")

    knl = lp.add_and_infer_dtypes(knl, dict(x=np.float32))

    unr = 4
    bx = 256

    knl = lp.split_iname(knl, "i", bx*unr, outer_tag="g.0", slabs=(0, 1))
    knl = lp.split_iname(knl, "i_inner", bx, outer_tag="unr", inner_tag="l.0")

    footprints = lp.gather_access_footprints(knl)
    x_l_foot = footprints[('x', 'read')]

    from loopy.statistics import count
    num = count(knl, x_l_foot).eval_with_dict(param_dict)
    denom = count(knl, x_l_foot.remove_divs()).eval_with_dict(param_dict)

    assert 2*num < denom


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

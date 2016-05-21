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
import numpy as np


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
                        dict(a=np.float32, b=np.float32, g=np.float64, h=np.float64))
    poly = lp.get_op_poly(knl)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32add = poly[(np.dtype(np.float32), 'add')].eval_with_dict(params)
    f32mul = poly[(np.dtype(np.float32), 'mul')].eval_with_dict(params)
    f32div = poly[(np.dtype(np.float32), 'div')].eval_with_dict(params)
    f64mul = poly[(np.dtype(np.float64), 'mul')].eval_with_dict(params)
    i32add = poly[(np.dtype(np.int32), 'add')].eval_with_dict(params)
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
    poly = lp.get_op_poly(knl)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32add = poly[(np.dtype(np.float32), 'add')].eval_with_dict(params)
    f32mul = poly[(np.dtype(np.float32), 'mul')].eval_with_dict(params)
    assert f32add == f32mul == n*m*l


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
    poly = lp.get_op_poly(knl)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32mul = poly[(np.dtype(np.float32), 'mul')].eval_with_dict(params)
    f64add = poly[(np.dtype(np.float64), 'add')].eval_with_dict(params)
    f64div = poly[(np.dtype(np.float64), 'div')].eval_with_dict(params)
    i32add = poly[(np.dtype(np.int32), 'add')].eval_with_dict(params)
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
                        dict(a=np.float32, b=np.float32, g=np.float64, h=np.float64))
    poly = lp.get_op_poly(knl)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32mul = poly[(np.dtype(np.float32), 'mul')].eval_with_dict(params)
    f32div = poly[(np.dtype(np.float32), 'div')].eval_with_dict(params)
    f32add = poly[(np.dtype(np.float32), 'add')].eval_with_dict(params)
    f64pow = poly[(np.dtype(np.float64), 'pow')].eval_with_dict(params)
    f64add = poly[(np.dtype(np.float64), 'add')].eval_with_dict(params)
    i32add = poly[(np.dtype(np.int32), 'add')].eval_with_dict(params)
    f64rsqrt = poly[(np.dtype(np.float64), 'func:rsqrt')].eval_with_dict(params)
    f64sin = poly[(np.dtype(np.float64), 'func:sin')].eval_with_dict(params)
    assert f32div == 2*n*m*l
    assert f32mul == f32add == n*m*l
    assert f64add == 3*n*m
    assert f64pow == i32add == f64rsqrt == f64sin == n*m


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

    poly = lp.get_op_poly(knl)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    i32add = poly[(np.dtype(np.int32), 'add')].eval_with_dict(params)
    i32bw = poly[(np.dtype(np.int32), 'bw')].eval_with_dict(params)
    i64bw = poly[(np.dtype(np.int64), 'bw')].eval_with_dict(params)
    i64mul = poly[(np.dtype(np.int64), 'mul')].eval_with_dict(params)
    i64add = poly[(np.dtype(np.int64), 'add')].eval_with_dict(params)
    i64shift = poly[(np.dtype(np.int64), 'shift')].eval_with_dict(params)
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

    poly = lp.get_op_poly(knl)[(np.dtype(np.float64), 'mul')]
    value_dict = dict(m=13, n=200)
    flops = poly.eval_with_dict(value_dict)

    if expect_fallback:
        assert flops == 144
    else:
        assert flops == 78


def test_gmem_access_counter_basic():

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
    poly = lp.get_gmem_access_poly(knl)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32 = poly[
                    (np.dtype(np.float32), 'uniform', 'load')
                   ].eval_with_dict(params)
    f64 = poly[
                    (np.dtype(np.float64), 'uniform', 'load')
                   ].eval_with_dict(params)
    assert f32 == 3*n*m*l
    assert f64 == 2*n*m

    f32 = poly[
                    (np.dtype(np.float32), 'uniform', 'store')
                   ].eval_with_dict(params)
    f64 = poly[
                    (np.dtype(np.float64), 'uniform', 'store')
                   ].eval_with_dict(params)
    assert f32 == n*m*l
    assert f64 == n*m


def test_gmem_access_counter_reduction():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],
            name="matmul", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
    poly = lp.get_gmem_access_poly(knl)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32 = poly[
                    (np.dtype(np.float32), 'uniform', 'load')
                    ].eval_with_dict(params)
    assert f32 == 2*n*m*l

    f32 = poly[
                    (np.dtype(np.float32), 'uniform', 'store')
                    ].eval_with_dict(params)
    assert f32 == n*l


def test_gmem_access_counter_logic():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                """
                e[i,k] = if(not(k<l-2) and k>6 or k/2==l, g[i,k]*2, g[i,k]+h[i,k]/2)
                """
            ],
            name="logic", assumptions="n,m,l >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(g=np.float32, h=np.float64))
    poly = lp.get_gmem_access_poly(knl)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32 = poly[
                    (np.dtype(np.float32), 'uniform', 'load')
                    ].eval_with_dict(params)
    f64 = poly[
                    (np.dtype(np.float64), 'uniform', 'load')
                    ].eval_with_dict(params)
    assert f32 == 2*n*m
    assert f64 == n*m

    f64 = poly[
                    (np.dtype(np.float64), 'uniform', 'store')
                    ].eval_with_dict(params)
    assert f64 == n*m


def test_gmem_access_counter_specialops():

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
    poly = lp.get_gmem_access_poly(knl)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f32 = poly[
                    (np.dtype(np.float32), 'uniform', 'load')
                    ].eval_with_dict(params)
    f64 = poly[
                    (np.dtype(np.float64), 'uniform', 'load')
                    ].eval_with_dict(params)
    assert f32 == 2*n*m*l
    assert f64 == 2*n*m

    f32 = poly[
                    (np.dtype(np.float32), 'uniform', 'store')
                    ].eval_with_dict(params)
    f64 = poly[
                    (np.dtype(np.float64), 'uniform', 'store')
                    ].eval_with_dict(params)
    assert f32 == n*m*l
    assert f64 == n*m


def test_gmem_access_counter_bitwise():

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

    poly = lp.get_gmem_access_poly(knl)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    i32 = poly[
                    (np.dtype(np.int32), 'uniform', 'load')
                    ].eval_with_dict(params)
    assert i32 == 4*n*m+2*n*m*l

    i32 = poly[
                    (np.dtype(np.int32), 'uniform', 'store')
                    ].eval_with_dict(params)
    assert i32 == n*m+n*m*l


def test_gmem_access_counter_mixed():

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
    threads = 16
    knl = lp.split_iname(knl, "j", threads)
    knl = lp.tag_inames(knl, {"j_inner": "l.0", "j_outer": "g.0"})

    poly = lp.get_gmem_access_poly(knl)  # noqa
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f64uniform = poly[
                    (np.dtype(np.float64), 'uniform', 'load')
                    ].eval_with_dict(params)
    f32uniform = poly[
                    (np.dtype(np.float32), 'uniform', 'load')
                    ].eval_with_dict(params)
    f32nonconsec = poly[
                    (np.dtype(np.float32), 'nonconsecutive', 'load')
                    ].eval_with_dict(params)
    assert f64uniform == 2*n*m
    assert f32uniform == n*m*l/threads
    assert f32nonconsec == 3*n*m*l

    f64uniform = poly[
                    (np.dtype(np.float64), 'uniform', 'store')
                    ].eval_with_dict(params)
    f32nonconsec = poly[
                    (np.dtype(np.float32), 'nonconsecutive', 'store')
                    ].eval_with_dict(params)
    assert f64uniform == n*m
    assert f32nonconsec == n*m*l


def test_gmem_access_counter_nonconsec():

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

    poly = lp.get_gmem_access_poly(knl)  # noqa
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    f64nonconsec = poly[
                    (np.dtype(np.float64), 'nonconsecutive', 'load')
                    ].eval_with_dict(params)
    f32nonconsec = poly[
                    (np.dtype(np.float32), 'nonconsecutive', 'load')
                    ].eval_with_dict(params)
    assert f64nonconsec == 2*n*m
    assert f32nonconsec == 3*n*m*l

    f64nonconsec = poly[
                    (np.dtype(np.float64), 'nonconsecutive', 'store')
                    ].eval_with_dict(params)
    f32nonconsec = poly[
                    (np.dtype(np.float32), 'nonconsecutive', 'store')
                    ].eval_with_dict(params)
    assert f64nonconsec == n*m
    assert f32nonconsec == n*m*l


def test_gmem_access_counter_consec():

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

    poly = lp.get_gmem_access_poly(knl)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}

    f64consec = poly[
                    (np.dtype(np.float64), 'consecutive', 'load')
                    ].eval_with_dict(params)
    f32consec = poly[
                    (np.dtype(np.float32), 'consecutive', 'load')
                    ].eval_with_dict(params)
    assert f64consec == 2*n*m
    assert f32consec == 3*n*m*l

    f64consec = poly[
                    (np.dtype(np.float64), 'consecutive', 'store')
                    ].eval_with_dict(params)
    f32consec = poly[
                    (np.dtype(np.float32), 'consecutive', 'store')
                    ].eval_with_dict(params)
    assert f64consec == n*m
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

    knl = lp.add_and_infer_dtypes(knl,
                        dict(a=np.float32, b=np.float32, g=np.float64, h=np.float64))
    sync_poly = lp.get_synchronization_poly(knl)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    assert len(sync_poly) == 1
    assert sync_poly["kernel_launch"].eval_with_dict(params) == 1


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
    knl = lp.split_iname(knl, "k", 128, outer_tag="g.0", inner_tag="l.0")
    poly = lp.get_synchronization_poly(knl)
    print(poly)
    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}
    barrier_count = poly["barrier_local"].eval_with_dict(params)
    assert barrier_count == 50*10*2


def test_all_counters_parallel_matmul():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],
            name="matmul", assumptions="n,m,l >= 1")
    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
    knl = lp.split_iname(knl, "i", 16, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_iname(knl, "j", 16, outer_tag="g.1", inner_tag="l.0")

    n = 512
    m = 256
    l = 128
    params = {'n': n, 'm': m, 'l': l}

    sync_poly = lp.get_synchronization_poly(knl)
    assert len(sync_poly) == 1
    assert sync_poly["kernel_launch"].eval_with_dict(params) == 1

    op_map = lp.get_op_poly(knl)
    f32mul = op_map[
                        (np.dtype(np.float32), 'mul')
                        ].eval_with_dict(params)
    f32add = op_map[
                        (np.dtype(np.float32), 'add')
                        ].eval_with_dict(params)
    i32ops = op_map[
                        (np.dtype(np.int32), 'add')
                        ].eval_with_dict(params)
    i32ops += op_map[
                        (np.dtype(np.int32), 'mul')
                        ].eval_with_dict(params)

    assert f32mul+f32add == n*m*l*2
    assert i32ops == n*m*l*4 + l*n*4

    subscript_map = lp.get_gmem_access_poly(knl)
    f32uncoal = subscript_map[
                        (np.dtype(np.float32), 'nonconsecutive', 'load')
                        ].eval_with_dict(params)
    f32coal = subscript_map[
                        (np.dtype(np.float32), 'consecutive', 'load')
                        ].eval_with_dict(params)

    assert f32uncoal == n*m*l
    assert f32coal == n*m*l

    f32coal = subscript_map[
                        (np.dtype(np.float32), 'consecutive', 'store')
                        ].eval_with_dict(params)

    assert f32coal == n*l


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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

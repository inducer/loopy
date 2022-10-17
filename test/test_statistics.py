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
from loopy.types import to_loopy_type
import numpy as np
from pytools import div_ceil
from loopy.statistics import CountGranularity as CG

from pymbolic.primitives import Variable


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


SGS = 32  # Subgroup size


def test_op_counter_basic():

    knl = lp.make_kernel(
            "[n,m,ell] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                """
                c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
                e[i, k+1] = -g[i,k]*h[i,k+1]
                """
            ],
            name="basic", assumptions="n,m,ell >= 1")

    knl = lp.add_and_infer_dtypes(knl,
                                  dict(a=np.float32, b=np.float32,
                                       g=np.float64, h=np.float64))
    op_map = lp.get_op_map(knl, subgroup_size=SGS, count_redundant_work=True,
                           count_within_subscripts=True)
    n_workgroups = 1
    group_size = 1
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}
    f32add = op_map[lp.Op(np.float32, "add", CG.SUBGROUP, "basic")].eval_with_dict(
            params)
    f32mul = op_map[lp.Op(np.float32, "mul", CG.SUBGROUP, "basic")].eval_with_dict(
            params)
    f32div = op_map[lp.Op(np.float32, "div", CG.SUBGROUP, "basic")].eval_with_dict(
            params)
    f64mul = op_map[lp.Op(np.dtype(np.float64), "mul", CG.SUBGROUP, "basic")
                    ].eval_with_dict(params)
    i32add = op_map[lp.Op(np.dtype(np.int32), "add", CG.SUBGROUP, "basic")
                    ].eval_with_dict(params)
    # (count-per-sub-group)*n_subgroups
    assert f32add == f32mul == f32div == n*m*ell*n_subgroups
    assert f64mul == n*m*n_subgroups
    assert i32add == n*m*2*n_subgroups


def test_op_counter_reduction():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],
            name="matmul_serial", assumptions="n,m,ell >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
    op_map = lp.get_op_map(knl, subgroup_size=SGS, count_redundant_work=True)
    n_workgroups = 1
    group_size = 1
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}
    f32add = op_map[lp.Op(np.float32, "add", CG.SUBGROUP,
        "matmul_serial")].eval_with_dict(params)
    f32mul = op_map[lp.Op(np.dtype(np.float32), "mul", CG.SUBGROUP,
        "matmul_serial")].eval_with_dict(params)
    # (count-per-sub-group)*n_subgroups
    assert f32add == f32mul == n*m*ell*n_subgroups

    op_map_dtype = op_map.group_by("dtype")
    f32 = op_map_dtype[lp.Op(dtype=np.float32)].eval_with_dict(params)
    assert f32 == f32add + f32mul


def test_op_counter_logic():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                """
                e[i,k] = \
                        (g[i, k] * 2) \
                        if (not(k<ell-2) and k>6 or k/2==ell) else \
                        (g[i, k] + h[i, k] / 2)
                """
            ],
            name="logic", assumptions="n,m,ell >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(g=np.float32, h=np.float64))
    op_map = lp.get_op_map(knl, subgroup_size=SGS, count_redundant_work=True)
    n_workgroups = 1
    group_size = 1
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}
    f32mul = op_map[lp.Op(np.float32, "mul", CG.SUBGROUP, "logic")].eval_with_dict(
            params)
    f64add = op_map[lp.Op(np.float64, "add", CG.SUBGROUP, "logic")].eval_with_dict(
            params)
    f64div = op_map[lp.Op(np.dtype(np.float64), "div", CG.SUBGROUP, "logic")
                    ].eval_with_dict(params)
    i32add = op_map[lp.Op(np.dtype(np.int32), "add", CG.SUBGROUP, "logic")
                    ].eval_with_dict(params)
    # (count-per-sub-group)*n_subgroups
    assert f32mul == n*m*n_subgroups
    assert f64div == 2*n*m*n_subgroups  # TODO why?
    assert f64add == n*m*n_subgroups
    assert i32add == n*m*n_subgroups


def test_op_counter_special_ops():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                """
                c[i, j, k] = (2*a[i,j,k])%(2+b[i,j,k]/3.0)
                e[i, k] = (1+g[i,k])**(1+h[i,k+1])+rsqrt(g[i,k])*sin(g[i,k])
                """
            ],
            name="special_ops", assumptions="n,m,ell >= 1")

    knl = lp.add_and_infer_dtypes(knl,
                                  dict(a=np.float32, b=np.float32,
                                       g=np.float64, h=np.float64))
    op_map = lp.get_op_map(knl, subgroup_size=SGS, count_redundant_work=True,
                           count_within_subscripts=True)
    n_workgroups = 1
    group_size = 1
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}
    f32mul = op_map[lp.Op(np.float32, "mul", CG.SUBGROUP,
        "special_ops")].eval_with_dict(params)
    f32div = op_map[lp.Op(np.float32, "div", CG.SUBGROUP,
        "special_ops")].eval_with_dict(params)
    f32add = op_map[lp.Op(np.float32, "add", CG.SUBGROUP,
        "special_ops")].eval_with_dict(params)
    f64pow = op_map[lp.Op(np.float64, "pow", CG.SUBGROUP,
        "special_ops")].eval_with_dict(params)
    f64add = op_map[lp.Op(np.dtype(np.float64), "add", CG.SUBGROUP, "special_ops")
                    ].eval_with_dict(params)
    i32add = op_map[lp.Op(np.dtype(np.int32), "add", CG.SUBGROUP, "special_ops")
                    ].eval_with_dict(params)
    f64rsq = op_map[lp.Op(np.dtype(np.float64), "func:rsqrt", CG.SUBGROUP,
        "special_ops")].eval_with_dict(params)
    f64sin = op_map[lp.Op(np.dtype(np.float64), "func:sin", CG.SUBGROUP,
        "special_ops")].eval_with_dict(params)
    # (count-per-sub-group)*n_subgroups
    assert f32div == 2*n*m*ell*n_subgroups
    assert f32mul == f32add == n*m*ell*n_subgroups
    assert f64add == 3*n*m*n_subgroups
    assert f64pow == i32add == f64rsq == f64sin == n*m*n_subgroups


def test_op_counter_bitwise():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                """
                c[i, j, k] = (a[i,j,k] | 1) + (b[i,j,k] & 1)
                e[i, k] = (g[i,k] ^ k)*(~h[i,k+1]) + (g[i, k] << (h[i,k] >> k))
                """
            ],
            name="bitwise", assumptions="n,m,ell >= 1")

    knl = lp.add_and_infer_dtypes(
            knl, dict(
                a=np.int32, b=np.int32,
                g=np.int64, h=np.int64))

    op_map = lp.get_op_map(knl, subgroup_size=SGS, count_redundant_work=True,
                           count_within_subscripts=False)
    n_workgroups = 1
    group_size = 1
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}
    print(op_map)
    i32add = op_map[
            lp.Op(np.int32, "add", CG.SUBGROUP, "bitwise")
            ].eval_with_dict(params)
    i32bw = op_map[
            lp.Op(np.int32, "bw", CG.SUBGROUP, "bitwise")
            ].eval_with_dict(params)
    i64bw = op_map[
            lp.Op(np.dtype(np.int64), "bw", CG.SUBGROUP, "bitwise")
            ].eval_with_dict(params)
    i64mul = op_map[
            lp.Op(np.dtype(np.int64), "mul", CG.SUBGROUP, "bitwise")
            ].eval_with_dict(params)
    i64add = op_map[
            lp.Op(np.dtype(np.int64), "add", CG.SUBGROUP, "bitwise")
            ].eval_with_dict(params)
    i64shift = op_map[
            lp.Op(np.dtype(np.int64), "shift", CG.SUBGROUP, "bitwise")
            ].eval_with_dict(params)
    # (count-per-sub-group)*n_subgroups
    assert i32add == n*m*ell*n_subgroups
    assert i32bw == 2*n*m*ell*n_subgroups
    assert i64bw == 2*n*m*n_subgroups
    assert i64add == i64mul == n*m*n_subgroups
    assert i64shift == 2*n*m*n_subgroups


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

    op_map = lp.get_op_map(
                    knl,
                    subgroup_size=SGS,
                    count_redundant_work=True
                    )[lp.Op(np.float64, "mul", CG.SUBGROUP, "bitwise")]
    value_dict = dict(m=13, n=200)
    flops = op_map.eval_with_dict(value_dict)

    n_workgroups = 1
    group_size = 1
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group

    if expect_fallback:
        assert flops == 144*n_subgroups
    else:
        assert flops == 78*n_subgroups


def test_mem_access_counter_basic():

    knl = lp.make_kernel(
            "[n,m,ell] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                """
                c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
                e[i, k] = g[i,k]*h[i,k+1]
                """
            ],
            name="basic", assumptions="n,m,ell >= 1")

    knl = lp.add_and_infer_dtypes(knl,
                    dict(a=np.float32, b=np.float32, g=np.float64, h=np.float64))

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True,
                                    subgroup_size=SGS)

    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}

    n_workgroups = 1
    group_size = 1
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group

    f32l = mem_map[lp.MemAccess("global", np.float32,
                        lid_strides={}, gid_strides={},
                        direction="load", variable="a",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="basic")
                   ].eval_with_dict(params)
    f32l += mem_map[lp.MemAccess("global", np.float32,
                        lid_strides={}, gid_strides={},
                        direction="load", variable="b",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="basic")
                    ].eval_with_dict(params)
    f64l = mem_map[lp.MemAccess("global", np.float64,
                        lid_strides={}, gid_strides={},
                        direction="load", variable="g",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="basic")
                   ].eval_with_dict(params)
    f64l += mem_map[lp.MemAccess("global", np.float64,
                        lid_strides={}, gid_strides={},
                        direction="load", variable="h",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="basic")
                    ].eval_with_dict(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert f32l == (3*n*m*ell)*n_subgroups
    assert f64l == (2*n*m)*n_subgroups

    f32s = mem_map[lp.MemAccess("global", np.dtype(np.float32),
                        lid_strides={}, gid_strides={},
                        direction="store", variable="c",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="basic")
                   ].eval_with_dict(params)
    f64s = mem_map[lp.MemAccess("global", np.dtype(np.float64),
                        lid_strides={}, gid_strides={},
                        direction="store", variable="e",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="basic")
                   ].eval_with_dict(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert f32s == (n*m*ell)*n_subgroups
    assert f64s == (n*m)*n_subgroups


def test_mem_access_counter_reduction():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],
            name="matmul", assumptions="n,m,ell >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True,
                                    subgroup_size=SGS)
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}

    n_workgroups = 1
    group_size = 1
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group

    f32l = mem_map[lp.MemAccess("global", np.float32,
                        lid_strides={}, gid_strides={},
                        direction="load", variable="a",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="matmul")
                   ].eval_with_dict(params)
    f32l += mem_map[lp.MemAccess("global", np.float32,
                        lid_strides={}, gid_strides={},
                        direction="load", variable="b",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="matmul")
                    ].eval_with_dict(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert f32l == (2*n*m*ell)*n_subgroups

    f32s = mem_map[lp.MemAccess("global", np.dtype(np.float32),
                        lid_strides={}, gid_strides={},
                        direction="store", variable="c",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="matmul")
                   ].eval_with_dict(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert f32s == (n*ell)*n_subgroups

    ld_bytes = mem_map.filter_by(mtype=["global"], direction=["load"]
                                 ).to_bytes().eval_and_sum(params)
    st_bytes = mem_map.filter_by(mtype=["global"], direction=["store"]
                                 ).to_bytes().eval_and_sum(params)
    assert ld_bytes == 4*f32l
    assert st_bytes == 4*f32s


def test_mem_access_counter_logic():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                """
                e[i,k] = if(not(k<ell-2) and k>6 or k/2==ell,
                    g[i,k]*2,
                    g[i,k]+h[i,k]/2)
                """
            ],
            name="logic", assumptions="n,m,ell >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(g=np.float32, h=np.float64))

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True,
                                    subgroup_size=SGS)
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}

    n_workgroups = 1
    group_size = 1
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group

    reduced_map = mem_map.group_by("mtype", "dtype", "direction")

    f32_g_l = reduced_map[lp.MemAccess("global", to_loopy_type(np.float32),
                                       direction="load")
                          ].eval_with_dict(params)
    f64_g_l = reduced_map[lp.MemAccess("global", to_loopy_type(np.float64),
                                       direction="load")
                          ].eval_with_dict(params)
    f64_g_s = reduced_map[lp.MemAccess("global", to_loopy_type(np.float64),
                                       direction="store")
                          ].eval_with_dict(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert f32_g_l == (2*n*m)*n_subgroups
    assert f64_g_l == (n*m)*n_subgroups
    assert f64_g_s == (n*m)*n_subgroups


def test_mem_access_counter_special_ops():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                """
                c[i, j, k] = (2*a[i,j,k])%(2+b[i,j,k]/3.0)
                e[i, k] = (1+g[i,k])**(1+h[i,k+1])
                """
            ],
            name="special_ops", assumptions="n,m,ell >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32,
                                            g=np.float64, h=np.float64))

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True,
                                    subgroup_size=SGS)
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}

    n_workgroups = 1
    group_size = 1
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group

    f32 = mem_map[lp.MemAccess("global", np.float32,
                        lid_strides={}, gid_strides={},
                        direction="load", variable="a",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="special_ops")
                  ].eval_with_dict(params)
    f32 += mem_map[lp.MemAccess("global", np.float32,
                        lid_strides={}, gid_strides={},
                        direction="load", variable="b",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="special_ops")
                   ].eval_with_dict(params)
    f64 = mem_map[lp.MemAccess("global", np.dtype(np.float64),
                        lid_strides={}, gid_strides={},
                        direction="load", variable="g",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="special_ops")
                  ].eval_with_dict(params)
    f64 += mem_map[lp.MemAccess("global", np.dtype(np.float64),
                        lid_strides={}, gid_strides={},
                        direction="load", variable="h",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="special_ops")
                   ].eval_with_dict(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert f32 == (2*n*m*ell)*n_subgroups
    assert f64 == (2*n*m)*n_subgroups

    f32 = mem_map[lp.MemAccess("global", np.float32,
                        lid_strides={}, gid_strides={},
                        direction="store", variable="c",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="special_ops")
                  ].eval_with_dict(params)
    f64 = mem_map[lp.MemAccess("global", np.float64,
                        lid_strides={}, gid_strides={},
                        direction="store", variable="e",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="special_ops")
                  ].eval_with_dict(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert f32 == (n*m*ell)*n_subgroups
    assert f64 == (n*m)*n_subgroups

    filtered_map = mem_map.filter_by(direction=["load"], variable=["a", "g"],
                         count_granularity=CG.SUBGROUP)
    tot = filtered_map.eval_and_sum(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert tot == (n*m*ell + n*m)*n_subgroups


def test_mem_access_counter_bitwise():

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                """
                c[i, j, k] = (a[i,j,k] | 1) + (b[i,j,k] & 1)
                e[i, k] = (g[i,k] ^ k)*(~h[i,k+1]) + (g[i, k] << (h[i,k] >> k))
                """
            ],
            name="bitwise", assumptions="n,m,ell >= 1")

    knl = lp.add_and_infer_dtypes(
            knl, dict(
                a=np.int32, b=np.int32,
                g=np.int32, h=np.int32))

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True,
                                    subgroup_size=SGS)
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}

    n_workgroups = 1
    group_size = 1
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group

    i32 = mem_map[lp.MemAccess("global", np.int32,
                        lid_strides={}, gid_strides={},
                        direction="load", variable="a",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="bitwise")
                  ].eval_with_dict(params)
    i32 += mem_map[lp.MemAccess("global", np.int32,
                        lid_strides={}, gid_strides={},
                        direction="load", variable="b",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="bitwise")
                   ].eval_with_dict(params)
    i32 += mem_map[lp.MemAccess("global", np.int32,
                        lid_strides={}, gid_strides={},
                        direction="load", variable="g",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="bitwise")
                   ].eval_with_dict(params)
    i32 += mem_map[lp.MemAccess("global", np.dtype(np.int32),
                        lid_strides={}, gid_strides={},
                        direction="load", variable="h",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="bitwise")
                   ].eval_with_dict(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert i32 == (4*n*m+2*n*m*ell)*n_subgroups

    i32 = mem_map[lp.MemAccess("global", np.int32,
                        lid_strides={}, gid_strides={},
                        direction="store", variable="c",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="bitwise")
                  ].eval_with_dict(params)
    i32 += mem_map[lp.MemAccess("global", np.int32,
                        lid_strides={}, gid_strides={},
                        direction="store", variable="e",
                        count_granularity=CG.SUBGROUP,
                        kernel_name="bitwise")
                   ].eval_with_dict(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert i32 == (n*m+n*m*ell)*n_subgroups


def test_mem_access_counter_mixed():
    knl = lp.make_kernel(
            "[n,m,ell] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                """
            c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]+x[i,k]
            e[i, k] = g[i,k]*(2+h[i,k])
            """
            ],
            name="mixed", assumptions="n,m,ell >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(
                a=np.float32, b=np.float32, g=np.float64, h=np.float64,
                x=np.float32))

    group_size_0 = 65

    knl = lp.split_iname(knl, "j", group_size_0)
    knl = lp.tag_inames(knl, {"j_inner": "l.0", "j_outer": "g.0"})

    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}

    n_workgroups = div_ceil(ell, group_size_0)
    group_size = group_size_0
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True,
                                    subgroup_size=SGS)
    f64uniform = mem_map[lp.MemAccess("global", np.float64,
                                lid_strides={}, gid_strides={},
                                direction="load", variable="g",
                                count_granularity=CG.SUBGROUP,
                                kernel_name="mixed")
                         ].eval_with_dict(params)
    f64uniform += mem_map[lp.MemAccess("global", np.float64,
                                lid_strides={}, gid_strides={},
                                direction="load", variable="h",
                                count_granularity=CG.SUBGROUP,
                                kernel_name="mixed")
                          ].eval_with_dict(params)
    f32uniform = mem_map[lp.MemAccess("global", np.float32,
                                lid_strides={}, gid_strides={},
                                direction="load", variable="x",
                                count_granularity=CG.SUBGROUP,
                                kernel_name="mixed")
                         ].eval_with_dict(params)
    f32nonconsec = mem_map[lp.MemAccess("global", np.dtype(np.float32),
                                lid_strides={0: Variable("m")},
                                gid_strides={0: Variable("m")*group_size_0},
                                direction="load",
                                variable="a",
                                count_granularity=CG.WORKITEM,
                                kernel_name="mixed")
                           ].eval_with_dict(params)
    f32nonconsec += mem_map[lp.MemAccess("global", np.dtype(np.float32),
                                lid_strides={0: Variable("m")},
                                gid_strides={0: Variable("m")*group_size_0},
                                direction="load",
                                variable="b",
                                count_granularity=CG.WORKITEM,
                                kernel_name="mixed")
                            ].eval_with_dict(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert f64uniform == (2*n*m)*n_subgroups
    assert f32uniform == (m*n)*n_subgroups

    expect_fallback = False
    import islpy as isl
    try:
        isl.BasicSet.card
    except AttributeError:
        expect_fallback = True
    else:
        expect_fallback = False

    if expect_fallback:
        if ell < group_size_0:
            assert f32nonconsec == 3*n*m*ell*n_workgroups
        else:
            assert f32nonconsec == 3*n*m*n_workgroups*group_size_0
    else:
        assert f32nonconsec == 3*n*m*ell

    f64uniform = mem_map[lp.MemAccess("global", np.float64,
                                lid_strides={}, gid_strides={},
                                direction="store", variable="e",
                                count_granularity=CG.SUBGROUP,
                                kernel_name="mixed")
                         ].eval_with_dict(params)
    f32nonconsec = mem_map[lp.MemAccess("global", np.float32,
                                lid_strides={0: Variable("m")},
                                gid_strides={0: Variable("m")*group_size_0},
                                direction="store",
                                variable="c",
                                count_granularity=CG.WORKITEM,
                                kernel_name="mixed")
                           ].eval_with_dict(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert f64uniform == m*n*n_subgroups

    if expect_fallback:
        if ell < group_size_0:
            assert f32nonconsec == n*m*ell*n_workgroups
        else:
            assert f32nonconsec == n*m*n_workgroups*group_size_0
    else:
        assert f32nonconsec == n*m*ell


def test_mem_access_counter_nonconsec():

    knl = lp.make_kernel(
            "[n,m,ell] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                """
            c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
            e[i, k] = g[i,k]*(2+h[i,k])
            """
            ],
            name="non_consec", assumptions="n,m,ell >= 1")
    knl = lp.add_and_infer_dtypes(knl, dict(
                a=np.float32, b=np.float32, g=np.float64, h=np.float64))
    lsize0 = 16
    knl = lp.split_iname(knl, "i", lsize0)
    knl = lp.tag_inames(knl, {"i_inner": "l.0", "i_outer": "g.0"})

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True,
                                    subgroup_size=SGS)  # noqa
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}
    f64nonconsec = mem_map[lp.MemAccess("global", np.float64,
                                lid_strides={0: Variable("m")},
                                gid_strides={0: Variable("m")*lsize0},
                                direction="load",
                                variable="g",
                                count_granularity=CG.WORKITEM,
                                kernel_name="non_consec")
                           ].eval_with_dict(params)
    f64nonconsec += mem_map[lp.MemAccess("global", np.float64,
                                lid_strides={0: Variable("m")},
                                gid_strides={0: Variable("m")*lsize0},
                                direction="load",
                                variable="h",
                                count_granularity=CG.WORKITEM,
                                kernel_name="non_consec")
                            ].eval_with_dict(params)
    f32nonconsec = mem_map[lp.MemAccess(
                            "global", np.dtype(np.float32),
                            lid_strides={0: Variable("m")*Variable("ell")},
                            gid_strides={0: Variable("m")*Variable("ell")*lsize0},
                            direction="load", variable="a",
                            count_granularity=CG.WORKITEM,
                            kernel_name="non_consec")
                           ].eval_with_dict(params)
    f32nonconsec += mem_map[lp.MemAccess(
                            "global", np.dtype(np.float32),
                            lid_strides={0: Variable("m")*Variable("ell")},
                            gid_strides={0: Variable("m")*Variable("ell")*lsize0},
                            direction="load", variable="b",
                            count_granularity=CG.WORKITEM,
                            kernel_name="non_consec")
                            ].eval_with_dict(params)
    assert f64nonconsec == 2*n*m
    assert f32nonconsec == 3*n*m*ell

    f64nonconsec = mem_map[lp.MemAccess("global", np.float64,
                                lid_strides={0: Variable("m")},
                                gid_strides={0: Variable("m")*lsize0},
                                direction="store",
                                variable="e",
                                count_granularity=CG.WORKITEM,
                                kernel_name="non_consec")
                           ].eval_with_dict(params)
    f32nonconsec = mem_map[lp.MemAccess(
                            "global", np.float32,
                            lid_strides={0: Variable("m")*Variable("ell")},
                            gid_strides={0: Variable("m")*Variable("ell")*lsize0},
                            direction="store", variable="c",
                            count_granularity=CG.WORKITEM,
                            kernel_name="non_consec")
                           ].eval_with_dict(params)
    assert f64nonconsec == n*m
    assert f32nonconsec == n*m*ell

    mem_map64 = lp.get_mem_access_map(knl, count_redundant_work=True,
                                      subgroup_size=64)
    f64nonconsec = mem_map64[lp.MemAccess(
                    "global",
                    np.float64,
                    lid_strides={0: Variable("m")},
                    gid_strides={0: Variable("m")*lsize0},
                    direction="load", variable="g",
                    count_granularity=CG.WORKITEM,
                    kernel_name="non_consec")
                    ].eval_with_dict(params)
    f64nonconsec += mem_map64[lp.MemAccess(
                    "global",
                    np.float64,
                    lid_strides={0: Variable("m")},
                    gid_strides={0: Variable("m")*lsize0},
                    direction="load", variable="h",
                    count_granularity=CG.WORKITEM,
                    kernel_name="non_consec")
                    ].eval_with_dict(params)
    f32nonconsec = mem_map64[lp.MemAccess(
                    "global",
                    np.dtype(np.float32),
                    lid_strides={0: Variable("m")*Variable("ell")},
                    gid_strides={0: Variable("m")*Variable("ell")*lsize0},
                    direction="load",
                    variable="a",
                    count_granularity=CG.WORKITEM,
                    kernel_name="non_consec")
                    ].eval_with_dict(params)
    f32nonconsec += mem_map64[lp.MemAccess(
                    "global",
                    np.dtype(np.float32),
                    lid_strides={0: Variable("m")*Variable("ell")},
                    gid_strides={0: Variable("m")*Variable("ell")*lsize0},
                    direction="load",
                    variable="b",
                    count_granularity=CG.WORKITEM,
                    kernel_name="non_consec")
                    ].eval_with_dict(params)
    assert f64nonconsec == 2*n*m
    assert f32nonconsec == 3*n*m*ell


def test_mem_access_counter_consec():

    knl = lp.make_kernel(
            "[n,m,ell] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                """
            c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
            e[i, k] = g[i,k]*(2+h[i,k])
            """
            ],
            name="consec", assumptions="n,m,ell >= 1")
    knl = lp.add_and_infer_dtypes(knl, dict(
                a=np.float32, b=np.float32, g=np.float64, h=np.float64))
    knl = lp.tag_inames(knl, {"k": "l.0", "i": "g.0", "j": "g.1"})

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True,
                                    subgroup_size="guess")
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}

    f64consec = mem_map[lp.MemAccess(
                    "global", np.float64,
                    lid_strides={0: 1}, gid_strides={0: Variable("m")},
                    direction="load", variable="g",
                    count_granularity=CG.WORKITEM,
                    kernel_name="consec")
                    ].eval_with_dict(params)
    f64consec += mem_map[lp.MemAccess(
                    "global", np.float64,
                    lid_strides={0: 1}, gid_strides={0: Variable("m")},
                    direction="load", variable="h",
                    count_granularity=CG.WORKITEM,
                    kernel_name="consec")
                    ].eval_with_dict(params)
    f32consec = mem_map[lp.MemAccess(
                    "global", np.float32,
                    lid_strides={0: 1},
                    gid_strides={0: Variable("m")*Variable("ell"), 1: Variable("m")},
                    direction="load", variable="a",
                    count_granularity=CG.WORKITEM,
                    kernel_name="consec")
                    ].eval_with_dict(params)
    f32consec += mem_map[lp.MemAccess(
                    "global", np.dtype(np.float32),
                    lid_strides={0: 1},
                    gid_strides={0: Variable("m")*Variable("ell"), 1: Variable("m")},
                    direction="load", variable="b",
                    count_granularity=CG.WORKITEM,
                    kernel_name="consec")
                    ].eval_with_dict(params)
    assert f64consec == 2*n*m*ell
    assert f32consec == 3*n*m*ell

    f64consec = mem_map[lp.MemAccess(
                    "global", np.float64,
                    lid_strides={0: 1}, gid_strides={0: Variable("m")},
                    direction="store", variable="e",
                    count_granularity=CG.WORKITEM,
                    kernel_name="consec")
                    ].eval_with_dict(params)
    f32consec = mem_map[lp.MemAccess(
                    "global", np.float32,
                    lid_strides={0: 1},
                    gid_strides={0: Variable("m")*Variable("ell"), 1: Variable("m")},
                    direction="store", variable="c",
                    count_granularity=CG.WORKITEM,
                    kernel_name="consec")
                    ].eval_with_dict(params)
    assert f64consec == n*m*ell
    assert f32consec == n*m*ell


def test_mem_access_counter_global_temps():

    knl = lp.make_kernel(
            "[n,m,ell] -> {[i,j,k]: 0<=i<n and 0<=j<m and 0<=k<ell}",
            """
            <>a[i, j, k] = 3.1
            <>b[i, j] = 3.2
            """,
            assumptions="n,m,ell >= 1")
    knl = lp.add_and_infer_dtypes(knl, {"a,b": np.float32})

    # Change temporary b address space
    knl = lp.privatize_temporaries_with_inames(knl, "i,j", "b")
    knl = lp.set_temporary_address_space(knl, "b", "global")

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True,
                                    subgroup_size="guess")
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}

    # Count global accesses
    global_accesses = mem_map.filter_by(
        mtype=["global"]).sum().eval_with_dict(params)

    assert global_accesses == n*m


def test_count_granularity_val_checks():

    try:
        lp.MemAccess(count_granularity=CG.WORKITEM)
        lp.MemAccess(count_granularity=CG.SUBGROUP)
        lp.MemAccess(count_granularity=CG.WORKGROUP)
        lp.MemAccess(count_granularity=None)
        assert True
        lp.MemAccess(count_granularity="bushel")
        raise AssertionError()
    except ValueError:
        assert True

    try:
        lp.Op(count_granularity=CG.WORKITEM)
        lp.Op(count_granularity=CG.SUBGROUP)
        lp.Op(count_granularity=CG.WORKGROUP)
        lp.Op(count_granularity=None)
        assert True
        lp.Op(count_granularity="bushel")
        raise AssertionError()
    except ValueError:
        assert True


def test_barrier_counter_nobarriers():

    knl = lp.make_kernel(
            "[n,m,ell] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                """
                c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
                e[i, k] = g[i,k]*h[i,k+1]
                """
            ],
            name="basic", assumptions="n,m,ell >= 1")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32,
                                            g=np.float64, h=np.float64))
    sync_map = lp.get_synchronization_map(knl)
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}
    assert len(sync_map) == 1
    assert sync_map.filter_by(kind="kernel_launch").eval_and_sum(params) == 1


def test_barrier_counter_barriers():
    knl = lp.make_kernel(
            "[n,m,ell] -> {[i,k,j]: 0<=i<50 and 1<=k<98 and 0<=j<10}",
            [
                """
            c[i,j,k] = 2*a[i,j,k] {id=first}
            e[i,j,k] = c[i,j,k+1]+c[i,j,k-1] {dep=first}
            """
            ], [
                lp.TemporaryVariable("c", shape=(50, 10, 99)),
                ...
            ],
            name="weird2",
            )
    knl = lp.add_and_infer_dtypes(knl, dict(a=np.int32))
    knl = lp.split_iname(knl, "k", 128, inner_tag="l.0")
    sync_map = lp.get_synchronization_map(knl)
    print(sync_map)
    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}
    barrier_count = sync_map.filter_by(kind="barrier_local").eval_and_sum(params)
    assert barrier_count == 50*10*2


def test_barrier_count_single():
    knl = lp.make_kernel(
            "{[i]: 0<=i<128}",
            """
            <> c[i] = 15*i {id=yoink}
            c[i+1] = c[i]  {dep=yoink}
            """)

    knl = lp.tag_inames(knl, {"i": "l.0"})
    sync_map = lp.get_synchronization_map(knl)
    print(sync_map)
    barrier_count = sync_map.filter_by(kind="barrier_local").eval_and_sum()
    assert barrier_count == 1


def test_all_counters_parallel_matmul():
    bsize = 16
    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],
            name="matmul", assumptions="n,m,ell >= 1")
    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
    knl = lp.split_iname(knl, "i", bsize, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_iname(knl, "j", bsize, outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_iname(knl, "k", bsize)
    knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"], default_tag="l.auto")
    knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner"], default_tag="l.auto")

    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}
    group_size = bsize*bsize
    n_workgroups = div_ceil(n, bsize)*div_ceil(ell, bsize)
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group

    sync_map = lp.get_synchronization_map(knl)
    assert len(sync_map) == 2
    assert sync_map.filter_by(kind="kernel_launch").eval_and_sum(params) == 1
    assert sync_map.filter_by(kind="barrier_local").eval_and_sum(params) == 2*m/bsize

    op_map = lp.get_op_map(knl, subgroup_size=SGS, count_redundant_work=True)
    f32mul = op_map[
                        lp.Op(np.float32, "mul", CG.SUBGROUP, "matmul")
                        ].eval_with_dict(params)
    f32add = op_map[
                        lp.Op(np.float32, "add", CG.SUBGROUP, "matmul")
                        ].eval_with_dict(params)
    i32ops = op_map[
                        lp.Op(np.int32, "add", CG.SUBGROUP, "matmul")
                        ].eval_with_dict(params)
    i32ops += op_map[
                        lp.Op(np.dtype(np.int32), "mul", CG.SUBGROUP, "matmul")
                        ].eval_with_dict(params)

    # (count-per-sub-group)*n_subgroups
    assert f32mul+f32add == m*2*n_subgroups

    mem_access_map = lp.get_mem_access_map(knl, count_redundant_work=True,
                                           subgroup_size=SGS)

    f32s1lb = mem_access_map[lp.MemAccess("global", np.float32,
                             lid_strides={0: 1, 1: Variable("ell")},
                             gid_strides={1: bsize},
                             direction="load", variable="b",
                             count_granularity=CG.WORKITEM,
                             kernel_name="matmul")
                             ].eval_with_dict(params)
    f32s1la = mem_access_map[lp.MemAccess("global", np.float32,
                             lid_strides={0: 1, 1: Variable("m")},
                             gid_strides={0: Variable("m")*bsize},
                             direction="load",
                             variable="a", count_granularity=CG.WORKITEM,
                             kernel_name="matmul")
                             ].eval_with_dict(params)

    assert f32s1lb == n*m*ell/bsize
    assert f32s1la == n*m*ell/bsize

    f32coal = mem_access_map[lp.MemAccess("global", np.float32,
                             lid_strides={0: 1, 1: Variable("ell")},
                             gid_strides={0: Variable("ell")*bsize, 1: bsize},
                             direction="store", variable="c",
                             count_granularity=CG.WORKITEM,
                             kernel_name="matmul")
                             ].eval_with_dict(params)

    assert f32coal == n*ell

    local_mem_map = lp.get_mem_access_map(knl,
                        count_redundant_work=True,
                        subgroup_size=SGS).filter_by(mtype=["local"])

    local_mem_l = local_mem_map.filter_by(direction=["load"]
                                          ).eval_and_sum(params)
    # (count-per-sub-group)*n_subgroups
    assert local_mem_l == m*2*n_subgroups

    local_mem_l_a = local_mem_map[lp.MemAccess("local", np.dtype(np.float32),
                                               direction="load",
                                               lid_strides={1: 16},
                                               gid_strides={},
                                               variable="a_fetch",
                                               count_granularity=CG.SUBGROUP,
                                               kernel_name="matmul")
                                  ].eval_with_dict(params)
    local_mem_l_b = local_mem_map[lp.MemAccess("local", np.dtype(np.float32),
                                               direction="load",
                                               lid_strides={0: 1},
                                               gid_strides={},
                                               variable="b_fetch",
                                               count_granularity=CG.SUBGROUP,
                                               kernel_name="matmul")
                                  ].eval_with_dict(params)

    # (count-per-sub-group)*n_subgroups
    assert local_mem_l_a == local_mem_l_b == m*n_subgroups

    local_mem_s = local_mem_map.filter_by(direction=["store"]
                                          ).eval_and_sum(params)

    # (count-per-sub-group)*n_subgroups
    assert local_mem_s == m*2/bsize*n_subgroups


def test_floor_div_coefficient_collector():
    bsize = 16

    # kernel that shuffles local mem
    knl = lp.make_kernel(
        "{[i_outer,j_outer,i_inner,j_inner,r]: "
        "0<=i_outer<n/%d and 0<=j_outer<n/%d and "
        "0<=i_inner<%d and 0<=j_inner<%d and "
        "0<=r<rept}"
        % (bsize, bsize, bsize, bsize),
        [
            "for i_outer",
            "for j_outer",
            "<> loc[i_inner,j_inner] = 3.14f  {id=loc_init}",
            "loc[i_inner,(j_inner+r+4) %% %d] = loc[i_inner,(j_inner+r) %% %d]"
            "  {id=add,dep=loc_init}" % (bsize, bsize),
            "out0[i_outer*16+i_inner,j_outer*16+j_inner] = loc[i_inner,j_inner]"
            "  {id=store,dep=add}",
            "end",
            "end",
        ],
        name="local",
        lang_version=(2018, 2))

    knl = lp.add_and_infer_dtypes(knl, dict(out0=np.float32))
    knl = lp.tag_inames(knl, "i_outer:g.1,i_inner:l.1,j_outer:g.0,j_inner:l.0")

    n = 512
    rept = 64
    params = {"n": n, "rept": rept}
    group_size = bsize*bsize
    n_workgroups = div_ceil(n, bsize)*div_ceil(n, bsize)
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group

    # count local f32 accesses
    m = lp.get_mem_access_map(knl, count_redundant_work=True, subgroup_size=SGS)
    f32_local = m.filter_by(dtype=[np.float32], mtype=["local"]).eval_and_sum(params)

    # (count-per-sub-group)*n_subgroups
    assert f32_local == 2*(rept+1)*n_subgroups


def test_mem_access_tagged_variables():
    bsize = 16
    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                "c$mmresult[i, j] = sum(k, a$mmaload[i, k]*b$mmbload[k, j])"
            ],
            name="matmul", assumptions="n,m,ell >= 1")
    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
    knl = lp.split_iname(knl, "i", bsize, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_iname(knl, "j", bsize, outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_iname(knl, "k", bsize)
    # knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"], default_tag="l.auto")
    # knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner"], default_tag="l.auto")

    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}
    group_size = bsize*bsize
    n_workgroups = div_ceil(n, bsize)*div_ceil(ell, bsize)
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group

    mem_access_map = lp.get_mem_access_map(knl, count_redundant_work=True,
                                           subgroup_size=SGS)

    f32s1lb = mem_access_map[
            lp.MemAccess("global", np.float32,
                lid_strides={0: 1},
                gid_strides={1: bsize},
                direction="load", variable="b",
                variable_tags=frozenset([lp.LegacyStringInstructionTag("mmbload")]),
                count_granularity=CG.WORKITEM,
                kernel_name="matmul")
            ].eval_with_dict(params)
    f32s1la = mem_access_map[
            lp.MemAccess("global", np.float32,
                lid_strides={1: Variable("m")},
                gid_strides={0: Variable("m")*bsize},
                direction="load",
                variable="a",
                variable_tags=frozenset([lp.LegacyStringInstructionTag("mmaload")]),
                count_granularity=CG.SUBGROUP,
                kernel_name="matmul")
            ].eval_with_dict(params)

    assert f32s1lb == n*m*ell

    # uniform: (count-per-sub-group)*n_subgroups
    assert f32s1la == m*n_subgroups

    f32coal = mem_access_map[
            lp.MemAccess("global", np.float32,
                lid_strides={0: 1, 1: Variable("ell")},
                gid_strides={0: Variable("ell")*bsize, 1: bsize},
                direction="store", variable="c",
                variable_tags=frozenset([lp.LegacyStringInstructionTag("mmresult")]),
                count_granularity=CG.WORKITEM,
                kernel_name="matmul")
            ].eval_with_dict(params)

    assert f32coal == n*ell


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

    for key, footprint in fp.items():
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
    for key, footprint in fp.items():
        assert count(knl, footprint).eval_with_dict(params) == 200
        print(key, count(knl, footprint))


def test_summations_and_filters():

    knl = lp.make_kernel(
            "[n,m,ell] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                """
                c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
                e[i, k+1] = -g[i,k]*h[i,k+1]
                """
            ],
            name="basic", assumptions="n,m,ell >= 1")

    knl = lp.add_and_infer_dtypes(knl,
                    dict(a=np.float32, b=np.float32, g=np.float64, h=np.float64))

    n = 512
    m = 256
    ell = 128
    params = {"n": n, "m": m, "ell": ell}

    n_workgroups = 1
    group_size = 1
    subgroups_per_group = div_ceil(group_size, SGS)
    n_subgroups = n_workgroups*subgroups_per_group

    mem_map = lp.get_mem_access_map(knl, count_redundant_work=True,
                                    subgroup_size=SGS)

    loads_a = mem_map.filter_by(direction=["load"], variable=["a"],
                                count_granularity=[CG.SUBGROUP]
                                ).eval_and_sum(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert loads_a == (2*n*m*ell)*n_subgroups

    global_stores = mem_map.filter_by(mtype=["global"], direction=["store"],
                                      count_granularity=[CG.SUBGROUP]
                                      ).eval_and_sum(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert global_stores == (n*m*ell + n*m)*n_subgroups

    ld_bytes = mem_map.filter_by(mtype=["global"], direction=["load"],
                                 count_granularity=[CG.SUBGROUP]
                                 ).to_bytes().eval_and_sum(params)
    st_bytes = mem_map.filter_by(mtype=["global"], direction=["store"],
                                 count_granularity=[CG.SUBGROUP]
                                 ).to_bytes().eval_and_sum(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert ld_bytes == (4*n*m*ell*3 + 8*n*m*2)*n_subgroups
    assert st_bytes == (4*n*m*ell + 8*n*m)*n_subgroups

    # ignore stride and variable names in this map
    reduced_map = mem_map.group_by("mtype", "dtype", "direction")
    f32lall = reduced_map[lp.MemAccess("global", np.float32, direction="load")
                          ].eval_with_dict(params)
    f64lall = reduced_map[lp.MemAccess("global", np.float64, direction="load")
                          ].eval_with_dict(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert f32lall == (3*n*m*ell)*n_subgroups
    assert f64lall == (2*n*m)*n_subgroups

    op_map = lp.get_op_map(knl, subgroup_size=SGS, count_redundant_work=True,
                           count_within_subscripts=True)
    #for k, v in op_map.items():
    #    print(type(k), "\n", k.name, k.dtype, type(k.dtype), " :\n", v)

    op_map_dtype = op_map.group_by("dtype")
    f32 = op_map_dtype[lp.Op(dtype=np.float32)].eval_with_dict(params)
    f64 = op_map_dtype[lp.Op(dtype=np.float64)].eval_with_dict(params)
    i32 = op_map_dtype[lp.Op(dtype=np.int32)].eval_with_dict(params)
    assert f32 == n*m*ell*3
    assert f64 == n*m
    assert i32 == n*m*2

    addsub_all = op_map.filter_by(name=["add", "sub"]).eval_and_sum(params)
    f32ops_all = op_map.filter_by(dtype=[np.float32]).eval_and_sum(params)
    assert addsub_all == n*m*ell + n*m*2
    assert f32ops_all == n*m*ell*3

    non_field = op_map.filter_by(xxx=[np.float32]).eval_and_sum(params)
    assert non_field == 0

    ops_nodtype = op_map.group_by("name")
    ops_noname = op_map.group_by("dtype")
    mul_all = ops_nodtype[lp.Op(name="mul")].eval_with_dict(params)
    f64ops_all = ops_noname[lp.Op(dtype=np.float64)].eval_with_dict(params)
    assert mul_all == n*m*ell + n*m
    assert f64ops_all == n*m

    def func_filter(key):
        return key.lid_strides == {} and key.dtype == to_loopy_type(np.float64) and \
               key.direction == "load"
    f64l = mem_map.filter_by_func(func_filter).eval_and_sum(params)

    # uniform: (count-per-sub-group)*n_subgroups
    assert f64l == (2*n*m)*n_subgroups


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
    x_l_foot = footprints[("x", "read")]

    from loopy.statistics import count
    num = count(knl, x_l_foot).eval_with_dict(param_dict)
    denom = count(knl, x_l_foot.remove_divs()).eval_with_dict(param_dict)

    assert 2*num < denom


def test_stats_on_callable_kernel():
    callee = lp.make_function(
            "{[i, j]: 0<=i, j< 20}",
            """
            y[i] = sum(j, A[i,j]*x[j])
            """, name="matvec20x20")

    caller = lp.make_kernel(
            "{:}",
            """
            y[:]  = matvec20x20(A[:,:], x[:])
            """,
            [
                lp.GlobalArg("x,y", shape=(20,), dtype=np.float64),
                lp.GlobalArg("A", shape=(20, 20), dtype=np.float64),
                ],
            name="matvec")
    caller = lp.merge([caller, callee])

    op_map = lp.get_op_map(caller, subgroup_size=SGS, count_redundant_work=True,
                           count_within_subscripts=True)
    f64_add = op_map.filter_by(name="add").eval_and_sum({})
    assert f64_add == 400


def test_stats_on_callable_kernel_within_loop():
    callee = lp.make_function(
            "{[i, j]: 0<=i, j< 20}",
            """
            y[i] = sum(j, A[i,j]*x[j])
            """, name="matvec20x20")

    caller = lp.make_kernel(
            "{[i]: 0<=i< 20}",
            """
            y[i, :]  = matvec20x20(A[:,:], x[i, :])
            """,
            [
                lp.GlobalArg("x,y", shape=(20, 20), dtype=np.float64),
                lp.GlobalArg("A", shape=(20, 20), dtype=np.float64),
                ],
            name="matmat")
    caller = lp.merge([caller, callee])

    op_map = lp.get_op_map(caller, subgroup_size=SGS, count_redundant_work=True,
                           count_within_subscripts=True)

    f64_add = op_map.filter_by(name="add").eval_and_sum({})
    assert f64_add == 8000


def test_callable_kernel_with_substitution():
    callee = lp.make_function(
            "{[i, j]: 0<=i, j< n}",
            """
            y[i] = sum(j, A[i,j]*x[j])
            """,
            [lp.ValueArg("n"), ...],
            name="matvec")

    caller = lp.make_kernel(
            "{[i]: 0<=i< 20}",
            """
            y[i, :]  = matvec(20, A[:,:], x[i, :])
            """,
            [
                lp.GlobalArg("x,y", shape=(20, 20), dtype=np.float64),
                lp.GlobalArg("A", shape=(20, 20), dtype=np.float64),
                ],
            name="matmat")
    caller = lp.merge([caller, callee])

    op_map = lp.get_op_map(caller, subgroup_size=SGS, count_redundant_work=True,
                           count_within_subscripts=True)

    f64_add = op_map.filter_by(name="add").eval_and_sum({})
    assert f64_add == 8000


def test_no_loop_ops():
    # See https://github.com/inducer/loopy/issues/211

    knl = lp.make_kernel(
        "{ : }",
        """
        c = a + b
        d = 2*c + a + b
        """)

    knl = lp.add_dtypes(knl, {"a": np.float64, "b": np.float64})

    op_map = lp.get_op_map(knl, subgroup_size=SGS, count_redundant_work=True,
                           count_within_subscripts=True)
    f64_add = op_map.filter_by(name="add").eval_and_sum({})
    f64_mul = op_map.filter_by(name="mul").eval_and_sum({})
    assert f64_add == 3
    assert f64_mul == 1


def test_within_stats():
    import loopy as lp
    import numpy as np
    import pytest

    knl = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
        out1[i] = 2.0f * i  {tags=writes_float}
        out2[i] = 3 * i     {tags=writes_int}
        out3[i] = 7.0f * i  {tags=writes_float}
        """)

    op_map = lp.get_op_map(knl, subgroup_size="guess", within="tag:writes_float")

    ops_dtype = op_map.group_by("dtype")

    f32ops = ops_dtype[lp.Op(dtype=np.float32)].eval_with_dict({})

    assert f32ops == 20

    with pytest.raises(KeyError):
        _ = ops_dtype[lp.Op(dtype=np.int32)].eval_with_dict({})

    mem_map = lp.get_mem_access_map(knl, subgroup_size="guess",
                                    within="tag:writes_float")

    mem_dtype = mem_map.group_by("dtype")
    mem_ops = mem_dtype[lp.MemAccess(dtype=np.float32)].eval_with_dict({})
    assert mem_ops == 20

    with pytest.raises(KeyError):
        _ = ops_dtype[lp.MemAccess(dtype=np.int32)].eval_with_dict({})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

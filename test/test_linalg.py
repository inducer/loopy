__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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
import sys
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array  # noqa: F401
import pyopencl.cltypes as cltypes
import loopy as lp

import logging

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)


DEBUG_PREAMBLE = r"""
    #pragma OPENCL EXTENSION cl_amd_printf: enable
    #define MY_J (j_outer*64+j_inner_outer*16+j_inner_inner)
    #define MY_I (i_outer*16+i_inner)
    #define IFDIAG if (MY_I == MY_J)
    #define TST(S) if (MY_J == 144 && MY_I == 16-48) \
            for (int aa = 0; aa < 16: ++ab) \
                for (int bb = 0; bb < 16: ++bb)
    """


def get_suitable_size(ctx):
    dev, = ctx.devices
    if dev.type & cl.device_type.CPU:
        return 160
    else:
        return 1600


def check_float4(result, ref_result):
    for comp in ["x", "y", "z", "w"]:
        return np.allclose(
                ref_result[comp], result[comp], rtol=1e-3, atol=1e-3), None


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


def test_axpy(ctx_factory):
    logging.basicConfig(level="INFO")
    ctx = ctx_factory()

    n = 3145182

    if ctx.devices[0].platform.vendor.startswith("Advanced Micro"):
        pytest.skip("crashes on AMD 15.12")

    for dtype, check, a, b in [
            (np.complex64, None, 5, 7),
            (cltypes.float4, check_float4,  # pylint:disable=no-member
                cltypes.make_float4(1, 2, 3, 4),  # pylint:disable=no-member
                cltypes.make_float4(6, 7, 8, 9)),  # pylint:disable=no-member
            (np.float32, None, 5, 7),
            ]:
        knl = lp.make_kernel(
                "[n] -> {[i]: 0<=i<n}",
                [
                    "z[i] = a*x[i]+b*y[i]"
                    ],
                [
                    lp.ValueArg("a", dtype),
                    lp.GlobalArg("x", dtype, shape="n,"),
                    lp.ValueArg("b", dtype),
                    lp.GlobalArg("y", dtype, shape="n,"),
                    lp.GlobalArg("z", dtype, shape="n,"),
                    lp.ValueArg("n", np.int32, approximately=n),
                    ],
                name="axpy", assumptions="n>=1")

        seq_knl = knl

        def variant_cpu(knl):
            unroll = 16
            block_size = unroll*4096
            knl = lp.split_iname(knl, "i", block_size, outer_tag="g.0", slabs=(0, 1))
            knl = lp.split_iname(knl, "i_inner", unroll, inner_tag="unr")
            return knl

        def variant_gpu(knl):
            unroll = 4
            block_size = 256
            knl = lp.split_iname(knl, "i", unroll*block_size,
                    outer_tag="g.0", slabs=(0, 1))
            knl = lp.split_iname(knl, "i_inner", block_size,
                    outer_tag="unr", inner_tag="l.0")
            return knl

        #for variant in [ variant_gpu]:
        for variant in [variant_cpu, variant_gpu]:
            lp.auto_test_vs_ref(seq_knl, ctx, variant(knl),
                    op_count=[np.dtype(dtype).itemsize*n*3/1e9],
                    op_label=["GBytes"],
                    parameters={"a": a, "b": b, "n": n}, check_result=check,
                    blacklist_ref_vendors=["Advanced Micro"])


def test_transpose(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    order = "C"

    n = get_suitable_size(ctx)

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<%d}" % n,
            [
                "b[i, j] = a[j, i]"
                ],
            [
                lp.GlobalArg("a", dtype, shape=(n, n), order=order),
                lp.GlobalArg("b", dtype, shape=(n, n), order=order),
                ],
            name="transpose")

    seq_knl = knl

    knl = lp.split_iname(knl, "i", 16,
            outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_iname(knl, "j", 16,
            outer_tag="g.1", inner_tag="l.0")
    knl = lp.add_prefetch(knl, "a", ["i_inner", "j_inner"],
            default_tag="l.auto")

    from loopy.transform.data import add_padding_to_avoid_bank_conflicts
    knl = add_padding_to_avoid_bank_conflicts(knl, ctx.devices[0])
    lp.auto_test_vs_ref(seq_knl, ctx, knl,
            op_count=[dtype.itemsize*n**2*2/1e9], op_label=["GByte"],
            parameters={})


def test_plain_matrix_mul(ctx_factory):
    ctx = ctx_factory()
    order = "C"

    n = get_suitable_size(ctx)

    for dtype, check, vec_size in [
            (cltypes.float4, check_float4, 4),  # pylint:disable=no-member
            (np.float32, None, 1),
            ]:
        knl = lp.make_kernel(
                "{[i,j,k]: 0<=i,j,k<%d}" % n,
                [
                    "c[i, j] = sum(k, a[i, k]*b[k, j])"
                    ],
                [
                    lp.GlobalArg("a", dtype, shape=(n, n), order=order),
                    lp.GlobalArg("b", dtype, shape=(n, n), order=order),
                    lp.GlobalArg("c", dtype, shape=(n, n), order=order),
                    ],
                name="matmul")

        ref_knl = knl

        knl = lp.split_iname(knl, "i", 16,
                outer_tag="g.0", inner_tag="l.1")
        knl = lp.split_iname(knl, "j", 16,
                outer_tag="g.1", inner_tag="l.0")
        knl = lp.split_iname(knl, "k", 16)
        knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"],
                fetch_outer_inames="i_outer, j_outer, k_outer",
                default_tag="l.auto")
        knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner", ],
                fetch_outer_inames="i_outer, j_outer, k_outer",
                default_tag="l.auto")

        lp.auto_test_vs_ref(ref_knl, ctx, knl,
                op_count=[vec_size*2*n**3/1e9], op_label=["GFlops"],
                parameters={"n": n}, check_result=check)


def test_variable_size_matrix_mul(ctx_factory):
    ctx = ctx_factory()

    if (not ctx.devices[0].image_support
            or ctx.devices[0].platform.name == "Portable Computing Language"):
        pytest.skip("crashes on pocl")

    n = get_suitable_size(ctx)

    knl = lp.make_kernel(
            "{[i,j,k]: 0<=i,j,k<n}",
            "c[i, j] = sum(k, a[i, k]*b[k, j])")

    knl = lp.add_dtypes(knl, {
        "a": np.float32,
        "b": np.float32,
        })

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 16,
            outer_tag="g.0", inner_tag="l.1",
            slabs=(0, 1))
    knl = lp.split_iname(knl, "j", 16,
            outer_tag="g.1", inner_tag="l.0",
            slabs=(0, 1))
    knl = lp.split_iname(knl, "k", 8, slabs=(0, 1))

    knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"],
            fetch_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")
    knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner"],
            fetch_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")

    lp.auto_test_vs_ref(ref_knl, ctx, knl,
            op_count=[2*n**3/1e9], op_label=["GFlops"],
            parameters={"n": n})


def test_funny_shape_matrix_mul(ctx_factory):
    ctx = ctx_factory()

    n = get_suitable_size(ctx)
    m = n+12
    ell = m+12

    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
                ],
            name="matmul", assumptions="n,m,ell >= 1")

    knl = lp.add_dtypes(knl, {
        "a": np.float32,
        "b": np.float32,
        })

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 16,
            outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_iname(knl, "j", 8,
            outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_iname(knl, "k", 32)

    #knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"], default_tag="l.auto")
    #knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner"], default_tag="l.auto")
    knl = lp.extract_subst(knl, "a_acc", "a[i1,i2]", parameters="i1, i2")
    knl = lp.extract_subst(knl, "b_acc", "b[i1,i2]", parameters="i1, i2")
    knl = lp.precompute(knl, "a_acc", "k_inner,i_inner",
            precompute_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")
    knl = lp.precompute(knl, "b_acc", "j_inner,k_inner",
            precompute_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")

    lp.auto_test_vs_ref(ref_knl, ctx, knl,
            op_count=[2*n**3/1e9], op_label=["GFlops"],
            parameters={"n": n, "m": m, "ell": ell})


def test_rank_one(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "F"

    #n = int(get_suitable_size(ctx)**(2.7/2))
    n = 16**3

    knl = lp.make_kernel(
            "[n] -> {[i,j]: 0<=i,j<n}",
            [
                "c[i, j] = a[i]*b[j] {id=mylabel, priority =5}"
                ],
            [
                lp.GlobalArg("a", dtype, shape=("n",), order=order),
                lp.GlobalArg("b", dtype, shape=("n",), order=order),
                lp.GlobalArg("c", dtype, shape=("n, n"), order=order),
                lp.ValueArg("n", np.int32, approximately=n),
                ],
            name="rank_one",
            assumptions="n >= 16")

    def variant_1(knl):
        knl = lp.add_prefetch(knl, "a", default_tag="l.auto")
        knl = lp.add_prefetch(knl, "b", default_tag="l.auto")
        knl = lp.prioritize_loops(knl, ["i", "j"])
        knl = lp.add_inames_to_insn(knl, "i", "writes:b_fetch")
        return knl

    def variant_2(knl):
        knl = lp.split_iname(knl, "i", 16,
                outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_iname(knl, "j", 16,
                outer_tag="g.1", inner_tag="l.1")

        knl = lp.add_prefetch(knl, "a",
                fetch_outer_inames="i_outer, i_inner, j_outer, j_inner")
        knl = lp.add_prefetch(knl, "b",
                fetch_outer_inames="i_outer, i_inner, j_outer, j_inner")
        return knl

    def variant_3(knl):
        knl = lp.split_iname(knl, "i", 16,
                outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_iname(knl, "j", 16,
                outer_tag="g.1", inner_tag="l.1")

        knl = lp.add_prefetch(knl, "a", ["i_inner"],
                    fetch_outer_inames="i_outer, j_outer, j_inner",
                    temporary_address_space=lp.AddressSpace.LOCAL,
                    default_tag="l.auto")
        knl = lp.add_prefetch(knl, "b", ["j_inner"],
                    fetch_outer_inames="i_outer, j_outer, j_inner",
                    temporary_address_space=lp.AddressSpace.LOCAL,
                    default_tag="l.auto")

        return knl

    def variant_4(knl):
        knl = lp.split_iname(knl, "i", 256,
                outer_tag="g.0", slabs=(0, 1))
        knl = lp.split_iname(knl, "j", 256,
                outer_tag="g.1", slabs=(0, 1))

        knl = lp.add_prefetch(knl, "a", ["i_inner"],
                fetch_outer_inames="i_outer, j_outer", default_tag=None)
        knl = lp.add_prefetch(knl, "b", ["j_inner"],
                fetch_outer_inames="i_outer, j_outer", default_tag=None)

        knl = lp.split_iname(knl, "i_inner", 16,
                inner_tag="l.0")
        knl = lp.split_iname(knl, "j_inner", 16,
                inner_tag="l.1")

        knl = lp.split_iname(knl, "a_dim_0", 16,
                outer_tag="l.1", inner_tag="l.0")
        knl = lp.split_iname(knl, "b_dim_0", 16,
                outer_tag="l.1", inner_tag="l.0")
        return knl

    seq_knl = knl

    for variant in [
            variant_1,
            variant_2,
            variant_3,
            variant_4
            ]:
        lp.auto_test_vs_ref(seq_knl, ctx, variant(knl),
                op_count=[np.dtype(dtype).itemsize*n**2/1e9], op_label=["GBytes"],
                parameters={"n": n})


def test_troublesome_premagma_fermi_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    n = 6*16*2

    knl = lp.make_kernel(
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
                ],
            [
                lp.GlobalArg("a", dtype, shape=(n, n), order=order),
                lp.GlobalArg("b", dtype, shape=(n, n), order=order),
                lp.GlobalArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    seq_knl = knl

    i_reg = 2
    j_reg = 2
    i_chunks = 16
    j_chunks = 16
    knl = lp.split_iname(knl, "i", i_reg*i_chunks, outer_tag="g.0")
    knl = lp.split_iname(knl, "i_inner", i_reg, outer_tag="l.0", inner_tag="ilp")
    knl = lp.split_iname(knl, "j", j_reg*j_chunks, outer_tag="g.1")
    knl = lp.split_iname(knl, "j_inner", j_reg, outer_tag="l.1", inner_tag="ilp")
    knl = lp.split_iname(knl, "k", 16)
    knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner_inner", "i_inner_outer"],
            fetch_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")

    lp.auto_test_vs_ref(seq_knl, ctx, knl,
            op_count=[2*n**3/1e9], op_label=["GFlops"],
            parameters={})


def test_intel_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    n = 128+32

    knl = lp.make_kernel(
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
                ],
            [
                lp.GlobalArg("a", dtype, shape=(n, n), order=order),
                lp.GlobalArg("b", dtype, shape=(n, n), order=order),
                lp.GlobalArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    seq_knl = knl

    i_reg = 4
    j_reg = 4
    i_chunks = 16
    j_chunks = 16
    knl = lp.split_iname(knl, "i", i_reg*i_chunks, outer_tag="g.0")
    knl = lp.split_iname(knl, "i_inner", i_reg, outer_tag="l.0", inner_tag="ilp")
    knl = lp.split_iname(knl, "j", j_reg*j_chunks, outer_tag="g.1")
    knl = lp.split_iname(knl, "j_inner", j_reg, outer_tag="l.1", inner_tag="ilp")
    knl = lp.split_iname(knl, "k", 16)
    #knl = lp.split_iname(knl, "k_inner", 8, outer_tag="unr")

    knl = lp.add_prefetch(knl, "a", ["i_inner_inner", "k_inner", "i_inner_outer"],
            fetch_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")
    knl = lp.add_prefetch(knl, "b", ["j_inner_inner", "k_inner", "j_inner_outer"],
            fetch_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")

    # FIXME: Grouped prefetch
    #knl = lp.add_prefetch(knl, "a", ["k_inner", ("i_inner_inner", "i_inner_outer")],
    #           default_tag="l.auto")
    #knl = lp.add_prefetch(knl, "b",
    # ["k_inner", ("j_inner_inner", "j_inner_outer"),], default_tag="l.auto")

    #hints=["k_outer", "k_inner_outer", "k_inner_inner"]

    lp.auto_test_vs_ref(seq_knl, ctx, knl,
            op_count=[2*n**3/1e9], op_label=["GFlops"],
            parameters={})


def test_magma_fermi_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    n = get_suitable_size(ctx)

    if (not ctx.devices[0].image_support
            or ctx.devices[0].platform.name == "Portable Computing Language"):
        pytest.skip("crashes on pocl")

    image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    if image_format not in cl.get_supported_image_formats(
            ctx, cl.mem_flags.READ_ONLY, cl.mem_object_type.IMAGE2D):
        pytest.skip("image format not supported")

    knl = lp.make_kernel(
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
                ],
            [
                lp.ImageArg("a", dtype, shape=(n, n)),
                lp.ImageArg("b", dtype, shape=(n, n)),
                lp.GlobalArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    seq_knl = knl

    i_reg = 4
    j_reg = 4
    i_chunks = 16
    j_chunks = 16

    knl = lp.split_iname(knl, "i", i_reg*i_chunks, outer_tag="g.0")
    knl = lp.split_iname(knl, "i_inner", i_reg, outer_tag="l.0", inner_tag="ilp")
    knl = lp.split_iname(knl, "j", j_reg*j_chunks, outer_tag="g.1")
    knl = lp.split_iname(knl, "j_inner", j_reg, outer_tag="l.1", inner_tag="ilp")
    knl = lp.split_iname(knl, "k", 16)
    knl = lp.split_iname(knl, "k_inner", 8, outer_tag="unr")
    # FIXME
    #knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner_inner", "i_inner_outer"],
    #           default_tag="l.auto")
    #knl = lp.add_prefetch(knl, "b",
    #    ["k_inner", ("j_inner_inner", "j_inner_outer"),], default_tag="l.auto")

    lp.auto_test_vs_ref(seq_knl, ctx, knl,
            op_count=[2*n**3/1e9], op_label=["GFlops"],
            parameters={}, blacklist_ref_vendors="pocl")


def test_image_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    n = get_suitable_size(ctx)

    if (not ctx.devices[0].image_support
            or ctx.devices[0].platform.name == "Portable Computing Language"):
        pytest.skip("crashes on pocl")

    image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    if image_format not in cl.get_supported_image_formats(
            ctx, cl.mem_flags.READ_ONLY, cl.mem_object_type.IMAGE2D):
        pytest.skip("image format not supported")

    knl = lp.make_kernel(
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
                ],
            [
                lp.ImageArg("a", dtype, shape=(n, n)),
                lp.ImageArg("b", dtype, shape=(n, n)),
                lp.GlobalArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    seq_knl = knl

    knl = lp.split_iname(knl, "i", 16, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_iname(knl, "j", 16, outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_iname(knl, "k", 32)
    # conflict-free
    knl = lp.add_prefetch(knl, "a", ["i_inner", "k_inner"],
            fetch_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")
    knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner"],
            fetch_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")

    lp.auto_test_vs_ref(seq_knl, ctx, knl,
            op_count=[2*n**3/1e9], op_label=["GFlops"],
            parameters={}, print_ref_code=True)


def no_test_image_matrix_mul_ilp(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    if (not ctx.devices[0].image_support
            or ctx.devices[0].platform.name == "Portable Computing Language"):
        pytest.skip("crashes on pocl")

    image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    if image_format not in cl.get_supported_image_formats(
            ctx, cl.mem_flags.READ_ONLY, cl.mem_object_type.IMAGE2D):
        pytest.skip("image format not supported")

    n = get_suitable_size(ctx)

    knl = lp.make_kernel(
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
                ],
            [
                lp.ImageArg("a", dtype, shape=(n, n)),
                lp.ImageArg("b", dtype, shape=(n, n)),
                lp.GlobalArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    seq_knl = knl

    ilp = 4
    knl = lp.split_iname(knl, "i", 2, outer_tag="g.0", inner_tag="l.1")
    j_inner_split = 4
    knl = lp.split_iname(knl, "j", ilp*j_inner_split, outer_tag="g.1")
    knl = lp.split_iname(knl, "j_inner", j_inner_split,
            outer_tag="ilp", inner_tag="l.0")
    knl = lp.split_iname(knl, "k", 2)
    # conflict-free?
    knl = lp.add_prefetch(knl, "a", ["i_inner", "k_inner"], default_tag="l.auto")
    knl = lp.add_prefetch(knl, "b", ["j_inner_outer", "j_inner_inner", "k_inner"],
            default_tag="l.auto")

    lp.auto_test_vs_ref(seq_knl, ctx, knl,
            op_count=[2*n**3/1e9], op_label=["GFlops"],
            parameters={})


def test_fancy_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()

    order = "C"

    n = get_suitable_size(ctx)

    knl = lp.make_kernel(
            "[n] -> {[i,j,k]: 0<=i,j,k<n }",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
                ],
            [
                lp.GlobalArg("a", dtype, shape="(n, n)", order=order),
                lp.GlobalArg("b", dtype, shape="(n, n)", order=order),
                lp.GlobalArg("c", dtype, shape="(n, n)", order=order),
                lp.ValueArg("n", np.int32, approximately=1000),
                ], name="fancy_matmul", assumptions="n>=1")

    seq_knl = knl

    knl = lp.split_iname(knl, "i", 16, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_iname(knl, "j", 16, outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_iname(knl, "k", 16, slabs=(0, 1))
    knl = lp.add_prefetch(knl, "a", ["i_inner", "k_inner"],
            fetch_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")
    knl = lp.add_prefetch(knl, "b", ["k_inner", "j_inner"],
            fetch_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")

    lp.auto_test_vs_ref(seq_knl, ctx, knl,
            op_count=[2*n**3/1e9], op_label=["GFlops"],
            parameters=dict(n=n))


def test_small_batched_matvec(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()

    order = "C"

    K = 9997  # noqa
    Np = 36  # noqa

    knl = lp.make_kernel(
            "{[i,j,k]: 0<=k<K and 0<= i,j < %d}" % Np,
            [
                "result[k, i] = sum(j, d[i, j]*f[k, j])"
                ],
            [
                lp.GlobalArg("d", dtype, shape=(Np, Np), order=order),
                lp.GlobalArg("f", dtype, shape=("K", Np), order=order),
                lp.GlobalArg("result", dtype, shape=("K", Np), order=order),
                lp.ValueArg("K", np.int32, approximately=1000),
                ], name="batched_matvec", assumptions="K>=1")

    seq_knl = knl

    align_bytes = 64
    knl = lp.add_prefetch(knl, "d[:,:]", default_tag="l.auto")
    pad_mult = lp.find_padding_multiple(knl, "f", 0, align_bytes)
    knl = lp.split_array_dim(knl, ("f", 0), pad_mult)
    knl = lp.add_padding(knl, "f", 0, align_bytes)

    lp.auto_test_vs_ref(seq_knl, ctx, knl,
            op_count=[K*2*Np**2/1e9], op_label=["GFlops"],
            parameters=dict(K=K))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

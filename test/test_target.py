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

from loopy.diagnostic import LoopyError
import sys
import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.clmath
import pyopencl.clrandom
import pyopencl.tools
import pytest
import pymbolic.primitives as prim

from loopy.target.c import CTarget
from loopy.target.opencl import OpenCLTarget

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

__all__ = [
        "pytest_generate_tests",
        "cl"  # "cl.create_some_context"
        ]


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


def test_ispc_target():
    from loopy.target.ispc import ISPCTarget

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = 2*a[i]",
            [
                lp.GlobalArg("out,a", np.float32, shape=lp.auto),
                "..."
                ],
            target=ISPCTarget())

    knl = lp.split_iname(knl, "i", 8, inner_tag="l.0")
    knl = lp.split_iname(knl, "i_outer", 4, outer_tag="g.0", inner_tag="ilp")
    knl = lp.add_prefetch(knl, "a", ["i_inner", "i_outer_inner"],
            default_tag="l.auto")

    codegen_result = lp.generate_code_v2(knl)

    print(codegen_result.device_code())
    print(codegen_result.host_code())


def test_cuda_target():
    from loopy.target.cuda import CudaTarget

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = 2*a[i]",
            [
                lp.GlobalArg("out,a", np.float32, shape=lp.auto),
                "..."
                ],
            target=CudaTarget())

    knl = lp.split_iname(knl, "i", 8, inner_tag="l.0")
    knl = lp.split_iname(knl, "i_outer", 4, outer_tag="g.0", inner_tag="ilp")
    knl = lp.add_prefetch(knl, "a", ["i_inner", "i_outer_inner"],
            default_tag="l.auto")

    print(
            lp.generate_code_v2(
                knl).device_code())


def test_generate_c_snippet():
    from pymbolic import var
    I = var("I")  # noqa
    f = var("f")
    df = var("df")
    q_v = var("q_v")
    eN = var("eN")  # noqa
    k = var("k")
    u = var("u")

    from functools import partial
    l_sum = partial(lp.Reduction, "sum", allow_simultaneous=True)

    Instr = lp.Assignment  # noqa

    knl = lp.make_kernel(
        "{[I, k]: 0<=I<nSpace and 0<=k<nQuad}",
        [
            Instr(f[I], l_sum(k, q_v[k, I]*u)),
            Instr(df[I], l_sum(k, q_v[k, I])),
            ],
        [
            lp.GlobalArg("q_v", np.float64, shape="nQuad, nSpace"),
            lp.GlobalArg("f,df", np.float64, shape="nSpace"),
            lp.ValueArg("u", np.float64),
            "...",
            ],
        target=CTarget(),
        assumptions="nQuad>=1")

    if 0:  # enable to play with prefetching
        # (prefetch currently requires constant sizes)
        knl = lp.fix_parameters(knl, nQuad=5, nSpace=3)
        knl = lp.add_prefetch(knl, "q_v", "k,I", default_tag=None)

    knl = lp.split_iname(knl, "k", 4, inner_tag="unr", slabs=(0, 1))
    knl = lp.prioritize_loops(knl, "I,k_outer,k_inner")
    print(lp.generate_code_v2(knl))


@pytest.mark.parametrize("target", [CTarget, OpenCLTarget])
@pytest.mark.parametrize("tp", ["f32", "f64"])
def test_math_function(target, tp):
    # Test correct maths functions are generated for C and OpenCL
    # backend instead for different data type

    data_type = {"f32": np.float32,
                 "f64": np.float64}[tp]

    import pymbolic.primitives as p

    i = p.Variable("i")
    xi = p.Subscript(p.Variable("x"), i)
    yi = p.Subscript(p.Variable("y"), i)
    zi = p.Subscript(p.Variable("z"), i)

    n = 100
    domain = "{[i]: 0<=i<%d}" % n
    data = [lp.GlobalArg("x", data_type, shape=(n,)),
            lp.GlobalArg("y", data_type, shape=(n,)),
            lp.GlobalArg("z", data_type, shape=(n,))]

    inst = [lp.Assignment(xi, p.Variable("min")(yi, zi))]
    knl = lp.make_kernel(domain, inst, data, target=target())
    code = lp.generate_code_v2(knl).device_code()

    assert "fmin" in code

    if tp == "f32" and target == CTarget:
        assert "fminf" in code
    else:
        assert "fminf" not in code

    inst = [lp.Assignment(xi, p.Variable("max")(yi, zi))]
    knl = lp.make_kernel(domain, inst, data, target=target())
    code = lp.generate_code_v2(knl).device_code()

    assert "fmax" in code

    if tp == "f32" and target == CTarget:
        assert "fmaxf" in code
    else:
        assert "fmaxf" not in code


@pytest.mark.parametrize("tp", ["f32", "f64"])
def test_random123(ctx_factory, tp):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    import pyopencl.version  # noqa
    if cl.version.VERSION < (2016, 2):
        pytest.skip("Random123 RNG not supported in PyOpenCL < 2016.2")

    n = 150000

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            """
            <> key2 = make_uint2(i, 324830944) {inames=i}
            <> key4 = make_uint4(i, 324830944, 234181, 2233) {inames=i}
            <> ctr = make_uint4(0, 1, 2, 3)  {inames=i,id=init_ctr}
            <> real, ctr = philox4x32_TYPE(ctr, key2)  {id=realpart,dep=init_ctr}
            <> imag, ctr = threefry4x32_TYPE(ctr, key4)  {dep=init_ctr:realpart}

            out[i, 0] = real.s0 + 1j * imag.s0
            out[i, 1] = real.s1 + 1j * imag.s1
            out[i, 2] = real.s2 + 1j * imag.s2
            out[i, 3] = real.s3 + 1j * imag.s3
            """.replace("TYPE", tp))

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")
    knl = lp.set_options(knl, write_code=True)

    evt, (out,) = knl(queue, n=n)

    out = out.get()
    assert (out < 1).all()
    assert (0 <= out).all()


def test_tuple(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    import islpy as isl
    knl = lp.make_kernel(
            [isl.BasicSet("[] -> {[]: }")],
            """
            a, b = make_tuple(1, 2.)
            """)

    evt, (a, b) = knl(queue)

    assert a.get() == 1
    assert b.get() == 2.


def test_clamp(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 15 * 10**6
    x = cl.clrandom.rand(queue, n, dtype=np.float32)

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = clamp(x[i], a, b)")

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")
    knl = lp.set_options(knl, write_code=True)

    evt, (out,) = knl(queue, x=x, a=np.float32(12), b=np.float32(15))


def test_sized_integer_c_codegen(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from pymbolic import var
    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        [lp.Assignment("a[i]", lp.TypeCast(np.int64, 1) << var("i"))]
        )

    knl = lp.set_options(knl, write_code=True)
    n = 40

    evt, (a,) = knl(queue, n=n)

    a_ref = 1 << np.arange(n, dtype=np.int64)

    assert np.array_equal(a_ref, a.get())


def test_child_invalid_type_cast():
    from pymbolic import var
    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        ["<> ctr = make_uint2(0, 0)",
         lp.Assignment("a[i]", lp.TypeCast(np.int64, var("ctr")) << var("i"))]
        )

    with pytest.raises(lp.LoopyError):
        knl = lp.preprocess_kernel(knl)


def test_target_invalid_type_cast():
    dtype = np.dtype([("", "<u4"), ("", "<i4")])
    with pytest.raises(lp.LoopyError):
        lp.TypeCast(dtype, 1)


def test_ispc_streaming_stores():
    stream_dtype = np.float32
    index_dtype = np.int32

    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            "a[i] = b[i] + scalar * c[i]",
            target=lp.ISPCTarget(), index_dtype=index_dtype,
            name="stream_triad")

    vars = ["a", "b", "c", "scalar"]
    knl = lp.assume(knl, "n>0")
    knl = lp.split_iname(
        knl, "i", 2**18, outer_tag="g.0", slabs=(0, 1))
    knl = lp.split_iname(knl, "i_inner", 8, inner_tag="l.0")
    knl = lp.tag_instructions(knl, "!streaming_store")

    knl = lp.add_and_infer_dtypes(knl, {
        var: stream_dtype
        for var in vars
        })

    knl = lp.set_argument_order(knl, vars + ["n"])

    lp.generate_code_v2(knl).all_code()
    assert "streaming_store(" in lp.generate_code_v2(knl).all_code()


def test_cuda_short_vector():
    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]",
        target=lp.CudaTarget())

    knl = lp.set_options(knl, write_code=True)
    knl = lp.split_iname(knl, "i", 4, slabs=(0, 1), inner_tag="vec")
    knl = lp.split_array_axis(knl, "a,out", axis_nr=0, count=4)
    knl = lp.tag_array_axes(knl, "a,out", "C,vec")

    knl = lp.set_options(knl, write_wrapper=True)
    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})

    print(lp.generate_code_v2(knl).device_code())


def test_pyopencl_execution_numpy_handling(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    # test numpy input for x is written to and returned
    knl = lp.make_kernel("{:}", ["x[0] = y[0] + x[0]"])

    y = np.array([3.])
    x = np.array([4.])
    evt, out = knl(queue, y=y, x=x)
    assert out[0] is x
    assert x[0] == 7.

    # test numpy input for x is written to and returned, even when a pyopencl array
    # is passed for y
    import pyopencl.array as cla
    y = cla.zeros(queue, shape=(1), dtype="float64") + 3.
    x = np.array([4.])
    evt, out = knl(queue, y=y, x=x)
    assert out[0] is x
    assert x[0] == 7.

    # test numpy input for x is written to and returned, even when output-only
    knl = lp.make_kernel("{:}", ["x[0] = y[0] + 2"])

    y = np.array([3.])
    x = np.array([4.])
    evt, out = knl(queue, y=y, x=x)
    assert out[0] is x
    assert x[0] == 5.


def test_opencl_support_for_bool(ctx_factory):
    knl = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
        y[i] = i%2
        """,
        [lp.GlobalArg("y", dtype=np.bool_, shape=lp.auto)])

    cl_ctx = ctx_factory()
    evt, (out, ) = knl(cl.CommandQueue(cl_ctx))
    out = out.get()

    np.testing.assert_equal(out, np.tile(np.array([0, 1], dtype=np.bool_), 5))


@pytest.mark.parametrize("target", [lp.PyOpenCLTarget, lp.ExecutableCTarget])
def test_nan_support(ctx_factory, target):
    from loopy.symbolic import parse
    from pymbolic.primitives import NaN, Variable
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{:}",
        [lp.Assignment(parse("a"), np.nan),
         lp.Assignment(parse("b"), parse("isnan(a)")),
         lp.Assignment(parse("c"), parse("isnan(3.14)")),
         lp.Assignment(parse("d"), parse("isnan(0.0)")),
         lp.Assignment(parse("e"), NaN(np.float32)),
         lp.Assignment(parse("f"), Variable("isnan")(NaN(None))),
         lp.Assignment(parse("g"), NaN(np.complex64)),
         lp.Assignment(parse("h"), NaN(np.complex128)),
         ],
        [lp.GlobalArg("a", is_input=False, shape=tuple()), ...],
        seq_dependencies=True, target=target())

    knl = lp.set_options(knl, return_dict=True)

    if target == lp.PyOpenCLTarget:
        evt, out_dict = knl(queue)
        out_dict = {k: v.get() for k, v in out_dict.items()}
    elif target == lp.ExecutableCTarget:
        evt, out_dict = knl()
    else:
        raise NotImplementedError("unsupported target")

    assert np.isnan(out_dict["a"])
    assert out_dict["b"] == 1
    assert out_dict["c"] == 0
    assert out_dict["d"] == 0
    assert np.isnan(out_dict["e"])
    assert out_dict["e"].dtype == np.float32
    assert out_dict["f"] == 1
    assert np.isnan(out_dict["g"])
    assert out_dict["g"].dtype == np.complex64
    assert np.isnan(out_dict["h"])
    assert out_dict["h"].dtype == np.complex128


@pytest.mark.parametrize("target", [lp.PyOpenCLTarget, lp.ExecutableCTarget])
def test_opencl_emits_ternary_operators_correctly(ctx_factory, target):
    # See: https://github.com/inducer/loopy/issues/390
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{:}",
            """
            <> tmp1 = 3.1416
            <> tmp2 = 0.000
            y1 = 1729 if tmp1 else 1.414
            y2 = 42   if 2.7183 else 13
            y3 = 127 if tmp2 else 128
            """, seq_dependencies=True,
            target=target())

    knl = lp.set_options(knl, return_dict=True)

    if target == lp.PyOpenCLTarget:
        evt, out_dict = knl(queue)
    elif target == lp.ExecutableCTarget:
        evt, out_dict = knl()
    else:
        raise NotImplementedError("unsupported target")

    assert out_dict["y1"] == 1729
    assert out_dict["y2"] == 42
    assert out_dict["y3"] == 128


def test_scalar_array_take_offset(ctx_factory):
    import pyopencl.array as cla

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{:}",
        """
        y = 133*x
        """,
        [lp.GlobalArg("x", shape=(), offset=lp.auto),
         ...])

    x_in_base = cla.arange(cq, 42, dtype=np.int32)
    x_in = x_in_base[13]

    evt, (out,) = knl(cq, x=x_in)
    np.testing.assert_allclose(out.get(), 1729)


@pytest.mark.parametrize("target", [lp.PyOpenCLTarget, lp.ExecutableCTarget])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inf_support(ctx_factory, target, dtype):
    from loopy.symbolic import parse
    import math
    # See: https://github.com/inducer/loopy/issues/443 for some laughs
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{:}",
        [lp.Assignment(parse("out_inf"),
                       math.inf),
         lp.Assignment(parse("out_neginf"),
                       -math.inf)],
        [lp.GlobalArg("out_inf", shape=lp.auto,
                      dtype=dtype),
         lp.GlobalArg("out_neginf", shape=lp.auto,
                      dtype=dtype)
         ], target=target())

    knl = lp.set_options(knl, return_dict=True)

    if target == lp.PyOpenCLTarget:
        _, out_dict = knl(queue)
        out_dict = {k: v.get() for k, v in out_dict.items()}
    elif target == lp.ExecutableCTarget:
        _, out_dict = knl()
    else:
        raise NotImplementedError("unsupported target")

    assert np.isinf(out_dict["out_inf"])
    assert np.isneginf(out_dict["out_neginf"])


def test_input_args_are_required(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl1 = lp.make_kernel(
        "{ [i]: 0<=i<2 }",
        """
        g[i] = f[i] + 1.5
        """,
        [lp.GlobalArg("f, g", shape=lp.auto, dtype="float64"), ...]
    )

    knl2 = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "g[i] = 3 * f[i] + g[i]",
    )

    f = np.zeros(2)
    g = np.zeros(2)

    for knl in [knl1, knl2]:
        with pytest.raises(LoopyError):
            _ = knl(queue)
            _ = knl(queue, g=g)

    _ = knl1(queue, f=f)
    _ = knl1(queue, f=f, g=g)

    knl = lp.make_kernel(
        "{ [i]: 0<=i<2 }",
        """
        f[i] = 3.
        g[i] = f[i] + 1.5
        """,
        [lp.GlobalArg("f, g", shape=lp.auto, dtype="float64"), ...]
    )

    # FIXME: this should not raise!
    # https://github.com/inducer/loopy/issues/450
    with pytest.raises(LoopyError):
        _ = knl(queue)


def test_pyopencl_target_with_global_temps_with_base_storage(ctx_factory):
    from pyopencl.tools import ImmediateAllocator

    class RecordingAllocator(ImmediateAllocator):
        def __init__(self, queue):
            super().__init__(queue)
            self.allocated_nbytes = 0

        def __call__(self, size):
            self.allocated_nbytes += size
            return super().__call__(size)

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{[i, j]: 0<=i, j<10}",
        """
        tmp1[i] = 2*i    {id=w_tmp1}
        y[i] = tmp1[i] {nosync=w_tmp1}
        ... gbarrier
        tmp2[j] = 3*j    {id=w_tmp2}
        z[j] = tmp2[j] {nosync=w_tmp2}
        """,
        [lp.TemporaryVariable("tmp1",
                            base_storage="base",
                            address_space=lp.AddressSpace.GLOBAL),
        lp.TemporaryVariable("tmp2",
                            base_storage="base",
                            address_space=lp.AddressSpace.GLOBAL),
        ...],
        seq_dependencies=True)
    knl = lp.tag_inames(knl, {"i": "g.0", "j": "g.0"})
    knl = lp.set_options(knl, return_dict=True)

    my_allocator = RecordingAllocator(cq)
    _, out = knl(cq, allocator=my_allocator)

    np.testing.assert_allclose(out["y"].get(), 2*np.arange(10))
    np.testing.assert_allclose(out["z"].get(), 3*np.arange(10))
    assert my_allocator.allocated_nbytes == (40    # base
                                             + 40  # y
                                             + 40  # z
                                             )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_glibc_bessel_functions(dtype):
    pytest.importorskip("scipy.special")
    from scipy.special import jn, yn  # pylint: disable=no-name-in-module
    from loopy.target.c.c_execution import CCompiler
    from numpy.random import default_rng

    rng = default_rng(0)
    compiler = CCompiler(cflags=["-O3"])

    n = 2
    knl = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
        first_kind_bessel[i]  = bessel_jn(n, x[i])
        second_kind_bessel[i] = bessel_yn(n, x[i])
        """, target=lp.ExecutableCWithGNULibcTarget(compiler))

    if knl.target.compiler.toolchain.cc not in ["gcc", "g++"]:  # pylint: disable=no-member  # noqa: E501
        pytest.skip("GNU-libc not found.")

    knl = lp.fix_parameters(knl, n=2)
    knl = lp.set_options(knl, return_dict=True)
    knl = lp.set_options(knl, write_code=True)
    x_in = np.abs(rng.random(10, dtype=dtype))
    _, out_dict = knl(x=x_in)
    np.testing.assert_allclose(jn(n, x_in), out_dict["first_kind_bessel"],
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(yn(n, x_in), out_dict["second_kind_bessel"],
                               rtol=1e-6, atol=1e-6)


def test_zero_size_temporaries(ctx_factory):
    """Zero-sized arrays in PyOpenCL allocate as "None". This tests that the
    invoker is OK with that.
    """
    # https://github.com/inducer/loopy/pull/588

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{[i]: i > 0 and i < 0}",
        """
        tmp[i] = i
        a[i] = tmp[i]
        """, [lp.TemporaryVariable("tmp", address_space=lp.AddressSpace.GLOBAL,
                                   shape=(0,)),
              lp.GlobalArg("a", shape=(0,)),
              ...])

    _evt, (out, ) = knl(cq)
    assert out.shape == (0,)


def test_empty_array_output(ctx_factory):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{[i]: i > 0 and i < 0}",
        [],
        [lp.GlobalArg("a", shape=(0,), dtype=np.float32,
            is_output=True, is_input=False)])

    _evt, (out, ) = knl(cq)
    assert out.shape == (0,)


def test_empty_array_stride_check(ctx_factory):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    einsum = lp.make_einsum("mij,j->mi", ["a", "x"])
    einsum(cq, a=np.random.randn(3, 0, 5), x=np.random.randn(5))

    if einsum.default_entrypoint.options.skip_arg_checks:
        pytest.skip("args checks disabled, cannot check")

    with pytest.raises(ValueError):
        einsum(cq, a=np.random.randn(3, 2, 5).copy(order="F"), x=np.random.randn(5))


def test_no_op_with_predicate(ctx_factory):
    ctx = ctx_factory()

    predicate = prim.Comparison(prim.Variable("a"), ">", 0)
    knl = lp.make_kernel([],
        ["<> a = 1", lp.NoOpInstruction(predicates=[predicate])])
    code = lp.generate_code_v2(knl).device_code()
    cl.Program(ctx, code).build()


def test_empty_array_stride_check_fortran(ctx_factory):
    # https://github.com/inducer/loopy/issues/583
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    import pyopencl.array as cla

    a_f = cla.Array(queue, (0, 2), np.float64, order="F")

    knl = lp.make_kernel(
        "{ [i,j]: 0<=i<n and 0<=j<m }",
        "output[i,j] = sqrt(input[i,j])")

    knl(queue, input=a_f)


@pytest.mark.parametrize("with_gbarrier", [False, True])
def test_passing_bajillions_of_svm_args(ctx_factory, with_gbarrier):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from pyopencl.characterize import has_coarse_grain_buffer_svm
    if not has_coarse_grain_buffer_svm(queue.device):
        pytest.skip("device does not support SVM, which is required for this test")

    if with_gbarrier:
        gbarrier_part = [
            # Make this artificially have multiple subkernels to check that
            # declarations are correctly emitted in that setting as well
            # https://github.com/inducer/loopy/pull/642#pullrequestreview-1087588248
            "z[j] = 0 {id=init_z}",
            "... gbarrier {dep=init_z,id=gb}"
             ]

        dep = "{dep=gb}"

    else:
        gbarrier_part = []
        dep = ""

    nargsets = 300
    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n}",
            gbarrier_part + [
                f"c{iargset}[i] = a{iargset}[i]+b{iargset}[i] {dep}"
                for iargset in range(nargsets)
            ], [
                lp.GlobalArg(f"{name}{iargset}", shape=lp.auto, dtype=np.float32)
                for name in "abc"
                for iargset in range(nargsets)
                ] + [...],
            target=lp.PyOpenCLTarget(limit_arg_size_nbytes=20),
            options=lp.Options(return_dict=True))

    alloc = cl.tools.SVMAllocator(
            ctx, flags=cl.svm_mem_flags.READ_WRITE, queue=queue)

    multiplier = 10_000
    args = {}
    for iargset in range(nargsets):
        args[f"a{iargset}"] = (
                cl.array.zeros(queue, 20, np.float32, allocator=alloc)
                + np.float32(multiplier * iargset))
        args[f"b{iargset}"] = (
                cl.array.zeros(queue, 20, np.float32, allocator=alloc)
                + np.float32(iargset))

    evt, res = knl(queue, **args, allocator=alloc)

    for iargset in range(nargsets):
        assert (res[f"c{iargset}"].get() == iargset * multiplier + iargset).all()


def test_no_uint_in_cuda_code():
    # https://github.com/inducer/compyte/pull/44
    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = a[i] + b[i]", target=lp.CudaTarget())

    knl = lp.add_and_infer_dtypes(knl, {"a": np.dtype(np.uint32)})
    knl = lp.add_and_infer_dtypes(knl, {"b": np.dtype(np.uint32)})
    assert "uint" not in lp.generate_code_v2(knl).device_code()


def test_ispc_private_var():
    # https://github.com/inducer/loopy/issues/763
    knl = lp.make_kernel(
            "{ [k]: 0<=k<K }",
            """
            <float32> b = 6.0 * float_pos[k]
            output[k] = 2.0 * b
            """, [lp.ValueArg("K", is_input=True),
                  lp.GlobalArg("float_pos", np.float32, shape=lp.auto,
                               is_input=True, is_output=False),
                  lp.GlobalArg("output", np.uint8, shape=lp.auto, is_input=False,
                               is_output=True)],
            target=lp.ISPCTarget(), assumptions="1<K")

    knl = lp.split_iname(knl, "k", 8, inner_tag="l.0")
    knl = lp.set_temporary_address_space(knl, "b", "private")

    cg_result = lp.generate_code_v2(knl)

    print(cg_result.device_code())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

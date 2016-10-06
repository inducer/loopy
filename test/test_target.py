from __future__ import division, absolute_import, print_function

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

import sys
import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.clmath  # noqa
import pyopencl.clrandom  # noqa
import pytest

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
        "cl"  # 'cl.create_some_context'
        ]


def test_ispc_target(occa_mode=False):
    from loopy.target.ispc import ISPCTarget

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = 2*a[i]",
            [
                lp.GlobalArg("out,a", np.float32, shape=lp.auto),
                "..."
                ],
            target=ISPCTarget(occa_mode=occa_mode))

    knl = lp.split_iname(knl, "i", 8, inner_tag="l.0")
    knl = lp.split_iname(knl, "i_outer", 4, outer_tag="g.0", inner_tag="ilp")
    knl = lp.add_prefetch(knl, "a", ["i_inner", "i_outer_inner"])

    codegen_result = lp.generate_code_v2(
                lp.get_one_scheduled_kernel(
                    lp.preprocess_kernel(knl)))

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
    knl = lp.add_prefetch(knl, "a", ["i_inner", "i_outer_inner"])

    print(
            lp.generate_code(
                lp.get_one_scheduled_kernel(
                    lp.preprocess_kernel(knl)))[0])


def test_generate_c_snippet():
    from loopy.target.c import CTarget

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
    knl = lp.set_loop_priority(knl, "I,k_outer,k_inner")

    knl = lp.preprocess_kernel(knl)
    knl = lp.get_one_scheduled_kernel(knl)
    print(lp.generate_body(knl))


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
            <> real, ctr = philox4x32_TYPE(ctr, key2)  {dep=init_ctr}
            <> imag, ctr = threefry4x32_TYPE(ctr, key4)  {dep=init_ctr}

            out[i, 0] = real.s0 + 1j * imag.s0
            out[i, 1] = real.s1 + 1j * imag.s1
            out[i, 2] = real.s2 + 1j * imag.s2
            out[i, 3] = real.s3 + 1j * imag.s3
            """.replace("TYPE", tp))

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")
    knl = lp.set_options(knl, write_cl=True)

    evt, (out,) = knl(queue, n=n)

    out = out.get()
    assert (out < 1).all()
    assert (0 <= out).all()


def test_clamp(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 15 * 10**6
    x = cl.clrandom.rand(queue, n, dtype=np.float32)

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = clamp(x[i], a, b)")

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")
    knl = lp.set_options(knl, write_cl=True)

    evt, (out,) = knl(queue, x=x, a=np.float32(12), b=np.float32(15))


def test_numba_target():
    knl = lp.make_kernel(
        "{[i,j,k]: 0<=i,j<M and 0<=k<N}",
        "D[i,j] = sqrt(sum(k, (X[i, k]-X[j, k])**2))",
        target=lp.NumbaTarget())

    knl = lp.add_and_infer_dtypes(knl, {"X": np.float32})

    print(lp.generate_code_v2(knl).device_code())


def test_numba_cuda_target():
    knl = lp.make_kernel(
        "{[i,j,k]: 0<=i,j<M and 0<=k<N}",
        "D[i,j] = sqrt(sum(k, (X[i, k]-X[j, k])**2))",
        target=lp.NumbaCudaTarget())

    knl = lp.assume(knl, "M>0")
    knl = lp.split_iname(knl, "i", 16, outer_tag='g.0')
    knl = lp.split_iname(knl, "j", 128, inner_tag='l.0', slabs=(0, 1))
    knl = lp.add_prefetch(knl, "X[i,:]")
    knl = lp.fix_parameters(knl, N=3)
    knl = lp.set_loop_priority(knl, "i_inner,j_outer")
    knl = lp.tag_inames(knl, "k:unr")
    knl = lp.tag_array_axes(knl, "X", "N0,N1")

    knl = lp.add_and_infer_dtypes(knl, {"X": np.float32})

    print(lp.generate_code_v2(knl).all_code())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: foldmethod=marker

__copyright__ = "Copyright (C) 2022 Kaushik Kulkarni"

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
import pytest
pytest.importorskip("pycuda")
import pycuda.gpuarray as cu_np
import itertools

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()

from typing import Tuple, Any
from pycuda.tools import init_cuda_context_fixture
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


@pytest.fixture(autouse=True)
def init_cuda_context():
    yield from init_cuda_context_fixture()


def get_random_array(rng, shape: Tuple[int, ...], dtype: np.dtype[Any]):
    if np.issubdtype(dtype, np.complexfloating):
        subdtype = np.empty(0, dtype=dtype).real.dtype
        return (get_random_array(rng, shape, subdtype)
                + dtype.type(1j) * get_random_array(rng, shape, subdtype))
    else:
        assert np.issubdtype(dtype, np.floating)
        return rng.random(shape, dtype=dtype)


@pytest.mark.parametrize("target", [lp.PyCudaTarget(),
                                    lp.PyCudaWithPackedArgsTarget()])
def test_pycuda_invoker(target):
    m = 5
    n = 6

    knl = lp.make_kernel(
        "{[i, j]: 0<=i<m and 0<=j<n}",
        """
        y[i, j] = i+j
        """,
        target=target)

    knl = lp.split_iname(knl, "i", 5, inner_tag="l.0", outer_tag="g.0")
    evt, (out,) = knl(n=n, m=m)
    assert isinstance(out, cu_np.GPUArray)
    np.testing.assert_array_equal(
        out.get(),
        np.arange(n) + np.arange(m).reshape(-1, 1)
    )


@pytest.mark.parametrize("target", [lp.PyCudaTarget(),
                                    lp.PyCudaWithPackedArgsTarget()])
def test_gbarrier(target):
    n = 5
    knl = lp.make_kernel(
        "{[i, j]: 0<=i,j<n}",
        """
        y[i] = i ** 2
        ...gbarrier
        y[j] = (n-j) ** 3
        """,
        seq_dependencies=True,
        target=target)

    knl = lp.split_iname(knl, "i", 5, inner_tag="l.0", outer_tag="g.0")
    knl = lp.split_iname(knl, "j", 2, inner_tag="l.0", outer_tag="g.0")
    evt, (out,) = knl(n=n)

    np.testing.assert_array_equal(out.get(), (n-np.arange(n))**3)


@pytest.mark.parametrize("target", [lp.PyCudaTarget(),
                                    lp.PyCudaWithPackedArgsTarget()])
def test_np_array_input(target):
    from numpy.random import default_rng

    n = 5

    knl = lp.make_kernel(
        "{[i, j]: 0<=i,j<n}",
        """
        z[i] = 2*x[i] + 3*y[i]
        """,
        target=target)

    knl = lp.split_iname(knl, "i", 5, inner_tag="l.0", outer_tag="g.0")
    rng = default_rng(seed=42)

    for x_is_np, y_is_np in itertools.product([False, True], [False, True]):
        x_np = rng.random(n)
        y_np = rng.random(n)
        x = x_np if x_is_np else cu_np.to_gpu(x_np)
        y = y_np if y_is_np else cu_np.to_gpu(y_np)

        evt, (out,) = knl(x=x, y=y)

        assert isinstance(out, (cu_np.GPUArray, np.ndarray))

        out_np = out.get() if isinstance(out, cu_np.GPUArray) else out
        np.testing.assert_allclose(out_np, 2*x_np + 3*y_np)


@pytest.mark.parametrize("target", [lp.PyCudaTarget(),
                                    lp.PyCudaWithPackedArgsTarget()])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_global_temporary(target, dtype):
    rng = np.random.default_rng(seed=314)
    x = rng.random(42, dtype=dtype)

    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        """
        <> tmp[i] = sin(x[i])
        z[i] = 2 * tmp[i]
        """,
        target=target)
    knl = lp.set_temporary_address_space(knl, "tmp", lp.AddressSpace.GLOBAL)

    evt, (out,) = knl(x=x, out_host=False)
    np.testing.assert_allclose(2*np.sin(x), out.get(), rtol=1e-6)


@pytest.mark.parametrize("target", [lp.PyCudaTarget(),
                                    lp.PyCudaWithPackedArgsTarget()])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multi_entrypoints(target, dtype):
    rng = np.random.default_rng(seed=314)
    x = rng.random(42, dtype=dtype)

    knl1 = lp.make_kernel(
        "{[i]: 0<=i<n}",
        """
        z[i] = sin(x[i])
        """,
        name="mysin",
        target=target)
    knl1 = lp.add_dtypes(knl1, {"x": dtype})

    knl2 = lp.make_kernel(
        "{[i]: 0<=i<n}",
        """
        z[i] = cos(x[i])
        """,
        name="mycos",
        target=target)
    knl2 = lp.add_dtypes(knl2, {"x": dtype})

    knl = lp.merge([knl1, knl2])

    evt, (out,) = knl(entrypoint="mycos", x=x)
    np.testing.assert_allclose(np.cos(x), out, rtol=1e-6)

    evt, (out,) = knl(entrypoint="mysin", x=x)
    np.testing.assert_allclose(np.sin(x), out, rtol=1e-6)


@pytest.mark.parametrize("target", [lp.PyCudaTarget(),
                                    lp.PyCudaWithPackedArgsTarget()])
def test_global_arg_with_offsets(target):

    rng = np.random.default_rng(seed=314)
    x = rng.random(104)
    y = rng.random(104)

    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        """
        <> tmp[i] = 21*sin(x[i])  + 864.5*cos(y[i])
        z[i] = 2 * tmp[i]
        """,
        [lp.GlobalArg("x,y",
                      offset=lp.auto, shape=lp.auto),
         ...],
        target=target)
    knl = lp.set_temporary_address_space(knl, "tmp", lp.AddressSpace.GLOBAL)

    evt, (out,) = knl(x=x, y=y)
    np.testing.assert_allclose(42*np.sin(x) + 1729*np.cos(y), out)


@pytest.mark.parametrize("target", [lp.PyCudaTarget(),
                                    lp.PyCudaWithPackedArgsTarget()])
@pytest.mark.parametrize("dtype,rtol", [(np.complex64, 1e-6),
                                        (np.complex128, 1e-14),
                                        (np.float32, 1e-6),
                                        (np.float64, 1e-14)])
def test_sum_of_array(target, dtype, rtol):
    # Reported by Mit Kotak
    rng = np.random.default_rng(seed=0)
    knl = lp.make_kernel(
        "{[i]: 0 <= i < N}",
        """
        out = sum(i, x[i])
        """,
        target=target)
    x = get_random_array(rng, (42,), np.dtype(dtype))
    evt, (out,) = knl(x=x)
    np.testing.assert_allclose(np.sum(x), out, rtol=rtol)


@pytest.mark.parametrize("target", [lp.PyCudaTarget(),
                                    lp.PyCudaWithPackedArgsTarget()])
@pytest.mark.parametrize("dtype,rtol", [(np.complex64, 1e-6),
                                        (np.complex128, 1e-14),
                                        (np.float32, 1e-6),
                                        (np.float64, 1e-14)])
def test_int_pow(target, dtype, rtol):
    rng = np.random.default_rng(seed=0)
    knl = lp.make_kernel(
        "{[i]: 0 <= i < N}",
        """
        out[i] = x[i] ** i
        """,
        target=target)
    x = get_random_array(rng, (10,), np.dtype(dtype))
    evt, (out,) = knl(x=x)
    np.testing.assert_allclose(x ** np.arange(10, dtype=np.int32), out,
                               rtol=rtol)


@pytest.mark.parametrize("target", [lp.PyCudaTarget(),
                                    lp.PyCudaWithPackedArgsTarget()])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128,
                                   np.float32, np.float64])
@pytest.mark.parametrize("func", ["abs", "sqrt",
                                  "sin",  "cos", "tan",
                                  "sinh",  "cosh", "tanh",
                                  "exp", "log", "log10"])
def test_math_functions(target, dtype, func):
    rng = np.random.default_rng(seed=0)
    knl = lp.make_kernel(
        "{[i]: 0 <= i < N}",
        f"""
        y[i] = {func}(x[i])
        """,
        target=target)
    x = get_random_array(rng, (42,), np.dtype(dtype))
    _, (out,) = knl(x=x)
    np.testing.assert_allclose(getattr(np, func)(x),
                               out, rtol=1e-6)


def test_pycuda_packargs_tgt_avoids_param_space_overflow():
    from pymbolic.primitives import Sum
    from loopy.symbolic import parse

    nargs = 1_000
    rng = np.random.default_rng(32)
    knl = lp.make_kernel(
        "{[i]: 0<=i<N}",
        [lp.Assignment("out[i]",
                       Sum(tuple(parse(f"arg{i}[i]")
                                 for i in range(nargs))))],
        [lp.GlobalArg(",".join(f"arg{i}" for i in range(nargs)) + ",out",
                      dtype="float64", shape=("N",)),
         lp.ValueArg("N", dtype="int32")],
        target=lp.PyCudaWithPackedArgsTarget()
    )

    args = {f"arg{i}": rng.random(10) for i in range(nargs)}

    evt, (out,) = knl(**args)
    np.testing.assert_allclose(sum(args.values()), out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

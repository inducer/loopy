__copyright__ = """
Copyright (C) 2012 Andreas Kloeckner
Copyright (C) 2016, 2017 Matt Wala
"""

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


# More things to test.
# - scan(a) + scan(b)
# - test for badly tagged inames

from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


@pytest.mark.parametrize("n", [1, 2, 3, 16])
@pytest.mark.parametrize("stride", [1, 2])
def test_sequential_scan(ctx_factory, n, stride):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "[n] -> {[i,j]: 0<=i<n and 0<=j<=%d*i}" % stride,
        """
        a[i] = sum(j, j**2)
        """
        )

    knl = lp.fix_parameters(knl, n=n)
    knl = lp.realize_reduction(knl, force_scan=True)

    evt, (a,) = knl(queue)

    assert (a.get() == np.cumsum(np.arange(stride*n)**2)[::stride]).all()


@pytest.mark.parametrize("sweep_lbound, scan_lbound", [
    (4, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 4),
    (5, -1),
    ])
def test_scan_with_different_lower_bound_from_sweep(
        ctx_factory, sweep_lbound, scan_lbound):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "[n, sweep_lbound, scan_lbound] -> "
        "{[i,j]: sweep_lbound<=i<n+sweep_lbound "
        "and scan_lbound<=j<=2*(i-sweep_lbound)+scan_lbound}",
        """
        out[i-sweep_lbound] = sum(j, j**2)
        """
        )

    n = 10

    knl = lp.fix_parameters(knl, sweep_lbound=sweep_lbound, scan_lbound=scan_lbound)
    knl = lp.realize_reduction(knl, force_scan=True)
    evt, (out,) = knl(queue, n=n)

    assert (out.get()
            == np.cumsum(np.arange(scan_lbound, 2*n+scan_lbound)**2)[::2]).all()


def test_automatic_scan_detection():
    knl = lp.make_kernel(
        [
            "[n] -> {[i]: 0<=i<n}",
            "{[j]: 0<=j<=2*i}"
        ],
        """
        a[i] = sum(j, j**2)
        """
        )

    cgr = lp.generate_code_v2(knl)
    assert "scan" not in cgr.device_code()


def test_selective_scan_realization():
    pass


def test_force_outer_iname_for_scan():
    knl = lp.make_kernel(
        "[n] -> {[i,j,k]: 0<=k<n and 0<=i<=k and 0<=j<=i}",
        "out[i] = product(j, a[j]) {inames=i:k}")

    knl = lp.add_dtypes(knl, dict(a=np.float32))

    # TODO: Maybe this deserves to work?
    with pytest.raises(lp.diagnostic.ReductionIsNotTriangularError):
        lp.realize_reduction(knl, force_scan=True)

    knl = lp.realize_reduction(knl, force_scan=True, force_outer_iname_for_scan="i")


def test_dependent_domain_scan(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        [
            "[n] -> {[i]: 0<=i<n}",
            "{[j]: 0<=j<=2*i}"
        ],
        """
        a[i] = sum(j, j**2) {id=scan}
        """
        )
    knl = lp.realize_reduction(knl, force_scan=True)
    evt, (a,) = knl(queue, n=100)

    assert (a.get() == np.cumsum(np.arange(200)**2)[::2]).all()


@pytest.mark.parametrize("i_tag, j_tag", [
    ("for", "for")
    ])
def test_nested_scan(ctx_factory, i_tag, j_tag):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        [
            "[n] -> {[i]: 0 <= i < n}",
            "[i] -> {[j]: 0 <= j <= i}",
            "[i] -> {[k]: 0 <= k <= i}"
        ],
        """
        <>tmp[i] = sum(k, 1)
        out[i] = sum(j, tmp[j])
        """)

    knl = lp.fix_parameters(knl, n=10)
    knl = lp.tag_inames(knl, dict(i=i_tag, j=j_tag))

    knl = lp.realize_reduction(knl, force_scan=True)

    print(knl)

    evt, (out,) = knl(queue)

    print(out)


def test_scan_not_triangular():
    knl = lp.make_kernel(
        "{[i,j]: 0<=i<100 and 1<=j<=2*i}",
        """
        a[i] = sum(j, j**2)
        """
        )

    with pytest.raises(lp.diagnostic.ReductionIsNotTriangularError):
        knl = lp.realize_reduction(knl, force_scan=True)


@pytest.mark.parametrize("n", [1, 2, 3, 16, 17])
def test_local_parallel_scan(ctx_factory, n):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "[n] -> {[i,j]: 0<=i<n and 0<=j<=i}",
        """
        out[i] = sum(j, a[j]**2)
        """,
        "..."
        )

    knl = lp.fix_parameters(knl, n=n)
    knl = lp.tag_inames(knl, dict(i="l.0"))
    knl = lp.realize_reduction(knl, force_scan=True)

    knl = lp.add_dtypes(knl, dict(a=int))

    evt, (a,) = knl(queue, a=np.arange(n))
    assert (a == np.cumsum(np.arange(n)**2)).all()


def test_local_parallel_scan_with_nonzero_lower_bounds(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "[n] -> {[i,j]: 1<=i<n+1 and 0<=j<=i-1}",
        """
        out[i-1] = sum(j, a[j]**2)
        """,
        "..."
        )

    knl = lp.fix_parameters(knl, n=16)
    knl = lp.tag_inames(knl, dict(i="l.0"))
    knl = lp.realize_reduction(knl, force_scan=True)

    knl = lp.add_dtypes(knl, dict(a=int))
    evt, (out,) = knl(queue, a=np.arange(1, 17))

    assert (out == np.cumsum(np.arange(1, 17)**2)).all()


def test_scan_extra_constraints_on_domain():
    knl = lp.make_kernel(
        "{[i,j,k]: 0<=i<n and 0<=j<=i and i=k}",
        "out[i] = sum(j, a[j])")

    with pytest.raises(lp.diagnostic.ReductionIsNotTriangularError):
        knl = lp.realize_reduction(
                knl, force_scan=True, force_outer_iname_for_scan="i")


@pytest.mark.parametrize("sweep_iname_tag", ["for", "l.1"])
def test_scan_with_outer_parallel_iname(ctx_factory, sweep_iname_tag):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        [
            "{[k]: 0<=k<=1}",
            "[n] -> {[i,j]: 0<=i<n and 0<=j<=i}"
        ],
        "out[k,i] = k + sum(j, j**2)"
        )

    knl = lp.tag_inames(knl, dict(k="l.0", i=sweep_iname_tag))
    n = 10
    knl = lp.fix_parameters(knl, n=n)
    knl = lp.realize_reduction(knl, force_scan=True)

    evt, (out,) = knl(queue)

    inner = np.cumsum(np.arange(n)**2)

    assert (out.get() == np.array([inner, 1 + inner])).all()


@pytest.mark.parametrize("dtype", [
    np.int32, np.int64, np.float32, np.float64])
def test_scan_data_types(ctx_factory, dtype):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{[i,j]: 0<=i<n and 0<=j<=i }",
            "res[i] = reduce(sum, j, a[j])",
            assumptions="n>=1")

    a = np.random.randn(20).astype(dtype)
    knl = lp.add_dtypes(knl, dict(a=dtype))
    knl = lp.realize_reduction(knl, force_scan=True)
    evt, (res,) = knl(queue, a=a)

    assert np.allclose(res, np.cumsum(a))


@pytest.mark.parametrize(("op_name", "np_op"), [
    ("sum", np.sum),
    ("product", np.prod),
    ("min", np.min),
    ("max", np.max),
    ])
def test_scan_library(ctx_factory, op_name, np_op):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{[i,j]: 0<=i<n and 0<=j<=i }",
            "res[i] = reduce(%s, j, a[j])" % op_name,
            assumptions="n>=1")

    a = np.random.randn(20)
    knl = lp.add_dtypes(knl, dict(a=np.float64))
    knl = lp.realize_reduction(knl, force_scan=True)
    evt, (res,) = knl(queue, a=a)

    assert np.allclose(res, np.array(
            [np_op(a[:i+1]) for i in range(len(a))]))


def test_scan_unsupported_tags():
    pass


@pytest.mark.parametrize("i_tag", ["for", "l.0"])
def test_argmax(ctx_factory, i_tag):
    logging.basicConfig(level=logging.INFO)

    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 128

    knl = lp.make_kernel(
            "{[i,j]: 0<=i<%d and 0<=j<=i}" % n,
            """
            max_vals[i], max_indices[i] = argmax(j, abs(a[j]), j)
            """)

    knl = lp.tag_inames(knl, dict(i=i_tag))
    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})
    knl = lp.realize_reduction(knl, force_scan=True)

    a = np.random.randn(n).astype(dtype)
    evt, (max_indices, max_vals) = knl(queue, a=a, out_host=True)

    assert (max_vals == [np.max(np.abs(a)[0:i+1]) for i in range(n)]).all()
    assert (max_indices == [np.argmax(np.abs(a[0:i+1])) for i in range(n)]).all()


def check_segmented_scan_output(arr, segment_boundaries_indices, out):
    class SegmentGrouper:

        def __init__(self):
            self.seg_idx = 0
            self.idx = 0

        def __call__(self, key):
            if self.idx in segment_boundaries_indices:
                self.seg_idx += 1
            self.idx += 1
            return self.seg_idx

    from itertools import groupby

    expected = [np.cumsum(list(group))
            for _, group in groupby(arr, SegmentGrouper())]
    actual = [np.array(list(group))
            for _, group in groupby(out, SegmentGrouper())]

    assert len(expected) == len(actual) == len(segment_boundaries_indices)
    assert [(e == a).all() for e, a in zip(expected, actual)]


@pytest.mark.parametrize("n, segment_boundaries_indices", [
    (1, (0,)),
    (2, (0,)),
    (2, (0, 1)),
    (3, (0,)),
    (3, (0, 1)),
    (3, (0, 2)),
    (3, (0, 1, 2)),
    (16, (0, 4, 8, 12))])
@pytest.mark.parametrize("iname_tag", ("for", "l.0"))
def test_segmented_scan(ctx_factory, n, segment_boundaries_indices, iname_tag):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    arr = np.ones(n, dtype=np.float32)
    segment_boundaries = np.zeros(n, dtype=np.int32)
    segment_boundaries[(segment_boundaries_indices,)] = 1

    knl = lp.make_kernel(
        "{[i,j]: 0<=i<n and 0<=j<=i}",
        "out[i], <>_ = reduce(segmented(sum), j, arr[j], segflag[j])",
        [
            lp.GlobalArg("arr", np.float32, shape=("n",)),
            lp.GlobalArg("segflag", np.int32, shape=("n",)),
            "..."
        ])

    knl = lp.fix_parameters(knl, n=n)
    knl = lp.tag_inames(knl, dict(i=iname_tag))
    knl = lp.realize_reduction(knl, force_scan=True)

    (evt, (out,)) = knl(queue, arr=arr, segflag=segment_boundaries)

    check_segmented_scan_output(arr, segment_boundaries_indices, out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

from __future__ import division, absolute_import, print_function

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
from pytools import memoize

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
# - test that dummy inames are removed
# - scan(a) + scan(b)
# - global parallel scan

# TO DO:
# segmented<sum>(...) syntax


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
    assert "scan" in cgr.device_code()


def test_selective_scan_realization():
    pass


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

    knl = lp.realize_reduction(knl)

    knl = lp.add_dtypes(knl, dict(a=int))

    print(knl)

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
    knl = lp.realize_reduction(knl)

    knl = lp.add_dtypes(knl, dict(a=int))
    evt, (out,) = knl(queue, a=np.arange(1, 17))

    assert (out == np.cumsum(np.arange(1, 17)**2)).all()


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


def test_scan_data_types():
    # TODO
    pass


def test_scan_library():
    # TODO
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
            max_vals[i], max_indices[i] = argmax(j, fabs(a[j]), j)
            """)

    knl = lp.tag_inames(knl, dict(i=i_tag))
    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})
    knl = lp.realize_reduction(knl, force_scan=True)

    print(knl)

    a = np.random.randn(n).astype(dtype)
    evt, (max_indices, max_vals) = knl(queue, a=a, out_host=True)

    assert (max_vals == [np.max(np.abs(a)[0:i+1]) for i in range(n)]).all()
    assert (max_indices == [np.argmax(np.abs(a[0:i+1])) for i in range(n)]).all()


@pytest.mark.parametrize("n, segment_boundaries_indices", (
    (1, (0,)),
    (2, (0,)),
    (2, (0, 1)),
    (3, (0,)),
    (3, (0, 1)),
    (3, (0, 2)),
    (3, (0, 1, 2)),
    (16, (0, 5)),
    ))
@pytest.mark.parametrize("iname_tag", ("for", "l.0"))
def test_segmented_scan(ctx_factory, n, segment_boundaries_indices, iname_tag):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    arr = np.ones(n, dtype=np.float32)
    segment_boundaries = np.zeros(n, dtype=np.int32)
    segment_boundaries[(segment_boundaries_indices,)] = 1

    knl = lp.make_kernel(
        "{[i,j]: 0<=i<n and 0<=j<=i}",
        "out[i], <>_ = segmented_sum(j, arr[j], segflag[j])",
        [
            lp.GlobalArg("arr", np.float32, shape=("n",)),
            lp.GlobalArg("segflag", np.int32, shape=("n",)),
            "..."
        ])

    knl = lp.fix_parameters(knl, n=n)
    knl = lp.tag_inames(knl, dict(i=iname_tag))
    knl = lp.realize_reduction(knl, force_scan=True)

    (evt, (out,)) = knl(queue, arr=arr, segflag=segment_boundaries)

    class SegmentGrouper(object):

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


# {{{ two and three level scan getters

@memoize
def _get_two_level_scan_kernel(g_size):
    knl = lp.make_kernel(
        [
            "[n] -> {[i,j]: 0 <= i < n and 0 <= j <= i}",
        ],
        """
        out[i] = sum(j, a[j]) {id=insn}
        """,
        "...",
        assumptions="n > 0")

    from loopy.transform.reduction import make_two_level_scan
    knl = make_two_level_scan(
        knl, "insn", inner_length=g_size,
        scan_iname="j",
        sweep_iname="i",
        local_storage_axes=(("i__l0",)),
        inner_iname="i__l0",
        inner_tag="l.0",
        outer_tag="g.0",
        local_storage_scope=lp.temp_var_scope.LOCAL,
        nonlocal_storage_scope=lp.temp_var_scope.GLOBAL,
        inner_local_tag="l.0",
        outer_local_tag="g.0")

    knl = lp.realize_reduction(knl, force_scan=True)

    from loopy.transform.instruction import add_nosync_to_instructions
    knl = add_nosync_to_instructions(
            knl,
            scope="global",
            source="writes:acc_j__l0",
            sink="reads:acc_j__l0")

    from loopy.transform.save import save_and_reload_temporaries

    knl = lp.preprocess_kernel(knl)
    knl = lp.get_one_scheduled_kernel(knl)
    knl = save_and_reload_temporaries(knl)
    knl = lp.get_one_scheduled_kernel(knl)

    return knl


@memoize
def _get_three_level_scan_kernel(g_size, p_size):
    knl = lp.make_kernel(
        [
            "[n] -> {[i,j]: 0 <= i < n and 0 <= j <= i}",
        ],
        """
        out[i] = sum(j, a[j]) {id=insn}
        """,
        "...",
        assumptions="n > 0")

    from loopy.transform.reduction import make_two_level_scan
    knl = make_two_level_scan(
        knl, "insn", inner_length=g_size,
        scan_iname="j",
        sweep_iname="i",
        local_storage_axes=(("i__l0",)),
        inner_iname="i__l0",
        inner_tag=None,
        outer_tag="g.0",
        local_storage_scope=lp.temp_var_scope.LOCAL,
        nonlocal_storage_scope=lp.temp_var_scope.GLOBAL,
        inner_local_tag=None,
        outer_local_tag="g.0")

    knl = make_two_level_scan(
        knl, "insn__l1", inner_length=p_size,
        scan_iname="j__l1",
        sweep_iname="i__l1",
        inner_tag="for",
        outer_tag="l.0",
        nonlocal_tag="l.0",
        local_storage_scope=lp.temp_var_scope.LOCAL,
        nonlocal_storage_scope=lp.temp_var_scope.LOCAL,
        inner_local_tag="for",
        outer_local_tag="l.0")

    knl = lp.tag_inames(knl, dict(i__l0="l.0",
                                  i__l0_nltail_inner="l.0"))

    knl = lp.realize_reduction(knl, force_scan=True)

    from loopy.transform.instruction import add_nosync_to_instructions
    knl = add_nosync_to_instructions(
            knl,
            scope="global",
            source="writes:acc_j__l0",
            sink="reads:acc_j__l0")

    knl = lp.alias_temporaries(knl,
            ("insn__l1", "insn__l2"),
            synchronize_for_exclusive_use=False)

    from loopy.transform.save import save_and_reload_temporaries

    knl = lp.preprocess_kernel(knl)
    knl = lp.get_one_scheduled_kernel(knl)
    knl = save_and_reload_temporaries(knl)
    knl = lp.get_one_scheduled_kernel(knl)

    return knl

# }}}


@pytest.mark.parametrize("input_len",
        (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 32))
@pytest.mark.parametrize("g_size", (16,))
def test_two_level_scan(ctx_getter, input_len, g_size):
    knl = _get_two_level_scan_kernel(g_size)

    import numpy as np
    np.random.seed(0)
    a = np.random.randint(low=0, high=100, size=input_len)

    c = ctx_getter()
    q = cl.CommandQueue(c)

    _, (out,) = knl(q, a=a)

    assert (out == np.cumsum(a)).all()


@pytest.mark.parametrize("input_len",
        (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 32))
@pytest.mark.parametrize("g_size", (16,))
@pytest.mark.parametrize("p_size", (4,))
def test_three_level_scan(ctx_getter, g_size, p_size, input_len):
    knl = _get_three_level_scan_kernel(g_size, p_size)

    import numpy as np
    np.random.seed(0)
    a = np.random.randint(low=0, high=100, size=input_len)

    c = ctx_getter()
    q = cl.CommandQueue(c)

    _, (out,) = knl(q, a=a)

    assert (out == np.cumsum(a)).all()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: foldmethod=marker

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
import pytest  # noqa

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


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


def test_assume():
    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            "a[i] = a[i] + 1",
            [lp.GlobalArg("a", np.float32, shape="n"), "..."],
            target=lp.PyOpenCLTarget())

    knl = lp.split_iname(knl, "i", 16)
    knl = lp.prioritize_loops(knl, "i_outer,i_inner")
    knl = lp.assume(knl, "n mod 16 = 0")
    knl = lp.assume(knl, "n > 10")
    code = lp.generate_code_v2(knl).device_code()
    assert "if" not in code


def test_divisibility_assumption(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "[n] -> {[i]: 0<=i<n}",
            [
                "b[i] = 2*a[i]"
                ],
            [
                lp.GlobalArg("a", np.float32, shape=("n",)),
                lp.GlobalArg("b", np.float32, shape=("n",)),
                lp.ValueArg("n", np.int32),
                ],
            assumptions="n>=1 and (exists zz: n = 16*zz)",
            target=lp.PyOpenCLTarget())

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 16)
    code = lp.generate_code_v2(knl).device_code()
    assert "if" not in code

    lp.auto_test_vs_ref(ref_knl, ctx, knl,
            parameters={"n": 16**3})


def test_eq_constraint():
    knl = lp.make_kernel(
            "{[i]: 0<= i < 32}",
            [
                "a[i] = b[i]"
                ],
            [
                lp.GlobalArg("a", np.float32, shape=(1000,)),
                lp.GlobalArg("b", np.float32, shape=(1000,))
                ],
            target=lp.PyOpenCLTarget())

    knl = lp.split_iname(knl, "i", 16, outer_tag="g.0")
    knl = lp.split_iname(knl, "i_inner", 16, outer_tag=None, inner_tag="l.0")
    print(lp.generate_code_v2(knl).device_code())


def test_dependent_loop_bounds():
    dtype = np.dtype(np.float32)

    knl = lp.make_kernel(
            [
                "{[i]: 0<=i<n}",
                "{[jj]: 0<=jj<row_len}",
                ],
            [
                "<> row_len = a_rowstarts[i+1] - a_rowstarts[i]",
                "a_sum[i] = sum(jj, a_values[[a_rowstarts[i]+jj]])",
                ],
            [
                lp.GlobalArg("a_rowstarts", np.int32, shape=lp.auto),
                lp.GlobalArg("a_indices", np.int32, shape=lp.auto),
                lp.GlobalArg("a_values", dtype),
                lp.GlobalArg("a_sum", dtype, shape=lp.auto),
                lp.ValueArg("n", np.int32),
                ],
            assumptions="n>=1 and row_len>=1",
            target=lp.PyOpenCLTarget())

    print(lp.generate_code_v2(knl).device_code())


def test_dependent_loop_bounds_2():
    dtype = np.dtype(np.float32)

    knl = lp.make_kernel(
            [
                "{[i]: 0<=i<n}",
                "{[jj]: 0<=jj<row_len}",
                ],
            [
                "<> row_start = a_rowstarts[i]",
                "<> row_len = a_rowstarts[i+1] - row_start",
                "ax[i] = sum(jj, a_values[[row_start+jj]])",
                ],
            [
                lp.GlobalArg("a_rowstarts", np.int32, shape=lp.auto),
                lp.GlobalArg("a_indices", np.int32, shape=lp.auto),
                lp.GlobalArg("a_values", dtype, strides=(1,)),
                lp.GlobalArg("ax", dtype, shape=lp.auto),
                lp.ValueArg("n", np.int32),
                ],
            assumptions="n>=1 and row_len>=1",
            target=lp.PyOpenCLTarget())

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0",
            inner_tag="l.0")

    print(lp.generate_code_v2(knl).device_code())


def test_dependent_loop_bounds_3():
    # The point of this test is that it shows a dependency between
    # domains that is exclusively mediated by the row_len temporary.
    # It also makes sure that row_len gets read before any
    # conditionals use it.

    dtype = np.dtype(np.float32)

    knl = lp.make_kernel(
            [
                "{[i]: 0<=i<n}",
                "{[jj]: 0<=jj<row_len}",
                ],
            [
                "<> row_len = a_row_lengths[i]",
                "a[i,jj] = 1",
                ],
            [
                lp.GlobalArg("a_row_lengths", np.int32, shape=lp.auto),
                lp.GlobalArg("a", dtype, shape=("n,n"), order="C"),
                lp.ValueArg("n", np.int32),
                ],
            target=lp.PyOpenCLTarget(),
            name="loopy_kernel")

    assert knl["loopy_kernel"].parents_per_domain()[1] == 0

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0",
            inner_tag="l.0")

    print(lp.generate_code_v2(knl).device_code())

    knl_bad = lp.split_iname(knl, "jj", 128, outer_tag="g.1",
            inner_tag="l.1")

    with pytest.raises(RuntimeError):
        list(lp.generate_code_v2(knl_bad))


def test_dependent_loop_bounds_4():
    # https://gitlab.tiker.net/inducer/loopy/issues/23
    import loopy as lp

    loopy_knl = lp.make_kernel(
        [
            "{[a]: 0<=a<10}",
            "{[b]: b_start<=b<b_end}",
            "{[c,idim]: c_start<=c<c_end and 0<=idim<dim}",
        ],
        """
        for a
         <> b_start = 1
         <> b_end = 2
         for b
          <> c_start = 1
          <> c_end = 2

          for c
           ... nop
          end

          <>t[idim] = 1
         end
        end
        """,
        "...",
        seq_dependencies=True)

    loopy_knl = lp.fix_parameters(loopy_knl, dim=3)

    with lp.CacheMode(False):
        lp.generate_code_v2(loopy_knl)


def test_independent_multi_domain(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            [
                "{[i]: 0<=i<n}",
                "{[j]: 0<=j<n}",
                ],
            [
                "a[i] = 1",
                "b[j] = 2",
                ],
            [
                lp.GlobalArg("a", dtype, shape=("n"), order="C"),
                lp.GlobalArg("b", dtype, shape=("n"), order="C"),
                lp.ValueArg("n", np.int32),
                ],
            name="loopy_kernel")

    knl = lp.split_iname(knl, "i", 16, outer_tag="g.0",
            inner_tag="l.0")
    knl = lp.split_iname(knl, "j", 16, outer_tag="g.0",
            inner_tag="l.0")
    assert knl["loopy_kernel"].parents_per_domain() == 2*[None]

    n = 50
    evt, (a, b) = knl(queue, n=n, out_host=True)

    assert a.shape == (50,)
    assert b.shape == (50,)
    assert (a == 1).all()
    assert (b == 2).all()


def test_equality_constraints(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()

    order = "C"

    n = 10

    knl = lp.make_kernel([
            "[n] -> {[i,j]: 0<=i,j<n }",
            "{[k]: k =i+5 and k < n}",
            ],
            [
                "a[i,j] = 5 {id=set_all}",
                "b[i,k] = 22 {id=set_b, dep=set_all}",
                ],
            [
                lp.GlobalArg("a,b", dtype, shape="n, n", order=order),
                lp.ValueArg("n", np.int32, approximately=1000),
                ],
            name="equality_constraints", assumptions="n>=1")

    seq_knl = knl

    knl = lp.split_iname(knl, "i", 16, outer_tag="g.0", inner_tag="l.0")
    knl = lp.split_iname(knl, "j", 16, outer_tag="g.1", inner_tag="l.1")

    knl = lp.add_inames_to_insn(knl, "j_inner, j_outer", "id:set_b")

    #print(knl)
    #print(knl.domains[0].detect_equalities())

    lp.auto_test_vs_ref(seq_knl, ctx, knl,
            parameters=dict(n=n), print_ref_code=True)


def test_stride(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()

    order = "C"

    n = 10

    knl = lp.make_kernel([
            "{[i]: 0<=i<n and (exists l: i = 2*l)}",
            ],
            [
                "a[i] = 5",
                ],
            [
                lp.GlobalArg("a", dtype, shape="n", order=order),
                lp.ValueArg("n", np.int32, approximately=1000),
                ],
            assumptions="n>=1")

    seq_knl = knl

    lp.auto_test_vs_ref(seq_knl, ctx, knl,
            parameters=dict(n=n))


def test_domain_dependency_via_existentially_quantified_variable(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()

    order = "C"

    n = 10

    knl = lp.make_kernel([
            "{[i]: 0<=i<n }",
            "{[k]: k=i and (exists l: k = 2*l) }",
            ],
            [
                "a[i] = 5 {id=set}",
                "b[k] = 6 {dep=set}",
                ],
            [
                lp.GlobalArg("a,b", dtype, shape="n", order=order),
                lp.ValueArg("n", np.int32, approximately=1000),
                ],
            assumptions="n>=1")

    seq_knl = knl

    lp.auto_test_vs_ref(seq_knl, ctx, knl,
            parameters=dict(n=n))


def test_triangle_domain():
    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n and i <= j}",
            "a[i,j] = 17",
            assumptions="n>=1",
            target=lp.PyOpenCLTarget())

    print(knl)
    print(lp.generate_code_v2(knl).device_code())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

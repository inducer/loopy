__copyright__ = "Copyright (C) 2018 Kaushik Kulkarni"

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

import numpy as np
import pyopencl as cl
import pyopencl.clrandom  # noqa: F401
import loopy as lp
import pytest
import sys
from pytools import ImmutableRecord


from pyopencl.tools import (  # noqa: F401
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


def test_register_function_lookup(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from testlib import Log2Callable

    x = np.random.rand(10)
    queue = cl.CommandQueue(ctx)

    prog = lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            y[i] = log2(x[i])
            """)
    prog = lp.register_callable(prog, "log2", Log2Callable("log2"))

    evt, (out, ) = prog(queue, x=x)

    assert np.linalg.norm(np.log2(x)-out)/np.linalg.norm(np.log2(x)) < 1e-15


@pytest.mark.parametrize("inline", [False, True])
def test_register_knl(ctx_factory, inline):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    n = 4

    x = np.random.rand(n, n, n, n, n)
    y = np.random.rand(n, n, n, n, n)

    grandchild_knl = lp.make_function(
            "{[i, j]:0<= i, j< 4}",
            """
            c[i, j] = 2*a[i, j] + 3*b[i, j]
            """, name="linear_combo1")

    child_knl = lp.make_function(
            "{[i, j]:0<=i, j < 4}",
            """
            [i, j]: g[i, j] = linear_combo1([i, j]: e[i, j], [i, j]: f[i, j])
            """, name="linear_combo2")

    parent_knl = lp.make_kernel(
            "{[i, j, k, l, m]: 0<=i, j, k, l, m<4}",
            """
            [j, l]: z[i, j, k, l, m] = linear_combo2([j, l]: x[i, j, k, l, m],
                                                     [j, l]: y[i, j, k, l, m])
            """,
            kernel_data=[
                lp.GlobalArg(
                    name="x, y",
                    dtype=np.float64,
                    shape=(n, n, n, n, n)),
                ...]
            )

    knl = lp.merge([grandchild_knl, child_knl, parent_knl])

    if inline:
        knl = lp.inline_callable_kernel(knl, "linear_combo2")
        knl = lp.inline_callable_kernel(knl, "linear_combo1")

    evt, (out, ) = knl(queue, x=x, y=y)

    assert (np.linalg.norm(2*x+3*y-out)/(
        np.linalg.norm(2*x+3*y))) < 1e-15


@pytest.mark.parametrize("inline", [False, True])
def test_slices_with_negative_step(ctx_factory, inline):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    n = 4

    x = np.random.rand(n, n, n, n, n)
    y = np.random.rand(n, n, n, n, n)

    child_knl = lp.make_function(
            "{[i, j]:0<=i, j < 4}",
            """
            g[i, j] = 2*e[i, j] + 3*f[i, j]
            """, name="linear_combo")

    parent_knl = lp.make_kernel(
            "{[i, k, m]: 0<=i, k, m<4}",
            """
            z[i, 3:-1:-1, k, :, m] = linear_combo(x[i, :, k, :, m],
                                                   y[i, :, k, :, m])
            """,
            kernel_data=[
                lp.GlobalArg(
                    name="x, y, z",
                    dtype=np.float64,
                    shape=(n, n, n, n, n)),
                ...]
            )

    knl = lp.merge([parent_knl, child_knl])
    if inline:
        knl = lp.inline_callable_kernel(knl, "linear_combo")

    evt, (out, ) = knl(queue, x=x, y=y)

    assert (np.linalg.norm(2*x+3*y-out[:, ::-1, :, :, :])/(
        np.linalg.norm(2*x+3*y))) < 1e-15


@pytest.mark.parametrize("inline", [False, True])
def test_register_knl_with_hw_axes(ctx_factory, inline):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 4

    x_dev = cl.clrandom.rand(queue, (n, n, n, n, n), np.float64)
    y_dev = cl.clrandom.rand(queue, (n, n, n, n, n), np.float64)

    callee_knl = lp.make_function(
            "{[i, j]:0<=i, j < 4}",
            """
            g[i, j] = 2*e[i, j] + 3*f[i, j]
            """, name="linear_combo")

    callee_knl = lp.split_iname(callee_knl, "i", 1, inner_tag="l.0", outer_tag="g.0")

    caller_knl = lp.make_kernel(
            "{[i, j, k, l, m]: 0<=i, j, k, l, m<4}",
            """
            [j, l]: z[i, j, k, l, m] = linear_combo([j, l]: x[i, j, k, l, m],
                                                     [j, l]: y[i, j, k, l, m])
            """, name="caller")
    caller_knl = lp.split_iname(caller_knl, "i", 4, inner_tag="l.1", outer_tag="g.1")

    knl = lp.merge([caller_knl, callee_knl])

    knl = lp.set_options(knl, return_dict=True)

    if inline:
        knl = lp.inline_callable_kernel(knl, "linear_combo")

    evt, out = knl(queue, x=x_dev, y=y_dev)

    x_host = x_dev.get()
    y_host = y_dev.get()

    assert np.linalg.norm(2*x_host+3*y_host-out["z"].get())/np.linalg.norm(
            2*x_host+3*y_host) < 1e-15


@pytest.mark.parametrize("inline", [False, True])
def test_shape_translation_through_sub_array_ref(ctx_factory, inline):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    x1 = cl.clrandom.rand(queue, (3, 2), dtype=np.float64)
    x2 = cl.clrandom.rand(queue, (6, ), dtype=np.float64)
    x3 = cl.clrandom.rand(queue, (6, 6), dtype=np.float64)

    callee1 = lp.make_function(
            "{[i]: 0<=i<6}",
            """
            b[i] = 2*abs(a[i])
            """, name="callee_fn1")

    callee2 = lp.make_function(
            "{[i, j]: 0<=i<3 and 0 <= j < 2}",
            """
            b[i, j] = 3*a[i, j]
            """, name="callee_fn2")

    callee3 = lp.make_function(
            "{[i]: 0<=i<6}",
            """
            b[i] = 5*a[i]
            """, name="callee_fn3")

    knl = lp.make_kernel(
            "{[i, j, k, l]:  0<= i < 6 and 0 <= j < 3 and 0 <= k < 2 and 0<=l<6}",
            """
            [i]: y1[i//2, i%2] = callee_fn1([i]: x1[i//2, i%2])
            [j, k]: y2[2*j+k] = callee_fn2([j, k]: x2[2*j+k])
            [l]: y3[l, l] = callee_fn3([l]: x3[l, l])
            """)

    knl = lp.merge([knl, callee1])
    knl = lp.merge([knl, callee2])
    knl = lp.merge([knl, callee3])

    if inline:
        knl = lp.inline_callable_kernel(knl, "callee_fn1")
        knl = lp.inline_callable_kernel(knl, "callee_fn2")
        knl = lp.inline_callable_kernel(knl, "callee_fn3")

    knl = lp.set_options(knl, write_code=True)
    knl = lp.set_options(knl, return_dict=True)
    evt, out_dict = knl(queue, x1=x1, x2=x2, x3=x3)

    y1 = out_dict["y1"].get()
    y2 = out_dict["y2"].get()
    y3 = out_dict["y3"].get()

    assert (np.linalg.norm(y1-2*x1.get())) < 1e-15
    assert (np.linalg.norm(y2-3*x2.get())) < 1e-15
    assert (np.linalg.norm(np.diag(y3-5*x3.get()))) < 1e-15


def test_multi_arg_array_call(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    import pymbolic.primitives as p
    n = 10
    acc_i = p.Variable("acc_i")
    i = p.Variable("i")
    index = p.Variable("index")
    a_i = p.Subscript(p.Variable("a"), p.Variable("i"))
    argmin_kernel = lp.make_function(
            "{[i]: 0 <= i < n}",
            [
                lp.Assignment(id="init2", assignee=index,
                    expression=0),
                lp.Assignment(id="init1", assignee=acc_i,
                    expression="214748367"),
                lp.Assignment(id="insn", assignee=index,
                    expression=p.If(p.Expression.eq(acc_i, a_i), i, index),
                    depends_on="update"),
                lp.Assignment(id="update", assignee=acc_i,
                    expression=p.Variable("min")(acc_i, a_i),
                    depends_on="init1,init2")],
            [
                lp.GlobalArg("a"),
                lp.GlobalArg("acc_i, index", is_input=False, is_output=True,
                             shape=lp.auto),
                ...],
            name="custom_argmin")

    argmin_kernel = lp.fix_parameters(argmin_kernel, n=n)

    knl = lp.make_kernel(
            "{[i]:0<=i<n}",
            """
            []: min_val[()], []: min_index[()] = custom_argmin([i]:b[i])
            """)

    knl = lp.fix_parameters(knl, n=n)
    knl = lp.set_options(knl, return_dict=True)

    knl = lp.merge([knl, argmin_kernel])
    b = np.random.randn(n)
    evt, out_dict = knl(queue, b=b)
    tol = 1e-15
    from numpy.linalg import norm
    assert norm(out_dict["min_val"] - np.min(b)) < tol
    assert norm(out_dict["min_index"] - np.argmin(b)) < tol


@pytest.mark.parametrize("inline", [False, True])
def test_packing_unpacking(ctx_factory, inline):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    x1 = cl.clrandom.rand(queue, (3, 2), dtype=np.float64)
    x2 = cl.clrandom.rand(queue, (6, ), dtype=np.float64)

    callee1 = lp.make_function(
            "{[i]: 0<=i<6}",
            """
            b[i] = 2*a[i]
            """, name="callee_fn1")

    callee2 = lp.make_function(
            "{[i, j]: 0<=i<2 and 0 <= j < 3}",
            """
            b[i, j] = 3*a[i, j]
            """, name="callee_fn2")

    knl = lp.make_kernel(
            "{[i, j, k]:  0<= i < 3 and 0 <= j < 2 and 0 <= k < 6}",
            """
            [i, j]: y1[i, j] = callee_fn1([i, j]: x1[i, j])
            [k]: y2[k] = callee_fn2([k]: x2[k])
            """)

    knl = lp.merge([knl, callee1])
    knl = lp.merge([knl, callee2])

    knl = lp.pack_and_unpack_args_for_call(knl, "callee_fn1")
    knl = lp.pack_and_unpack_args_for_call(knl, "callee_fn2")

    if inline:
        knl = lp.inline_callable_kernel(knl, "callee_fn1")
        knl = lp.inline_callable_kernel(knl, "callee_fn2")

    knl = lp.set_options(knl, write_code=True)
    knl = lp.set_options(knl, return_dict=True)
    evt, out_dict = knl(queue, x1=x1, x2=x2)

    y1 = out_dict["y1"].get()
    y2 = out_dict["y2"].get()

    assert np.linalg.norm(2*x1.get()-y1)/np.linalg.norm(
            2*x1.get()) < 1e-15
    assert np.linalg.norm(3*x2.get()-y2)/np.linalg.norm(
            3*x2.get()) < 1e-15


@pytest.mark.parametrize("inline", [False, True])
def test_empty_sub_array_refs(ctx_factory, inline):
    # See: https://github.com/OP2/PyOP2/pull/559#discussion_r272208618
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    x = np.random.randn(10)
    y = np.random.randn(10)

    callee = lp.make_function(
            "{[d]:0<=d<1}",
            """
            c[d] = a[d] - b[d]
            """, name="wence_function")

    caller = lp.make_kernel("{[i,k]: 0<=i<10 and 0<=k<1}",
            """
            [k]:z[i+k] = wence_function([k]:x[i+k], [k]:y[i+k])
            """,
            [lp.GlobalArg("x, y", dtype=np.float64, shape=(10, )), ...])

    caller = lp.merge([caller, callee])

    if inline:
        caller = lp.inline_callable_kernel(caller, "wence_function")

    evt, (out, ) = caller(queue, x=x, y=y)
    assert np.allclose(out, x-y)


@pytest.mark.parametrize("inline", [False, True])
def test_array_inputs_to_callee_kernels(ctx_factory, inline):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    n = 2 ** 3

    x = np.random.rand(n, n)
    y = np.random.rand(n, n)

    child_knl = lp.make_function(
            "{[i, j]:0<=i, j < 8}",
            """
            g[i, j] = 2*e[i, j] + 3*f[i, j]
            """, name="linear_combo")

    parent_knl = lp.make_kernel(
            "{:}",
            """
            z[:, :] = linear_combo(x, y)
            """,
            kernel_data=[
                lp.GlobalArg(
                    name="x, y, z",
                    dtype=np.float64,
                    shape=(n, n)),
                ...]
            )

    knl = lp.merge([parent_knl, child_knl])
    if inline:
        knl = lp.inline_callable_kernel(knl, "linear_combo")

    evt, (out, ) = knl(queue, x=x, y=y)

    assert (np.linalg.norm(2*x+3*y-out)/(
        np.linalg.norm(2*x+3*y))) < 1e-15


def test_stride_depending_on_args(ctx_factory):
    ctx = ctx_factory()

    twice = lp.make_function(
            "{[i, j]: 0<=i, j < n}",
            """
            b[i, j] = 2*a[i, j]
            """, [lp.ValueArg("n"), lp.GlobalArg("a"), lp.GlobalArg("b")],
            name="twice")

    thrice = lp.make_function(
            "{[i, j]: 0<=i, j < n}",
            """
            b[i, j] = 3*a[i, j]
            """, [lp.ValueArg("n"), lp.GlobalArg("a", shape=lp.auto),
                lp.GlobalArg("b", shape=lp.auto)],
            name="thrice")

    prog = lp.make_kernel(
            "{[i0,i1,i2,i3,i4,i5,i6,i7]: 0<=i0, i1, i2, i3, i4, i5, i6, i7< N}",
            """
            [i0, i1]: y[i0, i1] = twice(N, [i2, i3]: x[2*i2, i3])
            [i4, i5]: z[i4, i5] = thrice(N, [i6, i7]: x[2*i6+1, i7])
            """, [
                lp.ValueArg("N", dtype=np.int32), lp.GlobalArg("x",
                    shape=lp.auto, dtype=np.float64), ...])

    prog = lp.merge([prog, twice])
    prog = lp.merge([prog, thrice])

    lp.auto_test_vs_ref(prog, ctx, prog, parameters={"N": 4})


def test_unknown_stride_to_callee(ctx_factory):
    ctx = ctx_factory()

    twice = lp.make_function(
            "{[i, j]: 0<=i, j < n}",
            """
            b[i, j] = 2*a[i, j]
            """, [lp.ValueArg("n"), lp.GlobalArg("a"), lp.GlobalArg("b")],
            name="twice")

    prog = lp.make_kernel(
            "{[i,i0,i1,i2,i3]: 0<=i0, i1, i2, i3< N and 0<=i<Nvar}",
            """
            [i0, i1]: y[i0, i1, i] = twice(N, [i2, i3]: x[2*i2, i3, i])
            """, [
                lp.ValueArg("N", dtype=np.int32), lp.ValueArg("Nvar",
                    dtype=np.int32), lp.GlobalArg("x", shape=lp.auto,
                        dtype=np.float64), ...])

    prog = lp.merge([prog, twice])

    lp.auto_test_vs_ref(prog, ctx, prog, parameters={"N": 4, "Nvar": 5})


def test_argument_matching_for_inplace_update(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    twice = lp.make_function(
            "{[i]: 0<=i<10}",
            """
            x[i] = 2*x[i]
            """, name="twice")

    knl = lp.make_kernel(
            "{:}",
            """
            x[:] = twice(x[:])
            """, [lp.GlobalArg("x", shape=(10,), dtype=np.float64)])

    knl = lp.merge([knl, twice])

    x = np.random.randn(10)
    evt, (out, ) = knl(queue, x=np.copy(x))

    assert np.allclose(2*x, out)


def test_non_zero_start_in_subarray_ref(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    twice = lp.make_function(
            "{[i]: 0<=i<10}",
            """
            b[i] = 2*a[i]
            """, name="twice")

    knl = lp.make_kernel(
            "{[i, j]: -5<=i<5 and 0<=j<10}",
            """
            [i]:y[i+5] = twice([j]: x[j])
            """, [lp.GlobalArg("x, y", shape=(10,), dtype=np.float64)])

    knl = lp.merge([knl, twice])

    x = np.random.randn(10)
    evt, (out, ) = knl(queue, x=np.copy(x))

    assert np.allclose(2*x, out)


def test_incomplete_entrypoint_raises_type_inf_failure():
    from loopy.diagnostic import LoopyError

    twice = lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            y[i] = 2*x[i]
            """, name="dosify")

    quadr = lp.make_kernel(
            "{:}",
            """
            y[:] = dosify(x[:])
            y[:] = dosify(y[:])
            """, [lp.GlobalArg("x,y", shape=(10,))], name="cuatroify",
            seq_dependencies=True)

    prog = lp.merge([quadr, twice])

    with pytest.raises(LoopyError):
        # 'twice' is also registered as an entrypoint but provided args aren't
        # enough to infer the types
        lp.generate_code_v2(prog)


def test_callees_with_gbarriers_are_inlined(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    ones_and_zeros = lp.make_function(
            "{[i, j]: 0<=i<6 and 0<=j<3}",
            """
            x[i] = 0.0f
            ...gbarrier
            x[j] = 1.0f
            """,
            seq_dependencies=True,
            name="ones_and_zeros")

    t_unit = lp.make_kernel(
            "{ : }",
            """
            y[:] = ones_and_zeros()
            """, [lp.GlobalArg("y", shape=6, dtype=None)])

    t_unit = lp.merge([t_unit, ones_and_zeros])
    evt, (out,) = t_unit(queue)

    expected_out = np.array([1, 1, 1, 0, 0, 0]).astype(np.float32)

    assert (expected_out == out.get()).all()


def test_callees_with_gbarriers_are_inlined_with_nested_calls(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    ones_and_zeros = lp.make_function(
            "{[i, j]: 0<=i<6 and 0<=j<3}",
            """
            x[i] = 0.0f
            ...gbarrier
            x[j] = 1.0f
            """,
            seq_dependencies=True,
            name="ones_and_zeros")

    dummy_ones_and_zeros = lp.make_function(
            "{[i]: 0<=i<6}",
            """
            [i]: y[i] = ones_and_zeros()
            """,
            name="dummy_ones_and_zeros")

    t_unit = lp.make_kernel(
            "{ : }",
            """
            y[:] = dummy_ones_and_zeros()
            """, [lp.GlobalArg("y", shape=6, dtype=None)])

    t_unit = lp.merge([t_unit, dummy_ones_and_zeros, ones_and_zeros])
    evt, (out,) = t_unit(queue)

    expected_out = np.array([1, 1, 1, 0, 0, 0]).astype(np.float32)

    assert (expected_out == out.get()).all()


def test_inlining_with_indirections(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    ones_and_zeros = lp.make_function(
            "{[i, j]: 0<=i<6 and 0<=j<3}",
            """
            x[i] = 0.0f
            ...gbarrier
            x[map[j]] = 1.0f
            """,
            seq_dependencies=True,
            name="ones_and_zeros")

    t_unit = lp.make_kernel(
            "{ : }",
            """
            y[:] = ones_and_zeros(mymap[:])
            """, [lp.GlobalArg("y", shape=6, dtype=None),
                  lp.GlobalArg("mymap", dtype=np.int32, shape=3)])

    t_unit = lp.merge([t_unit, ones_and_zeros])
    t_unit = lp.inline_callable_kernel(t_unit, "ones_and_zeros")

    map_in = np.arange(3).astype(np.int32)

    evt, (out, ) = t_unit(queue, mymap=map_in)

    expected_out = np.array([1, 1, 1, 0, 0, 0]).astype(np.float32)
    assert (expected_out == out).all()


def test_inlining_with_callee_domain_param(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    fill2 = lp.make_function(
            "{[i]: 0<=i<n}",
            """
            y[i] = 2.0
            """,
            name="fill2")

    caller = lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            [i]: res[i] = fill2(10)
            """)

    caller = lp.merge([caller, fill2])
    caller = lp.inline_callable_kernel(caller, "fill2")
    evt, (out, ) = caller(queue)

    assert (out == 2).all()


def test_double_resolving():
    from loopy.translation_unit import resolve_callables
    from loopy.kernel import KernelState
    from loopy.symbolic import ResolvedFunction

    knl = lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            y[i] = sin(x[i])
            """,
            [
                lp.GlobalArg("x", dtype=float, shape=lp.auto),
                ...],
            name="foo"
            )

    knl = resolve_callables(knl)
    knl = knl.with_kernel(knl["foo"].copy(state=KernelState.INITIAL))
    knl = resolve_callables(knl)

    assert "sin" in knl.callables_table
    assert isinstance(knl["foo"].instructions[0].expression.function,
                      ResolvedFunction)


@pytest.mark.parametrize("inline", [False, True])
def test_passing_and_getting_scalar_in_clbl_knl(ctx_factory, inline):
    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)

    call_sin = lp.make_function(
        "{:}",
        """
        y = sin(x)
        """, name="call_sin")

    knl = lp.make_kernel(
        "{:}",
        """
        []: real_y[()] = call_sin(real_x)
        """)

    knl = lp.merge([knl, call_sin])
    knl = lp.set_options(knl, write_code=True)
    if inline:
        knl = lp.inline_callable_kernel(knl, "call_sin")

    evt, (out,) = knl(cq, real_x=np.asarray(3.0, dtype=float))


@pytest.mark.parametrize("inline", [False, True])
def test_passing_scalar_as_indexed_subcript_in_clbl_knl(ctx_factory, inline):
    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)
    x_in = np.random.rand()

    twice = lp.make_function(
        "{ : }",
        """
        y = 2*x
        """,
        name="twice")

    knl = lp.make_kernel(
        "{ : }",
        """
        []: Y[0, 0] = twice(X)
        """)

    knl = lp.add_dtypes(knl, {"X": np.float64})
    knl = lp.merge([knl, twice])

    if inline:
        knl = lp.inline_callable_kernel(knl, "twice")

    evt, (out,) = knl(cq, X=x_in)

    np.testing.assert_allclose(out.get(), 2*x_in)


def test_symbol_mangler_in_call(ctx_factory):
    from library_for_test import (symbol_x,
                                  preamble_for_x)
    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{:}",
        """
        y = sin(X)
        """,
        [lp.GlobalArg("y", shape=lp.auto)])

    knl = lp.register_symbol_manglers(knl, [symbol_x])

    knl = lp.register_preamble_generators(knl, [preamble_for_x])

    evt, (out,) = knl(cq)
    np.testing.assert_allclose(out.get(), np.sin(10))


@pytest.mark.parametrize("which", ["max", "min"])
def test_int_max_min_c_target(ctx_factory, which):
    from numpy.random import default_rng
    from pymbolic import parse
    rng = default_rng()

    n = 100
    arr1 = rng.integers(-100, 100, n)
    arr2 = rng.integers(-100, 100, n)
    np_func = getattr(np, f"{which}imum")

    knl = lp.make_kernel(
        "{[i]: 0<=i<100}",
        [lp.Assignment(parse("out[i]"),
                       parse(f"{which}(arr1[i], arr2[i])"))],
        target=lp.ExecutableCTarget())

    _, (out,) = knl(arr1=arr1, arr2=arr2)
    np.testing.assert_allclose(np_func(arr1, arr2), out)


def test_valueargs_being_mapped_in_inling(ctx_factory):
    doublify = lp.make_function(
            "{[i]: 0<=i<n}",
            """
            y[i] = n*x[i]
            """,
            [lp.ValueArg("n", dtype=np.int32), ...],
            name="doublify",
            )

    knl = lp.make_kernel(
            "{[i, j]: 0<=i, j<10}",
            """
            [i]: bar[i] = doublify(10, [j]: foo[j])
            """,
            [
                lp.GlobalArg("foo", dtype=float, shape=lp.auto),
                ...],
            )
    knl = lp.merge([knl, doublify])
    knl = lp.inline_callable_kernel(knl, "doublify")

    lp.auto_test_vs_ref(knl, ctx_factory(), knl)


@pytest.mark.parametrize("inline", [True, False])
def test_unused_hw_axes_in_callee(ctx_factory, inline):
    ctx = ctx_factory()

    twice = lp.make_function(
            "{[i]: 0<=i<10}",
            """
            y[i] = 2*x[i]
            """, name="twice")

    knl = lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            y[:, i] = twice(x[:, i])
            """, [lp.GlobalArg("x", shape=(10, 10), dtype=float),
                lp.GlobalArg("y", shape=(10, 10))],
            name="outer")

    twice = lp.tag_inames(twice, {"i": "l.1"})
    knl = lp.tag_inames(knl, {"i": "l.0"})
    knl = lp.merge([knl, twice])

    if inline:
        knl = lp.inline_callable_kernel(knl, "twice")

    lp.auto_test_vs_ref(knl, ctx, knl)


@pytest.mark.parametrize("inline", [True, False])
def test_double_hw_axes_used_in_knl_call(inline):
    from loopy.diagnostic import LoopyError

    twice = lp.make_function(
            "{[i]: 0<=i<10}",
            """
            y[i] = 2*x[i]
            """, name="twice")

    knl = lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            y[:, i] = twice(x[:, i])
            """, [lp.GlobalArg("x", shape=(10, 10), dtype=float),
                lp.GlobalArg("y", shape=(10, 10))],
            name="outer")

    twice = lp.tag_inames(twice, {"i": "l.0"})
    knl = lp.tag_inames(knl, {"i": "l.0"})
    knl = lp.merge([knl, twice])

    if inline:
        knl = lp.inline_callable_kernel(knl, "twice")

    with pytest.raises(LoopyError):
        lp.generate_code_v2(knl)


@pytest.mark.parametrize("inline", [True, False])
def test_kc_with_floor_div_in_expr(ctx_factory, inline):
    # See https://github.com/inducer/loopy/issues/366
    import loopy as lp

    ctx = ctx_factory()
    callee = lp.make_function(
            "{[i]: 0<=i<10}",
            """
            x[i] = 2*x[i]
            """, name="callee_with_update")

    knl = lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            [i]: x[2*(i//2) + (i%2)] = callee_with_update([i]: x[i])
            """)

    knl = lp.merge([knl, callee])

    if inline:
        knl = lp.inline_callable_kernel(knl, "callee_with_update")

    lp.auto_test_vs_ref(knl, ctx, knl)


@pytest.mark.parametrize("start", [5, 6, 7])
@pytest.mark.parametrize("inline", [True, False])
def test_non1_step_slices(ctx_factory, start, inline):
    # See https://github.com/inducer/loopy/pull/222#discussion_r645905188

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    callee = lp.make_function(
            "{[i]: 0<=i<n}",
            """
            y[i] = i**2
            """,
            [lp.ValueArg("n"), ...],
            name="squared_arange")

    t_unit = lp.make_kernel(
            "{[i_init, j_init]: 0<=i_init, j_init<40}",
            f"""
            X[i_init] = 42
            X[{start}:40:3] = squared_arange({len(range(start, 40, 3))})

            Y[j_init] = 1729
            Y[39:{start}:-3] = squared_arange({len(range(39, start, -3))})
            """,
            [lp.GlobalArg("X,Y", shape=40)],
            seq_dependencies=True)

    expected_out1 = 42*np.ones(40, dtype=np.int64)
    expected_out1[start:40:3] = np.arange(len(range(start, 40, 3)))**2

    expected_out2 = 1729*np.ones(40, dtype=np.int64)
    expected_out2[39:start:-3] = np.arange(len(range(39, start, -3)))**2

    t_unit = lp.merge([t_unit, callee])

    t_unit = lp.set_options(t_unit, return_dict=True)

    if inline:
        t_unit = lp.inline_callable_kernel(t_unit, "squared_arange")

    evt, out_dict = t_unit(cq)

    np.testing.assert_allclose(out_dict["X"].get(), expected_out1)
    np.testing.assert_allclose(out_dict["Y"].get(), expected_out2)


def test_check_bounds_with_caller_assumptions(ctx_factory):
    import islpy as isl
    from loopy.diagnostic import LoopyIndexError

    arange = lp.make_function(
        "{[i]: 0<=i<n}",
        """
        y[i] = i
        """, name="arange")

    knl = lp.make_kernel(
        "{[i]: 0<=i<20}",
        """
        [i]: Y[i] = arange(N)
        """,
        [lp.GlobalArg("Y", shape=(20,)), lp.ValueArg("N", dtype=np.int32)],
        name="epoint")

    knl = lp.merge([knl, arange])

    with pytest.raises(LoopyIndexError):
        lp.generate_code_v2(knl)

    knl = knl.with_kernel(lp.assume(knl.default_entrypoint,
                                    isl.BasicSet("[N] -> { : N <= 20}")))

    lp.auto_test_vs_ref(knl, ctx_factory(),
                        parameters={"N": 15})


def test_callee_with_auto_offset(ctx_factory):

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    arange = lp.make_function(
        "{[i]: 0<=i<7}",
        """
        y[i] = 2*y[i]
        """,
        [lp.GlobalArg("y",  offset=lp.auto)],
        name="dosify")

    knl = lp.make_kernel(
        "{[i]: 0<=i<7}",
        """
        [i]: y[i] = dosify([i]: y[i])
        """,
        [lp.GlobalArg("y", offset=3, shape=10)])

    knl = lp.merge([knl, arange])

    y = np.arange(10)
    knl(queue, y=y)
    np.testing.assert_allclose(y[:3], np.arange(3))
    np.testing.assert_allclose(y[3:], 2*np.arange(3, 10))


def test_callee_with_parameter_and_grid(ctx_factory):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    callee = lp.make_function(
        "{[i]: 0<=i<n}",
        """
        y[i] = i
        """, name="arange")

    knl = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
        [i]: y[i] = arange(10)
        """)

    knl = lp.merge([callee, knl])
    knl = lp.split_iname(knl, "i", 2,
                         outer_tag="g.0", within="in_kernel:arange")

    evt, (out,) = knl(cq)
    np.testing.assert_allclose(out.get(), np.arange(10))


@pytest.mark.parametrize("inline", [True, False])
def test_inlining_does_not_require_barrier(inline):

    # {{{ provided by isuruf in https://github.com/inducer/loopy/issues/578

    fft_knl = lp.make_function(
        [
            "{ [i] : 0 <= i <= 1322 }",
            "{ [ifft_0] : 0 <= ifft_0 <= 188 }",
            "{ [iN1_0] : 0 <= iN1_0 <= 6 }",
            "{ [iN1_sum_0] : 0 <= iN1_sum_0 <= 6 }",
            "{ [iN2_0] : iN2_0 = 0 }",
            "{ [i_0] : 0 <= i_0 <= 1322 }",
            "{ [i2_0] : 0 <= i2_0 <= 1322 }",
            "{ [ifft_1] : 0 <= ifft_1 <= 26 }",
            "{ [iN1_1] : 0 <= iN1_1 <= 6 }",
            "{ [iN1_sum_1] : 0 <= iN1_sum_1 <= 6 }",
            "{ [iN2_1] : 0 <= iN2_1 <= 6 }",
            "{ [i_1] : 0 <= i_1 <= 1322 }",
            "{ [i2_1] : 0 <= i2_1 <= 1322 }",
            "{ [ifft_2] : 0 <= ifft_2 <= 8 }",
            "{ [iN1_2] : 0 <= iN1_2 <= 2 }",
            "{ [iN1_sum_2] : 0 <= iN1_sum_2 <= 2 }",
            "{ [iN2_2] : 0 <= iN2_2 <= 48 }",
            "{ [i_2] : 0 <= i_2 <= 1322 }",
            "{ [i2_2] : 0 <= i2_2 <= 1322 }",
            "{ [ifft_3] : 0 <= ifft_3 <= 2 }",
            "{ [iN1_3] : 0 <= iN1_3 <= 2 }",
            "{ [iN1_sum_3] : 0 <= iN1_sum_3 <= 2 }",
            "{ [iN2_3] : 0 <= iN2_3 <= 146 }",
            "{ [i_3] : 0 <= i_3 <= 1322 }",
            "{ [i2_3] : 0 <= i2_3 <= 1322 }",
            "{ [ifft_4] : ifft_4 = 0 }",
            "{ [iN1_4] : 0 <= iN1_4 <= 2 }",
            "{ [iN1_sum_4] : 0 <= iN1_sum_4 <= 2 }",
            "{ [iN2_4] : 0 <= iN2_4 <= 440 }",
            "{ [i_4] : 0 <= i_4 <= 1322 }",
            "{ [i2_4] : 0 <= i2_4 <= 1322 }",
        ],
        """
        exp_table[i] = exp((-0.004749195243521985j)*i) {id=exp_table}
        temp[i_0] = x[i_0] {id=copy_0, dep=exp_table}
        x[i2_0] = 0 {id=reset_0, dep=copy_0}
        table_idx_0 = 189*iN1_sum_0*(iN2_0 + iN1_0) {id=idx_0, dep=reset_0}
        exp_0 = exp_table[table_idx_0 % 1323] {id=exp_0, dep=idx_0:exp_table}
        x[ifft_0 + 189*(iN1_0 + iN2_0)] = x[ifft_0 + 189*(iN1_0 + iN2_0)] + exp_0*temp[ifft_0 + 189*(iN2_0*7 + iN1_sum_0)] {id=update_0, dep=exp_0}
        temp[i_1] = x[i_1] {id=copy_1, dep=update_0}
        x[i2_1] = 0 {id=reset_1, dep=copy_1}
        table_idx_1 = 27*iN1_sum_1*(iN2_1 + 7*iN1_1) {id=idx_1, dep=reset_1}
        exp_1 = exp_table[table_idx_1 % 1323] {id=exp_1, dep=idx_1:exp_table}
        x[ifft_1 + 27*(iN1_1*7 + iN2_1)] = x[ifft_1 + 27*(iN1_1*7 + iN2_1)] + exp_1*temp[ifft_1 + 27*(iN2_1*7 + iN1_sum_1)] {id=update_1, dep=exp_1}
        temp[i_2] = x[i_2] {id=copy_2, dep=update_1}
        x[i2_2] = 0 {id=reset_2, dep=copy_2}
        table_idx_2 = 9*iN1_sum_2*(iN2_2 + 49*iN1_2) {id=idx_2, dep=reset_2}
        exp_2 = exp_table[table_idx_2 % 1323] {id=exp_2, dep=exp_table:idx_2}
        x[ifft_2 + 9*(iN1_2*49 + iN2_2)] = x[ifft_2 + 9*(iN1_2*49 + iN2_2)] + exp_2*temp[ifft_2 + 9*(iN2_2*3 + iN1_sum_2)] {id=update_2, dep=exp_2}
        temp[i_3] = x[i_3] {id=copy_3, dep=update_2}
        x[i2_3] = 0 {id=reset_3, dep=copy_3}
        table_idx_3 = 3*iN1_sum_3*(iN2_3 + 147*iN1_3) {id=idx_3, dep=reset_3}
        exp_3 = exp_table[table_idx_3 % 1323] {id=exp_3, dep=exp_table:idx_3}
        x[ifft_3 + 3*(iN1_3*147 + iN2_3)] = x[ifft_3 + 3*(iN1_3*147 + iN2_3)] + exp_3*temp[ifft_3 + 3*(iN2_3*3 + iN1_sum_3)] {id=update_3, dep=exp_3}
        temp[i_4] = x[i_4] {id=copy_4, dep=update_3}
        x[i2_4] = 0 {id=reset_4, dep=copy_4}
        table_idx_4 = iN1_sum_4*(iN2_4 + 441*iN1_4) {id=idx_4, dep=reset_4}
        exp_4 = exp_table[table_idx_4 % 1323] {id=exp_4, dep=exp_table:idx_4}
        x[ifft_4 + iN1_4*441 + iN2_4] = x[ifft_4 + iN1_4*441 + iN2_4] + exp_4*temp[ifft_4 + iN2_4*3 + iN1_sum_4] {id=update_4, dep=exp_4}
        """,  # noqa: E501
        [
            lp.GlobalArg(
                name="x", dtype=np.complex128,
                shape=(1323,), for_atomic=False),
            lp.TemporaryVariable(
                name="exp_table",
                dtype=np.complex128,
                shape=(1323,), for_atomic=False,
                address_space=lp.auto,
                read_only=False,
                ),
            lp.TemporaryVariable(
                name="temp",
                dtype=np.complex128,
                shape=(1323,), for_atomic=False,
                address_space=lp.auto,
                read_only=False,
                ),
            lp.TemporaryVariable(
                name="table_idx_0",
                dtype=np.uint32,
                shape=(), for_atomic=False,
                address_space=lp.auto,
                read_only=False,
                ),
            lp.TemporaryVariable(
                name="exp_0",
                dtype=np.complex128,
                shape=(), for_atomic=False,
                address_space=lp.auto,
                read_only=False,
                ),
            lp.TemporaryVariable(
                name="table_idx_1",
                dtype=np.uint32,
                shape=(), for_atomic=False,
                address_space=lp.auto,
                read_only=False,
                ),
            lp.TemporaryVariable(
                name="exp_1",
                dtype=np.complex128,
                shape=(), for_atomic=False,
                address_space=lp.auto,
                read_only=False,
                ),
            lp.TemporaryVariable(
                name="table_idx_2",
                dtype=np.uint32,
                shape=(), for_atomic=False,
                address_space=lp.auto,
                read_only=False,
                ),
            lp.TemporaryVariable(
                name="exp_2",
                dtype=np.complex128,
                shape=(), for_atomic=False,
                address_space=lp.auto,
                read_only=False,
                ),
            lp.TemporaryVariable(
                name="table_idx_3",
                dtype=np.uint32,
                shape=(), for_atomic=False,
                address_space=lp.auto,
                read_only=False,
                ),
            lp.TemporaryVariable(
                name="exp_3",
                dtype=np.complex128,
                shape=(), for_atomic=False,
                address_space=lp.auto,
                read_only=False,
                ),
            lp.TemporaryVariable(
                name="table_idx_4",
                dtype=np.uint32,
                shape=(), for_atomic=False,
                address_space=lp.auto,
                read_only=False,
                ),
            lp.TemporaryVariable(
                name="exp_4",
                dtype=np.complex128,
                shape=(), for_atomic=False,
                address_space=lp.auto,
                read_only=False,
                ),
            ], name="fft")
    loopy_kernel_knl = lp.make_kernel(
        [
            "{ [i] : 0 <= i <= 1322 }",
            "[m] -> { [j_outer, j_inner]"
            " : j_inner >= 0 and -64j_outer <= j_inner <= 63"
            " and j_inner < m - 64j_outer }",
        ],
        """
        [i]: y[j_inner + j_outer*64, i] = fft([i]: y[j_inner + j_outer*64, i])
        """, [
            lp.ValueArg(
                name="m",
                dtype=None),
            lp.GlobalArg(
                name="y", dtype=None,
                shape=lp.auto),
            ],
        iname_slab_increments={"j_outer": (0, 0)})

    # }}}

    loopy_kernel_knl = lp.tag_inames(loopy_kernel_knl, "j_outer:g.0")
    t_unit = lp.merge([fft_knl, loopy_kernel_knl])
    if inline:
        t_unit = lp.inline_callable_kernel(t_unit, "fft")

    # generate code to ensure that we don't emit spurious missing barrier
    print(lp.generate_code_v2(t_unit).device_code())


def test_inlining_w_zero_stride_callee_args(ctx_factory):
    # See https://github.com/inducer/loopy/issues/594
    ctx = ctx_factory()

    times_two = lp.make_function(
        "{[j1, j2, j3]: 0<=j1,j2,j3<1}",
        """
        out[j1, j2, j3] = 2 * inp[j1, j2, j3]
        """,
        name="times_two")

    knl = lp.make_kernel(
        "{[i1, i2, i3]: 0<=i1,i2,i3<1}",
        """
        tmp[0, 0, 0] = 2.0
        [i1,i2,i3]: y[i1,i2,i3] = times_two([i1,i2,i3]: tmp[0,i2,i3])
        """,
        [lp.TemporaryVariable("tmp", strides=(0, 1, 1)), ...])

    ref_knl = lp.merge([knl, times_two])
    knl = lp.inline_callable_kernel(ref_knl, "times_two")
    lp.auto_test_vs_ref(ref_knl, ctx, knl)


@pytest.mark.parametrize("inline", [True, False])
def test_call_kernel_w_preds(ctx_factory, inline):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    twice = lp.make_function(
        "{ [i] : 0<=i<10 }",
        """
        x[i] = 2*x[i]
        """, name="twice")

    knl = lp.make_kernel(
        "{[i] : 0<=i<10}",
        """
        for i
            if i >= 5
                x[i, :] = twice(x[i, :])
            end
        end
        """,
        [lp.GlobalArg("x",
                      shape=(10, 10),),
         ...])

    knl = lp.merge([knl, twice])

    if inline:
        knl = lp.inline_callable_kernel(knl, "twice")

    evt, (out,) = knl(cq, x=np.ones((10, 10)))

    np.testing.assert_allclose(out[:5], 1)
    np.testing.assert_allclose(out[5:], 2)


# {{{ test_inlining_does_not_lose_preambles

class OurPrintfDefiner(ImmutableRecord):
    def __call__(self, *args, **kwargs):
        return (("42", r"#define OURPRINTF printf"),)


@pytest.mark.parametrize("inline", [True, False])
def test_inlining_does_not_lose_preambles(ctx_factory, inline):
    # loopy.git<=95cc206 would miscompile this

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    callee = lp.make_function(
        "{ : }",
        [
            lp.CInstruction("",
                            r"""
                            MYPRINTF("Foo bar!\n");
                            """, assignees=())
        ],
        preambles=[("1729", r"#define MYPRINTF OURPRINTF")],
        preamble_generators=[OurPrintfDefiner()],
        name="print_foo")

    caller = lp.make_kernel(
        "{ : }",
        """
        print_foo()
        """,
        name="print_foo_caller")

    knl = lp.merge([caller, callee])
    knl = lp.set_options(knl, "write_code")

    if inline:
        knl = lp.inline_callable_kernel(knl, "print_foo")

    # run to make sure there is no compilation error
    knl(cq)

# }}}


@pytest.mark.parametrize("inline", [True, False])
def test_c_instruction_in_callee(ctx_factory, inline):
    from loopy.symbolic import parse

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    n = np.random.randint(3, 8)

    knl = lp.make_function(
        "{[i]: 0<=i<10}",
        [lp.CInstruction(iname_exprs=("i", "i"),
                         code="break;",
                         predicates={parse("i >= n")},
                         id="break",),
         lp.Assignment("out_callee", "i", depends_on=frozenset(["break"]))
         ],
        [lp.ValueArg("n", dtype="int32"), ...],
        name="circuit_breaker")

    t_unit = lp.make_kernel(
        "{ : }",
        """
        []: result[0] = circuit_breaker(N)
        """,
        [lp.ValueArg("N", dtype="int32"), ...],)

    t_unit = lp.merge([t_unit, knl])

    if inline:
        t_unit = lp.inline_callable_kernel(t_unit, "circuit_breaker")

    _, (out,) = t_unit(cq, N=n)

    assert out.get() == (n-1)


def test_global_temp_var_with_base_storage(ctx_factory):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{[i, i2] : 0<=i,i2<3}",
        """
        a[i] = 5
        b[i] = a[i] + 1
        ... gbarrier
        c[i2] = b[i2] + 2
        d[i2] = c[i2] + 3
        """, [
            lp.TemporaryVariable("a", dtype=np.int32, shape=(3,),
                address_space=lp.AddressSpace.GLOBAL, base_storage="bs"),
            lp.TemporaryVariable("b", dtype=np.int32, shape=(3,),
                address_space=lp.AddressSpace.GLOBAL, base_storage="bs"),
            lp.TemporaryVariable("c", dtype=np.int32, shape=(3,),
                address_space=lp.AddressSpace.GLOBAL, base_storage="bs"),
            ...
        ],
        seq_dependencies=True)

    knl = lp.allocate_temporaries_for_base_storage(knl, aliased=False)

    cl_prg = cl.Program(ctx, lp.generate_code_v2(knl).device_code()).build()
    assert [knl.num_args for knl in cl_prg.all_kernels()] == [1, 2]

    _evt, (d,) = knl(cq)
    assert (d.get() == 5 + 1 + 2 + 3).all()


def test_inline_deps(ctx_factory):
    # https://github.com/inducer/loopy/issues/564

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    child_knl = lp.make_function(
        "{[i]:0<=i < 4}",
        """
        a[i] = b[i]
        """, name="func")

    parent_knl = lp.make_kernel(
        "{[i]: 0<=i<4}",
        """
        <> b[i] = i  {id=init,dup=i}
        [i]: a[i] = func([i]: b[i]) {dep=init}
        """)

    prg = lp.merge([parent_knl, child_knl])
    inlined = lp.inline_callable_kernel(prg, "func")

    from loopy.kernel.creation import apply_single_writer_depencency_heuristic
    apply_single_writer_depencency_heuristic(inlined, error_if_used=True)

    _evt, (a_dev,) = inlined(cq)

    assert np.array_equal(a_dev.get(), np.arange(4))


def test_inline_stride():
    # https://github.com/inducer/loopy/issues/728
    child_knl = lp.make_function(
            [],
            """
            g[0] = 2*e[0] + 3*f[0] {id=a}
            g[1] = 2*e[1] + 3*f[1] {dep=a}
            """, name="linear_combo")
    parent_knl = lp.make_kernel(
            ["{[j]:0<=j<n}", "{[i]:0<=i<n}"],
            """
            [i]: z[i, j] = linear_combo([i]: x[i, j], [i]: y[i,j])
            """,
            kernel_data=[
                lp.GlobalArg(
                    name="x, y, z",
                    dtype=np.float64,
                    shape=("n", "n")),
                ...],
            assumptions="n>=2",
            )
    knl = lp.merge([parent_knl, child_knl])
    knl = lp.inline_callable_kernel(knl, "linear_combo")
    lp.generate_code_v2(knl).device_code()


def test_inline_predicate():
    # https://github.com/inducer/loopy/issues/739
    twice = lp.make_function(
        "{[i]: 0<=i<10}",
        """
        y[i] = 2*x[i]
        """,
        name="twice")

    knl = lp.make_kernel(
        "{[i,j]: 0<=i<10 and 0<=j<10}",
        """
        <> y[i] = 0  {id=y,dup=i}
        <> x[i] = 1  {id=x,dup=i}
        for j
            <> a = (j%2 == 0)
            y[i] = i                         {dep=y,if=a,id=y2}
            [i]: z[i, j] = twice([i]: x[i])  {if=a,dep=y}
        end
        """)

    knl = lp.add_dtypes(knl, {"z": np.float64})
    knl = lp.merge([knl, twice])
    knl = lp.inline_callable_kernel(knl, "twice")
    code = lp.generate_code_v2(knl).device_code()
    assert code.count("if (a)") == 1


def test_subarray_ref_with_repeated_indices(ctx_factory):
    # https://github.com/inducer/loopy/pull/735#discussion_r1071690388

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)
    child_knl = lp.make_function(
            ["{[i]: 0<=i<10}"],
            """
            g[i] = 1
            """, name="ones")

    parent_knl = lp.make_kernel(
            ["{[i]:0<=i<10}", "{[j]: 0<=j<10}"],
            """
            z[i, j] = 0                {id = a}
            [i]: z[i, i] = ones()  {dep=a,dup=i}
            """,
            kernel_data=[
                lp.GlobalArg(
                    name="z",
                    dtype=np.float64,
                    is_input=False,
                    shape=(10, 10)),
                ...],
            )
    knl = lp.merge([parent_knl, child_knl])
    knl = lp.inline_callable_kernel(knl, "ones")
    evt, (z_dev,) = knl(cq)
    assert np.allclose(z_dev.get(), np.eye(10))


def test_inline_constant_access():
    child_knl = lp.make_function(
            [],
            """
            g[0] = 2*e[0] + 3*f[0] {id=a}
            g[1] = 2*e[1] + 3*f[1] {dep=a}
            """, name="linear_combo")
    parent_knl = lp.make_kernel(
            ["{[j]:0<=j<n}", "{[i]:0<=i<n}"],
            """
            [i]: z[i, j] = linear_combo([i]: x[i, j], [i]: y[i,j])
            """,
            kernel_data=[
                lp.GlobalArg(
                    name="x, y, z",
                    dtype=np.float64,
                    shape=(3, "n")),
                ...],
            )
    knl = lp.merge([parent_knl, child_knl])
    knl = lp.inline_callable_kernel(knl, "linear_combo")
    knl = lp.tag_array_axes(knl, ["x", "y", "z"], "sep,C")
    lp.generate_code_v2(knl).device_code()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

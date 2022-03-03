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

    knl = lp.set_options(knl, "return_dict")

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

    knl = lp.set_options(knl, "write_cl")
    knl = lp.set_options(knl, "return_dict")
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
    assert(norm(out_dict["min_val"] - np.min(b)) < tol)
    assert(norm(out_dict["min_index"] - np.argmin(b)) < tol)


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

    knl = lp.set_options(knl, "write_cl")
    knl = lp.set_options(knl, "return_dict")
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
    knl = lp.set_options(knl, "write_cl")
    if inline:
        knl = lp.inline_callable_kernel(knl, "call_sin")

    evt, (out,) = knl(cq, real_x=np.asarray(3.0, dtype=float))


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

    t_unit = lp.set_options(t_unit, "return_dict")

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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

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

import six
from six.moves import range

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


def test_complicated_subst(ctx_factory):
    #ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            """
                f(x) := x*a[x]
                g(x) := 12 + f(x)
                h(x) := 1 + g(x) + 20*g$two(x)

                a[i] = h$one(i) * h$two(i)
                """)

    knl = lp.expand_subst(knl, "... > id:h and tag:two > id:g and tag:two")

    print(knl)

    sr_keys = list(knl.substitutions.keys())
    for letter, how_many in [
            ("f", 1),
            ("g", 1),
            ("h", 2)
            ]:
        substs_with_letter = sum(1 for k in sr_keys if k.startswith(letter))
        assert substs_with_letter == how_many


def test_type_inference_no_artificial_doubles(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            """
                <> bb = a[i] - b[i]
                c[i] = bb
                """,
            [
                lp.GlobalArg("a", np.float32, shape=("n",)),
                lp.GlobalArg("b", np.float32, shape=("n",)),
                lp.GlobalArg("c", np.float32, shape=("n",)),
                lp.ValueArg("n", np.int32),
                ],
            assumptions="n>=1")

    knl = lp.preprocess_kernel(knl, ctx.devices[0])
    for k in lp.generate_loop_schedules(knl):
        code = lp.generate_code(k)
        assert "double" not in code


def test_type_inference_with_type_dependencies():
    knl = lp.make_kernel(
            "{[i]: i=0}",
            """
            <>a = 99
            a = a + 1
            <>b = 0
            <>c = 1
            b = b + c + 1.0
            c = b + c
            <>d = b + 2 + 1j
            """,
            "...")
    knl = lp.infer_unknown_types(knl)

    from loopy.types import to_loopy_type
    assert knl.temporary_variables["a"].dtype == to_loopy_type(np.int32)
    assert knl.temporary_variables["b"].dtype == to_loopy_type(np.float32)
    assert knl.temporary_variables["c"].dtype == to_loopy_type(np.float32)
    assert knl.temporary_variables["d"].dtype == to_loopy_type(np.complex128)


def test_sized_and_complex_literals(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            """
                <> aa = 5jf
                <> bb = 5j
                a[i] = imag(aa)
                b[i] = imag(bb)
                c[i] = 5f
                """,
            [
                lp.GlobalArg("a", np.float32, shape=("n",)),
                lp.GlobalArg("b", np.float32, shape=("n",)),
                lp.GlobalArg("c", np.float32, shape=("n",)),
                lp.ValueArg("n", np.int32),
                ],
            assumptions="n>=1")

    lp.auto_test_vs_ref(knl, ctx, knl, parameters=dict(n=5))


def test_simple_side_effect(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<100}",
            """
                a[i] = a[i] + 1
                """,
            [lp.GlobalArg("a", np.float32, shape=(100,))]
            )

    knl = lp.preprocess_kernel(knl, ctx.devices[0])
    kernel_gen = lp.generate_loop_schedules(knl)

    for gen_knl in kernel_gen:
        print(gen_knl)
        compiled = lp.CompiledKernel(ctx, gen_knl)
        print(compiled.get_code())


def test_owed_barriers(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i]: 0<=i<100}",
            [
                "<float32> z[i] = a[i]"
                ],
            [lp.GlobalArg("a", np.float32, shape=(100,))]
            )

    knl = lp.tag_inames(knl, dict(i="l.0"))

    knl = lp.preprocess_kernel(knl, ctx.devices[0])
    kernel_gen = lp.generate_loop_schedules(knl)

    for gen_knl in kernel_gen:
        compiled = lp.CompiledKernel(ctx, gen_knl)
        print(compiled.get_code())


def test_wg_too_small(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i]: 0<=i<100}",
            [
                "<float32> z[i] = a[i] {id=copy}"
                ],
            [lp.GlobalArg("a", np.float32, shape=(100,))],
            local_sizes={0: 16})

    knl = lp.tag_inames(knl, dict(i="l.0"))

    knl = lp.preprocess_kernel(knl, ctx.devices[0])
    kernel_gen = lp.generate_loop_schedules(knl)

    import pytest
    for gen_knl in kernel_gen:
        with pytest.raises(RuntimeError):
            lp.CompiledKernel(ctx, gen_knl).get_code()


def test_multi_cse(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i]: 0<=i<100}",
            [
                "<float32> z[i] = a[i] + a[i]**2"
                ],
            [lp.GlobalArg("a", np.float32, shape=(100,))],
            local_sizes={0: 16})

    knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
    knl = lp.add_prefetch(knl, "a", [])

    knl = lp.preprocess_kernel(knl, ctx.devices[0])
    kernel_gen = lp.generate_loop_schedules(knl)

    for gen_knl in kernel_gen:
        compiled = lp.CompiledKernel(ctx, gen_knl)
        print(compiled.get_code())


# {{{ code generator fuzzing

def make_random_value():
    from random import randrange, uniform
    v = randrange(3)
    if v == 0:
        while True:
            z = randrange(-1000, 1000)
            if z:
                return z

    elif v == 1:
        return uniform(-10, 10)
    else:
        cval = uniform(-10, 10) + 1j*uniform(-10, 10)
        if randrange(0, 2) == 0:
            return np.complex128(cval)
        else:
            return np.complex128(cval)


def make_random_expression(var_values, size):
    from random import randrange
    import pymbolic.primitives as p
    v = randrange(1500)
    size[0] += 1
    if v < 500 and size[0] < 40:
        term_count = randrange(2, 5)
        if randrange(2) < 1:
            cls = p.Sum
        else:
            cls = p.Product
        return cls(tuple(
            make_random_expression(var_values, size)
            for i in range(term_count)))
    elif v < 750:
        return make_random_value()
    elif v < 1000:
        var_name = "var_%d" % len(var_values)
        assert var_name not in var_values
        var_values[var_name] = make_random_value()
        return p.Variable(var_name)
    elif v < 1250:
        # Cannot use '-' because that destroys numpy constants.
        return p.Sum((
            make_random_expression(var_values, size),
            - make_random_expression(var_values, size)))
    elif v < 1500:
        # Cannot use '/' because that destroys numpy constants.
        return p.Quotient(
                make_random_expression(var_values, size),
                make_random_expression(var_values, size))


def generate_random_fuzz_examples(count):
    for i in range(count):
        size = [0]
        var_values = {}
        expr = make_random_expression(var_values, size)
        yield expr, var_values


def test_fuzz_code_generator(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    if ctx.devices[0].platform.vendor.startswith("Advanced Micro"):
        pytest.skip("crashes on AMD 15.12")

    #from expr_fuzz import get_fuzz_examples
    #for expr, var_values in get_fuzz_examples():
    for expr, var_values in generate_random_fuzz_examples(50):
        from pymbolic import evaluate
        try:
            true_value = evaluate(expr, var_values)
        except ZeroDivisionError:
            continue

        def get_dtype(x):
            if isinstance(x, (complex, np.complexfloating)):
                return np.complex128
            else:
                return np.float64

        knl = lp.make_kernel("{ : }",
                [lp.Assignment("value", expr)],
                [lp.GlobalArg("value", np.complex128, shape=())]
                + [
                    lp.ValueArg(name, get_dtype(val))
                    for name, val in six.iteritems(var_values)
                    ])
        ck = lp.CompiledKernel(ctx, knl)
        evt, (lp_value,) = ck(queue, out_host=True, **var_values)
        err = abs(true_value-lp_value)/abs(true_value)
        if abs(err) > 1e-10:
            print(80*"-")
            print("WRONG: rel error=%g" % err)
            print("true=%r" % true_value)
            print("loopy=%r" % lp_value)
            print(80*"-")
            print(ck.get_code())
            print(80*"-")
            print(var_values)
            print(80*"-")
            print(repr(expr))
            print(80*"-")
            print(expr)
            print(80*"-")
            1/0

# }}}


def test_bare_data_dependency(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            [
                "[znirp] -> {[i]: 0<=i<znirp}",
                ],
            [
                "<> znirp = n",
                "a[i] = 1",
                ],
            [
                lp.GlobalArg("a", dtype, shape=("n"), order="C"),
                lp.ValueArg("n", np.int32),
                ])

    cknl = lp.CompiledKernel(ctx, knl)
    n = 20000
    evt, (a,) = cknl(queue, n=n, out_host=True)

    assert a.shape == (n,)
    assert (a == 1).all()


# {{{ test race detection

@pytest.mark.skipif("sys.version_info < (2,6)")
def test_ilp_write_race_detection_global(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "[n] -> {[i,j]: 0<=i,j<n }",
            [
                "a[i] = 5+i+j",
                ],
            [
                lp.GlobalArg("a", np.float32),
                lp.ValueArg("n", np.int32, approximately=1000),
                ],
            assumptions="n>=1")

    knl = lp.tag_inames(knl, dict(j="ilp"))

    knl = lp.preprocess_kernel(knl, ctx.devices[0])

    with lp.CacheMode(False):
        from loopy.diagnostic import WriteRaceConditionWarning
        from warnings import catch_warnings
        with catch_warnings(record=True) as warn_list:
            list(lp.generate_loop_schedules(knl))

            assert any(isinstance(w.message, WriteRaceConditionWarning)
                    for w in warn_list)


def test_ilp_write_race_avoidance_local(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i<16 and 0<=j<17 }",
            [
                "<> a[i] = 5+i+j",
                ],
            [])

    knl = lp.tag_inames(knl, dict(i="l.0", j="ilp"))

    knl = lp.preprocess_kernel(knl, ctx.devices[0])
    for k in lp.generate_loop_schedules(knl):
        assert k.temporary_variables["a"].shape == (16, 17)


def test_ilp_write_race_avoidance_private(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[j]: 0<=j<16 }",
            [
                "<> a = 5+j",
                ],
            [])

    knl = lp.tag_inames(knl, dict(j="ilp"))

    knl = lp.preprocess_kernel(knl, ctx.devices[0])
    for k in lp.generate_loop_schedules(knl):
        assert k.temporary_variables["a"].shape == (16,)

# }}}


def test_write_parameter(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n }",
            """
                a = sum((i,j), i*j)
                b = sum(i, sum(j, i*j))
                n = 15
                """,
            [
                lp.GlobalArg("a", dtype, shape=()),
                lp.GlobalArg("b", dtype, shape=()),
                lp.ValueArg("n", np.int32, approximately=1000),
                ],
            assumptions="n>=1")

    import pytest
    with pytest.raises(RuntimeError):
        lp.CompiledKernel(ctx, knl).get_code()


# {{{ arg guessing

def test_arg_shape_guessing(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n }",
            """
                a = 1.5 + sum((i,j), i*j)
                b[i, j] = i*j
                c[i+j, j] = b[j,i]
                """,
            [
                lp.GlobalArg("a", shape=lp.auto),
                lp.GlobalArg("b", shape=lp.auto),
                lp.GlobalArg("c", shape=lp.auto),
                lp.ValueArg("n"),
                ],
            assumptions="n>=1")

    print(knl)
    print(lp.CompiledKernel(ctx, knl).get_highlighted_code())


def test_arg_guessing(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n }",
            """
                a = 1.5 + sum((i,j), i*j)
                b[i, j] = i*j
                c[i+j, j] = b[j,i]
                """,
            assumptions="n>=1")

    print(knl)
    print(lp.CompiledKernel(ctx, knl).get_highlighted_code())


def test_arg_guessing_with_reduction(ctx_factory):
    #logging.basicConfig(level=logging.DEBUG)
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n }",
            """
                a = 1.5 + simul_reduce(sum, (i,j), i*j)
                d = 1.5 + simul_reduce(sum, (i,j), b[i,j])
                b[i, j] = i*j
                c[i+j, j] = b[j,i]
                """,
            assumptions="n>=1")

    print(knl)
    print(lp.CompiledKernel(ctx, knl).get_highlighted_code())

# }}}


def test_nonlinear_index(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n }",
            """
                a[i*i] = 17
                """,
            [
                lp.GlobalArg("a", shape="n"),
                lp.ValueArg("n"),
                ],
            assumptions="n>=1")

    print(knl)
    print(lp.CompiledKernel(ctx, knl).get_highlighted_code())


def test_offsets_and_slicing(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 20

    knl = lp.make_kernel(
            "{[i,j]: 0<=i<n and 0<=j<m }",
            """
                b[i,j] = 2*a[i,j]
                """,
            assumptions="n>=1 and m>=1",
            default_offset=lp.auto)

    knl = lp.tag_data_axes(knl, "a,b", "stride:auto,stride:1")

    cknl = lp.CompiledKernel(ctx, knl)

    a_full = cl.clrandom.rand(queue, (n, n), np.float64)
    a_full_h = a_full.get()
    b_full = cl.clrandom.rand(queue, (n, n), np.float64)
    b_full_h = b_full.get()

    a_sub = (slice(3, 10), slice(5, 10))
    a = a_full[a_sub]

    b_sub = (slice(3+3, 10+3), slice(5+4, 10+4))
    b = b_full[b_sub]

    b_full_h[b_sub] = 2*a_full_h[a_sub]

    print(cknl.get_highlighted_code({"a": a.dtype}))
    cknl(queue, a=a, b=b)

    import numpy.linalg as la
    assert la.norm(b_full.get() - b_full_h) < 1e-13


def test_vector_ilp_with_prefetch(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = 2*a[i]",
            [
                # Tests that comma'd arguments interoperate with
                # argument guessing.
                lp.GlobalArg("out,a", np.float32, shape=lp.auto),
                "..."
                ])

    knl = lp.split_iname(knl, "i", 128, inner_tag="l.0")
    knl = lp.split_iname(knl, "i_outer", 4, outer_tag="g.0", inner_tag="ilp")
    knl = lp.add_prefetch(knl, "a", ["i_inner", "i_outer_inner"])

    cknl = lp.CompiledKernel(ctx, knl)
    cknl.cl_kernel_info()

    import re
    code = cknl.get_code()
    assert len(list(re.finditer("barrier", code))) == 1


def test_c_instruction(ctx_factory):
    #logging.basicConfig(level=logging.DEBUG)
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n }",
            [
                lp.CInstruction("i,j", """
                    x = sin((float) i*j);
                    """, assignees="x"),
                "a[i,j] = x",
                ],
            [
                lp.GlobalArg("a", shape=lp.auto, dtype=np.float32),
                lp.TemporaryVariable("x", np.float32),
                "...",
                ],
            assumptions="n>=1")

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    print(knl)
    print(lp.CompiledKernel(ctx, knl).get_highlighted_code())


def test_dependent_domain_insn_iname_finding(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel([
            "{[isrc_box]: 0<=isrc_box<nsrc_boxes}",
            "{[isrc,idim]: isrc_start<=isrc<isrc_end and 0<=idim<dim}",
            ],
            """
                <> src_ibox = source_boxes[isrc_box]
                <> isrc_start = box_source_starts[src_ibox]
                <> isrc_end = isrc_start+box_source_counts_nonchild[src_ibox]
                <> strength = strengths[isrc] {id=set_strength}
                """,
            [
                lp.GlobalArg("box_source_starts,box_source_counts_nonchild",
                    None, shape=None),
                lp.GlobalArg("strengths",
                    None, shape="nsources"),
                "..."])

    print(knl)
    assert "isrc_box" in knl.insn_inames("set_strength")

    print(lp.CompiledKernel(ctx, knl).get_highlighted_code(
            dict(
                source_boxes=np.int32,
                box_source_starts=np.int32,
                box_source_counts_nonchild=np.int32,
                strengths=np.float64,
                nsources=np.int32,
                )))


def test_inames_deps_from_write_subscript(ctx_factory):
    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n}",
            """
                <> src_ibox = source_boxes[i]
                <int32> something = 5
                a[src_ibox] = sum(j, something) {id=myred}
                """,
            [
                lp.GlobalArg("box_source_starts,box_source_counts_nonchild,a",
                    None, shape=None),
                "..."])

    print(knl)
    assert "i" in knl.insn_inames("myred")


def test_modulo_indexing(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i<n and 0<=j<5}",
            """
                b[i] = sum(j, a[(i+j)%n])
                """,
            [
                lp.GlobalArg("a", None, shape="n"),
                "..."
                ]
            )

    print(knl)
    print(lp.CompiledKernel(ctx, knl).get_highlighted_code(
            dict(
                a=np.float32,
                )))


@pytest.mark.parametrize("vec_len", [2, 3, 4, 8, 16])
def test_vector_types(ctx_factory, vec_len):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{ [i,j]: 0<=i<n and 0<=j<vec_len }",
            "out[i,j] = 2*a[i,j]",
            [
                lp.GlobalArg("a", np.float32, shape=lp.auto),
                lp.GlobalArg("out", np.float32, shape=lp.auto),
                "..."
                ])

    knl = lp.fix_parameters(knl, vec_len=vec_len)

    ref_knl = knl

    knl = lp.tag_data_axes(knl, "out", "c,vec")
    knl = lp.tag_inames(knl, dict(j="unr"))

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    lp.auto_test_vs_ref(ref_knl, ctx, knl,
            parameters=dict(
                n=20000
                ))


def test_conditional(ctx_factory):
    #logging.basicConfig(level=logging.DEBUG)
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{ [i,j]: 0<=i,j<n }",
            """
                <> my_a = a[i,j] {id=read_a}
                <> a_less_than_zero = my_a < 0 {dep=read_a,inames=i:j}
                my_a = 2*my_a {id=twice_a,dep=read_a,if=a_less_than_zero}
                my_a = my_a+1 {id=aplus,dep=twice_a,if=a_less_than_zero}
                out[i,j] = 2*my_a {dep=aplus}
                """,
            [
                lp.GlobalArg("a", np.float32, shape=lp.auto),
                lp.GlobalArg("out", np.float32, shape=lp.auto),
                "..."
                ])

    ref_knl = knl

    lp.auto_test_vs_ref(ref_knl, ctx, knl,
            parameters=dict(
                n=200
                ))


def test_ilp_loop_bound(ctx_factory):
    # The salient bit of this test is that a joint bound on (outer, inner)
    # from a split occurs in a setting where the inner loop has been ilp'ed.
    # In 'normal' parallel loops, the inner index is available for conditionals
    # throughout. In ILP'd loops, not so much.

    ctx = ctx_factory()
    knl = lp.make_kernel(
            "{ [i,j,k]: 0<=i,j,k<n }",
            """
            out[i,k] = sum(j, a[i,j]*b[j,k])
            """,
            [
                lp.GlobalArg("a,b", np.float32, shape=lp.auto),
                "...",
                ],
            assumptions="n>=1")

    ref_knl = knl

    knl = lp.prioritize_loops(knl, "j,i,k")
    knl = lp.split_iname(knl,  "k", 4, inner_tag="ilp")

    lp.auto_test_vs_ref(ref_knl, ctx, knl,
            parameters=dict(
                n=200
                ))


def test_arg_shape_uses_assumptions(ctx_factory):
    # If arg shape determination does not use assumptions, then it won't find a
    # static shape for out, which is at least 1 x 1 in size, but otherwise of
    # size n x n.

    lp.make_kernel(
            "{ [i,j]: 0<=i,j<n }",
            """
            out[i,j] = 2*a[i,j]
            out[0,0] = 13.0
            """, assumptions="n>=1")


def test_slab_decomposition_does_not_double_execute(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "a[i] = 2*a[i]",
        assumptions="n>=1")

    ref_knl = knl

    for outer_tag in ["for", "g.0"]:
        knl = ref_knl
        knl = lp.split_iname(knl, "i", 4, slabs=(0, 1), inner_tag="unr",
                outer_tag=outer_tag)
        knl = lp.prioritize_loops(knl, "i_outer")

        a = cl.array.empty(queue, 20, np.float32)
        a.fill(17)
        a_ref = a.copy()
        a_knl = a.copy()

        knl = lp.set_options(knl, write_cl=True)
        print("TEST-----------------------------------------")
        knl(queue, a=a_knl)
        print("REF-----------------------------------------")
        ref_knl(queue, a=a_ref)
        print("DONE-----------------------------------------")

        print("REF", a_ref)
        print("KNL", a_knl)
        assert (a_ref == a_knl).get().all()

        print("_________________________________")


def test_multiple_writes_to_local_temporary():
    # Loopy would previously only handle barrier insertion correctly if exactly
    # one instruction wrote to each local temporary. This tests that multiple
    # writes are OK.

    knl = lp.make_kernel(
        "{[i,e]: 0<=i<5 and 0<=e<nelements}",
        """
        <> temp[i, 0] = 17
        temp[i, 1] = 15
        """)
    knl = lp.tag_inames(knl, dict(i="l.0"))

    knl = lp.preprocess_kernel(knl)
    for k in lp.generate_loop_schedules(knl):
        code, _ = lp.generate_code(k)
        print(code)


def test_make_copy_kernel(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    intermediate_format = "f,f,sep"

    a1 = np.random.randn(1024, 4, 3)

    cknl1 = lp.make_copy_kernel(intermediate_format)

    cknl1 = lp.fix_parameters(cknl1, n2=3)

    cknl1 = lp.set_options(cknl1, write_cl=True)
    evt, a2 = cknl1(queue, input=a1)

    cknl2 = lp.make_copy_kernel("c,c,c", intermediate_format)
    cknl2 = lp.fix_parameters(cknl2, n2=3)

    evt, a3 = cknl2(queue, input=a2)

    assert (a1 == a3).all()


def test_auto_test_can_detect_problems(ctx_factory):
    ctx = ctx_factory()

    ref_knl = lp.make_kernel(
        "{[i,j]: 0<=i,j<n}",
        """
        a[i,j] = 25
        """)

    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        """
        a[i,i] = 25
        """)

    ref_knl = lp.add_and_infer_dtypes(ref_knl, dict(a=np.float32))
    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32))

    from loopy.diagnostic import AutomaticTestFailure
    with pytest.raises(AutomaticTestFailure):
        lp.auto_test_vs_ref(
                ref_knl, ctx, knl,
                parameters=dict(n=123))


def test_sci_notation_literal(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    set_kernel = lp.make_kernel(
         ''' { [i]: 0<=i<12 } ''',
         ''' out[i] = 1e-12''')

    set_kernel = lp.set_options(set_kernel, write_cl=True)

    evt, (out,) = set_kernel(queue)

    assert (np.abs(out.get() - 1e-12) < 1e-20).all()


def test_variable_size_temporary():
    knl = lp.make_kernel(
         ''' { [i,j]: 0<=i,j<n } ''',
         ''' out[i] = sum(j, a[i,j])''')

    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})

    knl = lp.add_prefetch(
            knl, "a[:,:]", default_tag=None)

    # Make sure that code generation succeeds even if
    # there are variable-length arrays.
    knl = lp.preprocess_kernel(knl)
    for k in lp.generate_loop_schedules(knl):
        lp.generate_code(k)


def test_indexof(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
         ''' { [i,j]: 0<=i,j<5 } ''',
         ''' out[i,j] = indexof(out[i,j])''')

    knl = lp.set_options(knl, write_cl=True)

    (evt, (out,)) = knl(queue)
    out = out.get()

    assert np.array_equal(out.ravel(order="C"), np.arange(25))


def test_indexof_vec(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    if ctx.devices[0].platform.name.startswith("Portable"):
        # Accurate as of 2015-10-08
        pytest.skip("POCL miscompiles vector code")

    knl = lp.make_kernel(
         ''' { [i,j,k]: 0<=i,j,k<4 } ''',
         ''' out[i,j,k] = indexof_vec(out[i,j,k])''')

    knl = lp.tag_inames(knl, {"i": "vec"})
    knl = lp.tag_data_axes(knl, "out", "vec,c,c")
    knl = lp.set_options(knl, write_cl=True)

    (evt, (out,)) = knl(queue)
    #out = out.get()
    #assert np.array_equal(out.ravel(order="C"), np.arange(25))


def test_is_expression_equal():
    from loopy.symbolic import is_expression_equal
    from pymbolic import var

    x = var("x")
    y = var("y")

    assert is_expression_equal(x+2, 2+x)

    assert is_expression_equal((x+2)**2, x**2 + 4*x + 4)
    assert is_expression_equal((x+y)**2, x**2 + 2*x*y + y**2)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_atomic(ctx_factory, dtype):
    ctx = ctx_factory()

    if (
            np.dtype(dtype).itemsize == 8
            and "cl_khr_int64_base_atomics" not in ctx.devices[0].extensions):
        pytest.skip("64-bit atomics not supported on device")

    import pyopencl.version  # noqa
    if (
            cl.version.VERSION < (2015, 2)
            and dtype == np.int64):
        pytest.skip("int64 RNG not supported in PyOpenCL < 2015.2")

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i%20] = out[i%20] + 2*a[i] {atomic}",
            [
                lp.GlobalArg("out", dtype, shape=lp.auto, for_atomic=True),
                lp.GlobalArg("a", dtype, shape=lp.auto),
                "..."
                ],
            assumptions="n>0")

    ref_knl = knl
    knl = lp.split_iname(knl, "i", 512)
    knl = lp.split_iname(knl, "i_inner", 128, outer_tag="unr", inner_tag="g.0")
    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=10000))


def test_within_inames_and_reduction():
    # See https://github.com/inducer/loopy/issues/24

    # This is (purposefully) somewhat un-idiomatic, to replicate the conditions
    # under which the above bug was found. If assignees were phi[i], then the
    # iname propagation heuristic would not assume that dependent instructions
    # need to run inside of 'i', and hence the forced_iname_* bits below would not
    # be needed.

    i1 = lp.CInstruction("i",
            "doSomethingToGetPhi();",
            assignees="phi")

    from pymbolic.primitives import Subscript, Variable
    i2 = lp.Assignment("a",
            lp.Reduction("sum", "j", Subscript(Variable("phi"), Variable("j"))),
            within_inames=frozenset(),
            within_inames_is_final=True)

    k = lp.make_kernel("{[i,j] : 0<=i,j<n}",
            [i1, i2],
            [
                lp.GlobalArg("a", dtype=np.float32, shape=()),
                lp.ValueArg("n", dtype=np.int32),
                lp.TemporaryVariable("phi", dtype=np.float32, shape=("n",)),
                ],
            target=lp.CTarget(),
            )

    k = lp.preprocess_kernel(k)

    assert 'i' not in k.insn_inames("insn_0_j_update")
    print(k.stringify(with_dependencies=True))


def test_literal_local_barrier(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            """
            for i
                ... lbarrier
            end
            """, seq_dependencies=True)

    knl = lp.fix_parameters(knl, n=128)

    ref_knl = knl

    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=5))


def test_kernel_splitting(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            """
            for i
                c[i] = a[i + 1]
                ... gbarrier
                out[i] = c[i]
            end
            """, seq_dependencies=True)

    knl = lp.add_and_infer_dtypes(knl,
            {"a": np.float32, "c": np.float32, "out": np.float32, "n": np.int32})

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    # schedule
    from loopy.preprocess import preprocess_kernel
    knl = preprocess_kernel(knl)

    from loopy.schedule import get_one_scheduled_kernel
    knl = get_one_scheduled_kernel(knl)

    # map schedule onto host or device
    print(knl)

    cgr = lp.generate_code_v2(knl)

    assert len(cgr.device_programs) == 2

    print(cgr.device_code())
    print(cgr.host_code())

    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=5))


def test_kernel_splitting_with_loop(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{ [i,k]: 0<=i<n and 0<=k<3 }",
            """
            for i, k
                ... gbarrier
                c[k,i] = a[k, i + 1]
                ... gbarrier
                out[k,i] = c[k,i]
            end
            """, seq_dependencies=True)

    knl = lp.add_and_infer_dtypes(knl,
            {"a": np.float32, "c": np.float32, "out": np.float32, "n": np.int32})

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    # schedule
    from loopy.preprocess import preprocess_kernel
    knl = preprocess_kernel(knl)

    from loopy.schedule import get_one_scheduled_kernel
    knl = get_one_scheduled_kernel(knl)

    # map schedule onto host or device
    print(knl)

    cgr = lp.generate_code_v2(knl)

    assert len(cgr.device_programs) == 2

    print(cgr.device_code())
    print(cgr.host_code())

    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=5))


def save_and_reload_temporaries_test(queue, knl, out_expect, debug=False):
    from loopy.preprocess import preprocess_kernel
    from loopy.schedule import get_one_scheduled_kernel

    knl = preprocess_kernel(knl)
    knl = get_one_scheduled_kernel(knl)

    from loopy.transform.save import save_and_reload_temporaries
    knl = save_and_reload_temporaries(knl)
    knl = get_one_scheduled_kernel(knl)

    if debug:
        print(knl)
        cgr = lp.generate_code_v2(knl)
        print(cgr.device_code())
        print(cgr.host_code())
        1/0

    _, (out,) = knl(queue, out_host=True)
    assert (out == out_expect).all(), (out, out_expect)


@pytest.mark.parametrize("hw_loop", [True, False])
def test_save_of_private_scalar(ctx_factory, hw_loop, debug=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{ [i]: 0<=i<8 }",
        """
        for i
            <>t = i
            ... gbarrier
            out[i] = t
        end
        """, seq_dependencies=True)

    if hw_loop:
        knl = lp.tag_inames(knl, dict(i="g.0"))

    save_and_reload_temporaries_test(queue, knl, np.arange(8), debug)


def test_save_of_private_array(ctx_factory, debug=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{ [i]: 0<=i<8 }",
        """
        for i
            <>t[i] = i
            ... gbarrier
            out[i] = t[i]
        end
        """, seq_dependencies=True)

    knl = lp.set_temporary_scope(knl, "t", "private")
    save_and_reload_temporaries_test(queue, knl, np.arange(8), debug)


def test_save_of_private_array_in_hw_loop(ctx_factory, debug=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{ [i,j,k]: 0<=i,j,k<8 }",
        """
        for i
            for j
               <>t[j] = j
            end
            ... gbarrier
            for k
                out[i,k] = t[k]
            end
        end
        """, seq_dependencies=True)

    knl = lp.tag_inames(knl, dict(i="g.0"))
    knl = lp.set_temporary_scope(knl, "t", "private")

    save_and_reload_temporaries_test(
        queue, knl, np.vstack((8 * (np.arange(8),))), debug)


def test_save_of_private_multidim_array(ctx_factory, debug=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{ [i,j,k,l,m]: 0<=i,j,k,l,m<8 }",
        """
        for i
            for j, k
               <>t[j,k] = k
            end
            ... gbarrier
            for l, m
                out[i,l,m] = t[l,m]
            end
        end
        """, seq_dependencies=True)

    knl = lp.set_temporary_scope(knl, "t", "private")

    result = np.array([np.vstack((8 * (np.arange(8),))) for i in range(8)])
    save_and_reload_temporaries_test(queue, knl, result, debug)


def test_save_of_private_multidim_array_in_hw_loop(ctx_factory, debug=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{ [i,j,k,l,m]: 0<=i,j,k,l,m<8 }",
        """
        for i
            for j, k
               <>t[j,k] = k
            end
            ... gbarrier
            for l, m
                out[i,l,m] = t[l,m]
            end
        end
        """, seq_dependencies=True)

    knl = lp.set_temporary_scope(knl, "t", "private")
    knl = lp.tag_inames(knl, dict(i="g.0"))

    result = np.array([np.vstack((8 * (np.arange(8),))) for i in range(8)])
    save_and_reload_temporaries_test(queue, knl, result, debug)


@pytest.mark.parametrize("hw_loop", [True, False])
def test_save_of_multiple_private_temporaries(ctx_factory, hw_loop, debug=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{ [i,j,k]: 0<=i,j,k<10 }",
            """
            for i
                for k
                    <> t_arr[k] = k
                end
                <> t_scalar = 1
                for j
                    ... gbarrier
                    out[j] = t_scalar
                    ... gbarrier
                    t_scalar = 10
                end
                ... gbarrier
                <> flag = i == 9
                out[i] = t_arr[i] {if=flag}
            end
            """, seq_dependencies=True)

    knl = lp.set_temporary_scope(knl, "t_arr", "private")
    if hw_loop:
        knl = lp.tag_inames(knl, dict(i="g.0"))

    result = np.array([1, 10, 10, 10, 10, 10, 10, 10, 10, 9])

    save_and_reload_temporaries_test(queue, knl, result, debug)


def test_save_of_local_array(ctx_factory, debug=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{ [i,j]: 0<=i,j<8 }",
        """
        for i, j
            <>t[2*j] = j
            t[2*j+1] = j
            ... gbarrier
            out[i] = t[2*i]
        end
        """, seq_dependencies=True)

    knl = lp.set_temporary_scope(knl, "t", "local")
    knl = lp.tag_inames(knl, dict(i="g.0", j="l.0"))

    save_and_reload_temporaries_test(queue, knl, np.arange(8), debug)


def test_save_of_local_array_with_explicit_local_barrier(ctx_factory, debug=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{ [i,j]: 0<=i,j<8 }",
        """
        for i, j
            <>t[2*j] = j
            ... lbarrier
            t[2*j+1] = t[2*j]
            ... gbarrier
            out[i] = t[2*i]
        end
        """, seq_dependencies=True)

    knl = lp.set_temporary_scope(knl, "t", "local")
    knl = lp.tag_inames(knl, dict(i="g.0", j="l.0"))

    save_and_reload_temporaries_test(queue, knl, np.arange(8), debug)


def test_save_local_multidim_array(ctx_factory, debug=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{ [i,j,k]: 0<=i<2 and 0<=k<3 and 0<=j<2}",
            """
            for i, j, k
                ... gbarrier
                <> t_local[k,j] = 1
                ... gbarrier
                out[k,i*2+j] = t_local[k,j]
            end
            """, seq_dependencies=True)

    knl = lp.set_temporary_scope(knl, "t_local", "local")
    knl = lp.tag_inames(knl, dict(j="l.0", i="g.0"))

    save_and_reload_temporaries_test(queue, knl, 1, debug)


def test_save_with_base_storage(ctx_factory, debug=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{[i]: 0 <= i < 10}",
            """
            <>a[i] = 0
            <>b[i] = i
            ... gbarrier
            out[i] = a[i]
            """,
            "...",
            seq_dependencies=True)

    knl = lp.tag_inames(knl, dict(i="l.0"))
    knl = lp.set_temporary_scope(knl, "a", "local")
    knl = lp.set_temporary_scope(knl, "b", "local")

    knl = lp.alias_temporaries(knl, ["a", "b"],
            synchronize_for_exclusive_use=False)

    save_and_reload_temporaries_test(queue, knl, np.arange(10), debug)


def test_save_ambiguous_storage_requirements():
    knl = lp.make_kernel(
            "{[i,j]: 0 <= i < 10 and 0 <= j < 10}",
            """
            <>a[j] = j
            ... gbarrier
            out[i,j] = a[j]
            """,
            seq_dependencies=True)

    knl = lp.tag_inames(knl, dict(i="g.0", j="l.0"))
    knl = lp.duplicate_inames(knl, "j", within="writes:out", tags={"j": "l.0"})
    knl = lp.set_temporary_scope(knl, "a", "local")

    knl = lp.preprocess_kernel(knl)
    knl = lp.get_one_scheduled_kernel(knl)

    from loopy.diagnostic import LoopyError
    with pytest.raises(LoopyError):
        lp.save_and_reload_temporaries(knl)


def test_save_across_inames_with_same_tag(ctx_factory, debug=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{[i]: 0 <= i < 10}",
            """
            <>a[i] = i
            ... gbarrier
            out[i] = a[i]
            """,
            "...",
            seq_dependencies=True)

    knl = lp.tag_inames(knl, dict(i="l.0"))
    knl = lp.duplicate_inames(knl, "i", within="reads:a", tags={"i": "l.0"})

    save_and_reload_temporaries_test(queue, knl, np.arange(10), debug)


def test_missing_temporary_definition_detection():
    knl = lp.make_kernel(
            "{ [i]: 0<=i<10 }",
            """
            for i
                <> t = 1
                ... gbarrier
                out[i] = t
            end
            """, seq_dependencies=True)

    from loopy.diagnostic import MissingDefinitionError
    with pytest.raises(MissingDefinitionError):
        lp.generate_code_v2(knl)


def test_missing_definition_check_respects_aliases():
    # Based on https://github.com/inducer/loopy/issues/69
    knl = lp.make_kernel("{ [i] : 0<=i<n }",
         ["a[i] = 0",
          "c[i] = b[i]"],
         temporary_variables={
             "a": lp.TemporaryVariable("a",
                        dtype=np.float64, shape=("n",), base_storage="base"),
             "b": lp.TemporaryVariable("b",
                        dtype=np.float64, shape=("n",), base_storage="base")
         },
         target=lp.CTarget(),
         silenced_warnings=frozenset(["read_no_write(b)"]))

    lp.generate_code_v2(knl)


def test_global_temporary(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n}",
            """
            for i
                <> c[i] = a[i + 1]
                ... gbarrier
                out[i] = c[i]
            end
            """, seq_dependencies=True)

    knl = lp.add_and_infer_dtypes(knl,
            {"a": np.float32, "c": np.float32, "out": np.float32, "n": np.int32})
    knl = lp.set_temporary_scope(knl, "c", "global")

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    cgr = lp.generate_code_v2(knl)

    assert len(cgr.device_programs) == 2

    #print(cgr.device_code())
    #print(cgr.host_code())

    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=5))


def test_assign_to_linear_subscript(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl1 = lp.make_kernel(
            "{ [i]: 0<=i<n}",
            "a[i,i] = 1")
    knl2 = lp.make_kernel(
            "{ [i]: 0<=i<n}",
            "a[[i*n + i]] = 1",
            [lp.GlobalArg("a", shape="n,n"), "..."])

    a1 = cl.array.zeros(queue, (10, 10), np.float32)
    knl1(queue, a=a1)
    a2 = cl.array.zeros(queue, (10, 10), np.float32)
    knl2(queue, a=a2)

    assert np.array_equal(a1.get(),  a2.get())


def test_finite_difference_expr_subst(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    grid = np.linspace(0, 2*np.pi, 2048, endpoint=False)
    h = grid[1] - grid[0]
    u = cl.clmath.sin(cl.array.to_device(queue, grid))

    fin_diff_knl = lp.make_kernel(
        "{[i]: 1<=i<=n}",
        "out[i] = -(f[i+1] - f[i-1])/h",
        [lp.GlobalArg("out", shape="n+2"), "..."])

    flux_knl = lp.make_kernel(
        "{[j]: 1<=j<=n}",
        "f[j] = u[j]**2/2",
        [
            lp.GlobalArg("f", shape="n+2"),
            lp.GlobalArg("u", shape="n+2"),
            ])

    fused_knl = lp.fuse_kernels([fin_diff_knl, flux_knl],
            data_flow=[
                ("f", 1, 0)
                ])

    fused_knl = lp.set_options(fused_knl, write_cl=True)
    evt, _ = fused_knl(queue, u=u, h=np.float32(1e-1))

    fused_knl = lp.assignment_to_subst(fused_knl, "f")

    fused_knl = lp.set_options(fused_knl, write_cl=True)

    # This is the real test here: The automatically generated
    # shape expressions are '2+n' and the ones above are 'n+2'.
    # Is loopy smart enough to understand that these are equal?
    evt, _ = fused_knl(queue, u=u, h=np.float32(1e-1))

    fused0_knl = lp.affine_map_inames(fused_knl, "i", "inew", "inew+1=i")

    gpu_knl = lp.split_iname(
            fused0_knl, "inew", 128, outer_tag="g.0", inner_tag="l.0")

    precomp_knl = lp.precompute(
            gpu_knl, "f_subst", "inew_inner", fetch_bounding_box=True)

    precomp_knl = lp.tag_inames(precomp_knl, {"j_0_outer": "unr"})
    precomp_knl = lp.set_options(precomp_knl, return_dict=True)
    evt, _ = precomp_knl(queue, u=u, h=h)


# {{{ call without returned values

def test_call_with_no_returned_value(ctx_factory):
    import pymbolic.primitives as p

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{:}",
        [lp.CallInstruction((), p.Call(p.Variable("f"), ()))]
        )

    from library_for_test import no_ret_f_mangler, no_ret_f_preamble_gen
    knl = lp.register_function_manglers(knl, [no_ret_f_mangler])
    knl = lp.register_preamble_generators(knl, [no_ret_f_preamble_gen])

    evt, _ = knl(queue)

# }}}


# {{{ call with no return values and options

def test_call_with_options():
    knl = lp.make_kernel(
        "{:}",
        "f() {id=init}"
        )

    from library_for_test import no_ret_f_mangler
    knl = lp.register_function_manglers(knl, [no_ret_f_mangler])

    print(lp.generate_code_v2(knl).device_code())

# }}}


def test_unschedulable_kernel_detection():
    knl = lp.make_kernel(["{[i,j]:0<=i,j<n}"],
                         """
                         mat1[i,j] = mat1[i,j] + 1 {inames=i:j, id=i1}
                         mat2[j] = mat2[j] + 1 {inames=j, id=i2}
                         mat3[i] = mat3[i] + 1 {inames=i, id=i3}
                         """)

    knl = lp.preprocess_kernel(knl)

    # Check that loopy can detect the unschedulability of the kernel
    assert not lp.has_schedulable_iname_nesting(knl)
    assert len(list(lp.get_iname_duplication_options(knl))) == 4

    for inames, insns in lp.get_iname_duplication_options(knl):
        fixed_knl = lp.duplicate_inames(knl, inames, insns)
        assert lp.has_schedulable_iname_nesting(fixed_knl)

    knl = lp.make_kernel(["{[i,j,k,l,m]:0<=i,j,k,l,m<n}"],
                         """
                         mat1[l,m,i,j,k] = mat1[l,m,i,j,k] + 1 {inames=i:j:k:l:m}
                         mat2[l,m,j,k] = mat2[l,m,j,k] + 1 {inames=j:k:l:m}
                         mat3[l,m,k] = mat3[l,m,k] + 11 {inames=k:l:m}
                         mat4[l,m,i] = mat4[l,m,i] + 1 {inames=i:l:m}
                         """)

    assert not lp.has_schedulable_iname_nesting(knl)
    assert len(list(lp.get_iname_duplication_options(knl))) == 10


def test_regression_no_ret_call_removal(ctx_factory):
    # https://github.com/inducer/loopy/issues/32
    knl = lp.make_kernel(
            "{[i] : 0<=i<n}",
            "f(sum(i, x[i]))")
    knl = lp.add_and_infer_dtypes(knl, {"x": np.float32})
    knl = lp.preprocess_kernel(knl)
    assert len(knl.instructions) == 3


def test_regression_persistent_hash():
    knl1 = lp.make_kernel(
            "{[i] : 0<=i<n}",
            "cse_exprvar = d[2]*d[2]")

    knl2 = lp.make_kernel(
            "{[i] : 0<=i<n}",
            "cse_exprvar = d[0]*d[0]")
    from loopy.tools import LoopyKeyBuilder
    lkb = LoopyKeyBuilder()
    assert lkb(knl1.instructions[0]) != lkb(knl2.instructions[0])
    assert lkb(knl1) != lkb(knl2)


def test_sequential_dependencies(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            """
            for i
                <> aa = 5jf
                <> bb = 5j
                a[i] = imag(aa)
                b[i] = imag(bb)
                c[i] = 5f
            end
            """, seq_dependencies=True)

    print(knl.stringify(with_dependencies=True))

    lp.auto_test_vs_ref(knl, ctx, knl, parameters=dict(n=5))


def test_nop(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,itrip]: 0<=i<n and 0<=itrip<ntrips}",
            """
            for itrip,i
                <> z[i] = z[i+1] + z[i]  {id=wr_z}
                <> v[i] = 11  {id=wr_v}
                ... nop {dep=wr_z:wr_v,id=yoink}
                z[i] = z[i] - z[i+1] + v[i]
            end
            """)

    print(knl)

    knl = lp.fix_parameters(knl, n=15)
    knl = lp.add_and_infer_dtypes(knl, {"z": np.float64})

    lp.auto_test_vs_ref(knl, ctx, knl, parameters=dict(ntrips=5))


def test_global_barrier(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,itrip]: 0<=i<n and 0<=itrip<ntrips}",
            """
            for i
                for itrip
                    ... gbarrier {id=top}
                    <> z[i] = z[i+1] + z[i]  {id=wr_z,dep=top}
                    <> v[i] = 11  {id=wr_v,dep=top}
                    ... gbarrier {dep=wr_z:wr_v,id=yoink}
                    z[i] = z[i] - z[i+1] + v[i] {id=iupd}
                end
                ... gbarrier {dep=iupd,id=postloop}
                z[i] = z[i] - z[i+1] + v[i]  {dep=postloop}
            end
            """)

    knl = lp.fix_parameters(knl, ntrips=3)
    knl = lp.add_and_infer_dtypes(knl, {"z": np.float64})

    ref_knl = knl
    ref_knl = lp.set_temporary_scope(ref_knl, "z", "global")
    ref_knl = lp.set_temporary_scope(ref_knl, "v", "global")

    knl = lp.split_iname(knl, "i", 256, outer_tag="g.0", inner_tag="l.0")
    print(knl)

    knl = lp.preprocess_kernel(knl)
    assert knl.temporary_variables["z"].scope == lp.temp_var_scope.GLOBAL
    assert knl.temporary_variables["v"].scope == lp.temp_var_scope.GLOBAL

    print(knl)

    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(ntrips=5, n=10))


def test_missing_global_barrier():
    knl = lp.make_kernel(
            "{[i,itrip]: 0<=i<n and 0<=itrip<ntrips}",
            """
            for i
                for itrip
                    ... gbarrier {id=yoink}
                    <> z[i] = z[i] - z[i+1]  {id=iupd,dep=yoink}
                end
                # This is where the barrier should be
                z[i] = z[i] - z[i+1] + v[i]  {dep=iupd}
            end
            """)

    knl = lp.set_temporary_scope(knl, "z", "global")
    knl = lp.split_iname(knl, "i", 256, outer_tag="g.0")
    knl = lp.preprocess_kernel(knl)

    from loopy.diagnostic import MissingBarrierError
    with pytest.raises(MissingBarrierError):
        lp.get_one_scheduled_kernel(knl)


def test_index_cse(ctx_factory):
    knl = lp.make_kernel(["{[i,j,k,l,m]:0<=i,j,k,l,m<n}"],
                         """
                         for i
                            for j
                                c[i,j,m] = sum((k,l), a[i,j,l]*b[i,j,k,l])
                            end
                         end
                         """)
    knl = lp.tag_inames(knl, "l:unr")
    knl = lp.prioritize_loops(knl, "i,j,k,l")
    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32, "b": np.float32})
    knl = lp.fix_parameters(knl, n=5)
    print(lp.generate_code_v2(knl).device_code())


def test_ilp_and_conditionals(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel('{[k]: 0<=k<n}}',
         """
         for k
             <> Tcond = T[k] < 0.5
             if Tcond
                 cp[k] = 2 * T[k] + Tcond
             end
         end
         """)

    knl = lp.fix_parameters(knl, n=200)
    knl = lp.add_and_infer_dtypes(knl, {"T": np.float32})

    ref_knl = knl

    knl = lp.split_iname(knl, 'k', 2, inner_tag='ilp')

    lp.auto_test_vs_ref(ref_knl, ctx, knl)


def test_unr_and_conditionals(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel('{[k]: 0<=k<n}}',
         """
         for k
             <> Tcond[k] = T[k] < 0.5
             if Tcond[k]
                 cp[k] = 2 * T[k] + Tcond[k]
             end
         end
         """)

    knl = lp.fix_parameters(knl, n=200)
    knl = lp.add_and_infer_dtypes(knl, {"T": np.float32})

    ref_knl = knl

    knl = lp.split_iname(knl, 'k', 2, inner_tag='unr')

    lp.auto_test_vs_ref(ref_knl, ctx, knl)


def test_constant_array_args(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel('{[k]: 0<=k<n}}',
         """
         for k
             <> Tcond[k] = T[k] < 0.5
             if Tcond[k]
                 cp[k] = 2 * T[k] + Tcond[k]
             end
         end
         """,
         [lp.ConstantArg('T', shape=(200,), dtype=np.float32),
         '...'])

    knl = lp.fix_parameters(knl, n=200)

    lp.auto_test_vs_ref(knl, ctx, knl)


@pytest.mark.parametrize("src_order", ["C"])
@pytest.mark.parametrize("tmp_order", ["C", "F"])
def test_temp_initializer(ctx_factory, src_order, tmp_order):
    a = np.random.randn(3, 3).copy(order=src_order)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n}",
            "out[i,j] = tmp[i,j]",
            [
                lp.TemporaryVariable("tmp",
                    initializer=a,
                    shape=lp.auto,
                    scope=lp.temp_var_scope.PRIVATE,
                    read_only=True,
                    order=tmp_order),
                "..."
                ])

    knl = lp.set_options(knl, write_cl=True, highlight_cl=True)
    knl = lp.fix_parameters(knl, n=a.shape[0])

    evt, (a2,) = knl(queue, out_host=True)

    assert np.array_equal(a, a2)


def test_const_temp_with_initializer_not_saved():
    knl = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
        ... gbarrier
        out[i] = tmp[i]
        """,
        [
            lp.TemporaryVariable("tmp",
                initializer=np.arange(10),
                shape=lp.auto,
                scope=lp.temp_var_scope.PRIVATE,
                read_only=True),
            "..."
            ],
        seq_dependencies=True)

    knl = lp.preprocess_kernel(knl)
    knl = lp.get_one_scheduled_kernel(knl)
    knl = lp.save_and_reload_temporaries(knl)

    # This ensures no save slot was added.
    assert len(knl.temporary_variables) == 1


def test_header_extract():
    knl = lp.make_kernel('{[k]: 0<=k<n}}',
         """
         for k
             T[k] = k**2
         end
         """,
         [lp.GlobalArg('T', shape=(200,), dtype=np.float32),
         '...'])

    knl = lp.fix_parameters(knl, n=200)

    #test C
    cknl = knl.copy(target=lp.CTarget())
    assert str(lp.generate_header(cknl)[0]) == (
            'void loopy_kernel(float *__restrict__ T);')

    #test CUDA
    cuknl = knl.copy(target=lp.CudaTarget())
    assert str(lp.generate_header(cuknl)[0]) == (
            'extern "C" __global__ void __launch_bounds__(1) '
            'loopy_kernel(float *__restrict__ T);')

    #test OpenCL
    oclknl = knl.copy(target=lp.PyOpenCLTarget())
    assert str(lp.generate_header(oclknl)[0]) == (
            '__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) '
            'loopy_kernel(__global float *__restrict__ T);')


def test_scalars_with_base_storage(ctx_factory):
    """ Regression test for !50 """
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{ [i]: 0<=i<1}",
            "a = 1",
            [lp.TemporaryVariable("a", dtype=np.float64,
                                  shape=(), base_storage="base")])

    knl(queue, out_host=True)


def test_if_else(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{ [i]: 0<=i<50}",
            """
            if i % 3 == 0
                a[i] = 15
            elif i % 3 == 1
                a[i] = 11
            else
                a[i] = 3
            end
            """
            )

    evt, (out,) = knl(queue, out_host=True)

    out_ref = np.empty(50)
    out_ref[::3] = 15
    out_ref[1::3] = 11
    out_ref[2::3] = 3

    assert np.array_equal(out_ref, out)

    knl = lp.make_kernel(
            "{ [i]: 0<=i<50}",
            """
            for i
                if i % 2 == 0
                    if i % 3 == 0
                        a[i] = 15
                    elif i % 3 == 1
                        a[i] = 11
                    else
                        a[i] = 3
                    end
                else
                    a[i] = 4
                end
            end
            """
            )

    evt, (out,) = knl(queue, out_host=True)

    out_ref = np.zeros(50)
    out_ref[1::2] = 4
    out_ref[0::6] = 15
    out_ref[4::6] = 11
    out_ref[2::6] = 3

    assert np.array_equal(out_ref, out)


def test_tight_loop_bounds(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        ["{ [i] : 0 <= i <= 5 }",
         "[i] -> { [j] : 2 * i - 2 < j <= 2 * i and 0 <= j <= 9 }"],
        """
        for i
          for j
            out[j] = j
          end
        end
        """,
        silenced_warnings="write_race(insn)")

    knl = lp.split_iname(knl, "i", 5, inner_tag="l.0", outer_tag="g.0")

    evt, (out,) = knl(queue, out_host=True)

    assert (out == np.arange(10)).all()


def test_tight_loop_bounds_codegen():
    knl = lp.make_kernel(
        ["{ [i] : 0 <= i <= 5 }",
         "[i] -> { [j] : 2 * i - 2 <= j <= 2 * i and 0 <= j <= 9 }"],
        """
        for i
          for j
            out[j] = j
          end
        end
        """,
        silenced_warnings="write_race(insn)",
        target=lp.OpenCLTarget())

    knl = lp.split_iname(knl, "i", 5, inner_tag="l.0", outer_tag="g.0")

    cgr = lp.generate_code_v2(knl)
    #print(cgr.device_code())

    for_loop = \
        "for (int j = " \
        "(gid(0) == 0 && lid(0) == 0 ? 0 : -2 + 2 * lid(0) + 10 * gid(0)); " \
        "j <= (-1 + gid(0) == 0 && lid(0) == 0 ? 9 : 2 * lid(0)); ++j)"

    assert for_loop in cgr.device_code()


def test_unscheduled_insn_detection():
    knl = lp.make_kernel(
        "{ [i]: 0 <= i < 10 }",
        """
        out[i] = i {id=insn1}
        """,
        "...")

    knl = lp.get_one_scheduled_kernel(lp.preprocess_kernel(knl))
    insn1, = lp.find_instructions(knl, "id:insn1")
    knl.instructions.append(insn1.copy(id="insn2"))

    from loopy.diagnostic import UnscheduledInstructionError
    with pytest.raises(UnscheduledInstructionError):
        lp.generate_code(knl)


def test_integer_reduction(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from loopy.kernel.data import temp_var_scope as scopes
    from loopy.types import to_loopy_type

    n = 200
    for vtype in [np.int32, np.int64]:
        var_int = np.random.randint(1000, size=n).astype(vtype)
        var_lp = lp.TemporaryVariable('var', initializer=var_int,
                                   read_only=True,
                                   scope=scopes.PRIVATE,
                                   dtype=to_loopy_type(vtype),
                                   shape=lp.auto)

        from collections import namedtuple
        ReductionTest = namedtuple('ReductionTest', 'kind, check, args')

        reductions = [
            ReductionTest('max', lambda x: x == np.max(var_int), args='var[k]'),
            ReductionTest('min', lambda x: x == np.min(var_int), args='var[k]'),
            ReductionTest('sum', lambda x: x == np.sum(var_int), args='var[k]'),
            ReductionTest('product', lambda x: x == np.prod(var_int), args='var[k]'),
            ReductionTest('argmax',
                lambda x: (
                    x[0] == np.max(var_int) and var_int[out[1]] == np.max(var_int)),
                args='var[k], k'),
            ReductionTest('argmin',
                lambda x: (
                    x[0] == np.min(var_int) and var_int[out[1]] == np.min(var_int)),
                args='var[k], k')
        ]

        for reduction, function, args in reductions:
            kstr = ("out" if 'arg' not in reduction
                        else "out[0], out[1]")
            kstr += ' = {0}(k, {1})'.format(reduction, args)
            knl = lp.make_kernel('{[k]: 0<=k<n}',
                                kstr,
                                [var_lp, '...'])

            knl = lp.fix_parameters(knl, n=200)

            _, (out,) = knl(queue, out_host=True)

            assert function(out)


def test_complicated_argmin_reduction(ctx_factory):
    cl_ctx = ctx_factory()
    knl = lp.make_kernel(
            "{[ictr,itgt,idim]: "
            "0<=itgt<ntargets "
            "and 0<=ictr<ncenters "
            "and 0<=idim<ambient_dim}",

            """
            for itgt
                for ictr
                    <> dist_sq = sum(idim,
                            (tgt[idim,itgt] - center[idim,ictr])**2)
                    <> in_disk = dist_sq < (radius[ictr]*1.05)**2
                    <> matches = (
                            (in_disk
                                and qbx_forced_limit == 0)
                            or (in_disk
                                    and qbx_forced_limit != 0
                                    and qbx_forced_limit * center_side[ictr] > 0)
                            )

                    <> post_dist_sq = if(matches, dist_sq, HUGE)
                end
                <> min_dist_sq, <> min_ictr = argmin(ictr, ictr, post_dist_sq)

                tgt_to_qbx_center[itgt] = if(min_dist_sq < HUGE, min_ictr, -1)
            end
            """)

    knl = lp.fix_parameters(knl, ambient_dim=2)
    knl = lp.add_and_infer_dtypes(knl, {
            "tgt,center,radius,HUGE": np.float32,
            "center_side,qbx_forced_limit": np.int32,
            })

    lp.auto_test_vs_ref(knl, cl_ctx, knl, parameters={
            "HUGE": 1e20, "ncenters": 200, "ntargets": 300,
            "qbx_forced_limit": 1})


def test_nosync_option_parsing():
    knl = lp.make_kernel(
        "{[i]: 0 <= i < 10}",
        """
        <>t = 1 {id=insn1,nosync=insn1}
        t = 2   {id=insn2,nosync=insn1:insn2}
        t = 3   {id=insn3,nosync=insn1@local:insn2@global:insn3@any}
        t = 4   {id=insn4,nosync_query=id:insn*@local}
        t = 5   {id=insn5,nosync_query=id:insn1}
        """,
        options=lp.Options(allow_terminal_colors=False))
    kernel_str = str(knl)
    assert "# insn1,no_sync_with=insn1@any" in kernel_str
    assert "# insn2,no_sync_with=insn1@any:insn2@any" in kernel_str
    assert "# insn3,no_sync_with=insn1@local:insn2@global:insn3@any" in kernel_str
    assert "# insn4,no_sync_with=insn1@local:insn2@local:insn3@local:insn5@local" in kernel_str  # noqa
    assert "# insn5,no_sync_with=insn1@any" in kernel_str


def assert_barrier_between(knl, id1, id2, ignore_barriers_in_levels=()):
    from loopy.schedule import (RunInstruction, Barrier, EnterLoop, LeaveLoop)
    watch_for_barrier = False
    seen_barrier = False
    loop_level = 0

    for sched_item in knl.schedule:
        if isinstance(sched_item, RunInstruction):
            if sched_item.insn_id == id1:
                watch_for_barrier = True
            elif sched_item.insn_id == id2:
                assert watch_for_barrier
                assert seen_barrier
                return
        elif isinstance(sched_item, Barrier):
            if watch_for_barrier and loop_level not in ignore_barriers_in_levels:
                seen_barrier = True
        elif isinstance(sched_item, EnterLoop):
            loop_level += 1
        elif isinstance(sched_item, LeaveLoop):
            loop_level -= 1

    raise RuntimeError("id2 was not seen")


def test_barrier_insertion_near_top_of_loop():
    knl = lp.make_kernel(
        "{[i,j]: 0 <= i,j < 10 }",
        """
        for i
         <>a[i] = i  {id=ainit}
         for j
          <>t = a[(i + 1) % 10]  {id=tcomp}
          <>b[i,j] = a[i] + t   {id=bcomp1}
          b[i,j] = b[i,j] + 1  {id=bcomp2}
         end
        end
        """,
        seq_dependencies=True)

    knl = lp.tag_inames(knl, dict(i="l.0"))
    knl = lp.set_temporary_scope(knl, "a", "local")
    knl = lp.set_temporary_scope(knl, "b", "local")
    knl = lp.get_one_scheduled_kernel(lp.preprocess_kernel(knl))

    print(knl)

    assert_barrier_between(knl, "ainit", "tcomp")
    assert_barrier_between(knl, "tcomp", "bcomp1")
    assert_barrier_between(knl, "bcomp1", "bcomp2")


def test_barrier_insertion_near_bottom_of_loop():
    knl = lp.make_kernel(
        ["{[i]: 0 <= i < 10 }",
         "[jmax] -> {[j]: 0 <= j < jmax}"],
        """
        for i
         <>a[i] = i  {id=ainit}
         for j
          <>b[i,j] = a[i] + t   {id=bcomp1}
          b[i,j] = b[i,j] + 1  {id=bcomp2}
         end
         a[i] = i + 1 {id=aupdate}
        end
        """,
        seq_dependencies=True)
    knl = lp.tag_inames(knl, dict(i="l.0"))
    knl = lp.set_temporary_scope(knl, "a", "local")
    knl = lp.set_temporary_scope(knl, "b", "local")
    knl = lp.get_one_scheduled_kernel(lp.preprocess_kernel(knl))

    print(knl)

    assert_barrier_between(knl, "bcomp1", "bcomp2")
    assert_barrier_between(knl, "ainit", "aupdate", ignore_barriers_in_levels=[1])


def test_multi_argument_reduction_type_inference():
    from loopy.type_inference import TypeInferenceMapper
    from loopy.library.reduction import SegmentedSumReductionOperation
    from loopy.types import to_loopy_type
    op = SegmentedSumReductionOperation()

    knl = lp.make_kernel("{[i,j]: 0<=i<10 and 0<=j<i}", "")

    int32 = to_loopy_type(np.int32)

    expr = lp.symbolic.Reduction(
            operation=op,
            inames=("i",),
            expr=lp.symbolic.Reduction(
                operation=op,
                inames="j",
                expr=(1, 2),
                allow_simultaneous=True),
            allow_simultaneous=True)

    t_inf_mapper = TypeInferenceMapper(knl)

    assert (
            t_inf_mapper(expr, return_tuple=True, return_dtype_set=True)
            == [(int32, int32)])


def test_multi_argument_reduction_parsing():
    from loopy.symbolic import parse, Reduction

    assert isinstance(
            parse("reduce(argmax, i, reduce(argmax, j, i, j))").expr,
            Reduction)


def test_global_barrier_order_finding():
    knl = lp.make_kernel(
            "{[i,itrip]: 0<=i<n and 0<=itrip<ntrips}",
            """
            for i
                for itrip
                    ... gbarrier {id=top}
                    <> z[i] = z[i+1] + z[i]  {id=wr_z,dep=top}
                    <> v[i] = 11  {id=wr_v,dep=top}
                    ... gbarrier {dep=wr_z:wr_v,id=yoink}
                    z[i] = z[i] - z[i+1] + v[i] {id=iupd, dep=yoink}
                end
                ... nop {id=nop}
                ... gbarrier {dep=iupd,id=postloop}
                z[i] = z[i] - z[i+1] + v[i]  {id=zzzv,dep=postloop}
            end
            """)

    assert lp.get_global_barrier_order(knl) == ("top", "yoink", "postloop")

    for insn, barrier in (
            ("nop", None),
            ("top", None),
            ("wr_z", "top"),
            ("wr_v", "top"),
            ("yoink", "top"),
            ("postloop", "yoink"),
            ("zzzv", "postloop")):
        assert lp.find_most_recent_global_barrier(knl, insn) == barrier


def test_global_barrier_error_if_unordered():
    # FIXME: Should be illegal to declare this
    knl = lp.make_kernel("{[i]: 0 <= i < 10}",
            """
            ... gbarrier
            ... gbarrier
            """)

    from loopy.diagnostic import LoopyError
    with pytest.raises(LoopyError):
        lp.get_global_barrier_order(knl)


def test_struct_assignment(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    bbhit = np.dtype([
        ("tmin", np.float32),
        ("tmax", np.float32),
        ("bi", np.int32),
        ("hit", np.int32)])

    bbhit, bbhit_c_decl = cl.tools.match_dtype_to_c_struct(
            ctx.devices[0], "bbhit", bbhit)
    bbhit = cl.tools.get_or_register_dtype('bbhit', bbhit)

    preamble = bbhit_c_decl

    knl = lp.make_kernel(
        "{ [i]: 0<=i<N }",
        """
        for i
            result[i].hit = i % 2
            result[i].tmin = i
            result[i].tmax = i+10
            result[i].bi = i
        end
        """,
        [
            lp.GlobalArg("result", shape=("N",), dtype=bbhit),
            "..."],
        preambles=[("000", preamble)])

    knl = lp.set_options(knl, write_cl=True)
    knl(queue, N=200)


def test_inames_conditional_generation(ctx_factory):
    ctx = ctx_factory()
    knl = lp.make_kernel(
            "{[i,j,k]: 0 < k < i and 0 < j < 10 and 0 < i < 10}",
            """
            for k
                ... gbarrier
                <>tmp1 = 0
            end
            for j
                ... gbarrier
                <>tmp2 = i
            end
            """,
            "...",
            seq_dependencies=True)

    knl = lp.tag_inames(knl, dict(i="g.0"))

    with cl.CommandQueue(ctx) as queue:
        knl(queue)


def test_kernel_var_name_generator():
    knl = lp.make_kernel(
            "{[i]: 0 <= i <= 10}",
            """
            <>a = 0
            <>b_s0 = 0
            """)

    vng = knl.get_var_name_generator()

    assert vng("a_s0") != "a_s0"
    assert vng("b") != "b"


def test_fixed_parameters(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "[n] -> {[i]: 0 <= i < n}",
            """
            <>tmp[i] = i
            tmp[0] = 0
            """,
            fixed_parameters=dict(n=1))

    knl(queue)


def test_execution_backend_can_cache_dtypes(ctx_factory):
    # When the kernel is invoked, the execution backend uses it as a cache key
    # for the type inference and scheduling cache. This tests to make sure that
    # dtypes in the kernel can be cached, even though they may not have a
    # target.

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel("{[i]: 0 <= i < 10}", "<>tmp[i] = i")
    knl = lp.add_dtypes(knl, dict(tmp=int))

    knl(queue)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: foldmethod=marker

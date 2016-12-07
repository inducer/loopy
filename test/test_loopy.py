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
    assert (out == out_expect).all()


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
         silenced_warnings=frozenset({"read_no_write(b)"}))

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
        "(lid(0) == 0 && gid(0) == 0 ? 0 : -2 + 10 * gid(0) + 2 * lid(0)); " \
        "j <= (lid(0) == 0 && -1 + gid(0) == 0 ? 9 : 2 * lid(0)); ++j)"

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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: foldmethod=marker

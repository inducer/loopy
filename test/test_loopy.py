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


def test_extract_subst(ctx_factory):
    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            """
                a[i] = 23*b[i]**2 + 25*b[i]**2
                """)

    knl = lp.extract_subst(knl, "bsquare", "alpha*b[i]**2", "alpha")

    print(knl)

    from loopy.symbolic import parse

    insn, = knl.instructions
    assert insn.expression == parse("bsquare(23) + bsquare(25)")


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


def test_assume(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            "a[i] = a[i] + 1",
            [lp.GlobalArg("a", np.float32, shape="n"), "..."])

    knl = lp.split_iname(knl, "i", 16)
    knl = lp.set_loop_priority(knl, "i_outer,i_inner")
    knl = lp.assume(knl, "n mod 16 = 0")
    knl = lp.assume(knl, "n > 10")
    knl = lp.preprocess_kernel(knl, ctx.devices[0])
    kernel_gen = lp.generate_loop_schedules(knl)

    for gen_knl in kernel_gen:
        print(gen_knl)
        compiled = lp.CompiledKernel(ctx, gen_knl)
        print(compiled.get_code())
        assert "if" not in compiled.get_code()


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


def test_nonsense_reduction(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i]: 0<=i<100}",
            """
                a[i] = sum(i, 2)
                """,
            [lp.GlobalArg("a", np.float32, shape=(100,))]
            )

    import pytest
    with pytest.raises(RuntimeError):
        knl = lp.preprocess_kernel(knl, ctx.devices[0])


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


def test_join_inames(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<16}",
            [
                "b[i,j] = 2*a[i,j]"
                ],
            [
                lp.GlobalArg("a", np.float32, shape=(16, 16,)),
                lp.GlobalArg("b", np.float32, shape=(16, 16,))
                ],
            )

    ref_knl = knl

    knl = lp.add_prefetch(knl, "a", sweep_inames=["i", "j"])
    knl = lp.join_inames(knl, ["a_dim_0", "a_dim_1"])

    lp.auto_test_vs_ref(ref_knl, ctx, knl)


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
            assumptions="n>=1 and (exists zz: n = 16*zz)")

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 16)

    knl = lp.preprocess_kernel(knl, ctx.devices[0])
    for k in lp.generate_loop_schedules(knl):
        code = lp.generate_code(k)
        assert "if" not in code

    lp.auto_test_vs_ref(ref_knl, ctx, knl,
            parameters={"n": 16**3})


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


def test_stencil(ctx_factory):
    ctx = ctx_factory()

    # n=32 causes corner case behavior in size calculations for temprorary (a
    # non-unifiable, two-constant-segments PwAff as the base index)

    n = 256
    knl = lp.make_kernel(
            "{[i,j]: 0<= i,j < %d}" % n,
            [
                "a_offset(ii, jj) := a[ii+1, jj+1]",
                "z[i,j] = -2*a_offset(i,j)"
                " + a_offset(i,j-1)"
                " + a_offset(i,j+1)"
                " + a_offset(i-1,j)"
                " + a_offset(i+1,j)"
                ],
            [
                lp.GlobalArg("a", np.float32, shape=(n+2, n+2,)),
                lp.GlobalArg("z", np.float32, shape=(n+2, n+2,))
                ])

    ref_knl = knl

    def variant_1(knl):
        knl = lp.split_iname(knl, "i", 16, outer_tag="g.1", inner_tag="l.1")
        knl = lp.split_iname(knl, "j", 16, outer_tag="g.0", inner_tag="l.0")
        knl = lp.add_prefetch(knl, "a", ["i_inner", "j_inner"])
        knl = lp.set_loop_priority(knl, ["a_dim_0_outer", "a_dim_1_outer"])
        return knl

    def variant_2(knl):
        knl = lp.split_iname(knl, "i", 16, outer_tag="g.1", inner_tag="l.1")
        knl = lp.split_iname(knl, "j", 16, outer_tag="g.0", inner_tag="l.0")
        knl = lp.add_prefetch(knl, "a", ["i_inner", "j_inner"],
                fetch_bounding_box=True)
        knl = lp.set_loop_priority(knl, ["a_dim_0_outer", "a_dim_1_outer"])
        return knl

    for variant in [
            #variant_1,
            variant_2,
            ]:
        lp.auto_test_vs_ref(ref_knl, ctx, variant(knl),
                print_ref_code=False,
                op_count=[n*n], op_label=["cells"])


def test_stencil_with_overfetch(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<= i,j < n}",
            [
                "a_offset(ii, jj) := a[ii+2, jj+2]",
                "z[i,j] = -2*a_offset(i,j)"
                " + a_offset(i,j-1)"
                " + a_offset(i,j+1)"
                " + a_offset(i-1,j)"
                " + a_offset(i+1,j)"

                " + a_offset(i,j-2)"
                " + a_offset(i,j+2)"
                " + a_offset(i-2,j)"
                " + a_offset(i+2,j)"
                ],
            assumptions="n>=1")

    if ctx.devices[0].platform.name == "Portable Computing Language":
        # https://github.com/pocl/pocl/issues/205
        pytest.skip("takes very long to compile on pocl")

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32))

    ref_knl = knl

    def variant_overfetch(knl):
        knl = lp.split_iname(knl, "i", 16, outer_tag="g.1", inner_tag="l.1",
                slabs=(1, 1))
        knl = lp.split_iname(knl, "j", 16, outer_tag="g.0", inner_tag="l.0",
               slabs=(1, 1))
        knl = lp.add_prefetch(knl, "a", ["i_inner", "j_inner"],
                fetch_bounding_box=True)
        knl = lp.set_loop_priority(knl, ["a_dim_0_outer", "a_dim_1_outer"])
        return knl

    for variant in [variant_overfetch]:
        n = 200
        lp.auto_test_vs_ref(ref_knl, ctx, variant(knl),
                print_ref_code=False,
                op_count=[n*n], parameters=dict(n=n), op_label=["cells"])


def test_eq_constraint(ctx_factory):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<= i,j < 32}",
            [
                "a[i] = b[i]"
                ],
            [
                lp.GlobalArg("a", np.float32, shape=(1000,)),
                lp.GlobalArg("b", np.float32, shape=(1000,))
                ])

    knl = lp.split_iname(knl, "i", 16, outer_tag="g.0")
    knl = lp.split_iname(knl, "i_inner", 16, outer_tag=None, inner_tag="l.0")

    knl = lp.preprocess_kernel(knl, ctx.devices[0])
    kernel_gen = lp.generate_loop_schedules(knl)

    for knl in kernel_gen:
        print(lp.generate_code(knl))


def test_argmax(ctx_factory):
    logging.basicConfig(level=logging.INFO)

    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    order = "C"

    n = 10000

    knl = lp.make_kernel(
            "{[i]: 0<=i<%d}" % n,
            [
                "<> result = argmax(i, fabs(a[i]))",
                "max_idx = result.index",
                "max_val = result.value",
                ],
            [
                lp.GlobalArg("a", dtype, shape=(n,), order=order),
                lp.GlobalArg("max_idx", np.int32, shape=(), order=order),
                lp.GlobalArg("max_val", dtype, shape=(), order=order),
                ])

    a = np.random.randn(10000).astype(dtype)
    cknl = lp.CompiledKernel(ctx, knl)
    evt, (max_idx, max_val) = cknl(queue, a=a, out_host=True)
    assert max_val == np.max(np.abs(a))
    assert max_idx == np.where(np.abs(a) == max_val)[-1]


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


def test_empty_reduction(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            [
                "{[i]: 0<=i<20}",
                "[i] -> {[j]: 0<=j<0}"
                ],
            [
                "a[i] = sum(j, j)",
                ],
            [
                lp.GlobalArg("a", dtype, (20,)),
                ])
    cknl = lp.CompiledKernel(ctx, knl)

    evt, (a,) = cknl(queue)

    assert (a.get() == 0).all()


def test_nested_dependent_reduction(ctx_factory):
    dtype = np.dtype(np.int32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            [
                "{[i]: 0<=i<n}",
                "{[j]: 0<=j<i+sumlen}"
                ],
            [
                "<> sumlen = l[i]",
                "a[i] = sum(j, j)",
                ],
            [
                lp.ValueArg("n", np.int32),
                lp.GlobalArg("a", dtype, ("n",)),
                lp.GlobalArg("l", np.int32, ("n",)),
                ])

    cknl = lp.CompiledKernel(ctx, knl)

    n = 330
    l = np.arange(n, dtype=np.int32)
    evt, (a,) = cknl(queue, l=l, n=n, out_host=True)

    tgt_result = (2*l-1)*2*l/2
    assert (a == tgt_result).all()


def test_multi_nested_dependent_reduction(ctx_factory):
    dtype = np.dtype(np.int32)
    ctx = ctx_factory()

    knl = lp.make_kernel(
            [
                "{[itgt]: 0 <= itgt < ntgts}",
                "{[isrc_box]: 0 <= isrc_box < nboxes}",
                "{[isrc]: 0 <= isrc < npart}"
                ],
            [
                "<> npart = nparticles_per_box[isrc_box]",
                "a[itgt] = sum((isrc_box, isrc), 1)",
                ],
            [
                lp.ValueArg("n", np.int32),
                lp.GlobalArg("a", dtype, ("n",)),
                lp.GlobalArg("nparticles_per_box", np.int32, ("nboxes",)),
                lp.ValueArg("ntgts", np.int32),
                lp.ValueArg("nboxes", np.int32),
                ],
            assumptions="ntgts>=1")

    cknl = lp.CompiledKernel(ctx, knl)
    print(cknl.get_code())
    # FIXME: Actually test functionality.


def test_recursive_nested_dependent_reduction(ctx_factory):
    dtype = np.dtype(np.int32)
    ctx = ctx_factory()

    knl = lp.make_kernel(
            [
                "{[itgt]: 0 <= itgt < ntgts}",
                "{[isrc_box]: 0 <= isrc_box < nboxes}",
                "{[isrc]: 0 <= isrc < npart}"
                ],
            [
                "<> npart = nparticles_per_box[isrc_box]",
                "<> boxsum = sum(isrc, isrc+isrc_box+itgt)",
                "a[itgt] = sum(isrc_box, boxsum)",
                ],
            [
                lp.ValueArg("n", np.int32),
                lp.GlobalArg("a", dtype, ("n",)),
                lp.GlobalArg("nparticles_per_box", np.int32, ("nboxes",)),
                lp.ValueArg("ntgts", np.int32),
                lp.ValueArg("nboxes", np.int32),
                ],
            assumptions="ntgts>=1")

    cknl = lp.CompiledKernel(ctx, knl)
    print(cknl.get_code())
    # FIXME: Actually test functionality.


def test_dependent_loop_bounds(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()

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
            assumptions="n>=1 and row_len>=1")

    cknl = lp.CompiledKernel(ctx, knl)
    print("---------------------------------------------------")
    print(cknl.get_highlighted_code())
    print("---------------------------------------------------")


def test_dependent_loop_bounds_2(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()

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
            assumptions="n>=1 and row_len>=1")

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0",
            inner_tag="l.0")
    cknl = lp.CompiledKernel(ctx, knl)
    print("---------------------------------------------------")
    print(cknl.get_highlighted_code())
    print("---------------------------------------------------")


def test_dependent_loop_bounds_3(ctx_factory):
    # The point of this test is that it shows a dependency between
    # domains that is exclusively mediated by the row_len temporary.
    # It also makes sure that row_len gets read before any
    # conditionals use it.

    dtype = np.dtype(np.float32)
    ctx = ctx_factory()

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
                ])

    assert knl.parents_per_domain()[1] == 0

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0",
            inner_tag="l.0")

    cknl = lp.CompiledKernel(ctx, knl)
    print("---------------------------------------------------")
    print(cknl.get_highlighted_code())
    print("---------------------------------------------------")

    knl_bad = lp.split_iname(knl, "jj", 128, outer_tag="g.1",
            inner_tag="l.1")

    knl = lp.preprocess_kernel(knl, ctx.devices[0])

    import pytest
    with pytest.raises(RuntimeError):
        list(lp.generate_loop_schedules(knl_bad))


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
                ])

    knl = lp.split_iname(knl, "i", 16, outer_tag="g.0",
            inner_tag="l.0")
    knl = lp.split_iname(knl, "j", 16, outer_tag="g.0",
            inner_tag="l.0")
    assert knl.parents_per_domain() == 2*[None]

    n = 50
    cknl = lp.CompiledKernel(ctx, knl)
    evt, (a, b) = cknl(queue, n=n, out_host=True)

    assert a.shape == (50,)
    assert b.shape == (50,)
    assert (a == 1).all()
    assert (b == 2).all()


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
                "b[i,k] = 22 {dep=set_all}",
                ],
            [
                lp.GlobalArg("a,b", dtype, shape="n, n", order=order),
                lp.ValueArg("n", np.int32, approximately=1000),
                ],
            name="equality_constraints", assumptions="n>=1")

    seq_knl = knl

    knl = lp.split_iname(knl, "i", 16, outer_tag="g.0", inner_tag="l.0")
    knl = lp.split_iname(knl, "j", 16, outer_tag="g.1", inner_tag="l.1")
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


def test_double_sum(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 20

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n }",
            [
                "a = sum((i,j), i*j)",
                "b = sum(i, sum(j, i*j))",
                ],
            assumptions="n>=1")

    cknl = lp.CompiledKernel(ctx, knl)

    evt, (a, b) = cknl(queue, n=n)

    ref = sum(i*j for i in range(n) for j in range(n))
    assert a.get() == ref
    assert b.get() == ref


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
                a = 1.5 + sum((i,j), i*j)
                d = 1.5 + sum((i,j), b[i,j])
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


def test_triangle_domain(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n and i <= j}",
            "a[i,j] = 17",
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


# {{{ convolutions

def test_convolution(ctx_factory):
    ctx = ctx_factory()

    dtype = np.float32

    knl = lp.make_kernel(
        "{ [iimg, ifeat, icolor, im_x, im_y, f_x, f_y]: \
                -f_w <= f_x,f_y <= f_w \
                and 0 <= im_x < im_w and 0 <= im_y < im_h \
                and 0<=iimg<=nimgs and 0<=ifeat<nfeats and 0<=icolor<ncolors \
                }",
        """
        out[iimg, ifeat, im_x, im_y] = sum((f_x, f_y, icolor), \
            img[iimg, f_w+im_x-f_x, f_w+im_y-f_y, icolor] \
            * f[ifeat, f_w+f_x, f_w+f_y, icolor])
        """,
        [
            lp.GlobalArg("f", dtype, shape=lp.auto),
            lp.GlobalArg("img", dtype, shape=lp.auto),
            lp.GlobalArg("out", dtype, shape=lp.auto),
            "..."
            ],
        assumptions="f_w>=1 and im_w, im_h >= 2*f_w+1 and nfeats>=1 and nimgs>=0",
        flags="annotate_inames",
        defines=dict(ncolors=3))

    f_w = 3

    knl = lp.fix_parameters(knl, f_w=f_w)

    ref_knl = knl

    def variant_0(knl):
        #knl = lp.split_iname(knl, "im_x", 16, inner_tag="l.0")
        knl = lp.set_loop_priority(knl, "iimg,im_x,im_y,ifeat,f_x,f_y")
        return knl

    def variant_1(knl):
        knl = lp.split_iname(knl, "im_x", 16, inner_tag="l.0")
        knl = lp.set_loop_priority(knl, "iimg,im_x_outer,im_y,ifeat,f_x,f_y")
        return knl

    def variant_2(knl):
        knl = lp.split_iname(knl, "im_x", 16, outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_iname(knl, "im_y", 16, outer_tag="g.1", inner_tag="l.1")
        knl = lp.tag_inames(knl, dict(ifeat="g.2"))
        knl = lp.add_prefetch(knl, "f[ifeat,:,:,:]")
        knl = lp.add_prefetch(knl, "img", "im_x_inner, im_y_inner, f_x, f_y")
        return knl

    for variant in [
            variant_0,
            variant_1,
            variant_2
            ]:
        lp.auto_test_vs_ref(ref_knl, ctx, variant(knl),
                parameters=dict(
                    im_w=128, im_h=128, f_w=f_w,
                    nfeats=3, nimgs=3
                    ))


def test_convolution_with_nonzero_base(ctx_factory):
    # This is kept alive as a test for domains that don't start at zero.
    # These are a bad idea for split_iname, which places its origin at zero
    # and therefore produces a first block that is odd-sized.
    #
    # Therefore, for real tests, check test_convolution further up.

    ctx = ctx_factory()

    dtype = np.float32

    knl = lp.make_kernel(
        "{ [iimg, ifeat, icolor, im_x, im_y, f_x, f_y]: \
                -f_w <= f_x,f_y <= f_w \
                and f_w <= im_x < im_w-f_w and f_w <= im_y < im_h-f_w \
                and 0<=iimg<=nimgs and 0<=ifeat<nfeats and 0<=icolor<ncolors \
                }",
        """
        out[iimg, ifeat, im_x-f_w, im_y-f_w] = sum((f_x, f_y, icolor), \
            img[iimg, im_x-f_x, im_y-f_y, icolor] \
            * f[ifeat, f_w+f_x, f_w+f_y, icolor])
        """,
        [
            lp.GlobalArg("f", dtype, shape=lp.auto),
            lp.GlobalArg("img", dtype, shape=lp.auto),
            lp.GlobalArg("out", dtype, shape=lp.auto),
            "..."
            ],
        assumptions="f_w>=1 and im_w, im_h >= 2*f_w+1 and nfeats>=1 and nimgs>=0",
        flags="annotate_inames",
        defines=dict(ncolors=3))

    ref_knl = knl

    f_w = 3

    def variant_0(knl):
        #knl = lp.split_iname(knl, "im_x", 16, inner_tag="l.0")
        knl = lp.set_loop_priority(knl, "iimg,im_x,im_y,ifeat,f_x,f_y")
        return knl

    def variant_1(knl):
        knl = lp.split_iname(knl, "im_x", 16, inner_tag="l.0")
        knl = lp.set_loop_priority(knl, "iimg,im_x_outer,im_y,ifeat,f_x,f_y")
        return knl

    for variant in [
            variant_0,
            variant_1,
            ]:
        lp.auto_test_vs_ref(ref_knl, ctx, variant(knl),
                parameters=dict(
                    im_w=128, im_h=128, f_w=f_w,
                    nfeats=12, nimgs=17
                    ))

# }}}


def test_c_instruction(ctx_factory):
    #logging.basicConfig(level=logging.DEBUG)
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n }",
            [
                lp.CInstruction("i", """
                    x = sin((float) i);
                    """, assignees="x"),
                "a[i*i] = x",
                ],
            [
                lp.GlobalArg("a", shape="n"),
                lp.ValueArg("n"),
                lp.TemporaryVariable("x", np.float32),
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


def test_split_reduction(ctx_factory):
    knl = lp.make_kernel(
            "{[i,j,k]: 0<=i,j,k<n}",
            """
                b = sum((i,j,k), a[i,j,k])
                """,
            [
                lp.GlobalArg("box_source_starts,box_source_counts_nonchild,a",
                    None, shape=None),
                "..."])

    knl = lp.split_reduction_outward(knl, "j,k")
    # FIXME: finish test


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


def test_rob_stroud_bernstein(ctx_factory):
    ctx = ctx_factory()

    # NOTE: tmp would have to be zero-filled beforehand

    knl = lp.make_kernel(
            "{[el, i2, alpha1,alpha2]: \
                    0 <= el < nels and \
                    0 <= i2 < nqp1d and \
                    0 <= alpha1 <= deg and 0 <= alpha2 <= deg-alpha1 }",
            """
                <> xi = qpts[1, i2] {inames=+el}
                <> s = 1-xi
                <> r = xi/s
                <> aind = 0 {id=aind_init,inames=+i2:el}

                <> w = s**(deg-alpha1) {id=init_w}

                tmp[el,alpha1,i2] = tmp[el,alpha1,i2] + w * coeffs[aind] \
                        {id=write_tmp,inames=+alpha2}
                w = w * r * ( deg - alpha1 - alpha2 ) / (1 + alpha2) \
                        {id=update_w,dep=init_w:write_tmp}
                aind = aind + 1 \
                        {id=aind_incr,\
                        dep=aind_init:write_tmp:update_w, \
                        inames=+el:i2:alpha1:alpha2}
                """,
            [
                # Must declare coeffs to have "no" shape, to keep loopy
                # from trying to figure it out the shape automatically.

                lp.GlobalArg("coeffs", None, shape=None),
                "..."
                ],
            assumptions="deg>=0 and nels>=1"
            )

    knl = lp.fix_parameters(knl, nqp1d=7, deg=4)
    knl = lp.split_iname(knl, "el", 16, inner_tag="l.0")
    knl = lp.split_iname(knl, "el_outer", 2, outer_tag="g.0", inner_tag="ilp",
            slabs=(0, 1))
    knl = lp.tag_inames(knl, dict(i2="l.1", alpha1="unr", alpha2="unr"))

    print(lp.CompiledKernel(ctx, knl).get_highlighted_code(
            dict(
                qpts=np.float32,
                coeffs=np.float32,
                tmp=np.float32,
                )))


def test_rob_stroud_bernstein_full(ctx_factory):
    #logging.basicConfig(level=logging.DEBUG)
    ctx = ctx_factory()

    # NOTE: result would have to be zero-filled beforehand

    knl = lp.make_kernel(
            "{[el, i2, alpha1,alpha2, i1_2, alpha1_2, i2_2]: \
                    0 <= el < nels and \
                    0 <= i2 < nqp1d and \
                    0 <= alpha1 <= deg and 0 <= alpha2 <= deg-alpha1 and\
                    \
                    0 <= i1_2 < nqp1d and \
                    0 <= alpha1_2 <= deg and \
                    0 <= i2_2 < nqp1d \
                    }",
            """
                <> xi = qpts[1, i2] {inames=+el}
                <> s = 1-xi
                <> r = xi/s
                <> aind = 0 {id=aind_init,inames=+i2:el}

                <> w = s**(deg-alpha1) {id=init_w}

                <> tmp[alpha1,i2] = tmp[alpha1,i2] + w * coeffs[aind] \
                        {id=write_tmp,inames=+alpha2}
                w = w * r * ( deg - alpha1 - alpha2 ) / (1 + alpha2) \
                        {id=update_w,dep=init_w:write_tmp}
                aind = aind + 1 \
                        {id=aind_incr,\
                        dep=aind_init:write_tmp:update_w, \
                        inames=+el:i2:alpha1:alpha2}

                <> xi2 = qpts[0, i1_2] {dep=aind_incr,inames=+el}
                <> s2 = 1-xi2
                <> r2 = xi2/s2
                <> w2 = s2**deg

                result[el, i1_2, i2_2] = result[el, i1_2, i2_2] + \
                        w2 * tmp[alpha1_2, i2_2] \
                        {inames=el:alpha1_2:i1_2:i2_2}

                w2 = w2 * r2 * (deg-alpha1_2) / (1+alpha1_2)
                """,
            [
                # Must declare coeffs to have "no" shape, to keep loopy
                # from trying to figure it out the shape automatically.

                lp.GlobalArg("coeffs", None, shape=None),
                "..."
                ],
            assumptions="deg>=0 and nels>=1"
            )

    knl = lp.fix_parameters(knl, nqp1d=7, deg=4)

    if 0:
        knl = lp.split_iname(knl, "el", 16, inner_tag="l.0")
        knl = lp.split_iname(knl, "el_outer", 2, outer_tag="g.0", inner_tag="ilp",
                slabs=(0, 1))
        knl = lp.tag_inames(knl, dict(i2="l.1", alpha1="unr", alpha2="unr"))

    from pickle import dumps, loads
    knl = loads(dumps(knl))

    knl = lp.CompiledKernel(ctx, knl).get_highlighted_code(
            dict(
                qpts=np.float32,
                tmp=np.float32,
                coeffs=np.float32,
                result=np.float32,
                ))
    print(knl)


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


def test_tag_data_axes(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{ [i,j,k]: 0<=i,j,k<n }",
            "out[i,j,k] = 15")

    ref_knl = knl

    with pytest.raises(lp.LoopyError):
        lp.tag_data_axes(knl, "out", "N1,N0,N5")

    with pytest.raises(lp.LoopyError):
        lp.tag_data_axes(knl, "out", "N1,N0,c")

    knl = lp.tag_data_axes(knl, "out", "N1,N0,N2")
    knl = lp.tag_inames(knl, dict(j="g.0", i="g.1"))

    lp.auto_test_vs_ref(ref_knl, ctx, knl,
            parameters=dict(n=20))


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

    knl = lp.set_loop_priority(knl, "j,i,k")
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
        knl = lp.set_loop_priority(knl, "i_outer")

        a = cl.clrandom.rand(queue, 20, np.float32)
        a_ref = a.copy()
        a_knl = a.copy()

        knl = lp.set_options(knl, "write_cl")
        knl(queue, a=a_knl)
        ref_knl(queue, a=a_ref)

        queue.finish()

        assert (a_ref == a_knl).get().all()


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


def test_fd_demo():
    knl = lp.make_kernel(
        "{[i,j]: 0<=i,j<n}",
        "result[i,j] = u[i, j]**2 + -1 + (-4)*u[i + 1, j + 1] \
                + u[i + 1 + 1, j + 1] + u[i + 1 + -1, j + 1] \
                + u[i + 1, j + 1 + 1] + u[i + 1, j + 1 + -1]")
    #assumptions="n mod 16=0")
    knl = lp.split_iname(knl,
            "i", 16, outer_tag="g.1", inner_tag="l.1")
    knl = lp.split_iname(knl,
            "j", 16, outer_tag="g.0", inner_tag="l.0")
    knl = lp.add_prefetch(knl, "u",
            ["i_inner", "j_inner"],
            fetch_bounding_box=True)

    #n = 1000
    #u = cl.clrandom.rand(queue, (n+2, n+2), dtype=np.float32)

    knl = lp.set_options(knl, write_cl=True)
    knl = lp.add_and_infer_dtypes(knl, dict(u=np.float32))
    knl = lp.preprocess_kernel(knl)
    knl = lp.get_one_scheduled_kernel(knl)
    code, inf = lp.generate_code(knl)
    print(code)

    assert "double" not in code


def test_fd_1d(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        "result[i] = u[i+1]-u[i]")

    knl = lp.add_and_infer_dtypes(knl, {"u": np.float32})
    ref_knl = knl

    knl = lp.split_iname(knl, "i", 16)
    knl = lp.extract_subst(knl, "u_acc", "u[j]", parameters="j")
    knl = lp.precompute(knl, "u_acc", "i_inner", default_tag="for")
    knl = lp.assume(knl, "n mod 16 = 0")

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
            parameters=dict(n=2048))


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


def test_set_arg_order():
    knl = lp.make_kernel(
            "{ [i,j]: 0<=i,j<n }",
            "out[i,j] = a[i]*b[j]")

    knl = lp.set_argument_order(knl, "out,a,n,b")


def test_affine_map_inames():
    knl = lp.make_kernel(
        "{[e, i,j,n]: 0<=e<E and 0<=i,j,n<N}",
        "rhsQ[e, n+i, j] = rhsQ[e, n+i, j] - D[i, n]*x[i,j]")

    knl = lp.affine_map_inames(knl,
            "i", "i0",
            "i0 = n+i")

    print(knl)


def test_precompute_confusing_subst_arguments(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[i,j]: 0<=i<n and 0<=j<5}",
        """
        D(i):=a[i+1]-a[i]
        b[i,j] = D(j)
        """)

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32))

    ref_knl = knl

    knl = lp.tag_inames(knl, dict(j="g.1"))
    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    from loopy.symbolic import get_dependencies
    assert "i_inner" not in get_dependencies(knl.substitutions["D"].expression)
    knl = lp.precompute(knl, "D")

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
            parameters=dict(n=12345))


def test_precompute_nested_subst(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[i,j]: 0<=i<n and 0<=j<5}",
        """
        E:=a[i]
        D:=E*E
        b[i] = D
        """)

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32))

    ref_knl = knl

    knl = lp.tag_inames(knl, dict(j="g.1"))
    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    from loopy.symbolic import get_dependencies
    assert "i_inner" not in get_dependencies(knl.substitutions["D"].expression)
    knl = lp.precompute(knl, "D", "i_inner")

    # There's only one surviving 'E' rule.
    assert len([
        rule_name
        for rule_name in knl.substitutions
        if rule_name.startswith("E")]) == 1

    # That rule should use the newly created prefetch inames,
    # not the prior 'i_inner'
    assert "i_inner" not in get_dependencies(knl.substitutions["E"].expression)

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
            parameters=dict(n=12345))


def test_poisson(ctx_factory):
    # Stolen from Peter Coogan and Rob Kirby for FEM assembly
    ctx = ctx_factory()

    nbf = 5
    nqp = 5
    sdim = 3

    knl = lp.make_kernel(
            "{ [c,i,j,k,ell,ell2]: \
            0 <= c < nels and \
            0 <= i < nbf and \
            0 <= j < nbf and \
            0 <= k < nqp and \
            0 <= ell < sdim and \
            0 <= ell2 < sdim }",
            """
            dpsi(bf,k0,dir) := sum(ell2, DFinv[c,ell2,dir] * DPsi[bf,k0,ell2] )
            Ael[c,i,j] = J[c] * w[k] * sum(ell, dpsi(i,k,ell) * dpsi(j,k,ell))
            """,
            assumptions="nels>=1 and nbf >= 1 and nels mod 4 = 0")

    knl = lp.fix_parameters(knl, nbf=nbf, sdim=sdim, nqp=nqp)

    ref_knl = knl

    knl = lp.set_loop_priority(knl, ["c", "j", "i", "k"])

    def variant_1(knl):
        knl = lp.precompute(knl, "dpsi", "i,k,ell", default_tag='for')
        knl = lp.set_loop_priority(knl, "c,i,j")
        return knl

    def variant_2(knl):
        knl = lp.precompute(knl, "dpsi", "i,ell", default_tag='for')
        knl = lp.set_loop_priority(knl, "c,i,j")
        return knl

    def add_types(knl):
        return lp.add_and_infer_dtypes(knl, dict(
            w=np.float32,
            J=np.float32,
            DPsi=np.float32,
            DFinv=np.float32,
            ))

    for variant in [
            #variant_1,
            variant_2
            ]:
        knl = variant(knl)

        lp.auto_test_vs_ref(
                add_types(ref_knl), ctx, add_types(knl),
                parameters=dict(n=5, nels=15, nbf=5, sdim=2, nqp=7))


def test_auto_test_can_detect_problems(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[i,j]: 0<=i,j<n}",
        """
        a[i,j] = 25
        """)

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32))

    ref_knl = knl

    knl = lp.link_inames(knl, "i,j", "i0")

    from loopy.diagnostic import AutomaticTestFailure
    with pytest.raises(AutomaticTestFailure):
        lp.auto_test_vs_ref(
                ref_knl, ctx, knl,
                parameters=dict(n=123))


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
    l_sum = partial(lp.Reduction, "sum")

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


def test_precompute_with_preexisting_inames(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[e,i,j,k]: 0<=e<E and 0<=i,j,k<n}",
        """
        result[e,i] = sum(j, D1[i,j]*u[e,j])
        result2[e,i] = sum(k, D2[i,k]*u[e,k])
        """)

    knl = lp.add_and_infer_dtypes(knl, {
        "u": np.float32,
        "D1": np.float32,
        "D2": np.float32,
        })

    knl = lp.fix_parameters(knl, n=13)

    ref_knl = knl

    knl = lp.extract_subst(knl, "D1_subst", "D1[ii,jj]", parameters="ii,jj")
    knl = lp.extract_subst(knl, "D2_subst", "D2[ii,jj]", parameters="ii,jj")

    knl = lp.precompute(knl, "D1_subst", "i,j", default_tag="for",
            precompute_inames="ii,jj")
    knl = lp.precompute(knl, "D2_subst", "i,k", default_tag="for",
            precompute_inames="ii,jj")

    knl = lp.set_loop_priority(knl, "ii,jj,e,j,k")

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
            parameters=dict(E=200))


def test_precompute_with_preexisting_inames_fail():
    knl = lp.make_kernel(
        "{[e,i,j,k]: 0<=e<E and 0<=i,j<n and 0<=k<2*n}",
        """
        result[e,i] = sum(j, D1[i,j]*u[e,j])
        result2[e,i] = sum(k, D2[i,k]*u[e,k])
        """)

    knl = lp.add_and_infer_dtypes(knl, {
        "u": np.float32,
        "D1": np.float32,
        "D2": np.float32,
        })

    knl = lp.fix_parameters(knl, n=13)

    knl = lp.extract_subst(knl, "D1_subst", "D1[ii,jj]", parameters="ii,jj")
    knl = lp.extract_subst(knl, "D2_subst", "D2[ii,jj]", parameters="ii,jj")

    knl = lp.precompute(knl, "D1_subst", "i,j", default_tag="for",
            precompute_inames="ii,jj")
    with pytest.raises(lp.LoopyError):
        lp.precompute(knl, "D2_subst", "i,k", default_tag="for",
                precompute_inames="ii,jj")


def test_vectorize(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        """
        <> temp = 2*b[i]
        a[i] = temp
        """)
    knl = lp.add_and_infer_dtypes(knl, dict(b=np.float32))
    knl = lp.set_array_dim_names(knl, "a,b", "i")
    knl = lp.split_array_dim(knl, [("a", 0), ("b", 0)], 4,
            split_kwargs=dict(slabs=(0, 1)))

    knl = lp.tag_data_axes(knl, "a,b", "c,vec")
    ref_knl = knl
    ref_knl = lp.tag_inames(ref_knl, {"i_inner": "unr"})

    knl = lp.tag_inames(knl, {"i_inner": "vec"})

    knl = lp.preprocess_kernel(knl)
    knl = lp.get_one_scheduled_kernel(knl)
    code, inf = lp.generate_code(knl)

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
            parameters=dict(n=30))


def test_alias_temporaries(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        """
        times2(i) := 2*a[i]
        times3(i) := 3*a[i]
        times4(i) := 4*a[i]

        x[i] = times2(i)
        y[i] = times3(i)
        z[i] = times4(i)
        """)

    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 16, outer_tag="g.0", inner_tag="l.0")

    knl = lp.precompute(knl, "times2", "i_inner")
    knl = lp.precompute(knl, "times3", "i_inner")
    knl = lp.precompute(knl, "times4", "i_inner")

    knl = lp.alias_temporaries(knl, ["times2_0", "times3_0", "times4_0"])

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl,
            parameters=dict(n=30))


def test_fusion():
    exp_kernel = lp.make_kernel(
         ''' { [i]: 0<=i<n } ''',
         ''' exp[i] = pow(E, z[i])''',
         assumptions="n>0")

    sum_kernel = lp.make_kernel(
        '{ [j]: 0<=j<n }',
        'out2 = sum(j, exp[j])',
        assumptions='n>0')

    knl = lp.fuse_kernels([exp_kernel, sum_kernel])

    print(knl)


def test_sci_notation_literal(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    set_kernel = lp.make_kernel(
         ''' { [i]: 0<=i<12 } ''',
         ''' out[i] = 1e-12''')

    set_kernel = lp.set_options(set_kernel, write_cl=True)

    evt, (out,) = set_kernel(queue)

    assert (np.abs(out.get() - 1e-12) < 1e-20).all()


def test_to_batched(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
         ''' { [i,j]: 0<=i,j<n } ''',
         ''' out[i] = sum(j, a[i,j]*x[j])''')

    bknl = lp.to_batched(knl, "nbatches", "out,x")

    a = np.random.randn(5, 5)
    x = np.random.randn(7, 5)

    bknl(queue, a=a, x=x)


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

    fused_knl = lp.fuse_kernels([fin_diff_knl, flux_knl])

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


def test_is_expression_equal():
    from loopy.symbolic import is_expression_equal
    from pymbolic import var

    x = var("x")
    y = var("y")

    assert is_expression_equal(x+2, 2+x)

    assert is_expression_equal((x+2)**2, x**2 + 4*x + 4)
    assert is_expression_equal((x+y)**2, x**2 + 2*x*y + y**2)


def test_collect_common_factors(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,j,k]: 0<=i,j<n}",
            """
            <float32> out_tmp = 0 {id=out_init,inames=i}
            out_tmp = out_tmp + alpha[i]*a[i,j]*b1[j] {id=out_up1,dep=out_init}
            out_tmp = out_tmp + alpha[i]*a[j,i]*b2[j] {id=out_up2,dep=out_init}
            out[i] = out_tmp {dep=out_up1:out_up2}
            """)
    knl = lp.add_and_infer_dtypes(knl,
            dict(a=np.float32, alpha=np.float32, b1=np.float32, b2=np.float32))

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 256, outer_tag="g.0", inner_tag="l.0")
    knl = lp.collect_common_factors_on_increment(knl, "out_tmp")

    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=13))


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

    print(
            lp.generate_code(
                lp.get_one_scheduled_kernel(
                    lp.preprocess_kernel(knl)))[0])


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


def test_chunk_iname(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = 2*a[i]",
            [
                lp.GlobalArg("out,a", np.float32, shape=lp.auto),
                "..."
                ],
            assumptions="n>0")

    ref_knl = knl
    knl = lp.chunk_iname(knl, "i", 3, inner_tag="l.0")
    knl = lp.set_loop_priority(knl, "i_outer, i_inner")
    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=130))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: foldmethod=marker

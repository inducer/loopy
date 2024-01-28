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
import pyopencl.array  # noqa
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
        "cl"  # "cl.create_some_context"
        ]


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


def test_globals_decl_once_with_multi_subprogram(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    np.random.seed(17)
    a = np.random.randn(16)
    cnst = np.random.randn(16)
    knl = lp.make_kernel(
            "{[i, ii]: 0<=i, ii<n}",
            """
            out[i] = a[i]+cnst[i]{id=first}
            out[ii] = 2*out[ii]+cnst[ii]{id=second}
            """,
            [
                lp.TemporaryVariable(
                    "cnst", initializer=cnst, address_space=lp.AddressSpace.GLOBAL,
                    read_only=True),
                lp.GlobalArg("out", is_input=False, shape=lp.auto),
                "..."])
    knl = lp.fix_parameters(knl, n=16)
    knl = lp.add_barrier(knl, "id:first", "id:second")

    knl = lp.split_iname(knl, "i", 2, outer_tag="g.0", inner_tag="l.0")
    knl = lp.split_iname(knl, "ii", 2, outer_tag="g.0", inner_tag="l.0")
    evt, (out,) = knl(queue, a=a)
    assert np.linalg.norm(out-(2*(a+cnst)+cnst)) <= 1e-15


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

    sr_keys = list(knl["loopy_kernel"].substitutions.keys())
    for letter, how_many in [
            ("f", 1),
            ("g", 1),
            ("h", 2)
            ]:
        substs_with_letter = sum(1 for k in sr_keys if k.startswith(letter))
        assert substs_with_letter == how_many


def test_type_inference_no_artificial_doubles():
    prog = lp.make_kernel(
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
            assumptions="n>=1",
            target=lp.PyOpenCLTarget())

    code = lp.generate_code_v2(prog).device_code()
    assert "double" not in code


def test_type_inference_with_type_dependencies():
    prog = lp.make_kernel(
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
    prog = lp.infer_unknown_types(prog)

    from loopy.types import to_loopy_type
    assert prog["loopy_kernel"].temporary_variables["a"].dtype == to_loopy_type(
            np.int32)
    assert prog["loopy_kernel"].temporary_variables["b"].dtype == to_loopy_type(
            np.float32)
    assert prog["loopy_kernel"].temporary_variables["c"].dtype == to_loopy_type(
            np.float32)
    assert prog["loopy_kernel"].temporary_variables["d"].dtype == to_loopy_type(
            np.complex128)


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


def test_simple_side_effect():
    knl = lp.make_kernel(
            "{[i]: 0<=i<100}",
            """
                a[i] = a[i] + 1
                """,
            [lp.GlobalArg("a", np.float32, shape=(100,))],
            target=lp.PyOpenCLTarget()
            )

    print(knl)
    print(lp.generate_code_v2(knl))


def test_owed_barriers():
    knl = lp.make_kernel(
            "{[i]: 0<=i<100}",
            [
                "<float32> z[i] = a[i]"
                ],
            [lp.GlobalArg("a", np.float32, shape=(100,))],
            target=lp.PyOpenCLTarget()
            )

    knl = lp.tag_inames(knl, dict(i="l.0"))

    print(knl)
    print(lp.generate_code_v2(knl))


def test_multi_cse():
    knl = lp.make_kernel(
            "{[i]: 0<=i<100}",
            [
                "<float32> z[i] = a[i] + a[i]**2"
                ],
            [lp.GlobalArg("a", np.float32, shape=(100,))],
            target=lp.PyOpenCLTarget())

    knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
    knl = lp.add_prefetch(knl, "a", [])

    print(knl)
    print(lp.generate_code_v2(knl))


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

    n = 20000
    evt, (a,) = knl(queue, n=n, out_host=True)

    assert a.shape == (n,)
    assert (a == 1).all()


# {{{ test race detection

def test_ilp_write_race_detection_global():
    knl = lp.make_kernel(
            "[n] -> {[i,j]: 0<=i,j<n }",
            [
                "a[i] = 5+i+j",
                ],
            [
                lp.GlobalArg("a", np.float32),
                lp.ValueArg("n", np.int32, approximately=1000),
                ],
            assumptions="n>=1",
            target=lp.PyOpenCLTarget(),
            name="loopy_kernel")

    knl = lp.tag_inames(knl, dict(j="ilp"))

    knl = lp.preprocess_kernel(knl)

    with lp.CacheMode(False):
        from loopy.diagnostic import WriteRaceConditionWarning
        from warnings import catch_warnings
        from loopy.schedule import linearize
        with catch_warnings(record=True) as warn_list:
            linearize(knl)

            assert any(isinstance(w.message, WriteRaceConditionWarning)
                    for w in warn_list)


def test_ilp_write_race_avoidance_local():
    knl = lp.make_kernel(
            "{[i,j]: 0<=i<16 and 0<=j<17 }",
            [
                "<> a[i] = 5+i+j",
                ],
            [],
            target=lp.PyOpenCLTarget(),
            name="loopy_kernel")

    knl = lp.tag_inames(knl, dict(i="l.0", j="ilp"))

    knl = lp.preprocess_kernel(knl)
    assert knl["loopy_kernel"].temporary_variables["a"].shape == (16, 17)


def test_ilp_write_race_avoidance_private():
    knl = lp.make_kernel(
            "{[j]: 0<=j<16 }",
            [
                "<> a = 5+j",
                ],
            [],
            target=lp.PyOpenCLTarget(),
            name="loopy_kernel")

    knl = lp.tag_inames(knl, dict(j="ilp"))

    knl = lp.preprocess_kernel(knl)
    assert knl["loopy_kernel"].temporary_variables["a"].shape == (16,)

# }}}


def test_write_parameter(dtype=np.float32):
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
            assumptions="n>=1",
            target=lp.PyOpenCLTarget())

    import pytest
    with pytest.raises(RuntimeError):
        lp.generate_code_v2(knl).device_code()


# {{{ arg guessing

def test_arg_shape_guessing():
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
            assumptions="n>=1",
            target=lp.PyOpenCLTarget())

    print(knl)
    print(lp.generate_code_v2(knl).device_code())


def test_arg_guessing():
    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n }",
            """
                a = 1.5 + sum((i,j), i*j)
                b[i, j] = i*j
                c[i+j, j] = b[j,i]
                """,
            assumptions="n>=1",
            target=lp.PyOpenCLTarget())

    print(knl)
    print(lp.generate_code_v2(knl).device_code())


def test_arg_guessing_with_reduction():
    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<n }",
            """
                a = 1.5 + simul_reduce(sum, (i,j), i*j)
                d = 1.5 + simul_reduce(sum, [i,j], b[i,j])
                b[i, j] = i*j
                c[i+j, j] = b[j,i]
                """,
            assumptions="n>=1",
            target=lp.PyOpenCLTarget())

    print(knl)
    print(lp.generate_code_v2(knl).device_code())


def test_unknown_arg_shape():
    bsize = [256, 0]

    knl = lp.make_kernel(
        "{[i,j]: 0<=i<n and 0<=j<m}",
        """
        for i
            <int32> gid = i/256
            <int32> start = gid*256
            for j
                a[start + j] = a[start + j] + j
            end
        end
        """,
        seq_dependencies=True,
        name="uniform_l",
        target=lp.PyOpenCLTarget(),
        assumptions="m<=%d and m>=1 and n mod %d = 0" % (bsize[0], bsize[0]))

    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32))
    print(lp.generate_code_v2(knl).device_code())

# }}}


def test_nonlinear_index():
    knl = lp.make_kernel(
            "{[i]: 0<=i<n }",
            """
                a[i*i] = 17
                """,
            [
                lp.GlobalArg("a", shape="n"),
                lp.ValueArg("n"),
                ],
            assumptions="n>=1",
            target=lp.PyOpenCLTarget())

    print(knl)
    print(lp.generate_code_v2(knl).device_code())


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

    knl = lp.tag_array_axes(knl, "a,b", "stride:auto,stride:1")

    a_full = cl.clrandom.rand(queue, (n, n), np.float64)
    a_full_h = a_full.get()
    b_full = cl.clrandom.rand(queue, (n, n), np.float64)
    b_full_h = b_full.get()

    a_sub = (slice(3, 10), slice(5, 10))
    a = a_full[a_sub]

    b_sub = (slice(3+3, 10+3), slice(5+4, 10+4))
    b = b_full[b_sub]

    b_full_h[b_sub] = 2*a_full_h[a_sub]

    knl = lp.add_dtypes(knl, {"a": a.dtype})

    print(lp.generate_code_v2(knl))
    knl(queue, a=a, b=b)

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
                ],
            target=lp.PyOpenCLTarget())
    ref_knl = knl

    knl = lp.split_iname(knl, "i", 128, inner_tag="l.0")
    knl = lp.split_iname(knl, "i_outer", 4, outer_tag="g.0", inner_tag="ilp")
    knl = lp.add_prefetch(knl, "a", ["i_inner", "i_outer_inner"],
            default_tag="l.auto")

    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters={"n": 1024})


def test_c_instruction():
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
            assumptions="n>=1", target=lp.PyOpenCLTarget())

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    print(knl)
    print(lp.generate_code_v2(knl).device_code())


def test_dependent_domain_insn_iname_finding():
    prog = lp.make_kernel([
            "{[isrc_box]: 0<=isrc_box<nsrc_boxes}",
            "{[isrc]: isrc_start<=isrc<isrc_end}",
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
                "..."],
            target=lp.PyOpenCLTarget(),
            name="loopy_kernel")

    print(prog)
    assert "isrc_box" in prog["loopy_kernel"].insn_inames("set_strength")

    prog = lp.add_dtypes(prog,
        dict(
            source_boxes=np.int32,
            box_source_starts=np.int32,
            box_source_counts_nonchild=np.int32,
            strengths=np.float64,
            nsources=np.int32))
    print(lp.generate_code_v2(prog).device_code())


def test_inames_deps_from_write_subscript(ctx_factory):
    prog = lp.make_kernel(
            "{[i,j]: 0<=i,j<n}",
            """
                <> src_ibox = source_boxes[i]
                <int32> something = 5
                a[src_ibox] = sum(j, something) {id=myred}
                """,
            [
                lp.GlobalArg("box_source_starts,box_source_counts_nonchild,a",
                    None, shape=None),
                "..."],
            name="loopy_kernel")

    print(prog)
    assert "i" in prog["loopy_kernel"].insn_inames("myred")


def test_modulo_indexing():
    knl = lp.make_kernel(
            "{[i,j]: 0<=i<n and 0<=j<5}",
            """
                b[i] = sum(j, a[(i+j)%n])
                """,
            [
                lp.GlobalArg("a", None, shape="n"),
                "..."
                ], target=lp.PyOpenCLTarget()
            )

    print(knl)
    knl = lp.add_dtypes(knl, {"a": np.float32})
    print(lp.generate_code_v2(knl).device_code())


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

    knl = lp.tag_array_axes(knl, "out", "c,vec")
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


def test_conditional_two_ways(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{ [i,j]: 0<=i,j<n }",
        """
        <> b = i > 3
        <> c = i > 1
        out[i] = a[i] {id=init}
        if b
            out[i] = 2*a[i]  {if=c,dep=init}
        end
        """,
        [
            lp.GlobalArg("a", np.float32, shape=lp.auto),
            lp.GlobalArg("out", np.float32, shape=lp.auto),
            "..."
        ]
    )

    ref_knl = lp.make_kernel(
        "{ [i,j]: 0<=i,j<n }",
        """
        <> b = i > 3
        <> c = i > 1
        out[i] = a[i] {id=init}
        if b and c
            out[i] = 2*a[i]  {dep=init}
        end
        """,
        [
            lp.GlobalArg("a", np.float32, shape=lp.auto),
            lp.GlobalArg("out", np.float32, shape=lp.auto),
            "..."
        ]
    )
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

        knl = lp.set_options(knl, write_code=True)
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
        "{[i]: 0<=i<5}",
        """
        <> temp[i, 0] = 17
        temp[i, 1] = 15
        """)
    knl = lp.tag_inames(knl, dict(i="l.0"))
    print(lp.generate_code_v2(knl).device_code())


def test_make_copy_kernel(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    intermediate_format = "f,f,sep"

    a1 = np.random.randn(1024, 4, 3)

    cknl1 = lp.make_copy_kernel(intermediate_format)

    cknl1 = lp.fix_parameters(cknl1, n2=3)

    cknl1 = lp.set_options(cknl1, write_code=True)
    evt, a2 = cknl1(queue, input=a1)

    cknl2 = lp.make_copy_kernel("c,c,c", intermediate_format)
    cknl2 = lp.fix_parameters(cknl2, n2=3)

    evt, a3 = cknl2(queue, input=a2)

    assert (a1 == a3).all()


def test_make_copy_kernel_with_offsets(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    a1 = np.random.randn(3, 1024, 4)
    a1_dev = cl.array.to_device(queue, a1)

    cknl1 = lp.make_copy_kernel("c,c,c", "sep,c,c")

    cknl1 = lp.fix_parameters(cknl1, n0=3)

    cknl1 = lp.set_options(cknl1, write_code=True)
    evt, (a2_dev,) = cknl1(queue, input=a1_dev)

    assert (a1 == a2_dev.get()).all()


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


def test_auto_test_zero_warmup_rounds(ctx_factory):
    ctx = ctx_factory()

    ref_knl = lp.make_kernel(
        "{[i,j]: 0<=i,j<n}",
        """
        a[i,j] = 25
        """)

    ref_knl = lp.add_and_infer_dtypes(ref_knl, dict(a=np.float32))

    lp.auto_test_vs_ref(
            ref_knl, ctx, ref_knl,
            parameters=dict(n=12),
            warmup_rounds=0)


def test_variable_size_temporary():
    knl = lp.make_kernel(
         """{ [i,j]: 0<=i,j<n }""",
         """out[i] = sum(j, a[i,j])""")

    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})

    knl = lp.add_prefetch(
            knl, "a[:,:]", default_tag=None)

    # Make sure that code generation succeeds even if
    # there are variable-length arrays.
    lp.generate_code_v2(knl).device_code()


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


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_atomic_load(ctx_factory, dtype):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    dtype = np.float64
    n = 10

    knl = lp.make_kernel(
            "{ [j]: 0<=j<n}",
            """
            for j
                temp = 0 {id=init, atomic}
                ... lbarrier {id=lb1, dep=init}
                temp = temp + 1 {id=temp_sum, dep=lb1, atomic}
                ... lbarrier {id=lb2, dep=temp_sum}
                out[j] = temp {id=final, dep=lb2, atomic,nosync=init:temp_sum}
            end
            """,
            [
                lp.GlobalArg("out", dtype, shape=lp.auto, for_atomic=True),
                lp.TemporaryVariable("temp", dtype, for_atomic=True,
                                     address_space=lp.AddressSpace.LOCAL),
                "..."
                ],
            silenced_warnings=["write_race(temp_sum)", "write_race(init)"])
    knl = lp.fix_parameters(knl, n=n)
    knl = lp.tag_inames(knl, {"j": "l.0"})
    _, (out,) = knl(queue)
    assert (out.get() == n).all()


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_atomic_init(dtype):
    vec_width = 4

    knl = lp.make_kernel(
            "{ [i]: 0<=i<100 }",
            """
            out[i%4] = 0 {id=init, atomic=init}
            """,
            [
                lp.GlobalArg("out", dtype, shape=lp.auto, for_atomic=True),
                "..."
                ],
            silenced_warnings=["write_race(init)"])
    knl = lp.split_iname(knl, "i", vec_width, inner_tag="l.0")
    print(knl)
    print(lp.generate_code_v2(knl).device_code())


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

    prog = lp.make_kernel("{[i,j] : 0<=i,j<n}",
            [i1, i2],
            [
                lp.GlobalArg("a", dtype=np.float32, shape=()),
                lp.ValueArg("n", dtype=np.int32),
                lp.TemporaryVariable("phi", dtype=np.float32, shape=("n",)),
                ],
            target=lp.CTarget(),
            name="loopy_kernel"
            )

    prog = lp.preprocess_kernel(prog)

    assert "i" not in prog["loopy_kernel"].insn_inames("insn_0_j_update")
    print(prog["loopy_kernel"].stringify(with_dependencies=True))


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


def test_local_barrier_mem_kind():
    def _test_type(mtype, expected):
        insn = "... lbarrier"
        if mtype:
            insn += "{mem_kind=%s}" % mtype
        knl = lp.make_kernel(
                "{ [i]: 0<=i<n }",
                """
                for i
                    %s
                end
                """ % insn, seq_dependencies=True,
                target=lp.PyOpenCLTarget())

        cgr = lp.generate_code_v2(knl)
        assert "barrier(%s)" % expected in cgr.device_code()

    _test_type("", "CLK_LOCAL_MEM_FENCE")
    _test_type("global", "CLK_GLOBAL_MEM_FENCE")
    _test_type("local", "CLK_LOCAL_MEM_FENCE")


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

    # map schedule onto host or device
    print(knl)

    cgr = lp.generate_code_v2(knl)

    assert len(cgr.device_programs) == 2

    print(cgr.device_code())
    print(cgr.host_code())

    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=5))


def save_and_reload_temporaries_test(queue, prog, out_expect, debug=False):

    from loopy.transform.save import save_and_reload_temporaries
    prog = save_and_reload_temporaries(prog)
    prog = prog.with_kernel(lp.get_one_linearized_kernel(prog["loopy_kernel"],
        prog.callables_table))

    if debug:
        print(prog)
        cgr = lp.generate_code_v2(prog)
        print(cgr.device_code())
        print(cgr.host_code())

    _, (out,) = prog(queue, out_host=True)
    assert (out == out_expect).all(), (out, out_expect)


@pytest.mark.parametrize("hw_loop", [True, False])
def test_save_of_private_scalar(ctx_factory, hw_loop, debug=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    prog = lp.make_kernel(
        "{ [i]: 0<=i<8 }",
        """
        for i
            <>t = i
            ... gbarrier
            out[i] = t
        end
        """, seq_dependencies=True)

    if hw_loop:
        prog = lp.tag_inames(prog, dict(i="g.0"))

    save_and_reload_temporaries_test(queue, prog, np.arange(8), debug)


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

    knl = lp.set_temporary_address_space(knl, "t", "private")
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
    knl = lp.set_temporary_address_space(knl, "t", "private")

    save_and_reload_temporaries_test(
        queue, knl, np.vstack(8 * (np.arange(8),)), debug)


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

    knl = lp.set_temporary_address_space(knl, "t", "private")

    result = np.array([np.vstack(8 * (np.arange(8),)) for i in range(8)])
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

    knl = lp.set_temporary_address_space(knl, "t", "private")
    knl = lp.tag_inames(knl, dict(i="g.0"))

    result = np.array([np.vstack(8 * (np.arange(8),)) for i in range(8)])
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

    knl = lp.set_temporary_address_space(knl, "t_arr", "private")
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

    knl = lp.set_temporary_address_space(knl, "t", "local")
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

    knl = lp.set_temporary_address_space(knl, "t", "local")
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

    knl = lp.set_temporary_address_space(knl, "t_local", "local")
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
    knl = lp.set_temporary_address_space(knl, "a", "local")
    knl = lp.set_temporary_address_space(knl, "b", "local")

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
    knl = lp.set_temporary_address_space(knl, "a", "local")

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
            """
            a[i] = 0
            c[i] = b[i]  {dep_query=writes:a}
            """,
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
    knl = lp.set_temporary_address_space(knl, "c", "global")

    ref_knl = knl

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    cgr = lp.generate_code_v2(knl)

    assert len(cgr.device_programs) == 2

    print(cgr.device_code())
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
        [
            lp.GlobalArg("out", shape="n+2"),
            lp.GlobalArg("f", is_input=False, is_output=True, shape=lp.auto),
            "..."
        ])

    flux_knl = lp.make_kernel(
        "{[j]: 1<=j<=n}",
        "f[j] = u[j]**2/2",
        [
            lp.GlobalArg("f", shape="n+2"),
            lp.GlobalArg("u", shape="n+2"),
            ])

    fused_knl = lp.fuse_kernels(
            [fin_diff_knl["loopy_kernel"], flux_knl["loopy_kernel"]],
            data_flow=[
                ("f", 1, 0)
                ])

    fused_knl = lp.set_options(fused_knl, write_code=True)
    evt, _ = fused_knl(queue, u=u, h=np.float32(1e-1))

    fused_knl = lp.assignment_to_subst(fused_knl, "f")

    fused_knl = lp.set_options(fused_knl, write_code=True)

    # This is the real test here: The automatically generated
    # shape expressions are '2+n' and the ones above are 'n+2'.
    # Is loopy smart enough to understand that these are equal?
    evt, _ = fused_knl(queue, u=u, h=np.float32(1e-1))

    fused0_knl = lp.affine_map_inames(fused_knl, "i", "inew", "inew+1=i")

    gpu_knl = lp.split_iname(
            fused0_knl, "inew", 128, outer_tag="g.0", inner_tag="l.0")

    precomp_knl = lp.precompute(
            gpu_knl, "f_subst", "inew_inner", fetch_bounding_box=True,
            default_tag="l.auto")

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

    from library_for_test import NoRetFunction
    knl = lp.register_callable(knl, "f", NoRetFunction("f"))

    evt, _ = knl(queue)

# }}}


# {{{ call with no return values and options

def test_call_with_options():
    knl = lp.make_kernel(
        "{:}",
        "f() {id=init}"
        )

    from library_for_test import NoRetFunction
    knl = lp.register_callable(knl, "f", NoRetFunction("f"))

    print(lp.generate_code_v2(knl).device_code())

# }}}


def test_unschedulable_kernel_detection():
    # FIXME: does not work
    # Reason for multiple calllable kernels, not sure how this will go.
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
    prog = lp.make_kernel(
            "{[i] : 0<=i<n}",
            "f(sum(i, x[i]))")
    prog = lp.add_and_infer_dtypes(prog, {"x": np.float32})
    prog = lp.preprocess_kernel(prog)
    assert len(prog["loopy_kernel"].instructions) == 3


def test_regression_persistent_hash():
    knl1 = lp.make_kernel(
            "{[i] : 0<=i<n}",
            "cse_exprvar = d[2]*d[2]")

    knl2 = lp.make_kernel(
            "{[i] : 0<=i<n}",
            "cse_exprvar = d[0]*d[0]")
    from loopy.tools import LoopyKeyBuilder
    lkb = LoopyKeyBuilder()
    assert (lkb(knl1["loopy_kernel"].instructions[0]) !=
            lkb(knl2["loopy_kernel"].instructions[0]))
    assert lkb(knl1) != lkb(knl2)


def test_sequential_dependencies(ctx_factory):
    ctx = ctx_factory()

    prog = lp.make_kernel(
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

    print(prog["loopy_kernel"].stringify(with_dependencies=True))

    lp.auto_test_vs_ref(prog, ctx, prog, parameters=dict(n=5))


def test_nop(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "{[i,itrip]: 0<=i<n and 0<=itrip<ntrips}",
            """
            for itrip,i
                <> z[i] = z[i+1] + z[i]  {id=wr_z}
                <> v[i] = 11  {id=wr_v}
                ... nop {dep=wr_z:wr_v,id=yoink}
                z[i] = z[i] - z[i+1] + v[i]  {dep=yoink}
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
                    ... gbarrier {id=yoink,dep=wr_z:wr_v}
                    z[i] = z[i] - z[i+1] + v[i] {id=iupd, dep=yoink}
                end
                ... gbarrier {dep=iupd,id=postloop}
                z[i] = z[i] - z[i+1] + v[i]  {dep=postloop}
            end
            """)

    knl = lp.fix_parameters(knl, ntrips=3)
    knl = lp.add_and_infer_dtypes(knl, {"z": np.float64})

    ref_knl = knl
    ref_knl = lp.set_temporary_address_space(ref_knl, "z", "global")
    ref_knl = lp.set_temporary_address_space(ref_knl, "v", "global")

    knl = lp.split_iname(knl, "i", 256, outer_tag="g.0", inner_tag="l.0")
    print(knl)

    knl = lp.preprocess_kernel(knl)
    assert (
            knl["loopy_kernel"].temporary_variables["z"].address_space ==
            lp.AddressSpace.GLOBAL)
    assert (
            knl["loopy_kernel"].temporary_variables["v"].address_space ==
            lp.AddressSpace.GLOBAL)

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

    knl = lp.set_temporary_address_space(knl, "z", "global")
    knl = lp.split_iname(knl, "i", 256, outer_tag="g.0")
    knl = lp.add_dtypes(knl, {"z": np.float32, "v": np.float32})
    knl = lp.preprocess_kernel(knl)

    from loopy.diagnostic import MissingBarrierError
    with pytest.raises(MissingBarrierError):
        lp.generate_code_v2(knl)


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

    knl = lp.make_kernel("{[k]: 0<=k<n}}",
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

    knl = lp.split_iname(knl, "k", 2, inner_tag="ilp")

    lp.auto_test_vs_ref(ref_knl, ctx, knl)


@pytest.mark.parametrize("unr_tag", ["unr", "unr_hint"])
def test_unr_and_conditionals(ctx_factory, unr_tag):
    ctx = ctx_factory()

    knl = lp.make_kernel("{[k]: 0<=k<n}}",
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

    knl = lp.split_iname(knl, "k", 2, inner_tag=unr_tag)

    lp.auto_test_vs_ref(ref_knl, ctx, knl)


def test_constant_array_args(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel("{[k]: 0<=k<n}}",
         """
         for k
             <> Tcond[k] = T[k] < 0.5
             if Tcond[k]
                 cp[k] = 2 * T[k] + Tcond[k]
             end
         end
         """,
         [lp.ConstantArg("T", shape=(200,), dtype=np.float32),
         "..."])

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
                    address_space=lp.AddressSpace.PRIVATE,
                    read_only=True,
                    order=tmp_order),
                "..."
                ])

    knl = lp.set_options(knl, write_code=True)
    knl = lp.fix_parameters(knl, n=a.shape[0])

    evt, (a2,) = knl(queue, out_host=True)

    assert np.array_equal(a, a2)


def test_const_temp_with_initializer_not_saved():
    prog = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
        ... gbarrier
        out[i] = tmp[i]
        """,
        [
            lp.TemporaryVariable("tmp",
                initializer=np.arange(10),
                shape=lp.auto,
                address_space=lp.AddressSpace.PRIVATE,
                read_only=True),
            "..."
            ],
        seq_dependencies=True)

    prog = lp.preprocess_kernel(prog)
    prog = lp.save_and_reload_temporaries(prog)

    # This ensures no save slot was added.
    assert len(prog["loopy_kernel"].temporary_variables) == 1


def test_header_extract():
    knl = lp.make_kernel("{[k]: 0<=k<n}}",
         """
         for k
             T[k] = k**2
         end
         """,
         [lp.GlobalArg("T", shape=(200,), dtype=np.float32),
         "..."])

    knl = lp.fix_parameters(knl, n=200)

    #test C
    cknl = knl.copy(target=lp.CTarget())
    assert str(lp.generate_header(cknl)[0]) == (
            "void loopy_kernel(float *__restrict__ T);")

    #test CUDA
    cuknl = knl.copy(target=lp.CudaTarget())
    assert str(lp.generate_header(cuknl)[0]) == (
            'extern "C" __global__ void __launch_bounds__(1) '
            "loopy_kernel(float *__restrict__ T);")

    #test OpenCL
    oclknl = knl.copy(target=lp.PyOpenCLTarget())
    assert str(lp.generate_header(oclknl)[0]) == (
            "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) "
            "loopy_kernel(__global float *__restrict__ T);")


def test_scalars_with_base_storage(ctx_factory):
    """ Regression test for !50 """
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    import islpy as isl
    knl = lp.make_kernel(
            [isl.BasicSet("[] -> {[]: }")],  # empty (domain w/unused inames errors)
            "a = 1",
            [
                lp.TemporaryVariable("a", dtype=np.float64,
                                  shape=(), base_storage="base"),
                lp.TemporaryVariable("b", dtype=np.float64,
                                  shape=(), base_storage="base"),
                ])

    knl(queue, out_host=True)


def test_if_else(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{ [i]: 0<=i<50}",
            """
            if i % 3 == 0
                a[i] = 15  {nosync_query=writes:a}
            elif i % 3 == 1
                a[i] = 11  {nosync_query=writes:a}
            else
                a[i] = 3  {nosync_query=writes:a}
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
                        a[i] = 15  {nosync_query=writes:a}
                    elif i % 3 == 1
                        a[i] = 11  {nosync_query=writes:a}
                    else
                        a[i] = 3  {nosync_query=writes:a}
                    end
                else
                    a[i] = 4  {nosync_query=writes:a}
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

    knl = lp.make_kernel(
            "{ [i,j]: 0<=i,j<50}",
            """
            for i
                if i < 25
                    for j
                        if j % 2 == 0
                            a[i, j] = 1  {nosync_query=writes:a}
                        else
                            a[i, j] = 0  {nosync_query=writes:a}
                        end
                    end
                else
                    for j
                        if j % 2 == 0
                            a[i, j] = 0  {nosync_query=writes:a}
                        else
                            a[i, j] = 1  {nosync_query=writes:a}
                        end
                    end
                end
            end
            """
            )

    evt, (out,) = knl(queue, out_host=True)

    out_ref = np.zeros((50, 50))
    out_ref[:25, 0::2] = 1
    out_ref[25:, 1::2] = 1

    assert np.array_equal(out_ref, out)


def test_tight_loop_bounds(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    if (queue.device.platform.vendor == "Intel(R) Corporation"
            and queue.device.driver_version in [
                "2019.8.7.0",
                "2019.8.8.0",
                ]):
        pytest.skip("Intel CL miscompiles this kernel")

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

    knl = lp.set_options(knl, write_code=True)

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
        "((gid(0) == 0 && lid(0) == 0) ? 0 : -2 + 2 * lid(0) + 10 * gid(0)); " \
        "j <= ((-1 + gid(0) == 0 && lid(0) == 0) ? 9 : 2 * lid(0)); ++j)"

    assert for_loop in cgr.device_code()


def test_unscheduled_insn_detection():
    prog = lp.make_kernel(
        "{ [i]: 0 <= i < 10 }",
        """
        out[i] = i {id=insn1}
        """,
        "...")

    prog = lp.preprocess_kernel(prog)
    prog = lp.linearize(prog)
    insn1, = lp.find_instructions(prog, "id:insn1")
    insns = prog["loopy_kernel"].instructions[:]
    insns.append(insn1.copy(id="insn2", depends_on=frozenset({"insn1"})))
    prog = prog.with_kernel(prog["loopy_kernel"].copy(instructions=insns))

    from loopy.diagnostic import UnscheduledInstructionError
    with pytest.raises(UnscheduledInstructionError):
        lp.generate_code(prog)


def test_integer_reduction(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from loopy.types import to_loopy_type

    n = 200
    for vtype in [np.int32, np.int64]:
        var_int = np.random.randint(1000, size=n).astype(vtype)
        var_lp = lp.TemporaryVariable("var", initializer=var_int,
                                   read_only=True,
                                   address_space=lp.AddressSpace.PRIVATE,
                                   dtype=to_loopy_type(vtype),
                                   shape=lp.auto)

        from collections import namedtuple
        ReductionTest = namedtuple("ReductionTest", "kind, check, args")

        reductions = [
            ReductionTest("max", lambda x: x == np.max(var_int), args="var[k]"),
            ReductionTest("min", lambda x: x == np.min(var_int), args="var[k]"),
            ReductionTest("sum", lambda x: x == np.sum(var_int), args="var[k]"),
            ReductionTest("product", lambda x: x == np.prod(var_int), args="var[k]"),
            ReductionTest("argmax",
                lambda x: (
                    x[0] == np.max(var_int) and var_int[out[1]] == np.max(var_int)),
                args="var[k], k"),
            ReductionTest("argmin",
                lambda x: (
                    x[0] == np.min(var_int) and var_int[out[1]] == np.min(var_int)),
                args="var[k], k")
        ]

        for reduction, function, args in reductions:
            kstr = ("out" if "arg" not in reduction
                        else "out[0], out[1]")
            kstr += f" = {reduction}(k, {args})"
            knl = lp.make_kernel("{[k]: 0<=k<n}",
                                kstr,
                                [var_lp, "..."])

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

                    <> post_dist_sq = dist_sq if matches else HUGE
                end
                <> min_dist_sq, <> min_ictr = argmin(ictr, ictr, post_dist_sq)

                tgt_to_qbx_center[itgt] = min_ictr if min_dist_sq < HUGE else -1
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
    print(kernel_str)
    assert "id=insn1, no_sync_with=insn1@any" in kernel_str
    assert "id=insn2, no_sync_with=insn1@any:insn2@any" in kernel_str
    assert "id=insn3, no_sync_with=insn1@local:insn2@global:insn3@any" in kernel_str
    assert "id=insn4, no_sync_with=insn1@local:insn2@local:insn3@local:insn5@local" in kernel_str  # noqa
    assert "id=insn5, no_sync_with=insn1@any" in kernel_str


def barrier_between(knl, id1, id2, ignore_barriers_in_levels=()):
    from loopy.schedule import (RunInstruction, Barrier, EnterLoop, LeaveLoop,
            CallKernel, ReturnFromKernel)
    watch_for_barrier = False
    seen_barrier = False
    loop_level = 0

    for sched_item in knl.linearization:
        if isinstance(sched_item, RunInstruction):
            if sched_item.insn_id == id1:
                watch_for_barrier = True
            elif sched_item.insn_id == id2:
                return watch_for_barrier and seen_barrier
        elif isinstance(sched_item, Barrier):
            if watch_for_barrier and loop_level not in ignore_barriers_in_levels:
                seen_barrier = True
        elif isinstance(sched_item, EnterLoop):
            loop_level += 1
        elif isinstance(sched_item, LeaveLoop):
            loop_level -= 1
        elif isinstance(sched_item, (CallKernel, ReturnFromKernel)):
            pass
        else:
            raise RuntimeError("schedule item type '%s' not understood"
                    % type(sched_item).__name__)

    raise RuntimeError("id2 was not seen")


def test_barrier_insertion_near_top_of_loop():
    prog = lp.make_kernel(
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

    prog = lp.tag_inames(prog, dict(i="l.0"))
    prog = lp.set_temporary_address_space(prog, "a", "local")
    prog = lp.set_temporary_address_space(prog, "b", "local")
    prog = lp.preprocess_kernel(prog)
    knl = lp.get_one_linearized_kernel(prog["loopy_kernel"], prog.callables_table)

    print(knl)

    assert barrier_between(knl, "ainit", "tcomp")


def test_barrier_insertion_near_bottom_of_loop():
    prog = lp.make_kernel(
        ["{[i]: 0 <= i < 10 }",
         "[jmax] -> {[j]: 0 <= j < jmax}"],
        """
        for i
         <>a[i] = i  {id=ainit}
         for j
          <>b[i,j] = a[(i+1) % 10] + t   {id=bcomp1}
          b[i,j] = b[(i+2) % 10,j] + 1  {id=bcomp2}
         end
         a[10-i] = i + 1 {id=aupdate}
        end
        """,
        seq_dependencies=True)
    prog = lp.tag_inames(prog, dict(i="l.0"))
    prog = lp.set_temporary_address_space(prog, "a", "local")
    prog = lp.set_temporary_address_space(prog, "b", "local")
    prog = lp.preprocess_kernel(prog)
    knl = lp.get_one_linearized_kernel(prog["loopy_kernel"], prog.callables_table)

    print(knl)

    assert barrier_between(knl, "bcomp1", "bcomp2")
    assert barrier_between(knl, "ainit", "aupdate", ignore_barriers_in_levels=[1])


def test_barrier_in_overridden_get_grid_size_expanded_kernel():
    # make simple barrier'd kernel
    prog = lp.make_kernel("{[i]: 0 <= i < 10}",
                   """
              for i
                    a[i] = i {id=a}
                    ... lbarrier {id=barrier}
                    b[i + 1] = a[i] {nosync=a}
              end
                   """,
                   [lp.TemporaryVariable("a", np.float32, shape=(10,), order="C",
                                         address_space=lp.AddressSpace.LOCAL),
                    lp.GlobalArg("b", np.float32, shape=(11,), order="C")],
               seq_dependencies=True)

    # split into kernel w/ vesize larger than iname domain
    vecsize = 16
    prog = lp.split_iname(prog, "i", vecsize, inner_tag="l.0")

    from testlib import GridOverride

    # artifically expand via overridden_get_grid_sizes_for_insn_ids
    knl = prog["loopy_kernel"]
    knl = knl.copy(overridden_get_grid_sizes_for_insn_ids=GridOverride(
        knl.copy(), vecsize))
    prog = prog.with_kernel(knl)
    # make sure we can generate the code
    lp.generate_code_v2(prog)


def test_multi_argument_reduction_type_inference():
    from loopy.type_inference import TypeReader
    from loopy.library.reduction import SegmentedSumReductionOperation
    from loopy.types import to_loopy_type
    op = SegmentedSumReductionOperation()

    prog = lp.make_kernel("{[i,j]: 0<=i<10 and 0<=j<i}", "")

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

    t_inf_mapper = TypeReader(prog["loopy_kernel"],
            prog.callables_table)

    assert (
            t_inf_mapper(expr, return_tuple=True, return_dtype_set=True)
            == [(int32, int32)])


def test_multi_argument_reduction_parsing():
    from loopy.symbolic import parse, Reduction

    assert isinstance(
            parse("reduce(argmax, i, reduce(argmax, j, i, j))").expr,
            Reduction)


def test_global_barrier_order_finding():
    prog = lp.make_kernel(
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

    assert (lp.get_global_barrier_order(prog["loopy_kernel"]) == ("top", "yoink",
        "postloop"))

    for insn, barrier in (
            ("nop", None),
            ("top", None),
            ("wr_z", "top"),
            ("wr_v", "top"),
            ("yoink", "top"),
            ("postloop", "yoink"),
            ("zzzv", "postloop")):
        assert lp.find_most_recent_global_barrier(prog["loopy_kernel"],
                insn) == barrier


def test_global_barrier_error_if_unordered():
    # FIXME: Should be illegal to declare this
    prog = lp.make_kernel("{[i]: 0 <= i < 10}",
            """
            ... gbarrier
            ... gbarrier
            """)

    from loopy.diagnostic import LoopyError
    with pytest.raises(LoopyError):
        lp.get_global_barrier_order(prog["loopy_kernel"])


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
    bbhit = cl.tools.get_or_register_dtype("bbhit", bbhit)

    preamble = bbhit_c_decl

    knl = lp.make_kernel(
        "{ [i]: 0<=i<N }",
        """
        for i
            result[i].hit = i % 2  {nosync_query=writes:result}
            result[i].tmin = i  {nosync_query=writes:result}
            result[i].tmax = i+10  {nosync_query=writes:result}
            result[i].bi = i  {nosync_query=writes:result}
        end
        """,
        [
            lp.GlobalArg("result", shape=("N",), dtype=bbhit),
            "..."],
        preambles=[("000", preamble)])

    knl = lp.set_options(knl, write_code=True)
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


def test_fixed_parameters(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "[n] -> {[i]: 0 <= i < n}",
            """
            <>tmp[i] = i  {id=init}
            tmp[0] = 0  {dep=init}
            """,
            fixed_parameters=dict(n=1))

    knl(queue)


def test_parameter_inference():
    knl = lp.make_kernel("{[i]: 0 <= i < n and i mod 2 = 0}", "")
    assert knl["loopy_kernel"].all_params() == {"n"}


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


def test_wildcard_dep_matching():
    prog = lp.make_kernel(
            "{[i]: 0 <= i < 10}",
            """
            <>a = 0 {id=insn1}
            <>b = 0 {id=insn2,dep=insn?}
            <>c = 0 {id=insn3,dep=insn*}
            <>d = 0 {id=insn4,dep=insn[12]}
            <>e = 0 {id=insn5,dep=insn[!1]}
            """,
            "...")

    all_insns = {"insn%d" % i for i in range(1, 6)}

    assert prog["loopy_kernel"].id_to_insn["insn1"].depends_on == set()
    assert (prog["loopy_kernel"].id_to_insn["insn2"].depends_on == all_insns -
            {"insn2"})
    assert (prog["loopy_kernel"].id_to_insn["insn3"].depends_on == all_insns -
            {"insn3"})
    assert (prog["loopy_kernel"].id_to_insn["insn4"].depends_on == {"insn1",
        "insn2"})
    assert (prog["loopy_kernel"].id_to_insn["insn5"].depends_on == all_insns -
            {"insn1", "insn5"})


def test_arg_inference_for_predicates():
    prog = lp.make_kernel("{[i]: 0 <= i < 10}",
            """
            if incr[i]
              a = a + 1
            end
            """, name="loopy_kernel")

    knl = prog["loopy_kernel"]

    assert "incr" in knl.arg_dict
    assert knl.arg_dict["incr"].shape == (10,)


def test_relaxed_stride_checks(ctx_factory):
    # Check that loopy is compatible with numpy's relaxed stride rules.
    ctx = ctx_factory()

    knl = lp.make_kernel("{[i,j]: 0 <= i <= n and 0 <= j <= m}",
             """
             a[i] = sum(j, A[i,j] * b[j])
             """)

    with cl.CommandQueue(ctx) as queue:
        mat = np.zeros((1, 10), order="F")
        b = np.zeros(10)

        evt, (a,) = knl(queue, A=mat, b=b)

        assert a == 0


def test_add_prefetch_works_in_lhs_index():
    prog = lp.make_kernel(
            "{ [n,k,l,k1,l1,k2,l2]: "
            "start<=n<end and 0<=k,k1,k2<3 and 0<=l,l1,l2<2 }",
            """
            for n
                <> a1_tmp[k,l] = a1[a1_map[n, k],l]
                a1_tmp[k1,l1] = a1_tmp[k1,l1] + 1
                a1_out[a1_map[n,k2], l2] = a1_tmp[k2,l2]
            end
            """,
            [
                lp.GlobalArg("a1,a1_out", None, "ndofs,2"),
                lp.GlobalArg("a1_map", None, "nelements,3"),
                "..."
            ])

    prog = lp.add_prefetch(prog, "a1_map", "k", default_tag="l.auto")

    from loopy.symbolic import get_dependencies
    for insn in prog["loopy_kernel"].instructions:
        assert "a1_map" not in get_dependencies(insn.assignees)


def test_check_for_variable_access_ordering():
    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            """
            a[i] = 12
            a[i+1] = 13
            """)

    from loopy.diagnostic import VariableAccessNotOrdered
    with pytest.raises(VariableAccessNotOrdered):
        lp.generate_code_v2(knl)


def test_check_for_variable_access_ordering_with_aliasing():
    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            """
            a[i] = 12
            b[i+1] = 13
            """,
            [
                lp.TemporaryVariable("a", shape="n+1", base_storage="tmp"),
                lp.TemporaryVariable("b", shape="n+1", base_storage="tmp"),
                ])

    from loopy.diagnostic import VariableAccessNotOrdered
    with pytest.raises(VariableAccessNotOrdered):
        lp.generate_code_v2(knl)


@pytest.mark.parametrize(("second_index", "expect_barrier"),
        [
            ("2*i", False),
            ("2*i+1", False),
            ("2*i+2", True),
            ])
def test_no_barriers_for_nonoverlapping_access(second_index, expect_barrier):
    prog = lp.make_kernel(
            "{[i]: 0<=i<128}",
            """
            a[2*i] = 12  {id=first}
            a[%s] = 13  {id=second,dep=first}
            """ % second_index,
            [
                lp.TemporaryVariable("a", dtype=None, shape=(256,),
                    address_space=lp.AddressSpace.LOCAL),
                ])

    prog = lp.tag_inames(prog, "i:l.0")
    prog = lp.preprocess_kernel(prog)

    knl = lp.get_one_linearized_kernel(prog["loopy_kernel"],
            prog.callables_table)

    assert barrier_between(knl, "first", "second") == expect_barrier


def test_half_complex_conditional(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{[i]: 0 <= i < 10}",
            """
           tmp[i] = 0 if i < 5 else 0j
           """)

    knl(queue)


def test_dep_cycle_printing_and_error():
    # https://gitlab.tiker.net/inducer/loopy/issues/140
    # This kernel has two dep cycles.

    knl = lp.make_kernel("{[i,j,k]: 0 <= i,j,k < 12}",
    """
        for j
            for i
                <> nu = i - 4
                if nu > 0
                    <> P_val = a[i, j] {id=pset0}
                else
                    P_val = 0.1 * a[i, j] {id=pset1}
                end
                <> B_sum = 0
                for k
                    B_sum = B_sum + k * P_val {id=bset, dep=pset*}
                end
                # here, we are testing that Kc is properly promoted to a vector dtype
                <> Kc = P_val * B_sum {id=kset, dep=bset}
                a[i, j] = Kc {dep=kset}
            end
        end
    """,
    [lp.GlobalArg("a", shape=(12, 12), dtype=np.int32)])

    knl = lp.split_iname(knl, "j", 4, inner_tag="vec")
    knl = lp.split_array_axis(knl, "a", 1, 4)
    knl = lp.tag_array_axes(knl, "a", "N1,N0,vec")
    knl = lp.preprocess_kernel(knl)

    from loopy.diagnostic import DependencyCycleFound
    with pytest.raises(DependencyCycleFound):
        print(lp.generate_code_v2(knl).device_code())


def test_backwards_dep_printing_and_error():
    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            """
            c[i] = a[i] + b[i]                       {id=insn1}
            c[i] = 2*c[i]                            {id=insn2, dep=insn1}
            c[i] = 7*c[i] + a[i]*a[i] + b[i]*b[i]    {id=insn3, dep=insn2}
            b[i] = b[i] + c[i]                                 {id=insn4, dep=insn3}
            d[i] = 7*a[i ]                                     {id=insn5, dep=insn4}
            a[i] = a[i] + d[i]                                 {id=insn6, dep=insn5}
            """, [
                lp.GlobalArg("a, b", dtype=np.float64),
                "..."
            ])

    # Used to crash with KeyError
    print(knl)


def test_dump_binary(ctx_factory):
    pytest.skip("Not investing time in passing test depends on feature which was "
            "deprecated in 2016")
    ctx = ctx_factory()

    device = ctx.devices[0]

    if (device.platform.vendor == "Intel(R) Corporation"
            and device.driver_version in [
                "2019.8.7.0",
                "2019.8.8.0",
                ]):
        pytest.skip("Intel CL doesn't implement Kernel.program")

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            """
            out[i] = i
            """)

    knl = lp.fix_parameters(knl, n=128)
    ref_knl = knl

    lp.auto_test_vs_ref(
            ref_knl, ctx, knl, parameters=dict(n=5),
            dump_binary=True)


def test_temp_var_type_deprecated_usage():
    import warnings
    warnings.simplefilter("always")

    with pytest.warns(DeprecationWarning):
        lp.Assignment("x", 1, temp_var_type=lp.auto)

    with pytest.warns(DeprecationWarning):
        lp.Assignment("x", 1, temp_var_type=None)

    with pytest.warns(DeprecationWarning):
        lp.Assignment("x", 1, temp_var_type=np.dtype(np.int32))

    from loopy.symbolic import parse

    with pytest.warns(DeprecationWarning):
        lp.CallInstruction("(x,)", parse("f(1)"), temp_var_types=(lp.auto,))

    with pytest.warns(DeprecationWarning):
        lp.CallInstruction("(x,)", parse("f(1)"), temp_var_types=(None,))

    with pytest.warns(DeprecationWarning):
        lp.CallInstruction("(x,)", parse("f(1)"),
                temp_var_types=(np.dtype(np.int32),))


def test_shape_mismatch_check(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    t_unit = lp.make_kernel(
            "{[i,j]: 0 <= i < n and 0 <= j < m}",
            "c[i] = sum(j, a[i,j]*b[j])",
            default_order="F")

    a = np.random.rand(10, 10).astype(np.float32)
    b = np.random.rand(10).astype(np.float32)

    if t_unit["loopy_kernel"].options.skip_arg_checks:
        pytest.skip("args checks disabled, cannot check")

    with pytest.raises(ValueError, match="strides mismatch"):
        t_unit(queue, a=a, b=b)


def test_array_arg_extra_kwargs_persis_hash():
    from loopy.tools import LoopyKeyBuilder

    a = lp.ArrayArg("a", shape=(10, ), dtype=np.float64,
            address_space=lp.AddressSpace.LOCAL)
    not_a = lp.ArrayArg("a", shape=(10, ), dtype=np.float64,
            address_space=lp.AddressSpace.PRIVATE)

    key_builder = LoopyKeyBuilder()
    assert key_builder(a) != key_builder(not_a)


def test_type_inference_walks_fn_in_comparison():
    # Reported by Lawrence Mitchell
    # See: https://gitlab.tiker.net/inducer/loopy/issues/180

    knl = lp.make_kernel(
        [
            "{ [p] : 0 <= p <= 2 }",
            "{ [i] : 0 <= i <= 2 }",
        ],
        """
        t2 = 0.0 {id=insn}
        t1 = 0.0 {id=insn_0, dep=insn}
        t1 = t1 + t0[p, i]*w_0[1 + i*2] {id=insn_1, dep=insn_0}
        t2 = t2 + t0[p, i]*w_0[i*2] {id=insn_2, dep=insn_1}
        A[p] = A[p]+(0.2 if abs(-1.2+t2) <= 0.1 and abs(-0.15+t1) <= 0.05 else 0.0
                                            ) {dep=insn_2}
        """, [
            lp.GlobalArg(
                name="A", dtype=np.float64,
                shape=(3)),
            lp.GlobalArg(
                name="w_0", dtype=np.float64,
                shape=(6),),
            lp.TemporaryVariable(
                name="t0", dtype=np.float64,
                shape=(3, 3),
                read_only=True,
                address_space=lp.AddressSpace.LOCAL,
                initializer=np.array([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]]),),
            lp.TemporaryVariable(
                name="t1", dtype=np.float64,
                shape=()),
            lp.TemporaryVariable(
                name="t2", dtype=np.float64,
                shape=()),
            ],
        target=lp.CTarget())

    print(lp.generate_code_v2(knl).device_code())


def test_non_integral_array_idx_raises():
    knl = lp.make_kernel(
            "{[i, j]: 0<=i<=4 and 0<=j<16}",
            """
            out[j] = 0 {id=init}
            out[i] = a[1.94**i-1] {dep=init}
            """, [lp.GlobalArg("a", np.float64), "..."])

    from loopy.diagnostic import LoopyError
    with pytest.raises(LoopyError):
        print(lp.generate_code_v2(knl).device_code())


@pytest.mark.parametrize("tag", ["for", "l.0", "g.0", "fixed"])
def test_empty_domain(ctx_factory, tag):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    t_unit = lp.make_kernel(
            "{[i,j]: 0 <= i < n}",
            """
            for i
                c = 1
            end
            """)

    if tag == "fixed":
        t_unit = lp.fix_parameters(t_unit, n=0)
        kwargs = {}
    else:
        t_unit = lp.tag_inames(t_unit, {"i": tag})
        kwargs = {"n": 0}

    t_unit = lp.set_options(t_unit, write_code=True)
    c = cl.array.zeros(queue, (), dtype=np.int32)
    t_unit(queue, c=c, **kwargs)

    assert (c.get() == 0).all()


def test_access_check_with_conditionals():
    legal_knl = lp.make_kernel(
            "{[i]: 0<=i<20}",
            """
            z[i] = x[i] if i < 10 else y[i-10]
            z[i] = x[i] if 0 else 2.0f
            z[i] = in[i-1] if i else 3.14f
            """,
            [lp.GlobalArg("x,y", shape=(10,), dtype=float),
             lp.GlobalArg("in", shape=(19,), dtype=float),
             ...], seq_dependencies=True)
    lp.generate_code_v2(legal_knl)

    illegal_knl = lp.make_kernel(
            "{[i]: 0<=i<20}",
            """
            z[i] = x[i] if i < 10 else y[i]
            """,
            [lp.GlobalArg("x,y", shape=(10,), dtype=float),
             ...])

    from loopy.diagnostic import LoopyError
    with pytest.raises(LoopyError):
        lp.generate_code_v2(illegal_knl)

    # current limitation: cannot handle non-affine conditions
    legal_but_nonaffine_condition_knl = lp.make_kernel(
            "{[i]: 0<=i<20}",
            """
            z[i] = x[i] if i*i < 100 else y[i-10]
            """,
            [lp.GlobalArg("x,y", shape=(10,), dtype=float),
             ...])

    from loopy.diagnostic import LoopyError
    with pytest.raises(LoopyError):
        lp.generate_code_v2(legal_but_nonaffine_condition_knl)


def test_access_check_with_insn_predicates():
    knl = lp.make_kernel(
            "{[i]: 0<i<10}",
            """
            if i < 4
              y[i] = 2*x[i]
            end
            """, [lp.GlobalArg("x", dtype=float, shape=(4,)), ...])

    print(lp.generate_code_v2(knl).device_code())


def test_conditional_access_range_with_parameters(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            ["{[i]: 0 <= i < 10}",
             "{[j]: 0 <= j < problem_size+2}"],
            """
            if i < 8 and j < problem_size
                tmp[j, i] = tmp[j, i] + 1
            end
           """,
            [lp.GlobalArg("tmp", shape=("problem_size", 8,), dtype=np.int64),
             lp.ValueArg("problem_size", dtype=np.int64)])

    assert np.array_equal(knl(queue, tmp=np.arange(80).reshape((10, 8)),
                              problem_size=10)[1][0], np.arange(1, 81).reshape(
                                (10, 8)))

    # test a conditional that's only _half_ data-dependent to ensure the other
    # half works
    knl = lp.make_kernel(
            ["{[i]: 0 <= i < 10}",
             "{[j]: 0 <= j < problem_size}"],
            """
            if i < 8 and (j + offset) < problem_size
                tmp[j, i] = tmp[j, i] + 1
            end
           """,
            [lp.GlobalArg("tmp", shape=("problem_size", 8,), dtype=np.int64),
             lp.ValueArg("problem_size", dtype=np.int64),
             lp.ValueArg("offset", dtype=np.int64)])

    assert np.array_equal(knl(queue, tmp=np.arange(80).reshape((10, 8)),
                              problem_size=10,
                              offset=0)[1][0], np.arange(1, 81).reshape(
                                (10, 8)))


def test_split_iname_within(ctx_factory):
    # https://github.com/inducer/loopy/issues/163
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{ [i, j]: 0<=i<n and 0<=j<n }",
        """
        x[i, j] = 3 {id=a}
        y[i, j] = 2 * y[i, j] {id=b}
        """,
        options=dict(write_code=True))

    ref_knl = knl

    knl = lp.split_iname(knl, "j", 4,
                         outer_tag="g.0", inner_tag="l.0",
                         within="id:a")
    knl = lp.split_iname(knl, "i", 4,
                         outer_tag="g.0", inner_tag="l.0",
                         within="id:b")

    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=5))


@pytest.mark.parametrize("base_type,exp_type", [
    (np.int32, np.uint32), (np.int64, np.uint64),

    #  It looks like numpy thinks int32**float32 should be float64, which seems
    #  weird.
    # (np.int32, np.float32),

    (np.int32, np.float64),
    (np.float64, np.int32), (np.int64, np.int32),
    (np.float32, np.float64), (np.float64, np.float32)])
def test_pow(ctx_factory, base_type, exp_type):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    def _make_random_np_array(shape, dtype):
        from numpy.random import default_rng
        rng = default_rng(0)
        if isinstance(shape, int):
            shape = (shape,)

        dtype = np.dtype(dtype)
        if dtype.kind in ["u", "i"]:
            low = 0  # numpy might trigger error for -ve int exponents
            high = 6  # choosing numbers to avoid overflow (undefined behavior)
            return rng.integers(low=low, high=high, size=shape, dtype=dtype)
        elif dtype.kind == "f":
            return rng.random(*shape).astype(dtype)
        else:
            raise NotImplementedError()

    base = _make_random_np_array(10, base_type)
    power = _make_random_np_array(10, exp_type)
    expected_result = base ** power

    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            """
            res[i] = base[i] ** power[i]
            """)

    knl = lp.add_dtypes(knl, {"base": base_type, "power": exp_type})

    evt, (result,) = knl(queue, base=base, power=power)

    assert result.dtype == expected_result.dtype

    np.testing.assert_allclose(expected_result, result)


def test_deps_from_conditionals():
    # https://github.com/inducer/loopy/issues/231
    knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            """
            <> icontaining_tgt_box = 1 {id=flagset}
            if icontaining_tgt_box == 1
                result = result + simul_reduce(sum, i, i*i)
                result = result + simul_reduce(sum, i, 2*i*i)
            end
            """, name="lpy_knl")
    ppknl = lp.preprocess_kernel(knl)

    # accumulator initializers must be dependency-less
    assert all(not insn.depends_on
            for insn in ppknl["lpy_knl"].instructions
            if "init" in insn.id)
    # accumulator initializers must not have inherited the predicates
    assert all(not insn.predicates
            for insn in ppknl["lpy_knl"].instructions
            if "init" in insn.id)

    # Ensure valid linearization exists: No valid linearization unless the
    # accumulator initializers can move out of the loop.
    print(lp.generate_code_v2(ppknl).device_code())


def test_scalar_temporary(ctx_factory):
    from numpy.random import default_rng
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    rng = default_rng()
    x_in = rng.random()
    knl = lp.make_kernel(
        "{:}",
        """
        tmp = 2*x
        y = 2*tmp
        """,
        [lp.ValueArg("x", dtype=float),
        lp.TemporaryVariable("tmp", address_space=lp.AddressSpace.GLOBAL,
                             shape=lp.auto),
        ...])
    evt, (out, ) = knl(queue, x=x_in)
    np.testing.assert_allclose(4*x_in, out.get())


def test_cached_written_variables_doesnt_carry_over_invalidly():
    knl = lp.make_kernel(
            "{:}",
            """
            a[i] = 2*i {id=write_a}
            b[i] = 2*i {id=write_b}
            """)
    from pickle import dumps, loads
    knl2 = loads(dumps(knl))

    knl2 = lp.remove_instructions(knl2, {"write_b"})
    assert "b" not in knl2["loopy_kernel"].get_written_variables()


def test_kernel_tagging():
    from pytools.tag import Tag

    class LessInformativeTag(Tag):
        pass

    class SuperInformativeTag(Tag):
        pass

    class SuperDuperInformativeTag(SuperInformativeTag):
        pass

    t1 = SuperInformativeTag()
    t2 = LessInformativeTag()
    knl1 = lp.make_kernel(
        "{:}",
        "y = 0",
        tags=frozenset((t1, t2)))
    knl1 = knl1.default_entrypoint

    assert knl1.tags == frozenset((t1, t2))

    t3 = SuperDuperInformativeTag()
    knl2 = knl1.tagged(tags=frozenset((t3,)))
    assert knl2.tags == frozenset((t1, t2, t3))

    knl3 = knl2.without_tags(tags=frozenset((t2,)))

    assert knl3.tags == frozenset((t1, t3))
    assert knl3.copy().tags == knl3.tags


def test_split_iname_with_multiple_dim_params(ctx_factory):
    ctx = ctx_factory()

    ref_knl = lp.make_kernel(
        ["{[i, j]: 0<=i,j<16}",
        "[i,j] -> {[k]: 0<=k<=4}"],
        """
        foo[i, j, k] = i+j+k
        """)
    knl = lp.split_iname(ref_knl, "i", 4)

    lp.auto_test_vs_ref(ref_knl, ctx, knl)


@pytest.mark.parametrize("opt_name",
        ["trace_assignments", "trace_assignment_values"])
def test_trace_assignments(ctx_factory, opt_name):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{[i,j]: 0<=i,j<2}",
        """
        foo[i,j] = i+j
        """)
    knl = lp.tag_inames(knl, {"i": "g.0", "j": "l.0"})
    knl = lp.set_options(knl, **{opt_name: True})

    knl(queue)


def test_tunit_to_python():
    knl = lp.make_kernel(
            "{[i, j]: 0<=i,j<n}",
            """
            y[i] = sin(x[i])     {id=insn0}
            z[i] = sin(y[i])     {id=insn1, dep=insn0}
            w = sum(j, 2 * z[j]) {id=insn2, dep=insn1:insn0}
            """,
            name="my_kernel")

    knl = lp.split_iname(knl, "i", 4, inner_tag="l.0", outer_tag="g.0")
    lp.t_unit_to_python(knl)  # contains check to assert roundtrip equivalence

    mysin = lp.make_function(
        "{[i, j]: 0<=i<n and 0<=j<m}",
        """
        y[i, j] = sin(x[i, j])
        """, name="my_kernel")
    t_unit = lp.make_kernel(
        "{[i, j]: 0<=i, j<10}",
        """
        [i, j]: y[i,j] = mysin(10, 10, [i, j]: x[i, j])
        """)

    t_unit = lp.merge([t_unit, mysin])
    lp.t_unit_to_python(t_unit)  # contains check to assert roundtrip equivalence

    knl_explicit_iname = lp.make_kernel(
        ["{[i]: 0<=i<10}", "{[j]: 0<=j<10}"],
        ["""
        for i
            a[j] = 0       {id=a}
            b[i, j] = a[j] {dep=a}
        end"""],
        kernel_data=[
            lp.TemporaryVariable("a", dtype=np.int32),
            lp.GlobalArg("b"),
        ])
    # contains check to assert roundtrip equivalence
    lp.t_unit_to_python(knl_explicit_iname)


def test_global_tv_with_base_storage_across_gbarrier(ctx_factory):
    # see https://github.com/inducer/loopy/pull/466 for context
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    t_unit = lp.make_kernel(
        "{[i,j]: 0<=i,j<10}",
        """
        tmp[i] = i
        ... gbarrier
        out[j] = tmp[9-j]
        """,
        [lp.TemporaryVariable("tmp",
                              address_space=lp.AddressSpace.GLOBAL,
                              base_storage="base"),
         ...],
        seq_dependencies=True)

    t_unit = lp.tag_inames(t_unit, {"i": "g.0", "j": "g.0"})

    _, (out,) = t_unit(cq)
    np.testing.assert_allclose(out.get(), np.arange(9, -1, -1))


def test_get_return_from_kernel_mapping():
    from loopy.schedule.tools import get_return_from_kernel_mapping

    t_unit = lp.make_kernel(
        "{[i,j]: 0<=i,j<10}",
        """
        <> tmp[i] = i
        ... gbarrier
        out[j] = tmp[9-j]
        """,
        seq_dependencies=True)
    t_unit = lp.linearize(lp.preprocess_kernel(t_unit))
    ret_from_knl_idx = get_return_from_kernel_mapping(t_unit.default_entrypoint)
    assert ret_from_knl_idx[0] == 4
    assert ret_from_knl_idx[1] == 4
    assert ret_from_knl_idx[2] == 4
    assert ret_from_knl_idx[3] == 4

    assert ret_from_knl_idx[6] == 10
    assert ret_from_knl_idx[7] == 10
    assert ret_from_knl_idx[8] == 10
    assert ret_from_knl_idx[9] == 10


def test_zero_stride_array(ctx_factory):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        ["{[i]: 0<=i<10}",
         "{[j]: 1=0}"],
        """
        y[i, j] = 1
        """, [lp.GlobalArg("y", shape=(10, 0))])

    evt, (out,) = knl(cq)
    assert out.shape == (10, 0)


def test_sep_array_ordering(ctx_factory):
    # https://github.com/inducer/loopy/pull/667
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    # NOTE: this works with n = 10, but fails with n >= 11
    n = 11
    knl = lp.make_kernel(
        "{[i, k]: 0<=k<noutputs and 0<=i<m}",
        """
        x[k, i] = k
        """,
        [lp.GlobalArg("x", shape=("noutputs", "m"), dim_tags="sep,C")] + [...],
        fixed_parameters=dict(noutputs=n),
        )
    knl = lp.tag_inames(knl, "k:unr")

    x = [cl.array.empty(cq, (0,), dtype=np.float64) for i in range(n)]
    evt, out = knl(cq, x=x)

    for i in range(n):
        assert out[i] is x[i], f"failed on input x{i}: {id(out[i])} {id(x[i])}"


def test_predicated_redn(ctx_factory):
    # See https://github.com/inducer/loopy/issues/427
    ctx = ctx_factory()

    knl = lp.make_kernel(
        ["{[i]: 0<= i < 5}",
         "{[j]: 0<= j < 10}",
         "{[k]: 0<=k<10}"],
        """
        <> tmp[k] = k ** 2
        y[j] = 0 if j < 5 else sum(i, tmp[i+j-5])
        """, seq_dependencies=True)

    # if predicates are added correctly, access checker does not raise
    lp.auto_test_vs_ref(knl, ctx, knl)


def test_redn_in_predicate(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        ["{[i]: 0<= i < 5}",
         "{[j]: 0<= j < 10}",
         "{[k]: 0<=k<10}"],
        """
        y[j] = sum(i, i**3) if (sum(k, k**2) < 2) else (10 - j)
        """,
        seq_dependencies=True)

    lp.auto_test_vs_ref(knl, ctx, knl)


def test_obj_tagged_is_persistent_hashable():
    from loopy.tools import LoopyKeyBuilder
    from pytools.tag import tag_dataclass, Tag
    from loopy.match import ObjTagged

    lkb = LoopyKeyBuilder()

    @tag_dataclass
    class MyTag(Tag):
        pass

    assert lkb(ObjTagged(MyTag())) == lkb(ObjTagged(MyTag()))


@pytest.mark.xfail
def test_vec_loops_surrounded_by_preds(ctx_factory):
    # See https://github.com/inducer/loopy/issues/615
    ctx = ctx_factory()
    knl = lp.make_kernel(
        "{[i, j]: 0<=i<100 and 0<=j<4}",
        """
        for i
            for j
                if j
                    <> tmp[j] = 1
                end
                out[i, j] = 2*tmp[j]
            end
        end
        """, seq_dependencies=True)

    ref_knl = knl

    knl = lp.tag_array_axes(knl, "tmp", "vec")
    knl = lp.tag_inames(knl, "j:vec")
    lp.auto_test_vs_ref(ref_knl, ctx, knl)


def test_vec_inames_can_reenter(ctx_factory):
    # See https://github.com/inducer/loopy/issues/644
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{[i, j]: 0<=i,j<4}",
        """
        for i
            <> tmp0[i] = 1
            for j
                <> tmp1[i] = 2
            end
            <> tmp2[i] = 3
            out[i] = tmp0[i] + tmp1[i] + tmp2[i]
        end
        """,
        seq_dependencies=True)

    knl = lp.tag_inames(knl, "i:vec")
    knl = lp.tag_array_axes(knl, "tmp0,tmp1,tmp2", "vec")

    knl = lp.duplicate_inames(knl, "i",
                              within="writes:tmp1",
                              tags={"i": "vec"})

    _, (out,) = knl(cq)
    np.testing.assert_allclose(out.get(), 6*np.ones(4))


def test_split_and_join_inames(ctx_factory):
    # See https://github.com/inducer/loopy/issues/652
    ctx = ctx_factory()

    tunit = lp.make_kernel(
        "{[i]: 0<=i<16}",
        """
        y[i] = i
        """)
    ref_tunit = tunit

    tunit = lp.split_iname(tunit, "i", 4)
    tunit = lp.join_inames(tunit, ["i_inner", "i_outer"])

    lp.auto_test_vs_ref(ref_tunit, ctx, tunit)


def test_different_index_dtypes():
    from loopy.diagnostic import LoopyError

    doublify = lp.make_function(
        "{[i]: 0<=i<10}",
        """
        x[i] = x[i] * 2
        """,
        name="doublify",
        index_dtype=np.int64
    )

    knl = lp.make_kernel(
        "{[I]: 0<=I<10}",
        """
        [I]: X[I] = doublify([I]: X[I])
        """,
        index_dtype=np.int32
    )

    knl = lp.merge([knl, doublify])

    with pytest.raises(LoopyError):
        lp.generate_code_v2(knl)


def test_translation_unit_pickle():
    tunit = lp.make_kernel(
        "{[i]: 0<=i<16}",
        """
        y[i] = i
        """)
    assert isinstance(hash(tunit), int)

    from pickle import dumps, loads
    tunit = loads(dumps(tunit))
    assert isinstance(hash(tunit), int)


def test_creation_kwargs():
    # https://github.com/inducer/loopy/issues/705
    knl = lp.make_kernel(
        "{[i]: 0<=i<10}",
        "a[i] = foo() * i",
        substitutions={"foo": lp.SubstitutionRule("foo", (), 3.14)},
    )

    assert len(knl.default_entrypoint.substitutions) != 0

    # https://github.com/inducer/loopy/issues/705
    with pytest.raises(lp.LoopyError):
        lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            foo := 5
            a[i] = foo() * i
            """,
            substitutions={"foo": lp.SubstitutionRule("foo", (), 3.14)},
        )

    with pytest.raises(TypeError):
        knl = lp.make_kernel(
            "{[i]: 0<=i<10}",
            "a[i] = foo() * i",
            # not a known kwarg
            ksdfjlasdf=None)


def test_global_temps_with_multiple_base_storages(ctx_factory):
    # See https://github.com/inducer/loopy/issues/737

    n = 10
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    prg = lp.make_kernel(
        "{[r0, r1]: 0<=r0,r1< %s}" % n,
        """
        tmp0 = sum(r0, r0**2)
        ... gbarrier
        tmp1 = sum(r1, r1**3)
        ... gbarrier
        out = tmp0 + tmp1
        """,
        [lp.TemporaryVariable("tmp0",
                              shape=lp.auto,
                              address_space=lp.AddressSpace.GLOBAL,
                              base_storage="base1"),
         lp.TemporaryVariable("tmp1",
                              shape=lp.auto,
                              address_space=lp.AddressSpace.GLOBAL,
                              base_storage="base2"),
         ...],
        seq_dependencies=True
    )

    prg = lp.infer_unknown_types(prg)
    prg = lp.allocate_temporaries_for_base_storage(prg)
    print(prg)

    _, (out,) = prg(cq)

    assert out == sum(i**2 for i in range(n)) + sum(i**3 for i in range(n))


def test_t_unit_to_python_with_substs():
    t_unit = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
        subst_0(i) := abs(10.0 * (i-5))
        subst_1(i) := abs(10.0 * (i**2-5))

        y[i] = subst_0(i) + subst_1(i)
        """)

    lp.t_unit_to_python(t_unit)  # contains check to assert roundtrip equivalence


def test_type_inference_of_clbls_in_substitutions(ctx_factory):
    # Regression for https://github.com/inducer/loopy/issues/746
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
        subst_0(_0) := abs(10.0 * (_0-5))

        y[i] = subst_0(i)
        """)

    evt, (out,) = knl(cq)
    np.testing.assert_allclose(out.get(), np.abs(10.0*(np.arange(10)-5)))


def test_einsum_parsing(ctx_factory):
    ctx = ctx_factory()

    # See <https://github.com/inducer/loopy/issues/753>
    knl = lp.make_einsum("ik, kj -> ij", ["A", "B"])
    knl = lp.add_dtypes(knl, {"A": np.float32, "B": np.float32})
    lp.auto_test_vs_ref(knl, ctx, knl,
                        parameters={"Ni": 10, "Nj": 10, "Nk": 10})


def test_no_barrier_err_for_global_temps_with_base_storage(ctx_factory):
    # Regression for https://github.com/inducer/loopy/issues/748
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{[i,j]: 0<=i, j<16}",
        """
        for i
            tmp1[i] = i
            tmp2[i] = tmp1[i] + 2
        end
        ... gbarrier
        for j
            out[j] = tmp1[j] + tmp2[j]
        end
        """,
        [lp.TemporaryVariable("tmp1",
                              address_space=lp.AddressSpace.GLOBAL,
                              base_storage="base1",
                              shape=lp.auto),
         lp.TemporaryVariable("tmp2",
                              address_space=lp.AddressSpace.GLOBAL,
                              base_storage="base2",
                              shape=lp.auto),
         ...],
        seq_dependencies=True
    )
    knl = lp.split_iname(knl, "i", 4, inner_tag="l.0", outer_tag="g.0")
    knl = lp.split_iname(knl, "j", 4, inner_tag="l.0", outer_tag="g.0")

    _, (out,) = knl(cq, out_host=True)

    np.testing.assert_allclose(2*np.arange(16) + 2, out)


def test_dgemm_with_rectangular_tile_prefetch():
    # See <https://github.com/inducer/loopy/issues/724>
    t_unit = lp.make_kernel(
        "{[i,j,k]: 0<=i,j<72 and 0<=k<32}",
        """
        C[i,j] = sum(k, A[i,k] * B[k,j])
        """,
        [lp.GlobalArg("A,B", dtype=np.float64, shape=lp.auto),
         ...],
    )
    ref_t_unit = t_unit

    tx = 8
    ty = 23
    tk = 11

    t_unit = lp.split_iname(t_unit, "i", tx, inner_tag="l.0", outer_tag="g.0")
    t_unit = lp.split_iname(t_unit, "j", ty, inner_tag="l.1", outer_tag="g.1")
    t_unit = lp.split_iname(t_unit, "k", tk)
    t_unit = lp.add_prefetch(
        t_unit, "A",
        sweep_inames=["i_inner", "k_inner"],
        temporary_address_space=lp.AddressSpace.LOCAL,
        fetch_outer_inames=frozenset({"i_outer", "j_outer", "k_outer"}),
        dim_arg_names=["iprftch_A", "kprftch_A"],
        default_tag=None,
    )

    t_unit = lp.add_prefetch(
        t_unit, "B",
        sweep_inames=["k_inner", "j_inner"],
        temporary_address_space=lp.AddressSpace.LOCAL,
        fetch_outer_inames=frozenset({"i_outer", "j_outer", "k_outer"}),
        dim_arg_names=["kprftch_B", "jprftch_B"],
        default_tag=None,
    )

    t_unit = lp.split_iname(t_unit, "kprftch_A", tx, inner_tag="l.0")
    t_unit = lp.split_iname(t_unit, "iprftch_A", ty, inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, "jprftch_B", tx, inner_tag="l.0")
    t_unit = lp.split_iname(t_unit, "kprftch_B", ty, inner_tag="l.1")

    ctx = cl.create_some_context()
    lp.auto_test_vs_ref(ref_t_unit, ctx, t_unit)


def test_modulo_vs_type_context(ctx_factory):
    t_unit = lp.make_kernel(
            "{[i]: 0 <= i < 10}",
            """
            # previously, the float 'type context' would propagate into
            # the remainder, leading to 'i % 10.0' being generated, which
            # C/OpenCL did not like.
            <float64> a = i % 10
            """)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    t_unit(queue)


def test_barrier_non_zero_hw_lbound():
    t_unit = lp.make_kernel(
        ["{[i]: 1<=i<17}",
         "{[j]: 0<=j<16}"],
        """
        <> a[i] = i      {id=w_a}
        <> b[j] = 2*a[j] {id=w_b}
        """)

    t_unit = lp.tag_inames(t_unit, {"i": "l.0", "j": "l.0"})

    t_unit = lp.preprocess_kernel(t_unit)
    knl = lp.get_one_linearized_kernel(t_unit.default_entrypoint,
                                       t_unit.callables_table)

    assert barrier_between(knl, "w_a", "w_b")


def test_no_unnecessary_lbarrier(ctx_factory):
    # This regression would fail on loopy.git <= 268a7f4
    # (Issue reported by @thilinarmtb)

    t_unit = lp.make_kernel(
        "{[i_outer, i_inner]: 0 <= i_outer < n and 0 <= i_inner < 16}",
        """
        <> s_a[i_inner] = ai[i_outer * 16 + i_inner] {id=write_s_a}
        ao[i_outer * 16 + i_inner] = 2.0 * s_a[i_inner] {id=write_ao, dep=write_s_a}
        """,
        assumptions="n>=0")

    t_unit = lp.add_dtypes(t_unit, dict(ai=np.float32))
    t_unit = lp.tag_inames(t_unit, dict(i_inner="l.0", i_outer="g.0"))
    t_unit = lp.set_temporary_address_space(t_unit, "s_a", "local")
    t_unit = lp.prioritize_loops(t_unit, "i_outer,i_inner")

    t_unit = lp.preprocess_kernel(t_unit)
    knl = lp.get_one_linearized_kernel(t_unit.default_entrypoint,
                                       t_unit.callables_table)

    assert not barrier_between(knl, "write_s_a", "write_ao")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

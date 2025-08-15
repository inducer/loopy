__copyright__ = "Copyright (C) 2021-25 Kaushik Kulkarni"

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

import logging
import sys

import numpy as np
import pytest

import pyopencl as cl

import loopy as lp


logger = logging.getLogger(__name__)

import faulthandler

from pyopencl.tools import (
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

from loopy.version import (
    LOOPY_USE_LANGUAGE_VERSION_2018_2,  # noqa # pyright: ignore[reportUnusedImport]
)


__all__ = ["cl", "pytest_generate_tests"]  # "cl.create_some_context"


faulthandler.enable()


def test_same_loop_node():
    t_unit = lp.make_kernel(
        "{[i, j]: 0<=i,j<10}",
        """
        a[i, j] = i+j
        """,
        lang_version=(2018, 2),
        name="foo",
    )
    with pytest.raises(lp.LoopyError, match="can be nested within one another"):
        lp.get_kennedy_unweighted_fusion_candidates(t_unit.default_entrypoint,
                                                    {"i", "j"})


def test_loop_fusion_vanilla(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[i0, i1, j0, j1]: 0 <= i0, i1, j0, j1 < 10}",
        """
        a[i0]     = 1
        b[i1, j0] = 2  {id=write_b}
        c[j1]     = 3  {id=write_c}
        """,
    )
    ref_knl = knl

    fused_chunks = lp.get_kennedy_unweighted_fusion_candidates(
        knl["loopy_kernel"], frozenset(["j0", "j1"])
    )

    knl = knl.with_kernel(
        lp.rename_inames_in_batch(knl["loopy_kernel"], fused_chunks)
    )
    assert len(ref_knl["loopy_kernel"].all_inames()) == 4
    assert len(knl["loopy_kernel"].all_inames()) == 3
    assert (
        len(
            knl["loopy_kernel"].id_to_insn["write_b"].within_inames
            & knl["loopy_kernel"].id_to_insn["write_c"].within_inames
        )
        == 1
    )

    lp.auto_test_vs_ref(ref_knl, ctx, knl)


def test_loop_fusion_outer_iname_preventing_fusion(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[i0, j0, j1]: 0 <= i0, j0, j1 < 10}",
        """
        a[i0]     = 1
        b[i0, j0] = 2 {id=write_b}
        c[j1]     = 3 {id=write_c}
        """,
    )
    ref_knl = knl

    fused_chunks = lp.get_kennedy_unweighted_fusion_candidates(
        knl["loopy_kernel"], frozenset(["j0", "j1"])
    )

    knl = knl.with_kernel(
        lp.rename_inames_in_batch(knl["loopy_kernel"], fused_chunks)
    )

    assert len(knl["loopy_kernel"].all_inames()) == 3
    assert len(knl["loopy_kernel"].all_inames()) == 3
    assert (
        len(
            knl["loopy_kernel"].id_to_insn["write_b"].within_inames
            & knl["loopy_kernel"].id_to_insn["write_c"].within_inames
        )
        == 0
    )

    lp.auto_test_vs_ref(ref_knl, ctx, knl)


def test_loop_fusion_with_loop_independent_deps(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[j0, j1]: 0 <= j0, j1 < 10}",
        """
        a[j0]  = 1
        b[j1] = 2 * a[j1]
        """,
        seq_dependencies=True,
    )

    ref_knl = knl

    fused_chunks = lp.get_kennedy_unweighted_fusion_candidates(
        knl["loopy_kernel"], frozenset(["j0", "j1"])
    )

    knl = knl.with_kernel(
        lp.rename_inames_in_batch(knl["loopy_kernel"], fused_chunks)
    )

    assert len(ref_knl["loopy_kernel"].all_inames()) == 2
    assert len(knl["loopy_kernel"].all_inames()) == 1

    lp.auto_test_vs_ref(ref_knl, ctx, knl)


def test_loop_fusion_constrained_by_outer_loop_deps(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()

    knl = lp.make_kernel(
        "{[j0, j1]: 0 <= j0, j1 < 10}",
        """
        a[j0] = 1         {id=write_a}
        b     = 2         {id=write_b}
        c[j1] = 2 * a[j1] {id=write_c}
        """,
        seq_dependencies=True,
    )

    ref_knl = knl

    fused_chunks = lp.get_kennedy_unweighted_fusion_candidates(
        knl["loopy_kernel"], frozenset(["j0", "j1"])
    )

    knl = knl.with_kernel(
        lp.rename_inames_in_batch(knl["loopy_kernel"], fused_chunks)
    )

    assert len(ref_knl["loopy_kernel"].all_inames()) == 2
    assert len(knl["loopy_kernel"].all_inames()) == 2
    assert (
        len(
            knl["loopy_kernel"].id_to_insn["write_a"].within_inames
            & knl["loopy_kernel"].id_to_insn["write_c"].within_inames
        )
        == 0
    )

    lp.auto_test_vs_ref(ref_knl, ctx, knl)


def test_loop_fusion_with_loop_carried_deps1(ctx_factory: cl.CtxFactory):

    ctx = ctx_factory()
    knl = lp.make_kernel(
        "{[i0, i1]: 1<=i0, i1<10}",
        """
        x[i0] = i0         {id=first_write}
        x[i1-1] = i1 ** 2  {id=second_write}
        """,
        seq_dependencies=True,
    )

    ref_knl = knl

    fused_chunks = lp.get_kennedy_unweighted_fusion_candidates(
        knl["loopy_kernel"], frozenset(["i0", "i1"])
    )

    knl = knl.with_kernel(
        lp.rename_inames_in_batch(knl["loopy_kernel"], fused_chunks)
    )

    assert len(ref_knl["loopy_kernel"].all_inames()) == 2
    assert len(knl["loopy_kernel"].all_inames()) == 1
    assert (
        len(
            knl["loopy_kernel"].id_to_insn["first_write"].within_inames
            & knl["loopy_kernel"].id_to_insn["second_write"].within_inames
        )
        == 1
    )

    lp.auto_test_vs_ref(ref_knl, ctx, knl)


def test_loop_fusion_with_loop_carried_deps2(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()
    knl = lp.make_kernel(
        "{[i0, i1]: 1<=i0, i1<10}",
        """
        x[i0-1] = i0     {id=first_write}
        x[i1] = i1 ** 2  {id=second_write}
        """,
        seq_dependencies=True,
    )

    ref_knl = knl

    fused_chunks = lp.get_kennedy_unweighted_fusion_candidates(
        knl["loopy_kernel"], frozenset(["i0", "i1"])
    )

    knl = knl.with_kernel(
        lp.rename_inames_in_batch(knl["loopy_kernel"], fused_chunks)
    )

    assert len(ref_knl["loopy_kernel"].all_inames()) == 2
    assert len(knl["loopy_kernel"].all_inames()) == 2
    assert (
        len(
            knl["loopy_kernel"].id_to_insn["first_write"].within_inames
            & knl["loopy_kernel"].id_to_insn["second_write"].within_inames
        )
        == 0
    )

    lp.auto_test_vs_ref(ref_knl, ctx, knl)


def test_loop_fusion_with_indirection(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()
    rng = np.random.default_rng(42)
    map_ = rng.permutation(10)
    cq = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{[i0, i1]: 0<=i0, i1<10}",
        """
        x[i0] = i0            {id=first_write}
        x[map[i1]] = i1 ** 2  {id=second_write}
        """,
        seq_dependencies=True,
    )

    ref_knl = knl

    fused_chunks = lp.get_kennedy_unweighted_fusion_candidates(
        knl["loopy_kernel"], frozenset(["i0", "i1"])
    )

    knl = knl.with_kernel(
        lp.rename_inames_in_batch(knl["loopy_kernel"], fused_chunks)
    )

    assert len(ref_knl["loopy_kernel"].all_inames()) == 2
    assert len(knl["loopy_kernel"].all_inames()) == 2
    assert (
        len(
            knl["loopy_kernel"].id_to_insn["first_write"].within_inames
            & knl["loopy_kernel"].id_to_insn["second_write"].within_inames
        )
        == 0
    )

    _, (out1,) = ref_knl(cq, map=map_)
    _, (out2,) = knl(cq, map=map_)
    np.testing.assert_allclose(out1, out2)


def test_loop_fusion_with_induced_dependencies_from_sibling_nests(
        ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()
    t_unit = lp.make_kernel(
        "{[i0, j, i1, i2]: 0<=i0, j, i1, i2<10}",
        """
        <> tmp0[i0] = i0
        <> tmp1[j] = tmp0[j]
        <> tmp2[j] = j
        out1[i1] = tmp2[i1]
        out2[i2] = 2 * tmp1[i2]
        """,
    )
    ref_t_unit = t_unit
    knl = t_unit.default_entrypoint
    knl = lp.rename_inames_in_batch(
        knl,
        lp.get_kennedy_unweighted_fusion_candidates(knl, frozenset(["i0", "i1"])),
    )
    t_unit = t_unit.with_kernel(knl)

    # 'i1', 'i2' should not be fused. If fused that would lead to an
    # unshcedulable kernel. Making sure that the kernel 'runs' suffices that
    # the transformation was successful.
    lp.auto_test_vs_ref(ref_t_unit, ctx, t_unit)


def test_loop_fusion_on_reduction_inames(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()

    t_unit = lp.make_kernel(
        "{[i, j0, j1, j2]: 0<=i, j0, j1, j2<10}",
        """
        y0[i] = sum(j0, sum([j1], 2*A[i, j0, j1]))
        y1[i] = sum(j0, sum([j2], 3*A[i, j0, j2]))
        """,
        [lp.GlobalArg("A", dtype=np.float64, shape=lp.auto), ...],
    )
    ref_t_unit = t_unit
    knl = t_unit.default_entrypoint
    knl = lp.rename_inames_in_batch(
        knl,
        lp.get_kennedy_unweighted_fusion_candidates(knl, frozenset(["j1", "j2"])),
    )
    assert (
        knl.id_to_insn["insn"].reduction_inames()
        == knl.id_to_insn["insn_0"].reduction_inames()
    )

    t_unit = t_unit.with_kernel(knl)
    lp.auto_test_vs_ref(ref_t_unit, ctx, t_unit)


def test_loop_fusion_on_reduction_inames_with_depth_mismatch(
        ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()

    t_unit = lp.make_kernel(
        "{[i, j0, j1, j2, j3]: 0<=i, j0, j1, j2, j3<10}",
        """
        y0[i] = sum(j0, sum([j1], 2*A[i, j0, j1]))
        y1[i] = sum(j2, sum([j3], 3*A[i, j3, j2]))
        """,
        [lp.GlobalArg("A", dtype=np.float64, shape=lp.auto), ...],
    )
    ref_t_unit = t_unit
    knl = t_unit.default_entrypoint
    knl = lp.rename_inames_in_batch(
        knl,
        lp.get_kennedy_unweighted_fusion_candidates(knl, frozenset(["j1", "j3"])),
    )

    # cannot fuse 'j1', 'j3' because they are not nested within the same outer
    # inames.
    assert (
        knl.id_to_insn["insn"].reduction_inames()
        != knl.id_to_insn["insn_0"].reduction_inames()
    )

    t_unit = t_unit.with_kernel(knl)
    lp.auto_test_vs_ref(ref_t_unit, ctx, t_unit)


def test_loop_fusion_on_outer_reduction_inames(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()

    t_unit = lp.make_kernel(
        "{[i, j0, j1, j2, j3]: 0<=i, j0, j1, j2, j3<10}",
        """
        y0[i] = sum(j0, sum([j1], 2*A[i, j0, j1]))
        y1[i] = sum(j2, sum([j3], 3*A[i, j3, j2]))
        """,
        [lp.GlobalArg("A", dtype=np.float64, shape=lp.auto), ...],
    )
    ref_t_unit = t_unit
    knl = t_unit.default_entrypoint
    knl = lp.rename_inames_in_batch(
        knl,
        lp.get_kennedy_unweighted_fusion_candidates(knl, frozenset(["j0", "j2"])),
    )

    assert (
        len(
            knl.id_to_insn["insn"].reduction_inames()
            & knl.id_to_insn["insn_0"].reduction_inames()
        )
        == 1
    )

    t_unit = t_unit.with_kernel(knl)
    lp.auto_test_vs_ref(ref_t_unit, ctx, t_unit)


def test_loop_fusion_reduction_inames_simple(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()

    t_unit = lp.make_kernel(
        "{[i, j0, j1]: 0<=i, j0, j1<10}",
        """
        y0[i] = sum(j0, 2*A[i, j0])
        y1[i] = sum(j1, 3*A[i, j1])
        """,
        [lp.GlobalArg("A", dtype=np.float64, shape=lp.auto), ...],
    )
    ref_t_unit = t_unit
    knl = t_unit.default_entrypoint
    knl = lp.rename_inames_in_batch(
        knl,
        lp.get_kennedy_unweighted_fusion_candidates(knl, frozenset(["j0", "j1"])),
    )

    assert (
        knl.id_to_insn["insn"].reduction_inames()
        == knl.id_to_insn["insn_0"].reduction_inames()
    )

    t_unit = t_unit.with_kernel(knl)
    lp.auto_test_vs_ref(ref_t_unit, ctx, t_unit)


def test_redn_loop_fusion_with_non_candidates_loops_in_nest(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()
    t_unit = lp.make_kernel(
        "{[i, j1, j2, d]: 0<=i, j1, j2, d<10}",
        """
        for i
          for d
            out1[i, d] = sum(j1, 2 * j1*i)
          end
          out2[i] = sum(j2, 2 * j2)
        end
        """,
        seq_dependencies=True,
    )
    ref_t_unit = t_unit

    knl = t_unit.default_entrypoint
    knl = lp.rename_inames_in_batch(
        knl,
        lp.get_kennedy_unweighted_fusion_candidates(knl, frozenset(["j1", "j2"])),
    )

    assert not (
        knl.id_to_insn["insn"].reduction_inames()
        & knl.id_to_insn["insn_0"].reduction_inames()
    )

    lp.auto_test_vs_ref(ref_t_unit, ctx, t_unit.with_kernel(knl))


def test_reduction_loop_fusion_with_multiple_redn_in_same_insn(
        ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()
    t_unit = lp.make_kernel(
        "{[j1, j2]: 0<=j1, j2<10}",
        """
        out = sum(j1, 2*j1) + sum(j2, 2*j2)
        """,
        seq_dependencies=True,
    )
    ref_t_unit = t_unit

    knl = t_unit.default_entrypoint
    knl = lp.rename_inames_in_batch(
        knl,
        lp.get_kennedy_unweighted_fusion_candidates(knl, frozenset(["j1", "j2"])),
    )

    assert len(knl.id_to_insn["insn"].reduction_inames()) == 1

    lp.auto_test_vs_ref(ref_t_unit, ctx, t_unit.with_kernel(knl))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])

# vim: fdm=marker

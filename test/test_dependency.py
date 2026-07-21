__copyright__ = """
Copyright (C) 2026 Addison Alvey-Blanco
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


import namedisl as nisl
import numpy as np
import pytest

import pyopencl as cl
from pyopencl.tools import (  # ruff:ignore[unused-import]
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

from pymbolic import var

import loopy as lp
import loopy.kernel.dependency as dep
from loopy.diagnostic import LoopyError
from loopy.kernel.instruction import HappensAfter
from loopy.schedule import (
    Barrier,
    CallKernel,
    EnterLoop,
    LeaveLoop,
    ReturnFromKernel,
    RunInstruction,
)
from loopy.schedule.verification import (
    _BarrierRecord,
    _build_enforced_order,
    _build_strict_lexicographic_order,
    _build_timestamp_relations,
    _get_timestamp_points_from_linearization,
    _PreciseSchedule,
    _StatementRecord,
    verify_happens_after_is_enforced,
)
from loopy.symbolic import SubArrayRef
from loopy.version import (
    LOOPY_USE_LANGUAGE_VERSION_2018_2,  # ruff:ignore[unused-import]
)


def test_add_lexicographic_happens_after_is_strict_for_self() -> None:
    t_unit = lp.make_kernel(
        "{ [i] : 0 <= i < N }",
        """
        a[i] = 2 * a[i] {id=S}
        b[i] = a[i] {id=T}
        """,
    )

    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint

    self_relation = kernel.id_to_insn["T"].happens_after["T"].instances_rel
    previous_relation = kernel.id_to_insn["T"].happens_after["S"].instances_rel

    assert self_relation is not None
    assert previous_relation is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    self_relation = nisl.make_map(self_relation)
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    previous_relation = nisl.make_map(previous_relation)
    assert self_relation.equals(
        nisl.make_map("""
        [N] -> {
            [i_after] -> [i_before] :
                0 <= i_before < i_after < N
        }
        """)
    )
    assert previous_relation.equals(
        nisl.make_map("""
        [N] -> {
            [i_after] -> [i_before] :
                0 <= i_before <= i_after < N
        }
        """)
    )


def test_add_lexicographic_happens_after_uses_domain_dimension_order() -> None:
    t_unit = lp.make_kernel(
        "{ [z, a] : 0 <= z < NZ and 0 <= a < NA }",
        "out[z, a] = z + a {id=S}",
    )

    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint
    self_relation = kernel.id_to_insn["S"].happens_after["S"].instances_rel

    assert self_relation is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    self_relation = nisl.make_map(self_relation)
    assert self_relation.equals(
        nisl.make_map("""
        [NZ, NA] -> {
            [z_after, a_after] -> [z_before, a_before] :
                0 <= z_after < NZ and
                0 <= z_before < NZ and
                0 <= a_after < NA and
                0 <= a_before < NA and
                (z_before < z_after or
                 (z_before = z_after and a_before < a_after))
        }
        """)
    )


def test_add_lexicographic_happens_after_orders_mixed_loop_nests() -> None:
    t_unit = lp.make_kernel(
        [
            "[NI] -> { [i] : 0 <= i < NI }",
            "[i, NJ] -> { [j] : 0 <= j < NJ }",
            "[i, NK] -> { [k] : 0 <= k < NK }",
            "[NQ] -> { [q] : 0 <= q < NQ }",
        ],
        """
        a[i, j] = i + j {id=S}
        b[i, k] = i + k {id=T}
        c[q] = q {id=U}
        """,
    )

    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint
    shared_nest_relation = (
        kernel.id_to_insn["T"].happens_after["S"].instances_rel
    )
    disjoint_nest_relation = (
        kernel.id_to_insn["U"].happens_after["T"].instances_rel
    )

    assert kernel.id_to_insn["T"].happens_after.keys() == {"S", "T"}
    assert kernel.id_to_insn["U"].happens_after.keys() == {"T", "U"}
    assert shared_nest_relation is not None
    assert disjoint_nest_relation is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    shared_nest_relation = nisl.make_map(shared_nest_relation)
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    disjoint_nest_relation = nisl.make_map(disjoint_nest_relation)
    assert shared_nest_relation.equals(
        nisl.make_map("""
        [NI, NJ, NK] -> {
            [i_after, k_after] -> [i_before, j_before] :
                0 <= i_before <= i_after < NI and
                0 <= j_before < NJ and
                0 <= k_after < NK
        }
        """)
    )
    assert disjoint_nest_relation.equals(
        nisl.make_map("""
        [NI, NK, NQ] -> {
            [q_after] -> [i_before, k_before] :
                0 <= q_after < NQ and
                0 <= i_before < NI and
                0 <= k_before < NK
        }
        """)
    )


def test_access_relation_finder_tracks_reads_and_writes_per_statement() -> None:
    t_unit = lp.make_kernel(
        "{ [i] : 1 <= i < N }",
        """
        a[i] = b[i - 1] + c[i] {id=S}
        b[i] = a[i] {id=T}
        """,
    )

    kernel = t_unit.default_entrypoint
    rel_find = dep.AccessRelationFinder(kernel)
    for stmt in kernel.instructions:
        assert isinstance(stmt, lp.MultiAssignmentBase)
        for assignee in stmt.assignees:
            rel_find(assignee, stmt.id, dep.AccessType.write)
        rel_find(stmt.expression, stmt.id, dep.AccessType.read)

    assert rel_find.read_relations["S"].keys() == {"b", "c"}
    assert rel_find.write_relations["S"].keys() == {"a"}
    assert rel_find.read_relations["T"].keys() == {"a"}
    assert rel_find.write_relations["T"].keys() == {"b"}
    assert rel_find.read_relations["S"]["b"].equals(
        nisl.make_map("[N] -> { [i] -> [ax_0 = i - 1] : 1 <= i < N }")
    )
    assert rel_find.write_relations["S"]["a"].equals(
        nisl.make_map("[N] -> { [i] -> [ax_0 = i] : 1 <= i < N }")
    )


def test_access_relation_names_do_not_clash_with_inames() -> None:
    t_unit = lp.make_kernel(
        """
        { [ax_0, ax_1] :
            0 <= ax_0 < N and 0 <= ax_1 < M
        }
        """,
        "out[ax_0, ax_1] = a[ax_0] + b[ax_1] {id=S}",
    )

    kernel = t_unit.default_entrypoint
    insn = kernel.id_to_insn["S"]
    rel_find = dep.AccessRelationFinder(kernel)
    rel_find(insn.expression, insn.id, dep.AccessType.read)

    a_cell_names = rel_find.read_relations["S"]["a"].space.dimtype_to_names[
        nisl.DimType.out
    ]
    b_cell_names = rel_find.read_relations["S"]["b"].space.dimtype_to_names[
        nisl.DimType.out
    ]
    assert a_cell_names == b_cell_names
    assert set(a_cell_names).isdisjoint(kernel.all_variable_names())


def test_access_relation_finder_handles_reduction() -> None:
    t_unit = lp.make_kernel(
        """
        { [i, j, k] :
            0 <= i < NI and 0 <= j < NJ and 0 <= k < NK
        }
        """,
        "out[i, j] = sum(k, a[i, j, k]) {id=S}",
    )

    kernel = t_unit.default_entrypoint
    insn = kernel.id_to_insn["S"]
    rel_find = dep.AccessRelationFinder(kernel)
    rel_find(insn.expression, insn.id, dep.AccessType.read)

    assert rel_find.read_relations["S"]["a"].equals(
        nisl.make_map("""
            [NI, NJ, NK] -> {
                [i, j] -> [ax_0 = i, ax_1 = j, ax_2] :
                    0 <= i < NI and
                    0 <= j < NJ and
                    0 <= ax_2 < NK
            }
            """)
    )


def test_access_relation_finder_handles_sub_array_ref() -> None:
    t_unit = lp.make_kernel(
        """
        { [i, j, k] :
            0 <= i < NI and 0 <= j < NJ and 0 <= k < NK
        }
        """,
        "out[i, j] = a[i, j, 0] {id=S}",
    )

    kernel = t_unit.default_entrypoint
    rel_find = dep.AccessRelationFinder(kernel)
    # Build the swept access directly so this remains a mapper unit test and
    # does not require setting up an array-valued callable.
    sub_array_ref = SubArrayRef(
        (var("k"),),
        var("a")[var("i"), var("j"), var("k")],
    )
    rel_find(sub_array_ref, "S", dep.AccessType.read)

    assert rel_find.read_relations["S"]["a"].equals(
        nisl.make_map("""
            [NI, NJ, NK] -> {
                [i, j] -> [ax_0 = i, ax_1 = j, ax_2] :
                    0 <= i < NI and
                    0 <= j < NJ and
                    0 <= ax_2 < NK
            }
            """)
    )


def _relax_strict_happens_after(
    instructions: str,
    domain: str = "{ [i] : 0 <= i < N }",
) -> lp.LoopKernel:
    t_unit = lp.make_kernel(domain, instructions)
    t_unit = dep.add_lexicographic_happens_after(t_unit)
    return dep.relax_strict_happens_after(t_unit).default_entrypoint


@pytest.mark.parametrize(
    ("source", "sink"),
    (
        ("a[i, j] = 1", "b[i, j] = a[i, j]"),
        ("a[i, j] = 1", "a[i, j] = 2"),
        ("b[i, j] = a[i, j]", "a[i, j] = 2"),
    ),
    ids=("read-after-write", "write-after-write", "write-after-read"),
)
def test_relax_strict_happens_after_finds_direct_hazards(
    source: str, sink: str
) -> None:
    kernel = _relax_strict_happens_after(
        f"""
        {source} {{id=S}}
        {sink} {{id=T}}
        """,
        "{ [i, j] : 0 <= i < NI and 0 <= j < NJ }",
    )

    required_order = kernel.id_to_insn["T"].happens_after["S"].instances_rel
    assert required_order is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    required_order = nisl.make_map(required_order)
    assert required_order.equals(
        nisl.make_map("""
            [NI, NJ] -> {
                [i_after, j_after] ->
                [i_before = i_after, j_before = j_after] :
                    0 <= i_after < NI and 0 <= j_after < NJ
            }
            """)
    )


def test_has_precise_dependencies() -> None:
    legacy_kernel = lp.make_kernel(
        "{ [i] : 0 <= i < N }",
        """
        a[i] = i {id=S}
        b[i] = a[i] {id=T, dep=S}
        """,
    ).default_entrypoint
    assert not dep.has_precise_dependencies(legacy_kernel)

    precise_kernel = dep.add_lexicographic_happens_after(legacy_kernel)
    assert dep.has_precise_dependencies(precise_kernel)

    t_insn = precise_kernel.id_to_insn["T"]
    mixed_happens_after = dict(t_insn.happens_after)
    mixed_happens_after["S"] = HappensAfter(instances_rel=None)
    mixed_kernel = precise_kernel.copy(
        instructions=tuple(
            insn.copy(happens_after=mixed_happens_after)
            if insn.id == "T"
            else insn
            for insn in precise_kernel.instructions
        )
    )

    with pytest.raises(LoopyError, match="mixes precise and legacy"):
        dep.has_precise_dependencies(mixed_kernel)


def test_relax_strict_happens_after_tracks_scalar_accesses() -> None:
    kernel = _relax_strict_happens_after(
        """
        <> tmp = i {id=S}
        out[i] = tmp {id=T}
        """,
        "{ [i] : 0 <= i < N }",
    )

    required_order = kernel.id_to_insn["T"].happens_after["S"].instances_rel
    assert required_order is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    required_order = nisl.make_map(required_order)
    assert required_order.equals(
        nisl.make_map("""
            [N] -> {
                [i_after] -> [i_before] :
                    0 <= i_before <= i_after < N
            }
            """)
    )


@pytest.mark.parametrize(
    "instructions",
    (
        """
        a[i] = 1 {id=S}
        b[i] = 2 {id=T}
        """,
        """
        a[i] = 1 {id=S}
        b[i] = a[i + N] {id=T}
        """,
    ),
    ids=("different-variables", "disjoint-footprints"),
)
def test_relax_strict_happens_after_drops_nonconflicting_edges(
    instructions: str,
) -> None:
    kernel = _relax_strict_happens_after(instructions)

    assert "S" not in kernel.id_to_insn["T"].happens_after


def test_relax_strict_happens_after_tracks_live_footprints_through_a_chain() -> (
    None
):
    kernel = _relax_strict_happens_after(
        """
        a[i, j] = 1 {id=A}
        b[i, j] = 2 {id=B}
        a[2*i, j] = 3 {id=C}
        out[i, j] = a[i, j] {id=D}
        """,
        "{ [i, j] : 0 <= i < NI and 0 <= j < NJ }",
    )

    recent_order = kernel.id_to_insn["D"].happens_after["C"].instances_rel
    fallback_order = kernel.id_to_insn["D"].happens_after["A"].instances_rel
    assert recent_order is not None
    assert fallback_order is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    recent_order = nisl.make_map(recent_order)
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    fallback_order = nisl.make_map(fallback_order)
    assert recent_order.equals(
        nisl.make_map("""
            [NI, NJ] -> {
                [i_after, j_after] -> [i_before, j_before] :
                    i_after = 2*i_before and
                    j_after = j_before and
                    0 <= i_after < NI and
                    0 <= i_before < NI and
                    0 <= j_after < NJ
            }
            """)
    )
    assert "B" not in kernel.id_to_insn["D"].happens_after
    assert fallback_order.equals(
        nisl.make_map("""
            [NI, NJ] -> {
                [i_after, j_after] ->
                [i_before = i_after, j_before = j_after] :
                    0 <= i_after < NI and
                    0 <= j_after < NJ and
                    i_after mod 2 = 1
            }
            """)
    )


def test_relax_strict_happens_after_composes_user_supplied_relations() -> None:
    t_unit = lp.make_kernel(
        [
            "[N] -> { [i] : 0 <= i < 2*N }",
            "[N] -> { [j] : 0 <= j < N }",
            "[N] -> { [k] : 1 <= k < N }",
        ],
        """
        a[i] = 1 {id=S}
        tmp[j] = 0 {id=T}
        out[k] = a[2*k - 2] {id=U}
        """,
    )
    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint
    t_after_s = HappensAfter(
        instances_rel=nisl.make_map("""
            [N] -> {
                [j_after] -> [i_before = 2*j_after] :
                    0 <= j_after < N
            }
            """).as_isl()
    )
    u_after_t = HappensAfter(
        instances_rel=nisl.make_map("""
            [N] -> {
                [k_after] -> [j_before = k_after - 1] :
                    1 <= k_after < N
            }
            """).as_isl()
    )
    kernel = kernel.copy(
        instructions=tuple(
            stmt.copy(
                happens_after={
                    stmt.id: stmt.happens_after[stmt.id],
                    **({"S": t_after_s} if stmt.id == "T" else {}),
                    **({"T": u_after_t} if stmt.id == "U" else {}),
                }
            )
            for stmt in kernel.instructions
        )
    )
    kernel = dep.relax_strict_happens_after(kernel)

    required_order = kernel.id_to_insn["U"].happens_after["S"].instances_rel
    assert required_order is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    required_order = nisl.make_map(required_order)
    assert required_order.equals(
        nisl.make_map("""
            [N] -> {
                [k_after] -> [i_before = 2*k_after - 2] :
                    1 <= k_after < N
            }
            """)
    )


def test_relax_strict_happens_after_unions_branched_paths() -> None:
    t_unit = lp.make_kernel(
        "{ [i] : 0 <= i < N }",
        """
        a[i] = 1 {id=A}
        b[i] = 2 {id=B}
        c[i] = 3 {id=C}
        d[i] = a[i] {id=D}
        """,
    )
    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint
    cross_order = kernel.id_to_insn["B"].happens_after["A"]
    predecessors = {
        "A": (),
        "B": ("A",),
        "C": ("A",),
        "D": ("B", "C"),
    }
    kernel = kernel.copy(
        instructions=[
            insn.copy(
                happens_after={
                    insn.id: insn.happens_after[insn.id],
                    **dict.fromkeys(predecessors[insn.id], cross_order),
                }
            )
            for insn in kernel.instructions
        ]
    )

    kernel = dep.relax_strict_happens_after(kernel)

    required_order = kernel.id_to_insn["D"].happens_after["A"].instances_rel
    assert required_order is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    required_order = nisl.make_map(required_order)
    assert required_order.equals(
        nisl.make_map("""
            [N] -> {
                [i_after] -> [i_before = i_after] : 0 <= i_after < N
            }
            """)
    )


def test_statement_timestamps_with_calls_inside_outer_loop() -> None:
    kernel = lp.make_kernel(
        """{
            [batch, i, j, k] :
                0 <= batch < 4 and 0 <= i < 8 and
                0 <= j < 16 and 0 <= k < 32
        }""",
        """
        a[batch, i, j] = i + j {id=A}
        b[batch, i] = i {id=B}
        c[batch, k] = k {id=C}
        """,
    ).default_entrypoint.copy(
        state=lp.KernelState.LINEARIZED,
        linearization=(
            EnterLoop("batch"),
            CallKernel("phase0"),
            EnterLoop("i"),
            EnterLoop("j"),
            RunInstruction("A"),
            LeaveLoop("j"),
            RunInstruction("B"),
            LeaveLoop("i"),
            ReturnFromKernel("phase0"),
            CallKernel("phase1"),
            EnterLoop("k"),
            RunInstruction("C"),
            LeaveLoop("k"),
            ReturnFromKernel("phase1"),
            LeaveLoop("batch"),
        ),
    )

    assert _get_timestamp_points_from_linearization(kernel) == _PreciseSchedule(
        statements={
            "A": _StatementRecord(
                timestamp=(0, "batch", 1, 2, "i", 3, "j", 4),
                subkernel_idx=1,
            ),
            "B": _StatementRecord(
                timestamp=(0, "batch", 1, 2, "i", 6),
                subkernel_idx=1,
            ),
            "C": _StatementRecord(
                timestamp=(0, "batch", 9, 10, "k", 11),
                subkernel_idx=9,
            ),
        },
        barriers=(),
    )


def test_statement_timestamps_with_imperfect_loop_nesting() -> None:
    t_unit = lp.make_kernel(
        "{ [i, j, k] : 0 <= i, j, k < 8 }",
        """
        <> before = 0 {id=before}
        <> outer = i {id=outer}
        <> deep = i + j + k {id=deep}
        <> after_inner = i {id=after_inner}
        <> after = 0 {id=after}
        """,
    )
    t_unit = dep.add_lexicographic_happens_after(t_unit)
    t_unit = lp.preprocess_program(t_unit)
    kernel = lp.linearize(t_unit).default_entrypoint

    assert _get_timestamp_points_from_linearization(kernel) == _PreciseSchedule(
        statements={
            "before": _StatementRecord(timestamp=(0, 1), subkernel_idx=0),
            "outer": _StatementRecord(
                timestamp=(0, 2, "i", 3), subkernel_idx=0
            ),
            "deep": _StatementRecord(
                timestamp=(0, 2, "i", 4, "j", 5, "k", 6),
                subkernel_idx=0,
            ),
            "after_inner": _StatementRecord(
                timestamp=(0, 2, "i", 9), subkernel_idx=0
            ),
            "after": _StatementRecord(timestamp=(0, 11), subkernel_idx=0),
        },
        barriers=(),
    )


def test_timestamp_relation_keeps_parallel_inames_in_instance_domain() -> None:
    t_unit = lp.make_kernel(
        "{ [g, l, i] : 0 <= g < 4 and 0 <= l < 8 and 0 <= i < 16 }",
        "out[g, l, i] = g + l + i {id=S}",
    )
    t_unit = lp.tag_inames(t_unit, {"g": "g.0", "l": "l.0"})
    t_unit = lp.preprocess_program(t_unit)
    kernel = lp.linearize(t_unit).default_entrypoint

    precise_schedule = _get_timestamp_points_from_linearization(kernel)
    assert precise_schedule.statements["S"] == _StatementRecord(
        timestamp=(0, 1, "i", 2),
        subkernel_idx=0,
    )

    timestamp_relations, _, _ = _build_timestamp_relations(
        kernel, precise_schedule, kernel.get_var_name_generator()
    )
    timestamp_relation = timestamp_relations["S"]
    assert timestamp_relation.equals(
        nisl.make_map("""
        {
            [g, l, i] -> [__ts_0, __ts_1, __ts_2, __ts_3] :
                0 <= g < 4 and 0 <= l < 8 and 0 <= i < 16 and
                __ts_0 = 0 and __ts_1 = 1 and
                __ts_2 = i and __ts_3 = 2
        }
        """)
    )


def test_analysis_names_do_not_clash_with_kernel_names() -> None:
    t_unit = lp.make_kernel(
        """
        [__group_0] -> {
            [g, __ts_0, __ts_0_later] :
                0 <= g < __group_0 and
                0 <= __ts_0, __ts_0_later < 2
        }
        """,
        """
        a[g, __ts_0, __ts_0_later] = g {id=S}
        out[g, __ts_0, __ts_0_later] = a[g, __ts_0, __ts_0_later] {id=T}
        """,
    )
    t_unit = dep.add_lexicographic_happens_after(t_unit)
    t_unit = dep.relax_strict_happens_after(t_unit)
    t_unit = lp.tag_inames(t_unit, {"g": "g.0"})
    t_unit = lp.preprocess_program(t_unit)
    kernel = lp.linearize(t_unit).default_entrypoint

    precise_schedule = _get_timestamp_points_from_linearization(kernel)
    stmt_relations, _, timestamp_order = _build_timestamp_relations(
        kernel, precise_schedule, kernel.get_var_name_generator()
    )

    timestamp_names = stmt_relations["S"].space.dimtype_to_names[
        nisl.DimType.out
    ]
    later_names = timestamp_order.space.dimtype_to_names[nisl.DimType.in_]
    earlier_names = timestamp_order.space.dimtype_to_names[nisl.DimType.out]
    analysis_names = {*timestamp_names, *later_names, *earlier_names}

    assert len(analysis_names) == 3 * len(timestamp_names)
    assert analysis_names.isdisjoint(kernel.all_variable_names())
    verify_happens_after_is_enforced(kernel)


def test_strict_lexicographic_timestamp_order() -> None:
    order = _build_strict_lexicographic_order(
        ("t0_later", "t1_later", "t2_later"),
        ("t0_earlier", "t1_earlier", "t2_earlier"),
    )

    assert order.equals(
        nisl.make_map("""
        {
            [t0_later, t1_later, t2_later] ->
                [t0_earlier, t1_earlier, t2_earlier] :
                    t0_later > t0_earlier;
            [t0_later, t1_later, t2_later] ->
                [t0_earlier, t1_earlier, t2_earlier] :
                    t0_later = t0_earlier and
                    t1_later > t1_earlier;
            [t0_later, t1_later, t2_later] ->
                [t0_earlier, t1_earlier, t2_earlier] :
                    t0_later = t0_earlier and
                    t1_later = t1_earlier and
                    t2_later > t2_earlier
        }
    """)
    )


def test_statement_timestamps_record_local_and_global_barriers() -> None:
    local_barrier = Barrier(
        comment="local synchronization",
        synchronization_kind="local",
        mem_kind="local",
        originating_insn_id=None,
    )
    global_barrier = Barrier(
        comment="global synchronization",
        synchronization_kind="global",
        mem_kind="global",
        originating_insn_id=None,
    )
    kernel = lp.make_kernel(
        "{ [batch, i, j] : 0 <= batch, i, j < 8 }",
        """
        a[batch, i] = i {id=producer}
        b[batch, i] = a[batch, i] {id=after_local}
        c[batch, j] = j {id=consumer}
        """,
    ).default_entrypoint.copy(
        state=lp.KernelState.LINEARIZED,
        linearization=(
            EnterLoop("batch"),
            CallKernel("phase0"),
            EnterLoop("i"),
            RunInstruction("producer"),
            local_barrier,
            RunInstruction("after_local"),
            LeaveLoop("i"),
            ReturnFromKernel("phase0"),
            global_barrier,
            CallKernel("phase1"),
            EnterLoop("j"),
            RunInstruction("consumer"),
            LeaveLoop("j"),
            ReturnFromKernel("phase1"),
            LeaveLoop("batch"),
        ),
    )

    precise_schedule = _get_timestamp_points_from_linearization(kernel)
    assert precise_schedule == _PreciseSchedule(
        statements={
            "producer": _StatementRecord(
                timestamp=(0, "batch", 1, 2, "i", 3),
                subkernel_idx=1,
            ),
            "after_local": _StatementRecord(
                timestamp=(0, "batch", 1, 2, "i", 5),
                subkernel_idx=1,
            ),
            "consumer": _StatementRecord(
                timestamp=(0, "batch", 9, 10, "j", 11),
                subkernel_idx=9,
            ),
        },
        barriers=(
            _BarrierRecord(
                timestamp=(0, "batch", 1, 2, "i", 4),
                barrier=local_barrier,
                subkernel_idx=1,
            ),
            _BarrierRecord(
                timestamp=(0, "batch", 8),
                barrier=global_barrier,
                subkernel_idx=None,
            ),
        ),
    )

    _, barrier_relations, _ = _build_timestamp_relations(
        kernel, precise_schedule, kernel.get_var_name_generator()
    )
    assert len(barrier_relations) == 2
    assert barrier_relations[0].equals(
        nisl.make_map("""
        {
            [batch, i] ->
                [__ts_0, __ts_1, __ts_2, __ts_3, __ts_4, __ts_5] :
                    0 <= batch < 8 and 0 <= i < 8 and
                    __ts_0 = 0 and __ts_1 = batch and
                    __ts_2 = 1 and __ts_3 = 2 and
                    __ts_4 = i and __ts_5 = 4
        }
    """)
    )
    assert barrier_relations[1].equals(
        nisl.make_map("""
        {
            [batch] ->
                [__ts_0, __ts_1, __ts_2, __ts_3, __ts_4, __ts_5] :
                    0 <= batch < 8 and
                    __ts_0 = 0 and __ts_1 = batch and
                    __ts_2 = 8 and __ts_3 = 0 and
                    __ts_4 = 0 and __ts_5 = 0
        }
    """)
    )


def test_local_barrier_orders_work_items_in_the_same_group() -> None:
    t_unit = lp.make_kernel(
        """
        {
            [g, l, i] :
                0 <= g < 2 and 0 <= l < 4 and 0 <= i < 2
        }
        """,
        """
        <> tmp[g, l, i] = g + l + i {id=source}
        out[g, l, i] = tmp[g, (l + 1) % 4, i] {id=sink}
        """,
    )
    t_unit = dep.add_lexicographic_happens_after(t_unit)
    t_unit = dep.relax_strict_happens_after(t_unit)
    t_unit = lp.set_temporary_address_space(
        t_unit, "tmp", lp.AddressSpace.LOCAL
    )
    t_unit = lp.tag_inames(t_unit, {"g": "g.0", "l": "l.0"})
    t_unit = lp.preprocess_program(t_unit)
    kernel = lp.linearize(t_unit).default_entrypoint

    precise_schedule = _get_timestamp_points_from_linearization(kernel)
    name_generator = kernel.get_var_name_generator()
    stmt_relations, barrier_relations, timestamp_order = (
        _build_timestamp_relations(
            kernel, precise_schedule, name_generator
        )
    )
    enforced = _build_enforced_order(
        kernel,
        "sink",
        "source",
        precise_schedule,
        stmt_relations,
        barrier_relations,
        timestamp_order,
        name_generator,
    )

    assert enforced.equals(
        nisl.make_map("""
        {
            [g_after, l_after, i_after] ->
                [g_before, l_before, i_before] :
                    0 <= g_after < 2 and 0 <= l_after < 4 and
                    0 <= i_after < 2 and
                    g_before = g_after and
                    0 <= l_before < 4 and
                    0 <= i_before <= i_after
        }
    """)
    )


def test_global_barrier_orders_all_work_items() -> None:
    t_unit = lp.make_kernel(
        "{ [g, l] : 0 <= g < 2 and 0 <= l < 4 }",
        """
        a[g, l] = g + l {id=source}
        out[g, l] = a[(g + 1) % 2, l] {id=sink}
        """,
    )
    t_unit = dep.add_lexicographic_happens_after(t_unit)
    t_unit = dep.relax_strict_happens_after(t_unit)
    t_unit = lp.tag_inames(t_unit, {"g": "g.0", "l": "l.0"})
    t_unit = lp.set_options(t_unit, insert_gbarriers=True)
    t_unit = lp.preprocess_program(t_unit)
    kernel = lp.linearize(t_unit).default_entrypoint

    precise_schedule = _get_timestamp_points_from_linearization(kernel)
    name_generator = kernel.get_var_name_generator()
    stmt_relations, barrier_relations, timestamp_order = (
        _build_timestamp_relations(
            kernel, precise_schedule, name_generator
        )
    )
    enforced = _build_enforced_order(
        kernel,
        "sink",
        "source",
        precise_schedule,
        stmt_relations,
        barrier_relations,
        timestamp_order,
        name_generator,
    )

    assert enforced.equals(
        nisl.make_map("""
        {
            [g_after, l_after] -> [g_before, l_before] :
                0 <= g_after < 2 and 0 <= l_after < 4 and
                0 <= g_before < 2 and 0 <= l_before < 4
        }
    """)
    )


def test_verification_enforces_self_recurrence(
    ctx_factory: cl.CtxFactory,
) -> None:
    ref_t_unit = lp.make_kernel(
        "{ [i] : 1 <= i < N }",
        "a[i] = a[i - 1] + x[i] {id=S}",
        [lp.GlobalArg("a,x", dtype=np.int32, shape=lp.auto), "..."],
    )
    t_unit = ref_t_unit
    t_unit = dep.add_lexicographic_happens_after(t_unit)
    t_unit = dep.relax_strict_happens_after(t_unit)

    required_order = (
        t_unit.default_entrypoint
        .id_to_insn["S"]
        .happens_after["S"]
        .instances_rel
    )
    assert required_order is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    assert nisl.make_map(required_order).equals(
        nisl.make_map("""
            [N] -> {
                [i_after] -> [i_before = i_after - 1] :
                    2 <= i_after < N
            }
            """)
    )

    lp.auto_test_vs_ref(
        ref_t_unit,
        ctx_factory(),
        t_unit,
        parameters={"N": 64},
        print_code=False,
        quiet=True,
    )

    parallel_t_unit = lp.tag_inames(t_unit, {"i": "l.0"})
    parallel_kernel = parallel_t_unit.default_entrypoint.copy(
        state=lp.KernelState.LINEARIZED,
        linearization=(
            CallKernel("device_program"),
            RunInstruction("S"),
            ReturnFromKernel("device_program"),
        ),
    )
    with pytest.raises(
        LoopyError,
        match="schedule does not enforce 'S' after 'S'",
    ):
        verify_happens_after_is_enforced(
            parallel_t_unit.with_kernel(parallel_kernel)
        )


def test_verification_rejects_unenforced_order() -> None:
    t_unit = lp.make_kernel(
        "{ [i] : 0 <= i < N }",
        """
        a[i] = i {id=S}
        b[i] = a[i] {id=T}
        """,
    )
    t_unit = dep.add_lexicographic_happens_after(t_unit)

    kernel = t_unit.default_entrypoint.copy(
        state=lp.KernelState.LINEARIZED,
        linearization=(
            CallKernel("device_program"),
            EnterLoop("i"),
            RunInstruction("T"),
            RunInstruction("S"),
            LeaveLoop("i"),
            ReturnFromKernel("device_program"),
        ),
    )
    t_unit = t_unit.with_kernel(kernel)

    with pytest.raises(
        LoopyError,
        match="schedule does not enforce 'T' after 'S'",
    ):
        verify_happens_after_is_enforced(t_unit)

    with pytest.raises(
        LoopyError,
        match="schedule does not enforce 'T' after 'S'",
    ):
        lp.generate_code_v2(t_unit)


def test_verification_handles_explicit_barrier_instruction() -> None:
    t_unit = lp.make_kernel(
        "{ : }",
        """
        a[0] = 1 {id=S}
        ... gbarrier {id=B}
        b[0] = a[0] {id=T}
        """,
        seq_dependencies=True,
    )
    t_unit = dep.add_lexicographic_happens_after(t_unit)
    t_unit = lp.preprocess_program(t_unit)
    t_unit = lp.linearize(t_unit)

    verify_happens_after_is_enforced(t_unit)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])

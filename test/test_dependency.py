import namedisl as nisl
import pytest

from pymbolic import var

import loopy as lp
import loopy.kernel.dependency as dep
from loopy.diagnostic import LoopyError
from loopy.kernel.instruction import HappensAfter

from loopy.schedule.verification import (
    _BarrierRecord,
    _PreciseSchedule,
    _StatementRecord,
    _build_enforced_order,
    _build_strict_lexicographic_order,
    _build_timestamp_relations,
    _get_timestamp_points_from_linearization,
    verify_happens_after_is_enforced,
)
from loopy.schedule import (
    Barrier,
    CallKernel,
    EnterLoop,
    LeaveLoop,
    ReturnFromKernel,
    RunInstruction,
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


def test_add_lexicographic_happens_after_orders_distinct_loop_nests() -> None:
    t_unit = lp.make_kernel(
        [
            "{ [i] : 0 <= i < N }",
            "{ [j] : 0 <= j < M }",
        ],
        """
        a[i] = 2 * a[i] {id=S}
        b[j] = 2 * b[j] {id=T}
        """,
    )

    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint
    cross_relation = kernel.id_to_insn["T"].happens_after["S"].instances_rel

    assert cross_relation is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    cross_relation = nisl.make_map(cross_relation)
    assert cross_relation.equals(
        nisl.make_map("""
        [N, M] -> {
            [j_after] -> [i_before] :
                0 <= j_after < M and
                0 <= i_before < N
        }
        """)
    )


def test_add_lexicographic_happens_after_with_five_inames() -> None:
    t_unit = lp.make_kernel(
        """
        { [q, z, a, m, b] :
            0 <= q < 2 and
            0 <= z < 2 and
            0 <= a < 2 and
            0 <= m < 2 and
            0 <= b < 2
        }
        """,
        "out[q, z, a, m, b] = q + z + a + m + b {id=S}",
    )

    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint
    self_relation = kernel.id_to_insn["S"].happens_after["S"].instances_rel

    assert self_relation is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    self_relation = nisl.make_map(self_relation)
    assert self_relation.equals(
        nisl.make_map("""
        {
            [q_after, z_after, a_after, m_after, b_after] ->
                    [q_before, z_before, a_before, m_before, b_before] :
                0 <= q_after, q_before < 2 and
                0 <= z_after, z_before < 2 and
                0 <= a_after, a_before < 2 and
                0 <= m_after, m_before < 2 and
                0 <= b_after, b_before < 2 and
                (q_before < q_after or
                 (q_before = q_after and z_before < z_after) or
                 (q_before = q_after and z_before = z_after and
                  a_before < a_after) or
                 (q_before = q_after and z_before = z_after and
                  a_before = a_after and m_before < m_after) or
                 (q_before = q_after and z_before = z_after and
                  a_before = a_after and m_before = m_after and
                  b_before < b_after))
        }
        """)
    )


def test_access_relation_finder_keeps_instruction_maps_separate() -> None:
    t_unit = lp.make_kernel(
        "{ [i] : 0 <= i < N }",
        """
        a[i] = 1 {id=S}
        b[i] = 2 {id=T}
        """,
    )

    kernel = t_unit.default_entrypoint
    rel_find = dep.AccessRelationFinder(kernel)
    rel_find(kernel.id_to_insn["S"].assignee, "S", dep.AccessType.write)
    rel_find(kernel.id_to_insn["T"].assignee, "T", dep.AccessType.write)

    assert rel_find.write_relations["S"].keys() == {"a"}
    assert rel_find.write_relations["T"].keys() == {"b"}


def test_access_relation_finder_distinguishes_reads_and_writes() -> None:
    t_unit = lp.make_kernel(
        "{ [i] : 1 <= i < N }",
        "a[i] = a[i - 1] {id=S}",
    )

    kernel = t_unit.default_entrypoint
    insn = kernel.id_to_insn["S"]
    rel_find = dep.AccessRelationFinder(kernel)
    rel_find(insn.assignee, insn.id, dep.AccessType.write)
    rel_find(insn.expression, insn.id, dep.AccessType.read)

    assert rel_find.read_relations["S"]["a"].equals(
        nisl.make_map("[N] -> { [i] -> [ax_0 = i - 1] : 1 <= i < N }")
    )
    assert rel_find.write_relations["S"]["a"].equals(
        nisl.make_map("[N] -> { [i] -> [ax_0 = i] : 1 <= i < N }")
    )


def test_access_relation_finder_handles_linear_subscript() -> None:
    t_unit = lp.make_kernel(
        "{ [i, j] : 0 <= i < NI and 0 <= j < NJ }",
        "out[i, j] = a[[2*i + j]] {id=S}",
    )

    kernel = t_unit.default_entrypoint
    insn = kernel.id_to_insn["S"]
    rel_find = dep.AccessRelationFinder(kernel)
    rel_find(insn.expression, insn.id, dep.AccessType.read)

    assert rel_find.read_relations["S"]["a"].equals(
        nisl.make_map("""
            [NI, NJ] -> {
                [i, j] -> [ax_0 = 2*i + j] :
                    0 <= i < NI and 0 <= j < NJ
            }
            """)
    )


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


def test_relax_strict_happens_after_finds_direct_raw() -> None:
    kernel = _relax_strict_happens_after(
        """
        a[i, j] = 1 {id=S}
        b[i, j] = a[i, j] {id=T}
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

    precise_kernel = dep.add_lexicographic_happens_after(
        legacy_kernel
    )
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


def test_relax_strict_happens_after_finds_direct_waw() -> None:
    kernel = _relax_strict_happens_after(
        """
        a[i, j] = 1 {id=S}
        a[i, j] = 2 {id=T}
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


def test_relax_strict_happens_after_finds_direct_war() -> None:
    kernel = _relax_strict_happens_after(
        """
        b[i, j] = a[i, j] {id=S}
        a[i, j] = 2 {id=T}
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


def test_relax_strict_happens_after_finds_self_raw() -> None:
    kernel = _relax_strict_happens_after(
        "a[i] = a[i - 1] {id=S}",
        "{ [i] : 1 <= i < N }",
    )

    required_order = kernel.id_to_insn["S"].happens_after["S"].instances_rel
    assert required_order is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    required_order = nisl.make_map(required_order)
    assert required_order.equals(
        nisl.make_map("""
            [N] -> {
                [i_after] -> [i_before = i_after - 1] : 2 <= i_after < N
            }
            """)
    )


def test_relax_strict_happens_after_drops_conflict_free_edge() -> None:
    kernel = _relax_strict_happens_after("""
        a[i] = 1 {id=S}
        b[i] = 2 {id=T}
        """)

    assert "S" not in kernel.id_to_insn["T"].happens_after


def test_relax_strict_happens_after_finds_recursive_raw() -> None:
    kernel = _relax_strict_happens_after(
        """
        a[i, j] = 1 {id=S}
        b[i, j] = 2 {id=T}
        c[i, j] = a[i, j] {id=U}
        """,
        "{ [i, j] : 0 <= i < NI and 0 <= j < NJ }",
    )

    required_order = kernel.id_to_insn["U"].happens_after["S"].instances_rel
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
    assert "T" not in kernel.id_to_insn["U"].happens_after


def test_relax_strict_happens_after_stops_at_most_recent_writer() -> None:
    kernel = _relax_strict_happens_after(
        """
        a[i, j] = 1 {id=S}
        a[i, j] = 2 {id=T}
        c[i, j] = a[i, j] {id=U}
        """,
        "{ [i, j] : 0 <= i < NI and 0 <= j < NJ }",
    )

    required_order = kernel.id_to_insn["U"].happens_after["T"].instances_rel
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
    assert "S" not in kernel.id_to_insn["U"].happens_after


def test_relax_strict_happens_after_partitions_partial_writer_footprint() -> (
    None
):
    kernel = _relax_strict_happens_after(
        """
        a[i, j] = 1 {id=S}
        a[2*i, j] = 2 {id=T}
        c[i, j] = a[i, j] {id=U}
        """,
        "{ [i, j] : 0 <= i < NI and 0 <= j < NJ }",
    )

    recent_order = kernel.id_to_insn["U"].happens_after["T"].instances_rel
    fallback_order = kernel.id_to_insn["U"].happens_after["S"].instances_rel
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


def test_relax_strict_happens_after_composes_distinct_loop_nests() -> None:
    t_unit = lp.make_kernel(
        [
            "{ [i] : 0 <= i < 4 }",
            "{ [j] : 0 <= j < 3 }",
            "{ [k] : 0 <= k < 4 }",
        ],
        """
        a[i] = 1 {id=S}
        b[j] = 2 {id=T}
        c[k] = a[k] {id=U}
        """,
    )
    t_unit = dep.add_lexicographic_happens_after(t_unit)
    kernel = dep.relax_strict_happens_after(t_unit).default_entrypoint

    required_order = kernel.id_to_insn["U"].happens_after["S"].instances_rel
    assert required_order is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    required_order = nisl.make_map(required_order)
    assert required_order.equals(
        nisl.make_map("""
            {
                [k_after] -> [i_before = k_after] : 0 <= k_after < 4
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


def test_relax_strict_happens_after_drops_empty_same_variable_edge() -> None:
    kernel = _relax_strict_happens_after("""
        a[i] = 1 {id=S}
        b[i] = a[i + N] {id=T}
        """)

    assert "S" not in kernel.id_to_insn["T"].happens_after


def test_relax_strict_happens_after_inner_uses_live_sink_accesses() -> None:
    t_unit = lp.make_kernel(
        "{ [i] : 0 <= i < N }",
        """
        a[i] = 1 {id=S}
        b[i] = a[i] {id=T}
        """,
    )
    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint

    rel_finder = dep.AccessRelationFinder(kernel)
    for insn in kernel.instructions:
        assert isinstance(insn, lp.MultiAssignmentBase)
        rel_finder(insn.assignee, insn.id, dep.AccessType.write)
        rel_finder(insn.expression, insn.id, dep.AccessType.read)

    incoming_relation = kernel.id_to_insn["T"].happens_after["S"].instances_rel
    assert incoming_relation is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    incoming_relation = nisl.make_map(incoming_relation)

    live_access_relation = rel_finder.read_relations["T"]["a"].rename_dims((
        ("i", "i_after"),
    ))
    live_access_relation = live_access_relation & nisl.make_map("""
        [N] -> {
            [i_after] -> [ax_0] : i_after = 0
        }
        """)

    happens_after = dep._relax_strict_happens_after_inner(
        kernel,
        "T",
        "S",
        "a",
        dep.AccessType.read,
        incoming_relation,
        live_access_relation,
        rel_finder,
        {},
    )

    required_order = happens_after["S"].instances_rel
    assert required_order is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    required_order = nisl.make_map(required_order)
    assert required_order.equals(
        nisl.make_map("""
            [N] -> {
                [i_after = 0] -> [i_before = 0] : N > 0
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
    kernel = lp.make_kernel(
        "{ [i, j, k] : 0 <= i, j, k < 8 }",
        """
        <> before = 0 {id=before}
        <> outer = i {id=outer}
        <> deep = i + j + k {id=deep}
        <> after_inner = i {id=after_inner}
        <> after = 0 {id=after}
        """,
    ).default_entrypoint.copy(
        state=lp.KernelState.LINEARIZED,
        linearization=(
            CallKernel("device_program"),
            RunInstruction("before"),
            EnterLoop("i"),
            RunInstruction("outer"),
            EnterLoop("j"),
            EnterLoop("k"),
            RunInstruction("deep"),
            LeaveLoop("k"),
            LeaveLoop("j"),
            RunInstruction("after_inner"),
            LeaveLoop("i"),
            RunInstruction("after"),
            ReturnFromKernel("device_program"),
        ),
    )

    assert _get_timestamp_points_from_linearization(kernel) == _PreciseSchedule(
        statements={
            "before": _StatementRecord(timestamp=(0, 1), subkernel_idx=0),
            "outer": _StatementRecord(timestamp=(0, 2, "i", 3), subkernel_idx=0),
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
    kernel = t_unit.default_entrypoint.copy(
        state=lp.KernelState.LINEARIZED,
        linearization=(
            CallKernel("device_program"),
            EnterLoop("i"),
            RunInstruction("S"),
            LeaveLoop("i"),
            ReturnFromKernel("device_program"),
        ),
    )

    precise_schedule = _get_timestamp_points_from_linearization(kernel)
    assert precise_schedule.statements["S"] == _StatementRecord(
        timestamp=(0, 1, "i", 2),
        subkernel_idx=0,
    )

    timestamp_relations, _, _ = _build_timestamp_relations(
        kernel, precise_schedule
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


def test_strict_lexicographic_timestamp_order() -> None:
    order = _build_strict_lexicographic_order(("t0", "t1", "t2"))

    assert order.equals(nisl.make_map("""
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
    """))


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
        kernel, precise_schedule
    )
    assert len(barrier_relations) == 2
    assert barrier_relations[0].equals(nisl.make_map("""
        {
            [batch, i] ->
                [__ts_0, __ts_1, __ts_2, __ts_3, __ts_4, __ts_5] :
                    0 <= batch < 8 and 0 <= i < 8 and
                    __ts_0 = 0 and __ts_1 = batch and
                    __ts_2 = 1 and __ts_3 = 2 and
                    __ts_4 = i and __ts_5 = 4
        }
    """))
    assert barrier_relations[1].equals(nisl.make_map("""
        {
            [batch] ->
                [__ts_0, __ts_1, __ts_2, __ts_3, __ts_4, __ts_5] :
                    0 <= batch < 8 and
                    __ts_0 = 0 and __ts_1 = batch and
                    __ts_2 = 8 and __ts_3 = 0 and
                    __ts_4 = 0 and __ts_5 = 0
        }
    """))


def test_local_barrier_orders_work_items_in_the_same_group() -> None:
    local_barrier = Barrier(
        comment="local synchronization",
        synchronization_kind="local",
        mem_kind="local",
        originating_insn_id=None,
    )
    t_unit = lp.make_kernel(
        """
        {
            [g, l, i] :
                0 <= g < 2 and 0 <= l < 4 and 0 <= i < 2
        }
        """,
        """
        a[g, l, i] = g + l + i {id=source}
        b[g, l, i] = a[g, l, i] {id=sink}
        """,
    )
    t_unit = lp.tag_inames(t_unit, {"g": "g.0", "l": "l.0"})
    kernel = t_unit.default_entrypoint.copy(
        state=lp.KernelState.LINEARIZED,
        linearization=(
            CallKernel("device_program"),
            EnterLoop("i"),
            RunInstruction("source"),
            local_barrier,
            RunInstruction("sink"),
            LeaveLoop("i"),
            ReturnFromKernel("device_program"),
        ),
    )

    precise_schedule = _get_timestamp_points_from_linearization(kernel)
    stmt_relations, barrier_relations, timestamp_order = (
        _build_timestamp_relations(kernel, precise_schedule)
    )
    enforced = _build_enforced_order(
        kernel,
        "sink",
        "source",
        precise_schedule,
        stmt_relations,
        barrier_relations,
        timestamp_order,
    )

    assert enforced.equals(nisl.make_map("""
        {
            [g_after, l_after, i_after] ->
                [g_before, l_before, i_before] :
                    0 <= g_after < 2 and 0 <= l_after < 4 and
                    0 <= i_after < 2 and
                    g_before = g_after and
                    0 <= l_before < 4 and
                    0 <= i_before <= i_after
        }
    """))


def test_global_barrier_orders_all_work_items() -> None:
    global_barrier = Barrier(
        comment="global synchronization",
        synchronization_kind="global",
        mem_kind="global",
        originating_insn_id=None,
    )
    t_unit = lp.make_kernel(
        "{ [g, l] : 0 <= g < 2 and 0 <= l < 4 }",
        """
        a[g, l] = g + l {id=source}
        b[g, l] = a[g, l] {id=sink}
        """,
    )
    t_unit = lp.tag_inames(t_unit, {"g": "g.0", "l": "l.0"})
    kernel = t_unit.default_entrypoint.copy(
        state=lp.KernelState.LINEARIZED,
        linearization=(
            CallKernel("phase0"),
            RunInstruction("source"),
            ReturnFromKernel("phase0"),
            global_barrier,
            CallKernel("phase1"),
            RunInstruction("sink"),
            ReturnFromKernel("phase1"),
        ),
    )

    precise_schedule = _get_timestamp_points_from_linearization(kernel)
    stmt_relations, barrier_relations, timestamp_order = (
        _build_timestamp_relations(kernel, precise_schedule)
    )
    enforced = _build_enforced_order(
        kernel,
        "sink",
        "source",
        precise_schedule,
        stmt_relations,
        barrier_relations,
        timestamp_order,
    )

    assert enforced.equals(nisl.make_map("""
        {
            [g_after, l_after] -> [g_before, l_before] :
                0 <= g_after < 2 and 0 <= l_after < 4 and
                0 <= g_before < 2 and 0 <= l_before < 4
        }
    """))


def test_verification() -> None:
    t_unit = lp.make_kernel(
        "{ [i] : 0 <= i < N }",
        """
        a[i] = 2 * a[i] {id=S}
        b[i] = 2 * b[i] {id=T}
        c[i] = a[i] + b[i] {id=U}
        """,
    )

    t_unit = dep.add_lexicographic_happens_after(t_unit)
    t_unit = dep.relax_strict_happens_after(t_unit)
    t_unit = lp.preprocess_program(t_unit)
    t_unit = lp.linearize(t_unit)
    t_unit = verify_happens_after_is_enforced(t_unit)
    lp.generate_code_v2(t_unit)


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
        "{ [i] : 0 <= i < N }",
        """
        a[i] = i {id=S}
        ... gbarrier {id=B}
        b[i] = a[i] {id=T}
        """,
    )
    t_unit = dep.add_lexicographic_happens_after(t_unit)

    kernel = t_unit.default_entrypoint
    barrier_insn = kernel.id_to_insn["B"]
    barrier = Barrier(
        comment="explicit global barrier",
        synchronization_kind=barrier_insn.synchronization_kind,
        mem_kind=barrier_insn.mem_kind,
        originating_insn_id="B",
    )
    kernel = kernel.copy(
        state=lp.KernelState.LINEARIZED,
        linearization=(
            CallKernel("phase0"),
            EnterLoop("i"),
            RunInstruction("S"),
            LeaveLoop("i"),
            ReturnFromKernel("phase0"),
            barrier,
            CallKernel("phase1"),
            EnterLoop("i"),
            RunInstruction("T"),
            LeaveLoop("i"),
            ReturnFromKernel("phase1"),
        ),
    )
    t_unit = t_unit.with_kernel(kernel)

    verify_happens_after_is_enforced(t_unit)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])

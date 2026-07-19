import namedisl as nisl

from pymbolic import var

import loopy as lp
import loopy.kernel.dependency as dep
from loopy.symbolic import SubArrayRef


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
    assert required_order.equals(
        nisl.make_map("""
            [NI, NJ] -> {
                [i_after, j_after] ->
                [i_before = i_after, j_before = j_after] :
                    0 <= i_after < NI and 0 <= j_after < NJ
            }
            """)
    )


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


def test_relax_strict_happens_after_partitions_partial_writer_footprint() -> None:
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
    kernel = kernel.copy(instructions=[
        insn.copy(happens_after={
            insn.id: insn.happens_after[insn.id],
            **dict.fromkeys(predecessors[insn.id], cross_order),
        })
        for insn in kernel.instructions
    ])

    kernel = dep.relax_strict_happens_after(kernel)

    required_order = kernel.id_to_insn["D"].happens_after["A"].instances_rel
    assert required_order is not None
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
    assert required_order.equals(
        nisl.make_map("""
            [N] -> {
                [i_after = 0] -> [i_before = 0] : N > 0
            }
            """)
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])

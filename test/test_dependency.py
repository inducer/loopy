import namedisl as nisl

import loopy as lp
import loopy.kernel.dependency as dep


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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])

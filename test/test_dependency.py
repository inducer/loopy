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

import islpy as isl
import pyopencl as cl
from pymbolic import var
from pyopencl.tools import (  # ruff:ignore[unused-import]
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

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


def _get_precise_relation(
    kernel: lp.LoopKernel, sink_id: str, source_id: str
) -> nisl.Map:
    relation = kernel.id_to_insn[sink_id].happens_after[source_id].instances_rel
    assert relation is not None
    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    return nisl.make_map(relation)


def test_affine_happens_after_transform_tiling() -> None:
    t_unit = lp.make_kernel(
        [
            "[N] -> { [i] : 0 <= i < N }",
            "[M] -> { [j] : 0 <= j < M }",
        ],
        """
        a[j] = 1 {id=A}
        b[i, j] = 2 {id=B}
        c[i, j] = 3 {id=C}
        d[j] = 4 {id=D}
        """,
        [
            lp.GlobalArg("a,d", shape=(var("M"),)),
            lp.GlobalArg("b,c", shape=(var("N"), var("M"))),
            "...",
        ],
    )
    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint
    a_self = _get_precise_relation(kernel, "A", "A")
    d_self = _get_precise_relation(kernel, "D", "D")

    kernel = dep.apply_affine_transform_to_happens_afters(
        kernel,
        nisl.make_map("""
            [N] -> {
                [i] -> [io, ii] :
                    i = 4*io + ii and 0 <= ii < 4
            }
            """),
    )

    assert _get_precise_relation(kernel, "B", "B").equals(
        nisl.make_map("""
        [N, M] -> {
            [io_after, ii_after, j_after] ->
            [io_before, ii_before, j_before] :
                0 <= 4*io_after + ii_after < N and
                0 <= ii_after < 4 and
                0 <= j_after < M and
                0 <= 4*io_before + ii_before < N and
                0 <= ii_before < 4 and
                0 <= j_before < M and
                (4*io_before + ii_before < 4*io_after + ii_after or
                 (4*io_before + ii_before = 4*io_after + ii_after and
                  j_before < j_after))
        }
        """)
    )
    assert _get_precise_relation(kernel, "B", "A").equals(
        nisl.make_map("""
        [N, M] -> {
            [io_after, ii_after, j_after] -> [j_before] :
                0 <= 4*io_after + ii_after < N and
                0 <= ii_after < 4 and
                0 <= j_before <= j_after < M
        }
        """)
    )
    assert _get_precise_relation(kernel, "C", "B").equals(
        nisl.make_map("""
        [N, M] -> {
            [io_after, ii_after, j_after] ->
            [io_before, ii_before, j_before] :
                0 <= 4*io_after + ii_after < N and
                0 <= ii_after < 4 and
                0 <= j_after < M and
                0 <= 4*io_before + ii_before < N and
                0 <= ii_before < 4 and
                0 <= j_before < M and
                (4*io_before + ii_before < 4*io_after + ii_after or
                 (4*io_before + ii_before = 4*io_after + ii_after and
                  j_before <= j_after))
        }
        """)
    )
    assert _get_precise_relation(kernel, "D", "C").equals(
        nisl.make_map("""
        [N, M] -> {
            [j_after] -> [io_before, ii_before, j_before] :
                0 <= 4*io_before + ii_before < N and
                0 <= ii_before < 4 and
                0 <= j_before <= j_after < M
        }
        """)
    )
    assert _get_precise_relation(kernel, "A", "A").equals(a_self)
    assert _get_precise_relation(kernel, "D", "D").equals(d_self)


def test_affine_happens_after_transform_skew() -> None:
    t_unit = lp.make_kernel(
        """
        [NI, NJ] -> {
            [i, j] : 0 <= i < NI and 0 <= j < NJ
        }
        """,
        "out[i, j] = i + j {id=S}",
    )
    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint
    kernel = dep.apply_affine_transform_to_happens_afters(
        kernel,
        nisl.make_map("""
            { [i, j] -> [is, js] : is = i and js = i + j }
            """),
    )

    assert _get_precise_relation(kernel, "S", "S").equals(
        nisl.make_map("""
        [NI, NJ] -> {
            [is_after, js_after] -> [is_before, js_before] :
                0 <= is_after < NI and
                0 <= js_after - is_after < NJ and
                0 <= is_before < NI and
                0 <= js_before - is_before < NJ and
                (is_before < is_after or
                 (is_before = is_after and
                  js_before - is_before < js_after - is_after))
        }
        """)
    )


def test_affine_happens_after_transform_permuted_axes() -> None:
    t_unit = lp.make_kernel(
        """
        [NI, NJ, NK] -> {
            [i, j, k] :
                0 <= i < NI and 0 <= j < NJ and 0 <= k < NK
        }
        """,
        "out[i, j, k] = i + j + k {id=S}",
    )
    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint
    kernel = dep.apply_affine_transform_to_happens_afters(
        kernel,
        nisl.make_map("""
            {
                [i, j, k] -> [j_new, i_new, k_new] :
                    j_new = j and i_new = i and k_new = k
            }
            """),
    )

    assert _get_precise_relation(kernel, "S", "S").equals(
        nisl.make_map("""
        [NI, NJ, NK] -> {
            [j_new_after, i_new_after, k_new_after] ->
            [j_new_before, i_new_before, k_new_before] :
                0 <= i_new_after < NI and
                0 <= j_new_after < NJ and
                0 <= k_new_after < NK and
                0 <= i_new_before < NI and
                0 <= j_new_before < NJ and
                0 <= k_new_before < NK and
                (i_new_before < i_new_after or
                 (i_new_before = i_new_after and
                  j_new_before < j_new_after) or
                 (i_new_before = i_new_after and
                  j_new_before = j_new_after and
                  k_new_before < k_new_after))
        }
        """)
    )


def test_affine_happens_after_transform_scalar_endpoints() -> None:
    t_unit = lp.make_kernel(
        "[N] -> { [i] : 0 <= i < N }",
        """
        a[0] = 1 {id=A}
        b[i] = 2 {id=B}
        c[0] = 3 {id=C}
        """,
        [
            lp.GlobalArg("a,c", shape=(1,)),
            lp.GlobalArg("b", shape=(var("N"),)),
            "...",
        ],
    )
    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint
    kernel = dep.apply_affine_transform_to_happens_afters(
        kernel,
        nisl.make_map("{ [i] -> [ip = i + 2] }"),
    )

    assert _get_precise_relation(kernel, "B", "A").equals(
        nisl.make_map("""
        [N] -> { [ip_after] -> [] : 2 <= ip_after < N + 2 }
        """)
    )
    assert _get_precise_relation(kernel, "C", "B").equals(
        nisl.make_map("""
        [N] -> { [] -> [ip_before] : 2 <= ip_before < N + 2 }
        """)
    )


def test_affine_happens_after_transform_rejects_partial_nest() -> None:
    t_unit = lp.make_kernel(
        [
            "[NI] -> { [i] : 0 <= i < NI }",
            "[NJ] -> { [j] : 0 <= j < NJ }",
        ],
        "out[i] = i {id=S}",
    )
    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint

    with pytest.raises(LoopyError):
        dep.apply_affine_transform_to_happens_afters(
            kernel,
            nisl.make_map("""
                {
                    [i, j] -> [i_new, j_new] :
                        i_new = i and j_new = j
                }
                """),
        )


def test_affine_happens_after_transform_avoids_proxy_name_collisions() -> None:
    t_unit = lp.make_kernel(
        "[NI, NJ] -> { [i, j] : 0 <= i < NI and 0 <= j < NJ }",
        "out[i, j] = i + j {id=S}",
    )
    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint
    kernel = dep.apply_affine_transform_to_happens_afters(
        kernel,
        nisl.make_map("{ [i] -> [j_new_] : j_new_ = i }"),
    )

    assert _get_precise_relation(kernel, "S", "S").equals(
        nisl.make_map("""
        [NI, NJ] -> {
            [j_new__after, j_after] -> [j_new__before, j_before] :
                0 <= j_new__after < NI and 0 <= j_after < NJ and
                0 <= j_new__before < NI and 0 <= j_before < NJ and
                (j_new__before < j_new__after or
                 (j_new__before = j_new__after and j_before < j_after))
        }
        """)
    )


def test_map_domain_transforms_precise_happens_after() -> None:
    t_unit = lp.make_kernel(
        "[NI, NJ] -> { [i, j] : 0 <= i < NI and 0 <= j < NJ }",
        """
        a[i, j] = i + j {id=A}
        b[i, j] = a[i, j] {id=B}
        """,
    )
    t_unit = dep.add_lexicographic_happens_after(t_unit)
    transform_map = isl.BasicMap("""
        [NI] -> {
            [i] -> [io, ii] :
                i = 4*io + ii and 0 <= ii < 4
        }
        """)

    expected = dep.apply_affine_transform_to_happens_afters(
        t_unit.default_entrypoint,
        nisl.make_map(transform_map.to_map()),
    )
    kernel = lp.map_domain(t_unit, transform_map).default_entrypoint

    assert kernel.id_to_insn["A"].within_inames == frozenset({"io", "ii", "j"})
    assert kernel.id_to_insn["B"].within_inames == frozenset({"io", "ii", "j"})
    for sink_id, source_id in [("A", "A"), ("B", "A"), ("B", "B")]:
        assert _get_precise_relation(kernel, sink_id, source_id).equals(
            _get_precise_relation(expected, sink_id, source_id)
        )


def test_split_iname_transforms_precise_happens_after() -> None:
    t_unit = lp.make_kernel(
        "[NI, NJ] -> { [i, j] : 0 <= i < NI and 0 <= j < NJ }",
        """
        a[i, j] = i + j {id=A}
        b[i, j] = a[i, j] {id=B}
        """,
    )
    t_unit = dep.add_lexicographic_happens_after(t_unit)
    split_reln = nisl.make_map("""
        { [i] -> [io, ii] :
            i = 4*io + ii and 0 <= ii < 4
        }
        """)

    expected = dep.apply_affine_transform_to_happens_afters(
        t_unit.default_entrypoint, split_reln
    )
    kernel = lp.split_iname(
        t_unit, "i", 4, outer_iname="io", inner_iname="ii"
    ).default_entrypoint

    assert kernel.id_to_insn["A"].within_inames == frozenset({"io", "ii", "j"})
    assert kernel.id_to_insn["B"].within_inames == frozenset({"io", "ii", "j"})
    for sink_id, source_id in [("A", "A"), ("B", "A"), ("B", "B")]:
        assert _get_precise_relation(kernel, sink_id, source_id).equals(
            _get_precise_relation(expected, sink_id, source_id)
        )


def test_split_iname_rejects_within_for_precise_happens_after() -> None:
    t_unit = lp.make_kernel(
        "{ [i] : 0 <= i < 16 }",
        """
        a[i] = i {id=A}
        b[i] = a[i] {id=B}
        """,
    )
    t_unit = dep.add_lexicographic_happens_after(t_unit)

    with pytest.raises(LoopyError, match="does not support 'within'"):
        lp.split_iname(t_unit, "i", 4, within="id:A")


def test_splice_happens_after_as_consumer_inherits_branched_order() -> None:
    t_unit = lp.make_kernel(
        [
            "[NI, NJ] -> { [i, j] : 0 <= i < NI and 0 <= j < NJ }",
            """
            [NI, NJ] -> {
                [tile, lane, jc] :
                    0 <= 2*tile + lane < NI and
                    0 <= lane < 2 and 0 <= jc < NJ
            }
            """,
        ],
        """
        p[i, j] = i + j {id=P}
        q[i] = i {id=Q}
        a[i, j] = p[i, j] + q[i] {id=A}
        c[tile, lane, jc] = 0 {id=C}
        s[i, j] = a[i, j] {id=S}
        """,
    )
    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint

    a_after_p = HappensAfter(nisl.make_map("""
        [NI, NJ] -> {
            [i_after, j_after] -> [i_before, j_before] :
                i_before = i_after and j_before = j_after and
                0 <= i_after < NI and 0 <= j_after < NJ
        }
        """).as_isl())
    a_after_q = HappensAfter(nisl.make_map("""
        [NI, NJ] -> {
            [i_after, j_after] -> [i_before] :
                i_before = i_after and
                0 <= i_after < NI and 0 <= j_after < NJ
        }
        """).as_isl())
    s_after_a = HappensAfter(nisl.make_map("""
        [NI, NJ] -> {
            [i_after, j_after] -> [i_before, j_before] :
                i_before = i_after and j_before = j_after and
                0 <= i_after < NI and 0 <= j_after < NJ
        }
        """).as_isl())

    new_happens_after = {
        "P": {"P": kernel.id_to_insn["P"].happens_after["P"]},
        "Q": {"Q": kernel.id_to_insn["Q"].happens_after["Q"]},
        "A": {
            "A": kernel.id_to_insn["A"].happens_after["A"],
            "P": a_after_p,
            "Q": a_after_q,
        },
        "C": {"C": kernel.id_to_insn["C"].happens_after["C"]},
        "S": {
            "S": kernel.id_to_insn["S"].happens_after["S"],
            "A": s_after_a,
        },
    }
    kernel = kernel.copy(instructions=tuple(
        stmt.copy(happens_after=new_happens_after[stmt.id])
        for stmt in kernel.instructions
    ))
    original_anchor_order = kernel.id_to_insn["A"].happens_after
    original_successor_order = kernel.id_to_insn["S"].happens_after
    original_consumer_self_order = kernel.id_to_insn["C"].happens_after["C"]

    kernel = dep.splice_happens_after_as_consumer(
        kernel,
        "C",
        "A",
        nisl.make_map("""
        [NI, NJ] -> {
            [tile_after, lane_after, jc_after] -> [i_before, j_before] :
                i_before = 2*tile_after + lane_after and
                j_before = jc_after and
                0 <= 2*tile_after + lane_after < NI and
                0 <= lane_after < 2 and 0 <= jc_after < NJ
        }
        """),
    )

    assert kernel.id_to_insn["C"].happens_after.keys() == {"C", "P", "Q"}
    assert kernel.id_to_insn["C"].happens_after["C"] == (
        original_consumer_self_order
    )
    assert kernel.id_to_insn["A"].happens_after == original_anchor_order
    assert kernel.id_to_insn["S"].happens_after == original_successor_order
    assert _get_precise_relation(kernel, "C", "P").equals(
        nisl.make_map("""
        [NI, NJ] -> {
            [tile_after, lane_after, jc_after] -> [i_before, j_before] :
                i_before = 2*tile_after + lane_after and
                j_before = jc_after and
                0 <= 2*tile_after + lane_after < NI and
                0 <= lane_after < 2 and 0 <= jc_after < NJ
        }
        """)
    )
    assert _get_precise_relation(kernel, "C", "Q").equals(
        nisl.make_map("""
        [NI, NJ] -> {
            [tile_after, lane_after, jc_after] -> [i_before] :
                i_before = 2*tile_after + lane_after and
                0 <= 2*tile_after + lane_after < NI and
                0 <= lane_after < 2 and 0 <= jc_after < NJ
        }
        """)
    )


def test_splice_happens_after_as_consumer_unions_existing_order() -> None:
    t_unit = lp.make_kernel(
        [
            "[N] -> { [i] : 0 <= i < N }",
            "[N] -> { [ic, lane] : 0 <= ic < N and 0 <= lane < 4 }",
        ],
        """
        p[i] = i {id=P}
        a[i] = p[i] {id=A}
        c[ic, lane] = 0 {id=C}
        """,
    )
    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint
    a_after_p = HappensAfter(nisl.make_map("""
        [N] -> {
            [i_after] -> [i_before = i_after] : 0 <= i_after < N
        }
        """).as_isl())
    c_after_p = HappensAfter(nisl.make_map("""
        [N] -> {
            [ic_after, lane_after] -> [i_before = ic_after] :
                0 <= ic_after < N and lane_after = 0
        }
        """).as_isl())
    new_happens_after = {
        "P": {"P": kernel.id_to_insn["P"].happens_after["P"]},
        "A": {
            "A": kernel.id_to_insn["A"].happens_after["A"],
            "P": a_after_p,
        },
        "C": {
            "C": kernel.id_to_insn["C"].happens_after["C"],
            "P": c_after_p,
        },
    }
    kernel = kernel.copy(instructions=tuple(
        stmt.copy(happens_after=new_happens_after[stmt.id])
        for stmt in kernel.instructions
    ))

    kernel = dep.splice_happens_after_as_consumer(
        kernel,
        "C",
        "A",
        nisl.make_map("""
        [N] -> {
            [ic_after, lane_after] -> [i_before = ic_after] :
                0 <= ic_after < N and 1 <= lane_after < 4
        }
        """),
    )

    assert _get_precise_relation(kernel, "C", "P").equals(
        nisl.make_map("""
        [N] -> {
            [ic_after, lane_after] -> [i_before = ic_after] :
                0 <= ic_after < N and 0 <= lane_after < 4
        }
        """)
    )


def test_splice_happens_after_as_consumer_handles_scalar_anchor() -> None:
    t_unit = lp.make_kernel(
        "[N] -> { [i] : 0 <= i < N }",
        """
        p[0] = 1 {id=P}
        a[0] = p[0] {id=A}
        c[i] = 0 {id=C}
        """,
    )
    kernel = dep.add_lexicographic_happens_after(t_unit).default_entrypoint
    kernel = kernel.copy(instructions=tuple(
        stmt.copy(happens_after={
            "P": {"P": kernel.id_to_insn["P"].happens_after["P"]},
            "A": {
                "A": kernel.id_to_insn["A"].happens_after["A"],
                "P": HappensAfter(nisl.make_map("{ [] -> [] }").as_isl()),
            },
            "C": {"C": kernel.id_to_insn["C"].happens_after["C"]},
        }[stmt.id])
        for stmt in kernel.instructions
    ))

    kernel = dep.splice_happens_after_as_consumer(
        kernel,
        "C",
        "A",
        nisl.make_map("[N] -> { [i_after] -> [] : 0 <= i_after < N }"),
    )

    assert _get_precise_relation(kernel, "C", "P").equals(
        nisl.make_map("[N] -> { [i_after] -> [] : 0 <= i_after < N }")
    )


def test_splice_happens_after_as_producer_redirects_branched_order() -> None:
    t_unit = lp.make_kernel(
        [
            "[NI, NJ] -> { [i, j] : 0 <= i < NI and 0 <= j < NJ }",
            """
            [NI, NJ] -> {
                [tile, lane, jp] :
                    0 <= 2*tile + lane < NI and
                    0 <= lane < 2 and 0 <= jp < NJ
            }
            """,
        ],
        """
        q[i, j] = i + j {id=Q}
        a[i, j] = q[i, j] {id=A}
        g[tile, lane, jp] = 0 {id=G}
        s[i, j] = a[i, j] + q[i, j] {id=S}
        t[i] = sum(j, a[i, j]) {id=T}
        """,
    )
    kernel = t_unit.default_entrypoint

    a_after_q = HappensAfter(nisl.make_map("""
        [NI, NJ] -> {
            [i_after, j_after] -> [i_before, j_before] :
                i_before = i_after and j_before = j_after and
                0 <= i_after < NI and 0 <= j_after < NJ
        }
        """).as_isl())
    g_after_q = HappensAfter(nisl.make_map("""
        [NI, NJ] -> {
            [tile_after, lane_after, jp_after] -> [i_before, j_before] :
                i_before = 2*tile_after + lane_after and
                j_before = jp_after and
                0 <= 2*tile_after + lane_after < NI and
                0 <= lane_after < 2 and 0 <= jp_after < NJ
        }
        """).as_isl())
    s_after_a = HappensAfter(nisl.make_map("""
        [NI, NJ] -> {
            [i_after, j_after] -> [i_before, j_before] :
                i_before = i_after and j_before = j_after and
                0 <= i_after < NI and 0 <= j_after < NJ
        }
        """).as_isl())
    s_after_q = HappensAfter(nisl.make_map("""
        [NI, NJ] -> {
            [i_after, j_after] -> [i_before, j_before] :
                i_before = i_after and j_before = j_after and
                0 <= i_after < NI and 0 <= j_after < NJ
        }
        """).as_isl())
    t_after_a = HappensAfter(nisl.make_map("""
        [NI, NJ] -> {
            [i_after] -> [i_before, j_before] :
                i_before = i_after and
                0 <= i_after < NI and 0 <= j_before < NJ
        }
        """).as_isl())
    happens_after = {
        "Q": {},
        "A": {"Q": a_after_q},
        "G": {"Q": g_after_q},
        "S": {"A": s_after_a, "Q": s_after_q},
        "T": {"A": t_after_a},
    }
    kernel = kernel.copy(instructions=tuple(
        stmt.copy(happens_after=happens_after[stmt.id])
        for stmt in kernel.instructions
    ))
    original_anchor_order = kernel.id_to_insn["A"].happens_after
    original_producer_order = kernel.id_to_insn["G"].happens_after
    original_s_after_q = kernel.id_to_insn["S"].happens_after["Q"]

    kernel = dep.splice_happens_after_as_producer(
        kernel,
        "G",
        "A",
        nisl.make_map("""
        [NI, NJ] -> {
            [i_after, j_after] -> [tile_before, lane_before, jp_before] :
                i_after = 2*tile_before + lane_before and
                j_after = jp_before and
                0 <= i_after < NI and 0 <= j_after < NJ and
                0 <= lane_before < 2
        }
        """),
    )

    assert kernel.id_to_insn["A"].happens_after == original_anchor_order
    assert kernel.id_to_insn["G"].happens_after == original_producer_order
    assert kernel.id_to_insn["S"].happens_after.keys() == {"G", "Q"}
    assert kernel.id_to_insn["S"].happens_after["Q"] == original_s_after_q
    assert kernel.id_to_insn["T"].happens_after.keys() == {"G"}
    assert _get_precise_relation(kernel, "S", "G").equals(
        nisl.make_map("""
        [NI, NJ] -> {
            [i_after, j_after] -> [tile_before, lane_before, jp_before] :
                i_after = 2*tile_before + lane_before and
                j_after = jp_before and
                0 <= i_after < NI and 0 <= j_after < NJ and
                0 <= lane_before < 2
        }
        """)
    )
    assert _get_precise_relation(kernel, "T", "G").equals(
        nisl.make_map("""
        [NI, NJ] -> {
            [i_after] -> [tile_before, lane_before, jp_before] :
                i_after = 2*tile_before + lane_before and
                0 <= i_after < NI and 0 <= lane_before < 2 and
                0 <= jp_before < NJ
        }
        """)
    )


def test_splice_happens_after_as_producer_preserves_unmapped_order() -> None:
    t_unit = lp.make_kernel(
        [
            "[N] -> { [i] : 0 <= i < N }",
            "[N] -> { [ip] : 0 <= ip < N }",
        ],
        """
        a[i] = i {id=A}
        g[ip] = 0 {id=G}
        s[i] = a[i] {id=S}
        """,
    )
    kernel = t_unit.default_entrypoint
    s_after_a = HappensAfter(nisl.make_map("""
        [N] -> {
            [i_after] -> [i_before = i_after] : 0 <= i_after < N
        }
        """).as_isl())
    s_after_g = HappensAfter(nisl.make_map("""
        [N] -> {
            [i_after] -> [ip_before = i_after] :
                0 <= i_after < N and 2*i_after >= N
        }
        """).as_isl())
    kernel = kernel.copy(instructions=tuple(
        stmt.copy(happens_after={
            "A": {},
            "G": {},
            "S": {"A": s_after_a, "G": s_after_g},
        }[stmt.id])
        for stmt in kernel.instructions
    ))

    kernel = dep.splice_happens_after_as_producer(
        kernel,
        "G",
        "A",
        nisl.make_map("""
        [N] -> {
            [i_after] -> [ip_before = 0] :
                0 <= i_after < N and 2*i_after < N
        }
        """),
    )

    assert _get_precise_relation(kernel, "S", "A").equals(
        nisl.make_map("""
        [N] -> {
            [i_after] -> [i_before = i_after] :
                0 <= i_after < N and 2*i_after >= N
        }
        """)
    )
    assert _get_precise_relation(kernel, "S", "G").equals(
        nisl.make_map("""
        [N] -> {
            [i_after] -> [ip_before = 0] :
                0 <= i_after < N and 2*i_after < N;
            [i_after] -> [ip_before = i_after] :
                0 <= i_after < N and 2*i_after >= N
        }
        """)
    )


def test_splice_happens_after_as_producer_handles_scalar_producer() -> None:
    t_unit = lp.make_kernel(
        "[N] -> { [i] : 0 <= i < N }",
        """
        a[i] = i {id=A}
        g[0] = 0 {id=G}
        s[i] = a[i] {id=S}
        """,
    )
    kernel = t_unit.default_entrypoint
    kernel = kernel.copy(instructions=tuple(
        stmt.copy(happens_after={
            "A": {},
            "G": {},
            "S": {
                "A": HappensAfter(nisl.make_map("""
                    [N] -> {
                        [i_after] -> [i_before = i_after] :
                            0 <= i_after < N
                    }
                    """).as_isl()),
            },
        }[stmt.id])
        for stmt in kernel.instructions
    ))

    kernel = dep.splice_happens_after_as_producer(
        kernel,
        "G",
        "A",
        nisl.make_map("[N] -> { [i_after] -> [] : 0 <= i_after < N }"),
    )

    assert kernel.id_to_insn["S"].happens_after.keys() == {"G"}
    assert _get_precise_relation(kernel, "S", "G").equals(
        nisl.make_map("[N] -> { [i_after] -> [] : 0 <= i_after < N }")
    )


def test_splice_happens_after_as_consumer_and_producer() -> None:
    t_unit = lp.make_kernel(
        [
            "[NI, NJ] -> { [i, j] : 0 <= i < NI and 0 <= j < NJ }",
            """
            [NI, NJ] -> {
                [tile, lane, jg] :
                    0 <= 2*tile + lane < NI and
                    0 <= lane < 2 and 0 <= jg < NJ
            }
            """,
        ],
        """
        d[i, j] = i + j {id=D}
        a[i, j] = d[i, j] {id=A}
        g[tile, lane, jg] = 0 {id=G}
        s[i, j] = a[i, j] {id=S}
        """,
    )
    kernel = t_unit.default_entrypoint
    same_instance = nisl.make_map("""
        [NI, NJ] -> {
            [i_after, j_after] -> [i_before, j_before] :
                i_before = i_after and j_before = j_after and
                0 <= i_after < NI and 0 <= j_after < NJ
        }
        """)
    kernel = kernel.copy(instructions=tuple(
        stmt.copy(happens_after={
            "D": {},
            "A": {"D": HappensAfter(same_instance.as_isl())},
            "G": {},
            "S": {"A": HappensAfter(same_instance.as_isl())},
        }[stmt.id])
        for stmt in kernel.instructions
    ))

    instruction_to_anchor = nisl.make_map("""
        [NI, NJ] -> {
            [tile_after, lane_after, jg_after] -> [i_before, j_before] :
                i_before = 2*tile_after + lane_after and
                j_before = jg_after and
                0 <= 2*tile_after + lane_after < NI and
                0 <= lane_after < 2 and 0 <= jg_after < NJ
        }
        """)
    anchor_to_instruction = nisl.make_map("""
        [NI, NJ] -> {
            [i_after, j_after] -> [tile_before, lane_before, jg_before] :
                i_after = 2*tile_before + lane_before and
                j_after = jg_before and
                0 <= i_after < NI and 0 <= j_after < NJ and
                0 <= lane_before < 2
        }
        """)
    kernel = dep.splice_happens_after_as_consumer_and_producer(
        kernel,
        "G",
        "A",
        instruction_to_anchor,
        anchor_to_instruction,
    )

    assert kernel.id_to_insn["A"].happens_after.keys() == {"D"}
    assert kernel.id_to_insn["G"].happens_after.keys() == {"D"}
    assert kernel.id_to_insn["S"].happens_after.keys() == {"G"}
    assert _get_precise_relation(kernel, "G", "D").equals(
        instruction_to_anchor
    )
    assert _get_precise_relation(kernel, "S", "G").equals(
        anchor_to_instruction
    )


def test_combined_splice_accepts_independent_partial_maps() -> None:
    t_unit = lp.make_kernel(
        [
            "[N] -> { [i] : 0 <= i < N }",
            "[N] -> { [b] : 0 <= b < N }",
        ],
        """
        d[i] = i {id=D}
        a[i] = d[i] {id=A}
        g[b] = 0 {id=G}
        s[i] = a[i] {id=S}
        """,
    )
    kernel = t_unit.default_entrypoint
    same_instance = HappensAfter(nisl.make_map("""
        [N] -> {
            [i_after] -> [i_before = i_after] : 0 <= i_after < N
        }
        """).as_isl())
    kernel = kernel.copy(instructions=tuple(
        stmt.copy(happens_after={
            "D": {},
            "A": {"D": same_instance},
            "G": {},
            "S": {"A": same_instance},
        }[stmt.id])
        for stmt in kernel.instructions
    ))

    kernel = dep.splice_happens_after_as_consumer_and_producer(
        kernel,
        "G",
        "A",
        nisl.make_map("""
        [N] -> {
            [b_after] -> [i_before = 2*b_after] :
                0 <= 2*b_after < N
        }
        """),
        nisl.make_map("""
        [N] -> {
            [i_after] -> [b_before = i_after] :
                0 <= i_after < N and 2*i_after >= N
        }
        """),
    )

    assert _get_precise_relation(kernel, "G", "D").equals(
        nisl.make_map("""
        [N] -> {
            [b_after] -> [i_before = 2*b_after] :
                0 <= 2*b_after < N
        }
        """)
    )
    assert _get_precise_relation(kernel, "S", "A").equals(
        nisl.make_map("""
        [N] -> {
            [i_after] -> [i_before = i_after] :
                0 <= i_after < N and 2*i_after < N
        }
        """)
    )
    assert _get_precise_relation(kernel, "S", "G").equals(
        nisl.make_map("""
        [N] -> {
            [i_after] -> [b_before = i_after] :
                0 <= i_after < N and 2*i_after >= N
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


def test_relax_strict_happens_after_expands_substitution_rules() -> None:
    t_unit = lp.make_kernel(
        "{ [i] : 0 <= i < N }",
        """
        a[2*i + 3] = i {id=S}
        out[i] = outer(i) {id=T}
        """,
        substitutions={
            "inner": lp.SubstitutionRule(
                "inner", ("j",), var("a")[2 * var("j") + 1]
            ),
            "outer": lp.SubstitutionRule(
                "outer", ("k",), var("inner")(var("k") + 1)
            ),
        },
    )
    t_unit = dep.add_lexicographic_happens_after(t_unit)
    kernel = dep.relax_strict_happens_after(t_unit).default_entrypoint

    required_order = kernel.id_to_insn["T"].happens_after["S"].instances_rel
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
        _build_timestamp_relations(kernel, precise_schedule, name_generator)
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
        _build_timestamp_relations(kernel, precise_schedule, name_generator)
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


def test_numerical_affine_and_splicing_integration(
    ctx_factory: cl.CtxFactory,
) -> None:
    args = [
        lp.GlobalArg("x,out", dtype=np.int32, shape=(64,)),
        "...",
    ]
    ref_t_unit = lp.make_kernel(
        "{ [i] : 0 <= i < 64 }",
        "out[i] = 2*(x[i] + 1) + 3",
        args,
    )
    t_unit = lp.make_kernel(
        "{ [i] : 0 <= i < 64 }",
        """
        out[i] = g[i] + 3 {id=S}
        <> g[i] = 2*d[i] {id=G}
        <> a[i] = 2*d[i] {id=A}
        <> d[i] = x[i] + 1 {id=D}
        """,
        args,
    )
    same_instance = HappensAfter(nisl.make_map("""
        {
            [i_after] -> [i_before = i_after] : 0 <= i_after < 64
        }
        """).as_isl())
    kernel = t_unit.default_entrypoint
    kernel = kernel.copy(instructions=tuple(
        stmt.copy(happens_after={
            "S": {"A": same_instance},
            "G": {},
            "A": {"D": same_instance},
            "D": {},
        }[stmt.id])
        for stmt in kernel.instructions
    ))
    t_unit = t_unit.with_kernel(kernel)

    t_unit = lp.split_iname(
        t_unit,
        "i",
        4,
        outer_iname="io",
        inner_iname="ii",
    )
    same_tiled_instance = nisl.make_map("""
        {
            [io_after, ii_after] -> [io_before, ii_before] :
                io_before = io_after and ii_before = ii_after and
                0 <= 4*io_after + ii_after < 64 and 0 <= ii_after < 4
        }
        """)
    kernel = dep.splice_happens_after_as_consumer_and_producer(
        t_unit.default_entrypoint,
        "G",
        "A",
        same_tiled_instance,
        same_tiled_instance,
    )
    t_unit = t_unit.with_kernel(kernel)
    t_unit = lp.prioritize_loops(t_unit, "io,ii")

    lp.auto_test_vs_ref(
        ref_t_unit,
        ctx_factory(),
        t_unit,
        print_code=False,
        quiet=True,
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

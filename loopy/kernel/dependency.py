from __future__ import annotations


__copyright__ = "Copyright (C) 2025 Addison Alvey-Blanco"

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

import islpy as isl
from islpy import dim_type

from typing import Mapping

from loopy import HappensAfter, LoopKernel, for_each_kernel
from loopy.kernel.instruction import (
    InstructionBase,
    VariableSpecificHappensAfter,
)
from loopy.transform.dependency import AccessMapFinder


def _add_lexicographic_happens_after_inner(knl, after_insn, before_insn):
    domain_before = knl.get_inames_domain(before_insn.within_inames)
    domain_after = knl.get_inames_domain(after_insn.within_inames)

    happens_after = isl.Map.from_domain_and_range(domain_after,
                                                  domain_before)
    for idim in range(happens_after.dim(dim_type.out)):
        happens_after = happens_after.set_dim_name(
            dim_type.out,
            idim,
            happens_after.get_dim_name(dim_type.out, idim) + "'"
        )

    shared_inames = before_insn.within_inames & after_insn.within_inames

    # {{{ use whatever iname ordering exists at this point

    shared_inames_order_before = [
        domain_before.get_dim_name(dim_type.out, idim)
        for idim in range(domain_before.dim(dim_type.out))
        if domain_before.get_dim_name(dim_type.out, idim)
        in shared_inames
    ]

    shared_inames_order_after = [
        domain_after.get_dim_name(dim_type.out, idim)
        for idim in range(domain_after.dim(dim_type.out))
        if domain_after.get_dim_name(dim_type.out, idim)
        in shared_inames
    ]

    assert shared_inames_order_after == shared_inames_order_before
    shared_inames_order = shared_inames_order_after

    # }}}

    affs_in = isl.affs_from_space(happens_after.domain().space)
    affs_out = isl.affs_from_space(happens_after.range().space)

    lex_map = isl.Map.empty(happens_after.space)
    for iinnermost, innermost_iname in enumerate(shared_inames_order):
        innermost_map = affs_in[innermost_iname].gt_map(
            affs_out[innermost_iname + "'"]
        )

        for outer_iname in list(shared_inames_order)[:iinnermost]:
            innermost_map = innermost_map & (
                affs_in[outer_iname].eq_map(
                    affs_out[outer_iname + "'"]
                )
            )

        lex_map = lex_map | innermost_map

    return happens_after & lex_map


@for_each_kernel
def add_lexicographic_happens_after(knl: LoopKernel) -> LoopKernel:
    """
    Impose a sequential, top-down execution order to instructions in a program.
    It is expected that this strict order will be relaxed with
    :func:`reduce_strict_ordering_with_dependencies` using data dependencies.
    """

    rmap = knl.reader_map()
    wmap_r: dict[str, set[str]] = {}
    for var, insns in knl.writer_map().items():
        for insn in insns:
            wmap_r.setdefault(insn, set())
            wmap_r[insn].add(var)

    new_insns = []
    for iafter, after_insn in enumerate(knl.instructions):
        assert after_insn.id is not None

        new_happens_after = {}

        # check for self dependencies
        for var in wmap_r[after_insn.id]:
            if rmap.get(var) and after_insn.id in rmap[var]:
                self_happens_after = _add_lexicographic_happens_after_inner(
                    knl, after_insn, after_insn
                )
                new_happens_after[after_insn.id] = HappensAfter(
                    self_happens_after
                )

        # add happens after relation with previous instruction
        if iafter != 0:
            before_insn = knl.instructions[iafter - 1]

            happens_after = _add_lexicographic_happens_after_inner(
                knl, after_insn, before_insn
            )

            new_happens_after[before_insn.id] = HappensAfter(happens_after)

        new_insns.append(after_insn.copy(happens_after=new_happens_after))

    return knl.copy(instructions=new_insns)


@for_each_kernel
def reduce_strict_ordering(knl) -> LoopKernel:
    def narrow_dependencies(
            after: InstructionBase,
            before: InstructionBase,
            happens_afters: Mapping,
            remaining: isl.Map | None = None,  # type: ignore
        ) -> Mapping:
        assert isinstance(after.id, str)
        assert isinstance(before.id, str)

        if remaining is not None and remaining.is_empty():
            return happens_afters

        new_happens_after: dict[str, VariableSpecificHappensAfter] = {}
        for insn, happens_after in before.happens_after.items():
            if remaining is None:
                remaining = happens_after.instances_rel
            else:
                assert happens_after.instances_rel is not None
                if remaining.space != happens_after.instances_rel.space:
                    remaining = remaining.apply_range(happens_after.instances_rel)

            source_vars = access_mapper.get_accessed_variables(after.id)
            common_vars = wmap_r[insn] & source_vars  # type: ignore
            for var in common_vars:
                write_map = access_mapper.get_map(insn, var)
                source_map = access_mapper.get_map(after.id, var)
                assert write_map is not None
                assert source_map is not None

                source_to_writer = source_map.apply_range(write_map.reverse())
                dependency_map = source_to_writer & remaining
                remaining = remaining - dependency_map
                if dependency_map is not None and not dependency_map.is_empty():
                    new_happens_after[insn] = VariableSpecificHappensAfter(
                        instances_rel=dependency_map, variable_name=var
                    )
                    happens_afters.update(new_happens_after)

            if insn != after.id:
                happens_afters.update(
                    narrow_dependencies(
                        after,
                        knl.id_to_insn[insn],
                        happens_afters,
                        remaining,
                    )
                )

        return happens_afters

    access_mapper = AccessMapFinder(knl)
    for insn in knl.instructions:
        access_mapper(insn.expression, insn.id)
        access_mapper(insn.assignee, insn.id)

    wmap_r: dict[str, set[str]] = {}
    for var, insns in knl.writer_map().items():
        for insn in insns:
            wmap_r.setdefault(insn, set())
            wmap_r[insn].add(var)

    new_insns = []
    for insn in knl.instructions[::-1]:
        new_insns.append(
            insn.copy(happens_after=narrow_dependencies(insn, insn, {}))
        )

    return knl.copy(instructions=new_insns[::-1])

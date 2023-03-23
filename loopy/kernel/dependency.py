__copyright__ = "Copyright (C) 2023 Addison Alvey-Blanco"

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

from typing import Mapping

import islpy as isl
from islpy import dim_type

from loopy import LoopKernel
from loopy.kernel.instruction import HappensAfter
from loopy.symbolic import ArrayAccessFinder
from loopy.translation_unit import for_each_kernel
from loopy.symbolic import get_access_map

import operator
from functools import reduce


@for_each_kernel
def narrow_dependencies(knl: LoopKernel) -> LoopKernel:
    """Compute statement-instance-level data dependencies between statements in
    a program. Relies on an existing lexicographic ordering, i.e. computed by
    add_lexicographic_happens_after.

    In particular, this function computes three dependency relations:
        1. The flow dependency relation, aka read-after-write
        2. The anti-dependecny relation, aka write-after-reads
        3. The output dependency relation, aka write-after-write

    The full dependency relation between a two statements in a program is the
    union of these three dependency relations.
    """

    writer_map = knl.writer_map()
    reader_map = knl.reader_map()

    reads = {
            insn.id: insn.read_dependency_names() - insn.within_inames
            for insn in knl.instructions
    }

    writes = {
            insn.id: insn.write_dependency_names() - insn.within_inames
            for insn in knl.instructions
    }

    def get_unordered_deps(r: isl.Map, s: isl.Map) -> isl.Map:
        """Equivalent to computing the relation R^{-1} o S
        """
        return s.apply_range(r.reverse())

    def make_inames_unique(relation: isl.Map) -> isl.Map:
        """Make the inames of a particular map unique by adding a single quote
        to each iname in the range of the map
        """
        ndim = relation.dim(dim_type.out)
        for i in range(ndim):
            iname = relation.get_dim_name(dim_type.out, i) + "'"
            relation = relation.set_dim_name(dim_type.out, i, iname)

        return relation

    access_finder = ArrayAccessFinder()
    new_insns = []
    for insn in knl.instructions:

        new_happens_after: Mapping[str, HappensAfter] = {}

        # compute flow dependencies
        for variable in reads[insn.id]:
            for writer in writer_map.get(variable, set()):
                write_insn = knl.id_to_insn[writer]

                read_domain = knl.get_inames_domain(insn.within_inames)
                write_domain = knl.get_inames_domain(write_insn.within_inames)

                read_subscripts = access_finder(insn.expression)
                write_subscripts = access_finder(write_insn.assignee)

                assert isinstance(read_subscripts, set)
                assert isinstance(write_subscripts, set)

                read_maps = []
                for sub in read_subscripts:
                    if sub.aggregate.name == variable:
                        read_maps.append(
                                get_access_map(read_domain, sub.index_tuple)
                        )

                write_maps = []
                for sub in write_subscripts:
                    if sub.aggregate.name == variable:
                        write_maps.append(
                                get_access_map(write_domain, sub.index_tuple)
                        )

                assert len(read_maps) > 0
                assert len(write_maps) > 0

                read_map = reduce(operator.or_, read_maps)
                write_map = reduce(operator.or_, write_maps)

                read_write = get_unordered_deps(read_map, write_map)

                if writer in insn.happens_after:
                    lex_map = insn.happens_after[writer].instances_rel

                else:
                    lex_map = read_write.lex_lt_map(read_write)

                    if lex_map.space != read_write.space:
                        lex_map = make_inames_unique(lex_map)
                        read_write = make_inames_unique(read_write)

                        lex_map, read_write = isl.align_two(lex_map, read_write)

                flow_dependencies = read_write & lex_map

                if not flow_dependencies.is_empty():
                    if writer in new_happens_after:
                        old_rel = new_happens_after[writer].instances_rel
                        flow_dependencies |= old_rel

                    new_happens_after |= {
                            writer: HappensAfter(variable,
                                                 flow_dependencies)
                    }

        # compute anti and output dependencies
        for variable in writes[insn.id]:

            # compute anti dependencies
            for reader in reader_map.get(variable, set()):
                read_insn = knl.id_to_insn[reader]

                read_domain = knl.get_inames_domain(read_insn.within_inames)
                write_domain = knl.get_inames_domain(insn.within_inames)

                read_subscripts = access_finder(read_insn.expression)
                write_subscripts = access_finder(insn.assignee)

                assert isinstance(read_subscripts, set)
                assert isinstance(write_subscripts, set)

                read_maps = []
                for sub in read_subscripts:
                    if sub.aggregate.name == variable:
                        read_maps.append(
                                get_access_map(read_domain, sub.index_tuple)
                        )

                write_maps = []
                for sub in write_subscripts:
                    if sub.aggregate.name == variable:
                        write_maps.append(
                                get_access_map(write_domain, sub.index_tuple)
                        )

                assert len(read_maps) > 0
                assert len(write_maps) > 0

                read_map = reduce(operator.or_, read_maps)
                write_map = reduce(operator.or_, write_maps)

                write_read = get_unordered_deps(write_map, read_map)

                if reader in insn.happens_after:
                    lex_map = insn.happens_after[reader].instances_rel

                else:
                    lex_map = write_read.lex_lt_map(write_read)

                    if lex_map.space != write_read.space:
                        lex_map = make_inames_unique(lex_map)
                        write_read = make_inames_unique(lex_map)

                        lex_map, write_read = isl.align_two(lex_map, write_read)

                anti_dependencies = write_read & lex_map

                if not anti_dependencies.is_empty():
                    if reader in new_happens_after:
                        old_rel = new_happens_after[reader].instances_rel

                        anti_dependencies |= old_rel

                    new_happens_after |= {
                            reader: HappensAfter(variable, anti_dependencies)
                    }

            # compute output dependencies
            for writer in writer_map.get(variable, set()):
                before_write = knl.id_to_insn[writer]

                before_domain = knl.get_inames_domain(
                        before_write.within_inames
                )
                after_domain = knl.get_inames_domain(insn.within_inames)

                before_subscripts = access_finder(before_write.assignee)
                after_subscripts = access_finder(insn.assignee)

                assert isinstance(before_subscripts, set)
                assert isinstance(after_subscripts, set)

                before_maps = []
                for sub in before_subscripts:
                    if sub.aggregate.name == variable:
                        before_maps.append(
                                get_access_map(before_domain, sub.index_tuple)
                        )

                after_maps = []
                for sub in after_subscripts:
                    if sub.aggregate.name == variable:
                        after_maps.append(
                                get_access_map(after_domain, sub.index_tuple)
                        )

                assert len(before_maps) > 0
                assert len(after_maps) > 0

                before_map = reduce(operator.or_, before_maps)
                after_map = reduce(operator.or_, before_maps)

                write_write = get_unordered_deps(after_map, before_map)

                if writer in insn.happens_after:
                    lex_map = insn.happens_after[writer].instances_rel

                else:
                    lex_map = write_write.lex_lt_map(write_write)

                    if lex_map.space != write_write.space:
                        lex_map = make_inames_unique(lex_map)
                        write_write = make_inames_unique(write_write)

                        lex_map, write_write = isl.align_two(
                                lex_map, write_write
                        )

                output_dependencies = write_write & lex_map

                if not output_dependencies.is_empty():
                    if writer in new_happens_after[writer]:
                        old_rel = new_happens_after[writer].instances_rel

                        output_dependencies |= old_rel

                    new_happens_after |= {
                            writer: HappensAfter(variable, output_dependencies)
                    }

        new_insns.append(insn.copy(happens_after=new_happens_after))

    return knl.copy(instructions=new_insns)


@for_each_kernel
def add_lexicographic_happens_after(knl: LoopKernel) -> LoopKernel:
    """Compute an initial lexicographic happens-after ordering of the statments
    in a :class:`loopy.LoopKernel`. Statements are ordered in a sequential
    (C-like) manner.
    """

    new_insns = []

    for iafter, insn_after in enumerate(knl.instructions):

        if iafter == 0:
            new_insns.append(insn_after)

        else:

            insn_before = knl.instructions[iafter - 1]
            shared_inames = insn_after.within_inames & insn_before.within_inames

            domain_before = knl.get_inames_domain(insn_before.within_inames)
            domain_after = knl.get_inames_domain(insn_after.within_inames)
            happens_before = isl.Map.from_domain_and_range(
                    domain_before, domain_after
            )

            for idim in range(happens_before.dim(dim_type.out)):
                happens_before = happens_before.set_dim_name(
                        dim_type.out, idim,
                        happens_before.get_dim_name(dim_type.out, idim) + "'"
                )

            n_inames_before = happens_before.dim(dim_type.in_)
            happens_before_set = happens_before.move_dims(
                    dim_type.out, 0,
                    dim_type.in_, 0,
                    n_inames_before).range()

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

            affs = isl.affs_from_space(happens_before_set.space)

            lex_set = isl.Set.empty(happens_before_set.space)
            for iinnermost, innermost_iname in enumerate(shared_inames_order):

                innermost_set = affs[innermost_iname].lt_set(
                        affs[innermost_iname+"'"]
                )

                for outer_iname in shared_inames_order[:iinnermost]:
                    innermost_set = innermost_set & (
                            affs[outer_iname].eq_set(affs[outer_iname + "'"])
                    )

                lex_set = lex_set | innermost_set

            lex_map = isl.Map.from_range(lex_set).move_dims(
                    dim_type.in_, 0,
                    dim_type.out, 0,
                    n_inames_before)

            happens_before = happens_before & lex_map

            new_happens_after = {
                insn_before.id: HappensAfter(None, happens_before)
            }

            insn_after = insn_after.copy(happens_after=new_happens_after)

            new_insns.append(insn_after)

    return knl.copy(instructions=new_insns)

# vim: foldmethod=marker

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

from typing import Optional, FrozenSet, Set

import islpy as isl
from islpy import dim_type

import pymbolic.primitives as p

from loopy import LoopKernel
from loopy import InstructionBase
from loopy.kernel.instruction import HappensAfter
from loopy.symbolic import WalkMapper
from loopy.translation_unit import for_each_kernel
from loopy.symbolic import get_access_map


class AccessMapMapper(WalkMapper):
    """A subclass of :class:`loopy.symbolic.WalkMapper` used to generate
    individual array access maps of instructions. Similar to
    :class:`loopy.symbolic.BatchedAccessMapMapper`, except that it generates
    single access maps instead of combined access maps.

    .. attribute:: access_maps

        A dict containing the access map of a particular array as accessed by an
        instruction. These maps can be found via

        access_maps[insn_id][variable_name][inames]

    .. warning::
        This implementation of finding and storing access maps for instructions
        is subject to change.
    """

    def __init__(self, kernel: LoopKernel, variable_names: Set):
        self.kernel = kernel
        self._variable_names = variable_names

        # possibly not the final implementation of this
        from collections import defaultdict
        from typing import DefaultDict as Dict
        self.access_maps: Dict[Optional[str],\
                          Dict[Optional[str],\
                          Dict[FrozenSet, isl.Map]]] =\
                                                  defaultdict(lambda:
                                                  defaultdict(lambda:
                                                  defaultdict(lambda: None)))

        super().__init__()

    def map_subscript(self, expr: p.Subscript, inames: FrozenSet, insn_id: str):

        domain = self.kernel.get_inames_domain(inames)

        WalkMapper.map_subscript(self, expr, inames)

        assert isinstance(expr.aggregate, p.Variable)

        variable_name = expr.aggregate.name
        subscript = expr.index_tuple
        if variable_name not in self._variable_names:
            return

        from loopy.diagnostic import UnableToDetermineAccessRangeError

        try:
            access_map = get_access_map(domain, subscript)
        except UnableToDetermineAccessRangeError:
            return

        if self.access_maps[insn_id][variable_name][inames] is None:
            self.access_maps[insn_id][variable_name][inames] = access_map


@for_each_kernel
def compute_data_dependencies(knl: LoopKernel) -> LoopKernel:
    """Determine precise data dependencies between dynamic statement instances
    using the access relations of statements. Relies on there being an existing
    lexicographic ordering of statements.

    :arg knl:

        A :class:`loopy.LoopKernel` containing instructions to find the
        statement instance level dependencies of.
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

    variables = knl.all_variable_names()
    amap = AccessMapMapper(knl, variables)
    for insn in knl.instructions:
        amap(insn.assignee, insn.within_inames, insn.id)
        amap(insn.expression, insn.within_inames, insn.id)

    def get_relation(insn: InstructionBase, variable: Optional[str]) -> isl.Map:
        return amap.access_maps[insn.id][variable][insn.within_inames]

    def get_unordered_deps(r: isl.Map, s: isl.Map) -> isl.Map:
        # equivalent to the composition R^{-1} o S
        return s.apply_range(r.reverse())

    def make_out_inames_unique(relation: isl.Map) -> isl.Map:
        ndim = relation.dim(dim_type.out)
        for i in range(ndim):
            iname = relation.get_dim_name(dim_type.out, i) + "'"
            relation = relation.set_dim_name(dim_type.out, i, iname)

        return relation

    new_insns = []
    for cur_insn in knl.instructions:

        new_insn = cur_insn.copy()

        # handle read-after-write case
        for var in reads[cur_insn.id]:
            for writer in writer_map.get(var, set()) - {cur_insn.id}:

                # grab writer from knl.instructions
                write_insn = writer
                for insn in knl.instructions:
                    if writer == insn.id:
                        write_insn = insn.copy()
                        break

                read_rel = get_relation(cur_insn, var)
                write_rel = get_relation(write_insn, var)
                read_write = get_unordered_deps(read_rel, write_rel)

                # writer is immediately before reader
                if writer in cur_insn.happens_after:
                    lex_map = cur_insn.happens_after[writer].instances_rel

                    deps = lex_map & read_write
                    new_insn.happens_after.update(
                        {writer: HappensAfter(var, deps)}
                    )

                # writer is not immediately before reader
                else:
                    # generate a lexicographical map between the instructions
                    lex_map = read_write.lex_lt_map(read_write)

                    if lex_map.space != read_write.space:
                        # names may not be unique, make out names unique
                        lex_map = make_out_inames_unique(lex_map)
                        read_write = make_out_inames_unique(read_write)
                        lex_map, read_write = isl.align_two(lex_map, read_write)

                    deps = lex_map & read_write
                    new_insn.happens_after.update(
                            {writer: HappensAfter(var, deps)}
                    )

    # handle write-after-read and write-after-write
        for var in writes[cur_insn.id]:

            # write-after-read
            for reader in reader_map.get(var, set()) - {cur_insn.id}:

                read_insn = reader
                for insn in knl.instructions:
                    if reader == insn.id:
                        read_insn = insn.copy()
                        break

                write_rel = get_relation(cur_insn, var)
                read_rel = get_relation(read_insn, var)

                # calculate dependency map
                write_read = get_unordered_deps(write_rel, read_rel)

                # reader is immediately before writer
                if reader in cur_insn.happens_after:
                    lex_map = cur_insn.happens_after[reader].instances_rel
                    deps = lex_map & write_read
                    new_insn.happens_after.update(
                        {reader: HappensAfter(var, deps)}
                    )

                # reader is not immediately before writer
                else:
                    lex_map = write_read.lex_lt_map(write_read)

                    if lex_map.space != write_read.space:
                        # inames may not be unique, make out inames unique
                        lex_map = make_out_inames_unique(lex_map)
                        write_read = make_out_inames_unique(write_read)
                        lex_map, write_read = isl.align_two(lex_map, write_read)

                    deps = lex_map & write_read
                    new_insn.happens_after.update(
                            {reader: HappensAfter(var, deps)}
                    )

            # write-after-write
            for writer in writer_map.get(var, set()) - {cur_insn.id}:

                other_writer = writer
                for insn in knl.instructions:
                    if writer == insn.id:
                        other_writer = insn.copy()
                        break

                before_write_rel = get_relation(other_writer, var)
                after_write_rel = get_relation(cur_insn, var)

                write_write = get_unordered_deps(after_write_rel,
                                                 before_write_rel)

                # other writer is immediately before current writer
                if writer in cur_insn.happens_after:
                    lex_map = cur_insn.happens_after[writer].instances_rel
                    deps = lex_map & write_write
                    new_insn.happens_after.update(
                        {writer: HappensAfter(var, deps)}
                    )

                # there is not a writer immediately before current writer
                else:
                    lex_map = write_write.lex_lt_map(write_write)

                    if lex_map.space != write_write.space:
                        # make inames unique
                        lex_map = make_out_inames_unique(lex_map)
                        write_write = make_out_inames_unique(write_write)
                        lex_map, write_write = isl.align_two(lex_map,
                                                             write_write)

                    deps = lex_map & write_write
                    new_insn.happens_after.update(
                            {writer: HappensAfter(var, deps)}
                    )

        new_insns.append(new_insn)

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

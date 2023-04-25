"""
.. autoclass:: AccessMapFinder
.. autofunction:: compute_data_dependencies
"""

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

import islpy as isl
from islpy import dim_type

from loopy import LoopKernel
from loopy.kernel.instruction import HappensAfter
from loopy.translation_unit import for_each_kernel
from loopy.symbolic import UnableToDetermineAccessRangeError, get_access_map
from loopy.symbolic import WalkMapper

import pymbolic.primitives as p

from pyrsistent import pmap
from pyrsistent import PMap

from loopy.typing import Expression
from typing import List, Dict

from warnings import warn


class AccessMapFinder(WalkMapper):
    """Finds and stores relations representing the accesses to an array by
    statement instances. Access maps can be found using an instruction's ID and
    a variable's name. Essentially a specialized version of
    BatchedAccessMapMapper.
    """

    def __init__(self, knl: LoopKernel) -> None:
        self.kernel = knl
        self._access_maps: PMap[str, PMap[str, isl.Map]] = pmap({})
        from collections import defaultdict  # FIXME remove this
        self.bad_subscripts: Dict[str, List[Expression]] = defaultdict(list)

        super().__init__()

    def get_map(self, insn_id: str, variable_name: str) -> isl.Map | None:
        """Retrieve an access map indexed by an instruction ID and variable
        name.
        """
        try:
            return self._access_maps[insn_id][variable_name]
        except KeyError:
            return None

    def map_subscript(self, expr, insn_id):
        domain = self.kernel.get_inames_domain(
                self.kernel.id_to_insn[insn_id].within_inames
        )
        WalkMapper.map_subscript(self, expr, insn_id)

        assert isinstance(expr.aggregate, p.Variable)

        arg_name = expr.aggregate.name
        subscript = expr.index_tuple

        try:
            access_map = get_access_map(
                    domain, subscript, self.kernel.assumptions)
        except UnableToDetermineAccessRangeError:
            # may not have enough info to generate access map at current point
            self.bad_subscripts[arg_name].append(expr)
            return

        # analyze what we have in our access map dict before storing map
        insn_to_args = self._access_maps.get(insn_id)
        if insn_to_args is not None:
            existing_relation = insn_to_args.get(arg_name)

            if existing_relation is not None:
                access_map |= existing_relation

            self._access_maps = self._access_maps.set(
                    insn_id, self._access_maps[insn_id].set(
                        arg_name, access_map))

        else:
            self._access_maps = self._access_maps.set(
                    insn_id, pmap({arg_name: access_map}))

    def map_linear_subscript(self, expr, insn_id):
        raise NotImplementedError("linear subscripts cannot be used with "
                                  "precise dependency finding. Use "
                                  "multidimensional accesses to take advantage "
                                  "of this feature.")

    def map_reduction(self, expr, insn_id):
        return WalkMapper.map_reduction(self, expr, insn_id)

    def map_type_cast(self, expr, insn_id):
        return self.rec(expr.child, insn_id)

    def map_sub_array_ref(self, expr, insn_id):
        raise NotImplementedError("Not yet implemented")

@for_each_kernel
def narrow_dependencies(knl: LoopKernel) -> LoopKernel:
    """Attempt to relax a strict (lexical) ordering between statements in a
    kernel by way of finding data dependencies.
    """

    def make_inames_unique(relation: isl.Map) -> isl.Map:
        """Append a single-quote to all inames in the output dimension of a map
        ensure input/output inames do not match
        """
        for idim in range(relation.dim(dim_type.out)):
            iname = relation.get_dim_name(dim_type.out, idim) + "'"
            relation = relation.set_dim_name(dim_type.out, idim, iname)

        return relation


    def compute_data_dependencies(before_insn: str, after_insn: str,
                                  variable: str) -> isl.Map:
        """Compute a relation from instances of `after_insn` -> instances of
        `before_insn` that describes the instances of `after_insn` that must
        execute after instances of `before_insn`.

            .. arg:: before_insn
            The instruction id of the statement whose instances are in the range
            of the relation.

            .. arg:: after_insn
            The instruction id of the statement whose instances are in the
            domain of the relation.

            .. arg:: variable
            The variable responsible for the data dependency between the two
            statements.
        """

        assert isinstance(insn.id, str) # stop complaints

        before_map = amf.get_map(before_insn, variable)
        after_map = amf.get_map(after_insn, variable)

        if before_map is None:
            warn("unable to determine the access map for %s. "
                 "Defaulting to a conservative dependency relation between %s "
                 "and %s" % (before_insn, before_insn, after_insn))
            return

        if after_map is None:
            warn("unable to determine the access map for %s. "
                 "Defaulting to a conservative dependency relation between %s "
                 "and %s" % (after_insn, before_insn, after_insn))
            return

        dims = [before_map.dim(isl.dim_type.out),
                before_map.dim(isl.dim_type.in_),
                after_map.dim(isl.dim_type.out),
                after_map.dim(isl.dim_type.in_)]

        for i in range(len(dims)):
            if i == 0 or i == 1:
                scalar_insn = before_insn
            else:
                scalar_insn = after_insn

            if dims[i] == 0:
                warn("found evidence of a scalar access in %s. Defaulting to a "
                     "conservative dependency relation between "
                     "%s and %s" % (scalar_insn, before_insn, after_insn))
                return

        # map from after_instances -> before_instances
        unordered_deps = after_map.apply_range(before_map.reverse())
        identity = unordered_deps.identity(unordered_deps.get_space())
        unordered_deps -= identity

        if before_insn in insn.happens_after:
            lex_map = insn.happens_after[before_insn].instances_rel
        else:
            lex_map = unordered_deps.lex_lt_map(unordered_deps)

        assert lex_map is not None
        if lex_map.space != unordered_deps.space:
            lex_map = make_inames_unique(lex_map)
            unordered_deps = make_inames_unique(unordered_deps)

            lex_map, unordered_deps = isl.align_two(lex_map,
                                                    unordered_deps)

        deps = lex_map & unordered_deps

        return deps
    # end helper function definitions
    
    writer_map = knl.writer_map()
    reader_map = knl.reader_map()

    written_by = {insn.id: insn.write_dependency_names() - insn.within_inames
                  for insn in knl.instructions}
    read_by = {insn.id: insn.write_dependency_names() - insn.within_inames
               for insn in knl.instructions}

    # used to ensure that we are only storing dependencies at the instructions
    # that occur later in the program list
    ordered_insns = {insn.id: i for i, insn in enumerate(knl.instructions)}

    amf = AccessMapFinder(knl)
    for insn in knl.instructions:
        amf(insn.assignee, insn.id)
        amf(insn.expression, insn.id)

    new_insns = []
    for insn in knl.instructions:
        if ordered_insns[insn.id] == 0:
            new_insns.append(insn)
            continue

        new_happens_after = {}
        for variable in read_by[insn.id]:

            # handle flow-dependencies (read-after-write)
            for before_insn in writer_map.get(variable, set()) - {insn.id}:
                if ordered_insns[before_insn] > ordered_insns[insn.id]:
                    continue

                assert isinstance(insn.id, str) # stop complaints
                deps = compute_data_dependencies(before_insn, insn.id, variable)

                if deps is None or not deps.is_empty():
                    new_happens_after = dict(
                            list(insn.happens_after.items())
                            +
                            [(before_insn, HappensAfter(variable, deps))])

        for variable in written_by[insn.id]:

            # handle anti dependencies (write-after-read)
            for before_insn in reader_map.get(variable, set()) - {insn.id}:
                if ordered_insns[before_insn] > ordered_insns[insn.id]:
                    continue

                assert isinstance(insn.id, str) # stop complaints
                deps = compute_data_dependencies(before_insn, insn.id, variable)

                if deps is None or not deps.is_empty():
                    new_happens_after = dict(
                            list(insn.happens_after.items())
                            +
                            [(before_insn, HappensAfter(variable, deps))])

            # handle output dependencies (write-after-write)
            for before_insn in writer_map.get(variable, set()) - {insn.id}:
                if ordered_insns[before_insn] > ordered_insns[insn.id]:
                    continue

                assert isinstance(insn.id, str) # stop complaints
                deps = compute_data_dependencies(before_insn, insn.id, variable)

                if deps is None or not deps.is_empty():
                    new_happens_after = dict(
                            list(insn.happens_after.items())
                            +
                            [(before_insn, HappensAfter(variable, deps))])

        # clean up any deps that do not contain a variable name
        new_happens_after = {insn_id: dep
                             for insn_id, dep in new_happens_after.items()
                             if dep.variable_name is not None}

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

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

import pymbolic.primitives as p

from dataclasses import dataclass
from typing import Optional

from loopy import LoopKernel
from loopy.symbolic import WalkMapper
from loopy.translation_unit import for_each_kernel
from loopy.typing import ExpressionT

@dataclass(frozen=True)
class HappensAfter:
    variable_name: Optional[str]
    instances_rel: Optional[isl.Map]

class AccessMapMapper(WalkMapper):
    """
    TODO Update this documentation so it reflects proper formatting

    Used instead of BatchedAccessMapMapper to get single access maps for each
    instruction.
    """

    def __init__(self, kernel: LoopKernel, var_names: set):
        self.kernel = kernel
        self._var_names = var_names

        from collections import defaultdict
        self.access_maps = defaultdict(lambda:
                           defaultdict(lambda:
                           defaultdict(lambda: None)))

        super.__init__()

    def map_subscript(self, expr: ExpressionT, inames: frozenset, insn_id: str):

        domain = self.kernel.get_inames_domain(inames)

        WalkMapper.map_subscript(self, expr, inames)

        assert isinstance(expr.aggregate, p.Variable)

        if expr.aggregate.name not in self._var_names:
            return

        arg_name = expr.aggregate.name
        subscript = expr.index_tuple

        from loopy.diagnostic import UnableToDetermineAccessRangeError
        from loopy.symbolic import get_access_map

        try:
            access_map = get_access_map(domain, subscript)
        except UnableToDetermineAccessRangeError:
            return

        if self.access_maps[insn_id][arg_name][inames] is None:
            self.access_maps[insn_id][arg_name][inames] = access_map

def compute_happens_after(knl: LoopKernel) -> LoopKernel:
    """
    TODO Update documentation to reflect the proper format.

    Determine dependency relations that exist between instructions. Similar to
    apply_single_writer_dependency_heuristic. Extremely rough draft.
    """
    writer_map = knl.writer_map()
    variables = knl.all_variable_names - knl.inames.keys()

    # initialize the mapper
    amap = AccessMapMapper(knl, variables)

    for insn in knl.instructions:
        amap(insn.assignee, insn.within_inames)
        amap(insn.expression, insn.within_inames)

    # compute data dependencies
    dep_map = {
        insn.id: insn.read_dependency_names() - insn.within_inames
        for insn in knl.instructions
    }

    new_insns = []
    for insn in knl.instructions:
        current_insn = insn.id
        inames = insn.within_inames

        new_happens_after = []
        for var in dep_map[insn.id]:
            for writer in (writer_map.get(var, set()) - { current_insn }):

                # get relation for current instruction and a write instruction
                cur_relation = amap.access_maps[current_insn][var][inames]
                write_relation = amap.access_maps[writer][var][inames]

                # compute the dependency relation
                dep_relation = cur_relation.apply_range(write_relation.reverse())

                # create the mapping from writer -> (variable, dependency rel'n)
                happens_after = HappensAfter(var, dep_relation)
                happens_after_mapping = { writer: happens_after }

                # add to the new list of dependencies
                new_happens_after |= happens_after_mapping

        # update happens_after of our current instruction with the mapping
        insn = insn.copy(happens_after=new_happens_after)
        new_insns.append(insn)

    # return the kernel with the new instructions
    return knl.copy(instructions=new_insns)

@for_each_kernel
def add_lexicographic_happens_after(knl: LoopKernel) -> LoopKernel:
    """
    TODO update documentation to follow the proper format

    Determine a coarse "happens-before" relationship between an instruction and
    the instruction immediately preceeding it. This strict execution order is
    relaxed in accordance with the data dependency relations.

    See `loopy.dependency.compute_happens_after` for data dependency generation.
    """

    new_insns = []

    for iafter, insn_after in enumerate(knl.instructions):

        # the first instruction does not have anything preceding it
        if iafter == 0:
            new_insns.append(insn_after)
        
        # all other instructions "happen after" the instruction before it
        else:

            # not currently used
            # unshared_before = insn_before.within_inames
            
            # get information about the preceding instruction
            insn_before = knl.instructions[iafter - 1]
            shared_inames = insn_after.within_inames & insn_before.within_inames

            # generate a map from the preceding insn to the current insn
            domain_before = knl.get_inames_domain(insn_before.within_inames)
            domain_after = knl.get_inames_domain(insn_after.within_inames)
            happens_before = isl.Map.from_domain_and_range(
                    domain_before, domain_after
            )

            # update inames so they are unique 
            for idim in range(happens_before.dim(dim_type.out)):
                happens_before = happens_before.set_dim_name(
                        dim_type.out, idim,
                        happens_before.get_dim_name(dim_type.out, idim) + "'")
            
            # generate a set containing all inames (from both domains)
            n_inames_before = happens_before.dim(dim_type.in_)
            happens_before_set = happens_before.move_dims(
                    dim_type.out, 0,
                    dim_type.in_, 0,
                    n_inames_before).range()

            # verify the order of the inames
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

            # generate lexicographical map from space of preceding to current insn
            affs = isl.affs_from_space(happens_before_set.space)

            # start with an empty set
            lex_set = isl.Set.empty(happens_before_set.space)
            for iinnermost, innermost_iname in enumerate(shared_inames_order):
                
                innermost_set = affs[innermost_iname].lt_set(
                        affs[innermost_iname+"'"]
                )
                
                for  outer_iname in shared_inames_order[:iinnermost]:
                    innermost_set = innermost_set & (
                            affs[outer_iname].eq_set(affs[outer_iname + "'"])
                            )

                # update the set
                lex_set = lex_set | innermost_set

            # create the map
            lex_map = isl.Map.from_range(lex_set).move_dims(
                    dim_type.in_, 0,
                    dim_type.out, 0,
                    n_inames_before)

            # update happens_before map
            happens_before = happens_before & lex_map

            # create HappensAfter and add the relation to it
            new_happens_after = { 
                insn_before.id: HappensAfter(None, happens_before) 
            }

            insn_after = insn_after.copy(happens_after=new_happens_after)

            # update instructions
            new_insns.append(insn_after)

    return knl.copy(instructions=new_insns)


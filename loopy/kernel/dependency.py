# FIXME Add copyright header


import islpy as isl
from islpy import dim_type
import pymbolic.primitives as p

from dataclasses import dataclass
from islpy import Map
from typing import Optional

from loopy import LoopKernel
from loopy.symbolic import WalkMapper
from loopy.translation_unit import for_each_kernel
from loopy.typing import ExpressionT

@dataclass(frozen=True)
class HappensAfter:
    variable_name: Optional[str]
    instances_rel: Optional[Map]

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

def add_lexicographic_happens_after_orig(knl: LoopKernel) -> None:
    """
    TODO properly format this documentation.

    Creates a dependency relation between two instructions based on a
    lexicographic ordering of the statements in a program.

    For example, the C-like execution order (i.e. sequential ordering) of a
    program.
    """

    # we want to modify the output dimension and OUT = 3
    dim_type = isl.dim_type.out

    # generate an unordered mapping from statement instances to points in the
    # loop domain
    insn_number = 0
    schedules = {}
    for insn in knl.instructions:
        domain = knl.get_inames_domain(insn.within_inames)

        # if we do not set the dim name, the name is set as None
        domain = domain.insert_dims(dim_type, 0, 1).set_dim_name(dim_type, 0,
                                                                 insn.id)

        space = domain.get_space()
        domain = domain.add_constraint(
            isl.Constraint.eq_from_names(space, {1: -1*insn_number, insn.id: 1})
        )

        # this may not be the final way we keep track of the schedules
        schedule = isl.Map.from_domain_and_range(domain, domain)
        schedules[insn.id] = schedule

        insn_number += 1

        # determine a lexicographic order on the space the schedules belong to


@for_each_kernel
def add_lexicographic_happens_after(knl: LoopKernel) -> LoopKernel:

    new_insns = []

    for iafter, insn_after in enumerate(knl.instructions):
        if iafter == 0:
            new_insns.append(insn_after)
        else:
            insn_before = knl.instructions[iafter - 1]
            shared_inames = insn_after.within_inames & insn_before.within_inames
            unshared_before = insn_before.within_inames

            domain_before = knl.get_inames_domain(insn_before.within_inames)
            domain_after = knl.get_inames_domain(insn_after.within_inames)

            happens_before = isl.Map.from_domain_and_range(
                    domain_before, domain_after)
            for idim in range(happens_before.dim(dim_type.out)):
                happens_before = happens_before.set_dim_name(
                        dim_type.out, idim,
                        happens_before.get_dim_name(dim_type.out, idim) + "'")
            n_inames_before = happens_before.dim(dim_type.in_)
            happens_before_set = happens_before.move_dims(
                    dim_type.out, 0,
                    dim_type.in_, 0,
                    n_inames_before).range()

            shared_inames_order_before = [
                    domain_before.get_dim_name(dim_type.out, idim)
                    for idim in range(domain_before.dim(dim_type.out))
                    if domain_before.get_dim_name(dim_type.out, idim)
                    in shared_inames]
            shared_inames_order_after = [
                    domain_after.get_dim_name(dim_type.out, idim)
                    for idim in range(domain_after.dim(dim_type.out))
                    if domain_after.get_dim_name(dim_type.out, idim)
                    in shared_inames]

            assert shared_inames_order_after == shared_inames_order_before
            shared_inames_order = shared_inames_order_after

            affs = isl.affs_from_space(happens_before_set.space)

            lex_set = isl.Set.empty(happens_before_set.space)
            for iinnermost, innermost_iname in enumerate(shared_inames_order):
                innermost_set = affs[innermost_iname].lt_set(
                        affs[innermost_iname+"'"])

                for  outer_iname in shared_inames_order[:iinnermost]:
                    innermost_set = innermost_set & (
                            affs[outer_iname].eq_set(affs[outer_iname + "'"]))

                lex_set = lex_set | innermost_set

            lex_map = isl.Map.from_range(lex_set).move_dims(
                    dim_type.in_, 0,
                    dim_type.out, 0,
                    n_inames_before)

            happens_before = happens_before & lex_map

            pu.db

            new_insns.append(insn_after)

    return knl.copy(instructions=new_insns)




"""Code generation for Instruction objects."""
from __future__ import division

from pytools import Record
import islpy as isl




# {{{ ILP instance

class ILPInstance(Record):
    """
    :ivar ilp_key: a frozenset of tuples (iname, assignment)
    """
    __slots__ = ["implemented_domain", "assignments", "ilp_key"]

    def __init__(self, implemented_domain, assignments, ilp_key):
        Record.__init__(self,
                implemented_domain=implemented_domain,
                assignments=assignments,
                ilp_key=ilp_key)

    def fix(self, iname, aff):
        dt, pos = aff.get_space().get_var_dict()[iname]
        iname_plus_lb_aff = aff.add_coefficient(
                dt, pos, -1)

        from loopy.symbolic import pw_aff_to_expr
        cns = isl.Constraint.equality_from_aff(iname_plus_lb_aff)
        expr = pw_aff_to_expr(aff)

        return ILPInstance(
                implemented_domain=self.implemented_domain.add_constraint(cns),
                c_code_mapper=self.c_code_mapper.copy_and_assign(iname, expr),
                ilp_key=self.ilp_key | frozenset([(iname, expr)]))

# }}}




def generate_ilp_instances(kernel, insn, codegen_state):
    assignments = {}
    impl_domain = codegen_state.implemented_domain

    from loopy.kernel import (TAG_ILP,
            TAG_LOCAL_IDX, TAG_GROUP_IDX)

    from pymbolic import var

    # {{{ pass 1: assign all hw-parallel dimensions

    global_size, local_size = kernel.get_grid_sizes()

    for iname in insn.all_inames():
        tag = kernel.iname_to_tag.get(iname)

        if isinstance(tag, TAG_LOCAL_IDX):
            hw_axis_expr = var("(int) get_local_id")(tag.axis)
            hw_axis_size = local_size[tag.axis]

        elif isinstance(tag, TAG_GROUP_IDX):
            hw_axis_expr = var("(int) get_group_id")(tag.axis)
            hw_axis_size = global_size[tag.axis]

        else:
            continue

        bounds = kernel.get_iname_bounds(iname)

        from loopy.isl import make_slab
        impl_domain = impl_domain.intersect(
                make_slab(impl_domain.get_space(), iname,
                    bounds.lower_bound_pw_aff, bounds.lower_bound_pw_aff+hw_axis_size))

        from loopy.symbolic import pw_aff_to_expr
        assignments[iname] = pw_aff_to_expr(bounds.lower_bound_pw_aff + hw_axis_expr)

    # }}} 

    result = [ILPInstance(impl_domain, assignments, frozenset())]

    # {{{ pass 2: treat all ILP dimensions

    for iname in insn.all_inames():
        tag = kernel.iname_to_tag.get(iname)

        if not isinstance(tag, TAG_ILP):
            continue

        from warnings import warn
        warn("implement ILP instance generation")

    # }}}

    return result




def generate_instruction_code(kernel, insn, codegen_state):
    result = []

    for ilpi in generate_ilp_instances(kernel, insn, codegen_state):
        ccm = codegen_state.c_code_mapper.copy_and_assign_many(ilpi.assignments)

        # FIXME we should probably share some checks across ILP instances

        from cgen import Assign
        insn_code = Assign(ccm(insn.assignee), ccm(insn.expression))
        from loopy.codegen.bounds import wrap_in_bounds_checks
        insn_code = wrap_in_bounds_checks(
                ccm, kernel.domain, insn.all_inames(), ilpi.implemented_domain,
                insn_code)

        result.append(insn_code)

    from loopy.codegen import gen_code_block
    return gen_code_block(result)

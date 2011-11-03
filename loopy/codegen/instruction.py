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
        from loopy.isl_helpers import iname_rel_aff
        iname_plus_lb_aff = iname_rel_aff(
                self.implemented_domain.get_space(), iname, "==", aff)

        from loopy.symbolic import pw_aff_to_expr
        expr = pw_aff_to_expr(aff)

        cns = isl.Constraint.equality_from_aff(iname_plus_lb_aff)

        new_assignments = self.assignments.copy()
        new_assignments[iname] = expr
        return ILPInstance(
                implemented_domain=self.implemented_domain.add_constraint(cns),
                assignments=new_assignments,
                ilp_key=self.ilp_key | set([(iname, expr)]))

# }}}




def generate_ilp_instances(kernel, insn, codegen_state):
    impl_domain = codegen_state.implemented_domain

    from loopy.kernel import IlpTag

    result = [ILPInstance(impl_domain, {}, frozenset())]

    # {{{ pass 2: treat all ILP dimensions

    for iname in kernel.insn_inames(insn):
        tag = kernel.iname_to_tag.get(iname)

        if not isinstance(tag, IlpTag):
            continue


        bounds = kernel.get_iname_bounds(iname)

        from loopy.isl_helpers import (
                static_max_of_pw_aff, static_value_of_pw_aff)
        from loopy.symbolic import pw_aff_to_expr

        length = int(pw_aff_to_expr(
            static_max_of_pw_aff(bounds.size, constants_only=True)))
        lower_bound_aff = static_value_of_pw_aff(
                bounds.lower_bound_pw_aff.coalesce(),
                constants_only=False)

        new_result = []
        for ilpi in result:
            for i in range(length):
                idx_aff = lower_bound_aff + i
                new_result.append(ilpi.fix(iname, idx_aff))

        result = new_result

    # }}}

    return result




def generate_instruction_code(kernel, insn, codegen_state):
    result = []
    from loopy.codegen import GeneratedInstruction

    for ilpi in generate_ilp_instances(kernel, insn, codegen_state):
        ccm = codegen_state.c_code_mapper.copy_and_assign_many(ilpi.assignments)

        # FIXME we should probably share some checks across ILP instances

        from cgen import Assign
        insn_code = Assign(ccm(insn.assignee), ccm(insn.expression))
        from loopy.codegen.bounds import wrap_in_bounds_checks
        insn_code, impl_domain = wrap_in_bounds_checks(
                ccm, kernel.domain, kernel.insn_inames(insn), ilpi.implemented_domain,
                insn_code)

        result.append(GeneratedInstruction(
            insn_id=insn.id,
            implemented_domain=impl_domain,
            ast=insn_code))

    from loopy.codegen import gen_code_block
    return gen_code_block(result)





# vim: foldmethod=marker

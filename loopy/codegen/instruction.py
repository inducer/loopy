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
    impl_domain = codegen_state.implemented_domain

    from loopy.kernel import IlpTag

    result = [ILPInstance(impl_domain, {}, frozenset())]

    # {{{ pass 2: treat all ILP dimensions

    for iname in insn.all_inames():
        tag = kernel.iname_to_tag.get(iname)

        if not isinstance(tag, IlpTag):
            continue

        from warnings import warn
        warn("implement ILP instance generation")

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
        insn_code = wrap_in_bounds_checks(
                ccm, kernel.domain, insn.all_inames(), ilpi.implemented_domain,
                insn_code)

        result.append(GeneratedInstruction(
            insn_id=insn.id,
            implemented_domain=ilpi.implemented_domain,
            ast=insn_code))

    from loopy.codegen import gen_code_block
    return gen_code_block(result)





# vim: foldmethod=marker

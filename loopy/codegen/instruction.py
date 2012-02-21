"""Code generation for Instruction objects."""
from __future__ import division




def generate_instruction_code(kernel, insn, codegen_state):
    from loopy.codegen import GeneratedInstruction

    ccm = codegen_state.c_code_mapper

    from cgen import Assign
    insn_code = Assign(ccm(insn.assignee), ccm(insn.expression))
    from loopy.codegen.bounds import wrap_in_bounds_checks
    insn_code, impl_domain = wrap_in_bounds_checks(
            ccm, kernel.domain, kernel.insn_inames(insn),
            codegen_state.implemented_domain,
            insn_code)

    return GeneratedInstruction(
        insn_id=insn.id,
        implemented_domain=impl_domain,
        ast=insn_code)




# vim: foldmethod=marker

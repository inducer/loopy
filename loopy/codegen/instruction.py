"""Code generation for Instruction objects."""
from __future__ import division




def generate_instruction_code(kernel, insn, codegen_state):
    from loopy.codegen import GeneratedInstruction

    ccm = codegen_state.c_code_mapper

    expr = insn.expression

    from loopy.codegen.expression import perform_cast
    expr = perform_cast(ccm, expr, expr_dtype=ccm.infer_type(expr),
            target_dtype=kernel.get_var_descriptor(insn.get_assignee_var_name()).dtype)

    from cgen import Assign
    insn_code = Assign(ccm(insn.assignee), ccm(expr))
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

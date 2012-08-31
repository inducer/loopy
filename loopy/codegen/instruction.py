"""Code generation for Instruction objects."""
from __future__ import division




def generate_instruction_code(kernel, insn, codegen_state):
    from loopy.codegen import GeneratedInstruction

    ccm = codegen_state.c_code_mapper

    expr = insn.expression

    from loopy.codegen.expression import perform_cast
    target_dtype = kernel.get_var_descriptor(insn.get_assignee_var_name()).dtype
    expr_dtype = ccm.infer_type(expr)

    expr = perform_cast(ccm, expr,
            expr_dtype=expr_dtype,
            target_dtype=target_dtype)

    from cgen import Assign
    from loopy.codegen.expression import dtype_to_type_context
    insn_code = Assign(
            ccm(insn.assignee, prec=None, type_context=None),
            ccm(expr, prec=None, type_context=dtype_to_type_context(target_dtype)))

    from loopy.codegen.bounds import wrap_in_bounds_checks
    insn_inames = kernel.insn_inames(insn)
    insn_code, impl_domain = wrap_in_bounds_checks(
            ccm, kernel.get_inames_domain(insn_inames), insn_inames,
            codegen_state.implemented_domain,
            insn_code)

    result = GeneratedInstruction(
        insn_id=insn.id,
        implemented_domain=impl_domain,
        ast=insn_code)

    if 0:
        from loopy.codegen import gen_code_block
        from cgen import Statement as S
        idx = insn.get_assignee_indices()

        if idx:
            result = gen_code_block([
                GeneratedInstruction(
                    ast=S(r'printf("write %s[%s]\n", %s);'
                        % (insn.get_assignee_var_name(),
                            ",".join(len(idx) * ["%d"]),
                            ",".join(ccm(i, prec=None, type_context="i") for i in idx))),
                    implemented_domain=None),
                result
                ])

    return result





# vim: foldmethod=marker

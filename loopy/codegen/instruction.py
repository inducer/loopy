"""Code generation for Instruction objects."""
from __future__ import division

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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
from loopy.codegen import GeneratedInstruction


def wrap_in_bounds_checks(ccm, domain, check_inames, implemented_domain, stmt):
    from loopy.codegen.bounds import get_bounds_checks, constraint_to_code
    bounds_checks = get_bounds_checks(
            domain, check_inames,
            implemented_domain, overapproximate=False)

    bounds_check_set = isl.Set.universe(domain.get_space()) \
            .add_constraints(bounds_checks)
    bounds_check_set, new_implemented_domain = isl.align_two(
            bounds_check_set, implemented_domain)
    new_implemented_domain = new_implemented_domain & bounds_check_set

    condition_codelets = [constraint_to_code(ccm, cns) for cns in bounds_checks]

    if condition_codelets:
        from cgen import If
        stmt = If("\n&& ".join(condition_codelets), stmt)

    return stmt, new_implemented_domain


def generate_instruction_code(kernel, insn, codegen_state):
    from loopy.kernel.data import ExpressionInstruction, CInstruction

    if isinstance(insn, ExpressionInstruction):
        return generate_expr_instruction_code(kernel, insn, codegen_state)
    elif isinstance(insn, CInstruction):
        return generate_c_instruction_code(kernel, insn, codegen_state)
    else:
        raise RuntimeError("unexpected instruction type")


def generate_expr_instruction_code(kernel, insn, codegen_state):

    ccm = codegen_state.c_code_mapper

    expr = insn.expression

    (assignee_var_name, assignee_indices), = insn.assignees_and_indices()
    target_dtype = kernel.get_var_descriptor(assignee_var_name).dtype

    from cgen import Assign
    from loopy.codegen.expression import dtype_to_type_context
    insn_code = Assign(
            ccm(insn.assignee, prec=None, type_context=None),
            ccm(expr, prec=None, type_context=dtype_to_type_context(target_dtype),
                needed_dtype=target_dtype))

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

        if assignee_indices:
            result = gen_code_block([
                GeneratedInstruction(
                    ast=S(r'printf("write %s[%s]\n", %s);'
                        % (assignee_var_name,
                            ",".join(len(assignee_indices) * ["%d"]),
                            ",".join(
                                ccm(i, prec=None, type_context="i")
                                for i in assignee_indices))),
                    implemented_domain=None),
                result
                ])

    return result


def generate_c_instruction_code(kernel, insn, codegen_state):
    ccm = codegen_state.c_code_mapper

    body = []

    from loopy.codegen import POD
    from cgen import Initializer, Block, Line

    from pymbolic.primitives import Variable
    for name, iname_expr in insn.iname_exprs:
        if (isinstance(iname_expr, Variable)
                and name not in ccm.var_subst_map):
            # No need, the bare symbol will work
            continue

        body.append(
                Initializer(
                    POD(kernel.index_dtype, name),
                    codegen_state.c_code_mapper(
                        iname_expr, prec=None, type_context="i")))

    if body:
        body.append(Line())

    body.extend(Line(l) for l in insn.code.split("\n"))

    insn_inames = kernel.insn_inames(insn)
    insn_code, impl_domain = wrap_in_bounds_checks(
            ccm, kernel.get_inames_domain(insn_inames), insn_inames,
            codegen_state.implemented_domain,
            Block(body))

    return GeneratedInstruction(
        insn_id=insn.id,
        implemented_domain=impl_domain,
        ast=insn_code)


# vim: foldmethod=marker

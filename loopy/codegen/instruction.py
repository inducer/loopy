"""Code generation for Instruction objects."""


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
dim_type = isl.dim_type
from loopy.codegen import UnvectorizableError
from loopy.codegen.result import CodeGenerationResult
from pymbolic.mapper.stringifier import PREC_NONE
from pytools import memoize_on_first_arg


# These 'id' arguments are here because Set has a __hash__ supplied by isl,
# which ignores names. This may lead to incorrect things being returned from
# the cache. Passing Set id()s breaks that cache aliasing.
# This should be removed once there is a proper solution for the cache
# aliasing, such as what's under discussion in
# https://github.com/inducer/islpy/pull/103/.
@memoize_on_first_arg
def _get_new_implemented_domain(
        kernel,
        id_chk_domain, chk_domain,
        id_implemented_domain, implemented_domain):

    chk_domain, implemented_domain = isl.align_two(
            chk_domain, implemented_domain)
    chk_domain = chk_domain.gist(implemented_domain)

    new_implemented_domain = implemented_domain & chk_domain
    return chk_domain, new_implemented_domain


def to_codegen_result(
        codegen_state, insn_id, domain, check_inames, required_preds, ast):
    chk_domain = isl.Set.from_basic_set(domain)
    chk_domain = chk_domain.remove_redundancies()
    chk_domain = codegen_state.kernel.cache_manager.eliminate_except(chk_domain,
            check_inames, (dim_type.set,))

    chk_domain, new_implemented_domain = _get_new_implemented_domain(
            codegen_state.kernel,
            id(chk_domain), chk_domain,
            id(codegen_state.implemented_domain), codegen_state.implemented_domain)

    if chk_domain.is_empty():
        return None

    condition_exprs = []
    if not chk_domain.plain_is_universe():
        from loopy.symbolic import set_to_cond_expr
        condition_exprs.append(set_to_cond_expr(chk_domain))

    condition_exprs.extend(
            required_preds - codegen_state.implemented_predicates)

    if condition_exprs:
        from pymbolic.primitives import LogicalAnd
        from pymbolic.mapper.stringifier import PREC_NONE
        ast = codegen_state.ast_builder.emit_if(
                codegen_state.expression_to_code_mapper(
                    LogicalAnd(tuple(condition_exprs)), PREC_NONE),
                ast)

    return CodeGenerationResult.new(
            codegen_state, insn_id, ast, new_implemented_domain)


def generate_instruction_code(codegen_state, insn):
    kernel = codegen_state.kernel

    from loopy.kernel.instruction import (
        Assignment, CallInstruction, CInstruction, NoOpInstruction
    )

    if isinstance(insn, Assignment):
        ast = generate_assignment_instruction_code(codegen_state, insn)
    elif isinstance(insn, CallInstruction):
        ast = generate_call_code(codegen_state, insn)
    elif isinstance(insn, CInstruction):
        ast = generate_c_instruction_code(codegen_state, insn)
    elif isinstance(insn, NoOpInstruction):
        ast = generate_nop_instruction_code(codegen_state, insn)
    else:
        raise RuntimeError("unexpected instruction type")

    insn_inames = insn.within_inames

    return to_codegen_result(
            codegen_state,
            insn.id,
            kernel.get_inames_domain(insn_inames), insn_inames,
            insn.predicates,
            ast)


def generate_assignment_instruction_code(codegen_state, insn):
    kernel = codegen_state.kernel

    ecm = codegen_state.expression_to_code_mapper

    from loopy.expression import VectorizabilityChecker

    # {{{ vectorization handling

    if codegen_state.vectorization_info:
        if insn.atomicity:
            raise UnvectorizableError("atomic operation")

        vinfo = codegen_state.vectorization_info
        vcheck = VectorizabilityChecker(
                kernel, vinfo.iname, vinfo.length)
        lhs_is_vector = vcheck(insn.assignee)
        rhs_is_vector = vcheck(insn.expression)

        if not lhs_is_vector and rhs_is_vector:
            raise UnvectorizableError(
                    "LHS is scalar, RHS is vector, cannot assign")

        is_vector = lhs_is_vector

        del lhs_is_vector
        del rhs_is_vector

    # }}}

    from pymbolic.primitives import Variable, Subscript, Lookup
    from loopy.symbolic import LinearSubscript

    lhs = insn.assignee
    if isinstance(lhs, Lookup):
        lhs = lhs.aggregate

    if isinstance(lhs, Variable):
        assignee_var_name = lhs.name
        assignee_indices = ()

    elif isinstance(lhs, Subscript):
        assignee_var_name = lhs.aggregate.name
        assignee_indices = lhs.index_tuple

    elif isinstance(lhs, LinearSubscript):
        assignee_var_name = lhs.aggregate.name
        assignee_indices = (lhs.index,)

    else:
        raise RuntimeError("invalid lvalue '%s'" % lhs)

    del lhs

    result = codegen_state.ast_builder.emit_assignment(codegen_state, insn)

    # {{{ tracing

    lhs_dtype = codegen_state.kernel.get_var_descriptor(assignee_var_name).dtype

    if kernel.options.trace_assignments or kernel.options.trace_assignment_values:
        if codegen_state.vectorization_info and is_vector:
            raise UnvectorizableError("tracing does not support vectorization")

        from pymbolic.mapper.stringifier import PREC_NONE
        lhs_code = codegen_state.expression_to_code_mapper(insn.assignee, PREC_NONE)

        from cgen import Statement as S  # noqa

        gs, ls = kernel.get_grid_size_upper_bounds(codegen_state.callables_table)

        printf_format = "{}.{}[{}][{}]: {}".format(
                kernel.name,
                insn.id,
                ", ".join("gid%d=%%d" % i for i in range(len(gs))),
                ", ".join("lid%d=%%d" % i for i in range(len(ls))),
                assignee_var_name)

        printf_args = (
                ["gid(%d)" % i for i in range(len(gs))]
                +
                ["lid(%d)" % i for i in range(len(ls))]
                )

        if assignee_indices:
            printf_format += "[%s]" % ",".join(len(assignee_indices) * ["%d"])
            printf_args.extend(
                    ecm(i, prec=PREC_NONE, type_context="i")
                    for i in assignee_indices)

        if kernel.options.trace_assignment_values:
            if lhs_dtype.numpy_dtype.kind == "i":
                printf_format += " = %d"
                printf_args.append(lhs_code)
            elif lhs_dtype.numpy_dtype.kind == "f":
                printf_format += " = %g"
                printf_args.append(lhs_code)
            elif lhs_dtype.numpy_dtype.kind == "c":
                printf_format += " = %g + %gj"
                printf_args.extend([
                    "(%s).x" % lhs_code,
                    "(%s).y" % lhs_code])

        if printf_args:
            printf_args_str = ", " + ", ".join(str(v) for v in printf_args)
        else:
            printf_args_str = ""

        printf_insn = S('printf("{}\\n"{})'.format(
                    printf_format, printf_args_str))

        from cgen import Block
        if kernel.options.trace_assignment_values:
            result = Block([result, printf_insn])
        else:
            # print first, execute later -> helps find segfaults
            result = Block([printf_insn, result])

    # }}}

    return result


def generate_call_code(codegen_state, insn):
    kernel = codegen_state.kernel

    # {{{ vectorization handling

    if codegen_state.vectorization_info:
        if insn.atomicity:
            raise UnvectorizableError("atomic operation")

    # }}}

    result = codegen_state.ast_builder.emit_multiple_assignment(
            codegen_state, insn)

    # {{{ tracing

    if kernel.options.trace_assignments or kernel.options.trace_assignment_values:
        raise NotImplementedError("tracing of multi-output function calls")

    # }}}

    return result


def generate_c_instruction_code(codegen_state, insn):
    kernel = codegen_state.kernel

    if codegen_state.vectorization_info is not None:
        raise UnvectorizableError("C instructions cannot be vectorized")

    body = []

    from loopy.target.c import POD
    from cgen import Initializer, Block, Line

    from pymbolic.primitives import Variable
    for name, iname_expr in insn.iname_exprs:
        if (isinstance(iname_expr, Variable)
                and name not in codegen_state.var_subst_map):
            # No need, the bare symbol will work
            continue

        body.append(
                Initializer(
                    POD(codegen_state.ast_builder, kernel.index_dtype, name),
                    codegen_state.expression_to_code_mapper(
                        iname_expr, prec=PREC_NONE, type_context="i")))

    if body:
        body.append(Line())

    body.extend(Line(line) for line in insn.code.split("\n"))

    return Block(body)


def generate_nop_instruction_code(codegen_state, insn):
    if codegen_state.vectorization_info is not None:
        raise UnvectorizableError("C instructions cannot be vectorized")
    return codegen_state.ast_builder.emit_noop_with_comment(
        "no-op (insn=%s)" % (insn.id))

# vim: foldmethod=marker

"""Code generation for Instruction objects."""

from __future__ import division, absolute_import

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


from six.moves import range
import islpy as isl
from loopy.codegen import Unvectorizable
from loopy.codegen.result import CodeGenerationResult
from pymbolic.mapper.stringifier import PREC_NONE


def to_codegen_result(
        codegen_state, insn_id, domain, check_inames, required_preds, ast):
    from loopy.codegen.bounds import get_bounds_checks
    from loopy.symbolic import constraint_to_expr

    bounds_checks = get_bounds_checks(
            domain, check_inames,
            codegen_state.implemented_domain, overapproximate=False)
    bounds_check_set = isl.Set.universe(domain.get_space()) \
            .add_constraints(bounds_checks)
    bounds_check_set, new_implemented_domain = isl.align_two(
            bounds_check_set, codegen_state.implemented_domain)
    new_implemented_domain = new_implemented_domain & bounds_check_set

    if bounds_check_set.is_empty():
        return None

    condition_exprs = [
            constraint_to_expr(cns)
            for cns in bounds_checks]

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

    from loopy.kernel.data import Assignment, CallInstruction, CInstruction

    if isinstance(insn, Assignment):
        ast = generate_assignment_instruction_code(codegen_state, insn)
    elif isinstance(insn, CallInstruction):
        ast = generate_call_code(codegen_state, insn)
    elif isinstance(insn, CInstruction):
        ast = generate_c_instruction_code(codegen_state, insn)
    else:
        raise RuntimeError("unexpected instruction type")

    insn_inames = kernel.insn_inames(insn)

    return to_codegen_result(
            codegen_state,
            insn.id,
            kernel.get_inames_domain(insn_inames), insn_inames,
            insn.predicates,
            ast)


def generate_assignment_instruction_code(codegen_state, insn):
    kernel = codegen_state.kernel

    ecm = codegen_state.expression_to_code_mapper

    from loopy.expression import dtype_to_type_context, VectorizabilityChecker

    # {{{ vectorization handling

    if codegen_state.vectorization_info:
        if insn.atomicity:
            raise Unvectorizable("atomic operation")

        vinfo = codegen_state.vectorization_info
        vcheck = VectorizabilityChecker(
                kernel, vinfo.iname, vinfo.length)
        lhs_is_vector = vcheck(insn.assignee)
        rhs_is_vector = vcheck(insn.expression)

        if not lhs_is_vector and rhs_is_vector:
            raise Unvectorizable(
                    "LHS is scalar, RHS is vector, cannot assign")

        is_vector = lhs_is_vector

        del lhs_is_vector
        del rhs_is_vector

    # }}}

    from pymbolic.primitives import Variable, Subscript
    from loopy.symbolic import LinearSubscript

    lhs = insn.assignee
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

    lhs_var = kernel.get_var_descriptor(assignee_var_name)
    lhs_dtype = lhs_var.dtype

    if insn.atomicity is not None:
        lhs_atomicity = [
                a for a in insn.atomicity if a.var_name == assignee_var_name]
        assert len(lhs_atomicity) <= 1
        if lhs_atomicity:
            lhs_atomicity, = lhs_atomicity
        else:
            lhs_atomicity = None
    else:
        lhs_atomicity = None

    from loopy.kernel.data import AtomicInit, AtomicUpdate

    lhs_code = ecm(insn.assignee, prec=PREC_NONE, type_context=None)
    rhs_type_context = dtype_to_type_context(kernel.target, lhs_dtype)
    if lhs_atomicity is None:
        result = codegen_state.ast_builder.emit_assignment(
                codegen_state,
                lhs_code,
                ecm(insn.expression, prec=PREC_NONE,
                    type_context=rhs_type_context,
                    needed_dtype=lhs_dtype))

    elif isinstance(lhs_atomicity, AtomicInit):
        raise NotImplementedError("atomic init")

    elif isinstance(lhs_atomicity, AtomicUpdate):
        codegen_state.seen_atomic_dtypes.add(lhs_dtype)
        result = codegen_state.ast_builder.generate_atomic_update(
                kernel, codegen_state, lhs_atomicity, lhs_var,
                insn.assignee, insn.expression,
                lhs_dtype, rhs_type_context)

    else:
        raise ValueError("unexpected lhs atomicity type: %s"
                % type(lhs_atomicity).__name__)

    # {{{ tracing

    if kernel.options.trace_assignments or kernel.options.trace_assignment_values:
        if codegen_state.vectorization_info and is_vector:
            raise Unvectorizable("tracing does not support vectorization")

        from cgen import Statement as S  # noqa

        gs, ls = kernel.get_grid_size_upper_bounds()

        printf_format = "%s.%s[%s][%s]: %s" % (
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
            printf_args_str = ", " + ", ".join(printf_args)
        else:
            printf_args_str = ""

        printf_insn = S("printf(\"%s\\n\"%s)" % (
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
            raise Unvectorizable("function call")

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
        raise Unvectorizable("C instructions cannot be vectorized")

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

    body.extend(Line(l) for l in insn.code.split("\n"))

    return Block(body)


# vim: foldmethod=marker

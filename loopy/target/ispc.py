"""Target for Intel ISPC."""


__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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


from typing import cast, Tuple, Sequence

import numpy as np  # noqa
import pymbolic.primitives as p
from pymbolic import var
from pymbolic.mapper.stringifier import PREC_NONE
from pytools import memoize_method
from cgen import Generable, Declarator, Const, Collection

from loopy.target.c import CFamilyTarget, CFamilyASTBuilder
from loopy.target.c.codegen.expression import ExpressionToCExpressionMapper
from loopy.diagnostic import LoopyError
from loopy.symbolic import Literal
from loopy.schedule import CallKernel
from loopy.typing import ExpressionT
from loopy.types import LoopyType
from loopy.kernel.data import AddressSpace, TemporaryVariable, ArrayArg
from loopy.codegen import CodeGenerationState
from loopy.codegen.result import CodeGenerationResult


# {{{ expression mapper

class ExprToISPCExprMapper(ExpressionToCExpressionMapper):
    def _get_index_ctype(self):
        if self.kernel.index_dtype.numpy_dtype == np.int32:
            return "int32"
        elif self.kernel.index_dtype.numpy_dtype == np.int64:
            return "int64"
        else:
            raise ValueError("unexpected index_type")

    def map_group_hw_index(self, expr, type_context):
        return var(
                "((uniform %s) taskIndex%d)"
                % (self._get_index_ctype(), expr.axis))

    def map_local_hw_index(self, expr, type_context):
        if expr.axis == 0:
            return var("(varying %s) programIndex" % self._get_index_ctype())
        else:
            raise LoopyError("ISPC only supports one local axis")

    def map_constant(self, expr, type_context):
        if isinstance(expr, (complex, np.complexfloating)):
            raise NotImplementedError("complex numbers in ispc")
        else:
            if type_context == "f":
                return Literal(repr(float(expr)))
            elif type_context == "d":
                # Keepin' the good ideas flowin' since '66.
                return Literal(repr(float(expr))+"d")
            elif type_context in ["i", "b"]:
                return expr
            else:
                from loopy.tools import is_integer
                if is_integer(expr):
                    return expr

                raise RuntimeError("don't know how to generate code "
                        "for constant '%s'" % expr)

    def map_variable(self, expr, type_context):
        tv = self.kernel.temporary_variables.get(expr.name)

        if tv is not None and tv.address_space == AddressSpace.PRIVATE:
            # FIXME: This is a pretty coarse way of deciding what
            # private temporaries get duplicated. Refine? (See also
            # below in decl generation)
            _gsize, lsize = self.kernel.get_grid_size_upper_bounds_as_exprs(
                    self.codegen_state.callables_table)
            if lsize:
                return expr[var("programIndex")]
            else:
                return expr

        else:
            return super().map_variable(
                    expr, type_context)

    def map_subscript(self, expr, type_context):
        from loopy.kernel.data import TemporaryVariable

        ary = self.find_array(expr)

        if (isinstance(ary, TemporaryVariable)
                and ary.address_space == AddressSpace.PRIVATE):
            # generate access code for acccess to private-index temporaries

            gsize, lsize = self.kernel.get_grid_size_upper_bounds_as_exprs()
            if lsize:
                lsize, = lsize
                from loopy.kernel.array import get_access_info
                from pymbolic import evaluate

                access_info = get_access_info(self.kernel, ary, expr.index,
                    lambda expr: evaluate(expr, self.codegen_state.var_subst_map),
                    self.codegen_state.vectorization_info)

                subscript, = access_info.subscripts
                result = var(access_info.array_name)[
                        var("programIndex") + self.rec(lsize*subscript, "i")]

                if access_info.vector_index is not None:
                    return self.kernel.target.add_vector_access(
                        result, access_info.vector_index)
                else:
                    return result

        return super().map_subscript(
                expr, type_context)

# }}}


# {{{ type registry

def fill_registry_with_ispc_types(reg, respect_windows, include_bool=True):
    reg.get_or_register_dtype("bool", bool)

    reg.get_or_register_dtype(["int8", "signed char", "char"], np.int8)
    reg.get_or_register_dtype(["uint8", "unsigned char"], np.uint8)
    reg.get_or_register_dtype(["int16", "short", "signed short",
        "signed short int", "short signed int"], np.int16)
    reg.get_or_register_dtype(["uint16", "unsigned short",
        "unsigned short int", "short unsigned int"], np.uint16)
    reg.get_or_register_dtype(["int32", "int", "signed int"], np.int32)
    reg.get_or_register_dtype(["uint32", "unsigned", "unsigned int"], np.uint32)

    reg.get_or_register_dtype(["int64"], np.int64)
    reg.get_or_register_dtype(["uint64"], np.uint64)

    reg.get_or_register_dtype("float", np.float32)
    reg.get_or_register_dtype("double", np.float64)

# }}}


class ISPCTarget(CFamilyTarget):
    """A code generation target for Intel's `ISPC <https://ispc.github.io/>`_
    SPMD programming language, to target Intel's Knight's hardware and modern
    Intel CPUs with wide vector units.
    """

    host_program_name_suffix = ""
    device_program_name_suffix = "_inner"

    def pre_codegen_entrypoint_check(self, kernel, callables_table):
        gsize, lsize = kernel.get_grid_size_upper_bounds_as_exprs(
                callables_table)
        if len(lsize) > 1:
            for ls_i in lsize[1:]:
                if ls_i != 1:
                    raise LoopyError("local axis %d (0-based) "
                            "has length > 1, which is unsupported "
                            "by ISPC" % ls_i)

    def get_host_ast_builder(self):
        return ISPCASTBuilder(self)

    def get_device_ast_builder(self):
        return ISPCASTBuilder(self)

    # {{{ types

    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c.compyte.dtypes import DTypeRegistry
        result = DTypeRegistry()
        fill_registry_with_ispc_types(result, respect_windows=False,
                include_bool=True)
        return result

    # }}}


class ISPCASTBuilder(CFamilyASTBuilder):
    # {{{ top-level codegen

    def get_function_declaration(
            self, codegen_state: CodeGenerationState,
            codegen_result: CodeGenerationResult, schedule_index: int
            ) -> Tuple[Sequence[Tuple[str, str]], Generable]:
        name = codegen_result.current_program(codegen_state).name
        kernel = codegen_state.kernel

        assert codegen_state.kernel.linearization is not None
        subkernel_name = cast(
                        CallKernel,
                        codegen_state.kernel.linearization[schedule_index]
                        ).kernel_name

        from cgen import (FunctionDeclaration, Value)
        from cgen.ispc import ISPCExport, ISPCTask

        if codegen_state.is_entrypoint:
            # subkernel launches occur only as part of entrypoint kernels for now
            from loopy.schedule.tools import get_subkernel_arg_info
            skai = get_subkernel_arg_info(codegen_state.kernel, subkernel_name)
            passed_names = skai.passed_names
            written_names = skai.written_names
        else:
            passed_names = [arg.name for arg in kernel.args]
            written_names = kernel.get_written_variables()

        arg_decls = [self.arg_to_cgen_declarator(
                            kernel, arg_name,
                            is_written=arg_name in written_names)
                        for arg_name in passed_names]

        if codegen_state.is_generating_device_code:
            result = ISPCTask(
                        FunctionDeclaration(
                            Value("void", name),
                            arg_decls))
        else:
            result = ISPCExport(
                    FunctionDeclaration(
                        Value("void", name),
                        arg_decls))

        from loopy.target.c import FunctionDeclarationWrapper
        return [], FunctionDeclarationWrapper(result)

    def get_kernel_call(self, codegen_state: CodeGenerationState,
            subkernel_name: str,
            gsize: Tuple[ExpressionT, ...],
            lsize: Tuple[ExpressionT, ...]) -> Generable:
        kernel = codegen_state.kernel
        ecm = self.get_expression_to_code_mapper(codegen_state)

        from pymbolic.mapper.stringifier import PREC_NONE
        result = []
        from cgen import Statement as S, Block
        if lsize:
            result.append(
                    S(
                        "assert(programCount == (%s))"
                        % ecm(lsize[0], PREC_NONE)))

        if codegen_state.is_entrypoint:
            # subkernel launches occur only as part of entrypoint kernels for now
            from loopy.schedule.tools import get_subkernel_arg_info
            skai = get_subkernel_arg_info(codegen_state.kernel, subkernel_name)
            passed_names = skai.passed_names
        else:
            passed_names = [arg.name for arg in kernel.args]

        from cgen.ispc import ISPCLaunch
        result.append(
                ISPCLaunch(
                    tuple(ecm(gs_i, PREC_NONE) for gs_i in gsize),
                    "{}({})".format(
                        subkernel_name,
                        ", ".join(passed_names)
                        )))

        return Block(result)

    # }}}

    # {{{ code generation guts

    def get_expression_to_c_expression_mapper(self, codegen_state):
        return ExprToISPCExprMapper(codegen_state)

    def add_vector_access(self, access_expr, index):
        return access_expr[index]

    def emit_barrier(self, synchronization_kind, mem_kind, comment):
        from cgen import Comment, Statement

        assert comment

        if synchronization_kind == "local":
            return Comment("local barrier: %s" % comment)

        elif synchronization_kind == "global":
            return Statement("sync; /* %s */" % comment)

        else:
            raise LoopyError("unknown barrier kind")

    # }}}

    # {{{ declarators

    def get_value_arg_declaraotor(
            self, name: str, dtype: LoopyType, is_written: bool) -> Declarator:
        from cgen.ispc import ISPCUniform
        return ISPCUniform(super().get_value_arg_declaraotor(
                name, dtype, is_written))

    def get_array_arg_declarator(
            self, arg: ArrayArg, is_written: bool) -> Declarator:
        # FIXME restrict?
        from cgen.ispc import ISPCUniformPointer, ISPCUniform
        decl = ISPCUniform(
                ISPCUniformPointer(self.get_array_base_declarator(arg)))

        if not is_written:
            decl = Const(decl)

        return decl

    def get_temporary_var_declarator(self,
            codegen_state: CodeGenerationState,
            temp_var: TemporaryVariable) -> Declarator:
        temp_var_decl = self.get_array_base_declarator(temp_var)

        shape = temp_var.shape

        assert isinstance(shape, tuple)

        if temp_var.address_space == AddressSpace.PRIVATE:
            # FIXME: This is a pretty coarse way of deciding what
            # private temporaries get duplicated. Refine? (See also
            # above in expr to code mapper)
            _, lsize = codegen_state.kernel.get_grid_size_upper_bounds_as_exprs(
                    codegen_state.callables_table)
            shape = lsize + shape

        if shape:
            from cgen import ArrayOf
            ecm = self.get_expression_to_code_mapper(codegen_state)
            temp_var_decl = ArrayOf(
                    temp_var_decl,
                    ecm(p.flattened_product(shape),
                        prec=PREC_NONE, type_context="i"))

        return temp_var_decl

    # }}}

    # {{{ emit_...
    def emit_assignment(self, codegen_state, insn):
        kernel = codegen_state.kernel
        ecm = codegen_state.expression_to_code_mapper

        assignee_var_name, = insn.assignee_var_names()

        lhs_var = codegen_state.kernel.get_var_descriptor(assignee_var_name)
        lhs_dtype = lhs_var.dtype

        if insn.atomicity:
            raise NotImplementedError("atomic ops in ISPC")

        from loopy.expression import dtype_to_type_context
        from pymbolic.mapper.stringifier import PREC_NONE

        rhs_type_context = dtype_to_type_context(kernel.target, lhs_dtype)
        rhs_code = ecm(insn.expression, prec=PREC_NONE,
                    type_context=rhs_type_context,
                    needed_dtype=lhs_dtype)

        lhs = insn.assignee

        # {{{ handle streaming stores

        from loopy.kernel.instruction import UseStreamingStoreTag
        if UseStreamingStoreTag() in insn.tags:
            ary = ecm.find_array(lhs)

            from loopy.kernel.array import get_access_info
            from pymbolic import evaluate

            from loopy.symbolic import simplify_using_aff
            index_tuple = tuple(
                    simplify_using_aff(kernel, idx) for idx in lhs.index_tuple)

            access_info = get_access_info(kernel, ary, index_tuple,
                    lambda expr: evaluate(expr, codegen_state.var_subst_map),
                    codegen_state.vectorization_info)

            from loopy.kernel.data import ArrayArg, TemporaryVariable

            if not isinstance(ary, (ArrayArg, TemporaryVariable)):
                raise LoopyError("array type not supported in ISPC: %s"
                        % type(ary).__name)

            if len(access_info.subscripts) != 1:
                raise LoopyError("streaming stores must have a subscript")
            subscript, = access_info.subscripts

            from pymbolic.primitives import Sum, flattened_sum, Variable
            if isinstance(subscript, Sum):
                terms = subscript.children
            else:
                terms = (subscript.children,)

            new_terms = []

            from loopy.kernel.data import LocalInameTag, filter_iname_tags_by_type
            from loopy.symbolic import get_dependencies

            saw_l0 = False
            for term in terms:
                if (isinstance(term, Variable)
                            and kernel.iname_tags_of_type(term.name, LocalInameTag)):
                    tag, = kernel.iname_tags_of_type(
                        term.name, LocalInameTag, min_num=1, max_num=1)
                    if tag.axis == 0:
                        if saw_l0:
                            raise LoopyError(
                                "streaming store must have stride 1 in "
                                "local index, got: %s" % subscript)
                        saw_l0 = True
                        continue
                else:
                    for dep in get_dependencies(term):
                        if dep in kernel.all_inames() and (
                                filter_iname_tags_by_type(kernel.inames[dep].tags,
                                    LocalInameTag)):
                            tag, = filter_iname_tags_by_type(
                                kernel.inames[dep].tags, LocalInameTag, 1)
                            if tag.axis == 0:
                                raise LoopyError(
                                    "streaming store must have stride 1 in "
                                    "local index, got: %s" % subscript)

                    new_terms.append(term)

            if not saw_l0:
                raise LoopyError("streaming store must have stride 1 in "
                        "local index, got: %s" % subscript)

            if access_info.vector_index is not None:
                raise LoopyError("streaming store may not use a short-vector "
                        "data type")

            rhs_has_programindex = any(
                isinstance(tag, LocalInameTag) and tag.axis == 0
                for tag in kernel.iname_tags(dep)
                for dep in get_dependencies(insn.expression))

            if not rhs_has_programindex:
                rhs_code = "broadcast(%s, 0)" % rhs_code

            from cgen import Statement
            return Statement(
                    "streaming_store(%s + %s, %s)"
                    % (
                        access_info.array_name,
                        ecm(flattened_sum(new_terms), PREC_NONE, "i"),
                        rhs_code))

        # }}}

        from cgen import Assign
        return Assign(ecm(lhs, prec=PREC_NONE, type_context=None), rhs_code)

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            lbound, ubound, inner, hints):
        ecm = codegen_state.expression_to_code_mapper

        from loopy.target.c import POD

        from pymbolic.mapper.stringifier import PREC_NONE
        from cgen import For, InlineInitializer

        from cgen.ispc import ISPCUniform

        loop = For(
                InlineInitializer(
                    ISPCUniform(POD(self, iname_dtype, iname)),
                    ecm(lbound, PREC_NONE, "i")),
                ecm(
                    p.Comparison(var(iname), "<=", ubound),
                    PREC_NONE, "i"),
                "++%s" % iname,
                inner)

        if hints:
            return Collection(list(hints) + [loop])
        else:
            return loop

    # }}}


# TODO: Generate launch code
# TODO: Vector types (element access: done)

# vim: foldmethod=marker

"""Target for Intel ISPC."""

from __future__ import division, absolute_import

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


import numpy as np  # noqa
from loopy.target.c import CTarget, CASTBuilder, CExpression
from loopy.target.c.codegen.expression import ExpressionToCExpressionMapper
from loopy.diagnostic import LoopyError
from loopy.symbolic import Literal
from pymbolic import var
import pymbolic.primitives as p
from loopy.kernel.data import temp_var_scope
from pymbolic.mapper.stringifier import PREC_NONE

from pytools import memoize_method


class NewArray(p.Expression):
    """An expression node for allocating a new array."""

    # Only used in host-side code generation.

    init_arg_names = ("type", "size")

    def __init__(self, type, size):
        self.type = type
        self.size = size

    def __getinitargs__(self):
        return (self.type, self.size)

    init_arg_names = ("type", "size")

    mapper_method = "map_new_array"


class DeleteArray(p.Expression):
    """An expression node for deleting an array."""

    # Only used in host-side code generation.

    init_arg_names = ("array",)

    def __init__(self, array):
        self.array = array

    def __getinitargs__(self):
        return (self.array,)

    init_arg_names = ("array",)

    mapper_method = "map_delete_array"


def temporary_should_have_extra_local_axis(kernel, temporary_name):
    # This checks if a private temporary variable should be implemented as
    # having an extra axis or not. If there are no local-parallel tagged writers
    # on it, then we are safe implementing it without the extra axis.
    #
    # Note: this is currently needed as a workaround, because otherwise the
    # compiler will reject some code as mistyped. The problem is that a
    # temporary that is accessed with the programIndex variable will have a
    # varying type. However, the compiler always disallows assignments from
    # varying type to uniform type, even though in our case they may be
    # legitimate and race-free.  In order to make sure that compilable code is
    # always generated, it might be worth trying to case the result of
    # assignments from varying to uniform when a private temporary is involved.
    writers = kernel.writer_map()[temporary_name]
    from loopy.kernel.data import LocalIndexTagBase

    def insn_has_local_tag(id):
        for iname in kernel.id_to_insn[id].within_inames:
            if isinstance(kernel.iname_to_tag.get(iname), LocalIndexTagBase):
                return True
        return False

    if any(insn_has_local_tag(insn) for insn in writers):
        _, lsize = kernel.get_grid_size_upper_bounds_as_exprs()
        return lsize

    return False


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
            elif type_context == "i":
                return expr
            else:
                from loopy.tools import is_integer
                if is_integer(expr):
                    return expr

                raise RuntimeError("don't know how to generate code "
                        "for constant '%s'" % expr)

    def map_variable(self, expr, type_context):
        tv = self.kernel.temporary_variables.get(expr.name)

        if tv is not None and tv.scope == temp_var_scope.PRIVATE:
            # FIXME: This is a pretty coarse way of deciding what
            # private temporaries get duplicated. Refine? (See also
            # below in decl generation)
            if temporary_should_have_extra_local_axis(self.kernel, tv.name):
                return expr[var("programIndex")]
            else:
                return expr

        else:
            return super(ExprToISPCExprMapper, self).map_variable(
                    expr, type_context)

    def map_subscript(self, expr, type_context):
        from loopy.kernel.data import TemporaryVariable

        ary = self.find_array(expr)

        if (isinstance(ary, TemporaryVariable)
                and ary.scope == temp_var_scope.PRIVATE):
            # generate access code for acccess to private-index temporaries

            gsize, lsize = self.kernel.get_grid_size_upper_bounds_as_exprs()
            if lsize:
                lsize, = lsize
                from loopy.kernel.array import get_access_info
                from pymbolic import evaluate

                access_info = get_access_info(self.kernel.target, ary, expr.index,
                    lambda expr: evaluate(expr, self.codegen_state.var_subst_map),
                    self.codegen_state.vectorization_info)

                subscript, = access_info.subscripts
                result = var(access_info.array_name)[
                        var("programIndex") + self.rec(lsize*subscript, 'i')]

                if access_info.vector_index is not None:
                    return self.kernel.target.add_vector_access(
                        result, access_info.vector_index)
                else:
                    return result

        return super(ExprToISPCExprMapper, self).map_subscript(
                expr, type_context)

# }}}


# {{{ type registry

def fill_registry_with_ispc_types(reg, respect_windows, include_bool=True):
    reg.get_or_register_dtype("bool", np.bool)

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


class ISPCTarget(CTarget):
    """A code generation target for Intel's `ISPC <https://ispc.github.io/>`_
    SPMD programming language, to target Intel's Knight's hardware and modern
    Intel CPUs with wide vector units.
    """

    def __init__(self, occa_mode=False):
        """
        :arg occa_mode: Whether to modify the generated call signature to
            be compatible with OCCA
        """
        self.occa_mode = occa_mode

        super(ISPCTarget, self).__init__()

    host_program_name_suffix = ""
    device_program_name_suffix = "_inner"

    def split_kernel_at_global_barriers(self):
        return True

    def has_host_side_global_barriers(self):
        return True

    def pre_codegen_check(self, kernel):
        gsize, lsize = kernel.get_grid_size_upper_bounds_as_exprs()
        if len(lsize) > 1:
            for i, ls_i in enumerate(lsize[1:]):
                if ls_i != 1:
                    raise LoopyError("local axis %d (0-based) "
                            "has length > 1, which is unsupported "
                            "by ISPC" % ls_i)

    def get_host_ast_builder(self):
        return ISPCHostASTBuilder(self)

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


class ISPCASTBuilder(CASTBuilder):
    def _arg_names_and_decls(self, codegen_state, extra_args):
        implemented_data_info = codegen_state.implemented_data_info
        all_args = implemented_data_info + extra_args
        arg_names = [iai.name for iai in all_args]

        arg_decls = [
                self.idi_to_cgen_declarator(codegen_state.kernel, idi)
                for idi in all_args]

        # {{{ occa compatibility hackery

        from cgen import Value
        if self.target.occa_mode:
            from cgen import ArrayOf, Const
            from cgen.ispc import ISPCUniform

            arg_decls = [
                    Const(ISPCUniform(ArrayOf(Value("int", "loopy_dims")))),
                    Const(ISPCUniform(Value("int", "o1"))),
                    Const(ISPCUniform(Value("int", "o2"))),
                    Const(ISPCUniform(Value("int", "o3"))),
                    ] + arg_decls
            arg_names = ["loopy_dims", "o1", "o2", "o3"] + arg_names

        # }}}

        return arg_names, arg_decls

    # {{{ top-level codegen

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        name = codegen_result.current_program(codegen_state).name

        from cgen import (FunctionDeclaration, Value)
        from cgen.ispc import ISPCExport, ISPCTask

        arg_names, arg_decls = self._arg_names_and_decls(codegen_state, [])

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
        return FunctionDeclarationWrapper(result)

    # }}}

    def get_kernel_call(self, codegen_state, name, gsize, lsize, extra_args):
        ecm = self.get_expression_to_code_mapper(codegen_state)

        from pymbolic.mapper.stringifier import PREC_NONE
        result = []
        from cgen import Statement as S, Block
        if lsize:
            # FIXME: not sure
            result.append(
                    S(
                        "assert(programCount >= (%s))"
                        % ecm(lsize[0], PREC_NONE)))

        arg_names, arg_decls = self._arg_names_and_decls(codegen_state, extra_args)

        from cgen.ispc import ISPCLaunch
        result.append(
                ISPCLaunch(
                    tuple(ecm(gs_i, PREC_NONE) for gs_i in gsize),
                    "%s(%s)" % (
                        name,
                        ", ".join(arg_names)
                        )))

        return Block(result)

    # {{{ code generation guts

    def get_expression_to_c_expression_mapper(self, codegen_state):
        return ExprToISPCExprMapper(codegen_state)

    def add_vector_access(self, access_expr, index):
        return access_expr[index]

    def emit_barrier(self, kind, comment):
        from cgen import Comment, Statement

        assert comment

        if kind == "local":
            return Comment("local barrier: %s" % comment)

        elif kind == "global":
            return Statement("sync; /* %s */" % comment)

        else:
            raise LoopyError("unknown barrier kind")

    def get_temporary_decl(self, codegen_state, sched_index, temp_var, decl_info):
        from loopy.target.c import POD  # uses the correct complex type
        temp_var_decl = POD(self, decl_info.dtype, decl_info.name)

        shape = decl_info.shape

        if temp_var.scope == temp_var_scope.PRIVATE:
            if temporary_should_have_extra_local_axis(
                    codegen_state.kernel, temp_var.name):
                # FIXME: This is a pretty coarse way of deciding what
                # private temporaries get duplicated. Refine? (See also
                # above in expr to code mapper)
                _, lsize = codegen_state.kernel.get_grid_size_upper_bounds_as_exprs()
                shape = lsize + shape

        if shape:
            from cgen import ArrayOf
            ecm = self.get_expression_to_code_mapper(codegen_state)
            temp_var_decl = ArrayOf(
                    temp_var_decl,
                    ecm(p.flattened_product(shape),
                        prec=PREC_NONE, type_context="i"))

        return temp_var_decl

    def wrap_temporary_decl(self, decl, scope):
        from cgen.ispc import ISPCUniform
        return ISPCUniform(decl)

    def get_global_arg_decl(self, name, shape, dtype, is_written):
        from loopy.target.c import POD  # uses the correct complex type
        from cgen import Const
        from cgen.ispc import ISPCUniformPointer, ISPCUniform

        arg_decl = ISPCUniformPointer(POD(self, dtype, name))

        if not is_written:
            arg_decl = Const(arg_decl)

        arg_decl = ISPCUniform(arg_decl)

        return arg_decl

    def get_value_arg_decl(self, name, shape, dtype, is_written):
        result = super(ISPCASTBuilder, self).get_value_arg_decl(
                name, shape, dtype, is_written)

        from cgen import Reference, Const
        was_const = isinstance(result, Const)

        if was_const:
            result = result.subdecl

        if self.target.occa_mode:
            result = Reference(result)

        if was_const:
            result = Const(result)

        from cgen.ispc import ISPCUniform
        return ISPCUniform(result)

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

        if "!streaming_store" in insn.tags:
            ary = ecm.find_array(lhs)

            from loopy.kernel.array import get_access_info
            from pymbolic import evaluate

            from loopy.symbolic import simplify_using_aff
            index_tuple = tuple(
                    simplify_using_aff(kernel, idx) for idx in lhs.index_tuple)

            access_info = get_access_info(kernel.target, ary, index_tuple,
                    lambda expr: evaluate(expr, self.codegen_state.var_subst_map),
                    codegen_state.vectorization_info)

            from loopy.kernel.data import GlobalArg, TemporaryVariable

            if not isinstance(ary, (GlobalArg, TemporaryVariable)):
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

            from loopy.kernel.data import LocalIndexTag
            from loopy.symbolic import get_dependencies

            saw_l0 = False
            for term in terms:
                if (isinstance(term, Variable)
                        and isinstance(
                            kernel.iname_to_tag.get(term.name), LocalIndexTag)
                        and kernel.iname_to_tag.get(term.name).axis == 0):
                    if saw_l0:
                        raise LoopyError("streaming store must have stride 1 "
                                "in local index, got: %s" % subscript)
                    saw_l0 = True
                    continue
                else:
                    for dep in get_dependencies(term):
                        if (
                                isinstance(
                                    kernel.iname_to_tag.get(dep), LocalIndexTag)
                                and kernel.iname_to_tag.get(dep).axis == 0):
                            raise LoopyError("streaming store must have stride 1 "
                                    "in local index, got: %s" % subscript)

                    new_terms.append(term)

            if not saw_l0:
                raise LoopyError("streaming store must have stride 1 in "
                        "local index, got: %s" % subscript)

            if access_info.vector_index is not None:
                raise LoopyError("streaming store may not use a short-vector "
                        "data type")

            rhs_has_programindex = any(
                    isinstance(
                        kernel.iname_to_tag.get(dep), LocalIndexTag)
                    and kernel.iname_to_tag.get(dep).axis == 0
                    for dep in get_dependencies(insn.expression))

            if not rhs_has_programindex:
                rhs_code = "broadcast(%s, 0)" % rhs_code

            from cgen import Statement
            return Statement(
                    "streaming_store(%s + %s, %s)"
                    % (
                        access_info.array_name,
                        ecm(flattened_sum(new_terms), PREC_NONE, 'i'),
                        rhs_code))

        # }}}

        from cgen import Assign
        return Assign(ecm(lhs, prec=PREC_NONE, type_context=None), rhs_code)

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            lbound, ubound, inner):
        ecm = codegen_state.expression_to_code_mapper

        from loopy.target.c import POD

        from pymbolic.mapper.stringifier import PREC_NONE
        from cgen import For, InlineInitializer

        from cgen.ispc import ISPCUniform

        return For(
                InlineInitializer(
                    ISPCUniform(POD(self, iname_dtype, iname)),
                    ecm(lbound, PREC_NONE, "i")),
                ecm(
                    p.Comparison(var(iname), "<=", ubound),
                    PREC_NONE, "i"),
                "++%s" % iname,
                inner)
    # }}}


# {{{ host ast builder

class ISPCHostASTBuilder(ISPCASTBuilder):

    # FIXME: Rather than having to override get_temporary_decls /
    # get_function_definition, perhaps it makes more sense to have a
    # get_host_prologue / get_host_epilogue function pair

    def get_function_definition(
            self, codegen_state, codegen_result, schedule_index, function_decl,
            function_body):
        from loopy.kernel.data import temp_var_scope
        import six

        ecm = self.get_c_expression_to_code_mapper()

        global_temporaries = sorted(
                (tv for tv in
                    six.itervalues(codegen_state.kernel.temporary_variables)
                if tv.scope == temp_var_scope.GLOBAL),
                key=lambda tv: tv.name)

        # Add deallocation code.

        from cgen import ExpressionStatement, Line
        # FIXME: Not sure if it's okay to modify the function_body argument.
        function_body.extend([Line()] +
                [ExpressionStatement(
                    CExpression(ecm, DeleteArray(tv.name)))
                 for tv in global_temporaries])

        return ISPCASTBuilder.get_function_definition(
                self,
                codegen_state,
                codegen_result,
                schedule_index,
                function_decl,
                function_body)

    def get_temporary_decls(self, codegen_state, schedule_state):
        from loopy.kernel.data import temp_var_scope
        import six

        global_temporaries = sorted(
                (tv for tv in
                    six.itervalues(codegen_state.kernel.temporary_variables)
                if tv.scope == temp_var_scope.GLOBAL),
                key=lambda tv: tv.name)

        ecm = self.get_c_expression_to_code_mapper()

        from cgen import Assign, Line, dtype_to_ctype
        from six.moves import reduce
        from operator import mul

        assignments = []

        for tv in global_temporaries:
            decl_info = tv.decl_info(
                    self.target, index_dtype=codegen_state.kernel.index_dtype)

            # FIXME: I don't know if this inner loop is necessary.
            for idi in decl_info:
                from loopy.target.c import POD  # uses the correct complex type
                from cgen.ispc import ISPCUniformPointer, ISPCUniform

                decl = ISPCUniform(
                        ISPCUniformPointer(POD(self, idi.dtype, idi.name)))

                assignments.append(decl)
                assignments.append(
                        Assign(idi.name,
                               CExpression(ecm,
                               NewArray(
                                   # FIXME: Not sure how to get this properly
                                   dtype_to_ctype(idi.dtype),
                                   reduce(mul, idi.shape, 1)
                               ))))
                assignments.append(Line())

        return assignments

# }}}

# TODO: Generate launch code
# TODO: Vector types (element access: done)

# vim: foldmethod=marker

"""Python host AST builder for integration with PyOpenCL."""

from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2016 Andreas Kloeckner"

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

import six
import numpy as np

from pymbolic.mapper import Mapper
from pymbolic.mapper.stringifier import StringifyMapper
from loopy.expression import TypeInferenceMapper
from loopy.kernel.data import ValueArg
from loopy.diagnostic import LoopyError  # noqa
from loopy.target import ASTBuilderBase


# {{{ expression to code

class ExpressionToPythonMapper(StringifyMapper):
    def __init__(self, codegen_state, type_inf_mapper=None):
        self.kernel = codegen_state.kernel
        self.codegen_state = codegen_state

        if type_inf_mapper is None:
            type_inf_mapper = TypeInferenceMapper(self.kernel)
        self.type_inf_mapper = type_inf_mapper

    def handle_unsupported_expression(self, victim, enclosing_prec):
        return Mapper.handle_unsupported_expression(self, victim, enclosing_prec)

    def rec(self, expr, prec, type_context=None, needed_dtype=None):
        return super(ExpressionToPythonMapper, self).rec(expr, prec)

    __call__ = rec

    def map_constant(self, expr, enclosing_prec):
        return repr(expr)

    def map_variable(self, expr, enclosing_prec):
        if expr.name in self.codegen_state.var_subst_map:
            # Unimplemented: annotate_inames
            return str(self.rec(
                self.codegen_state.var_subst_map[expr.name],
                enclosing_prec))

        if expr.name in self.kernel.all_inames():
            return super(ExpressionToPythonMapper, self).map_variable(
                    expr, enclosing_prec)

        var_descr = self.kernel.get_var_descriptor(expr.name)
        if isinstance(var_descr, ValueArg):
            return super(ExpressionToPythonMapper, self).map_variable(
                    expr, enclosing_prec)

        return super(ExpressionToPythonMapper, self).map_variable(
                expr, enclosing_prec)

    def map_subscript(self, expr, enclosing_prec):
        return super(ExpressionToPythonMapper, self).map_subscript(
                expr, enclosing_prec)

    def map_call(self, expr, enclosing_prec):
        from pymbolic.primitives import Variable
        from pymbolic.mapper.stringifier import PREC_NONE

        identifier = expr.function

        if identifier.name in ["indexof", "indexof_vec"]:
            raise LoopyError(
                    "indexof, indexof_vec not yet supported in Python")

        if isinstance(identifier, Variable):
            identifier = identifier.name

        par_dtypes = tuple(self.type_inf_mapper(par) for par in expr.parameters)

        str_parameters = None

        mangle_result = self.kernel.mangle_function(
                identifier, par_dtypes,
                ast_builder=self.codegen_state.ast_builder)

        if mangle_result is None:
            raise RuntimeError("function '%s' unknown--"
                    "maybe you need to register a function mangler?"
                    % identifier)

        if len(mangle_result.result_dtypes) != 1:
            raise LoopyError("functions with more or fewer than one return value "
                    "may not be used in an expression")

        str_parameters = [
                self.rec(par, PREC_NONE)
                for par, par_dtype, tgt_dtype in zip(
                    expr.parameters, par_dtypes, mangle_result.arg_dtypes)]

        from loopy.codegen import SeenFunction
        self.codegen_state.seen_functions.add(
                SeenFunction(identifier,
                    mangle_result.target_name,
                    mangle_result.arg_dtypes or par_dtypes))

        return "%s(%s)" % (mangle_result.target_name, ", ".join(str_parameters))

    def map_group_hw_index(self, expr, enclosing_prec):
        raise LoopyError("plain Python does not have group hw axes")

    def map_local_hw_index(self, expr, enclosing_prec):
        raise LoopyError("plain Python does not have local hw axes")

# }}}


# {{{ ast builder

def _numpy_single_arg_function_mangler(kernel, name, arg_dtypes):
    if (not isinstance(name, str)
            or not hasattr(np, name)
            or len(arg_dtypes) != 1):
        return None

    arg_dtype, = arg_dtypes

    from loopy.kernel.data import CallMangleInfo
    return CallMangleInfo(
            target_name="_lpy_np."+name,
            result_dtypes=(arg_dtype,),
            arg_dtypes=arg_dtypes)


def _base_python_preamble_generator(preamble_info):
    yield ("00_future", "from __future__ import division, print_function\n")
    yield ("05_numpy_import", """
            import numpy as _lpy_np
            """)


class PythonASTBuilderBase(ASTBuilderBase):
    """A Python host AST builder for integration with PyOpenCL.
    """

    # {{{ code generation guts

    def function_manglers(self):
        return (
                super(PythonASTBuilderBase, self).function_manglers() + [
                    _numpy_single_arg_function_mangler,
                    ])

    def preamble_generators(self):
        return (
                super(PythonASTBuilderBase, self).preamble_generators() + [
                    _base_python_preamble_generator
                    ])

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        return None

    def get_function_definition(self, codegen_state, codegen_result,
            schedule_index,
            function_decl, function_body):

        assert function_decl is None

        from genpy import Function
        return Function(
                codegen_result.current_program(codegen_state).name,
                [idi.name for idi in codegen_state.implemented_data_info],
                function_body)

    def get_temporary_decls(self, codegen_state, schedule_index):
        kernel = codegen_state.kernel
        ecm = codegen_state.expression_to_code_mapper

        result = []

        from pymbolic.mapper.stringifier import PREC_NONE
        from genpy import Assign

        for tv in sorted(
                six.itervalues(kernel.temporary_variables),
                key=lambda tv: tv.name):
            if tv.shape:
                result.append(
                        Assign(
                            tv.name,
                            "_lpy_np.empty(%s, dtype=%s)"
                            % (
                                ecm(tv.shape, PREC_NONE, "i"),
                                "_lpy_np."+tv.dtype.numpy_dtype.name
                                )))

        return result

    def get_expression_to_code_mapper(self, codegen_state):
        return ExpressionToPythonMapper(codegen_state)

    @property
    def ast_block_class(self):
        from genpy import Suite
        return Suite

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            static_lbound, static_ubound, inner):
        ecm = codegen_state.expression_to_code_mapper

        from loopy.symbolic import aff_to_expr

        from pymbolic.mapper.stringifier import PREC_NONE
        from genpy import For

        return For(
                (iname,),
                "range(%s, %s + 1)"
                % (
                    ecm(aff_to_expr(static_lbound), PREC_NONE, "i"),
                    ecm(aff_to_expr(static_ubound), PREC_NONE, "i"),
                    ),
                inner)

    def emit_initializer(self, codegen_state, dtype, name, val_str, is_const):
        from genpy import Assign
        return Assign(name, val_str)

    def emit_blank_line(self):
        from genpy import Line
        return Line()

    def emit_comment(self, s):
        from genpy import Comment
        return Comment(s)

    def emit_if(self, condition_str, ast):
        from genpy import If
        return If(condition_str, ast)

    def emit_assignment(self, codegen_state, lhs, rhs):
        from genpy import Assign
        return Assign(lhs, rhs)

    # }}}

# }}}

# vim: foldmethod=marker

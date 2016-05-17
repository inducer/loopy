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


from pymbolic.mapper.stringifier import StringifyMapper
from loopy.expression import TypeInferenceMapper
from loopy.kernel.data import ValueArg
from loopy.diagnostic import LoopyError
from loopy.target import ASTBuilderBase


# {{{ expression to code

class ExpressionToPythonMapper(StringifyMapper):
    def __init__(self, codegen_state, type_inf_mapper=None):
        self.kernel = codegen_state.kernel
        self.codegen_state = codegen_state

        if type_inf_mapper is None:
            type_inf_mapper = TypeInferenceMapper(self.kernel)
        self.type_inf_mapper = type_inf_mapper

    def rec(self, expr, prec, type_context=None, needed_dtype=None):
        return super(ExpressionToPythonMapper, self).rec(expr, prec)

    __call__ = rec

    def map_constant(self, expr, enclosing_prec):
        return repr(expr)

    def map_variable(self, expr, enclosing_prec):
        if expr.name in self.kernel.all_inames():
            return super(ExpressionToPythonMapper, self).map_variable(
                    expr, enclosing_prec)

        var_descr = self.kernel.get_var_descriptor(expr.name)
        if isinstance(var_descr, ValueArg):
            return super(ExpressionToPythonMapper, self).map_variable(
                    expr, enclosing_prec)

        raise LoopyError("may not refer to %s '%s' in host code"
                % (type(var_descr).__name__, expr.name))

    def map_subscript(self, expr, enclosing_prec):
        raise LoopyError("may not subscript '%s' in host code"
                % expr.aggregate.name)

# }}}


# {{{ ast builder

class PythonASTBuilderBase(ASTBuilderBase):
    """A Python host AST builder for integration with PyOpenCL.
    """

    # {{{ code generation guts

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

    # }}}

# }}}

# vim: foldmethod=marker

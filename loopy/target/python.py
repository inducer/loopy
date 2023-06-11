"""Python host AST builder for integration with PyOpenCL."""


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

from typing import Tuple, Sequence

from pymbolic.mapper import Mapper
from pymbolic.mapper.stringifier import StringifyMapper
from genpy import Generable, Suite, Collection

from loopy.type_inference import TypeReader
from loopy.kernel.data import ValueArg
from loopy.diagnostic import LoopyError  # noqa
from loopy.target import ASTBuilderBase
from loopy.codegen import CodeGenerationState
from loopy.codegen.result import CodeGenerationResult


# {{{ expression to code

class ExpressionToPythonMapper(StringifyMapper):
    def __init__(self, codegen_state, type_inf_mapper=None):
        self.kernel = codegen_state.kernel
        self.codegen_state = codegen_state

        if type_inf_mapper is None:
            type_inf_mapper = TypeReader(self.kernel,
                    self.codegen_state.callables_table)
        self.type_inf_mapper = type_inf_mapper

    def handle_unsupported_expression(self, victim, enclosing_prec):
        return Mapper.handle_unsupported_expression(self, victim, enclosing_prec)

    def rec(self, expr, prec, type_context=None, needed_dtype=None):
        return super().rec(expr, prec)

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
            return super().map_variable(
                    expr, enclosing_prec)

        var_descr = self.kernel.get_var_descriptor(expr.name)
        if isinstance(var_descr, ValueArg):
            return super().map_variable(
                    expr, enclosing_prec)

        return super().map_variable(
                expr, enclosing_prec)

    def map_subscript(self, expr, enclosing_prec):
        return super().map_subscript(
                expr, enclosing_prec)

    def map_call(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE

        identifier_name = self.codegen_state.callables_table[
                expr.function.name].name

        if identifier_name in ["indexof", "indexof_vec"]:
            raise LoopyError(
                    "indexof, indexof_vec not yet supported in Python")

        clbl = self.codegen_state.callables_table[
                expr.function.name]

        str_parameters = None
        number_of_assignees = len([key for key in
            clbl.arg_id_to_dtype.keys() if key < 0])

        if number_of_assignees != 1:
            raise LoopyError("functions with more or fewer than one return value "
                    "may not be used in an expression")

        str_parameters = [self.rec(par, PREC_NONE) for par in expr.parameters]

        return "{}({})".format(clbl.name_in_target,
                               ", ".join(str_parameters))

    def map_group_hw_index(self, expr, enclosing_prec):
        raise LoopyError("plain Python does not have group hw axes")

    def map_local_hw_index(self, expr, enclosing_prec):
        raise LoopyError("plain Python does not have local hw axes")

    def map_if(self, expr, enclosing_prec):
        # Synthesize PREC_IFTHENELSE, make sure it is in the right place in the
        # operator precedence hierarchy (right above "or").
        from pymbolic.mapper.stringifier import PREC_LOGICAL_OR
        PREC_IFTHENELSE = PREC_LOGICAL_OR - 1  # noqa

        return self.parenthesize_if_needed(
            "{then} if {cond} else {else_}".format(
                # "1 if 0 if 1 else 2 else 3" is not valid Python.
                # So force parens by using an artificially higher precedence.
                then=self.rec(expr.then, PREC_LOGICAL_OR),
                cond=self.rec(expr.condition, PREC_LOGICAL_OR),
                else_=self.rec(expr.else_, PREC_LOGICAL_OR)),
            enclosing_prec, PREC_IFTHENELSE)

# }}}


# {{{ ast builder

def _base_python_preamble_generator(preamble_info):
    yield ("00_future", "from __future__ import division, print_function\n")
    yield ("05_numpy_import", """
            import numpy as _lpy_np
            """)


class PythonASTBuilderBase(ASTBuilderBase[Generable]):
    """A Python host AST builder for integration with PyOpenCL.
    """

    @property
    def known_callables(self):
        from loopy.target.c import get_c_callables
        callables = super().known_callables
        callables.update(get_c_callables())
        return callables

    def preamble_generators(self):
        return (
                super().preamble_generators() + [
                    _base_python_preamble_generator
                    ])

    # {{{ code generation guts

    @property
    def ast_module(self):
        import genpy
        return genpy

    def get_function_declaration(
            self, codegen_state: CodeGenerationState,
            codegen_result: CodeGenerationResult, schedule_index: int
            ) -> Tuple[Sequence[Tuple[str, str]], None]:
        return [], None

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
                kernel.temporary_variables.values(),
                key=lambda key_tv: key_tv.name):
            if tv.shape:
                result.append(
                        Assign(
                            tv.name,
                            "_lpy_np.empty(%s, dtype=%s)"
                            % (
                                ecm(tv.shape, PREC_NONE, "i"),
                                "_lpy_np."+(
                                    tv.dtype.numpy_dtype.name
                                    if tv.dtype.numpy_dtype.name != "bool"
                                    else "bool_")
                                )))

        return result

    def get_expression_to_code_mapper(self, codegen_state):
        return ExpressionToPythonMapper(codegen_state)

    @property
    def ast_block_class(self):
        return Suite

    @property
    def ast_block_scope_class(self):
        # Once a new version of genpy is released, switch to this:
        # from genpy import Collection
        # and delete the implementation above.
        return Collection

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            lbound, ubound, inner, hints):
        ecm = codegen_state.expression_to_code_mapper

        from pymbolic.mapper.stringifier import PREC_NONE, PREC_SUM
        from genpy import For

        if hints:
            raise ValueError("hints for python loops not supported")

        return For(
                (iname,),
                "range(%s, %s + 1)"
                % (
                    ecm(lbound, PREC_NONE, "i"),
                    ecm(ubound, PREC_SUM, "i"),
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

    def emit_noop_with_comment(self, s):
        from cgen import Line
        return Line(f"pass #{s}")

    @property
    def can_implement_conditionals(self):
        return True

    def emit_if(self, condition_str, ast):
        from genpy import If
        return If(condition_str, ast)

    def emit_assignment(self, codegen_state, insn):
        ecm = codegen_state.expression_to_code_mapper

        if insn.atomicity:
            raise NotImplementedError("atomic ops in Python")

        from pymbolic.mapper.stringifier import PREC_NONE
        from genpy import Assign

        return Assign(
                ecm(insn.assignee, prec=PREC_NONE, type_context=None),
                ecm(insn.expression, prec=PREC_NONE, type_context=None))

    # }}}

# }}}

# vim: foldmethod=marker

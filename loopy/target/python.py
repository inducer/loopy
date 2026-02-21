"""Python host AST builder for integration with PyOpenCL."""
from __future__ import annotations


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

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from typing_extensions import override

from genpy import Collection, Generable, Suite
from pymbolic.mapper import Mapper
from pymbolic.mapper.stringifier import PREC_NONE, StringifyMapper

from loopy.diagnostic import LoopyError
from loopy.kernel.data import ValueArg
from loopy.kernel.function_interface import ScalarCallable
from loopy.target import ASTBuilderBase
from loopy.type_inference import TypeReader
from loopy.typing import InameStr, not_none


if TYPE_CHECKING:
    from collections.abc import Sequence

    import pymbolic.primitives as p
    from pymbolic import Expression, ExpressionNode

    from loopy.codegen import CodeGenerationState, PreambleInfo
    from loopy.codegen.result import CodeGenerationResult
    from loopy.kernel import LoopKernel
    from loopy.kernel.instruction import Assignment
    from loopy.symbolic import (
        GroupHardwareAxisIndex,
        LocalHardwareAxisIndex,
        ResolvedFunction,
    )
    from loopy.types import LoopyType


# {{{ expression to code

class ExpressionToPythonMapper(StringifyMapper[[]]):
    def __init__(self,
                codegen_state: CodeGenerationState,
                type_inf_mapper: TypeReader | None = None):
        self.kernel: LoopKernel = codegen_state.kernel
        self.codegen_state: CodeGenerationState = codegen_state

        if type_inf_mapper is None:
            type_inf_mapper = TypeReader(self.kernel,
                    self.codegen_state.callables_table)
        self.type_inf_mapper: TypeReader = type_inf_mapper

    @override
    def handle_unsupported_expression(self, expr: ExpressionNode, enclosing_prec: int):
        return Mapper.handle_unsupported_expression(self, expr, enclosing_prec)

    @override
    def rec(self,
            expr: Expression,
            prec: int,
            type_context: str | None = None,
            needed_dtype: LoopyType | None = None):
        return super().rec(expr, prec)

    @override
    def __call__(self,
            expr: Expression,
            prec: int = PREC_NONE,
            *, type_context: str | None = None,
            needed_dtype: LoopyType | None = None):
        return super().rec(expr, prec)

    @override
    def map_constant(self, expr: object, enclosing_prec: int):
        if isinstance(expr, np.generic):
            return repr(expr).replace("np.", "_lpy_np.")
        else:
            return repr(expr)

    @override
    def map_variable(self, expr: p.Variable, enclosing_prec: int):
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

    @override
    def map_subscript(self, expr: p.Subscript, enclosing_prec: int):
        return super().map_subscript(
                expr, enclosing_prec)

    @override
    def map_call(self, expr: p.Call, enclosing_prec: int):
        from pymbolic.mapper.stringifier import PREC_NONE

        func = cast("p.Variable | ResolvedFunction", expr.function)
        clbl = self.codegen_state.callables_table[func.name]
        identifier_name = clbl.name

        if identifier_name in ["indexof", "indexof_vec"]:
            raise LoopyError(
                    "indexof, indexof_vec not yet supported in Python")

        str_parameters = None
        number_of_assignees = len([
                                  key for key in not_none(clbl.arg_id_to_dtype)
                                  if cast("int", key) < 0])

        if number_of_assignees != 1:
            raise LoopyError("functions with more or fewer than one return value "
                    "may not be used in an expression")

        str_parameters = [self.rec(par, PREC_NONE) for par in expr.parameters]

        assert isinstance(clbl, ScalarCallable)
        return "{}({})".format(clbl.name_in_target,
                               ", ".join(str_parameters))

    def map_group_hw_index(self, expr: GroupHardwareAxisIndex, enclosing_prec: int):
        raise LoopyError("plain Python does not have group hw axes")

    def map_local_hw_index(self, expr: LocalHardwareAxisIndex, enclosing_prec: int):
        raise LoopyError("plain Python does not have local hw axes")

    @override
    def map_if(self, expr: p.If, enclosing_prec: int):
        # Synthesize PREC_IFTHENELSE, make sure it is in the right place in the
        # operator precedence hierarchy (right above "or").
        from pymbolic.mapper.stringifier import PREC_LOGICAL_OR
        PREC_IFTHENELSE = PREC_LOGICAL_OR - 1  # noqa: N806

        then_ = self.rec(expr.then, PREC_LOGICAL_OR)
        cond_ = self.rec(expr.condition, PREC_LOGICAL_OR)
        else_ = self.rec(expr.else_, PREC_LOGICAL_OR)
        return self.parenthesize_if_needed(
            f"{then_} if {cond_} else {else_}",
            # "1 if 0 if 1 else 2 else 3" is not valid Python.
            # So force parens by using an artificially higher precedence.
            enclosing_prec, PREC_IFTHENELSE)

# }}}


# {{{ ast builder

def _base_python_preamble_generator(preamble_info: PreambleInfo):
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
                [*super().preamble_generators(), _base_python_preamble_generator])

    # {{{ code generation guts

    @property
    @override
    def ast_module(self):
        import genpy
        return genpy

    @override
    def get_function_declaration(
            self, codegen_state: CodeGenerationState,
            codegen_result: CodeGenerationResult[Any], schedule_index: int
            ) -> tuple[Sequence[tuple[str, str]], Generable | None]:
        return [], None

    @override
    def get_temporary_decls(self,
                codegen_state: CodeGenerationState,
                schedule_index: int) -> Sequence[Generable]:
        kernel = codegen_state.kernel
        ecm = codegen_state.expression_to_code_mapper

        from genpy import Assign
        from pymbolic.mapper.stringifier import PREC_NONE

        return [Assign(
                            tv.name,
                            "_lpy_np.empty(%s, dtype=%s)"
                            % (
                                ecm(tv.shape, PREC_NONE, "i"),
                                "_lpy_np."+(
                                    tv.dtype.numpy_dtype.name
                                    if tv.dtype.numpy_dtype.name != "bool"
                                    else "bool_")
                                )) for tv in sorted(
                kernel.temporary_variables.values(),
                key=lambda key_tv: key_tv.name) if tv.shape]

    @override
    def get_expression_to_code_mapper(self, codegen_state: CodeGenerationState):
        return ExpressionToPythonMapper(codegen_state)

    @property
    @override
    def ast_block_class(self):
        return Suite

    @property
    @override
    def ast_block_scope_class(self):
        # Once a new version of genpy is released, switch to this:
        # from genpy import Collection
        # and delete the implementation above.
        return Collection

    @override
    def emit_sequential_loop(self,
                codegen_state: CodeGenerationState,
                iname: InameStr,
                iname_dtype: LoopyType,
                lbound: Expression,
                ubound: Expression,
                inner: Generable,
                hints: Sequence[Generable],
            ):
        ecm = self.get_expression_to_code_mapper(codegen_state)

        from genpy import For
        from pymbolic.mapper.stringifier import PREC_NONE, PREC_SUM

        if hints:
            raise ValueError("hints for python loops not supported")

        return For(
                (iname,),
                "range(%s, %s + 1)"
                % (
                    ecm(lbound, PREC_NONE, type_context="i"),
                    ecm(ubound, PREC_SUM, type_context="i"),
                    ),
                inner)

    @override
    def emit_initializer(self,
                codegen_state: CodeGenerationState,
                dtype: LoopyType,
                name: str,
                val_str: str,
                is_const: bool):
        from genpy import Assign
        return Assign(name, val_str)

    @override
    def emit_blank_line(self):
        from genpy import Line
        return Line()

    @override
    def emit_comment(self, s: str):
        from genpy import Comment
        return Comment(s)

    @override
    def emit_noop_with_comment(self, s: str):
        from cgen import Line
        return Line(f"pass #{s}")

    @property
    @override
    def can_implement_conditionals(self):
        return True

    @override
    def emit_if(self, condition_str: str, ast: Generable):
        from genpy import If
        return If(condition_str, ast)

    @override
    def emit_assignment(self, codegen_state: CodeGenerationState, insn: Assignment):
        ecm = codegen_state.expression_to_code_mapper

        if insn.atomicity:
            raise NotImplementedError("atomic ops in Python")

        from genpy import Assign
        from pymbolic.mapper.stringifier import PREC_NONE

        return Assign(
                ecm(insn.assignee, prec=PREC_NONE, type_context=None),
                ecm(insn.expression, prec=PREC_NONE, type_context=None))

    # }}}

# }}}

# vim: foldmethod=marker

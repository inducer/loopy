"""
.. currentmodule:: loopy

.. autoclass:: TargetBase
.. autoclass:: ASTBuilderBase
.. autoclass:: CFamilyTarget
.. autoclass:: CTarget
.. autoclass:: ExecutableCTarget
.. autoclass:: CudaTarget
.. autoclass:: OpenCLTarget
.. autoclass:: PyOpenCLTarget
.. autoclass:: ISPCTarget

References to Canonical Names
-----------------------------

.. currentmodule:: loopy.target

.. class:: TargetBase

    See :class:`loopy.TargetBase`.
"""

from __future__ import annotations

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


from typing import (Any, Tuple, Generic, TypeVar, Sequence, ClassVar, Optional,
        TYPE_CHECKING)

if TYPE_CHECKING:
    from loopy.typing import ExpressionT
    from loopy.codegen import CodeGenerationState
    from loopy.codegen.result import CodeGenerationResult
    from loopy.target.execution import ExecutorBase
    from loopy.translation_unit import TranslationUnit, FunctionIdT


ASTType = TypeVar("ASTType")


class TargetBase:
    """Base class for all targets, i.e. different combinations of code that
    loopy can generate.

    Objects of this type must be picklable.
    """

    # {{{ hashing/equality

    hash_fields: ClassVar[Tuple[str, ...]] = ()
    comparison_fields: ClassVar[Tuple[str, ...]] = ()

    def __hash__(self):
        # NOTE: _hash_value may vanish during pickling
        if getattr(self, "_hash_value", None) is None:
            from loopy.tools import LoopyKeyBuilder
            key_hash = LoopyKeyBuilder.new_hash()
            LoopyKeyBuilder()(self)
            object.__setattr__(self, "_hash_value", hash(key_hash.digest()))

        return self._hash_value  # pylint: disable=no-member

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode())
        for field_name in self.hash_fields:
            key_builder.rec(key_hash, getattr(self, field_name))

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        for field_name in self.comparison_fields:
            if getattr(self, field_name) != getattr(other, field_name):
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    # }}}

    # {{{ preprocess

    def preprocess(self, kernel):
        return kernel

    def pre_codegen_entrypoint_check(self, kernel, callables_table):
        pass

    def pre_codegen_callable_check(self, kernel, callables_table):
        pass

    # }}}

    host_program_name_prefix = ""
    host_program_name_suffix = "_outer"
    device_program_name_prefix = ""
    device_program_name_suffix = ""

    def split_kernel_at_global_barriers(self) -> bool:
        """
        :returns: a :class:`bool` indicating whether the kernel should
            be split when a global barrier is encountered.
        """
        raise NotImplementedError()

    def get_host_ast_builder(self) -> ASTBuilderBase[Any]:
        """
        :returns: a class implementing :class:`ASTBuilderBase` for the host code
        """
        raise NotImplementedError()

    def get_device_ast_builder(self) -> ASTBuilderBase[Any]:
        """
        :returns: a class implementing :class:`ASTBuilderBase` for the host code
        """
        raise NotImplementedError()

    # {{{ types

    def get_dtype_registry(self):
        raise NotImplementedError()

    def is_vector_dtype(self, dtype):
        raise NotImplementedError()

    def vector_dtype(self, base, count):
        raise NotImplementedError()

    def alignment_requirement(self, type_decl):
        import struct
        return struct.calcsize(type_decl.struct_format())

    # }}}

    def get_kernel_executor_cache_key(self, *args, **kwargs):
        """
        :returns: an immutable type to be used as the cache key for
            kernel executor caching.
        """
        raise NotImplementedError()

    def get_kernel_executor(
            self, t_unit: TranslationUnit, *args, entrypoint: FunctionIdT,
            **kwargs) -> ExecutorBase:
        """
        :returns: an immutable type to be used as the cache key for
            kernel executor caching.
        """
        raise NotImplementedError()


class ASTBuilderBase(Generic[ASTType]):
    """An interface for generating (host or device) ASTs.
    """

    def __init__(self, target) -> None:
        self.target = target

    # {{{ library

    @property
    def known_callables(self):
        """
        Returns a mapping from function ids to corresponding
        :class:`loopy.kernel.function_interface.InKernelCallable` for the
        function ids known to *self.target*.
        """
        # FIXME: @inducer: Do we need to move this to TargetBase?
        return {}

    def symbol_manglers(self):
        return []

    def preamble_generators(self):
        return []

    # }}}

    # {{{ code generation guts

    @property
    def ast_module(self):
        raise NotImplementedError()

    def get_function_definition(
            self, codegen_state: CodeGenerationState,
            codegen_result: CodeGenerationResult,
            schedule_index: int, function_decl: ASTType, function_body: ASTType
            ) -> ASTType:
        raise NotImplementedError

    def get_function_declaration(
            self, codegen_state: CodeGenerationState,
            codegen_result: CodeGenerationResult, schedule_index: int
            ) -> Tuple[Sequence[Tuple[str, str]], ASTType]:
        """Returns preambles and the AST for the function declaration."""
        raise NotImplementedError

    def generate_top_of_body(
            self, codegen_state: CodeGenerationState) -> Sequence[ASTType]:
        return []

    def get_temporary_decls(self, codegen_state: CodeGenerationState,
            schedule_index: int) -> ASTType:
        raise NotImplementedError

    def get_kernel_call(self, codegen_state: CodeGenerationState,
            subkernel_name: str,
            gsize: Tuple[ExpressionT, ...],
            lsize: Tuple[ExpressionT, ...]) -> Optional[ASTType]:
        raise NotImplementedError()

    @property
    def ast_block_class(self):
        raise NotImplementedError()

    @property
    def ast_block_scope_class(self):
        raise NotImplementedError()

    def get_expression_to_code_mapper(self, codegen_state: CodeGenerationState):
        raise NotImplementedError()

    def add_vector_access(self, access_expr, index):
        raise NotImplementedError()

    def emit_barrier(self, synchronization_kind, mem_kind, comment):
        """
        :arg synchronization_kind: ``"local"`` or ``"global"``
        :arg mem_kind: ``"local"`` or ``"global"``
        """
        raise NotImplementedError()

    def emit_assignment(self, codegen_state, insn):
        raise NotImplementedError()

    def emit_multiple_assignment(self, codegen_state, insn):
        raise NotImplementedError()

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            static_lbound, static_ubound, inner, hints):
        raise NotImplementedError()

    def emit_unroll_hint(self, value):
        raise NotImplementedError()

    @property
    def can_implement_conditionals(self):
        return False

    def emit_if(self, condition_str, ast):
        raise NotImplementedError()

    def emit_initializer(self, codegen_state, dtype, name, val_str, is_const):
        raise NotImplementedError()

    def emit_declaration_scope(self, codegen_state, inner):
        raise NotImplementedError()

    def emit_blank_line(self):
        raise NotImplementedError()

    def emit_comment(self, s):
        raise NotImplementedError()

    def emit_noop_with_comment(self, s):
        raise NotImplementedError()

    # }}}

    def process_ast(self, node):
        return node


# {{{ dummy host ast builder

class _DummyExpressionToCodeMapper:
    def rec(self, expr, prec, type_context=None, needed_dtype=None):
        return ""

    __call__ = rec


class _DummyASTBlock:
    def __init__(self, arg):
        self.contents = []

    def __str__(self):
        return ""


class DummyHostASTBuilder(ASTBuilderBase[None]):
    def get_function_definition(self, codegen_state, codegen_result,
            schedule_index, function_decl, function_body):
        return function_body

    def get_function_declaration(
            self, codegen_state, codegen_result,
            schedule_index,
            ) -> Tuple[Sequence[Tuple[str, str]], None]:
        return [], None

    def get_temporary_decls(self, codegen_state, schedule_index):
        return []

    def get_expression_to_code_mapper(self, codegen_state):
        return _DummyExpressionToCodeMapper()

    def get_kernel_call(self, codegen_state, name, gsize, lsize):
        return None

    @property
    def ast_block_class(self):
        return _DummyASTBlock

    @property
    def ast_block_scope_class(self):
        return _DummyASTBlock

# }}}


# vim: foldmethod=marker

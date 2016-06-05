"""Base target interface."""

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

__doc__ = """

.. currentmodule:: loopy

.. autoclass:: TargetBase
.. autoclass:: ASTBuilderBase

.. autoclass:: CTarget
.. autoclass:: CudaTarget
.. autoclass:: OpenCLTarget
.. autoclass:: PyOpenCLTarget
.. autoclass:: ISPCTarget

"""


class TargetBase(object):
    """Base class for all targets, i.e. different combinations of code that
    loopy can generate.

    Objects of this type must be picklable.
    """

    # {{{ persistent hashing

    hash_fields = ()
    comparison_fields = ()

    def update_persistent_hash(self, key_hash, key_builder):
        for field_name in self.hash_fields:
            key_builder.rec(key_hash, getattr(self, field_name))

    def __eq__(self, other):
        if type(self) != type(other):
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

    def pre_codegen_check(self, kernel):
        pass

    # }}}

    host_program_name_prefix = ""
    host_program_name_suffix = "_outer"
    device_program_name_prefix = ""
    device_program_name_suffix = ""

    def split_kernel_at_global_barriers(self):
        """
        :returns: a :class:`bool` indicating whether the kernel should
            be split when a global barrier is encountered.
        """
        raise NotImplementedError()

    def get_host_ast_builder(self):
        """
        :returns: a class implementing :class:`ASTBuilderBase` for the host code
        """
        raise NotImplementedError()

    def get_device_ast_builder(self):
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


class ASTBuilderBase(object):
    """An interface for generating (host or device) ASTs.
    """

    def __init__(self, target):
        self.target = target

    # {{{ library

    def function_manglers(self):
        return []

    def symbol_manglers(self):
        return []

    def preamble_generators(self):
        return []

    # }}}

    # {{{ code generation guts

    def get_function_definition(self, codegen_state, codegen_result,
            schedule_index, function_decl, function_body):
        raise NotImplementedError

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        raise NotImplementedError

    def generate_top_of_body(self, codegen_state):
        return []

    def get_temporary_decls(self, codegen_state, schedule_index):
        raise NotImplementedError

    def get_kernel_call(self, codegen_state, name, gsize, lsize, extra_args):
        raise NotImplementedError

    @property
    def ast_block_class(self):
        raise NotImplementedError()

    def get_expression_to_code_mapper(self, codegen_state):
        raise NotImplementedError()

    def add_vector_access(self, access_str, index):
        raise NotImplementedError()

    def emit_barrier(self, kind, comment):
        """
        :arg kind: ``"local"`` or ``"global"``
        """
        raise NotImplementedError()

    def get_global_arg_decl(self, name, shape, dtype, is_written):
        raise NotImplementedError()

    def get_image_arg_decl(self, name, shape, num_target_axes, dtype, is_written):
        raise NotImplementedError()

    def emit_assignment(self, codegen_state, lhs, rhs):
        raise NotImplementedError()

    def emit_multiple_assignment(self, codegen_state, insn):
        raise NotImplementedError()

    def emit_atomic_update(self, kernel, codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype):
        raise NotImplementedError("atomic update in target %s" % type(self).__name__)

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            static_lbound, static_ubound, inner):
        raise NotImplementedError()

    def emit_if(self, condition_str, ast):
        raise NotImplementedError()

    def emit_initializer(self, codegen_state, dtype, name, val_str, is_const):
        raise NotImplementedError()

    def emit_blank_line(self):
        raise NotImplementedError()

    def emit_comment(self, s):
        raise NotImplementedError()

    # }}}


# {{{ dummy host ast builder

class _DummyExpressionToCodeMapper(object):
    def rec(self, expr, prec, type_context=None, needed_dtype=None):
        return ""

    __call__ = rec


class _DummyASTBlock(object):
    def __init__(self, arg):
        self.contents = []

    def __str__(self):
        return ""


class DummyHostASTBuilder(ASTBuilderBase):
    def get_function_definition(self, codegen_state, codegen_result,
            schedule_index, function_decl, function_body):
        return function_body

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        return None

    def get_temporary_decls(self, codegen_state, schedule_index):
        return []

    def get_expression_to_code_mapper(self, codegen_state):
        return _DummyExpressionToCodeMapper()

    def get_kernel_call(self, codegen_state, name, gsize, lsize, extra_args):
        return None

    @property
    def ast_block_class(self):
        return _DummyASTBlock

    def emit_assignment(self, codegen_state, lhs, rhs):
        return None

    def emit_multiple_assignment(self, codegen_state, insn):
        return None

    def emit_atomic_update(self, kernel, codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype):
        return None

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            static_lbound, static_ubound, inner):
        return None

    def emit_if(self, condition_str, ast):
        return None

    def emit_initializer(self, codegen_state, dtype, name, val_str, is_const):
        return None

    def emit_blank_line(self):
        return None

    def emit_comment(self, s):
        return None

# }}}


# vim: foldmethod=marker

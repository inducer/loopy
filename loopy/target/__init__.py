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


class TargetBase(object):
    """Objects of this type must be picklable."""

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

    # {{{ library

    def function_manglers(self):
        return []

    def symbol_manglers(self):
        return []

    def preamble_generators(self):
        return []

    # }}}

    # {{{ top-level codegen

    def preprocess(self, kernel):
        return kernel

    def pre_codegen_check(self, kernel):
        pass

    def generate_code(self, kernel, codegen_state, impl_arg_info):
        pass

    # }}}

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

    # {{{ code generation guts

    def get_expression_to_code_mapper(self, codegen_state):
        raise NotImplementedError()

    def add_vector_access(self, access_str, index):
        raise NotImplementedError()

    def emit_barrier(self, kind, comment):
        """
        :arg kind: ``"local"`` or ``"global"``
        :return: a :class:`loopy.codegen.GeneratedInstruction`.
        """
        raise NotImplementedError()

    def get_global_arg_decl(self, name, shape, dtype, is_written):
        raise NotImplementedError()

    def get_image_arg_decl(self, name, shape, num_target_axes, dtype, is_written):
        raise NotImplementedError()

    # }}}

# vim: foldmethod=marker

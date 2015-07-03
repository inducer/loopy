"""OpenCL target independent of PyOpenCL."""

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

import six

import numpy as np  # noqa
from loopy.target import TargetBase

from pytools import memoize_method


class CTarget(TargetBase):
    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c.compyte.dtypes import (
                DTypeRegistry, fill_with_registry_with_c_types)
        result = DTypeRegistry()
        fill_with_registry_with_c_types(result, respect_windows=False,
                include_bool=True)
        return result

    def is_vector_dtype(self, dtype):
        return False

    def get_vector_dtype(self, base, count):
        raise KeyError()

    def get_or_register_dtype(self, names, dtype=None):
        return self.get_dtype_registry().get_or_register_dtype(names, dtype)

    def dtype_to_typename(self, dtype):
        return self.get_dtype_registry().dtype_to_ctype(dtype)

    def get_expression_to_code_mapper(self, codegen_state):
        from loopy.target.c.codegen.expression import LoopyCCodeMapper
        return LoopyCCodeMapper(codegen_state)

    # {{{ code generation

    def generate_code(self, kernel, codegen_state, impl_arg_info):
        from cgen import FunctionBody, FunctionDeclaration, Value, Module

        body, implemented_domains = kernel.target.generate_body(
                kernel, codegen_state)

        mod = Module([
            FunctionBody(
                kernel.target.wrap_function_declaration(
                    kernel,
                    FunctionDeclaration(
                        Value("void", kernel.name),
                        [iai.cgen_declarator for iai in impl_arg_info])),
                body)
            ])

        return str(mod), implemented_domains

    def wrap_function_declaration(self, kernel, fdecl):
        return fdecl

    def generate_body(self, kernel, codegen_state):
        from cgen import Block
        body = Block()

        # {{{ declare temporaries

        body.extend(
                idi.cgen_declarator
                for tv in six.itervalues(kernel.temporary_variables)
                for idi in tv.decl_info(
                    kernel.target,
                    is_written=True, index_dtype=kernel.index_dtype))

        # }}}

        from loopy.codegen.loop import set_up_hw_parallel_loops
        gen_code = set_up_hw_parallel_loops(kernel, 0, codegen_state)

        from cgen import Line
        body.append(Line())

        if isinstance(gen_code.ast, Block):
            body.extend(gen_code.ast.contents)
        else:
            body.append(gen_code.ast)

        return body, gen_code.implemented_domains

    # }}}

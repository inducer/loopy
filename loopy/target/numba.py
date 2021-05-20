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


from pytools import memoize_method

from loopy.target.python import ExpressionToPythonMapper, PythonASTBuilderBase
from loopy.target import TargetBase, DummyHostASTBuilder

from loopy.diagnostic import LoopyWarning


# {{{ base numba

def _base_numba_preamble_generator(preamble_info):
    yield ("06_numba_imports", """
            import numba as _lpy_numba
            """)


class NumbaBaseASTBuilder(PythonASTBuilderBase):
    def preamble_generators(self):
        return (
                super().preamble_generators() + [
                    _base_numba_preamble_generator
                    ])

    def get_function_definition(self, kernel, name, implemented_data_info,
                                function_decl, function_body):

        assert function_decl is None

        from genpy import Function
        return Function(
                name,
                [idi.name for idi in implemented_data_info],
                function_body,
                decorators=self.get_python_function_decorators())

    def get_python_function_decorators(self):
        return ()

    def get_kernel_call(self, kernel, name, implemented_data_info, extra_args):
        ecm = self.get_expression_to_code_mapper(kernel)
        from pymbolic.mapper.stringifier import PREC_NONE
        from genpy import Statement

        ecm = self.get_expression_to_code_mapper(kernel)
        implemented_data_info = implemented_data_info

        from loopy.schedule.tree import get_insns_in_function
        gsize, lsize = kernel.get_grid_sizes_for_insn_ids_as_exprs(
            get_insns_in_function(kernel, name))

        return Statement(
            "{}[{}, {}]({})".format(
                name,
                ecm(gsize, PREC_NONE),
                ecm(lsize, PREC_NONE),
                ", ".join(idi.name for idi in implemented_data_info)
                ))


class NumbaJITASTBuilder(NumbaBaseASTBuilder):
    def get_python_function_decorators(self):
        return ("@_lpy_numba.jit",)


class NumbaTarget(TargetBase):
    """A target for plain Python as understood by Numba, without any parallel extensions.
    """

    def __init__(self):
        from warnings import warn
        warn("The Numba targets are not yet feature-complete",
                LoopyWarning, stacklevel=2)

    def split_kernel_at_global_barriers(self):
        return False

    def get_host_ast_builder(self):
        return DummyHostASTBuilder(self)

    def get_device_ast_builder(self):
        return NumbaJITASTBuilder(self)

    # {{{ types

    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c import DTypeRegistryWrapper
        from loopy.target.c.compyte.dtypes import (
                DTypeRegistry, fill_registry_with_c_types)
        result = DTypeRegistry()
        fill_registry_with_c_types(result, respect_windows=False,
                include_bool=True)
        return DTypeRegistryWrapper(result)

    def is_vector_dtype(self, dtype):
        return False

    def get_vector_dtype(self, base, count):
        raise KeyError()

    def get_or_register_dtype(self, names, dtype=None):
        # These kind of shouldn't be here.
        return self.get_dtype_registry().get_or_register_dtype(names, dtype)

    def dtype_to_typename(self, dtype):
        # These kind of shouldn't be here.
        return self.get_dtype_registry().dtype_to_ctype(dtype)

    # }}}

# }}}


# {{{ numba.cuda

class NumbaCudaExpressionToPythonMapper(ExpressionToPythonMapper):
    _GRID_AXES = "xyz"

    def map_group_hw_index(self, expr, enclosing_prec):
        return "_lpy_ncu.blockIdx.%s" % self._GRID_AXES[expr.axis]

    def map_local_hw_index(self, expr, enclosing_prec):
        return "_lpy_ncu.threadIdx.%s" % self._GRID_AXES[expr.axis]


def _cuda_numba_preamble_generator(preamble_info):
    yield ("06_import_numba_cuda", """
            import numba.cuda as _lpy_ncu
            """)


class NumbaCudaASTBuilder(NumbaBaseASTBuilder):
    def preamble_generators(self):
        return (
                super().preamble_generators() + [
                    _cuda_numba_preamble_generator
                    ])

    def get_python_function_decorators(self):
        return ("@_lpy_ncu.jit",)

    def get_expression_to_code_mapper(self, codegen_state):
        return NumbaCudaExpressionToPythonMapper(codegen_state)


class NumbaCudaTarget(TargetBase):
    """A target for Numba with CUDA extensions.
    """

    host_program_name_suffix = ""
    device_program_name_suffix = "_inner"

    def __init__(self):
        from warnings import warn
        warn("The Numba target is not yet feature-complete",
                LoopyWarning, stacklevel=2)

    def split_kernel_at_global_barriers(self):
        return True

    def get_host_ast_builder(self):
        return NumbaBaseASTBuilder(self)

    def get_device_ast_builder(self):
        return NumbaCudaASTBuilder(self)

    # {{{ types

    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c import DTypeRegistryWrapper
        from loopy.target.c.compyte.dtypes import (
                DTypeRegistry, fill_registry_with_c_types)
        result = DTypeRegistry()
        fill_registry_with_c_types(result, respect_windows=False,
                include_bool=True)
        return DTypeRegistryWrapper(result)

    def is_vector_dtype(self, dtype):
        return False

    def get_vector_dtype(self, base, count):
        raise KeyError()

    def get_or_register_dtype(self, names, dtype=None):
        # These kind of shouldn't be here.
        return self.get_dtype_registry().get_or_register_dtype(names, dtype)

    def dtype_to_typename(self, dtype):
        # These kind of shouldn't be here.
        return self.get_dtype_registry().dtype_to_ctype(dtype)

    # }}}

# }}}

# vim: foldmethod=marker

"""CUDA target independent of PyCUDA."""


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

import numpy as np

from pytools import memoize_method

from loopy.target.c import CFamilyTarget, CFamilyASTBuilder
from loopy.target.c.codegen.expression import ExpressionToCExpressionMapper
from loopy.diagnostic import LoopyError
from loopy.types import NumpyType
from loopy.kernel.data import AddressSpace
from pymbolic import var


# {{{ vector types

class vec:  # noqa
    pass


def _create_vector_types():
    field_names = ["x", "y", "z", "w"]

    import sys
    if sys.maxsize <= 2**33:
        long_dtype = np.int32
        ulong_dtype = np.uint32
    else:
        long_dtype = np.int64
        ulong_dtype = np.uint64

    vec.types = {}
    vec.names_and_dtypes = []
    vec.type_to_scalar_and_count = {}

    for base_name, base_type, counts in [
            ("char", np.int8, [1, 2, 3, 4]),
            ("uchar", np.uint8, [1, 2, 3, 4]),
            ("short", np.int16, [1, 2, 3, 4]),
            ("ushort", np.uint16, [1, 2, 3, 4]),
            ("int", np.int32, [1, 2, 3, 4]),
            ("uint", np.uint32, [1, 2, 3, 4]),
            ("long", long_dtype, [1, 2, 3, 4]),
            ("ulong", ulong_dtype, [1, 2, 3, 4]),
            ("longlong", np.int64, [1, 2]),
            ("ulonglong", np.uint64, [1, 2]),
            ("float", np.float32, [1, 2, 3, 4]),
            ("double", np.float64, [1, 2]),
            ]:
        for count in counts:
            name = "%s%d" % (base_name, count)

            titles = field_names[:count]

            names = ["s%d" % i for i in range(count)]
            if len(titles) < len(names):
                titles.extend((len(names)-len(titles))*[None])

            try:
                dtype = np.dtype(dict(
                    names=names,
                    formats=[base_type]*count,
                    titles=titles))
            except NotImplementedError:
                try:
                    dtype = np.dtype([((n, title), base_type)
                                      for (n, title) in zip(names, titles)])
                except TypeError:
                    dtype = np.dtype([(n, base_type) for (n, title)
                                      in zip(names, titles)])

            setattr(vec, name, dtype)

            vec.names_and_dtypes.append((name, dtype))

            vec.types[np.dtype(base_type), count] = dtype
            vec.type_to_scalar_and_count[dtype] = np.dtype(base_type), count


_create_vector_types()


def _register_vector_types(dtype_registry):
    for name, dtype in vec.names_and_dtypes:
        dtype_registry.get_or_register_dtype(name, dtype)

# }}}


# {{{ function mangler

def cuda_function_mangler(kernel, name, arg_dtypes):
    if not isinstance(name, str):
        return None

    if name in ["max", "min"] and len(arg_dtypes) == 2:
        dtype = np.find_common_type([], arg_dtypes)

        if dtype.kind == "c":
            raise RuntimeError("min/max do not support complex numbers")

        if dtype.kind == "f":
            name = "f" + name

        return dtype, name

    if name in ["pow"] and len(arg_dtypes) == 2:
        dtype = np.find_common_type([], arg_dtypes)

        if dtype == np.float64:
            pass  # pow
        elif dtype == np.float32:
            name = name + "f"  # powf
        else:
            raise RuntimeError(f"{name} does not support type {dtype}")

        return dtype, name

    if name in "atan2" and len(arg_dtypes) == 2:
        return arg_dtypes[0], name

    if name == "dot":
        scalar_dtype, offset, field_name = arg_dtypes[0].fields["x"]
        return scalar_dtype, name

    return None

# }}}


# {{{ expression mapper

class ExpressionToCudaCExpressionMapper(ExpressionToCExpressionMapper):
    _GRID_AXES = "xyz"

    @staticmethod
    def _get_index_ctype(kernel):
        if kernel.index_dtype.numpy_dtype == np.int32:
            return "int32_t"
        elif kernel.index_dtype.numpy_dtype == np.int64:
            return "int64_t"
        else:
            raise LoopyError("unexpected index type")

    def map_group_hw_index(self, expr, type_context):
        return var("(({}) blockIdx.{})".format(
            self._get_index_ctype(self.kernel),
            self._GRID_AXES[expr.axis]))

    def map_local_hw_index(self, expr, type_context):
        return var("(({}) threadIdx.{})".format(
            self._get_index_ctype(self.kernel),
            self._GRID_AXES[expr.axis]))

# }}}


# {{{ target

class CudaTarget(CFamilyTarget):
    """A target for Nvidia's CUDA GPU programming language."""

    def __init__(self, extern_c=True):
        """
        :arg extern_c: If *True*, declare kernels using "extern C" to
            avoid name mangling.
        """
        self.extern_c = extern_c

        super().__init__()

    def get_device_ast_builder(self):
        return CUDACASTBuilder(self)

    # {{{ types

    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c.compyte.dtypes import (DTypeRegistry,
                fill_registry_with_opencl_c_types)

        result = DTypeRegistry()
        fill_registry_with_opencl_c_types(result)

        # no complex number support--needs PyOpenCLTarget

        _register_vector_types(result)

        return result

    def is_vector_dtype(self, dtype):
        return (isinstance(dtype, NumpyType)
                and dtype.numpy_dtype in list(vec.types.values()))

    def vector_dtype(self, base, count):
        return NumpyType(
                vec.types[base.numpy_dtype, count],
                target=self)

    # }}}

# }}}


# {{{ ast builder

class CUDACASTBuilder(CFamilyASTBuilder):
    # {{{ library

    def function_manglers(self):
        return (
                super().function_manglers() + [
                    cuda_function_mangler
                    ])

    # }}}

    # {{{ top-level codegen

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        fdecl = super().get_function_declaration(
                codegen_state, codegen_result, schedule_index)

        from loopy.target.c import FunctionDeclarationWrapper
        assert isinstance(fdecl, FunctionDeclarationWrapper)
        fdecl = fdecl.subdecl

        from cgen.cuda import CudaGlobal, CudaLaunchBounds
        fdecl = CudaGlobal(fdecl)

        if self.target.extern_c:
            from cgen import Extern
            fdecl = Extern("C", fdecl)

        from loopy.schedule import get_insn_ids_for_block_at
        _, local_grid_size = \
                codegen_state.kernel.get_grid_sizes_for_insn_ids_as_exprs(
                        get_insn_ids_for_block_at(
                            codegen_state.kernel.schedule, schedule_index))

        from loopy.symbolic import get_dependencies
        if not get_dependencies(local_grid_size):
            # Sizes can't have parameter dependencies if they are
            # to be used in static thread block size.
            from pytools import product
            nthreads = product(local_grid_size)

            fdecl = CudaLaunchBounds(nthreads, fdecl)

        return FunctionDeclarationWrapper(fdecl)

    # }}}

    # {{{ code generation guts

    def get_expression_to_c_expression_mapper(self, codegen_state):
        return ExpressionToCudaCExpressionMapper(codegen_state)

    _VEC_AXES = "xyzw"

    def add_vector_access(self, access_expr, index):
        return access_expr.attr(self._VEC_AXES[index])

    def emit_barrier(self, synchronization_kind, mem_kind, comment):
        """
        :arg kind: ``"local"`` or ``"global"``
        :arg memkind: unused
        :return: a :class:`loopy.codegen.GeneratedInstruction`.
        """
        if synchronization_kind == "local":
            if comment:
                comment = " /* %s */" % comment

            from cgen import Statement
            return Statement("__syncthreads()%s" % comment)
        elif synchronization_kind == "global":
            raise LoopyError("CUDA does not have global barriers")
        else:
            raise LoopyError("unknown barrier kind")

    def wrap_temporary_decl(self, decl, scope):
        if scope == AddressSpace.LOCAL:
            from cgen.cuda import CudaShared
            return CudaShared(decl)
        elif scope == AddressSpace.PRIVATE:
            return decl
        else:
            raise ValueError("unexpected temporary variable scope: %s"
                    % scope)

    def wrap_global_constant(self, decl):
        from cgen.cuda import CudaConstant
        return CudaConstant(decl)

    def get_array_arg_decl(self, name, mem_address_space, shape, dtype, is_written):
        from loopy.target.c import POD  # uses the correct complex type
        from cgen import Const
        from cgen.cuda import CudaRestrictPointer

        arg_decl = CudaRestrictPointer(POD(self, dtype, name))

        if not is_written:
            arg_decl = Const(arg_decl)

        return arg_decl

    def get_global_arg_decl(self, name, shape, dtype, is_written):
        from warnings import warn
        warn("get_global_arg_decl is deprecated use get_array_arg_decl "
                "instead.", DeprecationWarning, stacklevel=2)
        return self.get_array_arg_decl(name, AddressSpace.GLOBAL, shape,
                dtype, is_written)

    def get_image_arg_decl(self, name, shape, num_target_axes, dtype, is_written):
        raise NotImplementedError("not yet: texture arguments in CUDA")

    def get_constant_arg_decl(self, name, shape, dtype, is_written):
        from loopy.target.c import POD  # uses the correct complex type
        from cgen import RestrictPointer, Const
        from cgen.cuda import CudaConstant

        arg_decl = RestrictPointer(POD(self, dtype, name))

        if not is_written:
            arg_decl = Const(arg_decl)

        return CudaConstant(arg_decl)

    # }}}

# }}}

# vim: foldmethod=marker

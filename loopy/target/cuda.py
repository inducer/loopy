"""CUDA target independent of PyCUDA."""

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

import numpy as np

from pytools import memoize_method

from loopy.target.c import CTarget, CASTBuilder
from loopy.target.c.codegen.expression import ExpressionToCExpressionMapper
from loopy.diagnostic import LoopyError
from loopy.types import NumpyType
from loopy.kernel.data import AddressSpace
from pymbolic import var
from loopy.kernel.function_interface import ScalarCallable


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
            ('char', np.int8, [1, 2, 3, 4]),
            ('uchar', np.uint8, [1, 2, 3, 4]),
            ('short', np.int16, [1, 2, 3, 4]),
            ('ushort', np.uint16, [1, 2, 3, 4]),
            ('int', np.int32, [1, 2, 3, 4]),
            ('uint', np.uint32, [1, 2, 3, 4]),
            ('long', long_dtype, [1, 2, 3, 4]),
            ('ulong', ulong_dtype, [1, 2, 3, 4]),
            ('longlong', np.int64, [1, 2]),
            ('ulonglong', np.uint64, [1, 2]),
            ('float', np.float32, [1, 2, 3, 4]),
            ('double', np.float64, [1, 2]),
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


# {{{ function scoper

_CUDA_SPECIFIC_FUNCTIONS = {
        "rsqrt": 1,
        "atan2": 2,
        }


class CudaCallable(ScalarCallable):

    def cuda_with_types(self, arg_id_to_dtype, kernel):

        name = self.name

        if name == "dot":
            for id in arg_id_to_dtype:
                if not -1 <= id <= 1:
                    raise LoopyError("%s can take only 2 arguments." % name)

            if 0 not in arg_id_to_dtype or 1 not in arg_id_to_dtype or (
                    arg_id_to_dtype[0] is None or arg_id_to_dtype[1] is None):
                # the types provided aren't mature enough to specialize the
                # callable
                return self.copy(arg_id_to_dtype=arg_id_to_dtype)

            dtype = arg_id_to_dtype[0]
            scalar_dtype, offset, field_name = dtype.numpy_dtype.fields["x"]
            return self.copy(name_in_target=name,
                    arg_id_to_dtype={-1: NumpyType(scalar_dtype),
                        0: dtype, 1: dtype})

        if name in _CUDA_SPECIFIC_FUNCTIONS:
            num_args = _CUDA_SPECIFIC_FUNCTIONS[name]
            for id in arg_id_to_dtype:
                if not -1 <= id < num_args:
                    raise LoopyError("%s can take only %d arguments." % (name,
                            num_args))

            for i in range(num_args):
                if i not in arg_id_to_dtype or arg_id_to_dtype[i] is None:
                    # the types provided aren't mature enough to specialize the
                    # callable
                    return self.copy(arg_id_to_dtype=arg_id_to_dtype)

            dtype = np.find_common_type(
                    [], [dtype.numpy_dtype for id, dtype in
                        arg_id_to_dtype.items() if id >= 0])

            if dtype.kind == "c":
                raise LoopyError("%s does not support complex numbers"
                        % name)

            updated_arg_id_to_dtype = dict((id, NumpyType(dtype)) for id in range(-1,
                num_args))

            return self.copy(name_in_target=name,
                    arg_id_to_dtype=updated_arg_id_to_dtype)

        return self.copy(arg_id_to_dtype=arg_id_to_dtype)


def scope_cuda_functions(target, identifier):
    if identifier in set(["dot"]) | set(
            _CUDA_SPECIFIC_FUNCTIONS):
        return CudaCallable(name=identifier)

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
        return var("((%s) blockIdx.%s)" % (
            self._get_index_ctype(self.kernel),
            self._GRID_AXES[expr.axis]))

    def map_local_hw_index(self, expr, type_context):
        return var("((%s) threadIdx.%s)" % (
            self._get_index_ctype(self.kernel),
            self._GRID_AXES[expr.axis]))

# }}}


# {{{ target

class CudaTarget(CTarget):
    """A target for Nvidia's CUDA GPU programming language."""

    def __init__(self, extern_c=True):
        """
        :arg extern_c: If *True*, declare kernels using "extern C" to
            avoid name mangling.
        """
        self.extern_c = extern_c

        super(CudaTarget, self).__init__()

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

class CUDACASTBuilder(CASTBuilder):

    # {{{ library

    def function_scopers(self):
        return [scope_cuda_functions] + (
                super(CUDACASTBuilder, self).function_scopers())

    # }}}

    # {{{ top-level codegen

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        fdecl = super(CUDACASTBuilder, self).get_function_declaration(
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

    def generate_code(self, kernel, codegen_state, impl_arg_info):
        code, implemented_domains = (
                super(CudaTarget, self).generate_code(
                    kernel, codegen_state, impl_arg_info))

        return code, implemented_domains

    def generate_body(self, kernel, codegen_state):
        body, implemented_domains = (
                super(CudaTarget, self).generate_body(kernel, codegen_state))

        from loopy.kernel.data import ImageArg

        if any(isinstance(arg, ImageArg) for arg in kernel.args):
            raise NotImplementedError("not yet: texture arguments in CUDA")

        return body, implemented_domains

    # }}}

    # {{{ code generation guts

    def get_expression_to_c_expression_mapper(self, codegen_state):
        return ExpressionToCudaCExpressionMapper(codegen_state)

    _VEC_AXES = "xyzw"

    def add_vector_access(self, access_expr, index):
        return access_expr.a(self._VEC_AXES[index])

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

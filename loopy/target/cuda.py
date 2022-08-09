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
from pymbolic import var
from pytools import memoize_method
from cgen import Declarator, Const

from loopy.target.c import CFamilyTarget, CFamilyASTBuilder
from loopy.target.c.codegen.expression import ExpressionToCExpressionMapper
from loopy.diagnostic import LoopyError, LoopyTypeError
from loopy.types import NumpyType
from loopy.kernel.array import ArrayBase, FixedStrideArrayDimTag, VectorArrayDimTag
from loopy.kernel.data import AddressSpace, ImageArg, ConstantArg, ArrayArg
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


# {{{ function scoper

_CUDA_SPECIFIC_FUNCTIONS = {
        "rsqrt": 1,
        "atan2": 2,
        }


class CudaCallable(ScalarCallable):

    def cuda_with_types(self, arg_id_to_dtype, callables_table):

        name = self.name

        if name in _CUDA_SPECIFIC_FUNCTIONS:
            num_args = _CUDA_SPECIFIC_FUNCTIONS[name]

            # {{{ sanity checks

            for id, dtype in arg_id_to_dtype.items():
                if not -1 <= id < num_args:
                    raise LoopyError("%s can take only %d arguments." % (name,
                            num_args))

                if dtype is not None and dtype.kind == "c":
                    raise LoopyTypeError(
                        f"'{name}' does not support complex arguments.")

            # }}}

            for i in range(num_args):
                if i not in arg_id_to_dtype or arg_id_to_dtype[i] is None:
                    # the types provided aren't mature enough to specialize the
                    # callable
                    return (
                            self.copy(arg_id_to_dtype=arg_id_to_dtype),
                            callables_table)

            dtype = np.find_common_type(
                    [], [dtype.numpy_dtype for id, dtype in
                        arg_id_to_dtype.items() if id >= 0])

            updated_arg_id_to_dtype = {id: NumpyType(dtype)
                    for id in range(-1, num_args)}

            return (
                    self.copy(name_in_target=name,
                        arg_id_to_dtype=updated_arg_id_to_dtype),
                    callables_table)

        if name == "dot":
            # CUDA dot function:
            # Performs dot product. Input types: vector and return type: scalar.
            for i in range(2):
                if i not in arg_id_to_dtype or arg_id_to_dtype[i] is None:
                    # the types provided aren't mature enough to specialize the
                    # callable
                    return (
                            self.copy(arg_id_to_dtype=arg_id_to_dtype),
                            callables_table)

            input_dtype = arg_id_to_dtype[0]

            scalar_dtype, offset, field_name = input_dtype.fields["x"]
            return_dtype = scalar_dtype
            return self.copy(arg_id_to_dtype={0: input_dtype, 1: input_dtype,
                                              -1: return_dtype})

        return (
                self.copy(arg_id_to_dtype=arg_id_to_dtype),
                callables_table)


def get_cuda_callables():
    cuda_func_ids = {"dot"} | set(_CUDA_SPECIFIC_FUNCTIONS)
    return {id_: CudaCallable(name=id_) for id_ in cuda_func_ids}

# }}}


# {{{ expression mapper

class ExpressionToCudaCExpressionMapper(ExpressionToCExpressionMapper):
    _GRID_AXES = "xyz"

    def map_group_hw_index(self, expr, type_context):
        return var(f"bIdx({self._GRID_AXES[expr.axis]})")

    def map_local_hw_index(self, expr, type_context):
        return var(f"tIdx({self._GRID_AXES[expr.axis]})")

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

    def split_kernel_at_global_barriers(self):
        return True

    def get_device_ast_builder(self):
        return CudaCASTBuilder(self)

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
        return NumpyType(vec.types[base.numpy_dtype, count])

    # }}}

# }}}


# {{{ preamable generator

def cuda_preamble_generator(preamble_info):
    from loopy.types import AtomicNumpyType
    seen_64_bit_atomics = any(
            isinstance(dtype, AtomicNumpyType) and dtype.numpy_dtype.itemsize == 8
            for dtype in preamble_info.seen_atomic_dtypes)

    if seen_64_bit_atomics:
        # Source:
        # docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
        yield ("00_enable_64bit_atomics", """
            #if __CUDA_ARCH__ < 600
            __device__ double atomicAdd(double* address, double val)
            {
                unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
                unsigned long long int old = *address_as_ull, assumed;

                do {
                    assumed = old;
                    old = atomicCAS(address_as_ull, assumed,
                                    __double_as_longlong(val +
                                           __longlong_as_double(assumed)));

                } while (assumed != old);

                return __longlong_as_double(old);
            }
            #endif
            """)

    from loopy.tools import remove_common_indentation
    kernel = preamble_info.kernel
    idx_ctype = kernel.target.dtype_to_typename(kernel.index_dtype)
    yield ("00_declare_gid_lid",
           remove_common_indentation(f"""
                #define bIdx(N) (({idx_ctype}) blockIdx.N)
                #define tIdx(N) (({idx_ctype}) threadIdx.N)
           """))

# }}}


# {{{ ast builder

class CudaCASTBuilder(CFamilyASTBuilder):

    preamble_function_qualifier = "inline __device__"

    # {{{ library

    @property
    def known_callables(self):
        callables = super().known_callables
        callables.update(get_cuda_callables())
        return callables

    def symbol_manglers(self):
        from loopy.target.opencl import opencl_symbol_mangler
        return (
                super().symbol_manglers() + [
                    opencl_symbol_mangler
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
                            codegen_state.kernel.linearization, schedule_index),
                        codegen_state.callables_table)

        from loopy.symbolic import get_dependencies
        if not get_dependencies(local_grid_size):
            # Sizes can't have parameter dependencies if they are
            # to be used in static thread block size.
            from pytools import product
            nthreads = product(local_grid_size)

            fdecl = CudaLaunchBounds(nthreads, fdecl)

        return FunctionDeclarationWrapper(fdecl)

    def preamble_generators(self):

        return (
                super().preamble_generators() + [
                    cuda_preamble_generator])

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

    # }}}

    # {{{ declarators

    def wrap_decl_for_address_space(
            self, decl: Declarator, address_space: AddressSpace) -> Declarator:
        from cgen.cuda import CudaGlobal, CudaShared
        if address_space == AddressSpace.GLOBAL:
            return CudaGlobal(decl)
        if address_space == AddressSpace.LOCAL:
            return CudaShared(decl)
        elif address_space == AddressSpace.PRIVATE:
            return decl
        else:
            raise ValueError("unexpected address_space: %s"
                    % address_space)

    def wrap_global_constant(self, decl: Declarator) -> Declarator:
        from cgen.cuda import CudaConstant, CudaGlobal
        assert isinstance(decl, CudaGlobal)
        decl = decl.subdecl
        return CudaConstant(decl)

    # duplicated in OpenCL, update there if updating here
    def get_array_base_declarator(self, ary: ArrayBase) -> Declarator:
        dtype = ary.dtype

        vec_size = ary.vector_size(self.target)
        if vec_size > 1:
            dtype = self.target.vector_dtype(dtype, vec_size)

        if ary.dim_tags:
            for dim_tag in ary.dim_tags:
                if isinstance(dim_tag, (FixedStrideArrayDimTag, VectorArrayDimTag)):
                    # we're OK with those
                    pass

                else:
                    raise NotImplementedError(
                        f"{type(self).__name__} does not understand axis tag "
                        f"'{type(dim_tag)}.")

        from loopy.target.c import POD
        return POD(self, dtype, ary.name)

    def get_array_arg_declarator(
            self, arg: ArrayArg, is_written: bool) -> Declarator:
        from cgen.cuda import CudaRestrictPointer
        arg_decl = CudaRestrictPointer(
                self.wrap_decl_for_address_space(
                    self.get_array_base_declarator(arg), arg.address_space))

        if not is_written:
            arg_decl = Const(arg_decl)

        return arg_decl

    def get_constant_arg_declarator(self, arg: ConstantArg) -> Declarator:
        from cgen import RestrictPointer
        from cgen.cuda import CudaConstant

        # constant *is* an address space as far as CUDA is concerned, do not re-wrap
        return CudaConstant(RestrictPointer(self.get_array_base_declarator(arg)))

    def get_image_arg_declarator(
            self, arg: ImageArg, is_written: bool) -> Declarator:
        raise NotImplementedError("not yet: texture arguments in CUDA")

    # }}}

    # {{{ atomics

    def emit_atomic_update(self, codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype, rhs_type_context):

        from pymbolic.primitives import Sum
        from cgen import Statement
        from pymbolic.mapper.stringifier import PREC_NONE

        if isinstance(lhs_dtype, NumpyType) and lhs_dtype.numpy_dtype in [
                np.int32, np.int64, np.float32, np.float64]:
            # atomicAdd
            if isinstance(rhs_expr, Sum):
                ecm = self.get_expression_to_code_mapper(codegen_state)

                new_rhs_expr = Sum(tuple(c for c in rhs_expr.children
                                         if c != lhs_expr))
                lhs_expr_code = ecm(lhs_expr)
                rhs_expr_code = ecm(new_rhs_expr)

                return Statement("atomicAdd(&{}, {})".format(
                    lhs_expr_code, rhs_expr_code))
            else:
                from cgen import Block, DoWhile, Assign
                from loopy.target.c import POD
                old_val_var = codegen_state.var_name_generator("loopy_old_val")
                new_val_var = codegen_state.var_name_generator("loopy_new_val")

                from loopy.kernel.data import TemporaryVariable
                ecm = codegen_state.expression_to_code_mapper.with_assignments(
                        {
                            old_val_var: TemporaryVariable(old_val_var, lhs_dtype),
                            new_val_var: TemporaryVariable(new_val_var, lhs_dtype),
                            })

                lhs_expr_code = ecm(lhs_expr, prec=PREC_NONE, type_context=None)

                from pymbolic.mapper.substitutor import make_subst_func
                from pymbolic import var
                from loopy.symbolic import SubstitutionMapper

                subst = SubstitutionMapper(
                        make_subst_func({lhs_expr: var(old_val_var)}))
                rhs_expr_code = ecm(subst(rhs_expr), prec=PREC_NONE,
                        type_context=rhs_type_context,
                        needed_dtype=lhs_dtype)

                cast_str = ""
                old_val = old_val_var
                new_val = new_val_var

                if lhs_dtype.numpy_dtype.kind == "f":
                    if lhs_dtype.numpy_dtype == np.float32:
                        ctype = "int"
                    elif lhs_dtype.numpy_dtype == np.float64:
                        ctype = "long"
                    else:
                        raise AssertionError()

                    old_val = "*(%s *) &" % ctype + old_val
                    new_val = "*(%s *) &" % ctype + new_val
                    cast_str = "(%s *) " % (ctype)

                return Block([
                    POD(self, NumpyType(lhs_dtype.dtype, target=self.target),
                        old_val_var),
                    POD(self, NumpyType(lhs_dtype.dtype, target=self.target),
                        new_val_var),
                    DoWhile(
                        "atomicCAS("
                        "%(cast_str)s&(%(lhs_expr)s), "
                        "%(old_val)s, "
                        "%(new_val)s"
                        ") != %(old_val)s"
                        % {
                            "cast_str": cast_str,
                            "lhs_expr": lhs_expr_code,
                            "old_val": old_val,
                            "new_val": new_val,
                            },
                        Block([
                            Assign(old_val_var, lhs_expr_code),
                            Assign(new_val_var, rhs_expr_code),
                            ])
                        )
                    ])
        else:
            raise NotImplementedError("atomic update for '%s'" % lhs_dtype)

    # }}}

# }}}

# vim: foldmethod=marker

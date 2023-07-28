"""OpenCL target independent of PyOpenCL."""


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

from typing import Tuple, Sequence

import numpy as np
from pymbolic import var
from pytools import memoize_method
from cgen import Declarator, Generable

from loopy.target.c import CFamilyTarget, CFamilyASTBuilder
from loopy.target.c.codegen.expression import ExpressionToCExpressionMapper
from loopy.diagnostic import LoopyError, LoopyTypeError
from loopy.types import NumpyType
from loopy.target.c import DTypeRegistryWrapper
from loopy.kernel.array import VectorArrayDimTag, FixedStrideArrayDimTag, ArrayBase
from loopy.kernel.data import AddressSpace, ImageArg, ConstantArg
from loopy.kernel.function_interface import ScalarCallable
from loopy.codegen import CodeGenerationState
from loopy.codegen.result import CodeGenerationResult


# {{{ dtype registry wrappers


class DTypeRegistryWrapperWithInt8ForBool(DTypeRegistryWrapper):
    """
    A DType registry that uses int8 for bool_ types.

    .. note::

        This sub-class is needed because compyte's type registry does
        not support type aliases.
    """
    def dtype_to_ctype(self, dtype):
        from loopy.types import NumpyType
        if isinstance(dtype, NumpyType) and dtype.dtype == np.bool_:
            return self.wrapped_registry.dtype_to_ctype(
                    NumpyType(np.int8))
        return self.wrapped_registry.dtype_to_ctype(dtype)


class DTypeRegistryWrapperWithAtomics(DTypeRegistryWrapper):
    def get_or_register_dtype(self, names, dtype=None):
        if dtype is not None:
            from loopy.types import AtomicNumpyType, NumpyType
            if isinstance(dtype, AtomicNumpyType):
                return self.wrapped_registry.get_or_register_dtype(
                        names, NumpyType(dtype.dtype))

        return self.wrapped_registry.get_or_register_dtype(names, dtype)


class DTypeRegistryWrapperWithCL1Atomics(DTypeRegistryWrapperWithAtomics):
    def dtype_to_ctype(self, dtype):
        from loopy.types import AtomicNumpyType

        if isinstance(dtype, AtomicNumpyType):
            return "volatile " + self.wrapped_registry.dtype_to_ctype(dtype)
        else:
            return self.wrapped_registry.dtype_to_ctype(dtype)

# }}}


# {{{ vector types

class vec:  # noqa
    pass


def _create_vector_types():
    field_names = ["x", "y", "z", "w"]

    vec.types = {}
    vec.names_and_dtypes = []
    vec.type_to_scalar_and_count = {}

    counts = [2, 3, 4, 8, 16]

    for base_name, base_type in [
            ("char", np.int8),
            ("uchar", np.uint8),
            ("short", np.int16),
            ("ushort", np.uint16),
            ("int", np.int32),
            ("uint", np.uint32),
            ("long", np.int64),
            ("ulong", np.uint64),
            ("float", np.float32),
            ("double", np.float64),
            ]:
        for count in counts:
            name = "%s%d" % (base_name, count)

            titles = field_names[:count]

            padded_count = count
            if count == 3:
                padded_count = 4

            names = ["s%d" % i for i in range(count)]
            while len(names) < padded_count:
                names.append("padding%d" % (len(names)-count))

            if len(titles) < len(names):
                titles.extend((len(names)-len(titles))*[None])

            try:
                dtype = np.dtype({
                    "names": names,
                    "formats": [base_type]*padded_count,
                    "titles": titles})
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

_CL_SIMPLE_MULTI_ARG_FUNCTIONS = {
        "rsqrt": 1,
        "clamp": 3,
        "atan2": 2,
        }


VECTOR_LITERAL_FUNCS = {
        "make_%s%d" % (name, count): (name, dtype, count)
        for name, dtype in [
            ("char", np.int8),
            ("uchar", np.uint8),
            ("short", np.int16),
            ("ushort", np.uint16),
            ("int", np.int32),
            ("uint", np.uint32),
            ("long", np.int64),
            ("ulong", np.uint64),
            ("float", np.float32),
            ("double", np.float64),
            ]
        for count in [2, 3, 4, 8, 16]
        }


class OpenCLCallable(ScalarCallable):
    """
    Records information about OpenCL functions which are not covered by
    :class:`loopy.target.c.CMathCallable`.
    """

    def with_types(self, arg_id_to_dtype, callables_table):
        name = self.name

        # {{{ unary functions
        if name == "abs":
            for id in arg_id_to_dtype:
                if not -1 <= id <= 0:
                    raise LoopyError(f"'{name}' can take only one argument.")

            if 0 not in arg_id_to_dtype or arg_id_to_dtype[0] is None:
                return (
                        self.copy(arg_id_to_dtype=arg_id_to_dtype),
                        callables_table)

            dtype = arg_id_to_dtype[0].numpy_dtype

            if dtype.kind in ("u", "i"):
                # OpenCL C 2.2, Section 6.13.3: abs returns *u*gentype
                from loopy.types import to_unsigned_dtype
                return (self.copy(name_in_target=name,
                            arg_id_to_dtype={
                                0: NumpyType(dtype),
                                -1: NumpyType(to_unsigned_dtype(dtype))}),
                        callables_table)
            elif dtype.kind == "f":
                name = "fabs"
            else:
                raise LoopyTypeError(f"'{name}' does not support type {dtype}")

        # deliberately not elif: abs branch above may end up taking this.
        if name in ["fabs", "acos", "asin", "atan", "cos", "cosh", "sin", "sinh",
                    "tan", "tanh", "exp", "log", "log10", "sqrt", "ceil", "floor",
                    "erf", "erfc"]:

            for id in arg_id_to_dtype:
                if not -1 <= id <= 0:
                    raise LoopyError(f"'{name}' can take only one argument.")

            if 0 not in arg_id_to_dtype or arg_id_to_dtype[0] is None:
                return (
                        self.copy(arg_id_to_dtype=arg_id_to_dtype),
                        callables_table)

            dtype = arg_id_to_dtype[0]
            dtype = dtype.numpy_dtype

            if dtype.kind in ("u", "i"):
                # ints and unsigned casted to float32
                dtype = np.float32
            elif dtype.kind == "c":
                raise LoopyTypeError(f"{name} does not support type {dtype}")

            return (
                    self.copy(name_in_target=name,
                        arg_id_to_dtype={0: NumpyType(dtype), -1:
                            NumpyType(dtype)}),
                    callables_table)

        # }}}

        # binary functions
        elif name in ["fmax", "fmin", "atan2", "copysign"]:

            for id in arg_id_to_dtype:
                if not -1 <= id <= 1:
                    #FIXME: Do we need to raise here?:
                    #   The pattern we generally follow is that if we don't find
                    #   a function, then we just return None
                    raise LoopyError("%s can take only two arguments." % name)

            if 0 not in arg_id_to_dtype or 1 not in arg_id_to_dtype or (
                    arg_id_to_dtype[0] is None or arg_id_to_dtype[1] is None):
                return (
                        self.copy(arg_id_to_dtype=arg_id_to_dtype),
                        callables_table)

            dtype = np.result_type(*[
                    dtype.numpy_dtype for id, dtype in arg_id_to_dtype.items()
                    if id >= 0])

            if dtype.kind == "c":
                raise LoopyTypeError(f"'{name}' does not support complex numbers")

            dtype = NumpyType(dtype)
            return (
                    self.copy(name_in_target=name,
                        arg_id_to_dtype={-1: dtype, 0: dtype, 1: dtype}),
                    callables_table)

        elif name in ["max", "min"]:
            for id in arg_id_to_dtype:
                if not -1 <= id <= 1:
                    raise LoopyError("%s can take only 2 arguments." % name)
            if 0 not in arg_id_to_dtype or 1 not in arg_id_to_dtype:
                return (
                        self.copy(arg_id_to_dtype=arg_id_to_dtype),
                        callables_table)
            common_dtype = np.result_type(*[
                    dtype.numpy_dtype for id, dtype in arg_id_to_dtype.items()
                    if (id >= 0 and dtype is not None)])

            if common_dtype.kind in ["u", "i", "f"]:
                if common_dtype.kind == "f":
                    name = "f"+name

                dtype = NumpyType(common_dtype)
                return (
                        self.copy(name_in_target=name,
                            arg_id_to_dtype={-1: dtype, 0: dtype, 1: dtype}),
                        callables_table)
            else:
                # Unsupported type.
                raise LoopyError("%s function not supported for the types %s" %
                        (name, common_dtype))

        elif name == "dot":
            for id in arg_id_to_dtype:
                if not -1 <= id <= 1:
                    raise LoopyError(f"'{name}' can take only 2 arguments.")

            if 0 not in arg_id_to_dtype or 1 not in arg_id_to_dtype or (
                    arg_id_to_dtype[0] is None or arg_id_to_dtype[1] is None):
                # the types provided aren't mature enough to specialize the
                # callable
                return (
                        self.copy(arg_id_to_dtype=arg_id_to_dtype),
                        callables_table)

            dtype = arg_id_to_dtype[0]
            scalar_dtype, offset, field_name = dtype.numpy_dtype.fields["s0"]
            return (
                    self.copy(name_in_target=name, arg_id_to_dtype={-1:
                        NumpyType(scalar_dtype), 0: dtype, 1: dtype}),
                    callables_table)

        elif name == "pow":
            for id in arg_id_to_dtype:
                if not -1 <= id <= 1:
                    raise LoopyError(f"'{name}' can take only 2 arguments.")

            common_dtype = np.result_type(*[
                    dtype.numpy_dtype for id, dtype in arg_id_to_dtype.items()
                    if (id >= 0 and dtype is not None)])

            if common_dtype == np.float64:
                name = "powf64"
            elif common_dtype == np.float32:
                name = "powf32"
            else:
                raise LoopyTypeError(f"'pow' does not support type {dtype}.")

            result_dtype = NumpyType(common_dtype)

            return (
                    self.copy(name_in_target=name,
                              arg_id_to_dtype={-1: result_dtype,
                                               0: common_dtype, 1: common_dtype}),
                    callables_table)

        elif name in _CL_SIMPLE_MULTI_ARG_FUNCTIONS:
            num_args = _CL_SIMPLE_MULTI_ARG_FUNCTIONS[name]
            for id in arg_id_to_dtype:
                if not -1 <= id < num_args:
                    raise LoopyError("%s can take only %d arguments." % (name,
                            num_args))

            for i in range(num_args):
                if i not in arg_id_to_dtype or arg_id_to_dtype[i] is None:
                    # the types provided aren't mature enough to specialize the
                    # callable
                    return (
                            self.copy(arg_id_to_dtype=arg_id_to_dtype),
                            callables_table)

            dtype = np.result_type(*[
                    dtype.numpy_dtype for id, dtype in arg_id_to_dtype.items()
                    if id >= 0])

            if dtype.kind == "c":
                raise LoopyError("%s does not support complex numbers"
                        % name)

            updated_arg_id_to_dtype = {id: NumpyType(dtype) for id in range(-1,
                num_args)}

            return (
                    self.copy(name_in_target=name,
                        arg_id_to_dtype=updated_arg_id_to_dtype),
                    callables_table)

        elif name in VECTOR_LITERAL_FUNCS:
            base_tp_name, dtype, count = VECTOR_LITERAL_FUNCS[name]

            for id in arg_id_to_dtype:
                if not -1 <= id < count:
                    raise LoopyError("%s can take only %d arguments." % (name,
                            num_args))

            for i in range(count):
                if i not in arg_id_to_dtype or arg_id_to_dtype[i] is None:
                    # the types provided aren't mature enough to specialize the
                    # callable
                    return (
                            self.copy(arg_id_to_dtype=arg_id_to_dtype),
                            callables_table)

            updated_arg_id_to_dtype = {id: NumpyType(dtype) for id in
                    range(count)}
            updated_arg_id_to_dtype[-1] = OpenCLTarget().vector_dtype(
                        NumpyType(dtype), count)

            return (
                    self.copy(name_in_target="(%s%d) " % (base_tp_name, count),
                        arg_id_to_dtype=updated_arg_id_to_dtype),
                    callables_table)

        # does not satisfy any of the conditions needed for specialization.
        # hence just returning a copy of the callable.
        return (
                self.copy(arg_id_to_dtype=arg_id_to_dtype),
                callables_table)


def get_opencl_callables():
    """
    Returns an instance of :class:`InKernelCallable` if the function defined by
    *identifier* is known in OpenCL.
    """
    opencl_function_ids = (
            {"dot", "abs",
             "cos", "cosh", "sin", "sinh", "tan", "tanh",
             "acos", "acosh", "asin", "asinh", "atan", "atanh", "atan2",
             "pow", "exp", "log", "log10", "sqrt", "ceil", "floor",
             "max", "min", "fmax", "fmin",
             "fabs",  "erf", "erfc"}
            | set(_CL_SIMPLE_MULTI_ARG_FUNCTIONS)
            | set(VECTOR_LITERAL_FUNCS))

    return {id_: OpenCLCallable(name=id_) for id_ in
        opencl_function_ids}

# }}}


# {{{ symbol mangler

def opencl_symbol_mangler(kernel, name):
    # FIXME: should be more picky about exact names
    if name.startswith("FLT_"):
        return NumpyType(np.dtype(np.float32)), name
    elif name.startswith("DBL_"):
        return NumpyType(np.dtype(np.float64)), name
    elif name.startswith("M_"):
        if name.endswith("_F"):
            return NumpyType(np.dtype(np.float32)), name
        else:
            return NumpyType(np.dtype(np.float64)), name
    elif name == "INFINITY":
        return NumpyType(np.dtype(np.float32)), name
    elif name.startswith("INT_"):
        return NumpyType(np.dtype(np.int32)), name
    elif name.startswith("LONG_"):
        return NumpyType(np.dtype(np.int64)), name
    elif name == "HUGE_VAL":
        return NumpyType(np.dtype(np.float64)), name
    else:
        return None

# }}}


# {{{ preamble generator

def opencl_preamble_generator(preamble_info):
    has_double = False

    for dtype in preamble_info.seen_dtypes:
        if (isinstance(dtype, NumpyType)
                and dtype.numpy_dtype in [np.float64, np.complex128]):
            has_double = True

    if has_double:
        yield ("00_enable_double", """
            #if __OPENCL_C_VERSION__ < 120
            #pragma OPENCL EXTENSION cl_khr_fp64: enable
            #endif
            """)

    from loopy.types import AtomicNumpyType
    seen_64_bit_atomics = any(
            isinstance(dtype, AtomicNumpyType) and dtype.numpy_dtype.itemsize == 8
            for dtype in preamble_info.seen_atomic_dtypes)

    if seen_64_bit_atomics:
        # FIXME: Should gate on "CL1" atomics style
        yield ("00_enable_64bit_atomics", """
            #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
            """)

    from loopy.tools import remove_common_indentation
    kernel = preamble_info.kernel

    idx_ctype = kernel.target.dtype_to_typename(kernel.index_dtype)
    yield ("00_declare_gid_lid",
            remove_common_indentation(f"""
                #define lid(N) (({idx_ctype}) get_local_id(N))
                #define gid(N) (({idx_ctype}) get_group_id(N))
                """))

    for func in preamble_info.seen_functions:
        if func.name == "pow" and func.c_name == "powf32":
            yield ("08_clpowf32", """
                inline float powf32(float x, float y) {
                return pow(x, y);
                }""")

        if func.name == "pow" and func.c_name == "powf64":
            yield ("08_clpowf64", """
                inline double powf64(double x, double y) {
                return pow(x, y);
                }""")

# }}}


# {{{ expression mapper

class ExpressionToOpenCLCExpressionMapper(ExpressionToCExpressionMapper):

    def wrap_in_typecast_lazy(self, actual_dtype, needed_dtype, s):
        if needed_dtype.dtype.kind == "b" and actual_dtype().dtype.kind == "f":
            # CL does not perform implicit conversion from float-type to a bool.
            from pymbolic.primitives import Comparison
            return Comparison(s, "!=", 0)

        return super().wrap_in_typecast_lazy(actual_dtype, needed_dtype, s)

    def map_group_hw_index(self, expr, type_context):
        return var("gid")(expr.axis)

    def map_local_hw_index(self, expr, type_context):
        return var("lid")(expr.axis)

# }}}


# {{{ target

class OpenCLTarget(CFamilyTarget):
    """A target for the OpenCL C heterogeneous compute programming language.
    """

    def __init__(self, atomics_flavor=None, use_int8_for_bool=True):
        """
        :arg atomics_flavor: one of ``"cl1"`` (C11-style atomics from OpenCL 2.0),
            ``"cl1"`` (OpenCL 1.1 atomics, using bit-for-bit compare-and-swap
            for floating point), ``"cl1-exch"`` (OpenCL 1.1 atomics, using
            double-exchange for floating point--not yet supported).
        :arg use_int8_for_bool: Size of *bool* is undefined as per
            OpenCL spec, if *True* all bool_ variables would be treated
            as int8's.
        """
        super().__init__()

        if atomics_flavor is None:
            atomics_flavor = "cl1"

        if atomics_flavor not in ["cl1", "cl2"]:
            raise ValueError("unsupported atomics flavor: %s" % atomics_flavor)

        self.atomics_flavor = atomics_flavor
        self.use_int8_for_bool = use_int8_for_bool

    def split_kernel_at_global_barriers(self):
        return True

    def get_device_ast_builder(self):
        return OpenCLCASTBuilder(self)

    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c.compyte.dtypes import (DTypeRegistry,
                fill_registry_with_opencl_c_types)

        result = DTypeRegistry()
        fill_registry_with_opencl_c_types(result)

        # no complex number support--needs PyOpenCLTarget

        _register_vector_types(result)

        if self.atomics_flavor == "cl1":
            result = DTypeRegistryWrapperWithCL1Atomics(result)
        else:
            raise NotImplementedError("atomics flavor: %s" % self.atomics_flavor)

        if self.use_int8_for_bool:
            result = DTypeRegistryWrapperWithInt8ForBool(result)

        return result

    def is_vector_dtype(self, dtype):
        return (isinstance(dtype, NumpyType)
                and dtype.numpy_dtype in list(vec.types.values()))

    def vector_dtype(self, base, count):
        return NumpyType(vec.types[base.numpy_dtype, count])

# }}}


# {{{ ast builder

class OpenCLCASTBuilder(CFamilyASTBuilder):
    # {{{ library

    @property
    def known_callables(self):
        callables = super().known_callables
        callables.update(get_opencl_callables())
        return callables

    def symbol_manglers(self):
        return (
                super().symbol_manglers() + [
                    opencl_symbol_mangler
                    ])

    def preamble_generators(self):

        return (
                super().preamble_generators() + [
                    opencl_preamble_generator])

    # }}}

    # {{{ top-level codegen

    def get_function_declaration(
            self, codegen_state: CodeGenerationState,
            codegen_result: CodeGenerationResult, schedule_index: int
            ) -> Tuple[Sequence[Tuple[str, str]], Generable]:
        preambles, fdecl = super().get_function_declaration(
                codegen_state, codegen_result, schedule_index)

        from loopy.target.c import FunctionDeclarationWrapper
        assert isinstance(fdecl, FunctionDeclarationWrapper)
        if not codegen_state.is_entrypoint:
            # auxiliary kernels need not mention opencl speicific qualifiers
            # for a functions signature
            return preambles, fdecl

        return preambles, FunctionDeclarationWrapper(
                self._wrap_kernel_decl(codegen_state, schedule_index, fdecl.subdecl))

    def _wrap_kernel_decl(
            self, codegen_state: CodeGenerationState, schedule_index: int,
            fdecl: Declarator) -> Declarator:
        from cgen.opencl import CLKernel, CLRequiredWorkGroupSize
        fdecl = CLKernel(fdecl)

        from loopy.schedule import get_insn_ids_for_block_at
        assert codegen_state.kernel.linearization is not None
        _, local_sizes = codegen_state.kernel.get_grid_sizes_for_insn_ids_as_exprs(
                get_insn_ids_for_block_at(
                    codegen_state.kernel.linearization, schedule_index),
                codegen_state.callables_table)

        from loopy.symbolic import get_dependencies
        if not get_dependencies(local_sizes):
            # sizes can't have parameter dependencies if they are
            # to be used in static WG size.

            fdecl = CLRequiredWorkGroupSize(local_sizes, fdecl)

        return fdecl

    def generate_top_of_body(self, codegen_state):
        from loopy.kernel.data import ImageArg
        if any(isinstance(arg, ImageArg) for arg in codegen_state.kernel.args):
            from cgen import Value, Const, Initializer
            return [
                    Initializer(Const(Value("sampler_t", "loopy_sampler")),
                        "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP "
                        "| CLK_FILTER_NEAREST")
                    ]

        return []

    # }}}

    def get_expression_to_c_expression_mapper(self, codegen_state):
        return ExpressionToOpenCLCExpressionMapper(codegen_state)

    def add_vector_access(self, access_expr, index):
        # The 'int' avoids an 'L' suffix for long ints.
        return access_expr.attr("s%s" % hex(int(index))[2:])

    def emit_barrier(self, synchronization_kind, mem_kind, comment):
        """
        :arg kind: ``"local"`` or ``"global"``
        :return: a :class:`loopy.codegen.GeneratedInstruction`.
        """
        if synchronization_kind == "local":
            if comment:
                comment = " /* %s */" % comment

            mem_kind = mem_kind.upper()

            from cgen import Statement
            return Statement(f"barrier(CLK_{mem_kind}_MEM_FENCE){comment}")
        elif synchronization_kind == "global":
            raise LoopyError("OpenCL does not have global barriers")
        else:
            raise LoopyError("unknown barrier kind")

    # {{{ declarators

    def wrap_decl_for_address_space(
            self, decl: Declarator, address_space: AddressSpace) -> Declarator:
        from cgen.opencl import CLGlobal, CLLocal
        if address_space == AddressSpace.GLOBAL:
            return CLGlobal(decl)
        elif address_space == AddressSpace.LOCAL:
            return CLLocal(decl)
        elif address_space == AddressSpace.PRIVATE:
            return decl
        else:
            raise ValueError("unexpected temporary variable address space: %s"
                    % address_space)

    def wrap_global_constant(self, decl: Declarator) -> Declarator:
        from cgen.opencl import CLGlobal, CLConstant
        assert isinstance(decl, CLGlobal)
        decl = decl.subdecl

        return CLConstant(decl)

    # duplicated in CUDA, update there if updating here
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

    def get_constant_arg_declarator(self, arg: ConstantArg) -> Declarator:
        from cgen import RestrictPointer
        from cgen.opencl import CLConstant

        # constant *is* an address space as far as CL is concerned, do not re-wrap
        return CLConstant(RestrictPointer(self.get_array_base_declarator(
                arg)))

    def get_image_arg_declarator(
            self, arg: ImageArg, is_written: bool) -> Declarator:
        if is_written:
            mode = "w"
        else:
            mode = "r"

        from cgen.opencl import CLImage
        return CLImage(arg.num_target_axes(), mode, arg.name)

    # }}}

    # {{{ atomics

    def emit_atomic_init(self, codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype, rhs_type_context):
        # for the CL1 flavor, this is as simple as a regular update with whatever
        # the RHS value is...

        return self.emit_atomic_update(codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype, rhs_type_context)

    def emit_unroll_hint(self, value):
        # See https://man.opencl.org/attributes-loopUnroll.html
        from cgen import Line
        if value:
            return Line(f"__attribute__((opencl_unroll_hint({value})))")
        else:
            return Line("__attribute__((opencl_unroll_hint))")

    def emit_atomic_update(self, codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype, rhs_type_context):
        from pymbolic.mapper.stringifier import PREC_NONE

        # FIXME: Could detect operations, generate atomic_{add,...} when
        # appropriate.

        if isinstance(lhs_dtype, NumpyType) and lhs_dtype.numpy_dtype in [
                np.int32, np.int64, np.float32, np.float64]:
            from cgen import Block, DoWhile, Assign
            from loopy.target.c import POD
            old_val_var = codegen_state.var_name_generator("loopy_old_val")
            new_val_var = codegen_state.var_name_generator("loopy_new_val")

            from loopy.kernel.data import TemporaryVariable, AddressSpace
            ecm = codegen_state.expression_to_code_mapper.with_assignments(
                    {
                        old_val_var: TemporaryVariable(old_val_var, lhs_dtype,
                            shape=()),
                        new_val_var: TemporaryVariable(new_val_var, lhs_dtype,
                            shape=()),
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

            if lhs_dtype.numpy_dtype.itemsize == 4:
                func_name = "atomic_cmpxchg"
            elif lhs_dtype.numpy_dtype.itemsize == 8:
                func_name = "atom_cmpxchg"
            else:
                raise LoopyError("unexpected atomic size")

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

                from loopy.kernel.data import (TemporaryVariable, ArrayArg)
                if (
                        isinstance(lhs_var, ArrayArg)
                        and
                        lhs_var.address_space == AddressSpace.GLOBAL):
                    var_kind = "__global"
                elif (
                        isinstance(lhs_var, ArrayArg)
                        and
                        lhs_var.address_space == AddressSpace.LOCAL):
                    var_kind = "__local"
                elif (
                        isinstance(lhs_var, TemporaryVariable)
                        and lhs_var.address_space == AddressSpace.LOCAL):
                    var_kind = "__local"
                elif (
                        isinstance(lhs_var, TemporaryVariable)
                        and lhs_var.address_space == AddressSpace.GLOBAL):
                    var_kind = "__global"
                else:
                    raise LoopyError("unexpected kind of variable '%s' in "
                            "atomic operation: '%s'"
                            % (lhs_var.name, type(lhs_var).__name__))

                old_val = "*(%s *) &" % ctype + old_val
                new_val = "*(%s *) &" % ctype + new_val
                cast_str = f"({var_kind} {ctype} *) "

            return Block([
                POD(self, NumpyType(lhs_dtype.dtype),
                    old_val_var),
                POD(self, NumpyType(lhs_dtype.dtype),
                    new_val_var),
                DoWhile(
                    "%(func_name)s("
                    "%(cast_str)s&(%(lhs_expr)s), "
                    "%(old_val)s, "
                    "%(new_val)s"
                    ") != %(old_val)s"
                    % {
                        "func_name": func_name,
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


# {{{ volatile mem acccess target

class VolatileMemExpressionToOpenCLCExpressionMapper(
        ExpressionToOpenCLCExpressionMapper):
    def make_subscript(self, array, base_expr, subscript):
        registry = self.codegen_state.ast_builder.target.get_dtype_registry()

        from loopy.kernel.data import AddressSpace
        if array.address_space == AddressSpace.GLOBAL:
            aspace = "__global "
        elif array.address_space == AddressSpace.LOCAL:
            aspace = "__local "
        elif array.address_space == AddressSpace.PRIVATE:
            aspace = ""
        else:
            raise ValueError("unexpected value of address space")

        from pymbolic import var
        return var(
                "(%s volatile %s *) "
                % (
                    registry.dtype_to_ctype(array.dtype),
                    aspace,
                    )
                )(base_expr)[subscript]


class VolatileMemOpenCLCASTBuilder(OpenCLCASTBuilder):
    def get_expression_to_c_expression_mapper(self, codegen_state):
        return VolatileMemExpressionToOpenCLCExpressionMapper(codegen_state)


class VolatileMemOpenCLTarget(OpenCLTarget):
    def get_device_ast_builder(self):
        return VolatileMemOpenCLCASTBuilder(self)

# }}}

# vim: foldmethod=marker

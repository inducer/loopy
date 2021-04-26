"""Plain C target and base for other C-family languages."""


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

import numpy as np  # noqa
from loopy.kernel.data import CallMangleInfo
from loopy.target import TargetBase, ASTBuilderBase, DummyHostASTBuilder
from loopy.diagnostic import LoopyError, LoopyTypeError
from cgen import Pointer, NestedDeclarator, Block
from cgen.mapper import IdentityMapper as CASTIdentityMapperBase
from pymbolic.mapper.stringifier import PREC_NONE
from loopy.symbolic import IdentityMapper
from loopy.types import NumpyType
import pymbolic.primitives as p

from loopy.tools import remove_common_indentation
import re

from pytools import memoize_method

__doc__ = """
.. currentmodule loopy.target.c

.. autoclass:: POD

.. autoclass:: ScopingBlock

.. automodule:: loopy.target.c.codegen.expression
"""


# {{{ dtype registry wrapper

class DTypeRegistryWrapper:
    def __init__(self, wrapped_registry):
        self.wrapped_registry = wrapped_registry

    def get_or_register_dtype(self, names, dtype=None):
        if dtype is not None:
            from loopy.types import LoopyType, NumpyType
            assert isinstance(dtype, LoopyType)

            if isinstance(dtype, NumpyType):
                return self.wrapped_registry.get_or_register_dtype(
                        names, dtype.dtype)
            else:
                raise LoopyError(
                        "unable to get or register type '%s'"
                        % dtype)
        else:
            return self.wrapped_registry.get_or_register_dtype(names, dtype)

    def dtype_to_ctype(self, dtype):
        from loopy.types import LoopyType, NumpyType
        assert isinstance(dtype, LoopyType)

        if isinstance(dtype, NumpyType):
            return self.wrapped_registry.dtype_to_ctype(dtype)
        else:
            raise LoopyError(
                    "unable to convert type '%s' to C"
                    % dtype)

# }}}


# {{{ preamble generator

def c99_preamble_generator(preamble_info):
    if any(dtype.is_integral() for dtype in preamble_info.seen_dtypes):
        yield("10_stdint", "#include <stdint.h>")
    if any(dtype.numpy_dtype == np.dtype("bool")
           for dtype in preamble_info.seen_dtypes):
        yield("10_stdbool", "#include <stdbool.h>")
    if any(dtype.is_complex() for dtype in preamble_info.seen_dtypes):
        yield("10_complex", "#include <complex.h>")


def _preamble_generator(preamble_info):
    integer_type_names = ["int8", "int16", "int32", "int64"]

    def_integer_types_macro = ("03_def_integer_types", r"""
            #define LOOPY_CALL_WITH_INTEGER_TYPES(MACRO_NAME) \
                MACRO_NAME(int8, char) \
                MACRO_NAME(int16, short) \
                MACRO_NAME(int32, int) \
                MACRO_NAME(int64, long)
            """)

    undef_integer_types_macro = ("05_undef_integer_types", """
            #undef LOOPY_CALL_WITH_INTEGER_TYPES
            """)

    function_defs = {
            "loopy_floor_div": r"""
            #define LOOPY_DEFINE_FLOOR_DIV(SUFFIX, TYPE) \
                inline TYPE loopy_floor_div_##SUFFIX(TYPE a, TYPE b) \
                { \
                    if ((a<0) != (b<0)) \
                        a = a - (b + (b<0) - (b>=0)); \
                    return a/b; \
                }
            LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_FLOOR_DIV)
            #undef LOOPY_DEFINE_FLOOR_DIV
            """,

            "loopy_floor_div_pos_b": r"""
            #define LOOPY_DEFINE_FLOOR_DIV_POS_B(SUFFIX, TYPE) \
                inline TYPE loopy_floor_div_pos_b_##SUFFIX(TYPE a, TYPE b) \
                { \
                    if (a<0) \
                        a = a - (b-1); \
                    return a/b; \
                }
            LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_FLOOR_DIV_POS_B)
            #undef LOOPY_DEFINE_FLOOR_DIV_POS_B
            """,

            "loopy_mod": r"""
            #define LOOPY_DEFINE_MOD(SUFFIX, TYPE) \
                inline TYPE loopy_mod_##SUFFIX(TYPE a, TYPE b) \
                { \
                    TYPE result = a%b; \
                    if (result < 0 && b > 0) \
                        result += b; \
                    if (result > 0 && b < 0) \
                        result = result + b; \
                    return result; \
                }
            LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_MOD)
            #undef LOOPY_DEFINE_MOD
            """,

            "loopy_mod_pos_b": r"""
            #define LOOPY_DEFINE_MOD_POS_B(SUFFIX, TYPE) \
                inline TYPE loopy_mod_pos_b_##SUFFIX(TYPE a, TYPE b) \
                { \
                    TYPE result = a%b; \
                    if (result < 0) \
                        result += b; \
                    return result; \
                }
            LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_MOD_POS_B)
            #undef LOOPY_DEFINE_MOD_POS_B
            """,
            }

    c_funcs = {func.c_name for func in preamble_info.seen_functions}

    for func_name, func_body in function_defs.items():
        if any((func_name + "_" + tpname) in c_funcs
                for tpname in integer_type_names):
            yield def_integer_types_macro
            yield ("04_%s" % func_name, func_body)
            yield undef_integer_types_macro

    for func in preamble_info.seen_functions:
        if func.name == "int_pow":
            base_ctype = preamble_info.kernel.target.dtype_to_typename(
                    func.arg_dtypes[0])
            exp_ctype = preamble_info.kernel.target.dtype_to_typename(
                    func.arg_dtypes[1])
            res_ctype = preamble_info.kernel.target.dtype_to_typename(
                    func.result_dtypes[0])

            if func.arg_dtypes[1].numpy_dtype.kind == "u":
                signed_exponent_preamble = ""
            else:
                signed_exponent_preamble = "\n" + remove_common_indentation(
                        """
                        if (n < 0) {
                          x = 1.0/x;
                          n =  -n;
                        }""")

            yield(f"07_{func.c_name}", f"""
            inline {res_ctype} {func.c_name}({base_ctype} x, {exp_ctype} n) {{
              if (n == 0)
                return 1;
              {re.sub("^", 14*" ", signed_exponent_preamble, flags=re.M)}

              {res_ctype} y = 1;

              while (n > 1) {{
                if (n % 2) {{
                  y = x * y;
                  x = x * x;
                }}
                else
                  x = x * x;
                n = n / 2;
              }}

              return x*y;
            }}""")

# }}}


# {{{ cgen overrides

from cgen import Declarator


class POD(Declarator):
    """A simple declarator: The type is given as a :class:`numpy.dtype`
    and the *name* is given as a string.
    """

    def __init__(self, ast_builder, dtype, name):
        from loopy.types import LoopyType
        assert isinstance(dtype, LoopyType)

        self.ast_builder = ast_builder
        self.ctype = ast_builder.target.dtype_to_typename(dtype)
        self.dtype = dtype
        self.name = name

    def get_decl_pair(self):
        return [self.ctype], self.name

    def struct_maker_code(self, name):
        return name

    def struct_format(self):
        return self.dtype.char

    def alignment_requirement(self):
        return self.ast_builder.target.alignment_requirement(self)

    def default_value(self):
        return 0

    mapper_method = "map_loopy_pod"


class ScopingBlock(Block):
    """A block that is mandatory for scoping and may not be simplified away
    by :func:`loopy.codegen.result.merge_codegen_results`.
    """


class FunctionDeclarationWrapper(NestedDeclarator):
    mapper_method = "map_function_decl_wrapper"

# }}}


# {{{ array literals

def generate_linearized_array(array, value):
    from pytools import product
    size = product(shape_ax for shape_ax in array.shape)

    if not isinstance(size, int):
        raise LoopyError("cannot produce literal for array '%s': "
                "shape is not a compile-time constant"
                % array.name)

    strides = []

    data = np.zeros(size, array.dtype.numpy_dtype)

    from loopy.kernel.array import FixedStrideArrayDimTag
    for i, dim_tag in enumerate(array.dim_tags):
        if isinstance(dim_tag, FixedStrideArrayDimTag):

            if not isinstance(dim_tag.stride, int):
                raise LoopyError("cannot produce literal for array '%s': "
                        "stride along axis %d (1-based) is not a "
                        "compile-time constant"
                        % (array.name, i+1))

            strides.append(dim_tag.stride)

        else:
            raise LoopyError("cannot produce literal for array '%s': "
                    "dim_tag type '%s' not supported"
                    % (array.name, type(dim_tag).__name__))

    assert array.offset == 0

    for ituple in np.ndindex(value.shape):
        i = sum(i_ax * strd_ax for i_ax, strd_ax in zip(ituple, strides))
        data[i] = value[ituple]

    return data


def generate_array_literal(codegen_state, array, value):
    data = generate_linearized_array(array, value)

    ecm = codegen_state.expression_to_code_mapper

    from loopy.expression import dtype_to_type_context
    from loopy.symbolic import ArrayLiteral

    type_context = dtype_to_type_context(codegen_state.kernel.target, array.dtype)
    return CExpression(
            codegen_state.ast_builder.get_c_expression_to_code_mapper(),
            ArrayLiteral(
                tuple(
                    ecm.map_constant(d_i, type_context)
                    for d_i in data)))

# }}}


# {{{ subscript CSE

class CASTIdentityMapper(CASTIdentityMapperBase):
    def map_loopy_pod(self, node, *args, **kwargs):
        return type(node)(node.ast_builder, node.dtype, node.name)

    def map_function_decl_wrapper(self, node, *args, **kwargs):
        return FunctionDeclarationWrapper(
                self.rec(node.subdecl, *args, **kwargs))


class SubscriptSubsetCounter(IdentityMapper):
    def __init__(self, subset_counters):
        self.subset_counters = subset_counters


class ASTSubscriptCollector(CASTIdentityMapper):
    def __init__(self):
        self.subset_counters = {}

    def map_expression(self, expr):
        from pymbolic.primitives import is_constant
        if isinstance(expr, CExpression) or is_constant(expr):
            return expr
        elif isinstance(expr, str):
            return expr
        else:
            raise LoopyError(
                    "Unexpected expression type: %s" % type(expr).__name__)

# }}}


# {{{ lazy expression generation

class CExpression:
    def __init__(self, to_code_mapper, expr):
        self.to_code_mapper = to_code_mapper
        self.expr = expr

    def __str__(self):
        return self.to_code_mapper(self.expr, PREC_NONE)

# }}}


class CFamilyTarget(TargetBase):
    """A target for "least-common denominator C", without any parallel
    extensions, and without use of any C99 specifics. Intended to be
    usable as a common base for C99, C++, OpenCL, CUDA, and the like.
    """

    hash_fields = TargetBase.hash_fields + ("fortran_abi",)
    comparison_fields = TargetBase.comparison_fields + ("fortran_abi",)

    def __init__(self, fortran_abi=False):
        self.fortran_abi = fortran_abi
        super().__init__()

    def split_kernel_at_global_barriers(self):
        return False

    def get_host_ast_builder(self):
        return DummyHostASTBuilder(self)

    def get_device_ast_builder(self):
        return CFamilyASTBuilder(self)

    # {{{ types

    @memoize_method
    def get_dtype_registry(self):
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

    def get_kernel_executor_cache_key(self, *args, **kwargs):
        return None  # TODO: ???

    def get_kernel_executor(self, knl, *args, **kwargs):
        raise NotImplementedError()

    # }}}


class _ConstRestrictPointer(Pointer):
    def get_decl_pair(self):
        sub_tp, sub_decl = self.subdecl.get_decl_pair()
        return sub_tp, ("*const __restrict__ %s" % sub_decl)


class _ConstPointer(Pointer):
    def get_decl_pair(self):
        sub_tp, sub_decl = self.subdecl.get_decl_pair()
        return sub_tp, ("*const %s" % sub_decl)


# {{{ symbol mangler

def c_symbol_mangler(kernel, name):
    # float NAN as defined in C99 standard
    if name == "NAN":
        return NumpyType(np.dtype(np.float32)), name
    return None

# }}}


# {{{ function mangler

def c_math_mangler(target, name, arg_dtypes, modify_name=True):
    # Function mangler for math functions defined in C standard
    # Convert abs, min, max to fabs, fmin, fmax.
    # If modify_name is set to True, function names are modified according to
    # floating point types of the arguments (e.g. cos(double), cosf(float))
    # This should be set to True for C and Cuda, False for OpenCL
    if not isinstance(name, str):
        return None

    # {{{ (abs|max|min) -> (fabs|fmax|fmin)

    if name in ["abs", "min", "max"]:
        dtype = np.find_common_type(
            [], [dtype.numpy_dtype for dtype in arg_dtypes])
        if dtype.kind == "f":
            name = "f" + name

    # }}}

    # unitary functions
    if (name in ["fabs", "acos", "asin", "atan", "cos", "cosh", "sin", "sinh",
                 "tanh", "exp", "log", "log10", "sqrt", "ceil", "floor"]
            and len(arg_dtypes) == 1
            and arg_dtypes[0].numpy_dtype.kind in "fc"):

        dtype = arg_dtypes[0].numpy_dtype
        real_dtype = np.empty(0, dtype=dtype).real.dtype

        if modify_name:
            if real_dtype == np.float64:
                pass  # fabs
            elif real_dtype == np.float32:
                name = name + "f"  # fabsf
            elif (hasattr(np, "float128")
                    and real_dtype == np.float128):  # pylint:disable=no-member
                name = name + "l"  # fabsl
            else:
                raise LoopyTypeError(f"{name} does not support type {real_dtype}")

            if dtype.kind == "c":
                name = "c" + name

        return CallMangleInfo(
                target_name=name,
                result_dtypes=arg_dtypes,
                arg_dtypes=arg_dtypes)

    # binary functions
    if (name in ["fmax", "fmin", "copysign", "pow"]
            and len(arg_dtypes) == 2):

        dtype = np.find_common_type(
            [], [dtype.numpy_dtype for dtype in arg_dtypes])
        real_dtype = np.empty(0, dtype=dtype).real.dtype

        if name in ["fmax", "fmin", "copysign"] and dtype.kind == "c":
            raise LoopyTypeError(f"{name} does not support complex numbers")

        elif real_dtype.kind in "fc":
            if modify_name:
                if real_dtype == np.float64:
                    pass  # fmin
                elif real_dtype == np.float32:
                    name = name + "f"  # fminf
                elif (hasattr(np, "float128")
                        and real_dtype == np.float128):  # pylint:disable=no-member
                    name = name + "l"  # fminl
                else:
                    raise LoopyTypeError("%s does not support type %s"
                                         % (name, real_dtype))

                if dtype.kind == "c":
                    name = "c" + name  # cpow

            result_dtype = NumpyType(dtype)
            return CallMangleInfo(
                    target_name=name,
                    result_dtypes=(result_dtype,),
                    arg_dtypes=2*(result_dtype,))

    # complex functions
    if (name in ["abs", "real", "imag"]
            and len(arg_dtypes) == 1
            and arg_dtypes[0].numpy_dtype.kind == "c"):
        dtype = arg_dtypes[0].numpy_dtype
        real_dtype = np.empty(0, dtype=dtype).real.dtype

        if modify_name:
            if real_dtype == np.float64:
                pass  # fabs
            elif real_dtype == np.float32:
                name = name + "f"  # fabsf
            elif (hasattr(np, "float128")
                    and real_dtype == np.float128):  # pylint:disable=no-member
                name = name + "l"  # fabsl
            else:
                raise LoopyTypeError(f"{name} does not support type {real_dtype}")

            name = "c" + name

        return CallMangleInfo(
                target_name=name,
                result_dtypes=(NumpyType(real_dtype),),
                arg_dtypes=arg_dtypes)

    if (name == "isnan" and len(arg_dtypes) == 1
            and arg_dtypes[0].numpy_dtype.kind == "f"):
        return CallMangleInfo(
                target_name=name,
                result_dtypes=(NumpyType(np.int32),),
                arg_dtypes=arg_dtypes)

    return None

# }}}


class CFamilyASTBuilder(ASTBuilderBase):
    # {{{ library

    def function_manglers(self):
        return (
                super().function_manglers() + [
                    c_math_mangler
                    ])

    def symbol_manglers(self):
        return (
                super().symbol_manglers() + [
                    c_symbol_mangler
                    ])

    def preamble_generators(self):
        return (
                super().preamble_generators() + [
                    _preamble_generator,
                    ])

    # }}}

    # {{{ code generation

    def get_function_definition(self, codegen_state, codegen_result,
            schedule_index,
            function_decl, function_body):
        kernel = codegen_state.kernel

        from cgen import (
                FunctionBody,

                # Post-mid-2016 cgens have 'Collection', too.
                Module as Collection,
                Initializer,
                Line)

        result = []

        from loopy.kernel.data import AddressSpace
        from loopy.schedule import CallKernel
        # We only need to write declarations for global variables with
        # the first device program. `is_first_dev_prog` determines
        # whether this is the first device program in the schedule.
        is_first_dev_prog = codegen_state.is_generating_device_code
        for i in range(schedule_index):
            if isinstance(kernel.schedule[i], CallKernel):
                is_first_dev_prog = False
                break
        if is_first_dev_prog:
            for tv in sorted(
                    kernel.temporary_variables.values(),
                    key=lambda tv: tv.name):

                if tv.address_space == AddressSpace.GLOBAL and (
                        tv.initializer is not None):
                    assert tv.read_only

                    decl_info, = tv.decl_info(self.target,
                                    index_dtype=kernel.index_dtype)
                    decl = self.wrap_global_constant(
                            self.get_temporary_decl(
                                codegen_state, schedule_index, tv,
                                decl_info))

                    if tv.initializer is not None:
                        decl = Initializer(decl, generate_array_literal(
                            codegen_state, tv, tv.initializer))

                    result.append(decl)

        fbody = FunctionBody(function_decl, function_body)
        if not result:
            return fbody
        else:
            return Collection(result+[Line(), fbody])

    def idi_to_cgen_declarator(self, kernel, idi):
        from loopy.kernel.data import InameArg
        if (idi.offset_for_name is not None
                or idi.stride_for_name_and_axis is not None):
            assert not idi.is_written
            from cgen import Const
            return Const(POD(self, idi.dtype, idi.name))
        elif issubclass(idi.arg_class, InameArg):
            return InameArg(idi.name, idi.dtype).get_arg_decl(self)
        else:
            name = idi.base_name or idi.name
            var_descr = kernel.get_var_descriptor(name)
            from loopy.kernel.data import ArrayBase
            if isinstance(var_descr, ArrayBase):
                return var_descr.get_arg_decl(
                        self,
                        idi.name[len(name):], idi.shape, idi.dtype,
                        idi.is_written)
            else:
                return var_descr.get_arg_decl(self)

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        from cgen import FunctionDeclaration, Value

        name = codegen_result.current_program(codegen_state).name
        if self.target.fortran_abi:
            name += "_"

        return FunctionDeclarationWrapper(
                FunctionDeclaration(
                    Value("void", name),
                    [self.idi_to_cgen_declarator(codegen_state.kernel, idi)
                        for idi in codegen_state.implemented_data_info]))

    def get_kernel_call(self, codegen_state, name, gsize, lsize, extra_args):
        return None

    def get_temporary_decls(self, codegen_state, schedule_index):
        from loopy.kernel.data import AddressSpace

        kernel = codegen_state.kernel

        base_storage_decls = []
        temp_decls = []

        # {{{ declare temporaries

        base_storage_sizes = {}
        base_storage_to_scope = {}
        base_storage_to_align_bytes = {}

        from cgen import ArrayOf, Initializer, AlignedAttribute, Value, Line
        # Getting the temporary variables that are needed for the current
        # sub-kernel.
        from loopy.schedule.tools import (
                temporaries_read_in_subkernel,
                temporaries_written_in_subkernel)
        subkernel = kernel.schedule[schedule_index].kernel_name
        sub_knl_temps = (
                temporaries_read_in_subkernel(kernel, subkernel) |
                temporaries_written_in_subkernel(kernel, subkernel))

        for tv in sorted(
                kernel.temporary_variables.values(),
                key=lambda tv: tv.name):
            decl_info = tv.decl_info(self.target, index_dtype=kernel.index_dtype)

            if not tv.base_storage:
                for idi in decl_info:
                    # global temp vars are mapped to arguments or global declarations
                    if tv.address_space != AddressSpace.GLOBAL and (
                            tv.name in sub_knl_temps):
                        decl = self.wrap_temporary_decl(
                                self.get_temporary_decl(
                                    codegen_state, schedule_index, tv, idi),
                                tv.address_space)

                        if tv.initializer is not None:
                            assert tv.read_only
                            decl = Initializer(decl, generate_array_literal(
                                codegen_state, tv, tv.initializer))

                        temp_decls.append(decl)

            else:
                assert tv.initializer is None

                offset = 0
                base_storage_sizes.setdefault(tv.base_storage, []).append(
                        tv.nbytes)
                base_storage_to_scope.setdefault(tv.base_storage, []).append(
                        tv.address_space)

                align_size = tv.dtype.itemsize

                from loopy.kernel.array import VectorArrayDimTag
                for dim_tag, axis_len in zip(tv.dim_tags, tv.shape):
                    if isinstance(dim_tag, VectorArrayDimTag):
                        align_size *= axis_len

                base_storage_to_align_bytes.setdefault(tv.base_storage, []).append(
                        align_size)

                for idi in decl_info:
                    cast_decl = POD(self, idi.dtype, "")
                    temp_var_decl = POD(self, idi.dtype, idi.name)

                    cast_decl = self.wrap_temporary_decl(cast_decl, tv.address_space)
                    temp_var_decl = self.wrap_temporary_decl(
                            temp_var_decl, tv.address_space)

                    if tv._base_storage_access_may_be_aliasing:
                        ptrtype = _ConstPointer
                    else:
                        # The 'restrict' part of this is a complete lie--of course
                        # all these temporaries are aliased. But we're promising to
                        # not use them to shovel data from one representation to the
                        # other. That counts, right?
                        ptrtype = _ConstRestrictPointer

                    cast_decl = ptrtype(cast_decl)
                    temp_var_decl = ptrtype(temp_var_decl)

                    cast_tp, cast_d = cast_decl.get_decl_pair()
                    temp_var_decl = Initializer(
                            temp_var_decl,
                            "({} {}) ({} + {})".format(
                                " ".join(cast_tp), cast_d,
                                tv.base_storage,
                                offset))

                    temp_decls.append(temp_var_decl)

                    from pytools import product
                    offset += (
                            idi.dtype.itemsize
                            * product(si for si in idi.shape))

        ecm = self.get_expression_to_code_mapper(codegen_state)

        for bs_name, bs_sizes in sorted(base_storage_sizes.items()):
            bs_var_decl = Value("char", bs_name)
            from pytools import single_valued
            bs_var_decl = self.wrap_temporary_decl(
                    bs_var_decl, single_valued(base_storage_to_scope[bs_name]))

            # FIXME: Could try to use isl knowledge to simplify max.
            if all(isinstance(bs, int) for bs in bs_sizes):
                bs_size_max = max(bs_sizes)
            else:
                bs_size_max = p.Max(tuple(bs_sizes))

            bs_var_decl = ArrayOf(bs_var_decl, ecm(bs_size_max))

            alignment = max(base_storage_to_align_bytes[bs_name])
            bs_var_decl = AlignedAttribute(alignment, bs_var_decl)

            base_storage_decls.append(bs_var_decl)

        # }}}

        result = base_storage_decls + temp_decls

        if result:
            result.append(Line())

        return result

    @property
    def ast_block_class(self):
        from cgen import Block
        return Block

    @property
    def ast_block_scope_class(self):
        return ScopingBlock

    # }}}

    # {{{ code generation guts

    def get_expression_to_code_mapper(self, codegen_state):
        return self.get_expression_to_c_expression_mapper(codegen_state)

    def get_expression_to_c_expression_mapper(self, codegen_state):
        from loopy.target.c.codegen.expression import ExpressionToCExpressionMapper
        return ExpressionToCExpressionMapper(
                codegen_state, fortran_abi=self.target.fortran_abi)

    def get_c_expression_to_code_mapper(self):
        from loopy.target.c.codegen.expression import CExpressionToCodeMapper
        return CExpressionToCodeMapper()

    def get_temporary_decl(self, codegen_state, schedule_index, temp_var, decl_info):
        temp_var_decl = POD(self, decl_info.dtype, decl_info.name)

        if temp_var.read_only:
            from cgen import Const
            temp_var_decl = Const(temp_var_decl)

        if decl_info.shape:
            from cgen import ArrayOf
            ecm = self.get_expression_to_code_mapper(codegen_state)
            temp_var_decl = ArrayOf(temp_var_decl,
                    ecm(p.flattened_product(decl_info.shape),
                        prec=PREC_NONE, type_context="i"))

        if temp_var.alignment:
            from cgen import AlignedAttribute
            temp_var_decl = AlignedAttribute(temp_var.alignment, temp_var_decl)

        return temp_var_decl

    def wrap_temporary_decl(self, decl, scope):
        return decl

    def wrap_global_constant(self, decl):
        from cgen import Static
        return Static(decl)

    def get_value_arg_decl(self, name, shape, dtype, is_written):
        assert shape == ()

        result = POD(self, dtype, name)

        if not is_written:
            from cgen import Const
            result = Const(result)

        if self.target.fortran_abi:
            from cgen import Pointer
            result = Pointer(result)

        return result

    def get_array_arg_decl(self, name, mem_address_space, shape, dtype, is_written):
        from cgen import RestrictPointer, Const

        arg_decl = RestrictPointer(POD(self, dtype, name))

        if not is_written:
            arg_decl = Const(arg_decl)

        return arg_decl

    def get_global_arg_decl(self, name, shape, dtype, is_written):
        from warnings import warn
        warn("get_global_arg_decl is deprecated use get_array_arg_decl "
                "instead.", DeprecationWarning, stacklevel=2)
        from loopy.kernel.data import AddressSpace
        return self.get_array_arg_decl(name, AddressSpace.GLOBAL, shape,
                dtype, is_written)

    def get_constant_arg_decl(self, name, shape, dtype, is_written):
        from loopy.target.c import POD  # uses the correct complex type
        from cgen import RestrictPointer, Const

        arg_decl = RestrictPointer(POD(self, dtype, name))

        if not is_written:
            arg_decl = Const(arg_decl)

        return arg_decl

    def emit_assignment(self, codegen_state, insn):
        kernel = codegen_state.kernel
        ecm = codegen_state.expression_to_code_mapper

        assignee_var_name, = insn.assignee_var_names()

        lhs_var = codegen_state.kernel.get_var_descriptor(assignee_var_name)
        lhs_dtype = lhs_var.dtype

        if insn.atomicity is not None:
            lhs_atomicity = [
                    a for a in insn.atomicity if a.var_name == assignee_var_name]
            assert len(lhs_atomicity) <= 1
            if lhs_atomicity:
                lhs_atomicity, = lhs_atomicity
            else:
                lhs_atomicity = None
        else:
            lhs_atomicity = None

        from loopy.kernel.data import AtomicInit, AtomicUpdate
        from loopy.expression import dtype_to_type_context

        lhs_code = ecm(insn.assignee, prec=PREC_NONE, type_context=None)
        rhs_type_context = dtype_to_type_context(kernel.target, lhs_dtype)
        if lhs_atomicity is None:
            from cgen import Assign
            return Assign(
                    lhs_code,
                    ecm(insn.expression, prec=PREC_NONE,
                        type_context=rhs_type_context,
                        needed_dtype=lhs_dtype))

        elif isinstance(lhs_atomicity, AtomicInit):
            codegen_state.seen_atomic_dtypes.add(lhs_dtype)
            return codegen_state.ast_builder.emit_atomic_init(
                    codegen_state, lhs_atomicity, lhs_var,
                    insn.assignee, insn.expression,
                    lhs_dtype, rhs_type_context)

        elif isinstance(lhs_atomicity, AtomicUpdate):
            codegen_state.seen_atomic_dtypes.add(lhs_dtype)
            return codegen_state.ast_builder.emit_atomic_update(
                    codegen_state, lhs_atomicity, lhs_var,
                    insn.assignee, insn.expression,
                    lhs_dtype, rhs_type_context)

        else:
            raise ValueError("unexpected lhs atomicity type: %s"
                    % type(lhs_atomicity).__name__)

    def emit_atomic_update(self, codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype):
        raise NotImplementedError("atomic updates in %s" % type(self).__name__)

    def emit_tuple_assignment(self, codegen_state, insn):
        ecm = codegen_state.expression_to_code_mapper

        from cgen import Assign, block_if_necessary
        assignments = []

        for i, (assignee, parameter) in enumerate(
                zip(insn.assignees, insn.expression.parameters)):
            lhs_code = ecm(assignee, prec=PREC_NONE, type_context=None)
            assignee_var_name = insn.assignee_var_names()[i]
            lhs_var = codegen_state.kernel.get_var_descriptor(assignee_var_name)
            lhs_dtype = lhs_var.dtype

            from loopy.expression import dtype_to_type_context
            rhs_type_context = dtype_to_type_context(
                    codegen_state.kernel.target, lhs_dtype)
            rhs_code = ecm(parameter, prec=PREC_NONE,
                    type_context=rhs_type_context, needed_dtype=lhs_dtype)

            assignments.append(Assign(lhs_code, rhs_code))

        return block_if_necessary(assignments)

    def emit_multiple_assignment(self, codegen_state, insn):
        ecm = codegen_state.expression_to_code_mapper

        from pymbolic.primitives import Variable
        from pymbolic.mapper.stringifier import PREC_NONE

        func_id = insn.expression.function
        parameters = insn.expression.parameters

        if isinstance(func_id, Variable):
            func_id = func_id.name

        assignee_var_descriptors = [
                codegen_state.kernel.get_var_descriptor(a)
                for a in insn.assignee_var_names()]

        par_dtypes = tuple(ecm.infer_type(par) for par in parameters)

        mangle_result = codegen_state.kernel.mangle_function(func_id, par_dtypes)
        if mangle_result is None:
            raise RuntimeError("function '%s' unknown--"
                    "maybe you need to register a function mangler?"
                    % func_id)

        assert mangle_result.arg_dtypes is not None

        if mangle_result.target_name == "loopy_make_tuple":
            # This shorcut avoids actually having to emit a 'make_tuple' function.
            return self.emit_tuple_assignment(codegen_state, insn)

        from loopy.expression import dtype_to_type_context
        c_parameters = [
                ecm(par, PREC_NONE,
                    dtype_to_type_context(self.target, tgt_dtype),
                    tgt_dtype).expr
                for par, par_dtype, tgt_dtype in zip(
                    parameters, par_dtypes, mangle_result.arg_dtypes)]

        from loopy.codegen import SeenFunction
        codegen_state.seen_functions.add(
                SeenFunction(func_id,
                    mangle_result.target_name,
                    mangle_result.arg_dtypes,
                    mangle_result.result_dtypes))

        from pymbolic import var
        for i, (a, tgt_dtype) in enumerate(
                zip(insn.assignees[1:], mangle_result.result_dtypes[1:])):
            if tgt_dtype != ecm.infer_type(a):
                raise LoopyError("type mismatch in %d'th (1-based) left-hand "
                        "side of instruction '%s'" % (i+1, insn.id))
            c_parameters.append(
                        # TODO Yuck: The "where-at function": &(...)
                        var("&")(
                            ecm(a, PREC_NONE,
                                dtype_to_type_context(self.target, tgt_dtype),
                                tgt_dtype).expr))

        from pymbolic import var
        result = var(mangle_result.target_name)(*c_parameters)

        # In case of no assignees, we are done
        if len(mangle_result.result_dtypes) == 0:
            from cgen import ExpressionStatement
            return ExpressionStatement(
                    CExpression(self.get_c_expression_to_code_mapper(), result))

        result = ecm.wrap_in_typecast_lazy(
                lambda: mangle_result.result_dtypes[0],
                assignee_var_descriptors[0].dtype,
                result)

        lhs_code = ecm(insn.assignees[0], prec=PREC_NONE, type_context=None)

        from cgen import Assign
        return Assign(
                lhs_code,
                CExpression(self.get_c_expression_to_code_mapper(), result))

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            lbound, ubound, inner):
        ecm = codegen_state.expression_to_code_mapper

        from pymbolic import var
        from pymbolic.primitives import Comparison
        from pymbolic.mapper.stringifier import PREC_NONE
        from cgen import For, InlineInitializer

        return For(
                InlineInitializer(
                    POD(self, iname_dtype, iname),
                    ecm(lbound, PREC_NONE, "i")),
                ecm(
                    Comparison(
                        var(iname),
                        "<=",
                        ubound),
                    PREC_NONE, "i"),
                "++%s" % iname,
                inner)

    def emit_initializer(self, codegen_state, dtype, name, val_str, is_const):
        decl = POD(self, dtype, name)

        from cgen import Initializer, Const

        if is_const:
            decl = Const(decl)

        return Initializer(decl, val_str)

    def emit_blank_line(self):
        from cgen import Line
        return Line()

    def emit_comment(self, s):
        from cgen import Comment
        return Comment(s)

    @property
    def can_implement_conditionals(self):
        return True

    def emit_if(self, condition_str, ast):
        from cgen import If
        return If(condition_str, ast)

    # }}}

    def process_ast(self, node):
        sc = ASTSubscriptCollector()
        sc(node)
        return node


# {{{ header generation

class CFunctionDeclExtractor(CASTIdentityMapper):
    def __init__(self):
        self.decls = []

    def map_expression(self, expr):
        return expr

    def map_function_decl_wrapper(self, node):
        self.decls.append(node.subdecl)
        return super()\
                .map_function_decl_wrapper(node)


def generate_header(kernel, codegen_result=None):
    """
    :arg kernel: a :class:`loopy.LoopKernel`
    :arg codegen_result: an instance of :class:`loopy.CodeGenerationResult`
    :returns: a list of AST nodes (which may have :class:`str`
        called on them to produce a string) representing
        function declarations for the generated device
        functions.
    """

    if not isinstance(kernel.target, CFamilyTarget):
        raise LoopyError(
                "Header generation for non C-based languages are not implemented")

    if codegen_result is None:
        from loopy.codegen import generate_code_v2
        codegen_result = generate_code_v2(kernel)

    fde = CFunctionDeclExtractor()
    for dev_prg in codegen_result.device_programs:
        fde(dev_prg.ast)

    return fde.decls

# }}}


# {{{ C99 target

class CTarget(CFamilyTarget):
    """This target may emit code using all features of C99.
    For a target base supporting "least-common-denominator" C,
    see :class:`CFamilyTarget`.
    """

    def get_device_ast_builder(self):
        return CASTBuilder(self)

    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c.compyte.dtypes import (
                DTypeRegistry, fill_registry_with_c99_stdint_types,
                fill_registry_with_c99_complex_types)
        result = DTypeRegistry()
        fill_registry_with_c99_stdint_types(result)
        fill_registry_with_c99_complex_types(result)
        return DTypeRegistryWrapper(result)


class CASTBuilder(CFamilyASTBuilder):
    def preamble_generators(self):
        return (
                super().preamble_generators() + [
                    c99_preamble_generator,
                    ])

# }}}


# {{{ executable c target

class ExecutableCTarget(CTarget):
    """
    An executable CFamilyTarget that uses (by default) JIT compilation of C-code
    """

    def __init__(self, compiler=None, fortran_abi=False):
        super().__init__(fortran_abi=fortran_abi)
        from loopy.target.c.c_execution import CCompiler
        self.compiler = compiler or CCompiler()

    def get_kernel_executor(self, knl, *args, **kwargs):
        from loopy.target.c.c_execution import CKernelExecutor
        return CKernelExecutor(knl, compiler=self.compiler)

    def get_host_ast_builder(self):
        # enable host code generation
        return CFamilyASTBuilder(self)

# }}}

# vim: foldmethod=marker

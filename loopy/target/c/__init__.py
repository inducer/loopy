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

from typing import cast, Tuple, Optional, Sequence, Any
import re

import numpy as np  # noqa

from cgen import (Collection, Pointer, NestedDeclarator, Block, Generable,
                  Declarator, Const)
from cgen.mapper import IdentityMapper as CASTIdentityMapperBase
from pymbolic.mapper.stringifier import PREC_NONE
import pymbolic.primitives as p
from pytools import memoize_method

from loopy.target import TargetBase, ASTBuilderBase, DummyHostASTBuilder
from loopy.diagnostic import LoopyError, LoopyTypeError
from loopy.symbolic import IdentityMapper
from loopy.target.execution import ExecutorBase
from loopy.translation_unit import FunctionIdT, TranslationUnit
from loopy.types import NumpyType, LoopyType
from loopy.typing import ExpressionT
from loopy.kernel import LoopKernel
from loopy.kernel.array import ArrayBase, FixedStrideArrayDimTag
from loopy.kernel.data import (TemporaryVariable, AddressSpace, ArrayArg,
        ConstantArg, ImageArg, ValueArg)
from loopy.kernel.function_interface import ScalarCallable
from loopy.schedule import CallKernel
from loopy.tools import remove_common_indentation
from loopy.codegen import CodeGenerationState
from loopy.codegen.result import CodeGenerationResult


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
        from loopy.types import LoopyType, NumpyType, OpaqueType
        assert isinstance(dtype, LoopyType)

        if isinstance(dtype, NumpyType):
            return self.wrapped_registry.dtype_to_ctype(dtype)
        elif isinstance(dtype, OpaqueType):
            return dtype.name
        else:
            raise LoopyError(
                    "unable to convert type '%s' to C"
                    % dtype)

# }}}


# {{{ preamble generator

class InfOrNanInExpressionRecorder(IdentityMapper):
    def __init__(self):
        self.saw_inf_or_nan = False
        super().__init__()

    def map_constant(self, expr):
        if (np.isinf(expr) or np.isnan(expr) or np.isnan(expr)):
            self.saw_inf_or_nan = True
        return super().map_constant(expr)

    def map_nan(self, expr):
        self.saw_inf_or_nan = True
        return super().map_nan(expr)


def c99_preamble_generator(preamble_info):
    if any(dtype.is_integral() for dtype in preamble_info.seen_dtypes):
        yield ("10_stdint", "#include <stdint.h>")
    if any(dtype.numpy_dtype == np.dtype("bool")
           for dtype in preamble_info.seen_dtypes
           if isinstance(dtype, NumpyType)):
        yield ("10_stdbool", "#include <stdbool.h>")
    if any(dtype.is_complex() for dtype in preamble_info.seen_dtypes):
        yield ("10_complex", "#include <complex.h>")

    # {{{ emit math.h

    inf_or_nan_recorder = InfOrNanInExpressionRecorder()

    for insn in preamble_info.codegen_state.kernel.instructions:
        insn.with_transformed_expressions(inf_or_nan_recorder)

    if inf_or_nan_recorder.saw_inf_or_nan:
        yield ("10_math", "#include <math.h>")

    # }}}


def _preamble_generator(preamble_info, func_qualifier="inline"):
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
                {} TYPE loopy_floor_div_##SUFFIX(TYPE a, TYPE b) \
                {{ \
                    if ((a<0) != (b<0)) \
                        a = a - (b + (b<0) - (b>=0)); \
                    return a/b; \
                }}
            LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_FLOOR_DIV)
            #undef LOOPY_DEFINE_FLOOR_DIV
            """.format(func_qualifier),

            "loopy_floor_div_pos_b": r"""
            #define LOOPY_DEFINE_FLOOR_DIV_POS_B(SUFFIX, TYPE) \
                {} TYPE loopy_floor_div_pos_b_##SUFFIX(TYPE a, TYPE b) \
                {{ \
                    if (a<0) \
                        a = a - (b-1); \
                    return a/b; \
                }}
            LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_FLOOR_DIV_POS_B)
            #undef LOOPY_DEFINE_FLOOR_DIV_POS_B
            """.format(func_qualifier),

            "loopy_mod": r"""
            #define LOOPY_DEFINE_MOD(SUFFIX, TYPE) \
                {} TYPE loopy_mod_##SUFFIX(TYPE a, TYPE b) \
                {{ \
                    TYPE result = a%b; \
                    if (result < 0 && b > 0) \
                        result += b; \
                    if (result > 0 && b < 0) \
                        result = result + b; \
                    return result; \
                }}
            LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_MOD)
            #undef LOOPY_DEFINE_MOD
            """.format(func_qualifier),

            "loopy_mod_pos_b": r"""
            #define LOOPY_DEFINE_MOD_POS_B(SUFFIX, TYPE) \
                {} TYPE loopy_mod_pos_b_##SUFFIX(TYPE a, TYPE b) \
                {{ \
                    TYPE result = a%b; \
                    if (result < 0) \
                        result += b; \
                    return result; \
                }}
            LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_MOD_POS_B)
            #undef LOOPY_DEFINE_MOD_POS_B
            """.format(func_qualifier),
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

            yield (f"07_{func.c_name}", f"""
            inline {res_ctype} {func.c_name}({base_ctype} x, {exp_ctype} n) {{
              if (n == 0)
                return 1;
              {re.sub("^", 14*" ", signed_exponent_preamble, flags=re.M)}

              {res_ctype} y = 1;

              while (n > 1) {{
                if (n % 2)
                  y = x * y;
                x = x * x;
                n = n / 2;
              }}

              return x*y;
            }}""")

# }}}


# {{{ cgen overrides

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

    if name in ["INT_MAX", "INT_MIN"]:
        return NumpyType(np.dtype(np.int32)), name

    return None

# }}}


# {{{ function scoping

class CMathCallable(ScalarCallable):
    """
    An umbrella callable for all the math functions which can be seen in a
    C-Target.
    """

    def with_types(self, arg_id_to_dtype, callables_table):
        name = self.name

        # {{{ (abs|max|min) -> (fabs|fmax|fmin)

        if name in ["abs", "min", "max"]:
            dtype = np.result_type(*[
                    dtype.numpy_dtype for dtype in arg_id_to_dtype.values()])
            if dtype.kind == "f":
                name = "f" + name

        # }}}

        # unary functions
        if name in ["fabs", "acos", "asin", "atan", "cos", "cosh", "sin", "sinh",
                    "tan", "tanh", "exp", "log", "log10", "sqrt", "ceil", "floor",
                    "erf", "erfc", "abs", "real", "imag", "conj"]:

            for id in arg_id_to_dtype:
                if not -1 <= id <= 0:
                    raise LoopyError(f"'{name}' can take only one argument.")

            if 0 not in arg_id_to_dtype or arg_id_to_dtype[0] is None:
                # the types provided aren't mature enough to specialize the
                # callable
                return (
                        self.copy(arg_id_to_dtype=arg_id_to_dtype),
                        callables_table)

            dtype = arg_id_to_dtype[0].numpy_dtype
            real_dtype = np.empty(0, dtype=dtype).real.dtype

            if dtype.kind in ("u", "i"):
                # ints and unsigned casted to float32
                dtype = np.float32

            # for CUDA, C Targets the name must be modified
            if real_dtype == np.float64:
                pass  # fabs
            elif real_dtype == np.float32:
                name = name + "f"  # fabsf
            elif (hasattr(np, "float128")
                    and real_dtype == np.float128):  # pylint:disable=no-member
                name = name + "l"  # fabsl
            else:
                raise LoopyTypeError("{} does not support type {}".format(name,
                    dtype))

            if name in ["abs", "real", "imag"]:
                dtype = real_dtype

            if dtype.kind == "c" or name in ["real", "imag", "abs"]:
                if name != "conj":
                    name = "c" + name

            return (
                    self.copy(name_in_target=name,
                        arg_id_to_dtype={0: NumpyType(dtype), -1:
                            NumpyType(dtype)}),
                    callables_table)

        # binary functions
        elif name in ["fmax", "fmin", "pow", "atan2", "copysign"]:

            for id in arg_id_to_dtype:
                if not -1 <= id <= 1:
                    raise LoopyError("%s can take only two arguments." % name)

            if 0 not in arg_id_to_dtype or 1 not in arg_id_to_dtype or (
                    arg_id_to_dtype[0] is None or arg_id_to_dtype[1] is None):
                # the types provided aren't mature enough to specialize the
                # callable
                return (
                        self.copy(arg_id_to_dtype=arg_id_to_dtype),
                        callables_table)

            dtype = np.result_type(*[
                    dtype.numpy_dtype for id, dtype in arg_id_to_dtype.items()
                    if id >= 0])
            real_dtype = np.empty(0, dtype=dtype).real.dtype

            if name in ["fmax", "fmin", "copysign"] and dtype.kind == "c":
                raise LoopyTypeError(f"{name} does not support complex numbers")

            elif real_dtype.kind in "fc":
                if real_dtype == np.float64:
                    pass  # fmin
                elif real_dtype == np.float32:
                    name = name + "f"  # fminf
                elif (hasattr(np, "float128")
                        and real_dtype == np.float128):  # pylint:disable=no-member
                    name = name + "l"  # fminl
                else:
                    raise LoopyTypeError("%s does not support type %s"
                                         % (name, dtype))
            if dtype.kind == "c":
                name = "c" + name  # cpow
            dtype = NumpyType(dtype)
            return (
                    self.copy(name_in_target=name,
                        arg_id_to_dtype={-1: dtype, 0: dtype, 1: dtype}),
                    callables_table)
        elif name in ["max", "min"]:

            for id in arg_id_to_dtype:
                if not -1 <= id <= 1:
                    raise LoopyError("%s can take only two arguments." % name)

            if 0 not in arg_id_to_dtype or 1 not in arg_id_to_dtype or (
                    arg_id_to_dtype[0] is None or arg_id_to_dtype[1] is None):
                # the types provided aren't resolved enough to specialize the
                # callable
                return (
                        self.copy(arg_id_to_dtype=arg_id_to_dtype),
                        callables_table)

            dtype = np.result_type(*[
                    dtype.numpy_dtype for id, dtype in arg_id_to_dtype.items()
                    if id >= 0])
            if dtype.kind not in "iu":
                # only support integers for now to avoid having to deal with NaNs
                raise LoopyError(f"{name} does not support '{dtype}' arguments.")

            return (
                    self.copy(name_in_target=f"lpy_{name}_{dtype.name}",
                              arg_id_to_dtype={-1: NumpyType(dtype),
                                               0: NumpyType(dtype),
                                               1: NumpyType(dtype)}),
                    callables_table)
        elif name == "isnan":
            for id in arg_id_to_dtype:
                if not -1 <= id <= 0:
                    raise LoopyError(f"'{name}' can take only one argument.")

            if 0 not in arg_id_to_dtype or arg_id_to_dtype[0] is None:
                # the types provided aren't mature enough to specialize the
                # callable
                return (
                        self.copy(arg_id_to_dtype=arg_id_to_dtype),
                        callables_table)

            dtype = arg_id_to_dtype[0].numpy_dtype

            if dtype.kind == "f":
                pass
            elif dtype == np.int32:
                name = "isnani32"
            elif dtype == np.int64:
                name = "isnani64"
            else:
                raise LoopyTypeError(f"'isnan' does not support type {dtype}.")

            return (
                    self.copy(
                        name_in_target=name,
                        arg_id_to_dtype={
                            0: NumpyType(dtype),
                            -1: NumpyType(np.int32)}),
                    callables_table)

    def generate_preambles(self, target):
        if self.name_in_target.startswith("lpy_max"):
            dtype = self.arg_id_to_dtype[-1]
            ctype = target.dtype_to_typename(dtype)

            yield ("40_lpy_max", f"""
            static inline {ctype} {self.name_in_target}({ctype} a, {ctype} b) {{
              return (a > b ? a : b);
            }}""")

        if self.name_in_target.startswith("lpy_min"):
            dtype = self.arg_id_to_dtype[-1]
            ctype = target.dtype_to_typename(dtype)
            yield ("40_lpy_min", f"""
            static inline {ctype} {self.name_in_target}({ctype} a, {ctype} b) {{
              return (a < b ? a : b);
            }}""")

        if self.name == "isnan" and self.name_in_target in {"isnani32",
                                                            "isnani64"}:
            dtype = self.arg_id_to_dtype[0]
            ctype = target.dtype_to_typename(dtype)
            yield (f"08_c_{self.name_in_target}", f"""
            inline static int {self.name_in_target}({ctype} x) {{
              return 0;
            }}""")


class GNULibcCallable(ScalarCallable):
    def with_types(self, arg_id_to_dtype, callables_table):
        name = self.name

        if name in ["bessel_jn", "bessel_yn"]:
            # bessel functions
            # https://www.gnu.org/software/libc/manual/html_node/Special-Functions.html
            for id in arg_id_to_dtype:
                if not -1 <= id <= 1:
                    raise LoopyError(f"'{name}' can take exactly 2 arguments.")

            if (not arg_id_to_dtype.get(0)) or (not arg_id_to_dtype.get(1)):
                # the types provided aren't mature enough to specialize the
                # callable
                return (
                        self.copy(arg_id_to_dtype=arg_id_to_dtype),
                        callables_table)

            if not arg_id_to_dtype[0].is_integral():
                raise LoopyTypeError(f"'{name}' needs order to be an int-type.")

            if arg_id_to_dtype[1].numpy_dtype == np.float32:
                # See e.g.
                # https://opensource.apple.com/source/Libm/Libm-2026/Source/Intel/math.h.auto.html
                # and
                # https://github.com/flang-compiler/flang/blob/81bebebb38177586f3c004f3c698a00a12bf094b/runtime/libpgmath/lib/common/mthdecls.h#L346-L402
                # for Bessel function names.
                import os
                if os.uname().sysname == "Linux":
                    name_in_target = name[-2:]+"f"
                else:
                    name_in_target = name[-2:]
            elif arg_id_to_dtype[1].numpy_dtype == np.float64:
                name_in_target = name[-2:]
            else:
                raise LoopyTypeError("argument to bessel function must be f32,"
                                     f"f64, got {arg_id_to_dtype[1].numpy_dtype}.")

            return (
                    self.copy(name_in_target=name_in_target,
                              arg_id_to_dtype={-1: arg_id_to_dtype[1],
                                               0: NumpyType(np.int32),
                                               1: arg_id_to_dtype[1]}),
                    callables_table)
        else:
            raise NotImplementedError(f"with_types for '{name}'")

    def generate_preambles(self, target):
        if self.name in ["bessel_yn", "bessel_jn"]:
            yield ("08_c_math", "#include <math.h>")


def get_c_callables():
    """
    Returns an instance of :class:`InKernelCallable` if the function
    represented by :arg:`identifier` is known in C, otherwise returns *None*.
    """
    cmath_ids = ["abs", "acos", "asin", "atan", "cos", "cosh", "sin",
                 "sinh", "pow", "atan2", "tanh", "exp", "log", "log10",
                 "sqrt", "ceil", "floor", "max", "min", "fmax", "fmin",
                 "fabs", "tan", "erf", "erfc", "isnan", "real", "imag",
                 "conj"]

    return {id_: CMathCallable(id_) for id_ in cmath_ids}


def get_gnu_libc_callables():
    # Support special functions from
    # https://www.gnu.org/software/libc/manual/html_node/Special-Functions.html
    func_ids = ["bessel_jn", "bessel_yn"]
    return {id_: GNULibcCallable(id_) for id_ in func_ids}

# }}}


class CFamilyASTBuilder(ASTBuilderBase[Generable]):

    preamble_function_qualifier = "inline"

    # {{{ library

    def symbol_manglers(self):
        return (
                super().symbol_manglers() + [
                    c_symbol_mangler
                    ])

    def preamble_generators(self):
        return (
                super().preamble_generators() + [
                    lambda preamble_info: _preamble_generator(preamble_info,
                        self.preamble_function_qualifier),
                    ])

    @property
    def known_callables(self):
        callables = super().known_callables
        callables.update(get_c_callables())
        return callables

    # }}}

    # {{{ code generation

    def get_function_definition(
            self, codegen_state: CodeGenerationState,
            codegen_result: CodeGenerationResult,
            schedule_index: int, function_decl: Generable, function_body: Generable
            ) -> Generable:
        kernel = codegen_state.kernel
        assert kernel.linearization is not None

        from cgen import (
                FunctionBody,
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
            if isinstance(kernel.linearization[i], CallKernel):
                is_first_dev_prog = False
                break
        if is_first_dev_prog:
            for tv in sorted(
                    kernel.temporary_variables.values(),
                    key=lambda key_tv: key_tv.name):

                if tv.address_space == AddressSpace.GLOBAL and (
                        tv.initializer is not None):
                    assert tv.read_only

                    decl = self.wrap_global_constant(
                            self.get_temporary_var_declarator(codegen_state, tv))

                    if tv.initializer is not None:
                        decl = Initializer(decl, generate_array_literal(
                            codegen_state, tv, tv.initializer))

                    result.append(decl)

        fbody = FunctionBody(function_decl, function_body)
        if not result:
            return fbody
        else:
            return Collection(result+[Line(), fbody])

    def get_function_declaration(
            self, codegen_state: CodeGenerationState,
            codegen_result: CodeGenerationResult, schedule_index: int
            ) -> Tuple[Sequence[Tuple[str, str]], Generable]:
        kernel = codegen_state.kernel

        assert codegen_state.kernel.linearization is not None
        subkernel_name = cast(
                        CallKernel,
                        codegen_state.kernel.linearization[schedule_index]
                        ).kernel_name

        from cgen import FunctionDeclaration, Value

        name = codegen_result.current_program(codegen_state).name
        if self.target.fortran_abi:
            name += "_"

        if codegen_state.is_entrypoint:
            name = Value("void", name)

            # subkernel launches occur only as part of entrypoint kernels for now
            from loopy.schedule.tools import get_subkernel_arg_info
            skai = get_subkernel_arg_info(kernel, subkernel_name)
            passed_names = skai.passed_names
            written_names = skai.written_names
        else:
            name = Value("static void", name)
            passed_names = [arg.name for arg in kernel.args]
            written_names = kernel.get_written_variables()

        return [], FunctionDeclarationWrapper(
                FunctionDeclaration(
                    name,
                    [self.arg_to_cgen_declarator(
                            kernel, arg_name,
                            is_written=arg_name in written_names)
                        for arg_name in passed_names]))

    def get_kernel_call(self, codegen_state: CodeGenerationState,
            subkernel_name: str,
            gsize: Tuple[ExpressionT, ...],
            lsize: Tuple[ExpressionT, ...]) -> Optional[Generable]:
        return None

    def get_temporary_decls(self, codegen_state, schedule_index):
        from loopy.kernel.data import AddressSpace

        kernel = codegen_state.kernel
        assert kernel.linearization is not None

        temp_decls = []
        temp_decls_using_base_storage = []

        # {{{ declare temporaries

        from cgen import Initializer, Line
        # Getting the temporary variables that are needed for the current
        # sub-kernel.
        from loopy.schedule.tools import (
                temporaries_read_in_subkernel,
                temporaries_written_in_subkernel,
                supporting_temporary_names)
        subkernel_name = kernel.linearization[schedule_index].kernel_name
        sub_knl_temps = (
                temporaries_read_in_subkernel(kernel, subkernel_name)
                | temporaries_written_in_subkernel(kernel, subkernel_name))
        sub_knl_temps = (
                sub_knl_temps
                | supporting_temporary_names(kernel, sub_knl_temps))

        ecm = self.get_expression_to_code_mapper(codegen_state)

        for tv_name in sorted(sub_knl_temps):
            tv = kernel.temporary_variables[tv_name]
            if not tv.base_storage:
                # global temp vars are mapped to arguments or global
                # declarations, no need to declare locally.
                if tv.address_space != AddressSpace.GLOBAL:
                    decl = self.get_temporary_var_declarator(codegen_state, tv)

                    if tv.initializer is not None:
                        assert tv.read_only
                        decl = Initializer(decl, generate_array_literal(
                            codegen_state, tv, tv.initializer))

                    temp_decls.append(decl)

            else:
                assert tv.initializer is None

                cast_decl = POD(self, tv.dtype, "")
                temp_var_decl = POD(self, tv.dtype, tv.name)

                if tv._base_storage_access_may_be_aliasing:
                    ptrtype = _ConstPointer
                else:
                    # The 'restrict' part of this is a complete lie--of course
                    # all these temporaries are aliased. But we're promising to
                    # not use them to shovel data from one representation to the
                    # other. That counts, right?
                    ptrtype = _ConstRestrictPointer

                cast_decl = self.wrap_decl_for_address_space(
                        ptrtype(cast_decl), tv.address_space)
                temp_var_decl = self.wrap_decl_for_address_space(
                        ptrtype(temp_var_decl), tv.address_space)

                cast_tp, cast_d = cast_decl.get_decl_pair()
                temp_var_decl = Initializer(
                        temp_var_decl,
                        "({} {}) ({} + {})".format(
                            " ".join(cast_tp), cast_d,
                            tv.base_storage,
                            ecm(tv.offset)
                            ))

                temp_decls_using_base_storage.append(temp_var_decl)

        # }}}

        result = temp_decls + temp_decls_using_base_storage

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

    @property
    def ast_module(self):
        import cgen
        return cgen

    def get_expression_to_code_mapper(self, codegen_state):
        return self.get_expression_to_c_expression_mapper(codegen_state)

    def get_expression_to_c_expression_mapper(self, codegen_state):
        from loopy.target.c.codegen.expression import ExpressionToCExpressionMapper
        return ExpressionToCExpressionMapper(
                codegen_state, fortran_abi=self.target.fortran_abi)

    def get_c_expression_to_code_mapper(self):
        from loopy.target.c.codegen.expression import CExpressionToCodeMapper
        return CExpressionToCodeMapper()

    # {{{ declarators

    def wrap_decl_for_address_space(
            self, decl: Declarator, address_space: AddressSpace) -> Declarator:
        return decl

    def wrap_global_constant(self, decl: Declarator) -> Declarator:
        from cgen import Static
        return Static(decl)

    def get_value_arg_declaraotor(
            self, name: str, dtype: LoopyType, is_written: bool) -> Declarator:
        result = POD(self, dtype, name)

        if not is_written:
            from cgen import Const
            result = Const(result)

        if self.target.fortran_abi:
            from cgen import Pointer
            result = Pointer(result)

        return result

    def get_array_base_declarator(self, ary: ArrayBase) -> Declarator:
        arg_decl = POD(self, ary.dtype, ary.name)

        if ary.dim_tags:
            for dim_tag in ary.dim_tags:
                if isinstance(dim_tag, FixedStrideArrayDimTag):
                    # we're OK with that
                    pass
                else:
                    raise NotImplementedError(
                        f"{type(self).__name__} does not understand axis tag "
                        f"'{type(dim_tag)}.")

        return arg_decl

    def get_array_arg_declarator(
            self, arg: ArrayArg, is_written: bool) -> Declarator:
        from cgen import RestrictPointer
        arg_decl = RestrictPointer(
                self.wrap_decl_for_address_space(
                    self.get_array_base_declarator(arg), arg.address_space))

        if not is_written:
            arg_decl = Const(arg_decl)

        return arg_decl

    def get_constant_arg_declarator(
            self, arg: ConstantArg) -> Declarator:
        from cgen import RestrictPointer
        return Const(self.wrap_decl_for_address_space(
            RestrictPointer(
                self.get_array_base_declarator(arg)), arg.address_space))

    def get_temporary_arg_decl(
            self, temp_var: TemporaryVariable, is_written: bool) -> Declarator:
        if temp_var.address_space == AddressSpace.GLOBAL:
            from cgen import RestrictPointer
            arg_decl = RestrictPointer(
                    self.wrap_decl_for_address_space(
                        self.get_array_base_declarator(temp_var),
                        temp_var.address_space))
            if not is_written:
                arg_decl = Const(arg_decl)

            return arg_decl
        else:
            raise LoopyError("unexpected request for argument declaration of "
                    "non-global temporary")

    def get_image_arg_declarator(
            self, arg: ImageArg, is_written: bool) -> Declarator:
        raise NotImplementedError()

    def arg_to_cgen_declarator(
            self, kernel: LoopKernel, passed_name: str, is_written: bool
            ) -> Declarator:
        if passed_name in kernel.all_inames():
            assert not is_written
            return self.get_value_arg_declaraotor(
                    passed_name, kernel.index_dtype, is_written)
        var_descr = kernel.get_var_descriptor(passed_name)
        if isinstance(var_descr, ValueArg):
            assert var_descr.dtype is not None
            return self.get_value_arg_declaraotor(
                    var_descr.name, var_descr.dtype, is_written)
        elif isinstance(var_descr, ArrayArg):
            return self.get_array_arg_declarator(var_descr, is_written)
        elif isinstance(var_descr, TemporaryVariable):
            return self.get_temporary_arg_decl(var_descr, is_written)
        elif isinstance(var_descr, ConstantArg):
            return self.get_constant_arg_declarator(var_descr)
        elif isinstance(var_descr, ImageArg):
            return self.get_image_arg_declarator(var_descr, is_written)
        else:
            raise ValueError(f"unexpected type of argument '{passed_name}': "
                    f"'{type(var_descr)}'")

    def get_temporary_var_declarator(self,
            codegen_state: CodeGenerationState,
            temp_var: TemporaryVariable) -> Declarator:
        temp_var_decl = self.get_array_base_declarator(temp_var)

        if temp_var.storage_shape:
            shape = temp_var.storage_shape
        else:
            assert isinstance(temp_var.shape, tuple)
            shape = temp_var.shape

        assert isinstance(shape, tuple)
        assert isinstance(temp_var.dim_tags, tuple)

        from loopy.kernel.array import drop_vec_dims
        unvec_shape = drop_vec_dims(temp_var.dim_tags, shape)

        if unvec_shape:
            from cgen import ArrayOf
            ecm = self.get_expression_to_code_mapper(codegen_state)
            temp_var_decl = ArrayOf(temp_var_decl,
                    ecm(p.flattened_product(unvec_shape),
                        prec=PREC_NONE, type_context="i"))

        if temp_var.alignment:
            from cgen import AlignedAttribute
            temp_var_decl = AlignedAttribute(temp_var.alignment, temp_var_decl)

        assert isinstance(temp_var.address_space, AddressSpace)
        return self.wrap_decl_for_address_space(temp_var_decl,
                temp_var.address_space)

    # }}}

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
        func_id = insn.expression.function.name
        in_knl_callable = codegen_state.callables_table[func_id]

        if isinstance(in_knl_callable, ScalarCallable) and (
                in_knl_callable.name_in_target == "loopy_make_tuple"):
            return self.emit_tuple_assignment(codegen_state, insn)

        # takes "is_returned" to infer whether insn.assignees[0] is a part of
        # LHS.
        in_knl_callable_as_call, is_returned = in_knl_callable.emit_call_insn(
                insn=insn,
                target=self.target,
                expression_to_code_mapper=ecm)

        if is_returned:
            from cgen import Assign
            lhs_code = ecm(insn.assignees[0], prec=PREC_NONE, type_context=None)
            return Assign(lhs_code,
                    CExpression(self.get_c_expression_to_code_mapper(),
                    in_knl_callable_as_call))
        else:
            from cgen import ExpressionStatement
            return ExpressionStatement(
                    CExpression(self.get_c_expression_to_code_mapper(),
                                in_knl_callable_as_call))

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            lbound, ubound, inner, hints):
        ecm = codegen_state.expression_to_code_mapper

        from pymbolic import var
        from pymbolic.primitives import Comparison
        from pymbolic.mapper.stringifier import PREC_NONE
        from cgen import For, InlineInitializer

        loop = For(
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

        if hints:
            return Collection(list(hints) + [loop])
        else:
            return loop

    def emit_unroll_hint(self, value):
        from cgen import Pragma
        if value:
            return Pragma(f"unroll {value}")
        else:
            return Pragma("unroll")

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

    def emit_noop_with_comment(self, s):
        from cgen import Line
        return Line(f"; /*{s}*/")

    @property
    def can_implement_conditionals(self):
        return True

    def emit_if(self, condition_str, ast):
        from cgen import If
        return If(condition_str, ast)

    # }}}

    def process_ast(self, node):
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

class _CExecutorCacheKey:
    pass


class ExecutableCTarget(CTarget):
    """
    An executable CFamilyTarget that uses (by default) JIT compilation of C-code
    """
    def __init__(self, compiler=None, fortran_abi=False):
        super().__init__(fortran_abi=fortran_abi)
        from loopy.target.c.c_execution import CCompiler
        self.compiler = compiler or CCompiler()

    def get_kernel_executor_cache_key(self, *args, **kwargs):
        # This is for things like the context in OpenCL. There is no such
        # thing that CPU JIT is specific to.

        # We can't use None here, because this will be a key in a WeakKeyDict,
        # and None isn't allowed in that setting.
        return _CExecutorCacheKey

    def get_kernel_executor(
            self, t_unit: TranslationUnit,
            *args: Any, entrypoint: FunctionIdT, **kwargs: Any) -> ExecutorBase:
        from loopy.target.c.c_execution import CExecutor
        return CExecutor(t_unit, entrypoint=entrypoint, compiler=self.compiler)

    def get_host_ast_builder(self):
        # enable host code generation
        return CFamilyASTBuilder(self)

# }}}


# {{{ C99 (with GNULibc) callable target

class CWithGNULibcTarget(CTarget):
    def get_device_ast_builder(self):
        return CWithGNULibcASTBuilder(self)


class CWithGNULibcASTBuilder(CASTBuilder):
    @property
    def known_callables(self):
        callables = super().known_callables
        callables.update(get_gnu_libc_callables())
        return callables


class ExecutableCWithGNULibcTarget(ExecutableCTarget):
    def get_device_ast_builder(self):
        return CWithGNULibcASTBuilder(self)

# }}}

# vim: foldmethod=marker

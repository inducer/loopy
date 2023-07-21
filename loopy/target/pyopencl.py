from __future__ import annotations
"""OpenCL target integrated with PyOpenCL."""

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

from warnings import warn
from typing import Sequence, Tuple, List, Union, Optional, cast, Any, TYPE_CHECKING

import numpy as np
import pymbolic.primitives as p
import genpy
from cgen import (Generable, Pointer, Const, FunctionBody, Collection, Initializer,
                Line, Block)
from cgen.opencl import CLGlobal

from loopy.target.opencl import (OpenCLTarget, OpenCLCASTBuilder,
        ExpressionToOpenCLCExpressionMapper)
from loopy.target.pyopencl_execution import PyOpenCLExecutor
from loopy.target.python import PythonASTBuilderBase
from loopy.kernel import LoopKernel
from loopy.translation_unit import FunctionIdT, TranslationUnit
from loopy.types import NumpyType
from loopy.typing import ExpressionT
from loopy.diagnostic import LoopyError, LoopyTypeError
from loopy.kernel.function_interface import ScalarCallable
from loopy.kernel.data import (
        TemporaryVariable, ValueArg, ArrayArg, ImageArg, ConstantArg)
from loopy.schedule import CallKernel
from loopy.codegen import CodeGenerationState
from loopy.codegen.result import CodeGenerationResult

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pyopencl as cl


# {{{ pyopencl function scopers

class PyOpenCLCallable(ScalarCallable):
    """
    Records information about the callables which are not covered by
    :class:`loopy.target.opencl.OpenCLCallable`
    """
    def with_types(self, arg_id_to_dtype, callables_table):

        name = self.name

        for id in arg_id_to_dtype:
            # since all the below functions are single arg.
            if not -1 <= id <= 0:
                raise LoopyError(f"{name} can only take one argument")

        if 0 not in arg_id_to_dtype or arg_id_to_dtype[0] is None:
            # the types provided aren't mature enough to specialize the
            # callable
            return (
                    self.copy(arg_id_to_dtype=arg_id_to_dtype),
                    callables_table)

        dtype = arg_id_to_dtype[0]

        if name in ["real", "imag", "abs"]:
            if dtype.is_complex():
                if dtype.numpy_dtype == np.complex64:
                    tpname = "cfloat"
                elif dtype.numpy_dtype == np.complex128:
                    tpname = "cdouble"
                else:
                    raise LoopyTypeError(f"unexpected complex type '{dtype}'")

                return (
                        self.copy(name_in_target=f"{tpname}_{name}",
                            arg_id_to_dtype={0: dtype, -1: NumpyType(
                                np.dtype(dtype.numpy_dtype.type(0).real))}),
                        callables_table)

        if name in ["real", "imag"]:
            if not dtype.is_complex():
                tpname = dtype.numpy_dtype.type.__name__
                return (
                        self.copy(
                            name_in_target=f"lpy_{name}_{tpname}",
                            arg_id_to_dtype={0: dtype, -1: dtype}),
                        callables_table)

        if name in ["sqrt", "exp", "log",
                "sin", "cos", "tan",
                "sinh", "cosh", "tanh",
                "conj"]:
            if dtype.is_complex():
                # function parameters are complex.
                if dtype.numpy_dtype == np.complex64:
                    tpname = "cfloat"
                elif dtype.numpy_dtype == np.complex128:
                    tpname = "cdouble"
                else:
                    raise LoopyTypeError("unexpected complex type '%s'" % dtype)

                return (
                        self.copy(name_in_target=f"{tpname}_{name}",
                            arg_id_to_dtype={0: dtype, -1: dtype}),
                        callables_table)

            # fall back to pure OpenCL for real-valued arguments

        from loopy.target.opencl import OpenCLCallable
        return OpenCLCallable(name,
                arg_id_to_dtype=self.arg_id_to_dtype,
                arg_id_to_descr=self.arg_id_to_descr,
                name_in_target=self.name_in_target).with_types(
                        arg_id_to_dtype, callables_table)

    def generate_preambles(self, target):
        name = self.name_in_target
        if name.startswith("lpy_real") or name.startswith("lpy_imag"):
            if name.startswith("lpy_real"):
                ret = "x"
            else:
                ret = "0"

            dtype = self.arg_id_to_dtype[-1]
            ctype = target.dtype_to_typename(dtype)

            yield (f"40_{name}", f"""
                static inline {ctype} {name}({ctype} x) {{
                    return {ret};
                }}
                """)


def get_pyopencl_callables():
    pyopencl_ids = ["sqrt", "exp", "log", "sin", "cos", "tan", "sinh", "cosh",
            "tanh", "conj", "real", "imag", "abs"]
    return {id_: PyOpenCLCallable(name=id_) for id_ in pyopencl_ids}

# }}}


# {{{ preamble generator

def pyopencl_preamble_generator(preamble_info):
    has_double = False
    has_complex = False

    from loopy.types import NumpyType
    for dtype in preamble_info.seen_dtypes:
        if (isinstance(dtype, NumpyType)
                and dtype.dtype in [np.float64, np.complex128]):
            has_double = True
        if dtype.involves_complex():
            has_complex = True

    if has_complex:
        if has_double:
            yield ("10_include_complex_header", """
                #define PYOPENCL_DEFINE_CDOUBLE
                #ifndef PYOPENCL_COMPLEX_ENABLE_EXTENDED_ALIGNMENT
                #define PYOPENCL_COMPLEX_ENABLE_EXTENDED_ALIGNMENT 1
                #endif

                #include <pyopencl-complex.h>
                """)
        else:
            yield ("10_include_complex_header", """
                #ifndef PYOPENCL_COMPLEX_ENABLE_EXTENDED_ALIGNMENT
                #define PYOPENCL_COMPLEX_ENABLE_EXTENDED_ALIGNMENT 1
                #endif

                #include <pyopencl-complex.h>
                """)

# }}}


# {{{ expression mapper

class ExpressionToPyOpenCLCExpressionMapper(ExpressionToOpenCLCExpressionMapper):
    def complex_type_name(self, dtype):
        from loopy.types import NumpyType
        if not isinstance(dtype, NumpyType):
            raise LoopyError("'%s' is not a complex type" % dtype)

        if dtype.dtype == np.complex64:
            return "cfloat"
        if dtype.dtype == np.complex128:
            return "cdouble"
        else:
            raise RuntimeError

    def wrap_in_typecast_lazy(self, actual_type_func, needed_dtype, s):
        if needed_dtype.is_complex():
            return self.wrap_in_typecast(actual_type_func(), needed_dtype, s)
        else:
            return super().wrap_in_typecast_lazy(actual_type_func,
                                                 needed_dtype, s)

    def wrap_in_typecast(self, actual_type, needed_dtype, s):
        if (actual_type.is_complex() and needed_dtype.is_complex()
                and actual_type != needed_dtype):
            return p.Variable("%s_cast" % self.complex_type_name(needed_dtype))(s)
        elif not actual_type.is_complex() and needed_dtype.is_complex():
            return p.Variable("%s_fromreal" % self.complex_type_name(needed_dtype))(
                    s)
        else:
            return super().wrap_in_typecast_lazy(actual_type,
                                                 needed_dtype, s)

    def map_sum(self, expr, type_context):
        # I've added 'type_context == "i"' because of the following
        # idiotic corner case: Code generation for subscripts comes
        # through here, and it may involve variables that we know
        # nothing about (offsets and such). If we fall into the allow_complex
        # branch, we'll try to do type inference on these variables,
        # and stuff breaks. This band-aid works around that. -AK
        if not self.allow_complex or type_context == "i":
            return super().map_sum(expr, type_context)

        tgt_dtype = self.infer_type(expr)
        is_complex = tgt_dtype.is_complex()

        if not is_complex:
            return super().map_sum(expr, type_context)
        elif not self.kernel.options.allow_fp_reordering:
            if len(expr.children) == 0:
                return tgt_dtype(0)

            tgt_name = self.complex_type_name(tgt_dtype)
            result = None
            lhs_is_complex = False

            for child in expr.children:
                rhs_is_complex = self.infer_type(child).is_complex()
                if rhs_is_complex:
                    child_val = self.rec(child, type_context, tgt_dtype)
                else:
                    child_val = self.rec(child, type_context)

                if result is None:
                    result = child_val
                elif lhs_is_complex and rhs_is_complex:
                    result = p.Variable(f"{tgt_name}_add")(result, child_val)
                elif lhs_is_complex and not rhs_is_complex:
                    result = p.Variable(f"{tgt_name}_addr")(result, child_val)
                elif not lhs_is_complex and rhs_is_complex:
                    result = p.Variable(f"{tgt_name}_radd")(result, child_val)
                else:
                    result = p.Sum((result, child_val))
                lhs_is_complex = lhs_is_complex or rhs_is_complex
            return result
        else:
            tgt_name = self.complex_type_name(tgt_dtype)

            reals = []
            complexes = []
            for child in expr.children:
                if self.infer_type(child).is_complex():
                    complexes.append(child)
                else:
                    reals.append(child)

            real_sum = p.flattened_sum([self.rec(r, type_context) for r in reals])

            c_applied = [self.rec(c, type_context, tgt_dtype) for c in complexes]

            mul_name = f"{tgt_name}_mul"

            def binary_tree_add(start, end):
                if start + 1 == end:
                    return c_applied[start]
                mid = (start + end)//2
                lsum = binary_tree_add(start, mid)
                rsum = binary_tree_add(mid, end)

                # FMAs should ideally be recognized by the compiler, but some
                # compilers fail to do so. For eg:
                #
                #    res = complex_add(c, complex_mul(a, b))
                #
                # leads to code that looks like below because of the temporary
                # given by ``complex_mul(a, b)``.
                #
                #    tmp.real = a.real * b.real - a.imag * b.imag
                #    tmp.imag = a.real * b.imag + a.imag * b.real
                #    res.real = c.real + tmp.real
                #    res.imag = c.imag + tmp.imag
                #
                # clang can fuse across multiple statements like this with
                # -ffp-contract=fast which is the default for PTX codegen, but
                # for some unknown reason, clang fails to see the FMAs.
                #
                # We need to do this only for complex as we haev temporaries
                # only in complex. For reals, the code generated looks like
                #
                #    res = c + a * b
                #
                # and clang is able to generate an FMA for this code.

                if isinstance(lsum, p.Call) and isinstance(lsum.function,
                        p.Variable) and lsum.function.name == mul_name:
                    return p.Variable(f"{tgt_name}_fma")(*lsum.parameters, rsum)

                elif isinstance(rsum, p.Call) and isinstance(rsum.function,
                        p.Variable) and rsum.function.name == mul_name:
                    return p.Variable(f"{tgt_name}_fma")(*rsum.parameters, lsum)

                else:
                    return p.Variable(f"{tgt_name}_add")(lsum, rsum)

            complex_sum = binary_tree_add(0, len(c_applied))

            if reals:
                return p.Variable(f"{tgt_name}_radd")(real_sum, complex_sum)
            else:
                return complex_sum

    def map_product(self, expr, type_context):
        # I've added 'type_context == "i"' because of the following
        # idiotic corner case: Code generation for subscripts comes
        # through here, and it may involve variables that we know
        # nothing about (offsets and such). If we fall into the allow_complex
        # branch, we'll try to do type inference on these variables,
        # and stuff breaks. This band-aid works around that. -AK
        if not self.allow_complex or type_context == "i":
            return super().map_product(expr, type_context)

        tgt_dtype = self.infer_type(expr)
        is_complex = tgt_dtype.is_complex()

        if not is_complex:
            return super().map_product(expr, type_context)
        elif not self.kernel.options.allow_fp_reordering:
            tgt_name = self.complex_type_name(tgt_dtype)

            result = None
            lhs_is_complex = False
            for child in expr.children:
                rhs_is_complex = self.infer_type(child).is_complex()
                if rhs_is_complex:
                    child_val = self.rec(child, type_context, tgt_dtype)
                else:
                    child_val = self.rec(child, type_context)

                if result is None:
                    result = child_val
                elif lhs_is_complex and rhs_is_complex:
                    result = p.Variable(f"{tgt_name}_mul")(result, child_val)
                elif lhs_is_complex and not rhs_is_complex:
                    result = p.Variable(f"{tgt_name}_mulr")(result, child_val)
                elif not lhs_is_complex and rhs_is_complex:
                    result = p.Variable(f"{tgt_name}_rmul")(result, child_val)
                else:
                    result = p.Product((result, child_val))
                lhs_is_complex = lhs_is_complex or rhs_is_complex
            return result
        else:
            tgt_name = self.complex_type_name(tgt_dtype)

            reals = []
            complexes = []
            for child in expr.children:
                if self.infer_type(child).is_complex():
                    complexes.append(child)
                else:
                    reals.append(child)

            real_prd = p.flattened_product(
                    [self.rec(r, type_context) for r in reals])

            c_applied = [self.rec(c, type_context, tgt_dtype) for c in complexes]

            def binary_tree_mul(start, end):
                if start + 1 == end:
                    return c_applied[start]
                mid = (start + end)//2
                lsum = binary_tree_mul(start, mid)
                rsum = binary_tree_mul(mid, end)
                return p.Variable("%s_mul" % tgt_name)(lsum, rsum)

            complex_prd = binary_tree_mul(0, len(complexes))

            if reals:
                return p.Variable("%s_rmul" % tgt_name)(real_prd, complex_prd)
            else:
                return complex_prd

    def map_quotient(self, expr, type_context):
        n_dtype = self.infer_type(expr.numerator).numpy_dtype
        d_dtype = self.infer_type(expr.denominator).numpy_dtype
        tgt_dtype = self.infer_type(expr)
        n_complex = "c" == n_dtype.kind
        d_complex = "c" == d_dtype.kind

        if not self.allow_complex or (not (n_complex or d_complex)):
            return super().map_quotient(expr, type_context)

        if n_complex and not d_complex:
            return p.Variable("%s_divider" % self.complex_type_name(tgt_dtype))(
                    self.rec(expr.numerator, type_context, tgt_dtype),
                    self.rec(expr.denominator, type_context))
        elif not n_complex and d_complex:
            return p.Variable("%s_rdivide" % self.complex_type_name(tgt_dtype))(
                    self.rec(expr.numerator, type_context),
                    self.rec(expr.denominator, type_context, tgt_dtype))
        else:
            return p.Variable("%s_divide" % self.complex_type_name(tgt_dtype))(
                    self.rec(expr.numerator, type_context, tgt_dtype),
                    self.rec(expr.denominator, type_context, tgt_dtype))

    def map_constant(self, expr, type_context):
        if isinstance(expr, (complex, np.complexfloating)):
            try:
                dtype = expr.dtype
            except AttributeError:
                # (COMPLEX_GUESS_LOGIC) This made it through type 'guessing' in
                # type inference, and it was concluded there (search for
                # COMPLEX_GUESS_LOGIC in loopy.type_inference), that no
                # accuracy was lost by using single precision.
                cast_type = "cfloat"
            else:
                if dtype == np.complex128:
                    cast_type = "cdouble"
                elif dtype == np.complex64:
                    cast_type = "cfloat"
                else:
                    raise RuntimeError("unsupported complex type in expression "
                            "generation: %s" % type(expr))

            return p.Variable("%s_new" % cast_type)(self.rec(expr.real,
                                                             type_context),
                                                    self.rec(expr.imag,
                                                             type_context))

        return super().map_constant(expr, type_context)

    def map_power(self, expr, type_context):
        tgt_dtype = self.infer_type(expr)
        base_dtype = self.infer_type(expr.base)
        exponent_dtype = self.infer_type(expr.exponent)

        if not self.allow_complex or (not tgt_dtype.is_complex()):
            return super().map_power(expr, type_context)

        if expr.exponent in [2, 3, 4]:
            value = expr.base
            for _i in range(expr.exponent-1):
                value = value * expr.base
            return self.rec(value, type_context)
        else:
            b_complex = base_dtype.is_complex()
            e_complex = exponent_dtype.is_complex()

            if b_complex and not e_complex:
                return p.Variable("%s_powr" % self.complex_type_name(tgt_dtype))(
                        self.rec(expr.base, type_context, tgt_dtype),
                        self.rec(expr.exponent, type_context))
            else:
                return p.Variable("%s_pow" % self.complex_type_name(tgt_dtype))(
                        self.rec(expr.base, type_context, tgt_dtype),
                        self.rec(expr.exponent, type_context, tgt_dtype))

# }}}


# {{{ target

class PyOpenCLTarget(OpenCLTarget):
    """A code generation target that takes special advantage of :mod:`pyopencl`
    features such as run-time knowledge of the target device (to generate
    warnings) and support for complex numbers.
    """

    # FIXME make prefixes conform to naming rules
    # (see Reference: Loopy’s Model of a Kernel)

    host_program_name_prefix = "_lpy_host_"
    host_program_name_suffix = ""

    # FIXME Not yet complete
    limit_arg_size_nbytes: Optional[int]
    pointer_size_nbytes: int

    def __init__(
            self, device=None, *, pyopencl_module_name: str = "_lpy_cl",
            atomics_flavor=None, use_int8_for_bool: bool = True,
            limit_arg_size_nbytes: Optional[int] = None,
            pointer_size_nbytes: Optional[int] = None
            ) -> None:
        # This ensures the dtype registry is populated.
        import pyopencl.tools  # noqa

        super().__init__(
            atomics_flavor=atomics_flavor,
            use_int8_for_bool=use_int8_for_bool)

        import pyopencl.version
        if pyopencl.version.VERSION < (2021, 2):
            raise RuntimeError("The version of loopy you have installed "
                    "generates invoker code that requires PyOpenCL 2021.2 "
                    "or newer.")

        if device is not None:
            warn("Passing device is deprecated, it will stop working in 2022.",
                    DeprecationWarning, stacklevel=2)

        self.pyopencl_module_name = pyopencl_module_name

        if pointer_size_nbytes is None:
            pointer_size_nbytes = tuple.__itemsize__

        self.limit_arg_size_nbytes = limit_arg_size_nbytes
        self.pointer_size_nbytes = pointer_size_nbytes

    @property
    def device(self):
        warn("PyOpenCLTarget.device is deprecated, it will stop working in 2022.",
                DeprecationWarning, stacklevel=2)
        return None

    # NB: Not including 'device', as that is handled specially here.
    hash_fields = OpenCLTarget.hash_fields + (
            "pyopencl_module_name",)
    comparison_fields = OpenCLTarget.comparison_fields + (
            "pyopencl_module_name",)

    def get_host_ast_builder(self):
        return PyOpenCLPythonASTBuilder(self)

    def get_device_ast_builder(self):
        return PyOpenCLCASTBuilder(self)

    # {{{ types

    def get_dtype_registry(self):
        from pyopencl.compyte.dtypes import TYPE_REGISTRY
        result = TYPE_REGISTRY

        from loopy.target.opencl import (
                DTypeRegistryWrapperWithCL1Atomics,
                DTypeRegistryWrapperWithInt8ForBool)

        result = DTypeRegistryWrapperWithInt8ForBool(result)
        if self.atomics_flavor == "cl1":
            result = DTypeRegistryWrapperWithCL1Atomics(result)
        else:
            raise NotImplementedError("atomics flavor: %s" % self.atomics_flavor)

        return result

    def is_vector_dtype(self, dtype):
        try:
            import pyopencl.cltypes as cltypes
            vec_types = cltypes.vec_types
        except ImportError:
            from pyopencl.array import vec
            vec_types = vec.types

        return (isinstance(dtype, NumpyType)
                and dtype.numpy_dtype in list(vec_types.values()))

    def vector_dtype(self, base, count):
        try:
            import pyopencl.cltypes as cltypes
            vec_types = cltypes.vec_types
        except ImportError:
            from pyopencl.array import vec
            vec_types = vec.types

        return NumpyType(vec_types[base.numpy_dtype, count])

    def alignment_requirement(self, type_decl):
        import struct

        fmt = (type_decl.struct_format()
                .replace("F", "ff")
                .replace("D", "dd"))

        return struct.calcsize(fmt)

    # }}}

    def get_kernel_executor_cache_key(self, queue, **kwargs):
        return (queue.context, kwargs["entrypoint"])

    # type-ignore because we're making things from *args: Any more concrete,
    # and mypy doesn't like it.
    def get_kernel_executor(self, t_unit: TranslationUnit,  # type: ignore[override]
                            queue_or_context: Union[cl.CommandQueue, cl.Context],
                            *args: Any, entrypoint: FunctionIdT, **kwargs: Any
                            ) -> PyOpenCLExecutor:
        from pyopencl import CommandQueue
        if isinstance(queue_or_context, CommandQueue):
            context = queue_or_context.context
        else:
            context = queue_or_context

        from loopy.target.pyopencl_execution import PyOpenCLExecutor
        return PyOpenCLExecutor(context, t_unit, entrypoint=entrypoint)

# }}}


# {{{ host code: value arg setup

def generate_value_arg_setup(
        kernel: LoopKernel, passed_names: Sequence[str]
        ) -> genpy.Suite:
    options = kernel.options

    import loopy as lp
    from loopy.kernel.array import ArrayBase

    from genpy import If, Raise, Statement as S, Suite

    result: List[str] = []
    gen = result.append

    buf_indices_and_args = []
    buf_pack_indices_and_args = []

    from pyopencl.invoker import BUF_PACK_TYPECHARS

    def add_buf_arg(arg_idx, typechar, expr_str):
        if typechar in BUF_PACK_TYPECHARS:
            buf_pack_indices_and_args.append(arg_idx)
            buf_pack_indices_and_args.append(repr(typechar.encode()))
            buf_pack_indices_and_args.append(expr_str)
        else:
            buf_indices_and_args.append(arg_idx)
            buf_indices_and_args.append(f"pack('{typechar}', {expr_str})")

    for arg_idx, passed_name in enumerate(passed_names):
        if passed_name in kernel.all_inames():
            add_buf_arg(arg_idx, kernel.index_dtype.numpy_dtype.char, passed_name)
            continue

        var_descr = kernel.get_var_descriptor(passed_name)
        assert var_descr.dtype is not None

        if not isinstance(var_descr, lp.ValueArg):
            assert isinstance(var_descr, ArrayBase)

            continue

        if not options.skip_arg_checks:
            gen(If(f"{passed_name} is None",
                Raise('RuntimeError("input argument \'{var_descr.name}\' '
                        'must be supplied")')))

        if var_descr.dtype.is_composite():
            buf_indices_and_args.append(arg_idx)
            buf_indices_and_args.append(f"{passed_name}")

        elif var_descr.dtype.is_complex():
            assert isinstance(var_descr.dtype, NumpyType)

            dtype = var_descr.dtype

            if dtype.numpy_dtype == np.complex64:
                arg_char = "f"
            elif dtype.numpy_dtype == np.complex128:
                arg_char = "d"
            else:
                raise TypeError("unexpected complex type: %s" % dtype)

            buf_indices_and_args.append(arg_idx)
            buf_indices_and_args.append(
                f"_lpy_pack('{arg_char}{arg_char}', "
                f"{passed_name}.real, {passed_name}.imag)")

        elif isinstance(var_descr.dtype, NumpyType):
            add_buf_arg(arg_idx, var_descr.dtype.dtype.char, passed_name)

        else:
            raise LoopyError("do not know how to pass argument of type '%s'"
                    % var_descr.dtype)

    for arg_kind, args_and_indices, entry_length in [
            ("_buf", buf_indices_and_args, 2),
            ("_buf_pack", buf_pack_indices_and_args, 3),
            ]:
        assert len(args_and_indices) % entry_length == 0
        if args_and_indices:
            gen(S(f"_lpy_knl._set_arg{arg_kind}_multi("
                    f"({', '.join(str(i) for i in args_and_indices)},), "
                    ")"))

    return Suite(result)

# }}}


def generate_array_arg_setup(
        kernel: LoopKernel, passed_names: Sequence[str],
        ) -> genpy.Generable:
    from loopy.kernel.array import ArrayBase
    from genpy import Statement as S, Suite

    result: List[str] = []
    gen = result.append

    cl_indices_and_args: List[Union[int, str]] = []
    for arg_idx, passed_name in enumerate(passed_names):
        if passed_name in kernel.all_inames():
            continue

        var_descr = kernel.get_var_descriptor(passed_name)
        if isinstance(var_descr, ArrayBase):
            cl_indices_and_args.append(arg_idx)
            cl_indices_and_args.append(passed_name)

    if cl_indices_and_args:
        assert len(cl_indices_and_args) % 2 == 0

        gen(S(f"_lpy_knl._set_arg_multi("
            f"({', '.join(str(i) for i in cl_indices_and_args)},)"
            ")"))

    return Suite(result)


# {{{ host ast builder

class PyOpenCLPythonASTBuilder(PythonASTBuilderBase):
    """A Python host AST builder for integration with PyOpenCL.
    """

    # {{{ code generation guts

    def get_function_definition(
            self, codegen_state, codegen_result,
            schedule_index: int, function_decl, function_body: genpy.Generable
            ) -> genpy.Function:
        assert schedule_index == 0

        from loopy.schedule.tools import get_kernel_arg_info
        kai = get_kernel_arg_info(codegen_state.kernel)

        args = (
                ["_lpy_cl_kernels", "queue"]
                + list(kai.passed_arg_names)
                + ["wait_for=None", "allocator=None"])

        from genpy import (For, Function, Suite, Return, Line, Statement as S)
        return Function(
                codegen_result.current_program(codegen_state).name,
                args,
                Suite([
                    Line(),
                    ] + [
                    Line(),
                    function_body,
                    Line(),
                    ] + ([
                        For("_tv", "_global_temporaries",
                            # Free global temporaries.
                            # Zero-size temporaries allocate as None, tolerate that.
                            # https://documen.tician.de/pyopencl/tools.html#pyopencl.tools.ImmediateAllocator
                            S("if _tv is not None: _tv.release()"))
                        ] if self._get_global_temporaries(codegen_state) else []
                    ) + [
                    Line(),
                    Return("_lpy_evt"),
                    ]))

    def get_function_declaration(
            self, codegen_state: CodeGenerationState,
            codegen_result: CodeGenerationResult, schedule_index: int
            ) -> Tuple[Sequence[Tuple[str, str]], genpy.Generable]:
        # no such thing in Python
        return [], None

    def _get_global_temporaries(self, codegen_state):
        from loopy.kernel.data import AddressSpace

        return sorted(
            (tv for tv in codegen_state.kernel.temporary_variables.values()
            if tv.address_space == AddressSpace.GLOBAL),
            key=lambda tv: tv.name)

    def get_temporary_decls(self, codegen_state, schedule_index):
        from genpy import Assign, Comment, Line

        from pymbolic.mapper.stringifier import PREC_NONE
        ecm = self.get_expression_to_code_mapper(codegen_state)

        global_temporaries = self._get_global_temporaries(codegen_state)
        if not global_temporaries:
            return []

        allocated_var_names = []
        code_lines = []
        code_lines.append(Line())
        code_lines.append(Comment("{{{ allocate global temporaries"))
        code_lines.append(Line())

        for tv in global_temporaries:
            if not tv.base_storage:
                nbytes_str = ecm(tv.nbytes, PREC_NONE, "i")
                allocated_var_names.append(tv.name)
                code_lines.append(Assign(tv.name,
                                         f"allocator({nbytes_str})"))

        code_lines.append(Assign("_global_temporaries", "[{tvs}]".format(
            tvs=", ".join(tv for tv in allocated_var_names))))

        code_lines.append(Line())
        code_lines.append(Comment("}}}"))
        code_lines.append(Line())

        return code_lines

    def get_kernel_call(
            self, codegen_state: CodeGenerationState,
            subkernel_name: str,
            gsize: Tuple[ExpressionT, ...], lsize: Tuple[ExpressionT, ...]
            ) -> genpy.Suite:
        from genpy import Suite, Assign, Assert, Line, Comment

        kernel = codegen_state.kernel

        from loopy.schedule.tools import get_subkernel_arg_info
        skai = get_subkernel_arg_info(kernel, subkernel_name)

        ecm = self.get_expression_to_code_mapper(codegen_state)

        if not gsize:
            gsize = (1,)
        if not lsize:
            lsize = (1,)

        assert isinstance(kernel.target, PyOpenCLTarget)
        regular_arg_names, struct_overflow_arg_names = split_args_for_overflow(
                kernel, skai.passed_names,
                limit_arg_size_nbytes=kernel.target.limit_arg_size_nbytes,
                pointer_size_nbytes=kernel.target.pointer_size_nbytes)

        value_arg_code = generate_value_arg_setup(
                codegen_state.kernel, regular_arg_names)
        arry_arg_code = generate_array_arg_setup(
                codegen_state.kernel, regular_arg_names)

        if struct_overflow_arg_names:
            regular_arg_names_set = frozenset(regular_arg_names)
            struct_overflow_arg_names_set = frozenset(
                    struct_overflow_arg_names)

            py_passed_args = []
            struct_pack_types = []
            struct_pack_args = []

            for arg_name in skai.passed_names:
                if arg_name in regular_arg_names_set:
                    py_passed_args.append(arg_name)
                else:
                    assert arg_name in struct_overflow_arg_names_set

                    arg = kernel.get_var_descriptor(arg_name)
                    assert arg.dtype is not None
                    if isinstance(arg, ValueArg):
                        struct_pack_types.append(arg.dtype.numpy_dtype.char)
                        struct_pack_args.append(arg_name)
                    elif isinstance(arg, (ArrayArg, ConstantArg, TemporaryVariable)):
                        struct_pack_types.append("P")
                        struct_pack_args.append(f"{arg_name}.svm_ptr")
                    elif isinstance(arg, ImageArg):
                        raise AssertionError()
                    else:
                        raise ValueError(f"unrecognized arg type: '{type(arg)}'")

            cl_arg_count = len(regular_arg_names)
            overflow_args_code = Suite([
                # It's important for _lpy_overflow_args_buf to be in a variable.
                # Otherwise, no reference to it will survive until the kernel
                # launch and the buffer may be released.
                Assign("_lpy_overflow_args_buf",
                    "_lpy_cl.Buffer(queue.context, "
                    "_lpy_cl.mem_flags.READ_ONLY "
                    "| _lpy_cl.mem_flags.COPY_HOST_PTR, "
                    "hostbuf="
                    f"_lpy_pack({repr(''.join(struct_pack_types))}, "
                    f"{', '.join(struct_pack_args)}))"),
                Line(f"_lpy_knl.set_arg({cl_arg_count}, _lpy_overflow_args_buf)")
                ])

            cl_arg_count += 1

        else:
            cl_arg_count = len(skai.passed_names)
            overflow_args_code = Suite([])

        from pymbolic.mapper.stringifier import PREC_NONE

        import pyopencl.version as cl_ver
        if cl_ver.VERSION < (2020, 2):
            from warnings import warn
            warn("Your kernel invocation will likely fail because your "
                    "version of PyOpenCL does not support allow_empty_ndrange. "
                    "Please upgrade to version 2020.2 or newer.")

        # TODO: Generate finer-grained dependency structure
        return Suite([
            Comment("{{{ enqueue %s" % subkernel_name),
            Line(),
            Assign("_lpy_knl", "_lpy_cl_kernels."+subkernel_name),
            Assert(f"_lpy_knl.num_args == {cl_arg_count}, "
                   f"f'Kernel \"{subkernel_name}\" "
                   f"invoker argument count ({cl_arg_count}) does not match the "
                   # No f"" here since {_lpy_knl.num_args} needs to be evaluated
                   # at runtime, not here.
                   "argument count of the kernel ({_lpy_knl.num_args}).'"),
            Line(),
            value_arg_code,
            arry_arg_code,
            overflow_args_code,
            Assign("_lpy_evt",
                   f"{self.target.pyopencl_module_name}.enqueue_nd_range_kernel("
                   "queue, _lpy_knl, "
                   f"{ecm(gsize, prec=PREC_NONE, type_context='i')}, "
                   f"{ecm(lsize, prec=PREC_NONE, type_context='i')}, "
                   # using positional args because pybind is slow with kwargs
                   "None, "  # offset
                   "wait_for, "
                   "True, "  # g_times_l
                   "True, "  # allow_empty_ndrange
                   ")"),
            Assign("wait_for", "[_lpy_evt]"),
            Line(),
            Comment("}}}"),
            Line(),
            ])

    # }}}

# }}}


# {{{ split_args_for_overflow

def split_args_for_overflow(
        kernel: LoopKernel, passed_names: Sequence[str],
        *, limit_arg_size_nbytes: Optional[int], pointer_size_nbytes: int
        ) -> Tuple[Sequence[str], Sequence[str]]:
    if limit_arg_size_nbytes is None:
        return passed_names, []

    regular_arg_names = []
    overflow_arg_names = []

    # Consider that the pointer to the arg overflow struct also occupies
    # argument space.
    running_arg_size = pointer_size_nbytes

    for arg_name in passed_names:
        arg = kernel.get_var_descriptor(arg_name)
        if isinstance(arg, (ValueArg, ArrayArg, ConstantArg, TemporaryVariable)):
            if isinstance(arg, ValueArg):
                assert arg.dtype is not None
                arg_size = arg.dtype.numpy_dtype.itemsize
            else:
                arg_size = pointer_size_nbytes

            if running_arg_size + arg_size > limit_arg_size_nbytes:
                overflow_arg_names.append(arg_name)
            else:
                regular_arg_names.append(arg_name)

            running_arg_size += arg_size

        elif isinstance(arg, ImageArg):
            regular_arg_names.append(arg_name)
        else:
            raise ValueError(f"unrecognized arg type: '{type(arg)}'")

    return regular_arg_names, overflow_arg_names

# }}}


# {{{ device ast builder

class PyOpenCLCASTBuilder(OpenCLCASTBuilder):
    """A C device AST builder for integration with PyOpenCL.
    """

    # {{{ function decl/def, with arg overflow handling

    def get_function_definition(
            self, codegen_state: CodeGenerationState,
            codegen_result: CodeGenerationResult,
            schedule_index: int, function_decl: Generable, function_body: Generable,
            ) -> Tuple[Sequence[Tuple[str, str]], Generable]:
        assert isinstance(function_body, Block)
        kernel = codegen_state.kernel
        assert kernel.linearization is not None

        subkernel_name = cast(CallKernel,
                kernel.linearization[schedule_index]).kernel_name

        result = []

        from loopy.kernel.data import AddressSpace
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
                        from loopy.target.c import generate_array_literal
                        decl = Initializer(decl, generate_array_literal(
                            codegen_state, tv, tv.initializer))

                    result.append(decl)

        # {{{ unpack overflow args

        if codegen_state.is_entrypoint:
            from loopy.schedule.tools import get_subkernel_arg_info
            skai = get_subkernel_arg_info(kernel, subkernel_name)

            _, struct_overflow_arg_names = split_args_for_overflow(
                    kernel, skai.passed_names,
                    limit_arg_size_nbytes=self.target.limit_arg_size_nbytes,
                    pointer_size_nbytes=self.target.pointer_size_nbytes)

            arg_unpack_code = [
                    Initializer(
                        self.arg_to_cgen_declarator(
                            kernel, arg_name,
                            is_written=arg_name in skai.written_names),
                        f"_lpy_overflow_args->{arg_name}")
                    for arg_name in struct_overflow_arg_names
                    ] + ([Line()] if struct_overflow_arg_names else [])

            function_body = Block(arg_unpack_code + function_body.contents)

        # }}}

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

        from cgen import FunctionDeclaration, Value, Struct

        name = codegen_result.current_program(codegen_state).name
        if self.target.fortran_abi:
            name += "_"

        from loopy.target.c import FunctionDeclarationWrapper

        if codegen_state.is_entrypoint:
            name = Value("void", name)

            # subkernel launches occur only as part of entrypoint kernels for now
            from loopy.schedule.tools import get_subkernel_arg_info
            skai = get_subkernel_arg_info(kernel, subkernel_name)
            passed_names = skai.passed_names
            written_names = skai.written_names

            regular_arg_names, struct_overflow_arg_names = split_args_for_overflow(
                    kernel, passed_names,
                    limit_arg_size_nbytes=self.target.limit_arg_size_nbytes,
                    pointer_size_nbytes=self.target.pointer_size_nbytes)

            arg_overflow_struct_name = f"_lpy_arg_struct_{subkernel_name}"
            arg_overflow_struct = Struct(
                    arg_overflow_struct_name, [
                        self.arg_to_cgen_declarator(
                            kernel, arg_name,
                            is_written=arg_name in written_names)
                        for arg_name in struct_overflow_arg_names])

            if struct_overflow_arg_names:
                logger.info(f"overflowing arguments into SVM buffer: "
                        f"{len(regular_arg_names)} regular/"
                        f"{len(struct_overflow_arg_names)} in buffer "
                        f"for '{subkernel_name}'")
                arg_struct_preambles = [
                        (f"declare-{arg_overflow_struct_name}",
                            str(arg_overflow_struct))
                        ] if struct_overflow_arg_names else []
                arg_struct_args = [CLGlobal(Const(Pointer(Value(
                                f"struct {arg_overflow_struct_name}",
                                "_lpy_overflow_args"))))]
            else:
                arg_struct_preambles = []
                arg_struct_args = []

            return arg_struct_preambles, FunctionDeclarationWrapper(
                    self._wrap_kernel_decl(
                        codegen_state, schedule_index,
                        FunctionDeclaration(
                            name,
                            [self.arg_to_cgen_declarator(
                                kernel, arg_name,
                                is_written=arg_name in written_names)
                                for arg_name in regular_arg_names]
                            + arg_struct_args
                            )))
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
    # }}}

    # {{{ library

    @property
    def known_callables(self):
        from loopy.library.random123 import get_random123_callables

        # order matters: e.g. prefer our abs() over that of the
        # superclass
        callables = super().known_callables
        callables.update(get_pyopencl_callables())
        callables.update(get_random123_callables(self.target))
        return callables

    def preamble_generators(self):
        return ([
            pyopencl_preamble_generator,
            ] + super().preamble_generators())

    # }}}

    def get_expression_to_c_expression_mapper(self, codegen_state):
        return ExpressionToPyOpenCLCExpressionMapper(codegen_state)

# }}}


# {{{ volatile mem acccess target

class VolatileMemPyOpenCLCASTBuilder(PyOpenCLCASTBuilder):
    def get_expression_to_c_expression_mapper(self, codegen_state):
        from loopy.target.opencl import \
                VolatileMemExpressionToOpenCLCExpressionMapper
        return VolatileMemExpressionToOpenCLCExpressionMapper(codegen_state)


class VolatileMemPyOpenCLTarget(PyOpenCLTarget):
    def get_device_ast_builder(self):
        return VolatileMemPyOpenCLCASTBuilder(self)

# }}}

# vim: foldmethod=marker

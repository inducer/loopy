"""CUDA target integrated with PyCUDA."""

__copyright__ = """
Copyright (C) 2015 Andreas Kloeckner
Copyright (C) 2022 Kaushik Kulkarni
"""

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
import pymbolic.primitives as p
import genpy

from loopy.target.cuda import (CudaTarget, CudaCASTBuilder,
                               ExpressionToCudaCExpressionMapper)
from loopy.target.python import PythonASTBuilderBase
from loopy.target.c import CMathCallable
from loopy.diagnostic import LoopyError
from loopy.types import NumpyType
from loopy.codegen import CodeGenerationState
from loopy.codegen.result import CodeGenerationResult
from cgen import Generable

import logging
logger = logging.getLogger(__name__)


# {{{ preamble generator

def pycuda_preamble_generator(preamble_info):
    has_complex = False

    for dtype in preamble_info.seen_dtypes:
        if dtype.involves_complex():
            has_complex = True

    if has_complex:
        yield ("03_include_complex_header", """
            #include <pycuda-complex.hpp>
            """)

# }}}


# {{{ PyCudaCallable

class PyCudaCallable(CMathCallable):
    def with_types(self, arg_id_to_dtype, callables_table):
        if any(dtype.is_complex() for dtype in arg_id_to_dtype.values()):
            if self.name in ["abs", "real", "imag"]:
                if not (set(arg_id_to_dtype) <= {0, -1}):
                    raise LoopyError(f"'{self.name}' takes only one argument")
                if arg_id_to_dtype.get(0) is None:
                    # not specialized enough
                    return (self.copy(arg_id_to_dtype=arg_id_to_dtype),
                            callables_table)
                else:
                    real_dtype = np.empty(0,
                                          arg_id_to_dtype[0].numpy_dtype).real.dtype
                    arg_id_to_dtype = arg_id_to_dtype.copy()
                    arg_id_to_dtype[-1] = NumpyType(real_dtype)
                    return (self.copy(arg_id_to_dtype=arg_id_to_dtype,
                                      name_in_target=self.name),
                            callables_table)
            elif self.name in ["sqrt", "conj",
                               "sin",  "cos", "tan",
                               "sinh",  "cosh", "tanh", "exp",
                               "log", "log10"]:
                if not (set(arg_id_to_dtype) <= {0, -1}):
                    raise LoopyError(f"'{self.name}' takes only one argument")
                if arg_id_to_dtype.get(0) is None:
                    # not specialized enough
                    return (self.copy(arg_id_to_dtype=arg_id_to_dtype),
                            callables_table)
                else:
                    arg_id_to_dtype = arg_id_to_dtype.copy()
                    arg_id_to_dtype[-1] = arg_id_to_dtype[0]
                    return (self.copy(arg_id_to_dtype=arg_id_to_dtype,
                                      name_in_target=self.name),
                            callables_table)
            else:
                raise LoopyError(f"'{self.name}' does not take complex"
                                 " arguments.")
        else:
            if self.name in ["real", "imag", "conj"]:
                if arg_id_to_dtype.get(0):
                    raise NotImplementedError("'{self.name}' for real arguments"
                                              ", not yet supported.")
            return super().with_types(arg_id_to_dtype, callables_table)


def get_pycuda_callables():
    cmath_ids = ["abs", "acos", "asin", "atan", "cos", "cosh", "sin",
                 "sinh", "pow", "atan2", "tanh", "exp", "log", "log10",
                 "sqrt", "ceil", "floor", "max", "min", "fmax", "fmin",
                 "fabs", "tan", "erf", "erfc", "isnan", "real", "imag",
                 "conj"]
    return {id_: PyCudaCallable(id_) for id_ in cmath_ids}

# }}}


# {{{ expression mapper

def _get_complex_tmplt_arg(dtype) -> str:
    if dtype == np.complex128:
        return "double"
    elif dtype == np.complex64:
        return "float"
    else:
        raise RuntimeError(f"unsupported complex type {dtype}.")


class ExpressionToPyCudaCExpressionMapper(ExpressionToCudaCExpressionMapper):
    """
    .. note::

        - PyCUDA (very conveniently) provides access to complex arithmetic
          headers which is not the default in CUDA. To access such additional
          features we introduce this mapper.
    """
    def wrap_in_typecast_lazy(self, actual_type_func, needed_dtype, s):
        if needed_dtype.is_complex():
            return self.wrap_in_typecast(actual_type_func(), needed_dtype, s)
        else:
            return super().wrap_in_typecast_lazy(actual_type_func,
                                                 needed_dtype, s)

    def wrap_in_typecast(self, actual_type, needed_dtype, s):
        if not actual_type.is_complex() and needed_dtype.is_complex():
            tmplt_arg = _get_complex_tmplt_arg(needed_dtype.numpy_dtype)
            return p.Variable(f"pycuda::complex<{tmplt_arg}>")(s)
        else:
            return super().wrap_in_typecast_lazy(actual_type,
                                                 needed_dtype, s)

    def map_constant(self, expr, type_context):
        if isinstance(expr, (complex, np.complexfloating)):
            try:
                dtype = expr.dtype
            except AttributeError:
                # (COMPLEX_GUESS_LOGIC) This made it through type 'guessing' in
                # type inference, and it was concluded there (search for
                # COMPLEX_GUESS_LOGIC in loopy.type_inference), that no
                # accuracy was lost by using single precision.
                dtype = np.complex64
            else:
                tmplt_arg = _get_complex_tmplt_arg(dtype)

            return p.Variable(f"pycuda::complex<{tmplt_arg}>")(self.rec(expr.real,
                                                                type_context),
                                                       self.rec(expr.imag,
                                                                type_context))

        return super().map_constant(expr, type_context)

# }}}


# {{{ target

class PyCudaTarget(CudaTarget):
    """A code generation target that takes special advantage of :mod:`pycuda`
    features such as run-time knowledge of the target device (to generate
    warnings) and support for complex numbers.
    """

    # FIXME make prefixes conform to naming rules
    # (see Reference: Loopy’s Model of a Kernel)

    host_program_name_prefix = "_lpy_host_"
    host_program_name_suffix = ""

    def __init__(self, pycuda_module_name="_lpy_cuda"):
        # import pycuda.tools import to populate the TYPE_REGISTRY
        import pycuda.tools  # noqa: F401
        super().__init__()
        self.pycuda_module_name = pycuda_module_name

    # NB: Not including 'device', as that is handled specially here.
    hash_fields = CudaTarget.hash_fields + (
            "pycuda_module_name",)
    comparison_fields = CudaTarget.comparison_fields + (
            "pycuda_module_name",)

    def get_host_ast_builder(self):
        return PyCudaPythonASTBuilder(self)

    def get_device_ast_builder(self):
        return PyCudaCASTBuilder(self)

    def get_kernel_executor_cache_key(self, **kwargs):
        return (kwargs["entrypoint"],)

    def get_dtype_registry(self):
        from pycuda.compyte.dtypes import TYPE_REGISTRY
        return TYPE_REGISTRY

    def preprocess_translation_unit_for_passed_args(self, t_unit, epoint,
                                                    passed_args_dict):

        # {{{ ValueArgs -> GlobalArgs if passed as array shapes

        from loopy.kernel.data import ValueArg, GlobalArg
        import pycuda.gpuarray as cu_np

        knl = t_unit[epoint]
        new_args = []

        for arg in knl.args:
            if isinstance(arg, ValueArg):
                if (arg.name in passed_args_dict
                        and isinstance(passed_args_dict[arg.name], cu_np.GPUArray)
                        and passed_args_dict[arg.name].shape == ()):
                    arg = GlobalArg(name=arg.name, dtype=arg.dtype, shape=(),
                                    is_output=False, is_input=True)

            new_args.append(arg)

        knl = knl.copy(args=new_args)

        t_unit = t_unit.with_kernel(knl)

        # }}}

        return t_unit

    def get_kernel_executor(self, t_unit, **kwargs):
        from loopy.target.pycuda_execution import PyCudaKernelExecutor

        epoint = kwargs.pop("entrypoint")
        t_unit = self.preprocess_translation_unit_for_passed_args(t_unit,
                                                                  epoint,
                                                                  kwargs)

        return PyCudaKernelExecutor(t_unit, entrypoint=epoint)


class PyCudaWithPackedArgsTarget(PyCudaTarget):

    def get_kernel_executor(self, t_unit, **kwargs):
        from loopy.target.pycuda_execution import PyCudaWithPackedArgsKernelExecutor

        epoint = kwargs.pop("entrypoint")
        t_unit = self.preprocess_translation_unit_for_passed_args(t_unit,
                                                                  epoint,
                                                                  kwargs)

        return PyCudaWithPackedArgsKernelExecutor(t_unit, entrypoint=epoint)

    def get_host_ast_builder(self):
        return PyCudaWithPackedArgsPythonASTBuilder(self)

    def get_device_ast_builder(self):
        return PyCudaWithPackedArgsCASTBuilder(self)

# }}}


# {{{ host ast builder

class PyCudaPythonASTBuilder(PythonASTBuilderBase):
    """A Python host AST builder for integration with PyCuda.
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
                ["_lpy_cuda_functions"]
                + [arg_name for arg_name in kai.passed_arg_names]
                + ["wait_for=()", "allocator=None", "stream=None"])

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
                            S("if _tv is not None: _tv.free()"))
                        ] if self._get_global_temporaries(codegen_state) else []
                    ) + [
                    Line(),
                    Return("_lpy_evt"),
                    ]))

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        # no such thing in Python
        return None

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

    def get_kernel_call(self,
                        codegen_state, subkernel_name, grid, block):
        from genpy import Suite, Assign, Line, Comment, Statement
        from pymbolic.mapper.stringifier import PREC_NONE

        from loopy.schedule.tools import get_subkernel_arg_info
        skai = get_subkernel_arg_info(
                codegen_state.kernel, subkernel_name)

        # make grid/block a 3-tuple
        grid = grid + (1,) * (3-len(grid))
        block = block + (1,) * (3-len(block))
        ecm = self.get_expression_to_code_mapper(codegen_state)

        grid_str = ecm(grid, prec=PREC_NONE, type_context="i")
        block_str = ecm(block, prec=PREC_NONE, type_context="i")

        return Suite([
            Comment("{{{ launch %s" % subkernel_name),
            Line(),
            Statement("for _lpy_cu_evt in wait_for: _lpy_cu_evt.synchronize()"),
            Line(),
            Assign("_lpy_knl", f"_lpy_cuda_functions['{subkernel_name}']"),
            Line(),
            Statement("_lpy_knl.prepared_async_call("
                      f"{grid_str}, {block_str}, "
                      "stream, "
                      f"{', '.join(arg_name for arg_name in skai.passed_names)}"
                      ")",),
            Assign("_lpy_evt", "_lpy_cuda.Event().record(stream)"),
            Assign("wait_for", "[_lpy_evt]"),
            Line(),
            Comment("}}}"),
            Line(),
        ])

    # }}}


class PyCudaWithPackedArgsPythonASTBuilder(PyCudaPythonASTBuilder):

    def get_kernel_call(self,
                        codegen_state, subkernel_name, grid, block):
        from genpy import Suite, Assign, Line, Comment, Statement
        from pymbolic.mapper.stringifier import PREC_NONE
        from loopy.kernel.data import ValueArg, ArrayArg

        from loopy.schedule.tools import get_subkernel_arg_info
        kernel = codegen_state.kernel
        skai = get_subkernel_arg_info(kernel, subkernel_name)

        # make grid/block a 3-tuple
        grid = grid + (1,) * (3-len(grid))
        block = block + (1,) * (3-len(block))
        ecm = self.get_expression_to_code_mapper(codegen_state)

        grid_str = ecm(grid, prec=PREC_NONE, type_context="i")
        block_str = ecm(block, prec=PREC_NONE, type_context="i")

        struct_format = []
        for arg_name in skai.passed_names:
            if arg_name in codegen_state.kernel.all_inames():
                struct_format.append(kernel.index_dtype.numpy_dtype.char)
                if kernel.index_dtype.numpy_dtype.itemsize < 8:
                    struct_format.append("x")
            elif arg_name in codegen_state.kernel.temporary_variables:
                struct_format.append("P")
            else:
                knl_arg = codegen_state.kernel.arg_dict[arg_name]
                if isinstance(knl_arg, ValueArg):
                    struct_format.append(knl_arg.dtype.numpy_dtype.char)
                    if knl_arg.dtype.numpy_dtype.itemsize < 8:
                        struct_format.append("x")
                else:
                    struct_format.append("P")

        def _arg_cast(arg_name: str) -> str:
            if arg_name in skai.passed_inames:
                return ("_lpy_np"
                        f".{codegen_state.kernel.index_dtype.numpy_dtype.name}"
                        f"({arg_name})")
            elif arg_name in skai.passed_temporaries:
                return f"_lpy_np.uintp(int({arg_name}))"
            else:
                knl_arg = codegen_state.kernel.arg_dict[arg_name]
                if isinstance(knl_arg, ValueArg):
                    return f"_lpy_np.{knl_arg.dtype.numpy_dtype.name}({arg_name})"
                else:
                    assert isinstance(knl_arg, ArrayArg)
                    return f"_lpy_np.uintp(int({arg_name}))"

        return Suite([
            Comment("{{{ launch %s" % subkernel_name),
            Line(),
            Statement("for _lpy_cu_evt in wait_for: _lpy_cu_evt.synchronize()"),
            Line(),
            Assign("_lpy_knl", f"_lpy_cuda_functions['{subkernel_name}']"),
            Line(),
            Assign("_lpy_args_on_dev", f"allocator({len(skai.passed_names)*8})"),
            Assign("_lpy_args_on_host",
                   f"_lpy_struct.pack('{''.join(struct_format)}',"
                   f"{','.join(_arg_cast(arg) for arg in skai.passed_names)})"),
            Statement("_lpy_cuda.memcpy_htod(_lpy_args_on_dev, _lpy_args_on_host)"),
            Line(),
            Statement("_lpy_knl.prepared_async_call("
                      f"{grid_str}, {block_str}, "
                      "stream, _lpy_args_on_dev)"),
            Assign("_lpy_evt", "_lpy_cuda.Event().record(stream)"),
            Assign("wait_for", "[_lpy_evt]"),
            Line(),
            Comment("}}}"),
            Line(),
        ])

# }}}


# {{{ device ast builder

class PyCudaCASTBuilder(CudaCASTBuilder):
    """A C device AST builder for integration with PyCUDA.
    """

    # {{{ library

    def preamble_generators(self):
        return ([pycuda_preamble_generator]
                + super().preamble_generators())

    @property
    def known_callables(self):
        callables = super().known_callables
        callables.update(get_pycuda_callables())
        return callables

    # }}}

    def get_expression_to_c_expression_mapper(self, codegen_state):
        return ExpressionToPyCudaCExpressionMapper(codegen_state)


class PyCudaWithPackedArgsCASTBuilder(PyCudaCASTBuilder):
    def arg_struct_name(self, kernel_name: str):
        return f"_lpy_{kernel_name}_packed_args"

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        from loopy.target.c import FunctionDeclarationWrapper
        from cgen import FunctionDeclaration, Value, Pointer, Extern
        from cgen.cuda import CudaGlobal, CudaDevice, CudaLaunchBounds

        kernel = codegen_state.kernel

        assert kernel.linearization is not None
        name = codegen_result.current_program(codegen_state).name
        arg_type = self.arg_struct_name(name)

        if self.target.fortran_abi:
            name += "_"

        fdecl = FunctionDeclaration(
            Value("void", name),
            [Pointer(Value(arg_type, "_lpy_args"))])

        if codegen_state.is_entrypoint:
            fdecl = CudaGlobal(fdecl)
            if self.target.extern_c:
                fdecl = Extern("C", fdecl)

            from loopy.schedule import get_insn_ids_for_block_at
            _, lsize = kernel.get_grid_sizes_for_insn_ids_as_exprs(
                get_insn_ids_for_block_at(kernel.linearization, schedule_index),
                codegen_state.callables_table)

            from loopy.symbolic import get_dependencies
            if not get_dependencies(lsize):
                # Sizes can't have parameter dependencies if they are
                # to be used in static thread block size.
                from pytools import product
                nthreads = product(lsize)

                fdecl = CudaLaunchBounds(nthreads, fdecl)

            return FunctionDeclarationWrapper(fdecl)
        else:
            return CudaDevice(fdecl)

    def get_function_definition(
            self, codegen_state: CodeGenerationState,
            codegen_result: CodeGenerationResult,
            schedule_index: int, function_decl: Generable, function_body: Generable
            ) -> Generable:
        from typing import cast
        from loopy.target.c import generate_array_literal
        from loopy.schedule import CallKernel
        from loopy.schedule.tools import get_subkernel_arg_info
        from loopy.kernel.data import ValueArg, AddressSpace
        from cgen import (FunctionBody,
                          Module as Collection,
                          Initializer,
                          Line, Value, Pointer, Struct as GenerableStruct)
        kernel = codegen_state.kernel
        assert kernel.linearization is not None
        assert isinstance(kernel.linearization[schedule_index], CallKernel)
        kernel_name = (cast(CallKernel,
                            kernel.linearization[schedule_index])
                       .kernel_name)

        skai = get_subkernel_arg_info(kernel, kernel_name)

        result = []

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

        # {{{ declare+unpack the struct type

        struct_fields = []

        for arg_name in skai.passed_names:
            if arg_name in skai.passed_inames:
                struct_fields.append(
                    Value(self.target.dtype_to_typename(kernel.index_dtype),
                          f"{arg_name}, __padding_{arg_name}"))
            elif arg_name in skai.passed_temporaries:
                tv = kernel.temporary_variables[arg_name]
                struct_fields.append(Pointer(
                    Value(self.target.dtype_to_typename(tv.dtype), arg_name)))
            else:
                knl_arg = kernel.arg_dict[arg_name]
                if isinstance(knl_arg, ValueArg):
                    struct_fields.append(
                        Value(self.target.dtype_to_typename(knl_arg.dtype),
                              f"{arg_name}, __padding_{arg_name}"))
                else:
                    struct_fields.append(
                        Pointer(Value(self.target.dtype_to_typename(knl_arg.dtype),
                                      arg_name)))

        function_body.insert(0, Line())
        for arg_name in skai.passed_names[::-1]:
            function_body.insert(0, Initializer(
                self.arg_to_cgen_declarator(
                    kernel, arg_name,
                    arg_name in kernel.get_written_variables()),
                f"_lpy_args->{arg_name}"
            ))

        # }}}

        fbody = FunctionBody(function_decl, function_body)

        return Collection([*result,
                           Line(),
                           GenerableStruct(self.arg_struct_name(kernel_name),
                                           struct_fields),
                           Line(),
                           fbody])


# }}}

# vim: foldmethod=marker

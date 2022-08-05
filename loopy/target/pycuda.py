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

import logging
logger = logging.getLogger(__name__)


# {{{ preamble generator

def pycuda_preamble_generator(preamble_info):
    has_complex = False

    for dtype in preamble_info.seen_dtypes:
        if dtype.involves_complex():
            has_complex = True

    if has_complex:
        yield ("10_include_complex_header", """
            #include <pycuda-complex.hpp>
            """)

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
          we introduce this mapper.
    """
    def wrap_in_typecast_lazy(self, actual_type_func, needed_dtype, s):
        if needed_dtype.is_complex():
            return self.wrap_in_typecast(actual_type_func(), needed_dtype, s)
        else:
            return super().wrap_in_typecast_lazy(actual_type_func,
                                                 needed_dtype, s)

    def wrap_in_typecast(self, actual_type, needed_dtype, s):
        if not actual_type.is_complex() and needed_dtype.is_complex():
            return p.Variable("complex")(s)
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

            return p.Variable(f"complex<{tmplt_arg}>")(self.rec(expr.real,
                                                                type_context),
                                                       self.rec(expr.imag,
                                                                type_context))

        return super().map_constant(expr, type_context)

    def map_power(self, expr, type_context):
        tgt_dtype = self.infer_type(expr)

        if not self.allow_complex or (not tgt_dtype.is_complex()):
            return super().map_power(expr, type_context)

        if expr.exponent in [2, 3, 4]:
            value = expr.base
            for _i in range(expr.exponent-1):
                value = value * expr.base
            return self.rec(value, type_context)
        else:
            return p.Variable("pow")(
                    self.rec(expr.base, type_context, tgt_dtype),
                    self.rec(expr.exponent, type_context))

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
        return ()

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

# }}}


# {{{ host code: value arg setup

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
                            # https://documen.tician.de/pyopencl/tools.html#pyopencl.tools.ImmediateAllocator
                            S("if _tv is not None: _tv.release()"))
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
                      f"{', '.join(arg_name for arg_name in skai.passed_arg_names)}"
                      ")",),
            Assign("_lpy_evt", "_lpy_cuda.Event().record(stream)"),
            Assign("wait_for", "[_lpy_evt]"),
            Line(),
            Comment("}}}"),
            Line(),
        ])

    # }}}

# }}}


# {{{ device ast builder

class PyCudaCASTBuilder(CudaCASTBuilder):
    """A C device AST builder for integration with PyCUDA.
    """

    # {{{ library

    def preamble_generators(self):
        return ([pycuda_preamble_generator]
                + super().preamble_generators())

    # }}}

    def get_expression_to_c_expression_mapper(self, codegen_state):
        return ExpressionToPyCudaCExpressionMapper(codegen_state)

# }}}

# vim: foldmethod=marker

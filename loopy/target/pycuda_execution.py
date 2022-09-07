__copyright__ = """
Copyright (C) 2012 Andreas Kloeckner
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


from typing import (Sequence, Tuple, Union, Callable, Any, Optional,
                    TYPE_CHECKING)
from dataclasses import dataclass

import numpy as np
from immutables import Map

from pytools import memoize_method
from pytools.codegen import Indentation, CodeGenerator

from loopy.types import LoopyType
from loopy.typing import ExpressionT
from loopy.kernel import LoopKernel
from loopy.kernel.data import ArrayArg
from loopy.translation_unit import TranslationUnit
from loopy.schedule.tools import KernelArgInfo
from loopy.target.execution import (
    KernelExecutorBase, ExecutionWrapperGeneratorBase)
import logging
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import pycuda.driver as cuda


# {{{ invoker generation

# /!\ This code runs in a namespace controlled by the user.
# Prefix all auxiliary variables with "_lpy".


class PyCudaExecutionWrapperGenerator(ExecutionWrapperGeneratorBase):
    """
    Specialized form of the :class:`ExecutionWrapperGeneratorBase` for
    pycuda execution
    """

    def __init__(self):
        system_args = [
            "_lpy_cuda_functions", "stream=None", "allocator=None", "wait_for=()",
            # ignored if options.no_numpy
            "out_host=None"
            ]
        super().__init__(system_args)

    def python_dtype_str_inner(self, dtype):
        from pycuda.tools import dtype_to_ctype
        # Test for types built into numpy. dtype.isbuiltin does not work:
        # https://github.com/numpy/numpy/issues/4317
        # Guided by https://numpy.org/doc/stable/reference/arrays.scalars.html
        if issubclass(dtype.type, (np.bool_, np.number)):
            name = dtype.name
            if dtype.type == np.bool_:
                name = "bool8"
            return f"_lpy_np.dtype(_lpy_np.{name})"
        else:
            return ('_lpy_cuda_tools.get_or_register_dtype("%s")'
                    % dtype_to_ctype(dtype))

    # {{{ handle non-numpy args

    def handle_non_numpy_arg(self, gen, arg):
        gen("if isinstance(%s, _lpy_np.ndarray):" % arg.name)
        with Indentation(gen):
            gen("# retain originally passed array")
            gen(f"_lpy_{arg.name}_np_input = {arg.name}")
            gen("# synchronous, nothing to worry about")
            gen("%s = _lpy_cuda_array.to_gpu_async("
                    "%s, allocator=allocator, stream=stream)"
                    % (arg.name, arg.name))
            gen("_lpy_encountered_numpy = True")
        gen("elif %s is not None:" % arg.name)
        with Indentation(gen):
            gen("_lpy_encountered_dev = True")
            gen("_lpy_%s_np_input = None" % arg.name)
        gen("else:")
        with Indentation(gen):
            gen("_lpy_%s_np_input = None" % arg.name)

        gen("")

    # }}}

    # {{{ handle allocation of unspecified arguments

    def handle_alloc(
            self, gen: CodeGenerator, arg: ArrayArg,
            strify: Callable[[Union[ExpressionT, Tuple[ExpressionT]]], str],
            skip_arg_checks: bool) -> None:
        """
        Handle allocation of non-specified arguments for pycuda execution
        """
        from pymbolic import var

        from loopy.kernel.array import get_strides
        strides = get_strides(arg)
        num_axes = len(strides)

        itemsize = arg.dtype.numpy_dtype.itemsize
        for i in range(num_axes):
            gen("_lpy_ustrides_%d = %s" % (i, strify(strides[i])))

        if not skip_arg_checks:
            for i in range(num_axes):
                gen("assert _lpy_ustrides_%d >= 0, "
                        "\"'%s' has negative stride in axis %d\""
                        % (i, arg.name, i))

        assert isinstance(arg.shape, tuple)
        sym_ustrides = tuple(
                var("_lpy_ustrides_%d" % i)
                for i in range(num_axes))
        sym_shape = tuple(arg.shape[i] for i in range(num_axes))

        size_expr = (sum(astrd*(alen-1)
            for alen, astrd in zip(sym_shape, sym_ustrides))
            + 1)

        gen("_lpy_size = %s" % strify(size_expr))
        sym_strides = tuple(itemsize*s_i for s_i in sym_ustrides)

        dtype_name = self.python_dtype_str(gen, arg.dtype.numpy_dtype)
        gen(f"{arg.name} = _lpy_cuda_array.GPUArray({strify(sym_shape)}, "
                f"{dtype_name}, strides={strify(sym_strides)}, "
                f"gpudata=allocator({strify(itemsize * var('_lpy_size'))}), "
                "allocator=allocator)")

        for i in range(num_axes):
            gen("del _lpy_ustrides_%d" % i)
        gen("del _lpy_size")
        gen("")

    # }}}

    def target_specific_preamble(self, gen):
        """
        Add default pycuda imports to preamble
        """
        gen.add_to_preamble("import numpy as _lpy_np")
        gen.add_to_preamble("import pycuda.driver as _lpy_cuda")
        gen.add_to_preamble("import pycuda.gpuarray as _lpy_cuda_array")
        gen.add_to_preamble("import pycuda.tools as _lpy_cuda_tools")
        gen.add_to_preamble("import struct as _lpy_struct")
        from loopy.target.c.c_execution import DEF_EVEN_DIV_FUNCTION
        gen.add_to_preamble(DEF_EVEN_DIV_FUNCTION)

    def initialize_system_args(self, gen):
        """
        Initializes possibly empty system arguments
        """
        gen("if allocator is None:")
        with Indentation(gen):
            gen("allocator = _lpy_cuda.mem_alloc")
        gen("")

    # {{{ generate invocation

    def generate_invocation(self, gen: CodeGenerator, kernel: LoopKernel,
            kai: KernelArgInfo, host_program_name: str, args: Sequence[str]) -> None:
        arg_list = (["_lpy_cuda_functions"]
                    + list(args)
                    + ["stream=stream", "wait_for=wait_for", "allocator=allocator"])
        gen(f"_lpy_evt = {host_program_name}({', '.join(arg_list)})")

    # }}}

    # {{{ generate_output_handler

    def generate_output_handler(self, gen: CodeGenerator,
            kernel: LoopKernel, kai: KernelArgInfo) -> None:
        options = kernel.options

        if not options.no_numpy:
            gen("if out_host is None and (_lpy_encountered_numpy "
                    "and not _lpy_encountered_dev):")
            with Indentation(gen):
                gen("out_host = True")

            for arg_name in kai.passed_arg_names:
                arg = kernel.arg_dict[arg_name]
                if arg.is_output:
                    np_name = "_lpy_%s_np_input" % arg.name
                    gen("if out_host or %s is not None:" % np_name)
                    with Indentation(gen):
                        gen("%s = %s.get(stream=stream, ary=%s)"
                            % (arg.name, arg.name, np_name))

            gen("")

        if options.return_dict:
            gen("return _lpy_evt, {%s}"
                    % ", ".join(f'"{arg_name}": {arg_name}'
                        for arg_name in kai.passed_arg_names
                        if kernel.arg_dict[arg_name].is_output))
        else:
            out_names = [arg_name for arg_name in kai.passed_arg_names
                    if kernel.arg_dict[arg_name].is_output]
            if out_names:
                gen("return _lpy_evt, (%s,)"
                        % ", ".join(out_names))
            else:
                gen("return _lpy_evt, ()")

    # }}}

    def generate_host_code(self, gen, codegen_result):
        gen.add_to_preamble(codegen_result.host_code())

    def get_arg_pass(self, arg):
        return "%s.gpudata" % arg.name

# }}}


@dataclass(frozen=True)
class _KernelInfo:
    t_unit: TranslationUnit
    cuda_functions: Map[str, "cuda.Function"]
    invoker: Callable[..., Any]


# {{{ kernel executor

class PyCudaKernelExecutor(KernelExecutorBase):
    """
    An object connecting a kernel to a :mod:`pycuda`
    for execution.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def get_invoker_uncached(self, t_unit, entrypoint, codegen_result):
        generator = PyCudaExecutionWrapperGenerator()
        return generator(t_unit, entrypoint, codegen_result)

    def get_wrapper_generator(self):
        return PyCudaExecutionWrapperGenerator()

    def _get_arg_dtypes(self, knl, subkernel_name):
        from loopy.schedule.tools import get_subkernel_arg_info
        from loopy.kernel.data import ValueArg

        skai = get_subkernel_arg_info(knl, subkernel_name)
        arg_dtypes = []
        for arg in skai.passed_names:
            if arg in skai.passed_inames:
                arg_dtypes.append(knl.index_dtype.numpy_dtype)
            elif arg in skai.passed_temporaries:
                arg_dtypes.append("P")
            else:
                assert arg in knl.arg_dict
                if isinstance(knl.arg_dict[arg], ValueArg):
                    arg_dtypes.append(knl.arg_dict[arg].dtype.numpy_dtype)
                else:
                    # Array Arg
                    arg_dtypes.append("P")

        return arg_dtypes

    @memoize_method
    def translation_unit_info(self,
                              arg_to_dtype: Optional[Map[str, LoopyType]] = None
                              ) -> _KernelInfo:
        t_unit = self.get_typed_and_scheduled_translation_unit(self.entrypoint,
                                                               arg_to_dtype)

        # FIXME: now just need to add the types to the arguments
        from loopy.codegen import generate_code_v2
        from loopy.target.execution import get_highlighted_code
        codegen_result = generate_code_v2(t_unit)

        dev_code = codegen_result.device_code()
        epoint_knl = t_unit[self.entrypoint]

        if t_unit[self.entrypoint].options.write_code:
            #FIXME: redirect to "translation unit" level option as well.
            output = dev_code
            if self.t_unit[self.entrypoint].options.allow_terminal_colors:
                output = get_highlighted_code(output)

            if epoint_knl.options.write_code is True:
                print(output)
            else:
                with open(epoint_knl.options.write_code, "w") as outf:
                    outf.write(output)

        if epoint_knl.options.edit_code:
            #FIXME: redirect to "translation unit" level option as well.
            from pytools import invoke_editor
            dev_code = invoke_editor(dev_code, "code.cu")

        from pycuda.compiler import SourceModule
        from loopy.kernel.tools import get_subkernels

        #FIXME: redirect to "translation unit" level option as well.
        src_module = SourceModule(dev_code,
                                  options=epoint_knl.options.build_options)

        cuda_functions = Map({name: (src_module
                                     .get_function(name)
                                     .prepare(self._get_arg_dtypes(epoint_knl, name))
                                     )
                              for name in get_subkernels(epoint_knl)})
        return _KernelInfo(
            t_unit=t_unit,
            cuda_functions=cuda_functions,
            invoker=self.get_invoker(t_unit, self.entrypoint, codegen_result))

    def __call__(self, *,
                 stream=None, allocator=None, wait_for=(), out_host=None,
                 **kwargs):
        """
        :arg allocator: a callable that accepts a byte count and returns
            an instance of :class:`pycuda.driver.DeviceAllocation`. Typically
            one of :func:`pycuda.driver.mem_alloc` or
            :meth:`pycuda.tools.DeviceMemoryPool.allocate`.
        :arg wait_for: A sequence of :class:`pycuda.driver.Event` instances
            for which to wait before launching the CUDA kernels.
        :arg out_host: :class:`bool`
            Decides whether output arguments (i.e. arguments
            written by the kernel) are to be returned as
            :mod:`numpy` arrays. *True* for yes, *False* for no.

            For the default value of *None*, if all (input) array
            arguments are :mod:`numpy` arrays, defaults to
            returning :mod:`numpy` arrays as well.

        :returns: ``(evt, output)`` where *evt* is a
            :class:`pycuda.driver.Event` that is recorded right after the
            kernel has been launched and output is a tuple of output arguments
            (arguments that are written as part of the kernel). The order is
            given by the order of kernel arguments. If this order is
            unspecified (such as when kernel arguments are inferred
            automatically), enable :attr:`loopy.Options.return_dict` to make
            *output* a :class:`dict` instead, with keys of argument names and
            values of the returned arrays.
        """

        if "entrypoint" in kwargs:
            assert kwargs.pop("entrypoint") == self.entrypoint
            from warnings import warn
            warn("Obtained a redundant argument 'entrypoint'. This will"
                 " be an error in 2023.", DeprecationWarning, stacklevel=2)

        if __debug__:
            self.check_for_required_array_arguments(kwargs.keys())

        if self.packing_controller is not None:
            kwargs = self.packing_controller(kwargs)

        translation_unit_info = self.translation_unit_info(self.arg_to_dtype(kwargs))

        return translation_unit_info.invoker(
                translation_unit_info.cuda_functions, stream, allocator, wait_for,
                out_host, **kwargs)


class PyCudaWithPackedArgsKernelExecutor(PyCudaKernelExecutor):

    def _get_arg_dtypes(self, knl, subkernel_name):
        return ["P"]

# }}}

# vim: foldmethod=marker

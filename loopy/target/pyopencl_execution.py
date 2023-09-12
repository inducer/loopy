__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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


from typing import Sequence, Tuple, Union, Callable, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from immutables import Map

from pytools import memoize_method
from pytools.codegen import Indentation, CodeGenerator

from loopy.types import LoopyType
from loopy.typing import ExpressionT
from loopy.kernel import LoopKernel
from loopy.kernel.data import ArrayArg
from loopy.schedule.tools import KernelArgInfo
from loopy.target.execution import (
    ExecutorBase, ExecutionWrapperGeneratorBase)
import logging
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import pyopencl as cl


# {{{ invoker generation

# /!\ This code runs in a namespace controlled by the user.
# Prefix all auxiliary variables with "_lpy".


class PyOpenCLExecutionWrapperGenerator(ExecutionWrapperGeneratorBase):
    """
    Specialized form of the :class:`ExecutionWrapperGeneratorBase` for
    pyopencl execution
    """

    def __init__(self):
        system_args = [
            "_lpy_cl_kernels", "queue", "allocator=None", "wait_for=None",
            # ignored if options.no_numpy
            "out_host=None"
            ]
        super().__init__(system_args)

    def python_dtype_str_inner(self, dtype):
        import pyopencl.tools as cl_tools
        # Test for types built into numpy. dtype.isbuiltin does not work:
        # https://github.com/numpy/numpy/issues/4317
        # Guided by https://numpy.org/doc/stable/reference/arrays.scalars.html
        if issubclass(dtype.type, (np.bool_, np.number)):
            name = dtype.name
            if dtype.type == np.bool_:
                name = "bool_"
            return f"_lpy_np.dtype(_lpy_np.{name})"
        else:
            return ('_lpy_cl_tools.get_or_register_dtype("%s")'
                    % cl_tools.dtype_to_ctype(dtype))

    # {{{ handle non-numpy args

    def handle_non_numpy_arg(self, gen, arg):
        gen("if isinstance(%s, _lpy_np.ndarray):" % arg.name)
        with Indentation(gen):
            gen("# retain originally passed array")
            gen(f"_lpy_{arg.name}_np_input = {arg.name}")
            gen("# synchronous, nothing to worry about")
            gen("%s = _lpy_cl_array.to_device("
                    "queue, %s, allocator=allocator)"
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
        Handle allocation of non-specified arguments for pyopencl execution
        """
        from pymbolic import var

        from loopy.kernel.array import get_strides
        strides = get_strides(arg)
        num_axes = len(strides)

        assert arg.dtype is not None
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
        gen(f"{arg.name} = _lpy_cl_array.Array(None, {strify(sym_shape)}, "
                f"{dtype_name}, strides={strify(sym_strides)}, "
                f"data=allocator({strify(itemsize * var('_lpy_size'))}), "
                "allocator=allocator, "
                "_fast=True, _size=_lpy_size, "
                "_context=queue.context, _queue=queue)")

        for i in range(num_axes):
            gen("del _lpy_ustrides_%d" % i)
        gen("del _lpy_size")
        gen("")

    # }}}

    def target_specific_preamble(self, gen):
        """
        Add default pyopencl imports to preamble
        """
        gen.add_to_preamble("import numpy as _lpy_np")
        gen.add_to_preamble("import pyopencl as _lpy_cl")
        gen.add_to_preamble("import pyopencl.array as _lpy_cl_array")
        gen.add_to_preamble("import pyopencl.tools as _lpy_cl_tools")
        gen.add_to_preamble("from struct import pack as _lpy_pack")
        from loopy.target.c.c_execution import DEF_EVEN_DIV_FUNCTION
        gen.add_to_preamble(DEF_EVEN_DIV_FUNCTION)

    def initialize_system_args(self, gen):
        """
        Initializes possibly empty system arguments
        """
        gen("if allocator is None:")
        with Indentation(gen):
            gen("allocator = _lpy_cl_tools.DeferredAllocator(queue.context)")
        gen("")

    # {{{ generate invocation

    def generate_invocation(self, gen: CodeGenerator, kernel: LoopKernel,
            kai: KernelArgInfo, host_program_name: str, args: Sequence[str]) -> None:
        if kernel.options.cl_exec_manage_array_events:
            gen("""
                if wait_for is None:
                    wait_for = []
                """)

            gen("")
            for arg_name in kai.passed_arg_names:
                arg = kernel.arg_dict[arg_name]
                if isinstance(arg, ArrayArg):
                    gen(
                            "wait_for.extend({arg_name}.events)"
                            .format(arg_name=arg.name))

            gen("")

        arg_list = (["_lpy_cl_kernels", "queue"]
                + list(args)
                + ["wait_for=wait_for", "allocator=allocator"])
        gen(f"_lpy_evt = {host_program_name}({', '.join(arg_list)})")

        if kernel.options.cl_exec_manage_array_events:
            gen("")
            for arg_name in kai.passed_arg_names:
                arg = kernel.arg_dict[arg_name]
                if (isinstance(arg, ArrayArg)
                        and arg.name in kernel.get_written_variables()):
                    gen(f"{arg.name}.add_event(_lpy_evt)")

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
                        gen("%s = %s.get(queue=queue, ary=%s)"
                            % (arg.name, arg.name, np_name))

            gen("")

        if options.return_dict:
            gen("return _lpy_evt, {%s}"
                    % ", ".join(f'"{arg_name}": {arg_name}'
                        for arg_name in kai.passed_arg_names
                        if kernel.arg_dict[arg_name].is_output))
        else:
            passed_arg_names_set = frozenset(kai.passed_arg_names)
            out_names = [
                    # Must ensure that these occur in the same order as in
                    # kernel.args.
                    arg.name
                    for arg in kernel.args
                    if arg.name in passed_arg_names_set
                    if arg.is_output]
            if out_names:
                gen("return _lpy_evt, (%s,)"
                        % ", ".join(out_names))
            else:
                gen("return _lpy_evt, ()")

    # }}}

    def generate_host_code(self, gen, codegen_result):
        gen.add_to_preamble(codegen_result.host_code())

    def get_arg_pass(self, arg):
        return "%s.base_data" % arg.name

# }}}


@dataclass(frozen=True)
class _KernelInfo:
    cl_kernels: "_Kernels"
    invoker: Callable[..., Any]


class _Kernels:
    pass


# {{{ kernel executor

class PyOpenCLExecutor(ExecutorBase):
    """An object connecting a kernel to a :class:`pyopencl.Context`
    for execution.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, context: "cl.Context", t_unit, entrypoint):
        super().__init__(t_unit, entrypoint)

        self.context = context

    def get_invoker_uncached(self, t_unit, entrypoint, codegen_result):
        generator = PyOpenCLExecutionWrapperGenerator()
        return generator(t_unit, entrypoint, codegen_result)

    def get_wrapper_generator(self):
        return PyOpenCLExecutionWrapperGenerator()

    @memoize_method
    def translation_unit_info(
            self,
            arg_to_dtype: Optional[Map[str, LoopyType]] = None) -> _KernelInfo:
        t_unit = self.get_typed_and_scheduled_translation_unit(arg_to_dtype)

        # FIXME: now just need to add the types to the arguments
        from loopy.codegen import generate_code_v2
        from loopy.target.execution import get_highlighted_code
        codegen_result = generate_code_v2(t_unit)

        dev_code = codegen_result.device_code()

        if t_unit[self.entrypoint].options.write_code:
            #FIXME: redirect to "translation unit" level option as well.
            output = dev_code
            if self.t_unit[self.entrypoint].options.allow_terminal_colors:
                output = get_highlighted_code(output)

            if self.t_unit[self.entrypoint].options.write_code is True:
                print(output)
            else:
                with open(
                        self.t_unit[self.entrypoint].options.write_code, "w"
                        ) as outf:
                    outf.write(output)

        if t_unit[self.entrypoint].options.edit_code:
            #FIXME: redirect to "translation unit" level option as well.
            from pytools import invoke_editor
            dev_code = invoke_editor(dev_code, "code.cl")

        import pyopencl as cl

        #FIXME: redirect to "translation unit" level option as well.
        cl_program = (
                cl.Program(self.context, dev_code)
                .build(options=t_unit[self.entrypoint].options.build_options))

        cl_kernels = _Kernels()
        for dp in cl_program.kernel_names.split(";"):
            setattr(cl_kernels, dp, getattr(cl_program, dp))

        return _KernelInfo(
                cl_kernels=cl_kernels,
                invoker=self.get_invoker(t_unit, self.entrypoint, codegen_result))

    def __call__(self, queue, *,
            allocator=None, wait_for=None, out_host=None,
            **kwargs):
        """
        :arg allocator: a callable passed a byte count and returning
            a :class:`pyopencl.Buffer`. A :mod:`pyopencl` allocator
            maybe.
        :arg wait_for: A list of :class:`pyopencl.Event` instances
            for which to wait.
        :arg out_host: :class:`bool`
            Decides whether output arguments (i.e. arguments
            written by the kernel) are to be returned as
            :mod:`numpy` arrays. *True* for yes, *False* for no.

            For the default value of *None*, if all (input) array
            arguments are :mod:`numpy` arrays, defaults to
            returning :mod:`numpy` arrays as well.

        :returns: ``(evt, output)`` where *evt* is a :class:`pyopencl.Event`
            associated with the execution of the kernel, and
            output is a tuple of output arguments (arguments that
            are written as part of the kernel). The order is given
            by the order of kernel arguments. If this order is unspecified
            (such as when kernel arguments are inferred automatically),
            enable :attr:`loopy.Options.return_dict` to make *output* a
            :class:`dict` instead, with keys of argument names and values
            of the returned arrays.
        """

        if __debug__:
            self.check_for_required_array_arguments(kwargs.keys())

        if self.packing_controller is not None:
            kwargs = self.packing_controller(kwargs)

        translation_unit_info = self.translation_unit_info(self.arg_to_dtype(kwargs))

        return translation_unit_info.invoker(
                translation_unit_info.cl_kernels, queue, allocator, wait_for,
                out_host, **kwargs)

# }}}

# vim: foldmethod=marker

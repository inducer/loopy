from __future__ import division, with_statement, absolute_import

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

from six.moves import range, zip

from pytools import memoize_method
from pytools.py_codegen import Indentation
from loopy.target.execution import (
    KernelExecutorBase, ExecutionWrapperGeneratorBase, _KernelInfo, _Kernels)
import logging
logger = logging.getLogger(__name__)


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
        super(PyOpenCLExecutionWrapperGenerator, self).__init__(system_args)

    def python_dtype_str(self, dtype):
        import pyopencl.tools as cl_tools
        if dtype.isbuiltin:
            return "_lpy_np."+dtype.name
        else:
            return ("_lpy_cl_tools.get_or_register_dtype(\"%s\")"
                    % cl_tools.dtype_to_ctype(dtype))

    # {{{ handle non-numpy args

    def handle_non_numpy_arg(self, gen, arg):
        gen("if isinstance(%s, _lpy_np.ndarray):" % arg.name)
        with Indentation(gen):
            gen("# retain originally passed array")
            gen("_lpy_%s_np_input = %s" % (arg.name, arg.name))
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

    def handle_alloc(self, gen, arg, kernel_arg, strify, skip_arg_checks):
        """
        Handle allocation of non-specified arguments for pyopencl execution
        """
        from pymbolic import var

        num_axes = len(arg.strides)
        for i in range(num_axes):
            gen("_lpy_shape_%d = %s" % (i, strify(arg.unvec_shape[i])))

        itemsize = kernel_arg.dtype.numpy_dtype.itemsize
        for i in range(num_axes):
            gen("_lpy_strides_%d = %s" % (i, strify(
                itemsize*arg.unvec_strides[i])))

        if not skip_arg_checks:
            for i in range(num_axes):
                gen("assert _lpy_strides_%d > 0, "
                        "\"'%s' has negative stride in axis %d\""
                        % (i, arg.name, i))

        sym_strides = tuple(
                var("_lpy_strides_%d" % i)
                for i in range(num_axes))
        sym_shape = tuple(
                var("_lpy_shape_%d" % i)
                for i in range(num_axes))

        alloc_size_expr = (sum(astrd*(alen-1)
            for alen, astrd in zip(sym_shape, sym_strides))
            + itemsize)

        gen("_lpy_alloc_size = %s" % strify(alloc_size_expr))
        gen("%(name)s = _lpy_cl_array.Array(queue, %(shape)s, "
                "%(dtype)s, strides=%(strides)s, "
                "data=allocator(_lpy_alloc_size), allocator=allocator)"
                % dict(
                    name=arg.name,
                    shape=strify(sym_shape),
                    strides=strify(sym_strides),
                    dtype=self.python_dtype_str(kernel_arg.dtype.numpy_dtype)))

        if not skip_arg_checks:
            for i in range(num_axes):
                gen("del _lpy_shape_%d" % i)
                gen("del _lpy_strides_%d" % i)
            gen("del _lpy_alloc_size")
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

    def initialize_system_args(self, gen):
        """
        Initializes possibly empty system arguments
        """
        gen("if allocator is None:")
        with Indentation(gen):
            gen("allocator = _lpy_cl_tools.DeferredAllocator(queue.context)")
        gen("")

    # {{{ generate invocation

    def generate_invocation(self, gen, kernel_name, args,
            kernel, implemented_data_info):
        if kernel.options.cl_exec_manage_array_events:
            gen("""
                if wait_for is None:
                    wait_for = []
                """)

            gen("")
            from loopy.kernel.data import ArrayArg
            for arg in implemented_data_info:
                if issubclass(arg.arg_class, ArrayArg):
                    gen(
                            "wait_for.extend({arg_name}.events)"
                            .format(arg_name=arg.name))

            gen("")

        gen("_lpy_evt = {kernel_name}({args})"
        .format(
            kernel_name=kernel_name,
            args=", ".join(
                ["_lpy_cl_kernels", "queue"]
                + args
                + ["wait_for=wait_for", "allocator=allocator"])))

        if kernel.options.cl_exec_manage_array_events:
            gen("")
            from loopy.kernel.data import ArrayArg
            for arg in implemented_data_info:
                if (issubclass(arg.arg_class, ArrayArg)
                        and arg.base_name in kernel.get_written_variables()):
                    gen("{arg_name}.add_event(_lpy_evt)".format(arg_name=arg.name))

    # }}}

    # {{{

    def generate_output_handler(
            self, gen, options, kernel, implemented_data_info):

        from loopy.kernel.data import KernelArgument

        if not options.no_numpy:
            gen("if out_host is None and (_lpy_encountered_numpy "
                    "and not _lpy_encountered_dev):")
            with Indentation(gen):
                gen("out_host = True")

            for arg in implemented_data_info:
                if not issubclass(arg.arg_class, KernelArgument):
                    continue

                is_written = arg.base_name in kernel.get_written_variables()
                if is_written:
                    np_name = "_lpy_%s_np_input" % arg.name
                    gen("if out_host or %s is not None:" % np_name)
                    with Indentation(gen):
                        gen("%s = %s.get(queue=queue, ary=%s)"
                            % (arg.name, arg.name, np_name))

            gen("")

        if options.return_dict:
            gen("return _lpy_evt, {%s}"
                    % ", ".join("\"%s\": %s" % (arg.name, arg.name)
                        for arg in implemented_data_info
                        if issubclass(arg.arg_class, KernelArgument)
                        if arg.base_name in kernel.get_written_variables()))
        else:
            out_args = [arg
                    for arg in implemented_data_info
                        if issubclass(arg.arg_class, KernelArgument)
                    if arg.base_name in kernel.get_written_variables()]
            if out_args:
                gen("return _lpy_evt, (%s,)"
                        % ", ".join(arg.name for arg in out_args))
            else:
                gen("return _lpy_evt, ()")

    # }}}

    def generate_host_code(self, gen, codegen_result):
        gen.add_to_preamble(codegen_result.host_code())

    def get_arg_pass(self, arg):
        return "%s.base_data" % arg.name

# }}}


# {{{ kernel executor


class PyOpenCLKernelExecutor(KernelExecutorBase):
    """An object connecting a kernel to a :class:`pyopencl.Context`
    for execution.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, context, kernel):
        """
        :arg context: a :class:`pyopencl.Context`
        :arg kernel: may be a loopy.LoopKernel, a generator returning kernels
            (a warning will be issued if more than one is returned). If the
            kernel has not yet been loop-scheduled, that is done, too, with no
            specific arguments.
        """

        super(PyOpenCLKernelExecutor, self).__init__(kernel)

        self.context = context

        from loopy.target.pyopencl import PyOpenCLTarget
        if isinstance(kernel.target, PyOpenCLTarget):
            self.kernel = kernel.copy(target=(
                kernel.target.with_device(context.devices[0])))

    def get_invoker_uncached(self, kernel, codegen_result):
        generator = PyOpenCLExecutionWrapperGenerator()
        return generator(kernel, codegen_result)

    @memoize_method
    def kernel_info(self, arg_to_dtype_set=frozenset(), all_kwargs=None):
        kernel = self.get_typed_and_scheduled_kernel(arg_to_dtype_set)

        from loopy.codegen import generate_code_v2
        from loopy.target.execution import get_highlighted_code
        codegen_result = generate_code_v2(kernel)

        dev_code = codegen_result.device_code()

        if self.kernel.options.write_cl:
            output = dev_code
            if self.kernel.options.highlight_cl:
                output = get_highlighted_code(output)

            if self.kernel.options.write_cl is True:
                print(output)
            else:
                with open(self.kernel.options.write_cl, "w") as outf:
                    outf.write(output)

        if self.kernel.options.edit_cl:
            from pytools import invoke_editor
            dev_code = invoke_editor(dev_code, "code.cl")

        import pyopencl as cl

        cl_program = (
                cl.Program(self.context, dev_code)
                .build(options=kernel.options.cl_build_options))

        cl_kernels = _Kernels()
        for dp in codegen_result.device_programs:
            setattr(cl_kernels, dp.name, getattr(cl_program, dp.name))

        return _KernelInfo(
                kernel=kernel,
                cl_kernels=cl_kernels,
                implemented_data_info=codegen_result.implemented_data_info,
                invoker=self.get_invoker(kernel, codegen_result))

    def __call__(self, queue, **kwargs):
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

        allocator = kwargs.pop("allocator", None)
        wait_for = kwargs.pop("wait_for", None)
        out_host = kwargs.pop("out_host", None)

        kwargs = self.packing_controller.unpack(kwargs)

        kernel_info = self.kernel_info(self.arg_to_dtype_set(kwargs))

        return kernel_info.invoker(
                kernel_info.cl_kernels, queue, allocator, wait_for,
                out_host, **kwargs)

# }}}

# vim: foldmethod=marker

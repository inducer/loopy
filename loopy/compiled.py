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

import six
from six.moves import range, zip

import numpy as np
from pytools import Record, memoize_method
from loopy.diagnostic import ParameterFinderWarning
from pytools.py_codegen import (
        Indentation, PythonFunctionGenerator)
from loopy.diagnostic import LoopyError
from loopy.types import NumpyType

import logging
logger = logging.getLogger(__name__)


# {{{ object array argument packing

class _PackingInfo(Record):
    """
    .. attribute:: name
    .. attribute:: sep_shape

    .. attribute:: subscripts_and_names

        A list of type ``[(index, unpacked_name), ...]``.
    """


class SeparateArrayPackingController(object):
    """For argument arrays with axes tagged to be implemented as separate
    arrays, this class provides preprocessing of the incoming arguments so that
    all sub-arrays may be passed in one object array (under the original,
    un-split argument name) and are unpacked into separate arrays before being
    passed to the kernel.

    It also repacks outgoing arrays of this type back into an object array.
    """

    def __init__(self, kernel):
        # map from arg name
        self.packing_info = {}

        from loopy.kernel.array import ArrayBase
        for arg in kernel.args:
            if not isinstance(arg, ArrayBase):
                continue

            if arg.shape is None or arg.dim_tags is None:
                continue

            subscripts_and_names = arg.subscripts_and_names()

            if subscripts_and_names is None:
                continue

            self.packing_info[arg.name] = _PackingInfo(
                    name=arg.name,
                    sep_shape=arg.sep_shape(),
                    subscripts_and_names=subscripts_and_names,
                    is_written=arg.name in kernel.get_written_variables())

    def unpack(self, kernel_kwargs):
        if not self.packing_info:
            return kernel_kwargs

        kernel_kwargs = kernel_kwargs.copy()

        for packing_info in six.itervalues(self.packing_info):
            arg_name = packing_info.name
            if packing_info.name in kernel_kwargs:
                arg = kernel_kwargs[arg_name]
                for index, unpacked_name in packing_info.subscripts_and_names:
                    assert unpacked_name not in kernel_kwargs
                    kernel_kwargs[unpacked_name] = arg[index]
                del kernel_kwargs[arg_name]

        return kernel_kwargs

    def pack(self, outputs):
        if not self.packing_info:
            return outputs

        for packing_info in six.itervalues(self.packing_info):
            if not packing_info.is_written:
                continue

            result = outputs[packing_info.name] = \
                    np.zeros(packing_info.sep_shape, dtype=np.object)

            for index, unpacked_name in packing_info.subscripts_and_names:
                result[index] = outputs.pop(unpacked_name)

        return outputs

# }}}


# {{{ invoker generation

# /!\ This code runs in a namespace controlled by the user.
# Prefix all auxiliary variables with "_lpy".


def python_dtype_str(dtype):
    import pyopencl.tools as cl_tools
    if dtype.isbuiltin:
        return "_lpy_np."+dtype.name
    else:
        return ("_lpy_cl_tools.get_or_register_dtype(\"%s\")"
                % cl_tools.dtype_to_ctype(dtype))


# {{{ integer arg finding from shapes

def generate_integer_arg_finding_from_shapes(gen, kernel, implemented_data_info):
    # a mapping from integer argument names to a list of tuples
    # (arg_name, expression), where expression is a
    # unary function of kernel.arg_dict[arg_name]
    # returning the desired integer argument.
    iarg_to_sources = {}

    from loopy.kernel.data import GlobalArg
    from loopy.symbolic import DependencyMapper, StringifyMapper
    dep_map = DependencyMapper()

    from pymbolic import var
    for arg in implemented_data_info:
        if arg.arg_class is GlobalArg:
            sym_shape = var(arg.name).attr("shape")
            for axis_nr, shape_i in enumerate(arg.shape):
                if shape_i is None:
                    continue

                deps = dep_map(shape_i)

                if len(deps) == 1:
                    integer_arg_var, = deps

                    if kernel.arg_dict[integer_arg_var.name].dtype.is_integral():
                        from pymbolic.algorithm import solve_affine_equations_for
                        try:
                            # friggin' overkill :)
                            iarg_expr = solve_affine_equations_for(
                                    [integer_arg_var.name],
                                    [(shape_i, sym_shape.index(axis_nr))]
                                    )[integer_arg_var]
                        except Exception as e:
                            #from traceback import print_exc
                            #print_exc()

                            # went wrong? oh well
                            from warnings import warn
                            warn("Unable to generate code to automatically "
                                    "find '%s' from the shape of '%s':\n%s"
                                    % (integer_arg_var.name, arg.name, str(e)),
                                    ParameterFinderWarning)
                        else:
                            iarg_to_sources.setdefault(integer_arg_var.name, []) \
                                    .append((arg.name, iarg_expr))

    gen("# {{{ find integer arguments from shapes")
    gen("")

    for iarg_name, sources in six.iteritems(iarg_to_sources):
        gen("if %s is None:" % iarg_name)
        with Indentation(gen):
            if_stmt = "if"
            for arg_name, value_expr in sources:
                gen("%s %s is not None:" % (if_stmt, arg_name))
                with Indentation(gen):
                    gen("%s = %s"
                            % (iarg_name, StringifyMapper()(value_expr)))

                if_stmt = "elif"

        gen("")

    gen("# }}}")
    gen("")

# }}}


# {{{ integer arg finding from offsets

def generate_integer_arg_finding_from_offsets(gen, kernel, implemented_data_info):
    options = kernel.options

    gen("# {{{ find integer arguments from offsets")
    gen("")

    for arg in implemented_data_info:
        impl_array_name = arg.offset_for_name
        if impl_array_name is not None:
            gen("if %s is None:" % arg.name)
            with Indentation(gen):
                gen("if %s is None:" % impl_array_name)
                with Indentation(gen):
                    gen("# Output variable, we'll be allocating "
                            "it, with zero offset.")
                    gen("%s = 0" % arg.name)
                gen("else:")
                with Indentation(gen):
                    if not options.no_numpy:
                        gen("_lpy_offset = getattr(%s, \"offset\", 0)"
                                % impl_array_name)
                    else:
                        gen("_lpy_offset = %s.offset" % impl_array_name)

                    base_arg = kernel.impl_arg_to_arg[impl_array_name]

                    if not options.skip_arg_checks:
                        gen("%s, _lpy_remdr = divmod(_lpy_offset, %d)"
                                % (arg.name, base_arg.dtype.itemsize))

                        gen("assert _lpy_remdr == 0, \"Offset of array '%s' is "
                                "not divisible by its dtype itemsize\""
                                % impl_array_name)
                        gen("del _lpy_remdr")
                    else:
                        gen("%s = _lpy_offset // %d"
                                % (arg.name, base_arg.dtype.itemsize))

                    if not options.skip_arg_checks:
                        gen("del _lpy_offset")

    gen("# }}}")
    gen("")

# }}}


# {{{ integer arg finding from strides

def generate_integer_arg_finding_from_strides(gen, kernel, implemented_data_info):
    options = kernel.options

    gen("# {{{ find integer arguments from strides")
    gen("")

    for arg in implemented_data_info:
        if arg.stride_for_name_and_axis is not None:
            impl_array_name, stride_impl_axis = arg.stride_for_name_and_axis

            gen("if %s is None:" % arg.name)
            with Indentation(gen):
                if not options.skip_arg_checks:
                    gen("if %s is None:" % impl_array_name)
                    with Indentation(gen):
                        gen("raise RuntimeError(\"required stride '%s' for "
                                "argument '%s' not given or deducible from "
                                "passed array\")"
                                % (arg.name, impl_array_name))

                    base_arg = kernel.impl_arg_to_arg[impl_array_name]

                    if not options.skip_arg_checks:
                        gen("%s, _lpy_remdr = divmod(%s.strides[%d], %d)"
                                % (arg.name, impl_array_name, stride_impl_axis,
                                    base_arg.dtype.dtype.itemsize))

                        gen("assert _lpy_remdr == 0, \"Stride %d of array '%s' is "
                                "not divisible by its dtype itemsize\""
                                % (stride_impl_axis, impl_array_name))
                        gen("del _lpy_remdr")
                    else:
                        gen("%s = _lpy_offset // %d"
                                % (arg.name, base_arg.dtype.itemsize))

    gen("# }}}")
    gen("")

# }}}


# {{{ check that value args are present

def generate_value_arg_check(gen, kernel, implemented_data_info):
    if kernel.options.skip_arg_checks:
        return

    from loopy.kernel.data import ValueArg

    gen("# {{{ check that value args are present")
    gen("")

    for arg in implemented_data_info:
        if not issubclass(arg.arg_class, ValueArg):
            continue

        gen("if %s is None:" % arg.name)
        with Indentation(gen):
            gen("raise TypeError(\"value argument '%s' "
                    "was not given and could not be automatically "
                    "determined\")" % arg.name)

    gen("# }}}")
    gen("")

# }}}


# {{{ arg setup

def generate_arg_setup(gen, kernel, implemented_data_info, options):
    import loopy as lp

    from loopy.kernel.data import KernelArgument
    from loopy.kernel.array import ArrayBase
    from loopy.symbolic import StringifyMapper
    from pymbolic import var

    gen("# {{{ set up array arguments")
    gen("")

    if not options.no_numpy:
        gen("_lpy_encountered_numpy = False")
        gen("_lpy_encountered_dev = False")
        gen("")

    args = []

    strify = StringifyMapper()

    expect_no_more_arguments = False

    for arg_idx, arg in enumerate(implemented_data_info):
        is_written = arg.base_name in kernel.get_written_variables()
        kernel_arg = kernel.impl_arg_to_arg.get(arg.name)

        if not issubclass(arg.arg_class, KernelArgument):
            expect_no_more_arguments = True
            continue

        if expect_no_more_arguments:
            raise LoopyError("Further arguments encountered after arg info "
                    "describing a global temporary variable")

        if not issubclass(arg.arg_class, ArrayBase):
            args.append(arg.name)
            continue

        gen("# {{{ process %s" % arg.name)
        gen("")

        if not options.no_numpy:
            gen("if isinstance(%s, _lpy_np.ndarray):" % arg.name)
            with Indentation(gen):
                gen("# synchronous, nothing to worry about")
                gen("%s = _lpy_cl_array.to_device("
                        "queue, %s, allocator=allocator)"
                        % (arg.name, arg.name))
                gen("_lpy_encountered_numpy = True")
            gen("elif %s is not None:" % arg.name)
            with Indentation(gen):
                gen("_lpy_encountered_dev = True")

            gen("")

        if not options.skip_arg_checks and not is_written:
            gen("if %s is None:" % arg.name)
            with Indentation(gen):
                gen("raise RuntimeError(\"input argument '%s' must "
                        "be supplied\")" % arg.name)
                gen("")

        if (is_written
                and arg.arg_class is lp.ImageArg
                and not options.skip_arg_checks):
            gen("if %s is None:" % arg.name)
            with Indentation(gen):
                gen("raise RuntimeError(\"written image '%s' must "
                        "be supplied\")" % arg.name)
                gen("")

        if is_written and arg.shape is None and not options.skip_arg_checks:
            gen("if %s is None:" % arg.name)
            with Indentation(gen):
                gen("raise RuntimeError(\"written argument '%s' has "
                        "unknown shape and must be supplied\")" % arg.name)
                gen("")

        possibly_made_by_loopy = False

        # {{{ allocate written arrays, if needed

        if is_written and arg.arg_class in [lp.GlobalArg, lp.ConstantArg] \
                and arg.shape is not None:

            if not isinstance(arg.dtype, NumpyType):
                raise LoopyError("do not know how to pass arg of type '%s'"
                        % arg.dtype)

            possibly_made_by_loopy = True
            gen("_lpy_made_by_loopy = False")
            gen("")

            gen("if %s is None:" % arg.name)
            with Indentation(gen):
                num_axes = len(arg.strides)
                for i in range(num_axes):
                    gen("_lpy_shape_%d = %s" % (i, strify(arg.unvec_shape[i])))

                itemsize = kernel_arg.dtype.numpy_dtype.itemsize
                for i in range(num_axes):
                    gen("_lpy_strides_%d = %s" % (i, strify(
                        itemsize*arg.unvec_strides[i])))

                if not options.skip_arg_checks:
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
                            dtype=python_dtype_str(kernel_arg.dtype.numpy_dtype)))

                if not options.skip_arg_checks:
                    for i in range(num_axes):
                        gen("del _lpy_shape_%d" % i)
                        gen("del _lpy_strides_%d" % i)
                    gen("del _lpy_alloc_size")
                    gen("")

                gen("_lpy_made_by_loopy = True")
                gen("")

        # }}}

        # {{{ argument checking

        if arg.arg_class in [lp.GlobalArg, lp.ConstantArg] \
                and not options.skip_arg_checks:
            if possibly_made_by_loopy:
                gen("if not _lpy_made_by_loopy:")
            else:
                gen("if True:")

            with Indentation(gen):
                gen("if %s.dtype != %s:"
                        % (arg.name, python_dtype_str(kernel_arg.dtype.numpy_dtype)))
                with Indentation(gen):
                    gen("raise TypeError(\"dtype mismatch on argument '%s' "
                            "(got: %%s, expected: %s)\" %% %s.dtype)"
                            % (arg.name, arg.dtype, arg.name))

                # {{{ generate shape checking code

                def strify_allowing_none(shape_axis):
                    if shape_axis is None:
                        return "None"
                    else:
                        return strify(shape_axis)

                def strify_tuple(t):
                    if len(t) == 0:
                        return "()"
                    else:
                        return "(%s,)" % ", ".join(
                                strify_allowing_none(sa)
                                for sa in t)

                shape_mismatch_msg = (
                        "raise TypeError(\"shape mismatch on argument '%s' "
                        "(got: %%s, expected: %%s)\" "
                        "%% (%s.shape, %s))"
                        % (arg.name, arg.name, strify_tuple(arg.unvec_shape)))

                if kernel_arg.shape is None:
                    pass

                elif any(shape_axis is None for shape_axis in kernel_arg.shape):
                    gen("if len(%s.shape) != %s:"
                            % (arg.name, len(arg.unvec_shape)))
                    with Indentation(gen):
                        gen(shape_mismatch_msg)

                    for i, shape_axis in enumerate(arg.unvec_shape):
                        if shape_axis is None:
                            continue

                        gen("if %s.shape[%d] != %s:"
                                % (arg.name, i, strify(shape_axis)))
                        with Indentation(gen):
                            gen(shape_mismatch_msg)

                else:  # not None, no Nones in tuple
                    gen("if %s.shape != %s:"
                            % (arg.name, strify(arg.unvec_shape)))
                    with Indentation(gen):
                        gen(shape_mismatch_msg)

                # }}}

                if arg.unvec_strides and kernel_arg.dim_tags:
                    itemsize = kernel_arg.dtype.numpy_dtype.itemsize
                    sym_strides = tuple(
                            itemsize*s_i for s_i in arg.unvec_strides)
                    gen("if %s.strides != %s:"
                            % (arg.name, strify(sym_strides)))
                    with Indentation(gen):
                        gen("raise TypeError(\"strides mismatch on "
                                "argument '%s' (got: %%s, expected: %%s)\" "
                                "%% (%s.strides, %s))"
                                % (arg.name, arg.name, strify(sym_strides)))

                if not arg.allows_offset:
                    gen("if %s.offset:" % arg.name)
                    with Indentation(gen):
                        gen("raise ValueError(\"Argument '%s' does not "
                                "allow arrays with offsets. Try passing "
                                "default_offset=loopy.auto to make_kernel()."
                                "\")" % arg.name)
                        gen("")

        # }}}

        if possibly_made_by_loopy and not options.skip_arg_checks:
            gen("del _lpy_made_by_loopy")
            gen("")

        if arg.arg_class in [lp.GlobalArg, lp.ConstantArg]:
            args.append("%s.base_data" % arg.name)
        else:
            args.append("%s" % arg.name)

        gen("")

        gen("# }}}")
        gen("")

    gen("# }}}")
    gen("")

    return args

# }}}


def generate_invoker(kernel, codegen_result):
    options = kernel.options
    implemented_data_info = codegen_result.implemented_data_info
    host_code = codegen_result.host_code()

    system_args = [
            "_lpy_cl_kernels", "queue", "allocator=None", "wait_for=None",
            # ignored if options.no_numpy
            "out_host=None"
            ]

    from loopy.kernel.data import KernelArgument
    gen = PythonFunctionGenerator(
            "invoke_%s_loopy_kernel" % kernel.name,
            system_args + [
                "%s=None" % idi.name
                for idi in implemented_data_info
                if issubclass(idi.arg_class, KernelArgument)
                ])

    gen.add_to_preamble("from __future__ import division")
    gen.add_to_preamble("")
    gen.add_to_preamble("import pyopencl as _lpy_cl")
    gen.add_to_preamble("import pyopencl.array as _lpy_cl_array")
    gen.add_to_preamble("import pyopencl.tools as _lpy_cl_tools")
    gen.add_to_preamble("import numpy as _lpy_np")
    gen.add_to_preamble("")
    gen.add_to_preamble(host_code)
    gen.add_to_preamble("")

    gen("if allocator is None:")
    with Indentation(gen):
        gen("allocator = _lpy_cl_tools.DeferredAllocator(queue.context)")
    gen("")

    generate_integer_arg_finding_from_shapes(gen, kernel, implemented_data_info)
    generate_integer_arg_finding_from_offsets(gen, kernel, implemented_data_info)
    generate_integer_arg_finding_from_strides(gen, kernel, implemented_data_info)
    generate_value_arg_check(gen, kernel, implemented_data_info)

    args = generate_arg_setup(gen, kernel, implemented_data_info, options)

    # {{{ generate invocation

    gen("_lpy_evt = {kernel_name}({args})"
            .format(
                kernel_name=codegen_result.host_program.name,
                args=", ".join(
                    ["_lpy_cl_kernels", "queue"]
                    + args
                    + ["wait_for=wait_for"])))

    # }}}

    # {{{ output

    if not options.no_numpy:
        gen("if out_host is None and (_lpy_encountered_numpy "
                "and not _lpy_encountered_dev):")
        with Indentation(gen):
            gen("out_host = True")

        gen("if out_host:")
        with Indentation(gen):
            gen("pass")  # if no outputs (?!)
            for arg in implemented_data_info:
                if not issubclass(arg.arg_class, KernelArgument):
                    continue

                is_written = arg.base_name in kernel.get_written_variables()
                if is_written:
                    gen("%s = %s.get(queue=queue)" % (arg.name, arg.name))

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

    if options.write_wrapper:
        output = gen.get()
        if options.highlight_wrapper:
            output = get_highlighted_python_code(output)

        if options.write_wrapper is True:
            print(output)
        else:
            with open(options.write_wrapper, "w") as outf:
                outf.write(output)

    return gen.get_function()


# }}}


# {{{ compiled kernel object

class _CLKernelInfo(Record):
    pass


class _CLKernels(object):
    pass


class CompiledKernel:
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

        self.context = context

        from loopy.target.pyopencl import PyOpenCLTarget
        self.kernel = kernel.copy(target=PyOpenCLTarget(context.devices[0]))

        self.packing_controller = SeparateArrayPackingController(kernel)

        self.output_names = tuple(arg.name for arg in self.kernel.args
                if arg.name in self.kernel.get_written_variables())

    @memoize_method
    def get_typed_and_scheduled_kernel(self, var_to_dtype_set):
        kernel = self.kernel

        from loopy.kernel.tools import add_dtypes

        if var_to_dtype_set:
            var_to_dtype = {}
            for var, dtype in var_to_dtype_set:
                try:
                    dest_name = kernel.impl_arg_to_arg[var].name
                except KeyError:
                    dest_name = var

                try:
                    var_to_dtype[dest_name] = dtype
                except KeyError:
                    raise LoopyError("cannot set type for '%s': "
                            "no known variable/argument with that name"
                            % var)

            kernel = add_dtypes(kernel, var_to_dtype)

            from loopy.preprocess import infer_unknown_types
            kernel = infer_unknown_types(kernel, expect_completion=True)

        if kernel.schedule is None:
            from loopy.preprocess import preprocess_kernel
            kernel = preprocess_kernel(kernel)

            from loopy.schedule import get_one_scheduled_kernel
            kernel = get_one_scheduled_kernel(kernel)

        return kernel

    @memoize_method
    def cl_kernel_info(self, arg_to_dtype_set=frozenset(), all_kwargs=None):
        kernel = self.get_typed_and_scheduled_kernel(arg_to_dtype_set)

        from loopy.codegen import generate_code_v2
        codegen_result = generate_code_v2(kernel)

        dev_code = codegen_result.device_code()

        if self.kernel.options.write_cl:
            output = dev_code
            if self.kernel.options.highlight_cl:
                output = get_highlighted_cl_code(output)

            if self.kernel.options.write_cl is True:
                print(output)
            else:
                with open(self.kernel.options.write_cl, "w") as outf:
                    outf.write(output)

        if self.kernel.options.edit_cl:
            from pytools import invoke_editor
            dev_code = invoke_editor(dev_code, "code.cl")

        import pyopencl as cl

        logger.info("%s: opencl compilation start" % self.kernel.name)

        cl_program = (
                cl.Program(self.context, dev_code)
                .build(options=kernel.options.cl_build_options))

        cl_kernels = _CLKernels()
        for dp in codegen_result.device_programs:
            setattr(cl_kernels, dp.name, getattr(cl_program, dp.name))

        logger.info("%s: opencl compilation done" % self.kernel.name)

        return _CLKernelInfo(
                kernel=kernel,
                cl_kernels=cl_kernels,
                implemented_data_info=codegen_result.implemented_data_info,
                invoker=generate_invoker(kernel, codegen_result))

    # {{{ debugging aids

    def get_code(self, arg_to_dtype=None):
        if arg_to_dtype is not None:
            arg_to_dtype = frozenset(six.iteritems(arg_to_dtype))

        kernel = self.get_typed_and_scheduled_kernel(arg_to_dtype)

        from loopy.codegen import generate_code_v2
        code = generate_code_v2(kernel)
        return code.device_code()

    def get_highlighted_code(self, arg_to_dtype=None):
        return get_highlighted_cl_code(
                self.get_code(arg_to_dtype))

    @property
    def code(self):
        from warnings import warn
        warn("CompiledKernel.code is deprecated. Use .get_code() instead.",
                DeprecationWarning, stacklevel=2)

        return self.get_code()

    # }}}

    def __call__(self, queue, **kwargs):
        """
        :arg allocator: a callable passed a byte count and returning
            a :class:`pyopencl.Buffer`. A :class:`pyopencl` allocator
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

        impl_arg_to_arg = self.kernel.impl_arg_to_arg
        arg_to_dtype = {}
        for arg_name, val in six.iteritems(kwargs):
            arg = impl_arg_to_arg.get(arg_name, None)

            if arg is None:
                # offsets, strides and such
                continue

            if arg.dtype is None and val is not None:
                try:
                    dtype = val.dtype
                except AttributeError:
                    pass
                else:
                    arg_to_dtype[arg_name] = dtype

        kernel_info = self.cl_kernel_info(
                frozenset(six.iteritems(arg_to_dtype)))

        return kernel_info.invoker(
                kernel_info.cl_kernels, queue, allocator, wait_for,
                out_host, **kwargs)

# }}}


def get_highlighted_python_code(text):
    try:
        from pygments import highlight
    except ImportError:
        return text
    else:
        from pygments.lexers import PythonLexer
        from pygments.formatters import TerminalFormatter

        return highlight(text, PythonLexer(), TerminalFormatter())


def get_highlighted_cl_code(text):
    try:
        from pygments import highlight
    except ImportError:
        return text
    else:
        from pygments.lexers import CLexer
        from pygments.formatters import TerminalFormatter

        return highlight(text, CLexer(), TerminalFormatter())


# vim: foldmethod=marker

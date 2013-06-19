from __future__ import division, with_statement

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


import pyopencl as cl
import pyopencl.tools  # noqa
import numpy as np
from pytools import Record, memoize_method
from loopy.diagnostic import ParameterFinderWarning


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

        for packing_info in self.packing_info.itervalues():
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

        for packing_info in self.packing_info.itervalues():
            if not packing_info.is_written:
                continue

            result = outputs[packing_info.name] = \
                    np.zeros(packing_info.sep_shape, dtype=np.object)

            for index, unpacked_name in packing_info.subscripts_and_names:
                result[index] = outputs.pop(unpacked_name)

        return outputs

# }}}


# {{{ python code generation helpers

class Indentation(object):
    def __init__(self, generator):
        self.generator = generator

    def __enter__(self):
        self.generator.indent()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.generator.dedent()


class PythonCodeGenerator(object):
    def __init__(self):
        self.preamble = []
        self.code = []
        self.level = 0

    def extend(self, sub_generator):
        self.code.extend(sub_generator.code)

    def extend_indent(self, sub_generator):
        with Indentation(self):
            for line in sub_generator.code:
                self.write(line)

    def get(self):
        result = "\n".join(self.code)
        if self.preamble:
            result = "\n".join(self.preamble) + "\n" + result
        return result

    def add_to_preamble(self, s):
        self.preamble.append(s)

    def __call__(self, string):
        self.code.append(" "*(4*self.level) + string)

    def indent(self):
        self.level += 1

    def dedent(self):
        if self.level == 0:
            raise RuntimeError("internal error in python code generator")
        self.level -= 1


class PythonFunctionGenerator(PythonCodeGenerator):
    def __init__(self, name, args):
        PythonCodeGenerator.__init__(self)
        self.name = name

        self("def %s(%s):" % (name, ", ".join(args)))
        self.indent()

    def get_function(self):
        result_dict = {}
        exec(compile(self.get(), "<generated function %s>" % self.name, "exec"),
                result_dict)
        return result_dict[self.name]


# }}}


# {{{ invoker generation

# /!\ This code runs in a namespace controlled by the user.
# Prefix all auxiliary variables with "_lpy".


def python_dtype_str(dtype):
    if dtype.isbuiltin:
        return "_lpy_np."+dtype.name
    else:
        return ("_lpy_cl_tools.get_or_register_dtype(\"%s\")"
                % cl.tools.dtype_to_ctype(dtype))


# {{{ integer arg finding from shapes

def generate_integer_arg_finding_from_shapes(gen, kernel, impl_arg_info, flags):
    # a mapping from integer argument names to a list of tuples
    # (arg_name, expression), where expression is a
    # unary function of kernel.arg_dict[arg_name]
    # returning the desired integer argument.
    iarg_to_sources = {}

    from loopy.kernel.data import GlobalArg
    from loopy.symbolic import DependencyMapper, StringifyMapper
    dep_map = DependencyMapper()

    from pymbolic import var
    for arg in impl_arg_info:
        if arg.arg_class is GlobalArg:
            sym_shape = var(arg.name).attr("shape")
            for axis_nr, shape_i in enumerate(arg.shape):
                deps = dep_map(shape_i)

                if len(deps) == 1:
                    integer_arg_var, = deps

                    if kernel.arg_dict[integer_arg_var.name].dtype.kind == "i":
                        from pymbolic.algorithm import solve_affine_equations_for
                        try:
                            # friggin' overkill :)
                            iarg_expr = solve_affine_equations_for(
                                    [integer_arg_var.name],
                                    [(shape_i, sym_shape[axis_nr])]
                                    )[integer_arg_var]
                        except Exception, e:
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

    for iarg_name, sources in iarg_to_sources.iteritems():
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

def generate_integer_arg_finding_from_offsets(gen, kernel, impl_arg_info, flags):
    gen("# {{{ find integer arguments from offsets")
    gen("")

    for arg in impl_arg_info:
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
                    if flags.allow_numpy:
                        gen("_lpy_offset = getattr(%s, \"offset\", 0)"
                                % impl_array_name)
                    else:
                        gen("_lpy_offset = %s.offset" % impl_array_name)

                    base_arg = kernel.impl_arg_to_arg[impl_array_name]

                    if flags.paranoid:
                        gen("%s, _lpy_remdr = divmod(_lpy_offset, %d)"
                                % (arg.name, base_arg.dtype.itemsize))

                        gen("assert _lpy_remdr == 0, \"Offset of array '%s' is "
                                "not divisible by its dtype itemsize\""
                                % impl_array_name)
                        gen("del _lpy_remdr")
                    else:
                        gen("%s = _lpy_offset // %d)"
                                % (arg.name, base_arg.dtype.itemsize))

                    if flags.paranoid:
                        gen("del _lpy_offset")

    gen("# }}}")
    gen("")

# }}}


# {{{ integer arg finding from offsets

def generate_integer_arg_finding_from_strides(gen, kernel, impl_arg_info, flags):
    gen("# {{{ find integer arguments from strides")
    gen("")

    for arg in impl_arg_info:
        if arg.stride_for_name_and_axis is not None:
            impl_array_name, stride_impl_axis = arg.stride_for_name_and_axis

            gen("if %s is None:" % arg.name)
            with Indentation(gen):
                if flags.paranoid:
                    gen("if %s is None:" % impl_array_name)
                    with Indentation(gen):
                        gen("raise RuntimeError(\"required stride '%s' for "
                                "argument '%s' not given or deducible from "
                                "passed array\")"
                                % (arg.name, impl_array_name))

                    base_arg = kernel.impl_arg_to_arg[impl_array_name]

                    if flags.paranoid:
                        gen("%s, _lpy_remdr = divmod(%s.strides[%d], %d)"
                                % (arg.name, impl_array_name, stride_impl_axis,
                                    base_arg.dtype.itemsize))

                        gen("assert _lpy_remdr == 0, \"Stride %d of array '%s' is "
                                "not divisible by its dtype itemsize\""
                                % (stride_impl_axis, impl_array_name))
                        gen("del _lpy_remdr")
                    else:
                        gen("%s = divmod(%s.strides[%d], %d)"
                                % (arg.name, impl_array_name, stride_impl_axis,
                                    base_arg.dtype.itemsize))
                        gen("%s = _lpy_offset // %d)"
                                % (arg.name, base_arg.dtype.itemsize))

    gen("# }}}")
    gen("")

# }}}


# {{{ value arg setup

def generate_value_arg_setup(gen, kernel, impl_arg_info, flags):
    import loopy as lp
    from loopy.kernel.array import ArrayBase

    for arg_idx, arg in enumerate(impl_arg_info):
        if arg.arg_class is not lp.ValueArg:
            assert issubclass(arg.arg_class, ArrayBase)
            continue

        gen("# {{{ process %s" % arg.name)
        gen("")

        if flags.paranoid:
            gen("if %s is None:" % arg.name)
            with Indentation(gen):
                gen("raise RuntimeError(\"input argument '%s' must "
                        "be supplied\")" % arg.name)
                gen("")

        if arg.dtype.kind == "i":
            gen("# cast to int to avoid numpy scalar trouble with Boost.Python")
            gen("%s = int(%s)" % (arg.name, arg.name))
            gen("")

        if arg.dtype.char == "V":
            gen("cl_kernel.set_arg(%d, %s)" % (arg_idx, arg.name))
        else:
            gen("cl_kernel.set_arg(%d, _lpy_pack(\"%s\", %s))"
                    % (arg_idx, arg.dtype.char, arg.name))
        gen("")

        gen("# }}}")
        gen("")

# }}}


# {{{ array arg setup

def generate_array_arg_setup(gen, kernel, impl_arg_info, flags):
    import loopy as lp

    from loopy.kernel.array import ArrayBase
    from loopy.symbolic import StringifyMapper
    from pymbolic import var

    gen("# {{{ set up array arguments")
    gen("")

    if flags.allow_numpy:
        gen("_lpy_encountered_numpy = False")
        gen("_lpy_encountered_dev = False")
        gen("")

    strify = StringifyMapper()

    for arg_idx, arg in enumerate(impl_arg_info):
        is_written = arg.base_name in kernel.get_written_variables()
        kernel_arg = kernel.impl_arg_to_arg.get(arg.name)

        gen("# {{{ process %s" % arg.name)
        gen("")

        if not issubclass(arg.arg_class, ArrayBase):
            continue

        if flags.allow_numpy:
            gen("if isinstance(%s, _lpy_np.ndarray):" % arg.name)
            with Indentation(gen):
                gen("# synchronous, nothing to worry about")
                gen("%s = _lpy_cl_array.to_device("
                        "queue, %s, allocator=allocator)"
                        % (arg.name, arg.name))
                gen("_lpy_encountered_numpy = True")
            gen("else:")
            with Indentation(gen):
                gen("_lpy_encountered_dev = True")

            gen("")

        if flags.paranoid and not is_written:
            gen("if %s is None:" % arg.name)
            with Indentation(gen):
                gen("raise RuntimeError(\"input argument '%s' must "
                        "be supplied\")" % arg.name)
                gen("")

        if is_written and arg.arg_class is lp.ImageArg and flags.paranoid:
            gen("if %s is None:" % arg.name)
            with Indentation(gen):
                gen("raise RuntimeError(\"written image '%s' must "
                        "be supplied\")" % arg.name)
                gen("")

        if is_written and arg.shape is None and flags.paranoid:
            gen("if %s is None:" % arg.name)
            with Indentation(gen):
                gen("raise RuntimeError(\"written argument '%s' has "
                        "unknown shape and must be supplied\")" % arg.name)
                gen("")

        possibly_made_by_loopy = False

        # {{{ allocate written arrays, if needed

        if is_written and arg.arg_class in [lp.GlobalArg, lp.ConstantArg] \
                and arg.shape is not None:

            possibly_made_by_loopy = True
            gen("_lpy_made_by_loopy = False")
            gen("")

            gen("if %s is None:" % arg.name)
            with Indentation(gen):
                num_axes = len(arg.strides)
                for i in xrange(num_axes):
                    gen("_lpy_shape_%d = %s" % (i, strify(arg.shape[i])))

                itemsize = kernel_arg.dtype.itemsize
                for i in xrange(num_axes):
                    gen("_lpy_strides_%d = %s" % (i, strify(
                        itemsize*arg.strides[i])))

                if flags.paranoid:
                    for i in xrange(num_axes):
                        gen("assert _lpy_strides_%d > 0, "
                                "\"'%s' has negative stride in axis %d\""
                                % (i, arg.name, i))

                sym_strides = tuple(
                        var("_lpy_strides_%d" % i)
                        for i in xrange(num_axes))
                sym_shape = tuple(
                        var("_lpy_shape_%d" % i)
                        for i in xrange(num_axes))

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
                            dtype=python_dtype_str(arg.dtype)))

                if flags.paranoid:
                    for i in xrange(num_axes):
                        gen("del _lpy_shape_%d" % i)
                        gen("del _lpy_strides_%d" % i)
                    gen("del _lpy_alloc_size")
                    gen("")

                gen("_lpy_made_by_loopy = True")
                gen("")

            # }}}

            # {{{ argument checking

            if arg.arg_class in [lp.GlobalArg, lp.ConstantArg] \
                    and flags.paranoid:
                if possibly_made_by_loopy:
                    gen("if not _lpy_made_by_loopy:")
                else:
                    gen("if True:")

                with Indentation(gen):
                    gen("if %s.dtype != %s:"
                            % (arg.name, python_dtype_str(arg.dtype)))
                    with Indentation(gen):
                        gen("raise TypeError(\"dtype mismatch on argument '%s' "
                                "(got: %%s, expected: %s)\" %% %s.dtype)"
                                % (arg.name, arg.dtype, arg.name))

                    if arg.shape is not None:
                        gen("if %s.shape != %s:"
                                % (arg.name, strify(arg.shape)))
                        with Indentation(gen):
                            gen("raise TypeError(\"shape mismatch on argument '%s' "
                                    "(got: %%s, expected: %%s)\" "
                                    "%% (%s.shape, %s))"
                                    % (arg.name, arg.name, strify(arg.shape)))

                    if arg.strides is not None:
                        itemsize = kernel_arg.dtype.itemsize
                        sym_strides = tuple(itemsize*s_i for s_i in arg.strides)
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

            if possibly_made_by_loopy and flags.paranoid:
                gen("del _lpy_made_by_loopy")
                gen("")

        if arg.arg_class in [lp.GlobalArg, lp.ConstantArg]:
            gen("cl_kernel.set_arg(%d, %s.base_data)" % (arg_idx, arg.name))
        else:
            gen("cl_kernel.set_arg(%d, %s)" % (arg_idx, arg.name))
        gen("")

        gen("# }}}")
        gen("")

    gen("# }}}")
    gen("")

# }}}


class InvocationFlags(Record):
    """
    .. attribute:: paranoid
    .. attribute:: allow_numpy
    .. attribute:: return_dict
    .. attribute:: print_wrapper
    .. attribute:: print_hl_wrapper
    """

    def __init__(self, paranoid=True, allow_numpy=True, return_dict=False,
            print_wrapper=False):
        Record.__init__(self, paranoid=paranoid, allow_numpy=allow_numpy,
                return_dict=return_dict, print_wrapper=print_wrapper)


def generate_invoker(kernel, impl_arg_info, flags):
    system_args = [
            "cl_kernel", "queue", "allocator=None", "wait_for=None",
            # ignored if not flags.allow_numpy
            "out_host=None"
            ]

    gen = PythonFunctionGenerator(
            "invoke_%s_loopy_kernel" % kernel.name,
            system_args + ["%s=None" % iai.name for iai in impl_arg_info])

    gen.add_to_preamble("from __future__ import division")
    gen.add_to_preamble("")
    gen.add_to_preamble("import pyopencl as _lpy_cl")
    gen.add_to_preamble("import pyopencl.array as _lpy_cl_array")
    gen.add_to_preamble("import pyopencl.tools as _lpy_cl_tools")
    gen.add_to_preamble("import numpy as _lpy_np")
    gen.add_to_preamble("from pyopencl._pvt_struct import pack as _lpy_pack")
    gen.add_to_preamble("")

    gen("if allocator is None:")
    with Indentation(gen):
        gen("allocator = _lpy_cl_tools.DeferredAllocator(queue.context)")
    gen("")

    generate_integer_arg_finding_from_shapes(gen, kernel, impl_arg_info, flags)
    generate_integer_arg_finding_from_offsets(gen, kernel, impl_arg_info, flags)
    generate_integer_arg_finding_from_strides(gen, kernel, impl_arg_info, flags)

    generate_value_arg_setup(gen, kernel, impl_arg_info, flags)
    generate_array_arg_setup(gen, kernel, impl_arg_info, flags)

    # {{{ generate invocation

    from loopy.symbolic import StringifyMapper

    strify = StringifyMapper()
    gsize_expr, lsize_expr = kernel.get_grid_sizes_as_exprs()

    if not gsize_expr:
        gsize_expr = (1,)
    if not lsize_expr:
        lsize_expr = (1,)

    gen("_lpy_evt = _lpy_cl.enqueue_nd_range_kernel(queue, cl_kernel, "
            "%(gsize)s, %(lsize)s,  wait_for=wait_for, g_times_l=True)"
            % dict(
                gsize=strify(gsize_expr),
                lsize=strify(lsize_expr)))
    gen("")

    # }}}

    # {{{ output

    if flags.allow_numpy:
        gen("if out_host is None and (_lpy_encountered_numpy "
                "and not _lpy_encountered_dev):")
        with Indentation(gen):
            gen("out_host = True")

        gen("if out_host:")
        with Indentation(gen):
            gen("pass")  # if no outputs (?!)
            for arg_idx, arg in enumerate(impl_arg_info):
                is_written = arg.base_name in kernel.get_written_variables()
                if is_written:
                    gen("%s = %s.get(queue=queue)" % (arg.name, arg.name))

        gen("")

    if flags.return_dict:
        gen("return _lpy_evt, {%s}"
                % ", ".join("\"%s\": %s" % (arg.name, arg.name)
                    for arg in impl_arg_info
                    if arg.base_name in kernel.get_written_variables()))
    else:
        gen("return _lpy_evt, (%s,)"
                % ", ".join(arg.name
                    for arg in impl_arg_info
                    if arg.base_name in kernel.get_written_variables()))

    # }}}

    if flags.print_wrapper == "hl":
        print get_highlighted_python_code(gen.get())
    elif flags.print_wrapper:
        print gen.get()

    return gen.get_function()


# }}}


# {{{ compiled kernel object

class _CLKernelInfo(Record):
    pass


class CompiledKernel:
    def __init__(self, context, kernel, options=[], codegen_kwargs={},
            iflags=InvocationFlags()):
        """
        :arg kernel: may be a loopy.LoopKernel, a generator returning kernels
            (a warning will be issued if more than one is returned). If the
            kernel has not yet been loop-scheduled, that is done, too, with no
            specific arguments.
        :arg iflags: An :class:`InvocationFlags` instance, or a dictionary
            of arguments with which a :class:`InvocationFlags` instance
            can be initialized.
        """

        self.context = context
        self.kernel = kernel
        self.codegen_kwargs = codegen_kwargs
        self.options = options

        if not isinstance(iflags, InvocationFlags):
            iflags = InvocationFlags(**iflags)
        self.iflags = iflags

        self.packing_controller = SeparateArrayPackingController(kernel)

        self.output_names = tuple(arg.name for arg in self.kernel.args
                if arg.name in self.kernel.get_written_variables())

    @memoize_method
    def get_kernel(self, arg_to_dtype_set):
        kernel = self.kernel

        from loopy.kernel.tools import add_argument_dtypes

        if arg_to_dtype_set:
            arg_to_dtype = {}
            for arg, dtype in arg_to_dtype_set:
                arg_to_dtype[kernel.impl_arg_to_arg[arg].name] = dtype

            kernel = add_argument_dtypes(kernel, arg_to_dtype)

            from loopy.preprocess import infer_unknown_types
            kernel = infer_unknown_types(kernel, expect_completion=True)

        if kernel.schedule is None:
            from loopy.schedule import get_one_scheduled_kernel
            kernel = get_one_scheduled_kernel(kernel)

        return kernel

    @memoize_method
    def cl_kernel_info(self, arg_to_dtype_set=frozenset(), code_op=None):
        kernel = self.get_kernel(arg_to_dtype_set)

        from loopy.codegen import generate_code
        code, impl_arg_info = generate_code(kernel, **self.codegen_kwargs)

        if code_op is None:
            code_op = ""

        code_op = code_op.split(",")
        if "print" in code_op:
            print code
        elif "print_hl" in code_op:
            print get_highlighted_cl_code(code)
        elif "edit" in code_op:
            from pytools import invoke_editor
            code = invoke_editor(code, "code.cl")

        cl_program = cl.Program(self.context, code)
        cl_kernel = getattr(
                cl_program.build(options=self.options),
                kernel.name)

        return _CLKernelInfo(
                kernel=kernel,
                cl_kernel=cl_kernel,
                impl_arg_info=impl_arg_info,
                invoker=generate_invoker(
                    kernel, impl_arg_info, self.iflags))

    # {{{ debugging aids

    def get_code(self, arg_to_dtype=None):
        if arg_to_dtype is not None:
            arg_to_dtype = frozenset(arg_to_dtype.iteritems())

        kernel = self.get_kernel(arg_to_dtype)

        from loopy.codegen import generate_code
        code, arg_info = generate_code(kernel, **self.codegen_kwargs)
        return code

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
        """If all array arguments are :mod:`numpy` arrays, defaults to
        returning numpy arrays as well.

        :arg allocator:
        :arg wait_for:
        :arg out_host:
        :arg code_op:
        """

        allocator = kwargs.pop("allocator", None)
        wait_for = kwargs.pop("wait_for", None)
        out_host = kwargs.pop("out_host", None)
        code_op = kwargs.pop("code_op", None)

        kwargs = self.packing_controller.unpack(kwargs)

        impl_arg_to_arg = self.kernel.impl_arg_to_arg
        arg_to_dtype = {}
        for arg_name, val in kwargs.iteritems():
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
                frozenset(arg_to_dtype.iteritems()),
                code_op)

        return kernel_info.invoker(
                kernel_info.cl_kernel, queue, allocator, wait_for,
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

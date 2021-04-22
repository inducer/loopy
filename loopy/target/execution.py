__copyright__ = "Copyright (C) 2012-17 Andreas Kloeckner, Nick Curtis"

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
from pytools import ImmutableRecord, memoize_method
from loopy.diagnostic import LoopyError
from pytools.py_codegen import (
        Indentation, PythonFunctionGenerator)

import logging
logger = logging.getLogger(__name__)

from pytools.persistent_dict import WriteOncePersistentDict
from loopy.tools import LoopyKeyBuilder
from loopy.version import DATA_MODEL_VERSION


# {{{ object array argument packing

class _PackingInfo(ImmutableRecord):
    """
    .. attribute:: name
    .. attribute:: sep_shape

    .. attribute:: subscripts_and_names

        A list of type ``[(index, unpacked_name), ...]``.
    """


class SeparateArrayPackingController:
    """For argument arrays with axes tagged to be implemented as separate
    arrays, this class provides preprocessing of the incoming arguments so that
    all sub-arrays may be passed in one object array (under the original,
    un-split argument name) and are unpacked into separate arrays before being
    passed to the kernel.

    It also repacks outgoing arrays of this type back into an object array.
    """

    def __init__(self, program, entrypoint):

        # map from arg name
        self.packing_info = {}

        from loopy.kernel.array import ArrayBase
        for arg in program[entrypoint].args:
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
                    is_written=arg.name in
                    program[entrypoint].get_written_variables())

    def unpack(self, kernel_kwargs):
        if not self.packing_info:
            return kernel_kwargs

        kernel_kwargs = kernel_kwargs.copy()

        for packing_info in self.packing_info.values():
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

        for packing_info in self.packing_info.values():
            if not packing_info.is_written:
                continue

            result = outputs[packing_info.name] = \
                    np.zeros(packing_info.sep_shape, dtype=np.object)

            for index, unpacked_name in packing_info.subscripts_and_names:
                result[index] = outputs.pop(unpacked_name)

        return outputs

# }}}


# {{{ ExecutionWrapperGeneratorBase

class ExecutionWrapperGeneratorBase:
    """
    A set of common methods for generating a wrapper
    for execution

    """

    def __init__(self, system_args):
        self.system_args = system_args[:]

    def python_dtype_str(self, dtype):
        raise NotImplementedError()

    # {{{ invoker generation

    # /!\ This code runs in a namespace controlled by the user.
    # Prefix all auxiliary variables with "_lpy".

    # {{{ integer arg finding from shapes

    def generate_integer_arg_finding_from_shapes(
            self, gen, program, implemented_data_info):
        # a mapping from integer argument names to a list of tuples
        # (arg_name, expression), where expression is a
        # unary function of kernel.arg_dict[arg_name]
        # returning the desired integer argument.
        iarg_to_sources = {}

        from loopy.kernel.data import ArrayArg
        from loopy.symbolic import DependencyMapper, StringifyMapper
        from loopy.diagnostic import ParameterFinderWarning
        dep_map = DependencyMapper()

        from pymbolic import var
        for arg in implemented_data_info:
            if arg.arg_class is ArrayArg:
                sym_shape = var(arg.name).attr("shape")
                for axis_nr, shape_i in enumerate(arg.shape):
                    if shape_i is None:
                        continue

                    deps = dep_map(shape_i)

                    if len(deps) == 1:
                        integer_arg_var, = deps

                        if program.arg_dict[
                                integer_arg_var.name].dtype.is_integral():
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
                                iarg_to_sources.setdefault(integer_arg_var.name, [])\
                                        .append((arg.name, iarg_expr))

        gen("# {{{ find integer arguments from shapes")
        gen("")

        for iarg_name, sources in iarg_to_sources.items():
            gen("if %s is None:" % iarg_name)
            with Indentation(gen):
                if_stmt = "if"
                for arg_name, value_expr in sources:
                    gen(f"{if_stmt} {arg_name} is not None:")
                    with Indentation(gen):
                        gen("%s = %s"
                                % (iarg_name, StringifyMapper()(value_expr)))

                    if_stmt = "elif"

            gen("")

        gen("# }}}")
        gen("")

    # }}}

    # {{{ integer arg finding from offsets

    def generate_integer_arg_finding_from_offsets(self, gen, kernel,
            implemented_data_info):
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
                            gen('_lpy_offset = getattr(%s, "offset", 0)'
                                    % impl_array_name)
                        else:
                            gen("_lpy_offset = %s.offset" % impl_array_name)

                        base_arg = kernel.impl_arg_to_arg[impl_array_name]

                        if not options.skip_arg_checks:
                            gen("%s, _lpy_remdr = divmod(_lpy_offset, %d)"
                                    % (arg.name, base_arg.dtype.itemsize))

                            gen("assert _lpy_remdr == 0, \"Offset of array '%s' is "
                                    'not divisible by its dtype itemsize"'
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

    def generate_integer_arg_finding_from_strides(
            self, gen, kernel, implemented_data_info):
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
                                    'passed array")'
                                    % (arg.name, impl_array_name))

                    base_arg = kernel.impl_arg_to_arg[impl_array_name]

                    if not options.skip_arg_checks:
                        gen("%s, _lpy_remdr = divmod(%s.strides[%d], %d)"
                                % (arg.name, impl_array_name, stride_impl_axis,
                                    base_arg.dtype.dtype.itemsize))

                        gen("assert _lpy_remdr == 0, \"Stride %d of array '%s' "
                                ' is not divisible by its dtype itemsize"'
                                % (stride_impl_axis, impl_array_name))
                        gen("del _lpy_remdr")
                    else:
                        gen("%s = %s.strides[%d] // %d"
                                % (arg.name,  impl_array_name, stride_impl_axis,
                                    base_arg.dtype.itemsize))

        gen("# }}}")
        gen("")

    # }}}

    # {{{ check that value args are present

    def generate_value_arg_check(
            self, gen, kernel, implemented_data_info):
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
                        'determined")' % arg.name)

        gen("# }}}")
        gen("")

    # }}}

    # {{{ handle non numpy arguements

    def handle_non_numpy_arg(self, gen, arg):
        raise NotImplementedError()

    # }}}

    # {{{ handle allocation of unspecified arguements

    def handle_alloc(self, gen, arg, kernel_arg, strify, skip_arg_checks):
        """
        Handle allocation of non-specified arguements for C-execution
        """
        raise NotImplementedError()

    # }}}

    def get_arg_pass(self, arg):
        raise NotImplementedError()

    def get_strides_check_expr(self, shape, strides, sym_strides):
        # Returns an expression suitable for use for checking the strides of an
        # argument. Arguments should be sequences of strings.
        return " and ".join(
                "(%s == 1 or %s == %s)" % elem
                for elem in zip(shape, strides, sym_strides)) or "True"

    # {{{ arg setup

    def generate_arg_setup(
            self, gen, kernel, implemented_data_info, options):
        import loopy as lp

        from loopy.kernel.data import KernelArgument
        from loopy.kernel.array import ArrayBase
        from loopy.symbolic import StringifyMapper
        from loopy.types import NumpyType

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
                self.handle_non_numpy_arg(gen, arg)

            if not options.skip_arg_checks and not is_written:
                gen("if %s is None:" % arg.name)
                with Indentation(gen):
                    gen("raise RuntimeError(\"input argument '%s' must "
                            'be supplied")' % arg.name)
                    gen("")

            if (is_written
                    and arg.arg_class is lp.ImageArg
                    and not options.skip_arg_checks):
                gen("if %s is None:" % arg.name)
                with Indentation(gen):
                    gen("raise RuntimeError(\"written image '%s' must "
                            'be supplied")' % arg.name)
                    gen("")

            if is_written and arg.shape is None and not options.skip_arg_checks:
                gen("if %s is None:" % arg.name)
                with Indentation(gen):
                    gen("raise RuntimeError(\"written argument '%s' has "
                            'unknown shape and must be supplied")' % arg.name)
                    gen("")

            possibly_made_by_loopy = False

            # {{{ allocate written arrays, if needed

            if is_written and arg.arg_class in [lp.ArrayArg, lp.ConstantArg] \
                    and arg.shape is not None \
                    and all(si is not None for si in arg.shape):

                if not isinstance(arg.dtype, NumpyType):
                    raise LoopyError("do not know how to pass arg of type '%s'"
                            % arg.dtype)

                possibly_made_by_loopy = True
                gen("_lpy_made_by_loopy = False")
                gen("")

                gen("if %s is None:" % arg.name)
                with Indentation(gen):
                    self.handle_alloc(
                        gen, arg, kernel_arg, strify, options.skip_arg_checks)
                    gen("_lpy_made_by_loopy = True")
                    gen("")

            # }}}

            # {{{ argument checking

            if arg.arg_class in [lp.ArrayArg, lp.ConstantArg] \
                    and not options.skip_arg_checks:
                if possibly_made_by_loopy:
                    gen("if not _lpy_made_by_loopy:")
                else:
                    gen("if True:")

                with Indentation(gen):
                    gen("if %s.dtype != %s:"
                            % (arg.name, self.python_dtype_str(
                                kernel_arg.dtype.numpy_dtype)))
                    with Indentation(gen):
                        gen("raise TypeError(\"dtype mismatch on argument '%s' "
                                '(got: %%s, expected: %s)" %% %s.dtype)'
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
                            '(got: %%s, expected: %%s)" '
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

                        ndim = len(arg.unvec_shape)
                        shape = ["_lpy_shape_%d" % i for i in range(ndim)]
                        strides = ["_lpy_stride_%d" % i for i in range(ndim)]

                        gen("({},) = {}.shape".format(", ".join(shape), arg.name))
                        gen("({},) = {}.strides".format(
                            ", ".join(strides), arg.name))

                        gen("if not (%s):"
                                % self.get_strides_check_expr(
                                    shape, strides,
                                    (strify(s) for s in sym_strides)))
                        with Indentation(gen):
                            gen("_lpy_got = tuple(stride "
                                    "for (dim, stride) in zip(%s.shape, %s.strides) "
                                    "if dim > 1)"
                                    % (arg.name, arg.name))
                            gen("_lpy_expected = tuple(stride "
                                    "for (dim, stride) in zip(%s.shape, %s) "
                                    "if dim > 1)"
                                    % (arg.name, strify_tuple(sym_strides)))

                            gen('raise TypeError("strides mismatch on '
                                    "argument '%s' "
                                    "(after removing unit length dims, "
                                    'got: %%s, expected: %%s)" '
                                    "%% (_lpy_got, _lpy_expected))"
                                    % arg.name)

                    if not arg.allows_offset:
                        gen("if hasattr({}, 'offset') and {}.offset:".format(
                                arg.name, arg.name))
                        with Indentation(gen):
                            gen("raise ValueError(\"Argument '%s' does not "
                                    "allow arrays with offsets. Try passing "
                                    "default_offset=loopy.auto to make_kernel()."
                                    '")' % arg.name)
                            gen("")

            # }}}

            if possibly_made_by_loopy and not options.skip_arg_checks:
                gen("del _lpy_made_by_loopy")
                gen("")

            if arg.arg_class in [lp.ArrayArg, lp.ConstantArg]:
                args.append(self.get_arg_pass(arg))
            else:
                args.append("%s" % arg.name)

            gen("")

            gen("# }}}")
            gen("")

        gen("# }}}")
        gen("")

        return args

    # }}}

    def target_specific_preamble(self, gen):
        """
        Add target specific imports to preamble
        """
        raise NotImplementedError()

    def initialize_system_args(self, gen):
        """
        Override to intialize any default system args
        """
        raise NotImplementedError()

    # {{{ generate invocation

    def generate_invocation(self, gen, kernel_name, args,
            kernel, implemented_data_info):
        raise NotImplementedError()

    # }}}

    # {{{ output

    def generate_output_handler(
            self, gen, options, kernel, implemented_data_info):

        raise NotImplementedError()

    # }}}

    def generate_host_code(self, gen, codegen_result):
        raise NotImplementedError

    def __call__(self, program, entrypoint, codegen_result):
        """
        Generates the wrapping python invoker for this execution target

        :arg kernel: the loopy :class:`LoopKernel`(s) to be executued
        :codegen_result: the loopy :class:`CodeGenerationResult` created
        by code generation

        :returns: A python callable that handles execution of this
            kernel
        """

        options = program[entrypoint].options
        #FIXME: endswith is ugly maybe make
        # codegen_result.implemented_data_infos a dict?
        implemented_data_info = codegen_result.implemented_data_infos[entrypoint]

        from loopy.kernel.data import KernelArgument
        gen = PythonFunctionGenerator(
                "invoke_%s_loopy_kernel" % entrypoint,
                self.system_args + [
                    "%s=None" % idi.name
                    for idi in implemented_data_info
                    if issubclass(idi.arg_class, KernelArgument)
                    ])

        self.target_specific_preamble(gen)
        gen.add_to_preamble("")
        self.generate_host_code(gen, codegen_result)
        gen.add_to_preamble("")

        self.initialize_system_args(gen)

        self.generate_integer_arg_finding_from_shapes(
            gen, program[entrypoint], implemented_data_info)
        self.generate_integer_arg_finding_from_offsets(
            gen, program[entrypoint], implemented_data_info)
        self.generate_integer_arg_finding_from_strides(
            gen, program[entrypoint], implemented_data_info)
        self.generate_value_arg_check(
            gen, program[entrypoint], implemented_data_info)
        args = self.generate_arg_setup(
            gen, program[entrypoint], implemented_data_info, options)

        #FIXME: should we make this as a dict as well.
        host_program_name = codegen_result.host_programs[entrypoint].name

        self.generate_invocation(gen, host_program_name, args,
                program[entrypoint], implemented_data_info)

        self.generate_output_handler(gen, options, program[entrypoint],
                implemented_data_info)

        if options.write_wrapper:
            output = gen.get()
            if options.highlight_wrapper:
                output = get_highlighted_python_code(output)

            if options.write_wrapper is True:
                print(output)
            else:
                with open(options.write_wrapper, "w") as outf:
                    outf.write(output)

        return gen.get_picklable_function()

# }}}

# }}}


class _KernelInfo(ImmutableRecord):
    pass


class _Kernels:
    pass


typed_and_scheduled_cache = WriteOncePersistentDict(
        "loopy-typed-and-scheduled-cache-v1-"+DATA_MODEL_VERSION,
        key_builder=LoopyKeyBuilder())


invoker_cache = WriteOncePersistentDict(
        "loopy-invoker-cache-v1-"+DATA_MODEL_VERSION,
        key_builder=LoopyKeyBuilder())


# {{{ kernel executor

class KernelExecutorBase:
    """An object connecting a kernel to a :class:`pyopencl.Context`
    for execution.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, program, entrypoint):
        """
        :arg kernel: a loopy.LoopKernel
        """

        self.program = program
        self.entrypoint = entrypoint

        self.packing_controller = SeparateArrayPackingController(program,
                entrypoint)

        self.output_names = tuple(arg.name for arg in self.program[entrypoint].args
                if arg.is_output)

        self.has_runtime_typed_args = any(
                arg.dtype is None
                for arg in program[entrypoint].args)

    def get_typed_and_scheduled_program_uncached(self, entrypoint, arg_to_dtype_set):
        from loopy.kernel.tools import add_dtypes
        from loopy.kernel import KernelState
        from loopy.translation_unit import resolve_callables

        program = resolve_callables(self.program)

        if arg_to_dtype_set:
            var_to_dtype = {}
            entry_knl = program[entrypoint]
            for var, dtype in arg_to_dtype_set:
                if var in entry_knl.impl_arg_to_arg:
                    dest_name = entry_knl.impl_arg_to_arg[var].name
                else:
                    dest_name = var

                var_to_dtype[dest_name] = dtype

            program = program.with_kernel(add_dtypes(entry_knl, var_to_dtype))

            from loopy.type_inference import infer_unknown_types
            program = infer_unknown_types(program, expect_completion=True)

        if program.state < KernelState.SCHEDULED:
            from loopy.preprocess import preprocess_program
            program = preprocess_program(program)

            from loopy.schedule import get_one_linearized_kernel
            for e in program.entrypoints:
                program = program.with_kernel(
                    get_one_linearized_kernel(program[e], program.callables_table))

        return program

    def get_typed_and_scheduled_program(self, entrypoint, arg_to_dtype_set):
        from loopy import CACHING_ENABLED

        from loopy.preprocess import prepare_for_caching
        # prepare_for_caching() gets run by preprocess, but the kernel at this
        # stage is not guaranteed to be preprocessed.
        cacheable_program = prepare_for_caching(self.program)
        cache_key = (type(self).__name__, cacheable_program, arg_to_dtype_set)

        if CACHING_ENABLED:
            try:
                return typed_and_scheduled_cache[cache_key]
            except KeyError:
                pass

        logger.debug("%s: typed-and-scheduled cache miss" %
                self.program.entrypoints)

        kernel = self.get_typed_and_scheduled_program_uncached(entrypoint,
                arg_to_dtype_set)

        if CACHING_ENABLED:
            typed_and_scheduled_cache.store_if_not_present(cache_key, kernel)

        return kernel

    def arg_to_dtype_set(self, kwargs):
        kwargs = kwargs.copy()
        if not self.has_runtime_typed_args:
            return None

        entrypoint = kwargs.pop("entrypoint")

        impl_arg_to_arg = self.program[entrypoint].impl_arg_to_arg
        arg_to_dtype = {}
        for arg_name, val in kwargs.items():
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

        return frozenset(arg_to_dtype.items())

    # {{{ debugging aids

    def get_highlighted_code(self, entrypoint, arg_to_dtype=None, code=None):
        if code is None:
            code = self.get_code(entrypoint, arg_to_dtype)
        return get_highlighted_code(code)

    def get_code(self, entrypoint, arg_to_dtype=None):
        def process_dtype(dtype):
            if isinstance(dtype, type) and issubclass(dtype, np.generic):
                dtype = np.dtype(dtype)
            if isinstance(dtype, np.dtype):
                from loopy.types import NumpyType
                dtype = NumpyType(dtype, self.program.target)

            return dtype

        if arg_to_dtype is not None:
            arg_to_dtype = frozenset(
                    (k, process_dtype(v)) for k, v in arg_to_dtype.items())

        kernel = self.get_typed_and_scheduled_program(entrypoint, arg_to_dtype)

        from loopy.codegen import generate_code_v2
        code = generate_code_v2(kernel)
        return code.device_code()

    def get_invoker_uncached(self, program, entrypoint, *args):
        raise NotImplementedError()

    def get_invoker(self, program, entrypoint, *args):
        from loopy import CACHING_ENABLED

        cache_key = (self.__class__.__name__, (program, entrypoint))

        if CACHING_ENABLED:
            try:
                return invoker_cache[cache_key]
            except KeyError:
                pass

        logger.debug("%s: invoker cache miss" % entrypoint)

        invoker = self.get_invoker_uncached(program, entrypoint, *args)

        if CACHING_ENABLED:
            invoker_cache.store_if_not_present(cache_key, invoker)

        return invoker

    # }}}

    # {{{ call and info generator

    @memoize_method
    def kernel_info(self, arg_to_dtype_set=frozenset(), all_kwargs=None):
        raise NotImplementedError()

    def __call__(self, queue, **kwargs):
        raise NotImplementedError()

    # }}}

# }}}

# {{{ code highlighers


def get_highlighted_code(text, python=False):
    try:
        from pygments import highlight
    except ImportError:
        return text
    else:
        from pygments.lexers import CLexer, PythonLexer
        from pygments.formatters import TerminalFormatter

        return highlight(text, CLexer() if not python else PythonLexer(),
                         TerminalFormatter())


def get_highlighted_python_code(text):
    return get_highlighted_code(text, True)


# }}}

# vim: foldmethod=marker

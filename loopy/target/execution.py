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


from typing import (Callable, Tuple, Union, Set, FrozenSet, List, Dict,
        Optional, Sequence, Any)
from dataclasses import dataclass

from immutables import Map

from abc import ABC, abstractmethod
from loopy.diagnostic import LoopyError
from pytools.py_codegen import PythonFunctionGenerator
from pytools.codegen import Indentation, CodeGenerator

from pymbolic import var

import logging
logger = logging.getLogger(__name__)

from pytools.persistent_dict import WriteOncePersistentDict
from loopy.tools import LoopyKeyBuilder, caches
from loopy.typing import ExpressionT
from loopy.types import LoopyType, NumpyType
from loopy.kernel import KernelState, LoopKernel
from loopy.kernel.data import _ArraySeparationInfo, ArrayArg, auto
from loopy.translation_unit import TranslationUnit
from loopy.schedule.tools import KernelArgInfo
from loopy.version import DATA_MODEL_VERSION


# {{{ object array argument packing

class SeparateArrayPackingController:
    """For argument arrays with axes tagged to be implemented as separate
    arrays, this class provides preprocessing of the incoming arguments so that
    all sub-arrays may be passed in one object array (under the original,
    un-split argument name) and are unpacked into separate arrays before being
    passed to the kernel.

    It also repacks outgoing arrays of this type back into an object array.
    """

    def __init__(self, packing_info: Dict[str, _ArraySeparationInfo]) -> None:
        # These must work to index tuples if 1D.
        def untuple_length_1_indices(
                ind: Tuple[int, ...]) -> Union[int, Tuple[int, ...]]:
            if len(ind) == 1:
                return ind[0]
            else:
                return ind

        self.packing_info = {
                name: {
                    untuple_length_1_indices(ind): sep_name
                    for ind, sep_name in sep_info.subarray_names.items()
                    }
                for name, sep_info in packing_info.items()
                }

    def __call__(self, kernel_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kernel_kwargs = kernel_kwargs.copy()

        for name, ind_to_subary_name in self.packing_info.items():
            if name in kernel_kwargs:
                arg = kernel_kwargs[name]
                for index, unpacked_name in ind_to_subary_name.items():
                    assert unpacked_name not in kernel_kwargs
                    kernel_kwargs[unpacked_name] = arg[index]
                del kernel_kwargs[name]

        return kernel_kwargs

# }}}


# {{{ ExecutionWrapperGeneratorBase

def _str_to_expr(name_or_expr: Union[str, ExpressionT]) -> ExpressionT:
    if isinstance(name_or_expr, str):
        return var(name_or_expr)
    else:
        return name_or_expr


@dataclass(frozen=True)
class _ArgFindingEquation:
    lhs: ExpressionT
    rhs: ExpressionT

    # Arg finding code is sorted by priority, all equations (across all unknowns)
    # of lowest priority first.
    order: int

    based_on_names: FrozenSet[str]


class ExecutionWrapperGeneratorBase(ABC):
    """
    A set of common methods for generating a wrapper
    for execution

    """

    def __init__(self, system_args):
        self.system_args = system_args[:]

        from pytools import UniqueNameGenerator
        self.dtype_name_generator = UniqueNameGenerator(forced_prefix="_lpy_dtype_")
        self.dtype_str_to_name = {}

    @abstractmethod
    def python_dtype_str_inner(self, dtype):
        pass

    def python_dtype_str(self, gen: CodeGenerator, numpy_dtype):
        dtype_str = self.python_dtype_str_inner(numpy_dtype)
        try:
            return self.dtype_str_to_name[dtype_str]
        except KeyError:
            pass

        dtype_name = self.dtype_name_generator()
        gen.add_to_preamble(f"{dtype_name} = _lpy_np.dtype({dtype_str})")
        self.dtype_str_to_name[dtype_str] = dtype_name
        return dtype_name

    # {{{ invoker generation

    # /!\ This code runs in a namespace controlled by the user.
    # Prefix all auxiliary variables with "_lpy".

    # {{{ integer arg finding from array data

    def generate_integer_arg_finding_from_array_data(
            self, gen: CodeGenerator, kernel: LoopKernel, kai: KernelArgInfo
            ) -> None:
        from loopy.kernel.data import ArrayArg
        from loopy.kernel.array import get_strides
        from loopy.symbolic import DependencyMapper, StringifyMapper
        from loopy.diagnostic import ParameterFinderWarning
        dep_map = DependencyMapper()

        # {{{ find equations

        equations: List[_ArgFindingEquation] = []

        for arg_name in kai.passed_arg_names:
            arg = kernel.arg_dict[arg_name]
            assert arg.dtype is not None
            if isinstance(arg, ArrayArg):
                assert arg.shape is not auto
                if isinstance(arg.shape, tuple):
                    for axis_nr, shape_i in enumerate(arg.shape):
                        if shape_i is not None:
                            equations.append(
                                _ArgFindingEquation(
                                    lhs=var(arg.name).attr("shape").index(axis_nr),
                                    rhs=shape_i,
                                    order=0,
                                    based_on_names=frozenset({arg.name})))

                strides = get_strides(arg)
                for axis_nr, stride_i in enumerate(strides):
                    if stride_i is not None:
                        equations.append(
                                _ArgFindingEquation(
                                    lhs=var("_lpy_even_div")(
                                        var(arg.name).attr("strides").index(axis_nr),
                                        arg.dtype.itemsize),
                                    rhs=_str_to_expr(stride_i),
                                    order=0,
                                    based_on_names=frozenset({arg.name}),
                                    ))

                        if not arg.is_input and isinstance(arg.shape, tuple):
                            # If no value was found by other means, provide
                            # C-contiguous default strides for output-only
                            # arguments.
                            equations.append(
                                    _ArgFindingEquation(
                                        lhs=(strides[axis_nr + 1]
                                             * arg.shape[axis_nr + 1])
                                        if axis_nr + 1 < len(strides)
                                        else 1,
                                        rhs=_str_to_expr(stride_i),
                                        # Find strides from last dim to first,
                                        # starting at order=1 so that shape
                                        # parameters (found above) are
                                        # available.
                                        order=len(strides) - axis_nr,
                                        based_on_names=frozenset(),
                                        ))

                if arg.offset is not None:
                    equations.append(
                            _ArgFindingEquation(
                                lhs=var("_lpy_even_div_none")(
                                    var("getattr")(
                                        var(arg.name), var('"offset"'), var("None")),
                                    arg.dtype.itemsize),
                                rhs=_str_to_expr(arg.offset),
                                order=0,
                                based_on_names=frozenset([arg.name]),
                                ))

                    # If no value was found by other means, default to zero.
                    equations.append(
                            _ArgFindingEquation(
                                lhs=0,
                                rhs=_str_to_expr(arg.offset),
                                order=1,
                                based_on_names=frozenset(),
                                ))

        # }}}

        # {{{ regroup equations by unknown

        order_to_unknown_to_equations: \
                Dict[int, Dict[str, List[_ArgFindingEquation]]] = {}

        for eqn in equations:
            deps = dep_map(eqn.rhs)

            if len(deps) == 1:
                unknown_var, = deps
                order_to_unknown_to_equations \
                        .setdefault(eqn.order, {}) \
                        .setdefault(unknown_var.name, []) \
                        .append((eqn))
            else:
                # Zero deps: nothing to determine, forget about it.
                # 2+ deps: not implemented
                pass

        del equations

        # }}}

        # {{{ generate arg finding code

        from pymbolic.algorithm import solve_affine_equations_for
        from pymbolic.primitives import Variable
        from pytools.codegen import CodeGenerator

        gen("# {{{ find integer arguments from array data")
        gen("")

        for order_value in sorted(order_to_unknown_to_equations):
            for unknown_name in sorted(order_to_unknown_to_equations[order_value]):
                unk_equations = sorted(
                        order_to_unknown_to_equations[order_value][unknown_name],
                        key=lambda eqn: eqn.order)
                subgen = CodeGenerator()

                seen_based_on_names: Set[FrozenSet[str]] = set()

                if_or_elif = "if"

                for eqn in unk_equations:
                    if eqn.rhs == Variable(unknown_name):
                        # Some of the expressions above are non-affine. Let's not
                        # get carried away by trying to solve a much more complex
                        # problem than needed.
                        value_expr = eqn.lhs
                    else:
                        try:
                            # overkill :)
                            value_expr = solve_affine_equations_for(
                                    [unknown_name],
                                    [(eqn.lhs, eqn.rhs)]
                                    )[Variable(unknown_name)]
                        except Exception as e:
                            # went wrong? oh well
                            from warnings import warn
                            warn("Unable to generate code to automatically "
                                    f"find '{unknown_name}' "
                                    f"from '{', '.join(eqn.based_on_names)}':\n"
                                    f"{e}", ParameterFinderWarning)
                            continue

                    # Do not use more than one bit of data from each of the
                    # 'based_on_names' to find each value, i.e. if a value can be
                    # found via shape and strides, only one of them suffices.
                    # This also helps because strides can be unreliable in the
                    # face of zero-length axes.
                    if eqn.based_on_names in seen_based_on_names:
                        continue
                    seen_based_on_names.add(eqn.based_on_names)

                    if eqn.based_on_names:
                        condition = " and ".join(
                                f"{ary_name} is not None"
                                for ary_name in eqn.based_on_names)
                    else:
                        condition = "True"

                    subgen(f"{if_or_elif} {condition}:")
                    with Indentation(subgen):
                        subgen(
                                f"{unknown_name} = {StringifyMapper()(value_expr)}")
                    if_or_elif = "elif"

                    subgen("")

                if subgen.code:
                    gen(f"if {unknown_name} is None:")
                    with Indentation(gen):
                        gen.extend(subgen)

        gen("# }}}")
        gen("")

        # }}}

    # }}}

    # {{{ check that value args are present

    def generate_value_arg_check(
            self, gen: CodeGenerator, kernel: LoopKernel, kai: KernelArgInfo
            ) -> None:
        if kernel.options.skip_arg_checks:
            return

        from loopy.kernel.data import ValueArg

        gen("# {{{ check that value args are present")
        gen("")

        for arg_name in kai.passed_arg_names:
            arg = kernel.arg_dict[arg_name]
            if not isinstance(arg, ValueArg):
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

    def handle_non_numpy_arg(self, gen: CodeGenerator, arg):
        raise NotImplementedError()

    # }}}

    # {{{ handle allocation of unspecified arguements

    def handle_alloc(
            self, gen: CodeGenerator, arg: ArrayArg,
            strify: Callable[[Union[ExpressionT, Tuple[ExpressionT]]], str],
            skip_arg_checks: bool) -> None:
        """
        Handle allocation of non-specified arguements for C-execution
        """
        raise NotImplementedError()

    # }}}

    def get_arg_pass(self, arg):
        raise NotImplementedError()

    def get_strides_check_expr(self, shape, strides, expected_strides):
        assert len(shape) == len(strides) == len(expected_strides)

        # Returns an expression suitable for use for checking the strides of an
        # argument. Arguments should be sequences of strings.

        # Shape axes of length 1 are ignored because strides along these
        # axes are never used: The only valid index is 1.
        match_expr = " and ".join(
                f"({shape_i} == 1 or {strides_i} == {expected_strides_i})"
                for shape_i, strides_i, expected_strides_i
                in zip(shape, strides, expected_strides)) or "True"

        if shape:
            # If any shape component is zero, the array is empty and the strides
            # don't matter.
            match_expr = (f"({match_expr})"
            + "".join(f" or not {shape_i}" for shape_i in shape))

        return match_expr

    # {{{ arg setup

    def generate_arg_setup(
            self, gen: CodeGenerator, kernel: LoopKernel, kai: KernelArgInfo,
            ) -> Sequence[str]:
        options = kernel.options
        import loopy as lp

        from loopy.kernel.data import ImageArg
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

        for arg_name in kai.passed_arg_names:
            arg = kernel.arg_dict[arg_name]
            is_written = arg.name in kernel.get_written_variables()

            if not isinstance(arg, ArrayBase):
                args.append(arg.name)
                continue

            gen("# {{{ process %s" % arg.name)
            gen("")

            if not options.no_numpy:
                self.handle_non_numpy_arg(gen, arg)

            if not options.skip_arg_checks and arg.is_input:
                gen("if %s is None:" % arg.name)
                with Indentation(gen):
                    gen("raise RuntimeError(\"input argument '%s' must "
                            'be supplied")' % arg.name)
                    gen("")

            if (is_written and isinstance(arg, ImageArg)
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

            if arg.is_output \
                    and isinstance(arg, (lp.ArrayArg, lp.ConstantArg)) \
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
                        gen, arg, strify, options.skip_arg_checks)
                    gen("_lpy_made_by_loopy = True")
                    gen("")

            # }}}

            # {{{ argument checking

            if isinstance(arg, (lp.ArrayArg, lp.ConstantArg)) \
                    and not options.skip_arg_checks:
                if possibly_made_by_loopy:
                    gen("if not _lpy_made_by_loopy:")
                else:
                    gen("if True:")

                with Indentation(gen):
                    gen("if %s.dtype != %s:"
                            % (arg.name, self.python_dtype_str(
                                gen, arg.dtype.numpy_dtype)))
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

                    def strify_tuple(t: Optional[Tuple[ExpressionT, ...]]) -> str:
                        if t is None:
                            return "None"
                        if len(t) == 0:
                            return "()"
                        else:
                            return "(%s,)" % ", ".join(
                                    strify_allowing_none(sa)
                                    for sa in t)

                    shape_mismatch_msg = (
                            "raise ValueError(\"shape mismatch on argument '%s' "
                            '(got: %%s, expected: %%s)" '
                            "%% (%s.shape, %s))"
                            % (arg.name, arg.name, strify_tuple(arg.shape)))

                    if arg.shape is None:
                        pass

                    elif any(shape_axis is None for shape_axis in arg.shape):
                        assert isinstance(arg.shape, tuple)
                        gen("if len(%s.shape) != %s:"
                                % (arg.name, len(arg.shape)))
                        with Indentation(gen):
                            gen(shape_mismatch_msg)

                        for i, shape_axis in enumerate(arg.shape):
                            if shape_axis is None:
                                continue

                            gen("if %s.shape[%d] != %s:"
                                    % (arg.name, i, strify(shape_axis)))
                            with Indentation(gen):
                                gen(shape_mismatch_msg)

                    else:  # not None, no Nones in tuple
                        gen("if %s.shape != %s:"
                                % (arg.name, strify(arg.shape)))
                        with Indentation(gen):
                            gen(shape_mismatch_msg)

                    # }}}

                    from loopy.kernel.array import get_strides
                    strides = get_strides(arg)

                    if strides and arg.dim_tags and arg.shape is not None:
                        assert isinstance(arg.shape, tuple)
                        itemsize = arg.dtype.numpy_dtype.itemsize
                        sym_strides = tuple(itemsize*s_i for s_i in strides)

                        ndim = len(arg.shape)
                        shape = ["_lpy_shape_%d" % i for i in range(ndim)]
                        strides = ["_lpy_stride_%d" % i for i in range(ndim)]

                        gen("({},) = {}.shape".format(", ".join(shape), arg.name))
                        gen("({},) = {}.strides".format(
                            ", ".join(strides), arg.name))

                        gen("if not (%s):"
                                % self.get_strides_check_expr(
                                    shape, strides,
                                    [strify(s) for s in sym_strides]))
                        with Indentation(gen):
                            gen(f"_lpy_got = {arg.name}.strides")
                            gen(f"_lpy_expected = {strify_tuple(sym_strides)}")

                            gen('raise ValueError("strides mismatch on '
                                    "argument '%s' "
                                    '(got: %%s, expected: %%s)" '
                                    "%% (_lpy_got, _lpy_expected))"
                                    % arg.name)

                    if not arg.offset:
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

            if isinstance(arg, (lp.ArrayArg, lp.ConstantArg)):
                args.append(self.get_arg_pass(arg))
            else:
                args.append(arg.name)

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

    def generate_invocation(self, gen: CodeGenerator, kernel: LoopKernel,
            kai: KernelArgInfo, host_program_name: str, args: Sequence[str]) -> None:
        raise NotImplementedError()

    # }}}

    # {{{ output

    def generate_output_handler(self, gen: CodeGenerator,
            kernel: LoopKernel, kai: KernelArgInfo) -> None:
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

        kernel = program[entrypoint]
        options = kernel.options

        from loopy.schedule.tools import get_kernel_arg_info
        kai = get_kernel_arg_info(kernel)

        gen = PythonFunctionGenerator(
                "invoke_%s_loopy_kernel" % entrypoint,
                self.system_args + [
                    "%s=None" % arg_name
                    for arg_name in kai.passed_arg_names
                    ])

        self.target_specific_preamble(gen)
        gen.add_to_preamble("")
        self.generate_host_code(gen, codegen_result)
        gen.add_to_preamble("")

        self.initialize_system_args(gen)

        self.generate_integer_arg_finding_from_array_data(
                gen, program[entrypoint], kai)
        self.generate_value_arg_check(gen, program[entrypoint], kai)
        args = self.generate_arg_setup(gen, program[entrypoint], kai)

        #FIXME: should we make this as a dict as well.
        host_program_name = codegen_result.host_programs[entrypoint].name

        self.generate_invocation(gen, program[entrypoint], kai,
                host_program_name, args)

        self.generate_output_handler(gen, program[entrypoint], kai)

        if options.write_wrapper:
            output = gen.get()
            if options.allow_terminal_colors:
                output = get_highlighted_python_code(output)

            if options.write_wrapper is True:
                print(output)
            else:
                with open(options.write_wrapper, "w") as outf:
                    outf.write(output)

        return gen.get_picklable_function()

# }}}

# }}}


typed_and_scheduled_cache = WriteOncePersistentDict(
        "loopy-typed-and-scheduled-cache-v1-"+DATA_MODEL_VERSION,
        key_builder=LoopyKeyBuilder())


caches.append(typed_and_scheduled_cache)


invoker_cache = WriteOncePersistentDict(
        "loopy-invoker-cache-v10-"+DATA_MODEL_VERSION,
        key_builder=LoopyKeyBuilder())


caches.append(invoker_cache)


# {{{ kernel executor

class ExecutorBase:
    """An object allowing the execution of an entrypoint of a
    :class:`~loopy.TranslationUnit`. Create these objects using
    :meth:`loopy.TranslationUnit.executor`.

    .. automethod:: __call__
    """
    packing_controller: Optional[SeparateArrayPackingController]

    def __init__(self, t_unit: TranslationUnit, entrypoint: str):
        self.t_unit = t_unit
        self.entrypoint = entrypoint

        kernel = self.t_unit[entrypoint]
        self.output_names = {arg.name for arg in kernel.args if arg.is_output}

        from loopy import ArrayArg
        self.input_array_names = {
            arg.name for arg in kernel.args
            if arg.is_input and isinstance(arg, ArrayArg)}

        self.has_runtime_typed_args = any(arg.dtype is None for arg in kernel.args)

        # We're doing this ahead of time to learn about array separation.
        # This will be done again as part of preprocessing below, and we're
        # betting that it happens consistently both times. (No reason it wouldn't,
        # but it is done redundantly.) We can't *use* the result of this
        # because we need to do the 'official' array separation after type
        # inference has completed.
        from loopy.preprocess import make_arrays_for_sep_arrays
        self.separated_entry_knl = make_arrays_for_sep_arrays(
                self.t_unit[self.entrypoint])

        self.sep_info = self.separated_entry_knl._separation_info()
        if self.sep_info:
            self.packing_controller = SeparateArrayPackingController(self.sep_info)
        else:
            self.packing_controller = None

    def check_for_required_array_arguments(self, input_args):
        # Formerly, the first exception raised when a required argument is not
        # passed was often at type inference. This exists to raise a more meaningful
        # message in such scenarios. Since type inference precedes compilation, this
        # check cannot be deferred to the generated invoker code.
        # See discussion at
        # https://github.com/inducer/loopy/pull/160#issuecomment-867761204
        # and links therin for context.
        if not self.input_array_names <= set(input_args):
            missing_args = self.input_array_names - set(input_args)
            kernel = self.t_unit[self.entrypoint]
            raise LoopyError(
                f"Kernel {kernel.name}() missing required array input arguments: "
                f"{', '.join(missing_args)}. "
                "If this is a surprise, maybe you need to add is_input=False to "
                "your argument.")

    def get_typed_and_scheduled_translation_unit_uncached(
            self, arg_to_dtype: Optional[Map[str, LoopyType]]
            ) -> TranslationUnit:
        t_unit = self.t_unit

        if arg_to_dtype:
            entry_knl = t_unit[self.entrypoint]

            # FIXME: This is not so nice. This transfers types from the
            # subarrays of sep-tagged arrays to the 'main' array, because
            # type inference fails otherwise.
            with arg_to_dtype.mutate() as mm:
                for name, sep_info in self.sep_info.items():
                    if entry_knl.arg_dict[name].dtype is None:
                        for sep_name in sep_info.subarray_names.values():
                            if sep_name in arg_to_dtype:
                                mm.set(name, arg_to_dtype[sep_name])
                                del mm[sep_name]

                arg_to_dtype = mm.finish()

            from loopy.kernel.tools import add_dtypes
            t_unit = t_unit.with_kernel(add_dtypes(entry_knl, arg_to_dtype))

            from loopy.type_inference import infer_unknown_types
            t_unit = infer_unknown_types(t_unit, expect_completion=True)

        if t_unit.state < KernelState.PREPROCESSED:
            from loopy.preprocess import preprocess_program
            t_unit = preprocess_program(t_unit)

        if t_unit.state < KernelState.LINEARIZED:
            from loopy.schedule import linearize
            t_unit = linearize(t_unit)

        return t_unit

    def get_typed_and_scheduled_translation_unit(
            self, arg_to_dtype: Optional[Map[str, LoopyType]]
            ) -> TranslationUnit:
        from loopy import CACHING_ENABLED

        cache_key = (type(self).__name__, self.t_unit, arg_to_dtype)

        if CACHING_ENABLED:
            try:
                return typed_and_scheduled_cache[cache_key]
            except KeyError:
                pass

        logger.debug("%s: typed-and-scheduled cache miss" %
                self.t_unit.entrypoints)

        kernel = self.get_typed_and_scheduled_translation_unit_uncached(arg_to_dtype)

        if CACHING_ENABLED:
            typed_and_scheduled_cache.store_if_not_present(cache_key, kernel)

        return kernel

    def arg_to_dtype(self, kwargs) -> Optional[Map[str, LoopyType]]:
        if not self.has_runtime_typed_args:
            return None

        arg_dict = self.separated_entry_knl.arg_dict
        arg_to_dtype = {}
        for arg_name, val in kwargs.items():
            arg = arg_dict[arg_name]

            if arg.dtype is None and val is not None:
                try:
                    dtype = val.dtype
                except AttributeError:
                    pass
                else:
                    arg_to_dtype[arg_name] = NumpyType(dtype)

        return Map(arg_to_dtype)

    # {{{ debugging aids

    def get_highlighted_code(self, entrypoint, arg_to_dtype=None, code=None):
        if code is None:
            code = self.get_code(entrypoint, arg_to_dtype)
        return get_highlighted_code(code)

    def get_code(
            self, entrypoint: str,
            arg_to_dtype: Optional[Map[str, LoopyType]] = None) -> str:
        kernel = self.get_typed_and_scheduled_translation_unit(arg_to_dtype)

        from loopy.codegen import generate_code_v2
        code = generate_code_v2(kernel)
        return code.device_code()

    def get_invoker_uncached(self, program, entrypoint, *args):
        raise NotImplementedError()

    def get_invoker(self, t_unit, entrypoint, *args):
        from loopy import CACHING_ENABLED

        cache_key = (self.__class__.__name__, (t_unit, entrypoint))

        if CACHING_ENABLED:
            try:
                return invoker_cache[cache_key]
            except KeyError:
                pass

        logger.debug("%s: invoker cache miss" % entrypoint)

        invoker = self.get_invoker_uncached(t_unit, entrypoint, *args)

        if CACHING_ENABLED:
            invoker_cache.store_if_not_present(cache_key, invoker)

        return invoker

    # }}}

    # {{{ call and info generator

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

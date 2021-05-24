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

import logging
logger = logging.getLogger(__name__)

from loopy.diagnostic import LoopyError

from pytools import ImmutableRecord

from pytools.persistent_dict import WriteOncePersistentDict
from loopy.tools import LoopyKeyBuilder
from loopy.version import DATA_MODEL_VERSION


from loopy.symbolic import CombineMapper
from functools import reduce

from loopy.kernel.function_interface import CallableKernel

from pytools import ProcessLogger

__doc__ = """
.. currentmodule:: loopy.codegen

.. autoclass:: ImplementedDataInfo

.. autoclass:: PreambleInfo

.. autoclass:: VectorizationInfo

.. autoclass:: SeenFunction

.. autoclass:: CodeGenerationState

.. autoclass:: TranslationUnitCodeGenerationResult

.. automodule:: loopy.codegen.result

.. automodule:: loopy.codegen.tools
"""


# {{{ implemented data info

class ImplementedDataInfo(ImmutableRecord):
    """
    .. attribute:: name

        The expanded name of the array. Note that, for example
        in the case of separate-array-tagged axes, multiple
        implemented arrays may correspond to one user-facing
        array.

    .. attribute:: dtype

    .. attribute:: arg_class

    .. attribute:: base_name

        The user-facing name of the underlying array.
        May be *None* for non-array arguments.

    .. attribute:: shape
    .. attribute:: strides

        Strides in multiples of ``dtype.itemsize``.

    .. attribute:: unvec_shape
    .. attribute:: unvec_strides

        Strides in multiples of ``dtype.itemsize`` that accounts for
        :class:`loopy.kernel.array.VectorArrayDimTag` in a scalar
        manner


    .. attribute:: offset_for_name
    .. attribute:: stride_for_name_and_axis

        A tuple *(name, axis)* indicating the (implementation-facing)
        name of the array and axis number for which this argument provides
        the strides.

    .. attribute:: allows_offset
    .. attribute:: is_written
    """

    def __init__(self, target, name, dtype, arg_class,
            base_name=None,
            shape=None, strides=None,
            unvec_shape=None, unvec_strides=None,
            offset_for_name=None, stride_for_name_and_axis=None,
            allows_offset=None,
            is_written=None):

        from loopy.types import LoopyType
        assert isinstance(dtype, LoopyType)

        ImmutableRecord.__init__(self,
                name=name,
                dtype=dtype,
                arg_class=arg_class,
                base_name=base_name,
                shape=shape,
                strides=strides,
                unvec_shape=unvec_shape,
                unvec_strides=unvec_strides,
                offset_for_name=offset_for_name,
                stride_for_name_and_axis=stride_for_name_and_axis,
                allows_offset=allows_offset,
                is_written=is_written)

# }}}


class UnvectorizableError(Exception):
    pass


class VectorizationInfo:
    """
    .. attribute:: iname
    .. attribute:: length
    """

    def __init__(self, iname, length):
        self.iname = iname
        self.length = length


class SeenFunction(ImmutableRecord):
    """This is used to track functions that emerge late during code generation,
    e.g. C functions to realize arithmetic. No connection with
    :class:`~loopy.kernel.function_interface.InKernelCallable`.

    .. attribute:: name
    .. attribute:: c_name
    .. attribute:: arg_dtypes

        a tuple of arg dtypes

    .. attribute:: result_dtypes

        a tuple of result dtypes
    """

    def __init__(self, name, c_name, arg_dtypes, result_dtypes):
        ImmutableRecord.__init__(self,
                name=name,
                c_name=c_name,
                arg_dtypes=arg_dtypes,
                result_dtypes=result_dtypes)


code_gen_cache = WriteOncePersistentDict(
         "loopy-code-gen-cache-v3-"+DATA_MODEL_VERSION,
         key_builder=LoopyKeyBuilder())


class InKernelCallablesCollector(CombineMapper):
    """
    Returns an instance of :class:`frozenset` containing instances of
    :class:`loopy.kernel.function_interface.InKernelCallable` in the
    :attr:``kernel`.
    """
    def __init__(self, kernel):
        self.kernel = kernel

    def combine(self, values):
        import operator
        return reduce(operator.or_, values, frozenset())

    def map_resolved_function(self, expr):
        return frozenset([self.kernel.scoped_functions[
            expr.name]])

    def map_constant(self, expr):
        return frozenset()

    map_variable = map_constant
    map_function_symbol = map_constant
    map_tagged_variable = map_constant
    map_type_cast = map_constant


class PreambleInfo(ImmutableRecord):
    """
    .. attribute:: kernel
    .. attribute:: seen_dtypes
    .. attribute:: seen_functions
    .. attribute:: seen_atomic_dtypes
    """


# {{{ main code generation entrypoint

def generate_code_for_a_single_kernel(kernel, callables_table, target,
        is_entrypoint):
    """
    :returns: a :class:`CodeGenerationResult`

    :param kernel: An instance of :class:`loopy.LoopKernel`.
    """

    from loopy.kernel import KernelState
    if kernel.state != KernelState.LINEARIZED:
        raise LoopyError("cannot generate code for a kernel that has not been "
                "scheduled")

    codegen_plog = ProcessLogger(logger, f"{kernel.name}: generate code")

    # {{{ pre-codegen-process of non-entrypoint kernel

    if not is_entrypoint:
        from loopy.kernel.array import ArrayBase
        from loopy.kernel.data import auto

        new_args = [arg.copy(offset=0 if arg.offset is auto else arg.offset)
                    if isinstance(arg, ArrayBase)
                    else arg
                    for arg in kernel.args]
        kernel = kernel.copy(args=new_args)

    # }}}

    # {{{ make_schedule_tree

    from loopy.schedule.tree import (make_schedule_tree,
                                     insert_predicates_into_schedule)
    kernel = make_schedule_tree(kernel)
    kernel = insert_predicates_into_schedule(kernel)

    # }}}

    from loopy.codegen.result import get_idis_for_kernel, CodeGenMapper
    codegen_mapper = CodeGenMapper(kernel)

    codegen_result = codegen_mapper(kernel.schedule)

    seen_dtypes = (codegen_mapper.device_ast_builder.seen_dtypes
                   | codegen_mapper.host_ast_builder.seen_dtypes)
    seen_atomic_dtypes = (codegen_mapper.device_ast_builder.seen_atomic_dtypes
                          | codegen_mapper.host_ast_builder.seen_atomic_dtypes)
    seen_functions = (codegen_mapper.device_ast_builder.seen_functions
                      | codegen_mapper.host_ast_builder.seen_functions)

    # FIXME: Fix this!!!
    # from loopy.check import check_implemented_domains
    # assert check_implemented_domains(kernel,
    #                                  codegen_result.implemented_domains,
    #                                  codegen_result.device_code())

    # {{{ handle preambles

    for idi in get_idis_for_kernel(kernel):
        seen_dtypes.add(idi.dtype)

    for tv in kernel.temporary_variables.values():
        for idi in tv.decl_info(kernel.target, index_dtype=kernel.index_dtype):
            seen_dtypes.add(idi.dtype)

    if kernel.all_inames():
        seen_dtypes.add(kernel.index_dtype)

    preambles = kernel.preambles[:]

    preamble_info = PreambleInfo(
            kernel=kernel,
            seen_dtypes=seen_dtypes,
            seen_functions=seen_functions,
            # a set of LoopyTypes (!)
            seen_atomic_dtypes=seen_atomic_dtypes,
            )

    preamble_generators = (kernel.preamble_generators
            + target.get_device_ast_builder().preamble_generators())
    for prea_gen in preamble_generators:
        preambles.extend(prea_gen(preamble_info))

    codegen_result = codegen_result.copy(device_preambles=preambles)

    # }}}

    codegen_plog.done()

    return codegen_result


def diverge_callee_entrypoints(program):
    """
    If a :class:`loopy.kernel.function_interface.CallableKernel` is both an
    entrypoint and a callee, then rename the callee.
    """
    from loopy.translation_unit import (get_reachable_resolved_callable_ids,
                                        rename_resolved_functions_in_a_single_kernel,
                                        make_callable_name_generator)
    callable_ids = get_reachable_resolved_callable_ids(program.callables_table,
                                                       program.entrypoints)

    new_callables = {}
    todo_renames = {}

    vng = make_callable_name_generator(program.callables_table)

    for clbl_id in callable_ids & program.entrypoints:
        todo_renames[clbl_id] = vng(based_on=clbl_id)

    for name, clbl in program.callables_table.items():
        if name in todo_renames:
            name = todo_renames[name]

        if isinstance(clbl, CallableKernel):
            knl = rename_resolved_functions_in_a_single_kernel(clbl.subkernel,
                                                               todo_renames)
            knl = knl.copy(name=name)
            clbl = clbl.copy(subkernel=knl)

        new_callables[name] = clbl

    return program.copy(callables_table=new_callables)


class TranslationUnitCodeGenerationResult(ImmutableRecord):
    """
    .. attribute:: host_program

        A mapping from names of entrypoints to their host
        :class:`~loopy.codegen.result.GeneratedProgram`.

    .. attribute:: device_programs

        A list of :class:`~loopy.codegen.result.GeneratedProgram` instances
        intended to run on the compute device.

    .. attribute:: host_preambles
    .. attribute:: device_preambles

    .. attribute:: implemented_data_infos

        A mapping from names of entrypoints to their
        list of :class:`ImplementedDataInfo` objects.

    .. automethod:: host_code
    .. automethod:: device_code
    .. automethod:: all_code

    """
    def host_code(self):
        from loopy.codegen.result import process_preambles
        preamble_codes = process_preambles(getattr(self, "host_preambles", []))

        return (
                "".join(preamble_codes)
                + "\n"
                + "\n\n".join(str(hp.ast)
                              for hp in self.host_programs.values()))

    def device_code(self):
        from loopy.codegen.result import process_preambles
        preamble_codes = process_preambles(getattr(self, "device_preambles", []))

        return (
                "".join(preamble_codes)
                + "\n"
                + "\n\n".join(str(dp.ast) for dp in self.device_programs))

    def all_code(self):
        from loopy.codegen.result import process_preambles
        preamble_codes = process_preambles(
                getattr(self, "host_preambles", [])
                +
                getattr(self, "device_preambles", [])
                )

        return (
                "".join(preamble_codes)
                + "\n"
                + "\n\n".join(str(dp.ast) for dp in self.device_programs)
                + "\n\n"
                + "\n\n".join(str(hp.ast) for hp in
                    self.host_programs.values()))


def generate_code_v2(program):
    """
    Returns an instance of :class:`CodeGenerationResult`.

    :param program: An instance of :class:`loopy.TranslationUnit`.
    """

    from loopy.kernel import LoopKernel
    from loopy.translation_unit import make_program

    # {{{ cache retrieval

    from loopy import CACHING_ENABLED

    if CACHING_ENABLED:
        input_program = program
        try:
            result = code_gen_cache[input_program]
            logger.debug(f"TranslationUnit with entrypoints {program.entrypoints}:"
                         " code generation cache hit")
            return result
        except KeyError:
            pass

    # }}}

    if isinstance(program, LoopKernel):
        program = make_program(program)

    from loopy.kernel import KernelState
    if program.state < KernelState.PREPROCESSED:
        # Note that we cannot have preprocessing separately for everyone.
        # Since, now the preprocessing of each one depends on the other.
        # So we check if any one of the callable kernels are not preprocesses
        # then, we have to do the preprocessing of every other kernel.
        from loopy.preprocess import preprocess_program
        program = preprocess_program(program)

    from loopy.type_inference import infer_unknown_types
    program = infer_unknown_types(program, expect_completion=True)

    from loopy.schedule import linearize
    program = linearize(program)

    # Why diverge? Generated code for a non-entrypoint kernel and an entrypoint
    # kernel isn't same for a general loopy target. For example in OpenCL, a
    # kernel callable from host and the one supposed to be callable from device
    # have different function signatures. To generate correct code, each
    # callable should be exclusively an entrypoint or a non-entrypoint kernel.
    program = diverge_callee_entrypoints(program)

    from loopy.check import pre_codegen_checks
    pre_codegen_checks(program)

    host_programs = {}
    device_programs = []
    device_preambles = []
    callee_fdecls = []
    implemented_data_infos = {}

    # {{{ collect host/device programs

    for func_id in sorted(key for key, val in program.callables_table.items()
                          if isinstance(val, CallableKernel)):
        cgr = generate_code_for_a_single_kernel(program[func_id],
                                                program.callables_table,
                                                program.target,
                                                func_id in program.entrypoints)
        if func_id in program.entrypoints:
            host_programs[func_id] = cgr.host_program
            implemented_data_infos[func_id] = cgr.implemented_data_info
        else:
            assert len(cgr.device_programs) == 1
            callee_fdecls.append(cgr.device_programs[0].ast.fdecl)

        device_programs.extend(cgr.device_programs)
        device_preambles.extend(cgr.device_preambles)

    # }}}

    # {{{ collect preambles

    for clbl in program.callables_table.values():
        device_preambles.extend(list(clbl.generate_preambles(program.target)))

    # }}}

    # adding the callee fdecls to the device_programs
    device_programs = ([device_programs[0].copy(
            ast=program.target.get_device_ast_builder().ast_module.Collection(
                callee_fdecls+[device_programs[0].ast]))] +
            device_programs[1:])
    cgr = TranslationUnitCodeGenerationResult(
            host_programs=host_programs,
            device_programs=device_programs,
            device_preambles=device_preambles,
            implemented_data_infos=implemented_data_infos)

    if CACHING_ENABLED:
        code_gen_cache.store_if_not_present(input_program, cgr)

    return cgr


def generate_code(kernel, device=None):
    if device is not None:
        from warnings import warn
        warn("passing 'device' to generate_code() is deprecated",
                DeprecationWarning, stacklevel=2)

    codegen_result = generate_code_v2(kernel)

    if len(codegen_result.device_programs) > 1:
        raise LoopyError("kernel passed to generate_code yielded multiple "
                "device programs. Use generate_code_v2.")
    if len(codegen_result.host_programs) > 1:
        raise LoopyError("kernel passed to generate_code yielded multiple "
                "host programs. Use generate_code_v2.")

    assert len(codegen_result.implemented_data_infos) == 1
    implemented_data_info, = codegen_result.implemented_data_infos.values()

    return codegen_result.device_code(), implemented_data_info

# }}}


# {{{ generate function body

def generate_body(kernel):
    codegen_result = generate_code_v2(kernel)

    if len(codegen_result.device_programs) != 1:
        raise LoopyError("generate_body cannot be used on programs "
                "that yield more than one device program")

    dev_prg, = codegen_result.device_programs

    return str(dev_prg.body_ast)

# }}}

# vim: foldmethod=marker

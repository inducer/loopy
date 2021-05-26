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

import islpy as isl

from loopy.diagnostic import LoopyError, warn
from pytools import ImmutableRecord

from pytools.persistent_dict import WriteOncePersistentDict
from loopy.tools import LoopyKeyBuilder
from loopy.version import DATA_MODEL_VERSION


from loopy.symbolic import CombineMapper
from functools import reduce

from loopy.kernel.function_interface import CallableKernel, ScalarCallable

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


# {{{ code generation state

class Unvectorizable(Exception):
    pass


class VectorizationInfo:
    """
    .. attribute:: iname
    .. attribute:: length
    .. attribute:: space
    """

    def __init__(self, iname, length, space):
        self.iname = iname
        self.length = length
        self.space = space


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


class CodeGenerationState:
    """
    .. attribute:: kernel
    .. attribute:: target
    .. attribute:: implemented_data_info

        a list of :class:`ImplementedDataInfo` objects.

    .. attribute:: implemented_domain

        The entire implemented domain (as an :class:`islpy.Set`)
        i.e. all constraints that have been enforced so far.

    .. attribute:: implemented_predicates

        A :class:`frozenset` of predicates for which checks have been
        implemented.

    .. attribute:: seen_dtypes

        set of dtypes that were encountered

    .. attribute:: seen_functions

        set of :class:`SeenFunction` instances

    .. attribute:: seen_atomic_dtypes

    .. attribute:: var_subst_map

    .. attribute:: allow_complex

    .. attribute:: vectorization_info

        None or an instance of :class:`VectorizationInfo`

    .. attribute:: is_generating_device_code

    .. attribute:: gen_program_name

        None (indicating that host code is being generated)
        or the name of the device program currently being
        generated.

    .. attribute:: schedule_index_end

    .. attribute:: callables_table

        A mapping from callable names to instances of
        :class:`loopy.kernel.function_interface.InKernelCallable`.

    .. attribute:: is_entrypoint

        A :class:`bool` to indicate if the code is being generated for an
        entrypoint kernel

    .. attribute:: codegen_cache_manager

        An instance of :class:`loopy.codegen.tools.CodegenOperationCacheManager`.
    """

    def __init__(self, kernel, target,
            implemented_data_info, implemented_domain, implemented_predicates,
            seen_dtypes, seen_functions, seen_atomic_dtypes, var_subst_map,
            allow_complex,
            callables_table,
            is_entrypoint,
            vectorization_info=None, var_name_generator=None,
            is_generating_device_code=None,
            gen_program_name=None,
            schedule_index_end=None,
            codegen_cachemanager=None):
        self.kernel = kernel
        self.target = target
        self.implemented_data_info = implemented_data_info
        self.implemented_domain = implemented_domain
        self.implemented_predicates = implemented_predicates
        self.seen_dtypes = seen_dtypes
        self.seen_functions = seen_functions
        self.seen_atomic_dtypes = seen_atomic_dtypes
        self.var_subst_map = var_subst_map.copy()
        self.allow_complex = allow_complex
        self.callables_table = callables_table
        self.is_entrypoint = is_entrypoint
        self.vectorization_info = vectorization_info
        self.var_name_generator = var_name_generator
        self.is_generating_device_code = is_generating_device_code
        self.gen_program_name = gen_program_name
        self.schedule_index_end = schedule_index_end
        self.codegen_cachemanager = codegen_cachemanager

    # {{{ copy helpers

    def copy(self, kernel=None, target=None, implemented_data_info=None,
            implemented_domain=None, implemented_predicates=frozenset(),
            var_subst_map=None, is_entrypoint=None, vectorization_info=None,
            is_generating_device_code=None, gen_program_name=None,
            schedule_index_end=None):

        if kernel is None:
            kernel = self.kernel

        if target is None:
            target = self.target

        if implemented_data_info is None:
            implemented_data_info = self.implemented_data_info

        if is_entrypoint is None:
            is_entrypoint = self.is_entrypoint

        if vectorization_info is False:
            vectorization_info = None

        elif vectorization_info is None:
            vectorization_info = self.vectorization_info

        if is_generating_device_code is None:
            is_generating_device_code = self.is_generating_device_code

        if gen_program_name is None:
            gen_program_name = self.gen_program_name

        if schedule_index_end is None:
            schedule_index_end = self.schedule_index_end

        return CodeGenerationState(
                kernel=kernel,
                target=target,
                implemented_data_info=implemented_data_info,
                implemented_domain=implemented_domain or self.implemented_domain,
                implemented_predicates=(
                    implemented_predicates or self.implemented_predicates),
                seen_dtypes=self.seen_dtypes,
                seen_functions=self.seen_functions,
                seen_atomic_dtypes=self.seen_atomic_dtypes,
                var_subst_map=var_subst_map or self.var_subst_map,
                allow_complex=self.allow_complex,
                callables_table=self.callables_table,
                is_entrypoint=is_entrypoint,
                vectorization_info=vectorization_info,
                var_name_generator=self.var_name_generator,
                is_generating_device_code=is_generating_device_code,
                gen_program_name=gen_program_name,
                schedule_index_end=schedule_index_end,
                codegen_cachemanager=self.codegen_cachemanager.with_kernel(kernel),
                )

    def copy_and_assign(self, name, value):
        """Make a copy of self with variable *name* fixed to *value*."""
        var_subst_map = self.var_subst_map.copy()
        var_subst_map[name] = value
        return self.copy(var_subst_map=var_subst_map)

    def copy_and_assign_many(self, assignments):
        """Make a copy of self with *assignments* included."""

        var_subst_map = self.var_subst_map.copy()
        var_subst_map.update(assignments)
        return self.copy(var_subst_map=var_subst_map)

    # }}}

    @property
    def expression_to_code_mapper(self):
        return self.ast_builder.get_expression_to_code_mapper(self)

    def intersect(self, other):
        new_impl, new_other = isl.align_two(self.implemented_domain, other)
        return self.copy(implemented_domain=new_impl & new_other)

    def fix(self, iname, aff):
        new_impl_domain = self.implemented_domain

        impl_space = self.implemented_domain.get_space()
        if iname not in impl_space.get_var_dict():
            new_impl_domain = (new_impl_domain
                    .add_dims(isl.dim_type.set, 1)
                    .set_dim_name(
                        isl.dim_type.set,
                        new_impl_domain.dim(isl.dim_type.set),
                        iname))
            impl_space = new_impl_domain.get_space()

        from loopy.isl_helpers import iname_rel_aff
        iname_plus_lb_aff = iname_rel_aff(impl_space, iname, "==", aff)

        from loopy.symbolic import pw_aff_to_expr
        cns = isl.Constraint.equality_from_aff(iname_plus_lb_aff)
        expr = pw_aff_to_expr(aff)

        new_impl_domain = new_impl_domain.add_constraint(cns)
        return self.copy_and_assign(iname, expr).copy(
                implemented_domain=new_impl_domain)

    def try_vectorized(self, what, func):
        """If *self* is in a vectorizing state (:attr:`vectorization_info` is
        not None), tries to call func (which must be a callable accepting a
        single :class:`CodeGenerationState` argument). If this fails with
        :exc:`Unvectorizable`, it unrolls the vectorized loop instead.

        *func* should return a :class:`GeneratedCode` instance.

        :returns: :class:`GeneratedCode`
        """

        if self.vectorization_info is None:
            return func(self)

        try:
            return func(self)
        except Unvectorizable as e:
            warn(self.kernel, "vectorize_failed",
                    "Vectorization of '%s' failed because '%s'"
                    % (what, e))

            return self.unvectorize(func)

    def unvectorize(self, func):
        vinf = self.vectorization_info
        result = []
        novec_self = self.copy(vectorization_info=False)

        for i in range(vinf.length):
            idx_aff = isl.Aff.zero_on_domain(vinf.space.params()) + i
            new_codegen_state = novec_self.fix(vinf.iname, idx_aff)
            generated = func(new_codegen_state)

            if isinstance(generated, list):
                result.extend(generated)
            else:
                result.append(generated)

        from loopy.codegen.result import merge_codegen_results
        return merge_codegen_results(self, result)

    @property
    def ast_builder(self):
        if self.is_generating_device_code:
            return self.kernel.target.get_device_ast_builder()
        else:
            return self.kernel.target.get_host_ast_builder()

# }}}


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
    .. attribute:: codegen_state
    """


# {{{ main code generation entrypoint

def generate_code_for_a_single_kernel(kernel, callables_table, target,
        is_entrypoint):
    """
    :returns: a :class:`CodeGenerationResult`

    :param kernel: An instance of :class:`loopy.LoopKernel`.
    """

    from loopy.kernel import KernelState
    if kernel.state != KernelState.SCHEDULED:
        raise LoopyError("cannot generate code for a kernel that has not been "
                "scheduled")

    codegen_plog = ProcessLogger(logger, f"{kernel.name}: generate code")

    # {{{ examine arg list

    from loopy.kernel.data import ValueArg
    from loopy.kernel.array import ArrayBase

    implemented_data_info = []

    for arg in kernel.args:
        is_written = arg.name in kernel.get_written_variables()
        if isinstance(arg, ArrayBase):
            implemented_data_info.extend(
                    arg.decl_info(
                        target,
                        is_written=is_written,
                        index_dtype=kernel.index_dtype))

        elif isinstance(arg, ValueArg):
            implemented_data_info.append(ImplementedDataInfo(
                target=target,
                name=arg.name,
                dtype=arg.dtype,
                arg_class=ValueArg,
                is_written=is_written))

        else:
            raise ValueError("argument type not understood: '%s'" % type(arg))

    allow_complex = False
    for var in kernel.args + list(kernel.temporary_variables.values()):
        if var.dtype.involves_complex():
            allow_complex = True

    # }}}

    seen_dtypes = set()
    seen_functions = set()
    seen_atomic_dtypes = set()

    initial_implemented_domain = isl.BasicSet.from_params(kernel.assumptions)

    from loopy.codegen.tools import CodegenOperationCacheManager

    codegen_state = CodeGenerationState(
            kernel=kernel,
            target=target,
            implemented_data_info=implemented_data_info,
            implemented_domain=initial_implemented_domain,
            implemented_predicates=frozenset(),
            seen_dtypes=seen_dtypes,
            seen_functions=seen_functions,
            seen_atomic_dtypes=seen_atomic_dtypes,
            var_subst_map={},
            allow_complex=allow_complex,
            var_name_generator=kernel.get_var_name_generator(),
            is_generating_device_code=False,
            gen_program_name=(
                target.host_program_name_prefix
                + kernel.name
                + kernel.target.host_program_name_suffix),
            schedule_index_end=len(kernel.schedule),
            callables_table=callables_table,
            is_entrypoint=is_entrypoint,
            codegen_cachemanager=CodegenOperationCacheManager.from_kernel(kernel),
            )

    from loopy.codegen.result import generate_host_or_device_program

    codegen_result = generate_host_or_device_program(
            codegen_state,
            schedule_index=0)

    device_code_str = codegen_result.device_code()

    from loopy.check import check_implemented_domains
    assert check_implemented_domains(kernel, codegen_result.implemented_domains,
            device_code_str)

    # {{{ handle preambles

    for idi in codegen_state.implemented_data_info:
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
            codegen_state=codegen_state
            )

    preamble_generators = (kernel.preamble_generators
            + target.get_device_ast_builder().preamble_generators())
    for prea_gen in preamble_generators:
        preambles.extend(prea_gen(preamble_info))

    codegen_result = codegen_result.copy(device_preambles=preambles)

    # }}}

    # For faster unpickling in the common case when implemented_domains isn't needed.
    from loopy.tools import LazilyUnpicklingDict
    codegen_result = codegen_result.copy(
            implemented_domains=LazilyUnpicklingDict(
                    codegen_result.implemented_domains))

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
    from loopy.preprocess import prepare_for_caching

    if CACHING_ENABLED:
        input_program = prepare_for_caching(program)
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

    new_callables = {}

    for name, clbl in program.callables_table.items():
        if isinstance(clbl, CallableKernel):
            from loopy.schedule import get_one_linearized_kernel
            knl = clbl.subkernel
            if knl.schedule is None:
                knl = get_one_linearized_kernel(
                            knl, program.callables_table)
            new_callables[name] = clbl.copy(subkernel=knl)
        elif isinstance(clbl, ScalarCallable):
            new_callables[name] = clbl
        else:
            raise NotImplementedError(type(clbl))

    program = program.copy(callables_table=new_callables)

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

    for func_id, clbl in program.callables_table.items():
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

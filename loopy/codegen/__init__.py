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

import sys
from immutables import Map
from typing import (Set, Mapping, Sequence, Any, FrozenSet, Union,
       Optional, Tuple, TYPE_CHECKING)
from dataclasses import dataclass, replace
import logging
logger = logging.getLogger(__name__)

import islpy as isl

from loopy.diagnostic import LoopyError, warn
from pytools import UniqueNameGenerator

from pytools.persistent_dict import WriteOncePersistentDict
from loopy.tools import LoopyKeyBuilder, caches
from loopy.version import DATA_MODEL_VERSION
from loopy.types import LoopyType
from loopy.typing import ExpressionT
from loopy.kernel import LoopKernel
from loopy.target import TargetBase
from loopy.kernel.function_interface import InKernelCallable


from loopy.symbolic import CombineMapper
from functools import reduce

from loopy.kernel.function_interface import CallableKernel

from pytools import ProcessLogger

if TYPE_CHECKING:
    from loopy.codegen.tools import CodegenOperationCacheManager
    from loopy.codegen.result import GeneratedProgram


if getattr(sys, "_BUILDING_SPHINX_DOCS", False):
    from loopy.codegen.tools import CodegenOperationCacheManager  # noqa: F811
    from loopy.codegen.result import GeneratedProgram  # noqa: F811


__doc__ = """
.. currentmodule:: loopy.codegen

.. autoclass:: PreambleInfo

.. autoclass:: VectorizationInfo

.. autoclass:: SeenFunction

.. autoclass:: CodeGenerationState

.. autoclass:: TranslationUnitCodeGenerationResult

.. automodule:: loopy.codegen.result

.. automodule:: loopy.codegen.tools
"""


# {{{ code generation state

class UnvectorizableError(Exception):
    pass


@dataclass(frozen=True)
class VectorizationInfo:
    """
    .. attribute:: iname
    .. attribute:: length
    .. attribute:: space
    """

    iname: str
    length: int
    # FIXME why is this here?
    space: isl.Space


@dataclass(frozen=True)
class SeenFunction:
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
    name: str
    c_name: str
    arg_dtypes: Tuple[LoopyType, ...]
    result_dtypes: Tuple[LoopyType, ...]


@dataclass(frozen=True)
class CodeGenerationState:
    """
    .. attribute:: kernel
    .. attribute:: target
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

        *None* (to mean vectorization has not yet been applied),  or an instance of
        :class:`VectorizationInfo`.

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

    kernel: LoopKernel
    target: TargetBase
    implemented_domain: isl.Set
    implemented_predicates: FrozenSet[Union[str, ExpressionT]]

    # /!\ mutable
    seen_dtypes: Set[LoopyType]
    seen_functions: Set[SeenFunction]
    seen_atomic_dtypes: Set[LoopyType]

    var_subst_map: Map[str, ExpressionT]
    allow_complex: bool
    callables_table: Mapping[str, InKernelCallable]
    is_entrypoint: bool
    var_name_generator: UniqueNameGenerator
    is_generating_device_code: bool
    gen_program_name: str
    schedule_index_end: int
    codegen_cachemanager: "CodegenOperationCacheManager"
    vectorization_info: Optional[VectorizationInfo] = None

    def __post_init__(self):
        # FIXME: If this doesn't bomb during testing, we can get rid of target.
        assert self.target == self.kernel.target

        assert self.vectorization_info is None or isinstance(
                self.vectorization_info, VectorizationInfo)

    # {{{ copy helpers

    def copy(self, **kwargs: Any) -> "CodeGenerationState":
        return replace(self, **kwargs)

    def copy_and_assign(
            self, name: str, value: ExpressionT) -> "CodeGenerationState":
        """Make a copy of self with variable *name* fixed to *value*."""
        return self.copy(var_subst_map=self.var_subst_map.set(name, value))

    def copy_and_assign_many(self, assignments) -> "CodeGenerationState":
        """Make a copy of self with *assignments* included."""

        return self.copy(var_subst_map=self.var_subst_map.update(assignments))

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
        :exc:`UnvectorizableError`, it unrolls the vectorized loop instead.

        *func* should return a :class:`GeneratedCode` instance.

        :returns: :class:`GeneratedCode`
        """

        if self.vectorization_info is None:
            return func(self)

        try:
            return func(self)
        except UnvectorizableError as e:
            warn(self.kernel, "vectorize_failed",
                    "Vectorization of '%s' failed because '%s'"
                    % (what, e))

            return self.unvectorize(func)

    def unvectorize(self, func):
        vinf = self.vectorization_info
        assert vinf is not None

        result = []
        novec_self = self.copy(vectorization_info=None)

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


caches.append(code_gen_cache)


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


@dataclass(frozen=True)
class PreambleInfo:
    kernel: LoopKernel
    seen_dtypes: Set[LoopyType]
    seen_functions: Set[SeenFunction]
    seen_atomic_dtypes: Set[LoopyType]

    # FIXME: This makes all the above redundant. It probably shouldn't be here.
    codegen_state: CodeGenerationState


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

    # {{{ examine arg list

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
            implemented_domain=initial_implemented_domain,
            implemented_predicates=frozenset(),
            seen_dtypes=seen_dtypes,
            seen_functions=seen_functions,
            seen_atomic_dtypes=seen_atomic_dtypes,
            var_subst_map=Map(),
            allow_complex=allow_complex,
            var_name_generator=kernel.get_var_name_generator(),
            is_generating_device_code=False,
            gen_program_name=(
                target.host_program_name_prefix
                + kernel.name
                + kernel.target.host_program_name_suffix),
            schedule_index_end=len(kernel.linearization),
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

    for arg in kernel.args:
        seen_dtypes.add(arg.dtype)

    for tv in kernel.temporary_variables.values():
        seen_dtypes.add(tv.dtype)

    if kernel.all_inames():
        seen_dtypes.add(kernel.index_dtype)

    preambles = kernel.preambles + codegen_result.device_preambles

    preamble_info = PreambleInfo(
            kernel=kernel,
            seen_dtypes=seen_dtypes,
            seen_functions=seen_functions,
            # a set of LoopyTypes (!)
            seen_atomic_dtypes=seen_atomic_dtypes,
            codegen_state=codegen_state
            )

    preamble_generators = (list(kernel.preamble_generators)
            + list(target.get_device_ast_builder().preamble_generators()))
    for prea_gen in preamble_generators:
        preambles = preambles + tuple(prea_gen(preamble_info))

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

    return program.copy(callables_table=Map(new_callables))


@dataclass(frozen=True)
class TranslationUnitCodeGenerationResult:
    """
    .. attribute:: host_program

        A mapping from names of entrypoints to their host
        :class:`~loopy.codegen.result.GeneratedProgram`.

    .. attribute:: device_programs

        A list of :class:`~loopy.codegen.result.GeneratedProgram` instances
        intended to run on the compute device.

    .. attribute:: host_preambles
    .. attribute:: device_preambles

    .. automethod:: host_code
    .. automethod:: device_code
    .. automethod:: all_code

    """
    host_programs: Mapping[str, "GeneratedProgram"]
    device_programs: Sequence["GeneratedProgram"]
    host_preambles: Sequence[Tuple[int, str]] = ()
    device_preambles: Sequence[Tuple[int, str]] = ()

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
                tuple(getattr(self, "host_preambles", ()))
                +
                tuple(getattr(self, "device_preambles", ()))
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
            logger.debug(f"TranslationUnit with entrypoints {program.entrypoints}:"
                          " code generation cache miss")

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

    if program.state < KernelState.LINEARIZED:
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

    # {{{ collect host/device programs

    for func_id in sorted(key for key, val in program.callables_table.items()
                          if isinstance(val, CallableKernel)):
        cgr = generate_code_for_a_single_kernel(program[func_id],
                                                program.callables_table,
                                                program.target,
                                                func_id in program.entrypoints)
        if func_id in program.entrypoints:
            host_programs[func_id] = cgr.host_program
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
            device_preambles=device_preambles)

    if CACHING_ENABLED:
        code_gen_cache.store_if_not_present(input_program, cgr)

    return cgr


def generate_code(kernel, device=None):
    if device is not None:
        from warnings import warn
        warn("passing 'device' to generate_code() is deprecated",
                DeprecationWarning, stacklevel=2)

    if device is not None:
        from warnings import warn
        warn("generate_code is deprecated and will stop working in 2023. "
                "Call generate_code_v2 instead.", DeprecationWarning, stacklevel=2)

    codegen_result = generate_code_v2(kernel)

    if len(codegen_result.device_programs) > 1:
        raise LoopyError("kernel passed to generate_code yielded multiple "
                "device programs. Use generate_code_v2.")
    if len(codegen_result.host_programs) > 1:
        raise LoopyError("kernel passed to generate_code yielded multiple "
                "host programs. Use generate_code_v2.")

    return codegen_result.device_code(), None

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

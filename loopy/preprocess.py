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

from typing import Tuple, TypeVar, Iterable, Optional, List, FrozenSet, cast
import logging
logger = logging.getLogger(__name__)

from immutables import Map
import numpy as np

from loopy.diagnostic import (
        LoopyError, WriteRaceConditionWarning, warn_with_kernel,
        LoopyAdvisory)

from loopy.tools import memoize_on_disk
from loopy.kernel.data import filter_iname_tags_by_type, ArrayArg, auto, ValueArg

from loopy.kernel import LoopKernel
# for the benefit of loopy.statistics, for now
from loopy.type_inference import infer_unknown_types
from loopy.symbolic import RuleAwareIdentityMapper
# from loopy.transform.iname import remove_any_newly_unused_inames

from loopy.kernel.instruction import (MultiAssignmentBase, CInstruction,
        CallInstruction,  _DataObliviousInstruction)
from loopy.kernel.function_interface import CallableKernel, ScalarCallable
from loopy.transform.data import allocate_temporaries_for_base_storage
from loopy.kernel.array import ArrayDimImplementationTag
from loopy.kernel.data import _ArraySeparationInfo, KernelArgument
from loopy.translation_unit import TranslationUnit, for_each_kernel
from loopy.typing import ExpressionT

from pytools import ProcessLogger
from functools import partial


# {{{ check for writes to predicates

def check_for_writes_to_predicates(kernel):
    from loopy.symbolic import get_dependencies
    for insn in kernel.instructions:
        pred_vars = (
                frozenset.union(
                    *(get_dependencies(pred) for pred in insn.predicates))
                if insn.predicates else frozenset())
        written_pred_vars = frozenset(insn.assignee_var_names()) & pred_vars
        if written_pred_vars:
            raise LoopyError("In instruction '%s': may not write to "
                    "variable(s) '%s' involved in the instruction's predicates"
                    % (insn.id, ", ".join(written_pred_vars)))

# }}}


# {{{ check reduction iname uniqueness

def check_reduction_iname_uniqueness(kernel):
    iname_to_reduction_count = {}
    iname_to_nonsimultaneous_reduction_count = {}

    def map_reduction(expr, rec):
        rec(expr.expr)

        for iname in expr.inames:
            iname_to_reduction_count[iname] = (
                    iname_to_reduction_count.get(iname, 0) + 1)
            if not expr.allow_simultaneous:
                iname_to_nonsimultaneous_reduction_count[iname] = (
                        iname_to_nonsimultaneous_reduction_count.get(iname, 0) + 1)

        return expr

    from loopy.symbolic import ReductionCallbackMapper
    cb_mapper = ReductionCallbackMapper(map_reduction)

    for insn in kernel.instructions:
        insn.with_transformed_expressions(cb_mapper)

    for iname, count in iname_to_reduction_count.items():
        nonsimul_count = iname_to_nonsimultaneous_reduction_count.get(iname, 0)

        if nonsimul_count and count > 1:
            raise LoopyError("iname '%s' used in more than one reduction. "
                    "(%d of them, to be precise.) "
                    "Since this usage can easily cause loop scheduling "
                    "problems, this is prohibited by default. "
                    "Use loopy.make_reduction_inames_unique() to fix this. "
                    "If you are sure that this is OK, write the reduction "
                    "as 'simul_reduce(...)' instead of 'reduce(...)'"
                    % (iname, count))

# }}}


# {{{ make_arrays_for_sep_arrays

T = TypeVar("T")


def _remove_at_indices(
        indices: FrozenSet[int], values: Optional[Iterable[T]]
        ) -> Optional[Tuple[T, ...]]:
    """
    Assumes *indices* is sorted.
    """
    if values is None:
        return values

    return tuple(val for i, val in enumerate(values) if i not in indices)


@for_each_kernel
def make_arrays_for_sep_arrays(kernel: LoopKernel) -> LoopKernel:
    from loopy.kernel.array import SeparateArrayArrayDimTag
    new_args = []

    vng = kernel.get_var_name_generator()
    made_changes = False

    # {{{ rewrite arguments

    for arg in kernel.args:
        if not isinstance(arg, ArrayArg) or arg.dim_tags is None:
            new_args.append(arg)
            continue

        sep_axis_indices = [
                i for i, dim_tag in enumerate(arg.dim_tags)
                if isinstance(dim_tag, SeparateArrayArrayDimTag)]

        if not sep_axis_indices or arg._separation_info:
            new_args.append(arg)
            continue

        made_changes = True

        sep_axis_indices_set = frozenset(sep_axis_indices)

        assert isinstance(arg.shape, tuple)
        new_shape: Optional[Tuple[ExpressionT, ...]] = \
                _remove_at_indices(sep_axis_indices_set, arg.shape)
        new_dim_tags: Optional[Tuple[ArrayDimImplementationTag, ...]] = \
                _remove_at_indices(sep_axis_indices_set, arg.dim_tags)
        new_dim_names: Optional[Tuple[Optional[str], ...]] = \
                _remove_at_indices(sep_axis_indices_set, arg.dim_names)

        sep_shape: List[ExpressionT] = [arg.shape[i] for i in sep_axis_indices]
        for i, sep_shape_i in enumerate(sep_shape):
            if not isinstance(sep_shape_i, (int, np.integer)):
                raise LoopyError(
                        f"Axis {sep_axis_indices[i]+1} (1-based) of "
                        f"argument '{arg.name}' is tagged 'sep', but "
                        "does not have constant length.")

        sep_info = _ArraySeparationInfo(
                sep_axis_indices_set=sep_axis_indices_set,
                subarray_names=Map({
                    ind: vng(f"{arg.name}_s{'_'.join(str(i) for i in ind)}")
                    for ind in np.ndindex(*cast(List[int], sep_shape))}))

        new_args.append(arg.copy(_separation_info=sep_info))

        for _ind, san in sorted(sep_info.subarray_names.items()):
            new_args.append(
                    arg.copy(
                        name=san,
                        shape=new_shape,
                        dim_tags=new_dim_tags,
                        dim_names=new_dim_names))

    # }}}

    if not made_changes:
        return kernel

    kernel = kernel.copy(args=new_args)

    return kernel

# }}}


# {{{ make temporary variables for offsets and strides

def make_args_for_offsets_and_strides(kernel: LoopKernel) -> LoopKernel:
    additional_args: List[KernelArgument] = []

    vng = kernel.get_var_name_generator()

    from pymbolic.primitives import Expression, Variable
    from loopy.kernel.array import FixedStrideArrayDimTag

    # {{{ process arguments

    new_args = []
    for arg in kernel.args:
        if isinstance(arg, ArrayArg) and not arg._separation_info:
            what = f"offset for argument '{arg.name}'"
            if arg.offset is None:
                pass
            if arg.offset is auto:
                offset_name = vng(arg.name+"_offset")
                additional_args.append(ValueArg(
                        offset_name, kernel.index_dtype))
                arg = arg.copy(offset=offset_name)
            elif isinstance(arg.offset, (int, np.integer, Expression, str)):
                pass
            else:
                raise LoopyError(f"invalid value of {what}")

            if arg.dim_tags is None:
                new_dim_tags: Optional[Tuple[ArrayDimImplementationTag, ...]]  \
                        = arg.dim_tags
            else:
                new_dim_tags = ()
                for iaxis, dim_tag in enumerate(arg.dim_tags):
                    if isinstance(dim_tag, FixedStrideArrayDimTag):
                        what = ("axis stride for axis "
                                f"{iaxis+1} (1-based) of '{arg.name}'")
                        if dim_tag.stride is auto:
                            stride_name = vng(f"{arg.name}_stride{iaxis}")
                            dim_tag = dim_tag.copy(stride=Variable(stride_name))
                            additional_args.append(ValueArg(
                                    stride_name, kernel.index_dtype))
                        elif isinstance(
                                dim_tag.stride, (int, np.integer, Expression)):
                            pass
                        else:
                            raise LoopyError(f"invalid value of {what}")

                    new_dim_tags = new_dim_tags + (dim_tag,)

            arg = arg.copy(dim_tags=new_dim_tags)

        new_args.append(arg)

    # }}}

    if not additional_args:
        return kernel
    else:
        return kernel.copy(args=new_args + additional_args)

# }}}


# {{{ zero_offsets

def zero_offsets_and_strides(kernel: LoopKernel) -> LoopKernel:
    made_changes = False
    from pymbolic.primitives import Expression

    # {{{ process arguments

    new_args = []
    for arg in kernel.args:
        if isinstance(arg, ArrayArg):
            if arg.offset is None:
                pass
            if arg.offset is auto:
                made_changes = True
                arg = arg.copy(offset=0)
            elif isinstance(arg.offset, (int, np.integer, Expression, str)):
                from pymbolic.primitives import is_zero
                if not is_zero(arg.offset):
                    raise LoopyError(
                        f"Non-zero offset on argument '{arg.name}' "
                        f"of callable kernel '{kernel.name}. This is not allowed.")
            else:
                raise LoopyError(f"invalid value of offset for '{arg.name}'")

        new_args.append(arg)

    # }}}

    if not made_changes:
        return kernel
    else:
        return kernel.copy(args=new_args)

# }}}


# {{{ decide temporary address space

def _get_compute_inames_tagged(kernel, insn, tag_base):
    return {iname for iname in kernel.insn_inames(insn.id)
               if kernel.iname_tags_of_type(iname, tag_base)}


def _get_assignee_inames_tagged(kernel, insn, tag_base, tv_names):
    return {iname
            for aname, adeps in zip(
                insn.assignee_var_names(),
                insn.assignee_subscript_deps())
            for iname in adeps & kernel.all_inames()
            if aname in tv_names
            if kernel.iname_tags_of_type(iname, tag_base)}


def find_temporary_address_space(kernel):
    logger.debug("%s: find temporary address space" % kernel.name)

    new_temp_vars = {}
    from loopy.kernel.data import (LocalInameTagBase, GroupInameTag,
            AddressSpace)
    import loopy as lp

    writers = kernel.writer_map()

    base_storage_to_aliases = {}

    for temp_var in kernel.temporary_variables.values():
        if temp_var.base_storage is not None:
            base_storage_to_aliases.setdefault(
                    temp_var.base_storage, []).append(temp_var.name)

    for temp_var in kernel.temporary_variables.values():
        # Only fill out for variables that do not yet know if they're
        # local. (I.e. those generated by implicit temporary generation.)

        if temp_var.address_space is not lp.auto:
            new_temp_vars[temp_var.name] = temp_var
            continue

        tv_names = (frozenset([temp_var.name])
                | frozenset(base_storage_to_aliases.get(temp_var.base_storage, [])))
        my_writers = writers.get(temp_var.name, frozenset())
        if temp_var.base_storage is not None:
            for alias in base_storage_to_aliases.get(temp_var.base_storage, []):
                my_writers = my_writers | writers.get(alias, frozenset())

        desired_aspace_per_insn = []
        for insn_id in my_writers:
            insn = kernel.id_to_insn[insn_id]

            # A write race will emerge if:
            #
            # - the variable is local
            #   and
            # - the instruction is run across more inames (locally) parallel
            #   than are reflected in the assignee indices.

            locparallel_compute_inames = _get_compute_inames_tagged(
                    kernel, insn, LocalInameTagBase)

            locparallel_assignee_inames = _get_assignee_inames_tagged(
                    kernel, insn, LocalInameTagBase, tv_names)

            grpparallel_compute_inames = _get_compute_inames_tagged(
                    kernel, insn, GroupInameTag)

            grpparallel_assignee_inames = _get_assignee_inames_tagged(
                    kernel, insn, GroupInameTag, temp_var.name)

            assert locparallel_assignee_inames <= locparallel_compute_inames
            assert grpparallel_assignee_inames <= grpparallel_compute_inames

            desired_aspace = AddressSpace.PRIVATE
            for iname_descr, aspace_descr, apin, cpin, aspace in [
                    ("local", "local", locparallel_assignee_inames,
                        locparallel_compute_inames, AddressSpace.LOCAL),
                    ("group", "global", grpparallel_assignee_inames,
                        grpparallel_compute_inames, AddressSpace.GLOBAL),
                    ]:

                if (apin != cpin and bool(apin)):
                    warn_with_kernel(
                            kernel,
                            f"write_race_{aspace_descr}({insn_id})",
                            "instruction '%s' looks invalid: "
                            "it assigns to indices based on %s IDs, but "
                            "its temporary '%s' cannot be made %s because "
                            "a write race across the iname(s) '%s' would emerge. "
                            "(Do you need to add an extra iname to your prefetch?)"
                            % (insn_id, iname_descr, temp_var.name, aspace_descr,
                                ", ".join(cpin - apin)),
                            WriteRaceConditionWarning)

                if (apin == cpin
                        # doesn't want to be in this address space if there
                        # aren't any parallel inames of that kind
                        and bool(cpin)):
                    desired_aspace = max(desired_aspace, aspace)

            desired_aspace_per_insn.append(desired_aspace)

        if not desired_aspace_per_insn:
            warn_with_kernel(kernel, "temp_to_write(%s)" % temp_var.name,
                    "cannot automatically determine address space of '%s'"
                    % temp_var.name, LoopyAdvisory)

            new_temp_vars[temp_var.name] = temp_var
            continue

        overall_aspace = max(desired_aspace_per_insn)

        if not all(iaspace == overall_aspace for iaspace in desired_aspace_per_insn):
            raise LoopyError("not all instructions agree on the "
                    "the desired address space (private/local/global) of  the "
                    "temporary '%s'" % temp_var.name)

        new_temp_vars[temp_var.name] = temp_var.copy(address_space=overall_aspace)

    return kernel.copy(temporary_variables=new_temp_vars)

# }}}


# {{{ realize_ilp

def realize_ilp(kernel):
    logger.debug("%s: add axes to temporaries for ilp" % kernel.name)

    from loopy.kernel.data import (IlpBaseTag, VectorizeTag,
                                   filter_iname_tags_by_type)

    privatizing_inames = frozenset(
        name for name, iname in kernel.inames.items()
        if filter_iname_tags_by_type(iname.tags, (IlpBaseTag, VectorizeTag))
    )

    if not privatizing_inames:
        return kernel

    from loopy.transform.privatize import privatize_temporaries_with_inames
    return privatize_temporaries_with_inames(kernel, privatizing_inames)

# }}}


# {{{ check for loads of atomic variables

def check_atomic_loads(kernel):
    """Find instances of AtomicInit or AtomicUpdate with use of other atomic
    variables to update the atomicity
    """

    logger.debug("%s: check atomic loads" % kernel.name)
    from loopy.types import AtomicType
    from loopy.kernel.array import ArrayBase
    from loopy.kernel.instruction import Assignment, AtomicLoad

    # find atomic variables
    atomicity_candidates = (
            {v.name for v in kernel.temporary_variables.values()
                if isinstance(v.dtype, AtomicType)}
            |
            {v.name for v in kernel.args
                if isinstance(v, ArrayBase)
                and isinstance(v.dtype, AtomicType)})

    new_insns = []
    for insn in kernel.instructions:
        if isinstance(insn, Assignment):
            # look for atomic variables
            atomic_accesses = {a.var_name for a in insn.atomicity}
            accessed_atomic_vars = (insn.dependency_names() & atomicity_candidates)\
                - {insn.assignee_var_names()[0]}
            if not accessed_atomic_vars <= atomic_accesses:
                #if we're missing some
                missed = accessed_atomic_vars - atomic_accesses
                for x in missed:
                    if {x} & atomicity_candidates:
                        insn = insn.copy(
                            atomicity=insn.atomicity + (AtomicLoad(x),))

        new_insns.append(insn)

    return kernel.copy(instructions=new_insns)

# }}}


# {{{ arg_descr_inference

class ArgDescrInferenceMapper(RuleAwareIdentityMapper):
    """
    Infers :attr:`~loopy.kernel.function_interface.arg_id_to_descr` of
    callables visited in an expression.
    """

    def __init__(self, rule_mapping_context, caller_kernel, clbl_inf_ctx):
        super().__init__(rule_mapping_context)
        self.caller_kernel = caller_kernel
        self.clbl_inf_ctx = clbl_inf_ctx

    def map_call(self, expr, expn_state, assignees=None):
        from pymbolic.primitives import Call, Variable
        from loopy.kernel.function_interface import ValueArgDescriptor
        from loopy.symbolic import ResolvedFunction
        from loopy.kernel.array import ArrayBase
        from loopy.kernel.data import ValueArg
        from pymbolic.mapper.substitutor import make_subst_func
        from loopy.symbolic import SubstitutionMapper
        from loopy.kernel.function_interface import get_arg_descriptor_for_expression

        if not isinstance(expr.function, ResolvedFunction):
            # ignore if the call is not to a ResolvedFunction
            return super().map_call(expr, expn_state)

        arg_id_to_arg = dict(enumerate(expr.parameters))

        if assignees is not None:
            # If supplied with assignees then this is a CallInstruction
            for i, arg in enumerate(assignees):
                arg_id_to_arg[-i-1] = arg

        arg_id_to_descr = {
            arg_id: get_arg_descriptor_for_expression(self.caller_kernel, arg)
            for arg_id, arg in arg_id_to_arg.items()}
        clbl = self.clbl_inf_ctx[expr.function.name]

        # {{{ translating descriptor expressions to the callable's namespace

        deps_as_params = []
        subst_map = {}

        deps = frozenset().union(*(descr.depends_on()
                                   for descr in arg_id_to_descr.values()))

        assert deps <= self.caller_kernel.all_variable_names()

        for dep in deps:
            caller_arg = self.caller_kernel.arg_dict.get(dep, (self.caller_kernel
                                                               .temporary_variables
                                                               .get(dep)))
            if not (isinstance(caller_arg, ValueArg)
                    or (isinstance(caller_arg, ArrayBase)
                        and caller_arg.shape == ())):
                raise NotImplementedError(f"Obtained '{dep}' as a dependency for"
                        f" call '{expr.function.name}' which is not a scalar.")

            clbl, callee_name = clbl.with_added_arg(caller_arg.dtype,
                                                    ValueArgDescriptor())

            subst_map[dep] = Variable(callee_name)
            deps_as_params.append(Variable(dep))

        mapper = SubstitutionMapper(make_subst_func(subst_map))
        arg_id_to_descr = {id_: descr.map_expr(mapper)
                           for id_, descr in arg_id_to_descr.items()}

        # }}}

        # specializing the function according to the parameter description
        new_clbl, self.clbl_inf_ctx = clbl.with_descrs(arg_id_to_descr,
                                                       self.clbl_inf_ctx)

        self.clbl_inf_ctx, new_func_id = (self.clbl_inf_ctx
                                          .with_callable(expr.function.function,
                                                         new_clbl))

        return Call(ResolvedFunction(new_func_id),
                    tuple(self.rec(child, expn_state)
                          for child in expr.parameters)
                    + tuple(deps_as_params))

    def map_call_with_kwargs(self, expr):
        # See https://github.com/inducer/loopy/pull/323
        raise NotImplementedError

    def __call__(self, expr, kernel, insn, assignees=None):
        from loopy.kernel.data import InstructionBase
        from loopy.symbolic import UncachedIdentityMapper, ExpansionState
        import immutables
        assert insn is None or isinstance(insn, InstructionBase)

        return UncachedIdentityMapper.__call__(self, expr,
                ExpansionState(
                    kernel=kernel,
                    instruction=insn,
                    stack=(),
                    arg_context=immutables.Map()), assignees=assignees)

    def map_kernel(self, kernel):

        new_insns = []

        for insn in kernel.instructions:
            if isinstance(insn, CallInstruction):
                # In call instructions the assignees play an important in
                # determining the arg_id_to_descr
                mapper = partial(self, kernel=kernel, insn=insn,
                        assignees=insn.assignees)
                new_insns.append(insn.with_transformed_expressions(mapper))
            elif isinstance(insn, MultiAssignmentBase):
                mapper = partial(self, kernel=kernel, insn=insn)
                new_insns.append(insn.with_transformed_expressions(mapper))
            elif isinstance(insn, (_DataObliviousInstruction, CInstruction)):
                new_insns.append(insn)
            else:
                raise NotImplementedError("arg_descr_inference for %s instruction" %
                        type(insn))

        return kernel.copy(instructions=new_insns)


def traverse_to_infer_arg_descr(kernel, callables_table):
    """
    Returns a copy of *kernel* with the argument shapes and strides matching for
    resolved functions in the *kernel*. Refer
    :meth:`loopy.kernel.function_interface.InKernelCallable.with_descrs`.

    .. note::

        Initiates a walk starting from *kernel* to all its callee kernels.
    """
    from loopy.symbolic import SubstitutionRuleMappingContext

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())

    arg_descr_inf_mapper = ArgDescrInferenceMapper(rule_mapping_context,
            kernel, callables_table)

    descr_inferred_kernel = rule_mapping_context.finish_kernel(
            arg_descr_inf_mapper.map_kernel(kernel))

    return descr_inferred_kernel, arg_descr_inf_mapper.clbl_inf_ctx


def infer_arg_descr(program):
    """
    Returns a copy of *program* with the
    :attr:`loopy.InKernelCallable.arg_id_to_descr` inferred for all the
    callables.
    """
    from loopy.translation_unit import make_clbl_inf_ctx, resolve_callables
    from loopy.kernel.array import ArrayBase
    from loopy.kernel.function_interface import (ArrayArgDescriptor,
            ValueArgDescriptor)
    from loopy import auto, ValueArg

    program = resolve_callables(program)

    clbl_inf_ctx = make_clbl_inf_ctx(program.callables_table,
                                     program.entrypoints)

    for e in program.entrypoints:
        def _tuple_or_None(s):
            if isinstance(s, tuple):
                return s
            elif s in [None, auto]:
                return s
            else:
                return s,

        arg_id_to_descr = {}
        for arg in program[e].args:
            if isinstance(arg, ArrayBase):
                if arg.shape not in (None, auto):
                    arg_id_to_descr[arg.name] = ArrayArgDescriptor(
                            _tuple_or_None(arg.shape), arg.address_space,
                            arg.dim_tags)
            elif isinstance(arg, ValueArg):
                arg_id_to_descr[arg.name] = ValueArgDescriptor()
            else:
                raise NotImplementedError()
        new_callable, clbl_inf_ctx = program.callables_table[e].with_descrs(
                arg_id_to_descr, clbl_inf_ctx)
        clbl_inf_ctx, new_name = clbl_inf_ctx.with_callable(e, new_callable,
                                                            is_entrypoint=True)

    return clbl_inf_ctx.finish_program(program)

# }}}


# {{{  inline_kernels_with_gbarriers

def inline_kernels_with_gbarriers(program):
    from loopy.kernel.instruction import BarrierInstruction
    from loopy.transform.callable import inline_callable_kernel
    from loopy.kernel.tools import get_call_graph
    from pytools.graph import compute_topological_order

    def has_gbarrier(knl):
        return any((isinstance(insn, BarrierInstruction)
                    and insn.synchronization_kind == "global")
                   for insn in knl.instructions)

    call_graph = get_call_graph(program, only_kernel_callables=True)

    # traverse the kernel calls in a reverse topological sort so that barriers
    # are rightly passed to the entrypoints.
    toposort = compute_topological_order(call_graph,
                                         # pass key to have deterministic codegen
                                         key=lambda x: x
                                         )

    for name in toposort[::-1]:
        if has_gbarrier(program[name]):
            program = inline_callable_kernel(program, name)

    return program

# }}}


def filter_reachable_callables(t_unit):
    from loopy.translation_unit import get_reachable_resolved_callable_ids
    reachable_function_ids = get_reachable_resolved_callable_ids(t_unit
                                                                 .callables_table,
                                                                 t_unit.entrypoints)
    new_callables = {name: clbl for name, clbl in t_unit.callables_table.items()
                     if name in (reachable_function_ids | t_unit.entrypoints)}
    return t_unit.copy(callables_table=Map(new_callables))


def _preprocess_single_kernel(kernel: LoopKernel, is_entrypoint: bool) -> LoopKernel:
    from loopy.kernel import KernelState

    prepro_logger = ProcessLogger(logger, "%s: preprocess" % kernel.name)

    kernel = make_arrays_for_sep_arrays(kernel)

    if is_entrypoint:
        kernel = make_args_for_offsets_and_strides(kernel)
    else:
        # No need for offsets internally, we can pass arbitrary pointers.
        kernel = zero_offsets_and_strides(kernel)

    from loopy.check import check_identifiers_in_subst_rules
    check_identifiers_in_subst_rules(kernel)

    # {{{ check that there are no l.auto-tagged inames

    from loopy.kernel.data import AutoLocalInameTagBase
    for name, iname in kernel.inames.items():
        if (filter_iname_tags_by_type(iname.tags, AutoLocalInameTagBase)
                 and name in kernel.all_inames()):
            raise LoopyError("kernel with automatically-assigned "
                    "local axes passed to preprocessing")

    # }}}

    # Ordering restriction:
    # Type inference and reduction iname uniqueness don't handle substitutions.
    # Get them out of the way.

    check_for_writes_to_predicates(kernel)
    check_reduction_iname_uniqueness(kernel)

    # Ordering restriction:
    # add_axes_to_temporaries_for_ilp because reduction accumulators
    # need to be duplicated by this.

    kernel = realize_ilp(kernel)

    kernel = find_temporary_address_space(kernel)

    # Ordering restriction: temporary address spaces need to be found before
    # allocating base_storage
    kernel = allocate_temporaries_for_base_storage(kernel, _implicitly_run=True)

    # check for atomic loads, much easier to do here now that the dependencies
    # have been established
    kernel = check_atomic_loads(kernel)

    kernel = kernel.target.preprocess(kernel)

    kernel = kernel.copy(
            state=KernelState.PREPROCESSED)

    prepro_logger.done()

    return kernel


@memoize_on_disk
def preprocess_program(t_unit: TranslationUnit) -> TranslationUnit:

    from loopy.kernel import KernelState
    if t_unit.state >= KernelState.PREPROCESSED:
        return t_unit

    if len([clbl for clbl in t_unit.callables_table.values() if
            isinstance(clbl, CallableKernel)]) == 1:
        t_unit = t_unit.with_entrypoints(",".join(clbl.name for clbl in
            t_unit.callables_table.values() if isinstance(clbl,
                CallableKernel)))

    if not t_unit.entrypoints:
        raise LoopyError("Translation unit did not receive any entrypoints")

    from loopy.translation_unit import resolve_callables
    t_unit = resolve_callables(t_unit)

    t_unit = filter_reachable_callables(t_unit)

    t_unit = infer_unknown_types(t_unit, expect_completion=False)

    from loopy.transform.subst import expand_subst
    t_unit = expand_subst(t_unit)

    from loopy.kernel.creation import apply_single_writer_depencency_heuristic
    t_unit = apply_single_writer_depencency_heuristic(t_unit)

    # Ordering restrictions:
    #
    # - realize_reduction must happen after type inference because it needs
    #   to be able to determine the types of the reduced expressions.
    #
    # - realize_reduction must happen after default dependencies are added
    #   because it manipulates the depends_on field, which could prevent
    #   defaults from being applied.

    from loopy.transform.realize_reduction import realize_reduction
    t_unit = realize_reduction(t_unit, unknown_types_ok=False)

    # {{{ preprocess callable kernels

    # Callable editing restrictions:
    #
    # - should not edit callables_table in :meth:`preprocess_single_kernel`
    #   as we are iterating over it.[1]
    #
    # [1] https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects

    new_callables = {}
    for func_id, in_knl_callable in t_unit.callables_table.items():
        if isinstance(in_knl_callable, CallableKernel):
            new_subkernel = _preprocess_single_kernel(
                    in_knl_callable.subkernel,
                    is_entrypoint=func_id in t_unit.entrypoints)
            in_knl_callable = in_knl_callable.copy(
                    subkernel=new_subkernel)
        elif isinstance(in_knl_callable, ScalarCallable):
            pass
        else:
            raise NotImplementedError("Unknown callable type %s." % (
                type(in_knl_callable).__name__))

        new_callables[func_id] = in_knl_callable

    t_unit = t_unit.copy(callables_table=Map(new_callables))

    # }}}

    # infer arg descrs of the callables
    t_unit = infer_arg_descr(t_unit)

    # Ordering restriction:
    # callees with gbarrier in them must be inlined after inferrring arg_descr.
    t_unit = inline_kernels_with_gbarriers(t_unit)

    # {{{ prepare for caching

    # PicklableDtype instances for example need to know the target they're working
    # towards in order to pickle and unpickle them. This is the first pass that
    # uses caching, so we need to be ready to pickle. This means propagating
    # this target information.

    # }}}

    return t_unit


# FIXME: Do we add a deprecation warning?
preprocess_kernel = preprocess_program


# vim: foldmethod=marker

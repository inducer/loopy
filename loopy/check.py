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

from typing import Union, Tuple, Optional, List

import numpy as np
from islpy import dim_type
import islpy as isl

from loopy.symbolic import WalkMapper, CombineMapper, ResolvedFunction
from loopy.diagnostic import (LoopyError, WriteRaceConditionWarning,
        warn_with_kernel, LoopyIndexError)
from loopy.type_inference import TypeReader
from loopy.kernel.instruction import (MultiAssignmentBase, CallInstruction,
                                      CInstruction, _DataObliviousInstruction,
                                      NoOpInstruction)
from loopy.kernel import LoopKernel
from loopy.kernel.array import (
        FixedStrideArrayDimTag, SeparateArrayArrayDimTag, ArrayBase)
from loopy.kernel.data import auto, ArrayArg, ArrayDimImplementationTag
from loopy.translation_unit import for_each_kernel
from loopy.typing import ExpressionT

from pytools import memoize_method

from collections import defaultdict

from functools import reduce

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. currentmodule:: loopy.check

.. autofunction:: check_for_integer_subscript_indices

.. autofunction:: check_for_duplicate_insn_ids

.. autofunction:: check_for_double_use_of_hw_axes

.. autofunction:: check_insn_attributes

.. autofunction:: check_loop_priority_inames_known

.. autofunction:: check_multiple_tags_allowed

.. autofunction:: check_for_inactive_iname_access

.. autofunction:: check_for_unused_inames

.. autofunction:: check_for_write_races

.. autofunction:: check_for_data_dependent_parallel_bounds

.. autofunction:: check_bounds

.. autofunction:: check_variable_access_ordered
"""


# {{{ sanity checks run before preprocessing

def check_identifiers_in_subst_rules(knl):
    """Substitution rules may only refer to kernel-global quantities or their
    own arguments.
    """

    from loopy.symbolic import get_dependencies

    allowed_identifiers = knl.all_variable_names()

    for rule in knl.substitutions.values():
        deps = get_dependencies(rule.expression)
        rule_allowed_identifiers = allowed_identifiers | frozenset(rule.arguments)

        if not deps <= rule_allowed_identifiers:
            raise LoopyError("kernel '%s': substitution rule '%s' refers to "
                    "identifier(s) '%s' which are neither rule arguments nor "
                    "kernel-global identifiers"
                    % (knl.name, rule.name,
                       ", ".join(deps-rule_allowed_identifiers)))


class UnresolvedCallCollector(CombineMapper):
    """
    Collects all the unresolved calls within a kernel.

    :returns:
        A :class:`frozenset` of function names that are not resolved.
    """

    def combine(self, values):
        import operator
        return reduce(operator.or_, values, frozenset())

    def map_call(self, expr):
        if not isinstance(expr.function, ResolvedFunction):
            return frozenset([expr.function.name]) | self.rec(expr.parameters)
        else:
            return self.rec(expr.parameters)

    def map_call_with_kwargs(self, expr):
        # See: https://github.com/inducer/loopy/pull/323
        raise NotImplementedError

    def map_constant(self, expr):
        return frozenset()

    map_variable = map_constant
    map_function_symbol = map_constant
    map_tagged_variable = map_constant
    map_type_cast = map_constant
    map_nan = map_constant


@for_each_kernel
def check_functions_are_resolved(kernel):
    """ Checks if all call nodes in the *kernel* expression have been
    resolved.
    """
    from loopy.symbolic import SubstitutionRuleExpander
    subst_expander = SubstitutionRuleExpander(kernel.substitutions)

    for insn in kernel.instructions:
        if isinstance(insn, MultiAssignmentBase):
            unresolved_calls = UnresolvedCallCollector()(subst_expander(insn
                                                                        .expression))
            if unresolved_calls:
                raise LoopyError("Unknown function '%s' -- register a "
                                 "callable corresponding to it." %
                                 set(unresolved_calls).pop())
        elif isinstance(insn, (CInstruction, _DataObliviousInstruction)):
            pass
        else:
            raise NotImplementedError(type(insn))


@for_each_kernel
def check_separated_array_consistency(kernel: LoopKernel) -> None:
    # Boo. This is (part of) the price of redundant representation.
    for arg in kernel.args:
        if isinstance(arg, ArrayArg) and arg._separation_info is not None:
            sep_indices = arg._separation_info.sep_axis_indices_set
            for subarg_name in arg._separation_info.subarray_names.values():
                sub_arg = kernel.arg_dict[subarg_name]

                from loopy.preprocess import _remove_at_indices

                assert arg.shape is None or isinstance(arg.shape, tuple)
                if _remove_at_indices(sep_indices, arg.shape) != sub_arg.shape:
                    raise LoopyError(
                            f"Shapes of '{arg.name}' and associated sep array "
                            "'{sub_arg.name}' are not consistent.")

                assert arg.dim_tags is None or isinstance(arg.dim_tags, tuple)
                if _remove_at_indices(sep_indices, arg.dim_tags) != sub_arg.dim_tags:
                    raise LoopyError(
                            f"Dim tags of '{arg.name}' and associated sep array "
                            "'{sub_arg.name}' are not consistent.")

                for attr_name in ["address_space", "is_input", "is_output"]:
                    if getattr(arg, attr_name) != getattr(sub_arg, attr_name):
                        raise LoopyError(
                                "Attribute '{attr_name}' of "
                                f"'{arg.name}' and associated sep array "
                                f"'{sub_arg.name}' is not consistent.")


@for_each_kernel
def check_offsets_and_dim_tags(kernel: LoopKernel) -> None:
    from loopy.symbolic import DependencyMapper
    from pymbolic.primitives import Variable, Expression

    arg_name_vars = {Variable(name) for name in kernel.arg_dict}
    dep_mapper = DependencyMapper()

    def ensure_depends_only_on_arguments(
            what: str, expr: Union[str, ExpressionT]) -> None:
        if isinstance(expr, str):
            expr = Variable(expr)

        deps = dep_mapper(expr)
        if not deps <= arg_name_vars:
            raise LoopyError(
                    f"not all names in {what} are arguments: "
                    + ", ".join(str(v) for v in deps - arg_name_vars))

    # {{{ process arguments

    new_args = []
    for arg in kernel.args:
        if isinstance(arg, ArrayArg):
            what = f"offset of argument '{arg.name}'"
            if arg.offset is None:
                continue
            if arg.offset is auto:
                pass
            elif isinstance(arg.offset, (int, np.integer, Expression, str)):
                ensure_depends_only_on_arguments(what, arg.offset)

            else:
                raise LoopyError(f"invalid value of offset for '{arg.name}'")

            if arg.dim_tags is None:
                new_dim_tags: Optional[Tuple[ArrayDimImplementationTag, ...]] = \
                        arg.dim_tags
            else:
                new_dim_tags = ()
                for iaxis, dim_tag in enumerate(arg.dim_tags):
                    if isinstance(dim_tag, FixedStrideArrayDimTag):
                        what = (f"stride for axis {iaxis+1} (1-based) of "
                                        f"of argument '{arg.name}'")
                        if dim_tag.stride is auto:
                            pass
                        elif isinstance(
                                dim_tag.stride, (int, np.integer, Expression)):
                            ensure_depends_only_on_arguments(what, dim_tag.stride)
                        else:
                            raise LoopyError(f"invalid value of {what}")

                    assert new_dim_tags is not None
                    new_dim_tags = new_dim_tags + (dim_tag,)

            arg = arg.copy(dim_tags=new_dim_tags)

        new_args.append(arg)

    # }}}

    # {{{ process temporary variables

    for tv in kernel.temporary_variables.values():
        what = f"offset of temporary '{tv.name}'"
        if tv.offset is None:
            pass
        if tv.offset is auto:
            pass
        elif isinstance(tv.offset, (int, np.integer, Expression, str)):
            ensure_depends_only_on_arguments(what, tv.offset)
        else:
            raise LoopyError(f"invalid value of offset for '{tv.name}'")

        if tv.dim_tags is not None:
            for iaxis, dim_tag in enumerate(tv.dim_tags):
                if isinstance(dim_tag, FixedStrideArrayDimTag):
                    what = ("axis stride for axis "
                            f"{iaxis+1} (1-based) of temporary '{tv.name}'")
                    if dim_tag.stride is auto:
                        raise LoopyError(f"The {what}" f" is 'auto', "
                                "which is not allowed.")
                    elif isinstance(dim_tag.stride, (int, np.integer, Expression)):
                        ensure_depends_only_on_arguments(what, dim_tag.stride)
                    else:
                        raise LoopyError(f"invalid value of {what}")

                elif isinstance(dim_tag, SeparateArrayArrayDimTag):
                    raise LoopyError(f"Axis {iaxis+1} of temporary "
                            f"'{tv.name} is tagged 'sep'. This is not allowed.")

    # }}}

# }}}


# {{{ sanity checks run pre-scheduling

# FIXME: Replace with an enum. See
# https://gitlab.tiker.net/inducer/loopy/issues/85
VALID_NOSYNC_SCOPES = frozenset(["local", "global", "any"])


class SubscriptIndicesIsIntChecker(TypeReader):
    def map_subscript(self, expr):
        for idx in expr.index_tuple:
            type_inf_result = self.rec(idx)
            if not type_inf_result:
                raise LoopyError(
                        "When checking that subscript indices are integral: "
                        "Type inference did not find type of '%s'"
                        % idx)
            if not type_inf_result[0].is_integral():
                raise LoopyError("Non-integral array indices obtained in"
                        " {}.".format(expr))

        return self.rec(expr.aggregate)


def _check_for_integer_subscript_indices_inner(kernel, callables_table):

    from loopy.kernel.data import auto
    if any(arg.dtype in [None, auto] for arg in kernel.args) or (
            any(tv.dtype in [None, auto]
                for tv in kernel.temporary_variables.values())):
        # some types are not resolved => do not check.
        return

    from pymbolic.primitives import Subscript
    idx_int_checker = SubscriptIndicesIsIntChecker(kernel, callables_table)
    for insn in kernel.instructions:
        if isinstance(insn, MultiAssignmentBase):
            idx_int_checker(insn.expression, return_tuple=isinstance(insn,
                CallInstruction), return_dtype_set=True)
            [idx_int_checker(assignee) for assignee in insn.assignees if
                    isinstance(assignee, Subscript)]
        elif isinstance(insn, (CInstruction, _DataObliviousInstruction)):
            pass
        else:
            raise NotImplementedError("Unknown insn type %s." % (
                type(insn).__name__))


def check_for_integer_subscript_indices(t_unit):
    """
    Checks if every array access is of type :class:`int`.
    """
    from loopy.kernel.function_interface import (CallableKernel,
                                                 ScalarCallable)
    for clbl in t_unit.callables_table.values():
        if isinstance(clbl, CallableKernel):
            _check_for_integer_subscript_indices_inner(clbl.subkernel,
                                                       t_unit.callables_table)
        elif isinstance(clbl, ScalarCallable):
            pass
        else:
            raise NotImplementedError(type(clbl).__name__)


@for_each_kernel
def check_sub_array_ref_inames_not_within_or_redn_inames(kernel):
    all_within_inames = frozenset().union(*(insn.within_inames
                                            for insn in kernel.instructions))
    all_redn_inames = frozenset().union(*(insn.reduction_inames()
                                          for insn in kernel.instructions))
    all_sar_inames = frozenset().union(*(insn.sub_array_ref_inames()
                                         for insn in kernel.instructions))

    if all_sar_inames & all_within_inames:
        sample = next(iter(all_sar_inames & all_within_inames))
        raise LoopyError(f"Iname '{sample}' used as a sub-array ref's sweep"
                         " iname and an instruction's within inames. Such usage"
                         " is illegal.")

    if all_sar_inames & all_redn_inames:
        sample = next(iter(all_sar_inames & all_within_inames))
        raise LoopyError(f"Iname '{sample}' used as a sub-array ref's sweep"
                         " iname and a reduction iname. Such usage is"
                         " illegal.")


@for_each_kernel
def check_insn_attributes(kernel):
    """
    Check for legality of attributes of every instruction in *kernel*.
    """
    all_insn_ids = {insn.id for insn in kernel.instructions}

    for insn in kernel.instructions:
        if not insn.within_inames <= kernel.all_inames():
            raise LoopyError("insn '%s' has unknown forced iname "
                    "dependencies: %s"
                    % (insn.id, ", ".join(
                        insn.within_inames - kernel.all_inames())))

        if insn.depends_on is not None and not insn.depends_on <= all_insn_ids:
            raise LoopyError("insn '%s' has unknown instruction "
                    "dependencies: %s"
                    % (insn.id, ", ".join(
                        insn.depends_on - all_insn_ids)))

        no_sync_with_insn_ids = {id for id, scope in insn.no_sync_with}
        if not no_sync_with_insn_ids <= all_insn_ids:
            raise LoopyError("insn '%s' has nosync directive with unknown "
                    "instruction ids: %s"
                    % (insn.id,
                       ", ".join(no_sync_with_insn_ids - all_insn_ids)))

        no_sync_with_scopes = {scope for id, scope in insn.no_sync_with}
        if not no_sync_with_scopes <= VALID_NOSYNC_SCOPES:
            raise LoopyError("insn '%s' has invalid nosync scopes: %s"
                    % (insn.id,
                       ", ".join(no_sync_with_scopes - VALID_NOSYNC_SCOPES)))


@for_each_kernel
def check_for_duplicate_insn_ids(knl):
    """
    Check if multiple instructions of *knl* have the same
    :attr:`loopy.InstructionBase.id`.
    """
    insn_ids = set()

    for insn in knl.instructions:
        if not isinstance(insn.id, str):
            raise LoopyError("instruction id %r is not a string" % insn.id)
        if insn.id in insn_ids:
            raise LoopyError("duplicate instruction id: '%s'" % insn.id)
        insn_ids.add(insn.id)


@for_each_kernel
def check_loop_priority_inames_known(kernel):
    """
    Checks if the inames in :attr:`loopy.LoopKernel.loop_priority` are part of
    the *kernel*'s domain.
    """
    for prio in kernel.loop_priority:
        for iname in prio:
            if iname not in kernel.all_inames():
                raise LoopyError("unknown iname '%s' in loop priorities" % iname)


@for_each_kernel
def check_multiple_tags_allowed(kernel):
    """
    Checks if a multiple tags of an iname are compatible.
    """
    from loopy.kernel.data import (GroupInameTag, LocalInameTag, VectorizeTag,
                UnrollTag, ForceSequentialTag, IlpBaseTag, filter_iname_tags_by_type)
    illegal_combinations = [
        (GroupInameTag, LocalInameTag, VectorizeTag, UnrollTag, ForceSequentialTag),
        (IlpBaseTag, ForceSequentialTag)
    ]
    for iname in kernel.inames.values():
        for comb in illegal_combinations:
            if len(filter_iname_tags_by_type(iname.tags, comb)) > 1:
                raise LoopyError("iname {} has illegal combination of "
                                 "tags: {}".format(iname.name, iname.tags))


def _check_for_double_use_of_hw_axes_inner(kernel, callables_table):
    from loopy.kernel.data import UniqueInameTag, GroupInameTag, LocalInameTag
    from loopy.kernel.instruction import CallInstruction
    from loopy.symbolic import ResolvedFunction

    for insn in kernel.instructions:
        insn_tag_keys = set()
        if isinstance(insn, CallInstruction):
            assert isinstance(insn.expression.function, ResolvedFunction)
            clbl = callables_table[insn.expression.function.name]
            gsize, lsize = clbl.get_used_hw_axes(callables_table)
            insn_tag_keys |= {GroupInameTag(i).key for i in gsize}
            insn_tag_keys |= {LocalInameTag(i).key for i in lsize}

        for iname in insn.within_inames:
            for tag in kernel.iname_tags_of_type(iname, UniqueInameTag):
                key = tag.key
                if key in insn_tag_keys:
                    raise LoopyError("instruction '%s' has multiple "
                            "inames tagged '%s'" % (insn.id, tag))

                insn_tag_keys.add(key)


def check_for_double_use_of_hw_axes(t_unit):
    """
    Check if any instruction of *kernel* is within multiple inames tagged with
    the same hw axis tag.
    """
    from loopy.kernel.function_interface import (CallableKernel,
                                                 ScalarCallable)
    for clbl in t_unit.callables_table.values():
        if isinstance(clbl, CallableKernel):
            _check_for_double_use_of_hw_axes_inner(clbl.subkernel,
                                                   t_unit.callables_table)
        elif isinstance(clbl, ScalarCallable):
            pass
        else:
            raise NotImplementedError(type(clbl).__name__)


@for_each_kernel
def check_for_inactive_iname_access(kernel):
    """
    Check if any instruction accesses an iname but is not within it.
    """
    for insn in kernel.instructions:
        expression_inames = insn.read_dependency_names() & kernel.all_inames()

        if not expression_inames <= insn.within_inames:
            raise LoopyError(
                    "instruction '%s' references "
                    "inames '%s' that the instruction does not depend on in "
                    "the kernel '%s'"
                    % (insn.id,
                        ", ".join(expression_inames
                                  - insn.within_inames), kernel.name))


@for_each_kernel
def check_for_unused_inames(kernel):
    """
    Check if there are any unused inames in the kernel.
    """
    # Warn if kernel has unused inames
    from loopy.transform.iname import get_used_inames
    unused_inames = kernel.all_inames() - get_used_inames(kernel)
    if unused_inames:
        warn_with_kernel(
            kernel, "unused_inames",
            "Found unused inames in kernel: %s "
            "Unused inames during linearization will be prohibited in "
            "Loopy version 2021.X."
            % unused_inames)


def _is_racing_iname_tag(tv, tag):
    from loopy.kernel.data import (AddressSpace,
            LocalInameTagBase, GroupInameTag, ConcurrentTag, auto)

    if tv.address_space == AddressSpace.PRIVATE:
        return (
                isinstance(tag, ConcurrentTag)
                and not isinstance(tag, (LocalInameTagBase, GroupInameTag)))

    elif tv.address_space == AddressSpace.LOCAL:
        return (
                isinstance(tag, ConcurrentTag)
                and not isinstance(tag, GroupInameTag))

    elif tv.address_space == AddressSpace.GLOBAL:
        return isinstance(tag, ConcurrentTag)

    elif tv.address_space == auto:
        raise LoopyError("scope of temp var '%s' has not yet been"
                "determined" % tv.name)

    else:
        raise ValueError("unexpected value of temp_var.address_space for "
                "temporary variable '%s'" % tv.name)


@for_each_kernel
def check_for_write_races(kernel):
    """
    Check if any memory accesses lead to write races.
    """
    from loopy.kernel.data import ConcurrentTag

    for insn in kernel.instructions:
        for assignee_name, assignee_indices in zip(
                insn.assignee_var_names(),
                insn.assignee_subscript_deps()):
            assignee_inames = assignee_indices & kernel.all_inames()
            if not assignee_inames <= insn.within_inames:
                raise LoopyError(
                        "assignee of instructions '%s' references "
                        "iname that the instruction does not depend on"
                        % insn.id)

            if assignee_name in kernel.arg_dict:
                # Any concurrent tags that are not depended upon by the assignee
                # will cause write races.

                raceable_parallel_insn_inames = {
                    iname for iname in insn.within_inames
                    if kernel.iname_tags_of_type(iname, ConcurrentTag)}

            elif assignee_name in kernel.temporary_variables:
                temp_var = kernel.temporary_variables[assignee_name]
                raceable_parallel_insn_inames = {
                        iname for iname in insn.within_inames
                        if any(_is_racing_iname_tag(temp_var, tag)
                            for tag in kernel.iname_tags(iname))}

            else:
                raise LoopyError("invalid assignee name in instruction '%s'"
                        % insn.id)

            race_inames = \
                    raceable_parallel_insn_inames - assignee_inames

            if race_inames:
                warn_with_kernel(kernel, "write_race(%s)" % insn.id,
                        "instruction '%s' contains a write race: "
                        "instruction will be run across parallel iname(s) "
                        "'%s', which is/are not referenced in the lhs index"
                        % (insn.id, ",".join(race_inames)),
                        WriteRaceConditionWarning)


@for_each_kernel
def check_for_data_dependent_parallel_bounds(kernel):
    """
    Check that inames tagged as hw axes have bounds that are known at kernel
    launch.
    """
    from loopy.kernel.data import ConcurrentTag

    for i, dom in enumerate(kernel.domains):
        dom_inames = set(dom.get_var_names(dim_type.set))
        par_inames = {
                iname for iname in dom_inames
                if kernel.iname_tags_of_type(iname, ConcurrentTag)}

        if not par_inames:
            continue

        parameters = set(dom.get_var_names(dim_type.param))
        for par in parameters:
            if par in kernel.temporary_variables:
                raise LoopyError("Domain number %d has a data-dependent "
                        "parameter '%s' and contains parallel "
                        "inames '%s'. This is not allowed (for now)."
                        % (i, par, ", ".join(par_inames)))


# {{{ check access bounds

# {{{ helpers for _AccessCheckMapper

def _align_and_intersect(d1, d2):
    d1, d2 = isl.align_two(d1, d2)
    return (d1 & d2).params()


def _align_and_intersect_with_caller_assumption(callee_assumptions,
                                                caller_assumptions):

    for name, (dt, pos) in caller_assumptions.get_var_dict().items():
        caller_assumptions = caller_assumptions.set_dim_name(
            dt, pos, f"_lp_caller_{name}")

    return _align_and_intersect(callee_assumptions,
                                caller_assumptions)


def _mark_variables_from_caller(expr):
    from loopy.symbolic import SubstitutionMapper
    import pymbolic.primitives as prim

    def subst_func(x):
        if isinstance(x, prim.Variable):
            return prim.Variable(f"_lp_caller_{x.name}")

    return SubstitutionMapper(subst_func)(expr)

# }}}


class _AccessCheckMapper(WalkMapper):
    def __init__(self, kernel, callables_table):
        self.kernel = kernel
        self.callables_table = callables_table
        super().__init__()

    @memoize_method
    def _make_slab(self, space, iname, start, stop):
        from loopy.isl_helpers import make_slab
        return make_slab(space, iname, start, stop)

    @memoize_method
    def _get_access_range(self, domain, subscript):
        from loopy.symbolic import (get_access_map,
                                    UnableToDetermineAccessRangeError)
        try:
            return get_access_map(domain, subscript).range()
        except UnableToDetermineAccessRangeError:
            return None

    def map_subscript(self, expr, domain, insn_id):
        WalkMapper.map_subscript(self, expr, domain, insn_id)

        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        shape = None
        var_name = expr.aggregate.name
        if var_name in self.kernel.arg_dict:
            arg = self.kernel.arg_dict[var_name]
            shape = arg.shape
        elif var_name in self.kernel.temporary_variables:
            tv = self.kernel.temporary_variables[var_name]
            shape = tv.shape

        if shape is not None:
            subscript = expr.index

            if not isinstance(subscript, tuple):
                subscript = (subscript,)

            from loopy.symbolic import get_dependencies

            available_vars = set(domain.get_var_dict())
            shape_deps = set()
            for shape_axis in shape:
                if shape_axis is not None:
                    shape_deps.update(get_dependencies(shape_axis))

            if not (get_dependencies(subscript) <= available_vars
                    and shape_deps <= available_vars):
                return

            if len(subscript) != len(shape):
                raise LoopyError("subscript to '%s' in '%s' has the wrong "
                        "number of indices (got: %d, expected: %d)" % (
                            expr.aggregate.name, expr,
                            len(subscript), len(shape)))

            access_range = self._get_access_range(domain, subscript)
            if access_range is None:
                # Likely: index was non-affine, nothing we can do.
                return

            shape_domain = isl.BasicSet.universe(access_range.get_space())
            for idim in range(len(subscript)):
                shape_axis = shape[idim]

                if shape_axis is not None:
                    slab = self._make_slab(
                            shape_domain.get_space(), (dim_type.in_, idim),
                            0, shape_axis)

                    shape_domain = shape_domain.intersect(slab)

            if not access_range.is_subset(shape_domain):
                raise LoopyIndexError("'%s' in instruction '%s' "
                        "accesses out-of-bounds array element (could not"
                        " establish '%s' is a subset of '%s')."
                        % (expr, insn_id, access_range, shape_domain))

    def map_if(self, expr, domain, insn_id):
        from loopy.symbolic import condition_to_set
        then_set = condition_to_set(domain.space, expr.condition)
        if then_set is None:
            # condition cannot be inferred as ISL expression => ignore
            # for domain contributions enforced by it
            then_set = else_set = isl.BasicSet.universe(domain.space)
        else:
            else_set = then_set.complement()

        self.rec(expr.then, domain & then_set, insn_id)
        self.rec(expr.else_, domain & else_set, insn_id)

    def map_call(self, expr, domain, insn_id):
        # perform access checks on the call arguments
        super().map_call(expr, domain, insn_id)

        import pymbolic.primitives as prim
        from loopy.kernel.function_interface import (CallableKernel,
                                                     get_kw_pos_association)
        from loopy.symbolic import (guarded_aff_from_expr,
                                    get_dependencies)
        from loopy.diagnostic import ExpressionToAffineConversionError

        if (isinstance(expr.function, ResolvedFunction)
            and isinstance(self.callables_table[expr.function.name],
                           CallableKernel)):

            subkernel = self.callables_table[expr.function.name].subkernel

            # The plan here is to add the constraints coming from the values
            # args passed at a call-site as assumptions to the callee. To avoid
            # variable-naming collision between the caller and callee, we
            # prepend all the caller variable by "_lp_caller_".

            # {{{ get bset from kwargs

            kw_to_pos, _ = get_kw_pos_association(subkernel)

            arg_id_to_arg = self.kernel.id_to_insn[insn_id].arg_id_to_arg()

            kwargs = {k: _mark_variables_from_caller(arg_id_to_arg[kw_to_pos[k]])
                      for k in subkernel.get_unwritten_value_args()}

            kw_space = isl.Space.create_from_names(
                subkernel.isl_context, set=[],
                params=(get_dependencies(tuple(kwargs.values()))
                        | set(kwargs.keys())))

            extra_assumptions = isl.BasicSet.universe(kw_space).params()

            for kw, arg in kwargs.items():
                try:
                    aff = guarded_aff_from_expr(extra_assumptions.space,
                                                prim.Variable(kw) - arg)
                except ExpressionToAffineConversionError:
                    # arg expression not affine => don't add any constraints
                    # corresponding to it
                    continue

                extra_assumptions = (extra_assumptions
                                     .add_constraint(isl.Constraint
                                                     .equality_from_aff(aff)))

            # FIXME: caller inames could be arguments => should take that into
            # account as well

            # take into account caller's assumptions
            extra_assumptions = _align_and_intersect_with_caller_assumption(
                extra_assumptions, self.kernel.assumptions)

            # project out the assumptions on caller's variables as they don't
            # bear any semantic meaning in the callee.
            extra_assumptions = extra_assumptions.project_out_except(
                types=[isl.dim_type.param],
                names=subkernel.get_unwritten_value_args())

            subkernel = subkernel.copy(
                assumptions=_align_and_intersect(subkernel.assumptions,
                                                 extra_assumptions))

            # }}}

            _check_bounds_inner_rec(subkernel, self.callables_table)


def _check_bounds_inner(kernel, callables_table):
    from loopy.kernel.instruction import get_insn_domain

    temp_var_names = set(kernel.temporary_variables)
    acm = _AccessCheckMapper(kernel, callables_table)
    kernel_assumptions_is_universe = kernel.assumptions.is_universe()
    for insn in kernel.instructions:
        domain = get_insn_domain(insn, kernel)

        # data-dependent bounds? can't do much
        if set(domain.get_var_names(dim_type.param)) & temp_var_names:
            continue

        if kernel_assumptions_is_universe:
            domain_with_assumptions = domain
        else:
            domain, assumptions = isl.align_two(domain, kernel.assumptions)
            domain_with_assumptions = domain & assumptions

        def run_acm(expr):
            acm(expr, domain_with_assumptions, insn.id)  # noqa: B023
            return expr

        insn.with_transformed_expressions(run_acm)


def _check_bounds_inner_rec(kernel, callables_table):
    if kernel.options.enforce_array_accesses_within_bounds not in [
            "no_check",
            True,
            False]:
        raise LoopyError("invalid value for option "
                "'enforce_array_accesses_within_bounds': %s"
                % kernel.options.enforce_array_accesses_within_bounds)

    if kernel.options.enforce_array_accesses_within_bounds == "no_check":
        return

    from pytools import ProcessLogger
    with ProcessLogger(logger, "%s: check array access within bounds" % kernel.name):
        if kernel.options.enforce_array_accesses_within_bounds:
            _check_bounds_inner(kernel, callables_table)
        else:
            from loopy.diagnostic import LoopyIndexError
            try:
                _check_bounds_inner(kernel, callables_table)
            except LoopyIndexError as e:
                from loopy.diagnostic import warn_with_kernel
                warn_with_kernel(kernel, "array_access_out_of_bounds", str(e))


def check_bounds(t_unit):
    """
    Performs out-of-bound check for every array access.
    """
    for epoint in t_unit.entrypoints:
        _check_bounds_inner_rec(t_unit[epoint],
                                t_unit.callables_table)

# }}}


# {{{ check write destinations

@for_each_kernel
def check_write_destinations(kernel):
    for insn in kernel.instructions:
        for wvar in insn.assignee_var_names():
            if wvar in kernel.all_inames():
                raise LoopyError("iname '%s' may not be written" % wvar)

            insn_domain = kernel.get_inames_domain(insn.within_inames)
            insn_params = set(insn_domain.get_var_names(dim_type.param))

            if wvar in kernel.all_params():
                if wvar not in kernel.temporary_variables:
                    raise LoopyError("domain parameter '%s' may not be written"
                            "--it is not a temporary variable" % wvar)

                if wvar in insn_params:
                    raise LoopyError("domain parameter '%s' may not be written "
                            "inside a domain dependent on it" % wvar)

            if not (wvar in kernel.temporary_variables
                    or wvar in kernel.arg_dict) and wvar not in kernel.all_params():
                raise LoopyError

# }}}


# {{{ check_has_schedulable_iname_nesting

@for_each_kernel
def check_has_schedulable_iname_nesting(kernel):
    from loopy.transform.iname import (has_schedulable_iname_nesting,
            get_iname_duplication_options)
    if not has_schedulable_iname_nesting(kernel):
        import itertools as it
        opt = get_iname_duplication_options(kernel)
        opt_str = "\n".join(f"* Duplicate {i} within instructions {w}"
                            for i, w in it.islice(opt, 3))
        raise LoopyError("Kernel does not have a schedulable iname nesting. "
                "In order for there to exist a feasible loop nesting, you "
                "may need to duplicate an iname. To do so, call "
                "loopy.duplicate_iname. Use loopy.get_iname_duplication_options "
                "to get hints about which iname to duplicate. Here are some "
                "options:\n%s" % opt_str)

# }}}


# {{{ check_variable_access_ordered

def declares_nosync_with(kernel, var_address_space, dep_a, dep_b):
    dep_a = kernel.id_to_insn[dep_a]
    dep_b = kernel.id_to_insn[dep_b]
    from loopy.kernel.data import AddressSpace
    if var_address_space == AddressSpace.GLOBAL:
        search_scopes = ["global", "any"]
    elif var_address_space == AddressSpace.LOCAL:
        search_scopes = ["local", "any"]
    elif var_address_space == AddressSpace.PRIVATE:
        search_scopes = ["any"]
    else:
        raise ValueError("unexpected value of 'AddressSpace'")

    ab_nosync = False
    ba_nosync = False

    for scope in search_scopes:
        if (dep_a.id, scope) in dep_b.no_sync_with:
            ab_nosync = True
        if (dep_b.id, scope) in dep_a.no_sync_with:
            ba_nosync = True

    return ab_nosync and ba_nosync


def _get_address_space(kernel, var):
    from loopy.kernel.data import ValueArg, AddressSpace, ArrayArg
    if var in kernel.temporary_variables:
        address_space = kernel.temporary_variables[var].address_space
    else:
        arg = kernel.arg_dict[var]
        if isinstance(arg, ArrayArg):
            address_space = arg.address_space
        elif isinstance(arg, ValueArg):
            address_space = AddressSpace.PRIVATE
        else:
            # No need to consider ConstantArg and ImageArg (for now)
            # because those won't be written.
            raise ValueError("could not determine address_space of '%s'" % var)
    return address_space


def _get_topological_order(kernel):
    """
    Returns a :class:`list` of insn ids of *kernel* in a topological sort
    order.

    If there is a dependency cycle within the instructions of *kernel* raises a
    :class:`loopy.diagnostic.DependencyCycleFound` exception.
    """
    from pytools.graph import compute_sccs
    from loopy.diagnostic import DependencyCycleFound

    dep_map = {insn.id: insn.depends_on for insn in kernel.instructions}

    # pytools.graph.compute_sccs serves 2 purposes:
    #   1. computes topological sort order of instructions.
    #   2. provides info. about any cycles in the graph.
    sccs = compute_sccs(dep_map)
    order = []

    for scc in sccs:
        if len(scc) != 1:
            raise DependencyCycleFound(", ".join(scc))
        order.append(scc[0])

    return order


def _check_variable_access_ordered_inner(kernel):
    from loopy.kernel.tools import find_aliasing_equivalence_classes
    from loopy.symbolic import AccessRangeOverlapChecker
    overlap_checker = AccessRangeOverlapChecker(kernel)
    aliasing_equiv_classes = find_aliasing_equivalence_classes(kernel)

    # dep_reqs_to_vars: A mapping (writer_id, dep_req_id) -> set of variable names,
    # where the tuple denotes a pair of instructions IDs, and the variable
    # names are the ones that necessitate a dependency.
    #
    # This mapping describes all the pairs of instructions (involving at least
    # one write) that require ordering by way of a dependency.
    # This mapping is then "whittled down" to remove pairs that have
    # dependencies between them. E.g. a pair of writers will be contained in
    # the mapping in both directions.
    #
    # Note: This can be worst-case O(n^2) in the number of instructions.
    dep_reqs_to_vars = {}

    wmap = kernel.writer_map()
    rmap = kernel.reader_map()

    # {{{ populate 'dep_reqs_to_vars'

    for var in kernel.get_written_variables():
        address_space = _get_address_space(kernel, var)
        eq_class = aliasing_equiv_classes[var]

        readers = set.union(
                *[rmap.get(eq_name, set()) for eq_name in eq_class])
        writers = set.union(
                *[wmap.get(eq_name, set()) for eq_name in eq_class])

        for writer in writers:
            required_deps = (readers | writers) - {writer}
            required_deps = {req_dep
                for req_dep in required_deps
                if not declares_nosync_with(kernel, address_space, writer,
                    req_dep)}

            for req_dep in required_deps:
                dep_reqs_to_vars.setdefault((writer, req_dep), set()).add(var)

    # }}}

    # {{{ compute rev_depends, depends_on

    # depends_on: mapping from insn_ids to their dependencies
    depends_on = {insn.id: set() for insn in kernel.instructions}
    # rev_depends: mapping from insn_ids to their reverse deps.
    rev_depends = {insn.id: set() for insn in kernel.instructions}

    for insn in kernel.instructions:
        depends_on[insn.id].update(insn.depends_on)
        for dep in insn.depends_on:
            rev_depends[dep].add(insn.id)

    # }}}

    # {{{ remove pairs from dep_reqs_to_vars for which dependencies exist

    topological_order = _get_topological_order(kernel)

    def satisfy_dep_reqs_in_order(dep_reqs_to_vars, edges, order):
        """
        Considering a graph defined by *edges* (as ``key -> value``),
        remove pairs of nodes from *dep_reqs_to_vars* for which edges
        exist in the graph. In doing so, make use of the topological
        order *order* (given as a sequence of *nodes*). For ``i<j``,
        no direct or indirect edges from ``order[j]`` to ``order[i]``
        exist. (All edges go from 'low index' to 'high index'.)
        """

        insn_to_req_deps = defaultdict(set)
        insn_to_order = {insn: i for i, insn in enumerate(order)}

        # Use dep_reqs_to_vars to construct insn_to_req_deps, listing
        # path endpoints pred -> ... -> insn to be considered for elimination.
        for insn, pred in dep_reqs_to_vars:
            if insn_to_order[pred] > insn_to_order[insn]:
                # If *pred* happens after *insn*, then there is no path
                # *pred* -> ... -> *insn*.
                continue

            insn_to_req_deps[pred].add(insn)

        for pred, check_successors in insn_to_req_deps.items():
            # for each *pred*, we will calculate all the direct/indirect
            # instructions that can be reached.
            seen_successors = set()
            # first let us start with direct sucessors
            to_check = edges[pred].copy()
            while to_check:
                successor = to_check.pop()
                seen_successors.add(successor)

                # Next we check if this successor was in *dep_reqs_to_vars*
                # and remove the pair from there if it is
                if successor in check_successors:
                    dep_reqs_to_vars.pop((successor, pred))
                    check_successors.remove(successor)
                    if not check_successors:
                        break

                # next we iterate through the successors of successor.
                for edge in edges[successor]:
                    if edge not in seen_successors:
                        to_check.add(edge)

    # forward dep. graph traversal in reverse topological sort order
    # (proceeds "end of program" -> "beginning of program")
    satisfy_dep_reqs_in_order(dep_reqs_to_vars, depends_on,
            topological_order[::-1])

    # Run the same thing on the reversed graph, with the reverse topological_order.
    # All edges above are 'low index to high index'.
    # Reversing graph and order maintains this property.

    # reverse dep. graph traversal in topological sort order
    # (proceeds "beginning of program" -> "end of program")
    satisfy_dep_reqs_in_order(dep_reqs_to_vars, rev_depends, topological_order)

    # }}}

    # {{{ handle dependency requirements that weren't satisfied

    for (writer_id, other_id), variables in dep_reqs_to_vars.items():
        writer = kernel.id_to_insn[writer_id]
        other = kernel.id_to_insn[other_id]

        for var in variables:
            eq_class = aliasing_equiv_classes[var]
            unaliased_readers = rmap.get(var, set())
            unaliased_writers = wmap.get(var, set())

            is_relationship_by_aliasing = not (
                writer_id in unaliased_writers
                and (writer_id in unaliased_writers
                    or other_id in unaliased_readers))

            # Do not enforce ordering for disjoint access ranges
            if (not is_relationship_by_aliasing and not
                overlap_checker.do_access_ranges_overlap_conservative(
                        writer_id, "w", other_id, "any", var)):
                continue

            # Do not enforce ordering for aliasing-based relationships
            # in different groups.
            if (is_relationship_by_aliasing and (
                    bool(writer.groups & other.conflicts_with_groups)
                    or
                    bool(other.groups & writer.conflicts_with_groups))):
                continue

            msg = ("No dependency relationship found between "
                    "'{writer_id}' which writes {var} and "
                    "'{other_id}' which also accesses {var}. "
                    "Either add a (possibly indirect) dependency "
                    "between the two, or add them to each others' nosync "
                    "set to indicate that no ordering is intended, or "
                    "turn off this check by setting the "
                    "'enforce_variable_access_ordered' option "
                    "(more issues of this type may exist--only reporting "
                    "the first one)"
                    .format(
                        writer_id=writer_id,
                        other_id=other_id,
                        var=(
                            "the variable '%s'" % var
                            if len(eq_class) == 1
                            else (
                                "the aliasing equivalence class '%s'"
                                % ", ".join(eq_class))
                            )))

            from loopy.diagnostic import VariableAccessNotOrdered
            raise VariableAccessNotOrdered(msg)

    # }}}


@for_each_kernel
def check_variable_access_ordered(kernel):
    """Checks that between each write to a variable and all other accesses to
    the variable there is either:

    * a direct/indirect depdendency edge, or
    * an explicit statement that no ordering is necessary (expressed
      through a bi-directional :attr:`loopy.InstructionBase.no_sync_with`)
    """

    if kernel.options.enforce_variable_access_ordered not in [
            "no_check",
            True,
            False]:
        raise LoopyError("invalid value for option "
                "'enforce_variable_access_ordered': %s"
                % kernel.options.enforce_variable_access_ordered)

    if kernel.options.enforce_variable_access_ordered == "no_check":
        return

    from pytools import ProcessLogger
    with ProcessLogger(logger, "%s: check variable access ordered" % kernel.name):
        if kernel.options.enforce_variable_access_ordered:
            _check_variable_access_ordered_inner(kernel)
        else:
            from loopy.diagnostic import VariableAccessNotOrdered
            try:
                _check_variable_access_ordered_inner(kernel)
            except VariableAccessNotOrdered as e:
                from loopy.diagnostic import warn_with_kernel
                warn_with_kernel(kernel, "variable_access_ordered", str(e))

# }}}

# }}}


def pre_schedule_checks(t_unit):
    try:
        logger.debug("pre-schedule checks start for entrypoints: "
                     f"{t_unit.entrypoints}.")

        check_for_integer_subscript_indices(t_unit)

        check_functions_are_resolved(t_unit)
        check_separated_array_consistency(t_unit)
        check_offsets_and_dim_tags(t_unit)
        # Ordering restriction:
        # check_sub_array_ref_inames_not_within_or_redn_inames should be done
        # before check_bounds. See: BatchedAccessMapMapper.map_sub_array_ref.
        check_sub_array_ref_inames_not_within_or_redn_inames(t_unit)
        check_for_duplicate_insn_ids(t_unit)
        check_for_double_use_of_hw_axes(t_unit)
        check_insn_attributes(t_unit)
        check_loop_priority_inames_known(t_unit)
        check_multiple_tags_allowed(t_unit)
        check_for_inactive_iname_access(t_unit)
        check_for_unused_inames(t_unit)
        check_for_write_races(t_unit)
        check_for_data_dependent_parallel_bounds(t_unit)
        check_bounds(t_unit)
        check_write_destinations(t_unit)
        check_has_schedulable_iname_nesting(t_unit)
        check_variable_access_ordered(t_unit)

        logger.debug("pre-schedule checks done")
    except KeyboardInterrupt:
        raise
    except Exception:
        print(75*"=")
        print("failing translation unit during pre-schedule check:")
        print(75*"=")
        print(t_unit)
        print(75*"=")
        raise


# {{{ post-schedule / pre-code-generation checks

# {{{ check_for_nested_base_storage

def check_for_nested_base_storage(kernel: LoopKernel) -> None:
    # must run after preprocessing has created variables for base_storage

    from loopy.kernel.data import ArrayArg
    arrays: List[ArrayBase] = [
        arg for arg in kernel.args if isinstance(arg, ArrayArg)
        ]
    arrays = arrays + list(kernel.temporary_variables.values())

    name_to_array = {ary.name: ary for ary in arrays}

    for ary in kernel.temporary_variables.values():
        if ary.base_storage:
            storage_array = name_to_array.get(ary.base_storage, None)

            if storage_array is None:
                raise ValueError("nothing known about storage array "
                        f"'{ary.base_storage}' serving as base_storage of "
                        f"'{ary.name}'")

            if storage_array.base_storage:
                raise ValueError("storage array "
                        f"'{storage_array.name}' serving as base_storage of "
                        f"'{ary.name}' may not itself use base_storage "
                        "(currently given as '{storage_array.base_storage}'")

# }}}


# {{{ check for unused hw axes

def _check_for_unused_hw_axes_in_kernel_chunk(kernel, callables_table,
        sched_index=None):
    from loopy.schedule import (CallKernel, RunInstruction,
            Barrier, EnterLoop, LeaveLoop, ReturnFromKernel,
            get_insn_ids_for_block_at, gather_schedule_block)

    if sched_index is None:
        group_axes = set()
        local_axes = set()

        i = 0
        loop_end_i = past_end_i = len(kernel.linearization)
    else:
        assert isinstance(kernel.linearization[sched_index], CallKernel)
        _, past_end_i = gather_schedule_block(kernel.linearization, sched_index)
        group_size, local_size = kernel.get_grid_sizes_for_insn_ids_as_exprs(
                get_insn_ids_for_block_at(kernel.linearization, sched_index),
                callables_table, return_dict=True)

        group_axes = set(group_size.keys())
        local_axes = set(local_size.keys())

        i = sched_index + 1
        assert isinstance(kernel.linearization[past_end_i - 1], ReturnFromKernel)
        loop_end_i = past_end_i - 1

    # alternative: just disregard length-1 dimensions?

    from loopy.kernel.data import (LocalInameTag, AutoLocalInameTagBase,
                        GroupInameTag)

    while i < loop_end_i:
        sched_item = kernel.linearization[i]
        if isinstance(sched_item, CallKernel):
            i = _check_for_unused_hw_axes_in_kernel_chunk(kernel,
                    callables_table, i)

        elif isinstance(sched_item, RunInstruction):
            insn = kernel.id_to_insn[sched_item.insn_id]
            i += 1

            if isinstance(insn, NoOpInstruction):
                continue

            group_axes_used = set()
            local_axes_used = set()

            for iname in insn.within_inames:
                ltags = kernel.iname_tags_of_type(iname, LocalInameTag, max_num=1)
                gtags = kernel.iname_tags_of_type(iname, GroupInameTag, max_num=1)
                altags = kernel.iname_tags_of_type(
                        iname, AutoLocalInameTagBase, max_num=1)

                if ltags:
                    tag, = ltags
                    local_axes_used.add(tag.axis)
                elif gtags:
                    tag, = gtags
                    group_axes_used.add(tag.axis)
                elif altags:
                    raise LoopyError("auto local tag encountered")

            # {{{ account for any hw axes due to a callable

            if isinstance(insn, CallInstruction):
                assert isinstance(insn.expression.function, ResolvedFunction)
                clbl = callables_table[insn.expression.function.name]
                clbl_g_axes, clbl_l_axes = clbl.get_used_hw_axes(callables_table)
                assert len(group_axes_used & clbl_g_axes) == 0
                assert len(local_axes_used & clbl_l_axes) == 0
                group_axes_used |= clbl_g_axes
                local_axes_used |= clbl_l_axes

            # }}}

            if group_axes != group_axes_used:
                raise LoopyError(
                        f"instruction '{insn.id}' does not use all group hw axes "
                        "(available: %s used: %s). "
                        "Calling loopy.add_inames_for_unused_hw_axes(...) "
                        "might help."
                        % (
                            ",".join(str(i) for i in group_axes),
                            ",".join(str(i) for i in group_axes_used)))

            if local_axes != local_axes_used:
                raise LoopyError(
                        f"instruction '{insn.id}' does not use all local hw axes "
                        "(available: %s used: %s). "
                        "Calling loopy.add_inames_for_unused_hw_axes(...) "
                        "might help."
                        % (
                            ",".join(str(i) for i in local_axes),
                            ",".join(str(i) for i in local_axes_used)))

        elif isinstance(sched_item, (Barrier, EnterLoop, LeaveLoop)):
            i += 1
            continue

        else:
            raise TypeError(
                    "schedule item not understood: %s" % type(sched_item).__name__)

    return past_end_i


def check_for_unused_hw_axes_in_insns(kernel, callables_table):
    if kernel.linearization:
        _check_for_unused_hw_axes_in_kernel_chunk(kernel,
                callables_table)

# }}}


# {{{ check that atomic ops are used exactly on atomic arrays

def check_that_atomic_ops_are_used_exactly_on_atomic_arrays(kernel):
    from loopy.kernel.data import ArrayBase, Assignment
    from loopy.types import AtomicType
    atomicity_candidates = (
            {v.name for v in kernel.temporary_variables.values()
                if isinstance(v.dtype, AtomicType)}
            |
            {v.name for v in kernel.args
                if isinstance(v, ArrayBase)
                and isinstance(v.dtype, AtomicType)})

    for insn in kernel.instructions:
        if not isinstance(insn, Assignment):
            continue

        atomic_accesses = {a.var_name for a in insn.atomicity}
        if not atomic_accesses <= atomicity_candidates:
            raise LoopyError("atomic access in instruction '%s' to "
                    "non-atomic variable(s) '%s'"
                    % (insn.id,
                        ",".join(atomic_accesses - atomicity_candidates)))

        accessed_atomic_vars = insn.dependency_names() & atomicity_candidates
        if not accessed_atomic_vars <= atomic_accesses:
            raise LoopyError("atomic variable(s) '%s' in instruction '%s' "
                    "used in non-atomic access"
                    % (
                        ",".join(accessed_atomic_vars - atomic_accesses),
                        insn.id))

# }}}


# {{{ check that temporaries are defined in subkernels where used

def check_that_temporaries_are_defined_in_subkernels_where_used(kernel):
    from loopy.kernel.data import AddressSpace
    from loopy.kernel.tools import get_subkernels

    globally_defined_base_storage = set()

    for subkernel in get_subkernels(kernel):
        locally_defined_base_storage = set()

        from loopy.schedule.tools import (
                temporaries_written_in_subkernel, temporaries_read_in_subkernel)

        for temporary in temporaries_written_in_subkernel(kernel, subkernel):
            tval = kernel.temporary_variables[temporary]
            if tval.base_storage is not None:
                if tval.address_space in (AddressSpace.PRIVATE, AddressSpace.LOCAL):
                    locally_defined_base_storage.add(tval.base_storage)
                else:
                    assert tval.address_space == AddressSpace.GLOBAL
                    globally_defined_base_storage.add(tval.base_storage)

        for temporary in (
                temporaries_read_in_subkernel(kernel, subkernel) -
                temporaries_written_in_subkernel(kernel, subkernel)):
            tval = kernel.temporary_variables[temporary]

            if tval.initializer is not None:
                continue

            # For aliased temporaries, check if there is an aliased definition.
            if tval.base_storage is not None:
                if tval.base_storage not in (locally_defined_base_storage
                                             | globally_defined_base_storage):
                    from loopy.diagnostic import MissingDefinitionError
                    raise MissingDefinitionError("temporary variable '%s' gets "
                            "used in subkernel '%s' and neither it nor its "
                            "aliases have a definition" % (temporary, subkernel))
                continue

            if tval.address_space in (AddressSpace.PRIVATE, AddressSpace.LOCAL):
                from loopy.diagnostic import MissingDefinitionError
                raise MissingDefinitionError("temporary variable '%s' gets used "
                        "in subkernel '%s' without a definition (maybe you forgot "
                        "to call loopy.save_and_reload_temporaries?)"
                        % (temporary, subkernel))

# }}}


# {{{ check that all instructions are scheduled

def check_that_all_insns_are_scheduled(kernel):

    all_schedulable_insns = {insn.id for insn in kernel.instructions}
    from loopy.schedule import sched_item_to_insn_id
    scheduled_insns = {
        insn_id
        for sched_item in kernel.linearization
        for insn_id in sched_item_to_insn_id(sched_item)}

    assert scheduled_insns <= all_schedulable_insns

    if scheduled_insns < all_schedulable_insns:
        from loopy.diagnostic import UnscheduledInstructionError
        raise UnscheduledInstructionError(
            "unscheduled instructions: '%s'"
            % ", ".join(all_schedulable_insns - scheduled_insns))

# }}}


# {{{ check that shapes and strides are arguments

def check_that_shapes_and_strides_are_arguments(kernel):
    from loopy.kernel.data import ValueArg
    from loopy.kernel.array import ArrayBase, FixedStrideArrayDimTag
    from loopy.symbolic import get_dependencies
    import loopy as lp

    integer_arg_names = {
            arg.name
            for arg in kernel.args
            if isinstance(arg, ValueArg)
            and arg.dtype.is_integral()}

    for arg in kernel.args:
        if isinstance(arg, ArrayBase):
            if isinstance(arg.shape, tuple):
                shape_deps = set()
                for shape_axis in arg.shape:
                    if shape_axis is not None:
                        shape_deps.update(get_dependencies(shape_axis))

                if not shape_deps <= integer_arg_names:
                    raise LoopyError("'%s' has a shape that depends on "
                            "non-argument(s): %s" % (
                                arg.name, ", ".join(shape_deps-integer_arg_names)))

            if arg.dim_tags is None:
                continue

            for dim_tag in arg.dim_tags:
                if isinstance(dim_tag, FixedStrideArrayDimTag):
                    if dim_tag.stride is lp.auto:
                        continue

                    deps = get_dependencies(dim_tag.stride)
                    if not deps <= integer_arg_names:
                        raise LoopyError("'%s' has a stride that depends on "
                                "non-argument(s): %s" % (
                                    arg.name, ", ".join(deps-integer_arg_names)))

# }}}


# {{{ validate_kernel_call_sites

def _get_sub_array_ref_swept_range(kernel, sar):
    from loopy.symbolic import get_access_map
    domain = kernel.get_inames_domain(frozenset({iname_var.name
                                                 for iname_var in sar.swept_inames}))
    return get_access_map(domain, sar.swept_inames, kernel.assumptions).range()


def _are_sub_array_refs_equivalent(sar1, sar2, caller):
    """
    Returns *True* iff *sar1* and *sar2* are equivalent
    :class:`loopy.SubArrayRef`s.

    Two sub-array-refs are said to be equivalent iff they point to the same
    array sub-regions. This equivalence check is less strict than
    :meth:`~loopy.SubArrayRef.is_equal`.

    :arg caller: An instance of :class:`loopy.LoopKernel` in which they are
         referenced.
    """
    if len(sar1.swept_inames) != len(sar2.swept_inames):
        return False

    if sar1.subscript.aggregate.name != sar2.subscript.aggregate.name:
        return False

    if len(sar1.subscript.index_tuple) != len(sar2.subscript.index_tuple):
        return False

    if (_get_sub_array_ref_swept_range(caller, sar1)
            != _get_sub_array_ref_swept_range(caller, sar2)):
        return False

    from loopy.symbolic import SubstitutionMapper
    from pymbolic.mapper.substitutor import make_subst_func
    from loopy.symbolic import simplify_via_aff
    subst_func = make_subst_func({iname1.name:  iname2
                                  for iname1, iname2 in zip(sar1.swept_inames,
                                                            sar2.swept_inames)
                                  })

    # subst_mapper: maps swept inames from sar1 to sar2
    subst_mapper = SubstitutionMapper(subst_func)

    for idx1, idx2 in zip(sar1.subscript.index_tuple,
                          sar2.subscript.index_tuple):
        if simplify_via_aff(subst_mapper(idx1) - idx2) != 0:
            return False
    return True


def _validate_kernel_call_insn(caller, call_insn, callee):
    assert call_insn.expression.function.name == callee.name
    from loopy.symbolic import SubArrayRef
    from loopy.kernel.array import ArrayBase

    arg_id_to_arg = call_insn.arg_id_to_arg()

    next_iarg_input = 0
    next_iarg_output = -1

    for arg in callee.args:
        if arg.is_input:
            if next_iarg_input not in arg_id_to_arg:
                raise LoopyError(f"Call to '{callee.name}' in '{call_insn}' expects"
                                 f" a {next_iarg_input+1}-th positional "
                                 "argument corresponding"
                                 f" to '{arg.name}' in the callee.")
            in_val = arg_id_to_arg[next_iarg_input]
            next_iarg_input += 1
            if isinstance(arg, ArrayBase):
                if not isinstance(in_val, SubArrayRef):
                    raise LoopyError(f"Call to '{callee.name}' in '{call_insn}'"
                                     f" expects a sub-array-ref for '{arg.name}'"
                                     f" (got {in_val}).")
            else:
                if isinstance(in_val, SubArrayRef):
                    raise LoopyError(f"Call to '{callee.name}' in '{call_insn}'"
                                     f" expects a value argument for '{arg.name}'"
                                     f" (got {in_val}).")
        if arg.is_output:
            if next_iarg_output not in arg_id_to_arg:
                raise LoopyError(f"Call to '{callee.name}' in '{call_insn}' expects"
                                 f" a {-next_iarg_output}-th positional assignee"
                                 f" corresponding to '{arg.name}'in the callee.")

            out_val = arg_id_to_arg[next_iarg_output]
            next_iarg_output -= 1
            assert isinstance(arg, ArrayBase)
            if not isinstance(out_val, SubArrayRef):
                raise LoopyError(f"Call to '{callee.name}' in '{call_insn}'"
                                 f" expects a sub-array-ref for '{arg.name}'"
                                 f" (got {out_val}).")

        if arg.is_input and arg.is_output:
            if not _are_sub_array_refs_equivalent(in_val, out_val, caller):
                raise LoopyError(f"Call to '{callee.name}' in '{call_insn}' expects"
                                 f" equivalent sub-array-refs for '{arg.name}'"
                                 f" (got {in_val}, {out_val}).")


def _validate_kernel_call_sites_inner(kernel, callables):
    from pymbolic.primitives import Call
    from loopy.kernel.function_interface import CallableKernel

    for insn in kernel.instructions:
        if (isinstance(insn, CallInstruction)
                and isinstance(insn.expression, Call)
                and isinstance(insn.expression.function, ResolvedFunction)):
            clbl = callables[insn.expression.function.name]
            if isinstance(clbl, CallableKernel):
                _validate_kernel_call_insn(kernel, insn, clbl.subkernel)
        elif isinstance(insn, (MultiAssignmentBase, CInstruction,
                               _DataObliviousInstruction)):
            pass
        else:
            raise NotImplementedError(type(insn))


def validate_kernel_call_sites(translation_unit):
    for name in translation_unit.callables_table:
        clbl = translation_unit[name]
        if isinstance(clbl, LoopKernel):
            _validate_kernel_call_sites_inner(clbl, translation_unit.callables_table)


# }}}


# {{{ check_all_callees_have_same_index_dtype

def check_all_callees_have_same_index_dtype(epoint: LoopKernel,
                                            callables_table):
    from loopy.kernel.function_interface import CallableKernel

    epoint_clbl = callables_table[epoint.name]
    for clbl_name in epoint_clbl.get_called_callables(callables_table,
                                                      recursive=True):
        clbl = callables_table[clbl_name]
        if isinstance(clbl, CallableKernel):
            if clbl.subkernel.index_dtype != epoint.index_dtype:
                raise LoopyError(f"Callee '{clbl_name}' in entrypoint"
                                 f" {epoint.name}'s callgraph does not have the"
                                 " same index_dtype as the entrypoint. This is"
                                 " not allowed.")

# }}}


def pre_codegen_entrypoint_checks(kernel, callables_table):
    logger.debug("pre-codegen entrypoint check %s: start" % kernel.name)

    kernel.target.pre_codegen_entrypoint_check(kernel, callables_table)
    check_all_callees_have_same_index_dtype(kernel, callables_table)

    logger.debug("pre-codegen entrypoint check %s: done" % kernel.name)


def pre_codegen_callable_checks(kernel, callables_table):
    logger.debug("pre-codegen callable check %s: start" % kernel.name)

    check_for_nested_base_storage(kernel)
    check_for_unused_hw_axes_in_insns(kernel, callables_table)
    check_that_atomic_ops_are_used_exactly_on_atomic_arrays(kernel)
    check_that_temporaries_are_defined_in_subkernels_where_used(kernel)
    check_that_all_insns_are_scheduled(kernel)
    kernel.target.pre_codegen_callable_check(kernel, callables_table)
    check_that_shapes_and_strides_are_arguments(kernel)

    logger.debug("pre-codegen callable check %s: done" % kernel.name)


def pre_codegen_checks(t_unit):
    from loopy.kernel.function_interface import CallableKernel

    try:
        for e in t_unit.entrypoints:
            pre_codegen_entrypoint_checks(t_unit[e], t_unit.callables_table)

        for clbl in t_unit.callables_table.values():
            if isinstance(clbl, CallableKernel):
                pre_codegen_callable_checks(clbl.subkernel, t_unit.callables_table)
    except Exception:
        print(75*"=")
        print("failing kernel during pre-codegen check:")
        print(75*"=")
        print(t_unit)
        print(75*"=")
        raise

# }}}


# {{{ sanity-check for implemented domains of each instruction

def check_implemented_domains(kernel, implemented_domains, code=None):
    from islpy import dim_type

    from islpy import align_two

    last_idomains = None
    last_insn_inames = None

    for insn_id, idomains in implemented_domains.items():
        insn = kernel.id_to_insn[insn_id]

        assert idomains

        insn_inames = insn.within_inames

        # {{{ if we've checked the same thing before, no need to check it again

        if last_idomains is not None and last_insn_inames is not None:
            if idomains == last_idomains and insn_inames == last_insn_inames:
                continue

        last_idomains = idomains
        last_insn_inames = insn_inames

        # }}}

        insn_impl_domain = idomains[0]
        for idomain in idomains[1:]:
            insn_impl_domain = insn_impl_domain | idomain
        assumption_non_param = isl.BasicSet.from_params(kernel.assumptions)
        assumptions, insn_impl_domain = align_two(
                assumption_non_param, insn_impl_domain)
        insn_impl_domain = (
                (insn_impl_domain & assumptions)
                .project_out_except(insn_inames, [dim_type.set]))

        from loopy.kernel.instruction import BarrierInstruction
        from loopy.kernel.data import LocalInameTag
        if isinstance(insn, BarrierInstruction):
            # project out local-id-mapped inames, solves #94 on gitlab
            non_lid_inames = frozenset(iname for iname in insn_inames
                if not kernel.iname_tags_of_type(iname, LocalInameTag))
            insn_impl_domain = insn_impl_domain.project_out_except(
                non_lid_inames, [dim_type.set])

        insn_domain = kernel.get_inames_domain(insn_inames)
        insn_parameters = frozenset(insn_domain.get_var_names(dim_type.param))
        assumptions, insn_domain = align_two(assumption_non_param, insn_domain)
        desired_domain = ((insn_domain & assumptions)
            .project_out_except(insn_inames, [dim_type.set])
            .project_out_except(insn_parameters, [dim_type.param]))

        if isinstance(insn, BarrierInstruction):
            # project out local-id-mapped inames, solves #94 on gitlab
            desired_domain = desired_domain.project_out_except(
                non_lid_inames, [dim_type.set])

        insn_impl_domain = (insn_impl_domain
                .project_out_except(insn_parameters, [dim_type.param]))
        insn_impl_domain, desired_domain = align_two(
                insn_impl_domain, desired_domain)

        if insn_impl_domain != desired_domain:
            i_minus_d = insn_impl_domain - desired_domain
            d_minus_i = desired_domain - insn_impl_domain

            parameter_inames = {
                    insn_domain.get_dim_name(dim_type.param, i)
                    for i in range(insn_impl_domain.dim(dim_type.param))}

            lines = []
            for bigger, smaller, diff_set, gist_domain in [
                    ("implemented", "desired", i_minus_d,
                        desired_domain.gist(insn_impl_domain)),
                    ("desired", "implemented", d_minus_i,
                        insn_impl_domain.gist(desired_domain))]:

                if diff_set.is_empty():
                    continue

                diff_set = diff_set.coalesce()
                pt = diff_set.sample_point()
                assert not pt.is_void()

                #pt_set = isl.Set.from_point(pt)
                #lines.append("point implemented: %s" % (pt_set <= insn_impl_domain))
                #lines.append("point desired: %s" % (pt_set <= desired_domain))

                iname_to_dim = pt.get_space().get_var_dict()
                point_axes = []
                for iname in insn_inames | parameter_inames:
                    tp, dim = iname_to_dim[iname]
                    point_axes.append("%s=%d" % (
                        iname, pt.get_coordinate_val(tp, dim).to_python()))

                lines.append(
                        "sample point in {} but not {}: {}".format(
                            bigger, smaller, ", ".join(point_axes)))
                lines.append(
                        "gist of constraints in {} but not {}: {}".format(
                            bigger, smaller, gist_domain))

            if code is not None:
                print(79*"-")
                print("CODE:")
                print(79*"-")
                from loopy.target.execution import get_highlighted_code
                print(get_highlighted_code(code))
                print(79*"-")

            raise LoopyError("sanity check failed--implemented and desired "
                    "domain for instruction '%s' do not match\n\n"
                    "implemented: %s\n\n"
                    "desired:%s\n\n%s"
                    % (insn_id, insn_impl_domain, desired_domain, "\n".join(lines)))

    # placate the assert at the call site
    return True

# }}}


# vim: foldmethod=marker

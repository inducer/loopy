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


from dataclasses import dataclass
from typing import FrozenSet, List, Mapping, Optional, Sequence, Type, Union
from immutables import Map
import islpy as isl
from pytools.tag import Tag
from loopy.kernel import LoopKernel
from loopy.typing import ExpressionT, auto, not_none
from loopy.match import ToStackMatchCovertible
from loopy.symbolic import (get_dependencies,
        RuleAwareIdentityMapper, RuleAwareSubstitutionMapper,
        SubstitutionRuleMappingContext, CombineMapper)
from loopy.diagnostic import LoopyError
from pymbolic.mapper.substitutor import make_subst_func
from loopy.translation_unit import TranslationUnit
from loopy.kernel.instruction import InstructionBase, MultiAssignmentBase
from loopy.kernel.function_interface import (CallableKernel, InKernelCallable,
                                             ScalarCallable)
from loopy.kernel.tools import (kernel_has_global_barriers,
                                find_most_recent_global_barrier)
from loopy.kernel.data import AddressSpace
from loopy.types import LoopyType, ToLoopyTypeConvertible, to_loopy_type

from pymbolic import var
from pytools import memoize_on_first_arg

from loopy.transform.array_buffer_map import (ArrayToBufferMap,
                                              ArrayToBufferMapBase,
                                              NoOpArrayToBufferMap,
                                              AccessDescriptor)


# {{{ contains_subst_rule_invocation

class FunctionNameCollector(CombineMapper):
    def combine(self, values):
        from functools import reduce
        return reduce(frozenset.union, values, frozenset())

    def map_call(self, expr):
        return self.combine([frozenset([expr.function.name])]
                            + [self.rec(arg) for arg in expr.parameters])

    def map_call_with_kwargs(self, expr):
        return self.combine([frozenset([expr.function.name])]
                            + [self.rec(arg) for arg in expr.parameters]
                            + [self.rec(arg) for arg in expr.kw_parameters.values()])

    def map_algebraic_leaf(self, expr):
        return frozenset()

    def map_constant(self, expr):
        return frozenset()


@memoize_on_first_arg
def _get_calls_in_expr(expr):
    return FunctionNameCollector()(expr)


@memoize_on_first_arg
def _get_called_names(insn):
    assert isinstance(insn, MultiAssignmentBase)
    from pymbolic.primitives import Expression
    from functools import reduce
    return ((_get_calls_in_expr(insn.expression)
             if isinstance(insn.expression, Expression)
             else frozenset())
            # indices of assignees might call the subst rules
            | reduce(frozenset.union,
                     (_get_calls_in_expr(assignee)
                      for assignee in insn.assignees),
                     frozenset())
            | reduce(frozenset.union,
                     (_get_calls_in_expr(pred)
                      for pred in insn.predicates
                      if isinstance(pred, Expression)),
                     frozenset())
            )


@memoize_on_first_arg
def _get_all_substs(kernel):
    return frozenset(kernel.substitutions)


def contains_a_subst_rule_invocation(kernel, insn):
    all_substs = _get_all_substs(kernel)
    return ((all_substs & _get_called_names(insn))
            or (all_substs & insn.read_dependency_names()))

# }}}


@dataclass(frozen=True)
class RuleAccessDescriptor(AccessDescriptor):
    args: Optional[Sequence[ExpressionT]] = None


def access_descriptor_id(args, expansion_stack):
    return (args, expansion_stack)


def storage_axis_exprs(storage_axis_sources, args) -> Sequence[ExpressionT]:
    result = []

    for saxis_source in storage_axis_sources:
        if isinstance(saxis_source, int):
            result.append(args[saxis_source])
        else:
            result.append(var(saxis_source))

    return result


# {{{ gather rule invocations

class RuleInvocationGatherer(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, kernel, subst_name, subst_tag, within):
        super().__init__(rule_mapping_context)

        from loopy.symbolic import SubstitutionRuleExpander
        self.subst_expander = SubstitutionRuleExpander(
                kernel.substitutions)

        self.kernel = kernel
        self.subst_name = subst_name
        self.subst_tag = subst_tag
        self.within = within

        self.access_descriptors: List[RuleAccessDescriptor] = []

    def map_substitution(self, name, tag, arguments, expn_state):
        process_me = name == self.subst_name

        if self.subst_tag is not None and self.subst_tag != tag:
            process_me = False

        process_me = process_me and self.within(
                expn_state.kernel,
                expn_state.instruction,
                expn_state.stack)

        if not process_me:
            return super().map_substitution(
                    name, tag, arguments, expn_state)

        rule = self.rule_mapping_context.old_subst_rules[name]
        arg_context = self.make_new_arg_context(
                    name, rule.arguments, arguments, expn_state.arg_context)

        arg_deps = set()
        for arg_val in arg_context.values():
            arg_deps = (arg_deps
                    | get_dependencies(self.subst_expander(arg_val)))

        # FIXME: This is too strict--and the footprint machinery
        # needs to be taught how to deal with locally constant
        # variables.
        if not arg_deps <= self.kernel.all_inames():
            from warnings import warn
            warn("Precompute arguments in '%s(%s)' do not consist exclusively "
                    "of inames and constants--specifically, these are "
                    "not inames: %s. Ignoring." % (
                        name,
                        ", ".join(str(arg) for arg in arguments),
                        ", ".join(arg_deps - self.kernel.all_inames()),
                        ))

            return super().map_substitution(
                    name, tag, arguments, expn_state)

        args = [arg_context[arg_name] for arg_name in rule.arguments]

        self.access_descriptors.append(
                RuleAccessDescriptor(
                    identifier=access_descriptor_id(args, expn_state.stack),
                    args=args,
                    ))

        return 0  # exact value irrelevant

# }}}


# {{{ replace rule invocation

class RuleInvocationReplacer(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, subst_name, subst_tag, within,
            access_descriptors, array_base_map,
            storage_axis_names, storage_axis_sources,
            non1_storage_axis_names,
            temporary_name, compute_insn_id, compute_dep_id,
            compute_read_variables):
        super().__init__(rule_mapping_context)

        self.subst_name = subst_name
        self.subst_tag = subst_tag
        self.within = within

        self.access_descriptors = access_descriptors
        self.array_base_map = array_base_map

        self.storage_axis_names = storage_axis_names
        self.storage_axis_sources = storage_axis_sources
        self.non1_storage_axis_names = non1_storage_axis_names

        self.temporary_name = temporary_name
        self.compute_insn_id = compute_insn_id
        self.compute_dep_id = compute_dep_id

        self.compute_read_variables = compute_read_variables
        self.compute_insn_depends_on = set()

    def map_substitution(self, name, tag, arguments, expn_state):
        if not (
                name == self.subst_name
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)
                and (self.subst_tag is None or self.subst_tag == tag)):
            return super().map_substitution(
                    name, tag, arguments, expn_state)

        # {{{ check if in footprint

        rule = self.rule_mapping_context.old_subst_rules[name]
        arg_context = self.make_new_arg_context(
                    name, rule.arguments, arguments, expn_state.arg_context)
        args = [arg_context[arg_name] for arg_name in rule.arguments]

        accdesc = AccessDescriptor(
                storage_axis_exprs=storage_axis_exprs(
                    self.storage_axis_sources, args))

        if not self.array_base_map.is_access_descriptor_in_footprint(accdesc):
            return super().map_substitution(
                    name, tag, arguments, expn_state)

        # }}}

        assert len(arguments) == len(rule.arguments)

        abm = self.array_base_map

        stor_subscript = []
        for sax_name, sax_source, sax_base_idx in zip(
                self.storage_axis_names,
                self.storage_axis_sources,
                abm.storage_base_indices):
            if sax_name not in self.non1_storage_axis_names:
                continue

            if isinstance(sax_source, int):
                # an argument
                ax_index = arguments[sax_source]
            else:
                # an iname
                ax_index = var(sax_source)

            from loopy.symbolic import simplify_via_aff
            ax_index = simplify_via_aff(ax_index - sax_base_idx)
            stor_subscript.append(ax_index)

        new_outer_expr = var(self.temporary_name)
        if stor_subscript:
            new_outer_expr = new_outer_expr.index(tuple(stor_subscript))

        # Can't possibly be nested, and no need to traverse
        # further as compute expression has already been seen
        # by rule_mapping_context.

        self.replaced_something = True

        # {{{ add gbarriers that the replaced insn depends-on to compute insn's deps

        if (kernel_has_global_barriers(expn_state.kernel)
                and (find_most_recent_global_barrier(expn_state.kernel,
                                                     expn_state.instruction.id
                                                     ) is not None)):
            self.compute_insn_depends_on.add(
                find_most_recent_global_barrier(expn_state.kernel,
                                                expn_state.instruction.id))

        # }}}

        return new_outer_expr

    def map_kernel(self, kernel):
        new_insns = []

        excluded_insn_ids = {self.compute_insn_id, self.compute_dep_id}
        # precomputed_in_insns: set of insn ids in which the subst rule was
        # precomputed.
        precomputed_in_insns = set()

        for insn in kernel.instructions:
            self.replaced_something = False

            if (isinstance(insn, MultiAssignmentBase)
                    and not contains_a_subst_rule_invocation(kernel, insn)):
                # 'insn' does not call a subst => do not process
                new_insns.append(insn)
                continue

            insn = insn.with_transformed_expressions(
                    lambda expr: self(expr, kernel, insn))  # noqa: B023

            if self.replaced_something:
                insn = insn.copy(
                        depends_on=(
                            insn.depends_on
                            | frozenset([self.compute_dep_id])))

                precomputed_in_insns.add(insn.id)

                for dep in insn.depends_on:
                    if dep in excluded_insn_ids:
                        continue

                    dep_insn = kernel.id_to_insn[dep]
                    if (frozenset(dep_insn.assignee_var_names())
                            & self.compute_read_variables):
                        self.compute_insn_depends_on.update(
                                insn.depends_on - excluded_insn_ids)

            new_insns.append(insn)

        # compute_insn cannot depend on its users.
        self.compute_insn_depends_on -= precomputed_in_insns

        return kernel.copy(instructions=new_insns)

# }}}


def precompute_for_single_kernel(
        kernel: LoopKernel,
        callables_table: Mapping[str, InKernelCallable], subst_use,
        sweep_inames=None,
        within: ToStackMatchCovertible = None,
        *,
        storage_axes=None,
        temporary_name: Optional[str] = None,
        precompute_inames: Optional[Sequence[str]] = None,
        precompute_outer_inames: Optional[FrozenSet[str]] = None,
        storage_axis_to_tag=None,

        default_tag: Union[None, Tag, str] = None,

        dtype: Optional[ToLoopyTypeConvertible] = None,
        fetch_bounding_box: bool = False,
        temporary_address_space: Union[AddressSpace, None, Type[auto]] = None,
        compute_insn_id: Optional[str] = None,
        _enable_mirgecom_workaround: bool = False,
        ) -> LoopKernel:
    """Precompute the expression described in the substitution rule determined by
    *subst_use* and store it in a temporary array. A precomputation needs two
    things to operate, a list of *sweep_inames* (order irrelevant) and an
    ordered list of *storage_axes* (whose order will describe the axis ordering
    of the temporary array).

    :arg subst_use: Describes what to prefetch.

        The following objects may be given for *subst_use*:

        * The name of the substitution rule.

        * The tagged name ("name$tag") of the substitution rule.

        * A list of invocations of the substitution rule.
          This list of invocations, when swept across *sweep_inames*, then serves
          to define the footprint of the precomputation.

          Invocations may be tagged ("name$tag") to filter out a subset of the
          usage sites of the substitution rule. (Namely those usage sites that
          use the same tagged name.)

          Invocations may be given as a string or as a
          :class:`pymbolic.primitives.Expression` object.

          If only one invocation is to be given, then the only entry of the list
          may be given directly.

    If the list of invocations generating the footprint is not given,
    all (tag-matching, if desired) usage sites of the substitution rule
    are used to determine the footprint.

    The following cases can arise for each sweep axis:

    * The axis is an iname that occurs within arguments specified at
      usage sites of the substitution rule. This case is assumed covered
      by the storage axes provided for the argument.

    * The axis is an iname that occurs within the *value* of the rule, but not
      within its arguments. A new, dedicated storage axis is allocated for
      such an axis.

    :arg sweep_inames: A :class:`list` of inames to be swept.
        May also equivalently be a comma-separated string.
    :arg within: a stack match as understood by
        :func:`loopy.match.parse_stack_match`.
    :arg storage_axes: A :class:`list` of inames and/or rule argument
        names/indices to be used as storage axes.
        May also equivalently be a comma-separated string.
    :arg temporary_name:
        The temporary variable name to use for storing the precomputed data.
        If it does not exist, it will be created. If it does exist, its properties
        (such as size, type) are checked (and updated, if possible) to match
        its use.
    :arg precompute_inames:
        A tuple of inames to be used to carry out the precomputation.
        If the specified inames do not already exist, they will be
        created. If they do already exist, their loop domain is verified
        against the one required for this precomputation. This tuple may
        be shorter than the (provided or automatically found) *storage_axes*
        tuple, in which case names will be automatically created.
        May also equivalently be a comma-separated string.

    :arg precompute_outer_inames: A :class:`frozenset` of inames within which
        the compute instruction is nested. If *None*, make an educated guess.
        May also be specified as a comma-separated string.

    :arg default_tag: The :ref:`iname tag <iname-tags>` to be applied to the
        inames created to perform the precomputation. By default, new
        inames remain untagged.

    :arg dtype: The dtype of the temporary variable to precompute the result
        in. Can be either a dtype as understood by :class:`numpy.dtype` or
        *None* to let :mod:`loopy` infer it.

    :arg compute_insn_id: The ID of the instruction generated to perform the
        precomputation.

    If `storage_axes` is not specified, it defaults to the arrangement
    `<direct sweep axes><arguments>` with the direct sweep axes being the
    slower-varying indices.

    Trivial storage axes (i.e. axes of length 1 with respect to the sweep) are
    eliminated.
    """
    # {{{ check, standardize arguments

    if sweep_inames is None:
        sweep_inames = []
    if storage_axis_to_tag is None:
        storage_axis_to_tag = {}

    if isinstance(sweep_inames, str):
        sweep_inames = [iname.strip() for iname in sweep_inames.split(",")]

    for iname in sweep_inames:
        if iname not in kernel.all_inames():
            raise RuntimeError("sweep iname '%s' is not a known iname"
                    % iname)

    sweep_inames = list(sweep_inames)
    sweep_inames_set = frozenset(sweep_inames)

    if isinstance(storage_axes, str):
        storage_axes = [ax.strip() for ax in storage_axes.split(",")]

    if isinstance(precompute_inames, str):
        precompute_inames = [iname.strip() for iname in precompute_inames.split(",")]

    if isinstance(precompute_outer_inames, str):
        precompute_outer_inames = frozenset(
                iname.strip() for iname in precompute_outer_inames.split(","))

    if isinstance(subst_use, str):
        subst_use = [subst_use]

    footprint_generators = None

    subst_name: Optional[str] = None
    subst_tag = None

    from pymbolic.primitives import Variable, Call
    from loopy.symbolic import parse, TaggedVariable

    for use in subst_use:
        if isinstance(use, str):
            use = parse(use)

        if isinstance(use, Call):
            if footprint_generators is None:
                footprint_generators = []

            footprint_generators.append(use)
            subst_name_as_expr = use.function
        else:
            subst_name_as_expr = use

        if isinstance(subst_name_as_expr, TaggedVariable):
            new_subst_name = subst_name_as_expr.name
            new_subst_tag = subst_name_as_expr.tag
        elif isinstance(subst_name_as_expr, Variable):
            new_subst_name = subst_name_as_expr.name
            new_subst_tag = None
        else:
            raise ValueError("unexpected type of subst_name")

        if (subst_name, subst_tag) == (None, None):
            subst_name, subst_tag = new_subst_name, new_subst_tag
        else:
            if (subst_name, subst_tag) != (new_subst_name, new_subst_tag):
                raise ValueError("not all uses in subst_use agree "
                        "on rule name and tag")

    from loopy.match import parse_stack_match
    within = parse_stack_match(within)

    assert subst_name is not None

    try:
        subst = kernel.substitutions[subst_name]
    except KeyError:
        raise LoopyError("substitution rule '%s' not found"
                % subst_name)

    c_subst_name = subst_name.replace(".", "_")

    from loopy.kernel.data import parse_tag
    default_tag = parse_tag(default_tag)

    # }}}

    # {{{ process invocations in footprint generators, start access_descriptors

    if footprint_generators:
        from pymbolic.primitives import Variable, Call

        access_descriptors = []

        for fpg in footprint_generators:
            if isinstance(fpg, Variable):
                args = ()
            elif isinstance(fpg, Call):
                args = fpg.parameters
            else:
                raise ValueError("footprint generator must "
                        "be substitution rule invocation")

            access_descriptors.append(
                    RuleAccessDescriptor(
                        identifier=access_descriptor_id(args, None),
                        args=args
                        ))

    # }}}

    # {{{ gather up invocations in kernel code, finish access_descriptors

    if not footprint_generators:
        rule_mapping_context = SubstitutionRuleMappingContext(
                kernel.substitutions, kernel.get_var_name_generator())
        invg = RuleInvocationGatherer(
                rule_mapping_context, kernel, subst_name, subst_tag, within)
        del rule_mapping_context

        import loopy as lp
        for insn in kernel.instructions:
            if (isinstance(insn, lp.MultiAssignmentBase)
                    and contains_a_subst_rule_invocation(kernel, insn)):
                for assignee in insn.assignees:
                    invg(assignee, kernel, insn)
                invg(insn.expression, kernel, insn)

        access_descriptors = invg.access_descriptors
        if not access_descriptors:
            raise RuntimeError("no invocations of '%s' found" % subst_name)

    # }}}

    # {{{ find inames used in arguments

    expanding_usage_arg_deps = set()

    for accdesc in access_descriptors:
        assert accdesc.args is not None

        for arg in accdesc.args:
            expanding_usage_arg_deps.update(
                    get_dependencies(arg) & kernel.all_inames())

    # }}}

    var_name_gen = kernel.get_var_name_generator()

    # {{{ use given / find new storage_axes

    # extra axes made necessary because they don't occur in the arguments
    extra_storage_axes = set(sweep_inames_set - expanding_usage_arg_deps)

    from loopy.symbolic import SubstitutionRuleExpander
    submap = SubstitutionRuleExpander(kernel.substitutions)

    value_inames = (
            get_dependencies(submap(subst.expression))
            - frozenset(subst.arguments)
            ) & kernel.all_inames()
    if value_inames - expanding_usage_arg_deps < extra_storage_axes:
        raise RuntimeError("unreferenced sweep inames specified: "
                + ", ".join(extra_storage_axes
                    - value_inames - expanding_usage_arg_deps))

    new_iname_to_tag = {}

    if storage_axes is None:
        storage_axes = []

        # Add sweep_inames (in given--rather than arbitrary--order) to
        # storage_axes *if* they are part of extra_storage_axes.
        for iname in sweep_inames:
            if iname in extra_storage_axes:
                extra_storage_axes.remove(iname)
                storage_axes.append(iname)

        if extra_storage_axes:
            if (precompute_inames is not None
                    and len(storage_axes) < len(precompute_inames)):
                raise LoopyError("must specify a sufficient number of "
                        "storage_axes to uniquely determine the meaning "
                        "of the given precompute_inames. (%d storage_axes "
                        "needed)" % len(precompute_inames))
            storage_axes.extend(sorted(extra_storage_axes))

        storage_axes.extend(range(len(subst.arguments)))

    del extra_storage_axes

    prior_storage_axis_name_dict = {}

    storage_axis_names: List[str] = []
    storage_axis_sources: List[Union[str, int]] = []  # number for arg#, or iname

    # {{{ check for pre-existing precompute_inames

    if precompute_inames is not None:
        preexisting_precompute_inames = (
                set(precompute_inames) & kernel.all_inames())
    else:
        preexisting_precompute_inames = set()

    # }}}

    for i, saxis in enumerate(storage_axes):
        tag_lookup_saxis = saxis

        if saxis in subst.arguments:
            saxis = subst.arguments.index(saxis)

        storage_axis_sources.append(saxis)

        if isinstance(saxis, int):
            # argument index
            name = old_name = subst.arguments[saxis]
        else:
            old_name = saxis
            name = f"{c_subst_name}_{old_name}"

        if (precompute_inames is not None
                and i < len(precompute_inames)
                and precompute_inames[i]):
            name = precompute_inames[i]
            tag_lookup_saxis = name
            if (name not in preexisting_precompute_inames
                    and var_name_gen.is_name_conflicting(name)):
                raise RuntimeError("new storage axis name '%s' "
                        "conflicts with existing name" % name)
        else:
            name = var_name_gen(name)

        storage_axis_names.append(name)
        if name not in preexisting_precompute_inames:
            iname_tag = storage_axis_to_tag.get(tag_lookup_saxis, None)
            if iname_tag is None:
                iname_tag = default_tag
            if iname_tag is not None:
                new_iname_to_tag[name] = iname_tag

        prior_storage_axis_name_dict[name] = old_name

    del storage_axis_to_tag
    del storage_axes
    del precompute_inames

    # }}}

    # {{{ fill out access_descriptors[...].storage_axis_exprs

    access_descriptors = [
            accdesc.copy(
                storage_axis_exprs=storage_axis_exprs(
                    storage_axis_sources, accdesc.args))
            for accdesc in access_descriptors]

    # }}}

    expanding_inames = sweep_inames_set | frozenset(expanding_usage_arg_deps)
    assert expanding_inames <= kernel.all_inames()

    if storage_axis_names:
        # {{{ find domain to be changed

        change_inames = expanding_inames | preexisting_precompute_inames

        from loopy.kernel.tools import DomainChanger
        domch = DomainChanger(kernel, change_inames)

        if domch.leaf_domain_index is not None:
            # If the sweep inames are at home in parent domains, then we'll add
            # fetches with loops over copies of these parent inames that will end
            # up being scheduled *within* loops over these parents.

            for iname in sweep_inames_set:
                if kernel.get_home_domain_index(iname) != domch.leaf_domain_index:
                    raise RuntimeError("sweep iname '%s' is not 'at home' in the "
                            "sweep's leaf domain" % iname)

        # }}}

        abm: ArrayToBufferMapBase = ArrayToBufferMap(
                kernel, domch.domain, sweep_inames,
                access_descriptors, len(storage_axis_names))

        non1_storage_axis_names = []
        for i, saxis in enumerate(storage_axis_names):
            if abm.non1_storage_axis_flags[i]:
                non1_storage_axis_names.append(saxis)
            else:
                if saxis in new_iname_to_tag:
                    del new_iname_to_tag[saxis]

                if saxis in preexisting_precompute_inames:
                    raise LoopyError("precompute axis %d (1-based) was "
                            "eliminated as "
                            "having length 1 but also mapped to existing "
                            "iname '%s'" % (i+1, saxis))

        mod_domain = domch.domain

        # {{{ modify the domain, taking into account preexisting inames

        # inames may already exist in mod_domain, add them primed to start
        primed_non1_saxis_names = [
                iname+"'" for iname in non1_storage_axis_names]

        mod_domain = abm.augment_domain_with_sweep(
            domch.domain, primed_non1_saxis_names,
            boxify_sweep=fetch_bounding_box)

        check_domain = mod_domain

        for i, saxis in enumerate(non1_storage_axis_names):
            var_dict = mod_domain.get_var_dict(isl.dim_type.set)

            if saxis in preexisting_precompute_inames:
                # add equality constraint between existing and new variable

                dt, dim_idx = var_dict[saxis]
                saxis_aff = isl.Aff.var_on_domain(mod_domain.space, dt, dim_idx)

                dt, dim_idx = var_dict[primed_non1_saxis_names[i]]
                new_var_aff = isl.Aff.var_on_domain(mod_domain.space, dt, dim_idx)

                mod_domain = mod_domain.add_constraint(
                        isl.Constraint.equality_from_aff(new_var_aff - saxis_aff))

                # project out the new one
                mod_domain = mod_domain.project_out(dt, dim_idx, 1)

            else:
                # remove the prime from the new variable
                dt, dim_idx = var_dict[primed_non1_saxis_names[i]]
                mod_domain = mod_domain.set_dim_name(dt, dim_idx, saxis)

        def add_assumptions(d):
            assumption_non_param = isl.BasicSet.from_params(kernel.assumptions)
            assumptions, domain = isl.align_two(assumption_non_param, d)
            return assumptions & domain

        # {{{ check that we got the desired domain

        check_domain = add_assumptions(
            check_domain.project_out_except(
                primed_non1_saxis_names, [isl.dim_type.set]))

        mod_check_domain = add_assumptions(mod_domain)

        # re-add the prime from the new variable
        var_dict = mod_check_domain.get_var_dict(isl.dim_type.set)

        for saxis in non1_storage_axis_names:
            dt, dim_idx = var_dict[saxis]
            mod_check_domain = mod_check_domain.set_dim_name(dt, dim_idx, saxis+"'")

        mod_check_domain = mod_check_domain.project_out_except(
                primed_non1_saxis_names, [isl.dim_type.set])

        mod_check_domain, check_domain = isl.align_two(
                mod_check_domain, check_domain)

        # The modified domain can't get bigger by adding constraints
        assert mod_check_domain <= check_domain

        if not check_domain <= mod_check_domain:
            print(check_domain)
            print(mod_check_domain)
            raise LoopyError("domain of preexisting inames does not match "
                    "domain needed for precompute")

        # }}}

        # {{{ check that we didn't shrink the original domain

        # project out the new names from the modified domain
        orig_domain_inames = list(domch.domain.get_var_dict(isl.dim_type.set))
        mod_check_domain = add_assumptions(
                mod_domain.project_out_except(
                    orig_domain_inames, [isl.dim_type.set]))

        check_domain = add_assumptions(domch.domain)

        mod_check_domain, check_domain = isl.align_two(
                mod_check_domain, check_domain)

        # The modified domain can't get bigger by adding constraints
        assert mod_check_domain <= check_domain

        if not check_domain <= mod_check_domain:
            print(check_domain)
            print(mod_check_domain)
            raise LoopyError("original domain got shrunk by applying the precompute")

        # }}}

        # }}}

        new_kernel_domains = domch.get_domains_with(mod_domain)

    else:
        # leave kernel domains unchanged
        new_kernel_domains = kernel.domains

        non1_storage_axis_names = []
        abm = NoOpArrayToBufferMap()

    kernel = kernel.copy(domains=new_kernel_domains)

    # {{{ set up compute insn

    if temporary_name is None:
        temporary_name = var_name_gen(based_on=c_subst_name)

    assignee = var(temporary_name)

    if non1_storage_axis_names:
        assignee = assignee[
                tuple(var(iname) for iname in non1_storage_axis_names)]

    # {{{ process substitutions on compute instruction

    storage_axis_subst_dict = {}

    for i, (arg_name, base_index) in enumerate(
            zip(storage_axis_names, abm.storage_base_indices)):
        is_length_1 = arg_name not in non1_storage_axis_names
        if is_length_1:
            arg = 0
        else:
            arg = var(arg_name)

        # FIXME: Hacky workaround, remove when no longer needed.
        # Some transform code in the mirgecom transform stack
        # first deletes inames from instructions if they're unused and then
        # gets upset when they've disappeared. Without this 'special handling'
        # here, this code will replace 0-length axis subscripts with '0', as it
        # should.

        if _enable_mirgecom_workaround:
            from pymbolic.primitives import Expression
            if is_length_1 and not isinstance(base_index, Expression):
                # I.e. base_index is an integer.
                from pytools import is_single_valued
                if is_single_valued(
                        not_none(accdesc.storage_axis_exprs)[i]
                        for accdesc in access_descriptors):
                    assert access_descriptors[0].storage_axis_exprs is not None
                    storage_axis_expr = access_descriptors[0].storage_axis_exprs[i]
                    if not (get_dependencies(storage_axis_expr) & sweep_inames_set):
                        # I.e. no sweeping in this axis.
                        base_index = storage_axis_expr

        storage_axis_subst_dict[
                prior_storage_axis_name_dict.get(arg_name, arg_name)] = \
                        arg+base_index

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())

    from loopy.match import parse_stack_match
    expr_subst_map = RuleAwareSubstitutionMapper(
            rule_mapping_context,
            make_subst_func(storage_axis_subst_dict),
            within=parse_stack_match(None))

    compute_expression = expr_subst_map(subst.expression, kernel, None)

    # }}}

    from loopy.kernel.data import Assignment
    if compute_insn_id is None:
        compute_insn_id = kernel.make_unique_instruction_id(based_on=c_subst_name)

    compute_insn = Assignment(
            id=compute_insn_id,
            assignee=assignee,
            expression=compute_expression,
            # within_inames determined below
            )
    compute_dep_id = compute_insn_id
    added_compute_insns: List[InstructionBase] = [compute_insn]

    if temporary_address_space == AddressSpace.GLOBAL:
        barrier_insn_id = kernel.make_unique_instruction_id(
                based_on=c_subst_name+"_barrier")
        from loopy.kernel.instruction import BarrierInstruction
        barrier_insn = BarrierInstruction(
                id=barrier_insn_id,
                depends_on=frozenset([compute_insn_id]),
                synchronization_kind="global",
                mem_kind="global")
        compute_dep_id = barrier_insn_id

        added_compute_insns.append(barrier_insn)

    # }}}

    # {{{ substitute rule into expressions in kernel (if within footprint)

    from loopy.symbolic import SubstitutionRuleExpander
    expander = SubstitutionRuleExpander(kernel.substitutions)

    invr = RuleInvocationReplacer(rule_mapping_context,
            subst_name, subst_tag, within,
            access_descriptors, abm,
            storage_axis_names, storage_axis_sources,
            non1_storage_axis_names,
            temporary_name, compute_insn_id, compute_dep_id,
            compute_read_variables=get_dependencies(expander(compute_expression)))

    kernel = invr.map_kernel(kernel)
    kernel = kernel.copy(
            instructions=added_compute_insns + list(kernel.instructions))
    kernel = rule_mapping_context.finish_kernel(kernel)

    # }}}

    # {{{ add dependencies to compute insn

    kernel = kernel.copy(
            instructions=[
                insn.copy(depends_on=frozenset(invr.compute_insn_depends_on))
                if insn.id == compute_insn_id
                else insn
                for insn in kernel.instructions])

    # }}}

    # {{{ propagate storage iname subst to dependencies of compute instructions

    from loopy.kernel.tools import find_recursive_dependencies
    compute_deps = find_recursive_dependencies(
            kernel, frozenset([compute_insn_id]))

    # FIXME: Need to verify that there are no outside dependencies
    # on compute_deps

    prior_storage_axis_names = frozenset(storage_axis_subst_dict)

    new_insns = []
    for insn in kernel.instructions:
        if (insn.id in compute_deps
                and insn.within_inames & prior_storage_axis_names):
            insn = (insn
                    .with_transformed_expressions(
                        lambda expr: expr_subst_map(expr, kernel, insn))  # noqa: B023,E501
                    .copy(within_inames=frozenset(
                        new_iname
                        for iname in insn.within_inames
                        for new_iname in get_dependencies(
                                storage_axis_subst_dict.get(iname, var(iname)))
                        )))

            new_insns.append(insn)
        else:
            new_insns.append(insn)

    kernel = kernel.copy(instructions=new_insns)

    # }}}

    # {{{ determine inames for compute insn

    if precompute_outer_inames is None:
        from loopy.kernel.tools import guess_iname_deps_based_on_var_use
        precompute_outer_inames = (
                    frozenset(non1_storage_axis_names)
                    | frozenset(
                        (expanding_usage_arg_deps | value_inames)
                        - sweep_inames_set)
                    | guess_iname_deps_based_on_var_use(kernel, compute_insn))
    else:
        if not isinstance(precompute_outer_inames, frozenset):
            raise TypeError("precompute_outer_inames must be a frozenset")

        precompute_outer_inames = precompute_outer_inames \
                | frozenset(non1_storage_axis_names)

    kernel = kernel.copy(
            instructions=[
                insn.copy(within_inames=precompute_outer_inames)
                if insn.id == compute_insn_id
                else insn
                for insn in kernel.instructions])

    # }}}

    # {{{ set up temp variable

    import loopy as lp

    loopy_type = to_loopy_type(dtype, allow_none=True)

    if temporary_address_space is None:
        temporary_address_space = lp.auto

    new_temp_shape = tuple(abm.non1_storage_shape)

    new_temporary_variables = dict(kernel.temporary_variables)
    if temporary_name not in new_temporary_variables:
        temp_var = lp.TemporaryVariable(
                name=temporary_name,
                dtype=loopy_type,
                base_indices=(0,)*len(new_temp_shape),
                shape=tuple(abm.non1_storage_shape),
                address_space=temporary_address_space,
                dim_names=tuple(non1_storage_axis_names))

    else:
        temp_var = new_temporary_variables[temporary_name]

        # {{{ check and adapt existing temporary

        if temp_var.dtype is None:
            pass
        elif temp_var.dtype is not None and dtype is None:
            dtype = temp_var.dtype
        elif temp_var.dtype is not None and dtype is not None:
            assert isinstance(temp_var.dtype, LoopyType)
            if temp_var.dtype.numpy_dtype != dtype:
                raise LoopyError("Existing and new dtype of temporary '%s' "
                        "do not match (existing: %s, new: %s)"
                        % (temporary_name, temp_var.dtype, dtype))

        temp_var = temp_var.copy(dtype=dtype)

        assert isinstance(temp_var.shape, tuple)
        if len(temp_var.shape) != len(new_temp_shape):
            raise LoopyError("Existing and new temporary '%s' do not "
                    "have matching number of dimensions ('%d' vs. '%d') "
                    % (temporary_name,
                        len(temp_var.shape), len(new_temp_shape)))

        if temp_var.base_indices != (0,) * len(new_temp_shape):
            raise LoopyError("Existing and new temporary '%s' do not "
                    "have matching number of dimensions ('%d' vs. '%d') "
                    % (temporary_name,
                        len(temp_var.shape), len(new_temp_shape)))

        new_temp_shape = tuple(
                max(i, ex_i)
                for i, ex_i in zip(new_temp_shape, temp_var.shape))

        temp_var = temp_var.copy(shape=new_temp_shape)

        if temporary_address_space == temp_var.address_space:
            pass
        elif temporary_address_space is lp.auto:
            temporary_address_space = temp_var.address_space
        elif temp_var.address_space is lp.auto:
            pass
        else:
            raise LoopyError("Existing and new temporary '%s' do not "
                    "have matching address spaces (existing: %s, new: %s)"
                    % (temporary_name,
                        AddressSpace.stringify(temp_var.address_space),
                        AddressSpace.stringify(temporary_address_space)))

        temp_var = temp_var.copy(address_space=temporary_address_space)

        # }}}

    new_temporary_variables[temporary_name] = temp_var

    kernel = kernel.copy(
            temporary_variables=new_temporary_variables)

    # }}}

    from loopy.transform.iname import tag_inames
    kernel = tag_inames(kernel, new_iname_to_tag)

    from loopy.kernel.data import AutoFitLocalInameTag, filter_iname_tags_by_type

    if filter_iname_tags_by_type(new_iname_to_tag.values(), AutoFitLocalInameTag):
        from loopy.kernel.tools import assign_automatic_axes
        kernel = assign_automatic_axes(kernel, callables_table)

    return kernel


def precompute(program, *args, **kwargs):
    assert isinstance(program, TranslationUnit)
    new_callables = {}

    for func_id, clbl in program.callables_table.items():
        if isinstance(clbl, CallableKernel):
            knl = precompute_for_single_kernel(clbl.subkernel,
                    program.callables_table, *args, **kwargs)
            clbl = clbl.copy(subkernel=knl)
        elif isinstance(clbl, ScalarCallable):
            pass
        else:
            raise NotImplementedError()

        new_callables[func_id] = clbl

    return program.copy(callables_table=Map(new_callables))

# vim: foldmethod=marker

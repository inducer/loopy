from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, override

import namedisl as nisl

import islpy as isl
import pymbolic.primitives as p
from pymbolic import var
from pymbolic.mapper.substitutor import make_subst_func
from pytools.tag import Tag

import loopy as lp
from loopy.kernel.tools import DomainChanger
from loopy.match import StackMatch, parse_stack_match
from loopy.symbolic import (
    DependencyMapper,
    ExpansionState,
    RuleAwareIdentityMapper,
    RuleAwareSubstitutionMapper,
    SubstitutionRuleExpander,
    SubstitutionRuleMappingContext,
    multi_pw_aff_from_exprs,
    pw_aff_to_expr,
)
from loopy.transform.precompute import contains_a_subst_rule_invocation
from loopy.translation_unit import for_each_kernel
from loopy.types import ToLoopyTypeConvertible, to_loopy_type


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence, Set

    from pymbolic.typing import Expression

    from loopy.kernel import LoopKernel
    from loopy.kernel.data import AddressSpace


AccessTuple: TypeAlias = tuple[str, ...]


def _access_key(args: Sequence[Expression]) -> AccessTuple:
    return tuple(str(arg) for arg in args)


def _base_name(name: str) -> str:
    return name.removesuffix("_")


def _cur_name(name: str) -> str:
    return f"{_base_name(name)}_cur"


def _prev_name(name: str) -> str:
    return f"{_base_name(name)}_prev"


def _basic_set_to_predicates(bset: nisl.BasicSet) -> frozenset[Expression]:
    isl_bset = bset._reconstruct_isl_object()

    predicates = []
    for constraint in isl_bset.get_constraints():
        expr = pw_aff_to_expr(constraint.get_aff())
        if constraint.is_equality():
            predicates.append(p.Comparison(expr, "==", 0))
        else:
            predicates.append(p.Comparison(expr, ">=", 0))

    return frozenset(predicates)


def _set_to_predicate_options(
        set_: nisl.Set | nisl.BasicSet
    ) -> Sequence[frozenset[Expression]]:
    if isinstance(set_, nisl.BasicSet):
        if set_._reconstruct_isl_object().is_empty():
            return []
        return [_basic_set_to_predicates(set_)]

    predicate_options = []
    for bset in set_.get_basic_sets():
        if not bset._reconstruct_isl_object().is_empty():
            predicate_options.append(_basic_set_to_predicates(bset))

    return predicate_options


# helper for gathering names of variables in pymbolic expressions
def _gather_vars(expr: Expression) -> set[str]:
    deps = DependencyMapper()(expr)
    var_names = set()
    for dep in deps:
        if isinstance(dep, p.Variable):
            var_names.add(dep.name)
        elif (
                isinstance(dep, p.Subscript)
                and isinstance(dep.aggregate, p.Variable)):
            var_names.add(dep.aggregate.name)

    return var_names


def _existing_name_mapping(
        map_: nisl.Map | nisl.BasicMap,
        name_mapping: Mapping[str, str]
    ) -> Mapping[str, str]:
    names = map_.names
    return {
        source: target
        for source, target in name_mapping.items()
        if source in names and target in names
    }


def _normalize_renamed_dims(
        map_: nisl.Map | nisl.BasicMap,
        name_mapping: Mapping[str, str],
    ) -> nisl.Map | nisl.BasicMap:
    map_ = map_.equate_dims(_existing_name_mapping(map_, name_mapping))

    names = map_.names
    project_names = [
        renamed_name
        for original_name, renamed_name in name_mapping.items()
        if original_name in names and renamed_name in names
    ]
    map_ = map_.project_out(project_names)

    names = map_.names
    rename_mapping = {
        renamed_name: original_name
        for original_name, renamed_name in name_mapping.items()
        if original_name not in names and renamed_name in names
    }
    return map_.rename_dims(rename_mapping)


# {{{ gathering usage expressions

class UsageSiteExpressionGatherer(RuleAwareIdentityMapper[[]]):
    """
    Gathers all expressions used as inputs to a particular substitution rule,
    identified by name.
    """
    def __init__(
            self,
            rule_mapping_ctx: SubstitutionRuleMappingContext,
            subst_expander: SubstitutionRuleExpander,
            kernel: LoopKernel,
            subst_name: str,
            subst_tag: Set[Tag] | Tag | None = None
        ) -> None:

        super().__init__(rule_mapping_ctx)

        self.subst_expander: SubstitutionRuleExpander = subst_expander
        self.kernel: LoopKernel = kernel
        self.subst_name: str = subst_name
        self.subst_tag: Set[Tag] | None = (
            {subst_tag} if isinstance(subst_tag, Tag) else subst_tag
        )

        self.usage_expressions: list[Sequence[Expression]] = []

    @override
    def map_subst_rule(
            self,
            name: str,
            tags: Set[Tag] | None,
            arguments: Sequence[Expression],
            expn_state: ExpansionState,
        ) -> Expression:

        if name != self.subst_name:
            return super().map_subst_rule(
                name, tags, arguments, expn_state
            )

        if self.subst_tag is not None and self.subst_tag != tags:
            return super().map_subst_rule(
                name, tags, arguments, expn_state
            )

        rule = self.rule_mapping_context.old_subst_rules[name]
        arg_ctx = self.make_new_arg_context(
            name, rule.arguments, arguments, expn_state.arg_context
        )

        self.usage_expressions.append([
            arg_ctx[arg_name] for arg_name in rule.arguments
        ])

        return 0

# }}}


# {{{ substitution rule use replacement

class RuleInvocationReplacer(RuleAwareIdentityMapper[[]]):
    def __init__(
            self,
            ctx: SubstitutionRuleMappingContext,
            subst_name: str,
            subst_tag: Sequence[Tag] | None,
            usage_descriptors: Mapping[AccessTuple, nisl.Map | nisl.BasicMap],
            storage_indices: Sequence[str],
            temporary_name: str,
            compute_insn_ids: str | Sequence[str],
            footprint: nisl.Set
        ) -> None:

        super().__init__(ctx)

        self.subst_name: str = subst_name
        self.subst_tag: Sequence[Tag] | None = subst_tag

        self.usage_descriptors: Mapping[AccessTuple, nisl.Map | nisl.BasicMap] = \
            usage_descriptors
        self.storage_indices: Sequence[str] = storage_indices
        self.footprint: nisl.Set = footprint

        self.temporary_name: str = temporary_name
        self.compute_insn_ids: frozenset[str] = (
            frozenset([compute_insn_ids])
            if isinstance(compute_insn_ids, str)
            else frozenset(compute_insn_ids)
        )

        self.replaced_something: bool = False

        # FIXME: may not always be the case (i.e. global barrier between
        # compute insn and uses)
        self.compute_dep_ids: frozenset[str] = self.compute_insn_ids

    @override
    def map_subst_rule(
            self,
            name: str,
            tags: Set[Tag] | None,
            arguments: Sequence[Expression],
            expn_state: ExpansionState
        ) -> Expression:

        rule = self.rule_mapping_context.old_subst_rules[name]
        arg_ctx = self.make_new_arg_context(
            name, rule.arguments, arguments, expn_state.arg_context
        )
        args = [arg_ctx[arg_name] for arg_name in rule.arguments]

        # {{{ validation checks

        if name != self.subst_name:
            return super().map_subst_rule(name, tags, arguments, expn_state)

        access_key = _access_key(args)
        if access_key not in self.usage_descriptors:
            return super().map_subst_rule(name, tags, arguments, expn_state)

        if len(arguments) != len(rule.arguments):
            raise ValueError(
                f"Number of arguments passed to rule {name} "
                f"does not match the signature of {name}."
            )

        local_map = self.usage_descriptors[access_key]
        temp_footprint = self.footprint.move_dims(
            frozenset(self.footprint.names) - frozenset(self.storage_indices),
            isl.dim_type.param
        )

        if not local_map.range() <= temp_footprint:
            return super().map_subst_rule(name, tags, arguments, expn_state)

        # }}}

        # {{{ get index expression in terms of global inames

        local_pwmaff = self.usage_descriptors[access_key].as_pw_multi_aff()

        index_exprs: Sequence[Expression] = []
        for dim in range(local_pwmaff.dim(isl.dim_type.out)):
            index_exprs.append(pw_aff_to_expr(local_pwmaff.get_at(dim)))

        new_expression = var(self.temporary_name)[tuple(index_exprs)]

        # }}}

        self.replaced_something = True

        return new_expression

    @override
    def map_kernel(
            self,
            kernel: LoopKernel,
            within: StackMatch = lambda knl, insn, stack: True,
            map_args: bool = True,
            map_tvs: bool = True
        ) -> LoopKernel:

        new_insns: Sequence[lp.InstructionBase] = []
        for insn in kernel.instructions:
            self.replaced_something = False

            if (isinstance(insn, lp.MultiAssignmentBase) and not
                contains_a_subst_rule_invocation(kernel, insn)):
                new_insns.append(insn)
                continue

            insn = insn.with_transformed_expressions(
                lambda expr, insn=insn: self(expr, kernel, insn)
            )

            if self.replaced_something:
                insn = insn.copy(
                    depends_on=(
                        insn.depends_on | self.compute_dep_ids
                    )
                )

                # FIXME: determine compute insn dependencies

            new_insns.append(insn)

        return kernel.copy(instructions=new_insns)

# }}}


@for_each_kernel
def compute(
        kernel: LoopKernel,
        substitution: str,
        compute_map: nisl.Map,

        storage_indices: Sequence[str],

        # FIXME: can these two be deduced?
        temporal_inames: Sequence[str],
        inames_to_advance: Sequence[str] | None = None,

        temporary_name: str | None = None,
        temporary_address_space: AddressSpace | None = None,

        temporary_dtype: ToLoopyTypeConvertible = None,

        compute_insn_id: str | None = None
    ) -> LoopKernel:
    """
    Inserts an instruction to compute an expression given by :arg:`substitution`
    and replaces all invocations of :arg:`substitution` with the result of the
    inserted compute instruction.

    :arg substitution: The substitution rule for which the compute
    transform should be applied.

    :arg compute_map: An :class:`isl.Map` representing a relation between
    substitution rule indices and tuples `(a, l)`, where `a` is a vector of
    storage indices and `l` is a vector of "timestamps".

    :arg storage_indices: An ordered sequence of names of storage indices. Used
    to create inames for the loops that cover the required set of compute points.
    """

    name_mapping = {
        name: name + "_"
        for name in compute_map.output_names
        if name not in storage_indices
    }
    compute_map = compute_map.rename_dims(name_mapping)

    # {{{ setup and useful variables

    storage_set = frozenset(storage_indices)
    temporal_set = frozenset(temporal_inames)

    ctx = SubstitutionRuleMappingContext(
        kernel.substitutions, kernel.get_var_name_generator())
    expander = SubstitutionRuleExpander(kernel.substitutions)
    expr_gatherer = UsageSiteExpressionGatherer(
        ctx, expander, kernel, substitution, None
    )

    _ = expr_gatherer.map_kernel(kernel)
    usage_exprs = expr_gatherer.usage_expressions

    all_exprs = [expr for usage in usage_exprs for expr in usage]
    usage_inames: frozenset[str] = frozenset(
        set.union(*(_gather_vars(expr) for expr in all_exprs))
    )

    # }}}

    # {{{ construct necessary pieces; footprint, global usage map

    # add compute inames to domain / kernel
    domain_changer = DomainChanger(kernel, kernel.all_inames())
    named_domain = nisl.make_basic_set(domain_changer.domain)

    # restrict domain to used inames
    local_domain = named_domain.project_out_except(usage_inames)

    # FIXME: gross. find a cleaner way to generate a space for an empty map
    global_usage_map = nisl.make_map_from_domain_and_range(
        nisl.make_set(isl.Set.empty(local_domain.get_space())),
        compute_map.domain()
    )
    global_usage_map = nisl.make_map(isl.Map.empty(global_usage_map.get_space()))

    usage_substs: Mapping[AccessTuple, nisl.Map | nisl.BasicMap] = {}
    for usage in usage_exprs:

        # {{{ compute local usage map, update global usage map

        local_usage_mpwaff = multi_pw_aff_from_exprs(
            usage,
            global_usage_map.get_space()
        )

        local_usage_map = nisl.make_map(local_usage_mpwaff.as_map())

        local_usage_map = local_usage_map.intersect_domain(local_domain)
        global_usage_map = global_usage_map | local_usage_map

        # }}}

        # {{{ compute storage map

        local_storage_map = local_usage_map.apply_range(compute_map)
        local_storage_map = _normalize_renamed_dims(
            local_storage_map, name_mapping)

        # check that no restrictions happened during composition (i.e. tile
        # valid for a single point in the domain)
        if not local_usage_map.domain() <= local_storage_map.domain():
            continue

        # clean up names
        non_param_names = (usage_inames - temporal_set) | storage_set
        parameter_names = frozenset(local_storage_map.names) - non_param_names
        local_storage_map = local_storage_map.move_dims(parameter_names,
                                                        isl.dim_type.param)

        # }}}

        usage_substs[_access_key(usage)] = local_storage_map

    storage_map = global_usage_map.apply_range(compute_map)
    storage_map = _normalize_renamed_dims(storage_map, name_mapping)

    # }}}

    # {{{ compute bounds and update kernel domain

    storage_map = storage_map.move_dims(temporal_set, isl.dim_type.param)
    footprint = storage_map.range()

    # clean up ticked duplicate names
    footprint = footprint.project_out_except(temporal_set | storage_set)
    footprint = footprint.move_dims(temporal_set, isl.dim_type.set)

    # {{{ FIXME: use Sets instead of BasicSets when loopy is ready

    # FIXME: convex hull is not permanent
    footprint_isl = footprint._reconstruct_isl_object()
    footprint = nisl.make_set(isl.Set.from_basic_set(footprint_isl.convex_hull()))
    named_domain = named_domain & footprint

    if len(named_domain.get_basic_sets()) != 1:
        raise ValueError("New domain should be composed of a single basic set")

    # FIXME: use named object once loopy is name-ified
    domain = named_domain.get_basic_sets()[0]._reconstruct_isl_object()
    new_domains = domain_changer.get_domains_with(domain)

    # }}}

    kernel = kernel.copy(domains=new_domains)

    # }}}

    if not temporary_name:
        temporary_name = substitution + "_temp"

    if not compute_insn_id:
        compute_insn_id = substitution + "_compute"

    # {{{ reuse analysis

    update_insns: list[lp.InstructionBase] = []
    update_insn_ids: list[str] = []
    refill_predicate_options: Sequence[frozenset[Expression] | None] = [None]
    current_update_deps: frozenset[str] = frozenset()

    if inames_to_advance is not None:
        advancing_set = frozenset(inames_to_advance)

        compute_map_cur = compute_map.rename_dims({
            name: _cur_name(name) for name in compute_map.output_names
        })
        compute_map_prev = compute_map.rename_dims({
            name: _prev_name(name) for name in compute_map.output_names
        })

        cur_storage = global_usage_map.apply_range(compute_map_cur)
        prev_storage = global_usage_map.apply_range(compute_map_prev)

        reuse_map = prev_storage.reverse().apply_range(cur_storage)
        reuse_map = reuse_map.add_constraint([
            (
                f"{name}_cur = {name}_prev + 1"
                if name in advancing_set
                else
                f"{name}_cur = {name}_prev"
            )
            for name in temporal_inames
        ])

        current_footprint = footprint.rename_dims({
            name: _cur_name(name) for name in footprint.names
        })
        previous_footprint = footprint.rename_dims({
            name: _prev_name(name) for name in footprint.names
        })

        reuse_map = reuse_map.intersect_domain(previous_footprint)
        reuse_map = reuse_map.intersect_range(current_footprint)
        reuse_map = reuse_map - nisl.make_map(
            "{ ["
            + ", ".join(_prev_name(name) for name in footprint.names)
            + "] -> ["
            + ", ".join(_cur_name(name) for name in footprint.names)
            + "] : "
            + " and ".join(
                f"{_cur_name(name)} = {_prev_name(name)}"
                for name in storage_indices
            )
            + " }"
        )

        reused_current = reuse_map.range()
        refill = current_footprint - reused_current

        cur_to_normal = {
            _cur_name(name): name
            for name in footprint.names
        }
        reused_current = reused_current.rename_dims(cur_to_normal)
        refill = refill.rename_dims(cur_to_normal)

        reused_context = named_domain.project_out_except(reused_current.names)
        refill_context = named_domain.project_out_except(refill.names)

        reused_current = reused_current.gist(reused_context)
        refill = refill.gist(refill_context)

        refill_predicate_options = _set_to_predicate_options(refill)

        storage_reuse_map = reuse_map.project_out_except(
            frozenset(_prev_name(name) for name in storage_indices)
            | frozenset(_cur_name(name) for name in storage_indices)
        )
        storage_reuse_map = storage_reuse_map.rename_dims({
            _cur_name(name): name
            for name in storage_indices
        })
        cur_to_prev = storage_reuse_map.reverse()
        cur_to_prev_pwma = cur_to_prev.as_pw_multi_aff()
        prev_expr_by_name = {
            cur_to_prev_pwma.get_dim_name(isl.dim_type.out, dim):
            pw_aff_to_expr(cur_to_prev_pwma.get_at(dim))
            for dim in range(cur_to_prev_pwma.dim(isl.dim_type.out))
        }
        prev_storage_exprs = [
            prev_expr_by_name[_prev_name(name)]
            for name in storage_indices
        ]

        shift_assignee = var(temporary_name)[
            tuple(var(idx) for idx in storage_indices)
        ]
        shift_expression = var(temporary_name)[tuple(prev_storage_exprs)]

        shift_predicate_options = _set_to_predicate_options(reused_current)
        for i, predicates in enumerate(shift_predicate_options):
            shift_insn_id = (
                f"{compute_insn_id}_shift"
                if len(shift_predicate_options) == 1
                else f"{compute_insn_id}_shift_{i}"
            )
            update_insns.append(lp.Assignment(
                id=shift_insn_id,
                assignee=shift_assignee,
                expression=shift_expression,
                within_inames=frozenset(temporal_inames) | storage_set,
                predicates=predicates,
                depends_on=current_update_deps,
            ))
            update_insn_ids.append(shift_insn_id)
            current_update_deps = frozenset([shift_insn_id])

    # }}}

    # {{{ create compute instruction in kernel

    # FIXME: maybe just keep original around?
    compute_map = compute_map.rename_dims({
        value: key for key, value in name_mapping.items()
    })

    compute_pw_aff = compute_map.reverse().as_pw_multi_aff()
    storage_ax_to_global_expr = {
        compute_pw_aff.get_dim_name(isl.dim_type.out, dim):
        pw_aff_to_expr(compute_pw_aff.get_at(dim))
        for dim in range(compute_pw_aff.dim(isl.dim_type.out))
    }

    expr_subst_map = RuleAwareSubstitutionMapper(
        ctx,
        make_subst_func(storage_ax_to_global_expr),
        within=parse_stack_match(None)
    )

    subst_expr = kernel.substitutions[substitution].expression
    compute_expression = expr_subst_map(
        subst_expr,
        kernel,
        None,
    )
    compute_dep_ids = frozenset().union(*(
        kernel.writer_map().get(var_name, frozenset())
        for var_name in _gather_vars(compute_expression)
    ))

    assignee = var(temporary_name)[tuple(var(idx) for idx in storage_indices)]

    within_inames = compute_map.output_names

    new_insns = list(kernel.instructions)
    new_insns.extend(update_insns)

    for i, predicates in enumerate(refill_predicate_options):
        refill_insn_id = (
            compute_insn_id
            if len(refill_predicate_options) == 1
            else f"{compute_insn_id}_refill_{i}"
        )
        compute_insn = lp.Assignment(
            id=refill_insn_id,
            assignee=assignee,
            expression=compute_expression,
            within_inames=within_inames,
            predicates=predicates,
            depends_on=current_update_deps | compute_dep_ids,
        )
        new_insns.append(compute_insn)
        update_insn_ids.append(refill_insn_id)
        current_update_deps = frozenset([refill_insn_id])

    kernel = kernel.copy(instructions=new_insns)

    # }}}

    # {{{ replace invocations with new compute instruction

    ctx = SubstitutionRuleMappingContext(
        kernel.substitutions, kernel.get_var_name_generator()
    )

    replacer = RuleInvocationReplacer(
        ctx,
        substitution,
        None,
        usage_substs,
        storage_indices,
        temporary_name,
        update_insn_ids,
        footprint
    )

    kernel = replacer.map_kernel(kernel)

    # }}}

    # {{{ create temporary variable for result of compute

    loopy_type = to_loopy_type(temporary_dtype, allow_none=True)

    temp_shape = tuple(
        pw_aff_to_expr(footprint.dim_max(dim)) + 1
        for dim in storage_indices
    )

    new_temp_vars = dict(kernel.temporary_variables)

    # FIXME: temp_var might already exist, handle the case where it does
    temp_var = lp.TemporaryVariable(
        name=temporary_name,
        dtype=loopy_type,
        base_indices=(0,)*len(temp_shape),
        shape=temp_shape,
        address_space=temporary_address_space,
        dim_names=tuple(storage_indices)
    )

    new_temp_vars[temporary_name] = temp_var

    kernel = kernel.copy(
        temporary_variables=new_temp_vars
    )

    # }}}

    return kernel

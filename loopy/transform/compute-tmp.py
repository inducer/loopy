from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import namedisl as nisl
from typing_extensions import override

import islpy as isl
import pymbolic.primitives as p
from pymbolic import var
from pymbolic.mapper.substitutor import make_subst_func
from pymbolic.typing import Expression
from pytools.tag import Tag

from ..kernel.data import TemporaryVariable
from ..kernel.instruction import Assignment, InstructionBase, MultiAssignmentBase
from ..kernel.tools import DomainChanger
from ..match import StackMatch, parse_stack_match
from ..symbolic import (
    DependencyMapper,
    ExpansionState,
    RuleAwareIdentityMapper,
    RuleAwareSubstitutionMapper,
    SubstitutionRuleMappingContext,
    multi_pw_aff_from_exprs,
    pw_aff_to_expr,
)
from ..translation_unit import for_each_kernel
from ..types import ToLoopyTypeConvertible, to_loopy_type
from .precompute import contains_a_subst_rule_invocation


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence, Set

    from ..kernel import LoopKernel
    from ..kernel.data import AddressSpace


UsageKey: TypeAlias = tuple[str, int]
PredicateSet: TypeAlias = frozenset[Expression]
PredicateOptions: TypeAlias = tuple[PredicateSet | None, ...]


@dataclass(frozen=True)
class UsageSite:
    insn_id: str
    ordinal: int
    args: tuple[Expression, ...]
    predicates: PredicateSet

    @property
    def key(self) -> UsageKey:
        return self.insn_id, self.ordinal

    @property
    def domain_names(self) -> frozenset[str]:
        exprs = (*self.args, *self.predicates)
        return frozenset(set().union(*(_gather_vars(expr) for expr in exprs)))


@dataclass(frozen=True)
class NameState:
    internal_compute_map: nisl.Map
    renamed_to_original: Mapping[str, str]
    original_to_renamed: Mapping[str, str]


@dataclass(frozen=True)
class UsageInfo:
    global_usage_map: nisl.Map
    local_storage_maps: Mapping[UsageKey, nisl.Map | nisl.BasicMap]


@dataclass(frozen=True)
class FootprintInfo:
    loopy_footprint: nisl.Set
    named_domain: nisl.Set


@dataclass(frozen=True)
class ReuseRelations:
    shift_relation: nisl.Map
    reusable_footprint: nisl.Set
    refill_footprint: nisl.Set


@dataclass(frozen=True)
class ComputePlan:
    name_state: NameState
    usage_info: UsageInfo
    footprint_info: FootprintInfo
    storage_indices: tuple[str, ...]
    temporal_inames: tuple[str, ...]
    reuse_relations: ReuseRelations | None


@dataclass(frozen=True)
class ComputeInstructionInfo:
    expression: Expression
    dependencies: frozenset[str]
    within_inames: frozenset[str]


def _base_name(name: str) -> str:
    return name.removesuffix("_")


def _cur_name(name: str) -> str:
    return f"{_base_name(name)}_cur"


def _prev_name(name: str) -> str:
    return f"{_base_name(name)}_prev"


def _make_name_state(
    compute_map: nisl.Map,
    storage_indices: Sequence[str],
) -> NameState:
    original_to_renamed = {
        name: f"{name}_"
        for name in compute_map.output_names
        if name not in storage_indices
    }
    renamed_to_original = {
        renamed: original for original, renamed in original_to_renamed.items()
    }
    return NameState(
        internal_compute_map=compute_map.rename_dims(original_to_renamed),
        renamed_to_original=renamed_to_original,
        original_to_renamed=original_to_renamed,
    )


def _infer_temporal_inames(
    compute_map: nisl.Map,
    storage_indices: Sequence[str],
) -> tuple[str, ...]:
    storage_set = frozenset(storage_indices)
    return tuple(name for name in compute_map.output_names if name not in storage_set)


def _basic_set_to_predicates(bset: nisl.BasicSet) -> PredicateSet:
    return frozenset(
        p.Comparison(
            pw_aff_to_expr(constraint.get_aff()),
            "==" if constraint.is_equality() else ">=",
            0,
        )
        for constraint in bset._reconstruct_isl_object().get_constraints()
    )


def _set_to_predicate_options(
    set_: nisl.Set | nisl.BasicSet,
) -> tuple[PredicateSet, ...]:
    if isinstance(set_, nisl.BasicSet):
        if set_._reconstruct_isl_object().is_empty():
            return ()
        return (_basic_set_to_predicates(set_),)

    return tuple(
        _basic_set_to_predicates(bset)
        for bset in set_.get_basic_sets()
        if not bset._reconstruct_isl_object().is_empty()
    )


def _gather_vars(expr: Expression) -> set[str]:
    deps = DependencyMapper()(expr)
    var_names = set()
    for dep in deps:
        if isinstance(dep, p.Variable):
            var_names.add(dep.name)
        elif isinstance(dep, p.Subscript) and isinstance(dep.aggregate, p.Variable):
            var_names.add(dep.aggregate.name)

    return var_names


def _gather_usage_inames(sites: Sequence[UsageSite]) -> frozenset[str]:
    return frozenset(set().union(*(site.domain_names for site in sites)))


def _next_ordinal(counters: dict[str, int], insn_id: str) -> int:
    ordinal = counters.get(insn_id, 0)
    counters[insn_id] = ordinal + 1
    return ordinal


def _normalize_subst_tag(
    tag: Set[Tag] | Sequence[Tag] | Tag | None,
) -> frozenset[Tag] | None:
    if tag is None:
        return None
    if isinstance(tag, Tag):
        return frozenset([tag])
    return frozenset(tag)


def _add_predicates_to_domain(
    domain: nisl.BasicSet,
    predicates: PredicateSet,
) -> nisl.BasicSet:
    predicate_constraints = [str(predicate) for predicate in predicates]
    if not predicate_constraints:
        return domain
    return domain.add_constraint(predicate_constraints)


def _existing_name_mapping(
    map_: nisl.Map | nisl.BasicMap,
    name_mapping: Mapping[str, str],
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
    map_ = map_.project_out([
        renamed_name
        for original_name, renamed_name in name_mapping.items()
        if original_name in names and renamed_name in names
    ])

    names = map_.names
    return map_.rename_dims({
        renamed_name: original_name
        for original_name, renamed_name in name_mapping.items()
        if original_name not in names and renamed_name in names
    })


def _empty_usage_map(local_domain: nisl.BasicSet, range_: nisl.Set) -> nisl.Map:
    map_space = nisl.make_map_from_domain_and_range(
        nisl.make_set(isl.Set.empty(local_domain.get_space())),
        range_,
    ).get_space()
    return nisl.make_map(isl.Map.empty(map_space))


def _map_to_output_exprs(map_: nisl.Map | nisl.BasicMap) -> tuple[Expression, ...]:
    pwmaff = map_.as_pw_multi_aff()
    return tuple(
        pw_aff_to_expr(pwmaff.get_at(dim))
        for dim in range(pwmaff.dim(isl.dim_type.out))
    )


def _map_to_named_output_exprs(
    map_: nisl.Map | nisl.BasicMap,
) -> Mapping[str, Expression]:
    pwmaff = map_.as_pw_multi_aff()
    return {
        pwmaff.get_dim_name(isl.dim_type.out, dim): pw_aff_to_expr(pwmaff.get_at(dim))
        for dim in range(pwmaff.dim(isl.dim_type.out))
    }


class UsageSiteExpressionGatherer(RuleAwareIdentityMapper[[]]):
    def __init__(
        self,
        rule_mapping_ctx: SubstitutionRuleMappingContext,
        subst_name: str,
        subst_tag: Set[Tag] | Tag | None = None,
    ) -> None:
        super().__init__(rule_mapping_ctx)

        self.subst_name: str = subst_name
        self.subst_tag: frozenset[Tag] | None = _normalize_subst_tag(subst_tag)
        self.sites: list[UsageSite] = []
        self._next_ordinal_by_insn: dict[str, int] = {}

    @override
    def map_subst_rule(
        self,
        name: str,
        tags: Set[Tag] | None,
        arguments: Sequence[Expression],
        expn_state: ExpansionState,
    ) -> Expression:
        if name != self.subst_name:
            return super().map_subst_rule(name, tags, arguments, expn_state)

        if self.subst_tag is not None and self.subst_tag != frozenset(tags or ()):
            return super().map_subst_rule(name, tags, arguments, expn_state)

        rule = self.rule_mapping_context.old_subst_rules[name]
        arg_ctx = self.make_new_arg_context(
            name, rule.arguments, arguments, expn_state.arg_context
        )

        self.sites.append(
            UsageSite(
                insn_id=expn_state.insn_id,
                ordinal=_next_ordinal(self._next_ordinal_by_insn, expn_state.insn_id),
                args=tuple(arg_ctx[arg_name] for arg_name in rule.arguments),
                predicates=frozenset(expn_state.instruction.predicates),
            )
        )

        return 0


class RuleInvocationReplacer(RuleAwareIdentityMapper[[]]):
    def __init__(
        self,
        ctx: SubstitutionRuleMappingContext,
        subst_name: str,
        subst_tag: Sequence[Tag] | None,
        usage_descriptors: Mapping[UsageKey, nisl.Map | nisl.BasicMap],
        storage_indices: Sequence[str],
        temporary_name: str,
        compute_insn_ids: str | Sequence[str],
        footprint: nisl.Set,
    ) -> None:
        super().__init__(ctx)

        self.subst_name: str = subst_name
        self.subst_tag: frozenset[Tag] | None = _normalize_subst_tag(subst_tag)
        self.usage_descriptors: Mapping[UsageKey, nisl.Map | nisl.BasicMap] = (
            usage_descriptors
        )
        self.storage_indices: tuple[str, ...] = tuple(storage_indices)
        self.temp_footprint: nisl.Set = footprint.move_dims(
            frozenset(footprint.names) - frozenset(self.storage_indices),
            isl.dim_type.param,
        )
        self.temporary_name: str = temporary_name
        self.compute_dep_ids: frozenset[str] = (
            frozenset([compute_insn_ids])
            if isinstance(compute_insn_ids, str)
            else frozenset(compute_insn_ids)
        )
        self.replaced_something: bool = False
        self._next_ordinal_by_insn: dict[str, int] = {}

    @override
    def map_subst_rule(
        self,
        name: str,
        tags: Set[Tag] | None,
        arguments: Sequence[Expression],
        expn_state: ExpansionState,
    ) -> Expression:
        if name != self.subst_name:
            return super().map_subst_rule(name, tags, arguments, expn_state)

        if self.subst_tag is not None and self.subst_tag != frozenset(tags or ()):
            return super().map_subst_rule(name, tags, arguments, expn_state)

        rule = self.rule_mapping_context.old_subst_rules[name]
        if len(arguments) != len(rule.arguments):
            raise ValueError(
                f"Number of arguments passed to rule {name} "
                f"does not match the signature of {name}."
            )

        access_key = (
            expn_state.insn_id,
            _next_ordinal(self._next_ordinal_by_insn, expn_state.insn_id),
        )

        local_map = self.usage_descriptors.get(access_key)
        if local_map is None:
            return super().map_subst_rule(name, tags, arguments, expn_state)

        if not local_map.range() <= self.temp_footprint:
            return super().map_subst_rule(name, tags, arguments, expn_state)

        self.replaced_something = True
        return var(self.temporary_name)[_map_to_output_exprs(local_map)]

    @override
    def map_kernel(
        self,
        kernel: LoopKernel,
        within: StackMatch = lambda knl, insn, stack: True,
        map_args: bool = True,
        map_tvs: bool = True,
    ) -> LoopKernel:
        new_insns = []
        for insn in kernel.instructions:
            self.replaced_something = False

            if isinstance(
                insn, MultiAssignmentBase
            ) and not contains_a_subst_rule_invocation(kernel, insn):
                new_insns.append(insn)
                continue

            insn = insn.with_transformed_expressions(
                lambda expr, insn=insn: self(expr, kernel, insn)
            )

            if self.replaced_something:
                insn = insn.copy(depends_on=insn.depends_on | self.compute_dep_ids)

            new_insns.append(insn)

        return kernel.copy(instructions=new_insns)


def _gather_usage_sites(
    kernel: LoopKernel,
    substitution: str,
) -> tuple[UsageSite, ...]:
    ctx = SubstitutionRuleMappingContext(
        kernel.substitutions, kernel.get_var_name_generator()
    )
    gatherer = UsageSiteExpressionGatherer(ctx, substitution)

    _ = gatherer.map_kernel(kernel)
    return tuple(gatherer.sites)


def _build_usage_info(
    named_domain: nisl.BasicSet,
    name_state: NameState,
    storage_indices: Sequence[str],
    temporal_inames: Sequence[str],
    sites: Sequence[UsageSite],
) -> UsageInfo:
    if not sites:
        raise ValueError(
            "compute() did not find any invocation of the requested substitution rule."
        )

    usage_inames = _gather_usage_inames(sites)
    local_domain = named_domain.project_out_except(usage_inames)
    global_usage_map = _empty_usage_map(
        local_domain, name_state.internal_compute_map.domain()
    )

    storage_set = frozenset(storage_indices)
    temporal_set = frozenset(temporal_inames)
    usage_descriptors: dict[UsageKey, nisl.Map | nisl.BasicMap] = {}

    for site in sites:
        local_domain = _add_predicates_to_domain(
            named_domain.project_out_except(site.domain_names),
            site.predicates,
        )
        usage_mpwaff = multi_pw_aff_from_exprs(site.args, global_usage_map.get_space())
        local_usage_map = nisl.make_map(usage_mpwaff.as_map()).intersect_domain(
            local_domain
        )
        global_usage_map = global_usage_map | local_usage_map

        local_storage_map = local_usage_map.apply_range(name_state.internal_compute_map)
        local_storage_map = _normalize_renamed_dims(
            local_storage_map, name_state.original_to_renamed
        )
        if not local_usage_map.domain() <= local_storage_map.domain():
            continue

        non_param_names = (usage_inames - temporal_set) | storage_set
        usage_descriptors[site.key] = local_storage_map.move_dims(
            frozenset(local_storage_map.names) - non_param_names,
            isl.dim_type.param,
        )

    return UsageInfo(
        global_usage_map=global_usage_map,
        local_storage_maps=usage_descriptors,
    )


def _build_footprint_info(
    named_domain: nisl.BasicSet,
    name_state: NameState,
    usage_info: UsageInfo,
    storage_indices: Sequence[str],
    temporal_inames: Sequence[str],
) -> FootprintInfo:
    storage_set = frozenset(storage_indices)
    temporal_set = frozenset(temporal_inames)

    storage_map = usage_info.global_usage_map.apply_range(
        name_state.internal_compute_map
    )
    storage_map = _normalize_renamed_dims(storage_map, name_state.original_to_renamed)

    storage_map = storage_map.move_dims(temporal_set, isl.dim_type.param)
    exact_footprint = storage_map.range()
    exact_footprint = exact_footprint.project_out_except(temporal_set | storage_set)
    exact_footprint = exact_footprint.move_dims(temporal_set, isl.dim_type.set)

    # Loopy domains are still restricted to a single BasicSet in this path.
    footprint_isl = exact_footprint._reconstruct_isl_object()
    loopy_footprint = nisl.make_set(isl.Set.from_basic_set(footprint_isl.convex_hull()))

    loopy_domain = named_domain & loopy_footprint
    if len(loopy_domain.get_basic_sets()) != 1:
        raise ValueError("New domain should be composed of a single basic set")

    return FootprintInfo(loopy_footprint=loopy_footprint, named_domain=loopy_domain)


def _build_compute_plan(
    compute_map: nisl.Map,
    named_domain: nisl.BasicSet,
    sites: Sequence[UsageSite],
    storage_indices: Sequence[str],
    temporal_inames: Sequence[str],
    inames_to_advance: Sequence[str] | Literal["auto"] | None,
) -> ComputePlan:
    name_state = _make_name_state(compute_map, storage_indices)
    usage_info = _build_usage_info(
        named_domain,
        name_state,
        storage_indices,
        temporal_inames,
        sites,
    )
    footprint_info = _build_footprint_info(
        named_domain,
        name_state,
        usage_info,
        storage_indices,
        temporal_inames,
    )

    if inames_to_advance == "auto":
        inames_to_advance = _detect_inames_to_advance(
            name_state.internal_compute_map,
            usage_info.global_usage_map,
            footprint_info.loopy_footprint,
            storage_indices,
            temporal_inames,
        )

    reuse_relations = (
        None
        if inames_to_advance is None
        else _build_reuse_relations(
            name_state.internal_compute_map,
            usage_info.global_usage_map,
            footprint_info.loopy_footprint,
            footprint_info.named_domain,
            storage_indices,
            temporal_inames,
            frozenset(inames_to_advance),
        )
    )

    return ComputePlan(
        name_state=name_state,
        usage_info=usage_info,
        footprint_info=footprint_info,
        storage_indices=tuple(storage_indices),
        temporal_inames=tuple(temporal_inames),
        reuse_relations=reuse_relations,
    )


def _build_reuse_relations(
    compute_map: nisl.Map,
    global_usage_map: nisl.Map,
    footprint: nisl.Set,
    named_domain: nisl.Set,
    storage_indices: Sequence[str],
    temporal_inames: Sequence[str],
    advancing_set: frozenset[str],
) -> ReuseRelations:
    predecessor_context = _build_predecessor_context(temporal_inames, advancing_set)
    shift_relation = _build_shift_relation(
        compute_map,
        global_usage_map,
        footprint,
        storage_indices,
        predecessor_context,
    )
    reusable_footprint = shift_relation.range()
    current_footprint = footprint.rename_dims({
        name: _cur_name(name) for name in footprint.names
    })
    refill_footprint = current_footprint - reusable_footprint

    normal_names = {_cur_name(name): name for name in footprint.names}
    reusable_footprint = reusable_footprint.rename_dims(normal_names)
    refill_footprint = refill_footprint.rename_dims(normal_names)

    reusable_footprint = reusable_footprint.gist(
        named_domain.project_out_except(reusable_footprint.names)
    )
    refill_footprint = refill_footprint.gist(
        named_domain.project_out_except(refill_footprint.names)
    )

    return ReuseRelations(
        shift_relation=shift_relation,
        reusable_footprint=reusable_footprint,
        refill_footprint=refill_footprint,
    )


def _build_predecessor_context(
    temporal_inames: Sequence[str],
    advancing_set: frozenset[str],
) -> nisl.Map:
    constraints = [
        (
            f"{_cur_name(name)} = {_prev_name(name)} + 1"
            if name in advancing_set
            else f"{_cur_name(name)} = {_prev_name(name)}"
        )
        for name in temporal_inames
    ]

    return nisl.make_map(
        "{ ["
        + ", ".join(_prev_name(name) for name in temporal_inames)
        + "] -> ["
        + ", ".join(_cur_name(name) for name in temporal_inames)
        + "]"
        + (f" : {' and '.join(constraints)}" if constraints else "")
        + " }"
    )


def _build_shift_relation(
    compute_map: nisl.Map,
    global_usage_map: nisl.Map,
    footprint: nisl.Set,
    storage_indices: Sequence[str],
    predecessor_context: nisl.Map,
) -> nisl.Map:
    compute_map_cur = compute_map.rename_dims({
        name: _cur_name(name) for name in compute_map.output_names
    })
    compute_map_prev = compute_map.rename_dims({
        name: _prev_name(name) for name in compute_map.output_names
    })

    reuse_map = (
        global_usage_map
        .apply_range(compute_map_prev)
        .reverse()
        .apply_range(global_usage_map.apply_range(compute_map_cur))
    )
    reuse_map = reuse_map & predecessor_context

    current_footprint = footprint.rename_dims({
        name: _cur_name(name) for name in footprint.names
    })
    previous_footprint = footprint.rename_dims({
        name: _prev_name(name) for name in footprint.names
    })

    reuse_map = reuse_map.intersect_domain(previous_footprint)
    reuse_map = reuse_map.intersect_range(current_footprint)

    return reuse_map - _identity_storage_map(footprint, storage_indices)


def _detect_inames_to_advance(
    compute_map: nisl.Map,
    global_usage_map: nisl.Map,
    footprint: nisl.Set,
    storage_indices: Sequence[str],
    temporal_inames: Sequence[str],
) -> tuple[str, ...]:
    candidates = []
    for name in temporal_inames:
        shift_relation = _build_shift_relation(
            compute_map,
            global_usage_map,
            footprint,
            storage_indices,
            _build_predecessor_context(temporal_inames, frozenset([name])),
        )
        if not shift_relation._reconstruct_isl_object().is_empty():
            candidates.append(name)

    if len(candidates) > 1:
        raise ValueError(
            "Could not infer a unique advancing iname. "
            f"Candidates are {candidates}; pass inames_to_advance explicitly."
        )

    return tuple(candidates)


def _identity_storage_map(
    footprint: nisl.Set,
    storage_indices: Sequence[str],
) -> nisl.Map:
    return nisl.make_map(
        "{ ["
        + ", ".join(_prev_name(name) for name in footprint.names)
        + "] -> ["
        + ", ".join(_cur_name(name) for name in footprint.names)
        + "] : "
        + " and ".join(
            f"{_cur_name(name)} = {_prev_name(name)}" for name in storage_indices
        )
        + " }"
    )


def _make_shift_instructions(
    reuse_map: nisl.Map,
    reused_current: nisl.Set,
    storage_indices: Sequence[str],
    temporal_inames: Sequence[str],
    temporary_name: str,
    compute_insn_id: str,
) -> tuple[tuple[InstructionBase, ...], tuple[str, ...], frozenset[str]]:
    storage_reuse_map = reuse_map.project_out_except(
        frozenset(_prev_name(name) for name in storage_indices)
        | frozenset(_cur_name(name) for name in storage_indices)
    )
    storage_reuse_map = storage_reuse_map.rename_dims({
        _cur_name(name): name for name in storage_indices
    })

    cur_to_prev_exprs = _map_to_named_output_exprs(storage_reuse_map.reverse())
    prev_storage_exprs = tuple(
        cur_to_prev_exprs[_prev_name(name)] for name in storage_indices
    )

    shift_assignee = var(temporary_name)[tuple(var(idx) for idx in storage_indices)]
    shift_expression = var(temporary_name)[prev_storage_exprs]

    update_insns = []
    update_ids = []
    current_deps: frozenset[str] = frozenset()
    shift_predicate_options = _set_to_predicate_options(reused_current)

    for i, predicates in enumerate(shift_predicate_options):
        shift_insn_id = (
            f"{compute_insn_id}_shift"
            if len(shift_predicate_options) == 1
            else f"{compute_insn_id}_shift_{i}"
        )
        update_insns.append(
            Assignment(
                id=shift_insn_id,
                assignee=shift_assignee,
                expression=shift_expression,
                within_inames=frozenset(temporal_inames) | frozenset(storage_indices),
                predicates=predicates,
                depends_on=current_deps,
            )
        )
        update_ids.append(shift_insn_id)
        current_deps = frozenset([shift_insn_id])

    return tuple(update_insns), tuple(update_ids), current_deps


def _build_compute_instruction_info(
    kernel: LoopKernel,
    substitution: str,
    name_state: NameState,
) -> ComputeInstructionInfo:
    compute_map = name_state.internal_compute_map.rename_dims(
        name_state.renamed_to_original
    )
    compute_pw_aff = compute_map.reverse().as_pw_multi_aff()
    storage_axis_to_global_expr = {
        compute_pw_aff.get_dim_name(isl.dim_type.out, dim): pw_aff_to_expr(
            compute_pw_aff.get_at(dim)
        )
        for dim in range(compute_pw_aff.dim(isl.dim_type.out))
    }

    ctx = SubstitutionRuleMappingContext(
        kernel.substitutions, kernel.get_var_name_generator()
    )
    expr_subst_map = RuleAwareSubstitutionMapper(
        ctx,
        make_subst_func(storage_axis_to_global_expr),
        within=parse_stack_match(None),
    )

    compute_expression = expr_subst_map(
        kernel.substitutions[substitution].expression,
        kernel,
        cast("InstructionBase", cast("object", None)),
    )

    dependencies = frozenset().union(
        *(
            kernel.writer_map().get(var_name, frozenset())
            for var_name in _gather_vars(compute_expression)
        )
    )

    return ComputeInstructionInfo(
        expression=compute_expression,
        dependencies=dependencies,
        within_inames=frozenset(compute_map.output_names),
    )


def _add_update_and_compute_instructions(
    kernel: LoopKernel,
    update_insns: Sequence[InstructionBase],
    update_ids: Sequence[str],
    refill_options: PredicateOptions,
    final_deps: frozenset[str],
    compute_info: ComputeInstructionInfo,
    storage_indices: Sequence[str],
    temporary_name: str,
    compute_insn_id: str,
) -> tuple[LoopKernel, tuple[str, ...]]:
    new_insns = [*kernel.instructions, *update_insns]
    update_ids = list(update_ids)
    current_deps = final_deps
    assignee = var(temporary_name)[tuple(var(idx) for idx in storage_indices)]

    for i, predicates in enumerate(refill_options):
        refill_insn_id = (
            compute_insn_id
            if len(refill_options) == 1
            else f"{compute_insn_id}_refill_{i}"
        )
        new_insns.append(
            Assignment(
                id=refill_insn_id,
                assignee=assignee,
                expression=compute_info.expression,
                within_inames=compute_info.within_inames,
                predicates=predicates,
                depends_on=current_deps | compute_info.dependencies,
            )
        )
        update_ids.append(refill_insn_id)
        current_deps = frozenset([refill_insn_id])

    return kernel.copy(instructions=new_insns), tuple(update_ids)


def _add_temporary(
    kernel: LoopKernel,
    footprint: nisl.Set,
    storage_indices: Sequence[str],
    temporary_name: str,
    temporary_address_space: AddressSpace | None,
    temporary_dtype: ToLoopyTypeConvertible,
) -> LoopKernel:
    loopy_type = to_loopy_type(temporary_dtype, allow_none=True)
    bounds = tuple(
        (pw_aff_to_expr(footprint.dim_min(dim)), pw_aff_to_expr(footprint.dim_max(dim)))
        for dim in storage_indices
    )
    base_indices = tuple(lower for lower, _upper in bounds)
    temp_shape = tuple(upper - lower + 1 for lower, upper in bounds)

    new_temp_vars = dict(kernel.temporary_variables)
    new_temp_vars[temporary_name] = TemporaryVariable(
        name=temporary_name,
        dtype=loopy_type,
        base_indices=base_indices,
        shape=temp_shape,
        address_space=temporary_address_space,
        dim_names=tuple(storage_indices),
    )
    return kernel.copy(temporary_variables=new_temp_vars)


def _lower_compute_plan(
    kernel: LoopKernel,
    substitution: str,
    plan: ComputePlan,
    domain_changer: DomainChanger,
    temporary_name: str,
    temporary_address_space: AddressSpace | None,
    temporary_dtype: ToLoopyTypeConvertible,
    compute_insn_id: str,
) -> LoopKernel:
    domain = plan.footprint_info.named_domain.get_basic_sets()[
        0
    ]._reconstruct_isl_object()
    kernel = kernel.copy(domains=domain_changer.get_domains_with(domain))

    update_insns: tuple[InstructionBase, ...] = ()
    update_insn_ids: tuple[str, ...] = ()
    refill_options: PredicateOptions = (None,)
    final_deps: frozenset[str] = frozenset()
    if plan.reuse_relations is not None:
        update_insns, update_insn_ids, final_deps = _make_shift_instructions(
            plan.reuse_relations.shift_relation,
            plan.reuse_relations.reusable_footprint,
            plan.storage_indices,
            plan.temporal_inames,
            temporary_name,
            compute_insn_id,
        )
        refill_options = _set_to_predicate_options(
            plan.reuse_relations.refill_footprint
        )

    compute_info = _build_compute_instruction_info(
        kernel, substitution, plan.name_state
    )
    kernel, update_insn_ids = _add_update_and_compute_instructions(
        kernel,
        update_insns,
        update_insn_ids,
        refill_options,
        final_deps,
        compute_info,
        plan.storage_indices,
        temporary_name,
        compute_insn_id,
    )

    ctx = SubstitutionRuleMappingContext(
        kernel.substitutions, kernel.get_var_name_generator()
    )
    kernel = RuleInvocationReplacer(
        ctx,
        substitution,
        None,
        plan.usage_info.local_storage_maps,
        plan.storage_indices,
        temporary_name,
        update_insn_ids,
        plan.footprint_info.loopy_footprint,
    ).map_kernel(kernel)

    return _add_temporary(
        kernel,
        plan.footprint_info.loopy_footprint,
        plan.storage_indices,
        temporary_name,
        temporary_address_space,
        temporary_dtype,
    )


@for_each_kernel
def compute(
    kernel: LoopKernel,
    substitution: str,
    compute_map: nisl.Map,
    storage_indices: Sequence[str],
    temporal_inames: Sequence[str] | None = None,
    inames_to_advance: Sequence[str] | Literal["auto"] | None = None,
    temporary_name: str | None = None,
    temporary_address_space: AddressSpace | None = None,
    temporary_dtype: ToLoopyTypeConvertible = None,
    compute_insn_id: str | None = None,
) -> LoopKernel:
    """Compute a substitution into a temporary and replace covered uses."""
    temporary_name = temporary_name or f"{substitution}_temp"
    compute_insn_id = compute_insn_id or f"{substitution}_compute"
    if temporal_inames is None:
        temporal_inames = _infer_temporal_inames(compute_map, storage_indices)

    domain_changer = DomainChanger(kernel, kernel.all_inames())
    named_domain = nisl.make_basic_set(domain_changer.domain)

    plan = _build_compute_plan(
        compute_map,
        named_domain,
        _gather_usage_sites(kernel, substitution),
        tuple(storage_indices),
        temporal_inames,
        inames_to_advance,
    )
    return _lower_compute_plan(
        kernel,
        substitution,
        plan,
        domain_changer,
        temporary_name,
        temporary_address_space,
        temporary_dtype,
        compute_insn_id,
    )

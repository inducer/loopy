from collections.abc import Mapping, Sequence, Set
from typing import override
from typing_extensions import TypeAlias
import loopy as lp
from loopy.kernel.tools import DomainChanger
from loopy.types import ToLoopyTypeConvertible, to_loopy_type
import namedisl as nisl

from loopy.kernel import LoopKernel
from loopy.kernel.data import AddressSpace
from loopy.match import StackMatch, parse_stack_match
from loopy.symbolic import (
    ExpansionState,
    RuleAwareIdentityMapper,
    RuleAwareSubstitutionMapper,
    SubstitutionRuleExpander,
    SubstitutionRuleMappingContext,
    multi_pw_aff_from_exprs,
    pw_aff_to_expr,
)
from loopy.transform.precompute import (
    contains_a_subst_rule_invocation
)
from loopy.translation_unit import for_each_kernel
from pymbolic import var
from pymbolic.mapper.substitutor import make_subst_func

import islpy as isl
import pymbolic.primitives as p
from pymbolic.mapper.dependency import DependencyMapper
from pymbolic.typing import Expression
from pytools.tag import Tag


AccessTuple: TypeAlias = tuple[Expression, ...]


# helper for gathering names of variables in pymbolic expressions
def _gather_vars(expr: Expression) -> set[str]:
    deps = DependencyMapper()(expr)
    return {
        dep.name
        for dep in deps
        if isinstance(dep, p.Variable)
    }


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
            usage_descriptors: Mapping[AccessTuple, nisl.Map],
            storage_indices: Sequence[str],
            temporary_name: str,
            compute_insn_id: str,
            footprint: nisl.Set
        ) -> None:

        super().__init__(ctx)

        self.subst_name: str = subst_name
        self.subst_tag: Sequence[Tag] | None = subst_tag

        self.usage_descriptors: Mapping[AccessTuple, nisl.Map] = \
            usage_descriptors
        self.storage_indices: Sequence[str] = storage_indices
        self.footprint: nisl.Set = footprint

        self.temporary_name: str = temporary_name
        self.compute_insn_id: str = compute_insn_id

        self.replaced_something: bool = False

        # FIXME: may not always be the case (i.e. global barrier between
        # compute insn and uses)
        self.compute_dep_id: str = compute_insn_id


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

        if not name == self.subst_name:
            return super().map_subst_rule(name, tags, arguments, expn_state)

        if not tuple(args) in self.usage_descriptors:
            return super().map_subst_rule(name, tags, arguments, expn_state)

        if not len(arguments) == len(rule.arguments):
            raise ValueError("Number of arguments passed to rule {name} ",
                             "does not match the signature of {name}.")

        local_map = self.usage_descriptors[tuple(args)]
        temp_footprint = self.footprint.move_dims(
            frozenset(self.footprint.names) - frozenset(self.storage_indices),
            isl.dim_type.param
        )

        if not local_map.range() <= temp_footprint:
            return super().map_subst_rule(name, tags, arguments, expn_state)

        # }}}

        # {{{ get index expression in terms of global inames

        local_pwmaff = self.usage_descriptors[tuple(args)].as_pw_multi_aff()

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
                lambda expr: self(expr, kernel, insn)
            )

            if self.replaced_something:
                insn = insn.copy(
                    depends_on=(
                        insn.depends_on | frozenset([self.compute_insn_id])
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

        temporal_inames: Sequence[str],

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

    # {{{ setup and useful items

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
    usage_domain = named_domain.project_out_except(usage_inames)

    # FIXME: gross. find a cleaner way to generate a space for an empty map
    global_usage_map = nisl.make_map_from_domain_and_range(
        nisl.make_set(isl.Set.empty(usage_domain.get_space())),
        compute_map.domain()
    )
    global_usage_map = nisl.make_map(isl.Map.empty(global_usage_map.get_space()))

    usage_substs: Mapping[AccessTuple, nisl.Map] = {}
    for usage in usage_exprs:

        # {{{ compute local usage map, update global usage map

        local_usage_mpwaff = multi_pw_aff_from_exprs(
            usage,
            global_usage_map.get_space()
        )

        local_usage_map = nisl.make_map(local_usage_mpwaff.as_map())

        local_usage_map = local_usage_map.intersect_domain(usage_domain)
        global_usage_map = global_usage_map | local_usage_map

        # }}}

        # {{{ compute storage map

        local_storage_map = local_usage_map.apply_range(compute_map)

        # check that no restrictions happened during composition (i.e. tile
        # valid for a single point in the domain)
        if not local_usage_map.domain() <= local_storage_map.domain():
            continue

        non_param_names = (usage_inames - temporal_set) | storage_set
        parameter_names = frozenset(local_storage_map.names) - non_param_names

        local_storage_map = local_storage_map.move_dims(parameter_names,
                                                        isl.dim_type.param)

        # }}}

        usage_substs[tuple(usage)] = local_storage_map

    global_usage_map = global_usage_map.apply_range(compute_map)

    # }}}

    # {{{ compute bounds and update kernel domain

    global_usage_map = global_usage_map.move_dims(
        temporal_set,
        isl.dim_type.param
    )
    footprint = global_usage_map.range()

    # clean up ticked duplicate names
    footprint = footprint.project_out_except(temporal_set | storage_set)
    footprint = footprint.move_dims(temporal_set, isl.dim_type.set)

    # {{{ FIXME: use Sets instead of BasicSets when loopy is ready

    footprint = nisl.make_set(footprint._reconstruct_isl_object().convex_hull())
    named_domain = named_domain & footprint

    if len(named_domain.get_basic_sets()) != 1:
        raise ValueError("New domain should be composed of a single basic set")

    # FIXME: use named object once loopy is name-ified
    domain = named_domain.get_basic_sets()[0]._reconstruct_isl_object()
    new_domains = domain_changer.get_domains_with(domain)

    # }}}

    kernel = kernel.copy(domains=new_domains)

    # }}}

    # {{{ create compute instruction in kernel

    # FIXME:
    compute_pw_aff = compute_map.reverse().as_pw_multi_aff()
    storage_ax_to_global_expr = {
        compute_pw_aff.get_dim_name(isl.dim_type.out, dim) :
        pw_aff_to_expr(compute_pw_aff.get_at(dim))
       for dim in range(compute_pw_aff.dim(isl.dim_type.out))
    }

    expr_subst_map = RuleAwareSubstitutionMapper(
        ctx,
        make_subst_func(storage_ax_to_global_expr),
        within=parse_stack_match(None)
    )

    subst_expr = kernel.substitutions[substitution].expression
    compute_expression = expr_subst_map(subst_expr, kernel, None)

    if not temporary_name:
        temporary_name = substitution + "_temp"

    assignee = var(temporary_name)[tuple(var(idx) for idx in storage_indices)]

    within_inames = compute_map.output_names

    if not compute_insn_id:
        compute_insn_id = substitution + "_compute"
    compute_insn = lp.Assignment(
        id=compute_insn_id,
        assignee=assignee,
        expression=compute_expression,
        within_inames=within_inames
    )

    new_insns = list(kernel.instructions)
    new_insns.append(compute_insn)
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
        compute_insn_id,
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

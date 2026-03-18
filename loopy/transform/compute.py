from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import override
import loopy as lp
from loopy.kernel.tools import DomainChanger
from loopy.types import to_loopy_type
import namedisl as nisl

from loopy.kernel import LoopKernel
from loopy.kernel.data import AddressSpace, SubstitutionRule
from loopy.match import StackMatch, parse_stack_match
from loopy.symbolic import (
    ExpansionState,
    RuleAwareIdentityMapper,
    RuleAwareSubstitutionMapper,
    SubstitutionRuleExpander,
    SubstitutionRuleMappingContext,
    get_dependencies,
    pw_aff_to_expr,
    pwaff_from_expr
)
from loopy.transform.precompute import (
    contains_a_subst_rule_invocation
)
from loopy.translation_unit import for_each_kernel
from pymbolic import ArithmeticExpression, var
from pymbolic.mapper.substitutor import make_subst_func

import islpy as isl
import pymbolic.primitives as p
from pymbolic.mapper.dependency import DependencyMapper
from pymbolic.typing import Expression
from pytools.tag import Tag


def gather_vars(expr) -> set[str]:
    deps = DependencyMapper()(expr)
    return {
        dep.name
        for dep in deps
        if isinstance(dep, p.Variable)
    }


def space_from_exprs(exprs, ctx=isl.DEFAULT_CONTEXT):
    names = sorted(set().union(*(gather_vars(expr) for expr in exprs)))
    set_names = [name for name in names]

    return isl.Space.create_from_names(
        ctx,
        set=set_names
    )

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


class RuleInvocationReplacer(RuleAwareIdentityMapper[[]]):
    def __init__(
            self,
            ctx: SubstitutionRuleMappingContext,
            subst_name: str,
            subst_tag: Sequence[Tag] | None,
            usage_descriptors: Mapping[tuple[Expression, ...], isl.Map],
            storage_indices: Sequence[str],
            temporary_name: str,
            compute_insn_id: str,
            compute_map: isl.Map
        ) -> None:

        super().__init__(ctx)

        self.subst_name: str = subst_name
        self.subst_tag: Sequence[Tag] | None = subst_tag

        self.usage_descriptors: Mapping[tuple[Expression, ...], isl.Map] = \
            usage_descriptors
        self.storage_indices: Sequence[str] = storage_indices

        self.temporary_name: str = temporary_name
        self.compute_insn_id: str = compute_insn_id

        # FIXME: may not always be the case (i.e. global barrier between
        # compute insn and uses)
        self.compute_dep_id: str = compute_insn_id

        self.replaced_something: bool = False


    @override
    def map_subst_rule(
            self,
            name: str,
            tags: Set[Tag] | None,
            arguments: Sequence[Expression],
            expn_state: ExpansionState
        ) -> Expression:

        if not name == self.subst_name:
            return super().map_subst_rule(name, tags, arguments, expn_state)

        rule = self.rule_mapping_context.old_subst_rules[name]
        arg_ctx = self.make_new_arg_context(
            name, rule.arguments, arguments, expn_state.arg_context
        )
        args = [arg_ctx[arg_name] for arg_name in rule.arguments]

        # FIXME: footprint check? likely required if user supplies bounds on
        # storage indices because we are not guaranteed to capture the footprint
        # of all usage sites

        if not len(arguments) == len(rule.arguments):
            raise ValueError("Number of arguments passed to rule {name} ",
                             "does not match the signature of {name}.")

        index_exprs: Sequence[Expression] = []

        # FIXME: make self.usage_descriptors a constantdict
        local_pwmaff = self.usage_descriptors[tuple(args)].as_pw_multi_aff()

        for dim in range(local_pwmaff.dim(isl.dim_type.out)):
            index_exprs.append(pw_aff_to_expr(local_pwmaff.get_at(dim)))

        new_expression = var(self.temporary_name)[tuple(index_exprs)]

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


@for_each_kernel
def compute(
        kernel: LoopKernel,
        substitution: str,
        compute_map: nisl.Map,
        storage_indices: Sequence[str],

        # NOTE: how can we deduce this?
        temporal_inames: Sequence[str],

        temporary_name: str | None = None,
        temporary_address_space: AddressSpace | None = None
    ) -> LoopKernel:
    """
    Inserts an instruction to compute an expression given by :arg:`substitution`
    and replaces all invocations of :arg:`substitution` with the result of the
    compute instruction.

    :arg substitution: The substitution rule for which the compute
    transform should be applied.

    :arg compute_map: An :class:`isl.Map` representing a relation between
    substitution rule indices and tuples `(a, l)`, where `a` is a vector of
    storage indices and `l` is a vector of "timestamps".

    :arg storage_indices: An ordered sequence of names of storage indices. Used
    to create inames for the loops that cover the required set of compute points.
    """
    compute_map = compute_map._reconstruct_isl_object()

    # construct union of usage footprints to determine bounds on compute inames
    ctx = SubstitutionRuleMappingContext(
        kernel.substitutions, kernel.get_var_name_generator())
    expander = SubstitutionRuleExpander(kernel.substitutions)
    expr_gatherer = UsageSiteExpressionGatherer(
        ctx, expander, kernel, substitution, None
    )

    _ = expr_gatherer.map_kernel(kernel)
    usage_exprs = expr_gatherer.usage_expressions

    all_exprs = [
        expr
        for usage in usage_exprs
        for expr in usage
    ]

    space = space_from_exprs(all_exprs)

    footprint = isl.Set.empty(
        isl.Space.create_from_names(
            ctx=space.get_ctx(),
            set=list(storage_indices)
        )
    )

    usage_descrs: Mapping[tuple[Expression, ...], isl.Map] = {}
    for usage in usage_exprs:

        range_space = isl.Space.create_from_names(
            ctx=space.get_ctx(),
            set=list(storage_indices)
        )
        map_space = space.map_from_domain_and_range(range_space)

        pw_multi_aff = isl.MultiPwAff.zero(map_space)

        for i, arg in enumerate(usage):
            pw_multi_aff = pw_multi_aff.set_pw_aff(
                i,
                pwaff_from_expr(space, arg)
            )

        usage_map = pw_multi_aff.as_map()

        iname_to_timespace = usage_map.apply_range(compute_map)
        iname_to_storage = iname_to_timespace.project_out_except(
            storage_indices, [isl.dim_type.out]
        )

        local_map = iname_to_storage.project_out_except(
            kernel.all_inames() - frozenset(temporal_inames),
            [isl.dim_type.in_]
        )

        footprint = footprint | iname_to_storage.range()

        usage_descrs[tuple(usage)] = local_map

    # add compute inames to domain / kernel
    domain_changer = DomainChanger(kernel, kernel.all_inames())
    domain = domain_changer.domain

    footprint_tmp, domain = isl.align_two(footprint, domain)
    domain = (domain & footprint_tmp).get_basic_sets()[0]

    new_domains = domain_changer.get_domains_with(domain)
    kernel = kernel.copy(domains=new_domains)

    # create compute instruction in kernel
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

    assignee = var(temporary_name)[tuple(
        var(iname) for iname in storage_indices
    )]

    within_inames = frozenset(
        compute_map.get_dim_name(isl.dim_type.out, dim)
        for dim in range(compute_map.dim(isl.dim_type.out))
    )

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

    ctx = SubstitutionRuleMappingContext(
        kernel.substitutions, kernel.get_var_name_generator()
    )

    replacer = RuleInvocationReplacer(
        ctx,
        substitution,
        None,
        usage_descrs,
        storage_indices,
        temporary_name,
        compute_insn_id,
        compute_map
    )

    kernel = replacer.map_kernel(kernel)

    # FIXME: accept dtype as an argument
    import numpy as np
    loopy_type = to_loopy_type(np.float64, allow_none=True)

    # WARNING: this can result in symbolic shapes, is that allowed?
    temp_shape = tuple(
        pw_aff_to_expr(footprint.dim_max(dim)) + 1
        for dim in range(footprint.dim(isl.dim_type.out))
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

    # FIXME: handle iname tagging

    return kernel

import loopy as lp
from loopy.kernel.tools import DomainChanger
import namedisl as nisl

from loopy.kernel import LoopKernel
from loopy.kernel.data import AddressSpace
from loopy.match import parse_stack_match
from loopy.symbolic import (
    RuleAwareIdentityMapper,
    RuleAwareSubstitutionMapper,
    SubstitutionRuleMappingContext,
    pw_aff_to_expr,
    pwaff_from_expr
)
from loopy.transform.precompute import (
    RuleInvocationGatherer,
    contains_a_subst_rule_invocation
)
from loopy.translation_unit import for_each_kernel
from pymbolic import var
from pymbolic.mapper.substitutor import make_subst_func

import islpy as isl
import pymbolic.primitives as p
from pymbolic.mapper.dependency import DependencyMapper

from pymbolic.mapper import IdentityMapper


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

@for_each_kernel
def compute(
        kernel: LoopKernel,
        substitution: str,
        compute_map: nisl.Map,
        storage_indices: frozenset[str],
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

    :arg storage_indices: A :class:`frozenset` of names of storage indices. Used
    to create inames for the loops that cover the required footprint.
    """
    compute_map = compute_map._reconstruct_isl_object()

    # construct union of usage footprints to determine bounds on compute inames
    ctx = SubstitutionRuleMappingContext(
        kernel.substitutions, kernel.get_var_name_generator())
    inv_gatherer = RuleInvocationGatherer(
        ctx, kernel, substitution, None, parse_stack_match(None)
    )

    for insn in kernel.instructions:
        if (isinstance(insn, lp.MultiAssignmentBase) and
            contains_a_subst_rule_invocation(kernel, insn)):
            for assignee in insn.assignees:
                _ = inv_gatherer(assignee, kernel, insn)
            _ = inv_gatherer(insn.expression, kernel, insn)

    access_descriptors = inv_gatherer.access_descriptors

    acc_desc_exprs = [
        arg
        for ad in access_descriptors
        if ad.args is not None
        for arg in ad.args
    ]

    space = space_from_exprs(acc_desc_exprs)

    footprint = isl.Set.empty(isl.Space.create_from_names(
        ctx=space.get_ctx(),
        set=list(storage_indices)
    ))
    for ad in access_descriptors:
        if not ad.args:
            continue

        nout = len(ad.args)

        range_space = isl.Space.alloc(space.get_ctx(), 0, nout, 0).domain()
        map_space = space.map_from_domain_and_range(range_space)
        pw_multi_aff = isl.MultiPwAff.zero(map_space)

        for i, arg in enumerate(ad.args):
            if arg is not None:
                pw_multi_aff = pw_multi_aff.set_pw_aff(
                    i,
                    pwaff_from_expr(space, arg)
                )

        usage_map = pw_multi_aff.as_map()
        iname_to_timespace = usage_map.apply_range(compute_map).coalesce()
        iname_to_storage = iname_to_timespace.project_out_except(
            storage_indices, [isl.dim_type.out]
        )

        footprint = footprint | iname_to_storage.range()

    # add compute inames to domain / kernel
    domain_changer = DomainChanger(kernel, kernel.all_inames())
    domain = domain_changer.domain

    footprint, domain = isl.align_two(footprint, domain)
    domain = domain & footprint

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

    print(kernel)
    return kernel

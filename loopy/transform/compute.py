import islpy as isl
import namedisl as nisl

import loopy as lp
from loopy.kernel import LoopKernel
from loopy.kernel.data import AddressSpace
from loopy.kernel.instruction import MultiAssignmentBase
from loopy.match import parse_stack_match
from loopy.symbolic import (
    RuleAwareSubstitutionMapper,
    SubstitutionRuleMappingContext,
    pw_aff_to_expr
)
from loopy.transform.precompute import contains_a_subst_rule_invocation
from loopy.translation_unit import for_each_kernel

from pymbolic import var
from pymbolic.mapper.substitutor import make_subst_func

from pytools.tag import Tag


@for_each_kernel
def compute(
        kernel: LoopKernel,
        substitution: str,
        compute_map: isl.Map | nisl.Map,
        storage_inames: list[str],
        default_tag: Tag | str | None = None,
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
    """
    if isinstance(compute_map, isl.Map):
        compute_map = nisl.make_map(compute_map)

    if not temporary_address_space:
        temporary_address_space = AddressSpace.GLOBAL

    # {{{ normalize names

    iname_to_storage_map = {
        iname : (iname + "_store" if iname in kernel.all_inames() else iname)
        for iname in storage_inames
    }

    compute_map = compute_map.rename_dims(iname_to_storage_map)

    # }}}

    # {{{ update kernel domain to contain storage inames

    new_storage_axes = list(iname_to_storage_map.values())

    # FIXME: use DomainChanger to add domain to kernel
    storage_domain = compute_map.range().project_out_except(new_storage_axes)
    new_domain = kernel.domains[0]

    # }}}

    # {{{ express substitution inputs as pw affs of (storage, time) names

    compute_pw_aff = compute_map.reverse().as_pw_multi_aff()

    storage_ax_to_global_expr = {
        dim_name : pw_aff_to_expr(compute_pw_aff.get_at(dim_name))
        for dim_name in compute_map.dim_type_names(isl.dim_type.in_)
    }

    # }}}

    # {{{ generate instruction from compute map

    rule_mapping_ctx = SubstitutionRuleMappingContext(
        kernel.substitutions, kernel.get_var_name_generator())

    expr_subst_map = RuleAwareSubstitutionMapper(
        rule_mapping_ctx,
        make_subst_func(storage_ax_to_global_expr),
        within=parse_stack_match(None)
    )

    subst_expr = kernel.substitutions[substitution].expression
    compute_expression = expr_subst_map(subst_expr, kernel, None)

    temporary_name = substitution + "_temp"
    assignee = var(temporary_name)[tuple(
        var(iname) for iname in new_storage_axes
    )]

    compute_insn_id = substitution + "_compute"
    compute_insn = lp.Assignment(
        id=compute_insn_id,
        assignee=assignee,
        expression=compute_expression,
    )

    # }}}

    # {{{ replace substitution rule with newly created instruction

    for insn in kernel.instructions:
        if contains_a_subst_rule_invocation(kernel, insn) \
                and isinstance(insn, MultiAssignmentBase):
            print(insn)


    # }}}

    return kernel

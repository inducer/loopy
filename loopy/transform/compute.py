# DomainChanger
# iname nesting order <=> tree
# loop transformations
# - traverse syntax tree
# - affine map inames
#
# index views for warp tiling

from pymbolic.mapper.substitutor import make_subst_func
from loopy.kernel import LoopKernel
import islpy as isl

import loopy as lp
from loopy.kernel.data import AddressSpace
from loopy.kernel.function_interface import CallableKernel, ScalarCallable
from loopy.kernel.instruction import MultiAssignmentBase
from loopy.kernel.tools import DomainChanger
from loopy.match import parse_stack_match
from loopy.symbolic import RuleAwareSubstitutionMapper, SubstitutionRuleMappingContext, aff_from_expr, aff_to_expr, pw_aff_to_expr
from loopy.transform.precompute import RuleInvocationGatherer, RuleInvocationReplacer, contains_a_subst_rule_invocation
from loopy.translation_unit import TranslationUnit

import pymbolic.primitives as prim
from pymbolic import var

from pytools.tag import Tag


def compute(
        t_unit: TranslationUnit,
substitution: str,
        *args,
        **kwargs
    ) -> TranslationUnit:
    """
    Entrypoint for performing a compute transformation on all kernels in a
    translation unit. See :func:`_compute_inner` for more details.
    """

    assert isinstance(t_unit, TranslationUnit)
    new_callables = {}

    for id, callable in t_unit.callables_table.items():
        if isinstance(callable, CallableKernel):
            kernel = _compute_inner(
                callable.subkernel,
                substitution,
                *args, **kwargs
            )

            callable = callable.copy(subkernel=kernel)
        elif isinstance(callable, ScalarCallable):
            pass
        else:
            raise NotImplementedError()

        new_callables[id] = callable

    return t_unit

def _compute_inner(
        kernel: LoopKernel,
        substitution: str,
        transform_map: isl.Map,
        compute_map: isl.Map,
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

    :arg transform_map: An :class:`isl.Map` representing the affine
    transformation from the original iname domain to the transformed iname
    domain.

    :arg compute_map: An :class:`isl.Map` representing a relation between
    substitution rule indices and tuples `(a, l)`, where `a` is a vector of
    storage indices and `l` is a vector of "timestamps". This map describes
    """

    if not temporary_address_space:
        temporary_address_space = AddressSpace.GLOBAL

    # {{{ normalize names

    iname_to_storage_map = {
        iname : (iname + "_store" if iname in kernel.all_inames() else iname)
        for iname in storage_inames
    }

    new_storage_axes = list(iname_to_storage_map.values())

    for dim in range(compute_map.dim(isl.dim_type.out)):
        for iname, storage_ax in iname_to_storage_map.items():
            if compute_map.get_dim_name(isl.dim_type.out, dim) == iname:
                compute_map = compute_map.set_dim_name(
                    isl.dim_type.out, dim, storage_ax
                )

    # }}}

    # {{{ update kernel domain to contain storage inames

    storage_domain = compute_map.range().project_out_except(
        new_storage_axes, [isl.dim_type.set]
    )

    # FIXME: likely need to do some more digging to find proper domain to update
    new_domain = kernel.domains[0]
    for ax in new_storage_axes:
        new_domain = new_domain.add_dims(isl.dim_type.set, 1)

        new_domain = new_domain.set_dim_name(
            isl.dim_type.set,
            new_domain.dim(isl.dim_type.set) - 1,
            ax
        )

    new_domain, storage_domain = isl.align_two(new_domain, storage_domain)
    new_domain = new_domain & storage_domain
    kernel = kernel.copy(domains=[new_domain])

    # }}}

    # {{{ express substitution inputs as pw affs of (storage, time) names

    compute_pw_aff = compute_map.reverse().as_pw_multi_aff()
    subst_inp_names = [
        compute_map.get_dim_name(isl.dim_type.in_, i)
        for i in range(compute_map.dim(isl.dim_type.in_))
    ]
    storage_ax_to_global_expr = dict.fromkeys(subst_inp_names)
    for dim in range(compute_pw_aff.dim(isl.dim_type.out)):
        subst_inp = compute_map.get_dim_name(isl.dim_type.in_, dim)
        storage_ax_to_global_expr[subst_inp] = \
            pw_aff_to_expr(compute_pw_aff.get_at(dim))

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

    compute_dep_id = compute_insn_id
    new_insns = [compute_insn]

    # add global sync if we are storing in global memory
    if temporary_address_space == lp.AddressSpace.GLOBAL:
        gbarrier_id = kernel.make_unique_instruction_id(
            based_on=substitution + "_barrier"
        )

        from loopy.kernel.instruction import BarrierInstruction
        barrier_insn = BarrierInstruction(
            id=gbarrier_id,
            depends_on=frozenset([compute_insn_id]),
            synchronization_kind="global",
            mem_kind="global"
        )

        compute_dep_id = gbarrier_id

    # }}}

    # {{{ replace substitution rule with newly created instruction

    # FIXME: get these properly (see `precompute`)
    subst_name = substitution
    subst_tag = None
    within = None  # do we need this?



    # }}}

    return kernel

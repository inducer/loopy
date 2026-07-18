from constantdict import constantdict
import namedisl as nisl
from namedisl import DimType

from loopy import for_each_kernel
from loopy.kernel import LoopKernel
from loopy.kernel.instruction import HappensAfter

from pytools.graph import compute_topological_order


def _prefix_names(obj: nisl.Set, prefix: str, dim_type: DimType) -> nisl.Set:
    return obj.rename_dims(
        ((name, name + prefix) for name in obj.space.dimtype_to_names[dim_type])
    )


@for_each_kernel
def add_lexicographic_happens_after(kernel: LoopKernel) -> LoopKernel:
    """
    Imposes a strict lexicographic order on all statements in *kernel*. The
    order of statements as they appear in the kernel is used to impose the
    dependence relations.

    The following two conditions are true of the order imposed by this routine:
    1. All statements will have a self-dependence relation defined
    2. All statements except the first statement (as dictated by kernel order)
       will have a dependence relation defined between itself and the
       immediately preceding statement.
    """

    new_insns = []
    for i, insn in enumerate(kernel.instructions):
        new_happens_after = {}

        preds = (insn,) if i == 0 else (insn, kernel.instructions[i - 1])

        # FIXME: yuck
        after_domain = nisl.make_set(
            kernel.get_inames_domain(insn.within_inames).to_set()
        )

        after_inames = after_domain.space.dimtype_to_names[DimType.out]
        after_domain = _prefix_names(after_domain, "_after", DimType.out)
        for pred in preds:
            before_domain = nisl.make_set(
                kernel.get_inames_domain(pred.within_inames).to_set()
            )

            before_inames = before_domain.space.dimtype_to_names[DimType.out]
            before_domain = _prefix_names(before_domain, "_before", DimType.out)

            # lexicographic order necessitates agreement between before and
            # after on the order of shared inames
            shared_inames = frozenset(before_inames) & frozenset(after_inames)
            before_order = tuple(
                iname for iname in before_inames if iname in shared_inames
            )
            after_order = tuple(
                iname for iname in after_inames if iname in shared_inames
            )

            assert before_order == after_order
            shared_order = after_order

            joint_domain = after_domain & before_domain
            affs = joint_domain.pw_affs

            strict_lex = joint_domain - joint_domain
            equal_prefix = joint_domain
            for iname in shared_order:
                after_aff = affs[f"{iname}_after"]
                before_aff = affs[f"{iname}_before"]

                strict_lex = strict_lex | (
                    equal_prefix & after_aff.gt_set(before_aff)
                )
                equal_prefix = equal_prefix & after_aff.eq_set(before_aff)

            if pred.id == insn.id:
                ordered_instances = strict_lex
            else:
                ordered_instances = strict_lex | equal_prefix

            instances_rel = ordered_instances.as_map(
                in_names=tuple(f"{name}_after" for name in after_inames)
            )

            new_happens_after[pred.id] = HappensAfter(
                instances_rel=instances_rel
            )

        new_insns.append(insn.copy(happens_after=new_happens_after))

    return kernel.copy(instructions=new_insns)

# pyright: reportAny=warning

from __future__ import annotations


__copyright__ = """
Copyright (C) 2022-26 University of Illinois Board of Trustees
"""

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
from typing import TYPE_CHECKING

from typing_extensions import override

from loopy.diagnostic import LoopyError
from loopy.kernel.data import AddressSpace
from loopy.kernel.instruction import (
    Assignment,
    BarrierInstruction,
    CallInstruction,
    NoOpInstruction,
)
from loopy.match import Matchable, MatchExpressionBase
from loopy.symbolic import Reduction, WalkMapper
from loopy.transform.add_barrier import add_barrier
from loopy.transform.iname import split_iname, split_reduction_outward
from loopy.transform.instruction import add_dependency
from loopy.translation_unit import TranslationUnit, for_each_kernel


if TYPE_CHECKING:
    from collections.abc import Mapping

    from loopy.kernel import LoopKernel
    from loopy.translation_unit import CallablesTable


__doc__ = """
.. autofunction:: split_iteration_domain_across_work_items
.. autofunction:: add_gbarrier_between_disjoint_loop_sets
"""


# {{{ _LoopSet class definition

@dataclass(frozen=True, eq=True)
class _LoopSet:
    inames: frozenset[str]
    insns_in_loop_set: frozenset[str]


def _get_disjoint_loop_sets(kernel: LoopKernel) -> frozenset[_LoopSet]:
    """
    Returns information about the disjoint loop sets in *kernel*.
    """
    disjoint_inames_and_insns: list[tuple[set[str], set[str]]] = []
    iname_to_associated_inames_and_insns: dict[str, tuple[set[str], set[str]]] = {}
    for insn in kernel.instructions:
        inames = insn.within_inames | insn.reduction_inames()
        associated_inames_and_insns: tuple[set[str], set[str]] | None = None
        for iname in inames:
            try:
                associated_inames_and_insns = \
                    iname_to_associated_inames_and_insns[iname]
            except KeyError:
                pass
        if associated_inames_and_insns is not None:
            associated_inames, associated_insns = associated_inames_and_insns
            associated_inames.update(inames)
            associated_insns.add(insn.id)
        else:
            associated_inames_and_insns = (set(inames), {insn.id})
            disjoint_inames_and_insns.append(associated_inames_and_insns)
        for iname in inames:
            iname_to_associated_inames_and_insns[iname] = associated_inames_and_insns

    return frozenset({
        _LoopSet(
            frozenset(associated_inames),
            frozenset(associated_insns))
        for associated_inames, associated_insns in disjoint_inames_and_insns})

# }}}


# {{{ split_iteration_domain_across_work_items

def get_iname_length(kernel: LoopKernel, iname: str) -> float | int:
    from loopy.isl_helpers import static_max_of_pw_aff
    max_domain_size = static_max_of_pw_aff(
        kernel.get_iname_bounds(iname).size,
        constants_only=False).to_pw_aff().max_val()
    if max_domain_size.is_infty():
        import math
        return math.inf
    else:
        return max_domain_size.to_python()


class OuterReductionNestCollector(WalkMapper[[]]):
    def __init__(self, outer_inames: frozenset[str]) -> None:
        super().__init__()
        self.outer_inames: frozenset[str] = outer_inames
        # Since we're only looking for the reductions that are on the outside, we can
        # use a list instead of a full graph
        self.outer_redn_nest: list[frozenset[str]] = []

    @override
    def map_reduction(self, expr: Reduction) -> None:
        if not self.visit(expr):
            return

        outer_redn_inames = frozenset(expr.inames) & self.outer_inames

        if outer_redn_inames:
            self.outer_redn_nest.append(outer_redn_inames)

        self.rec(expr.expr)


class InconsistentInameOrdersError(LoopyError):
    """Raised when the iname orders implied by different assignees in a loop
    set cannot be merged into a single consistent total order."""


def _get_outer_iname_pos_from_loop_set(
        kernel: LoopKernel, loop_set: _LoopSet, outer_inames: frozenset[str]
        ) -> Mapping[str, int]:
    if not outer_inames:
        return {}

    import pymbolic.primitives as prim

    iname_orders: set[tuple[frozenset[str], ...]] = set()

    for insn_id in loop_set.insns_in_loop_set:
        insn = kernel.id_to_insn[insn_id]
        if isinstance(insn, Assignment):
            insn_iname_order: list[frozenset[str]] = []
            if isinstance(insn.assignee, prim.Subscript):
                insn_iname_order.extend(
                    frozenset({idx.name})
                    for idx in insn.assignee.index_tuple
                    if (
                        isinstance(idx, prim.Variable)
                        and idx.name in outer_inames))
            ornc = OuterReductionNestCollector(outer_inames)
            ornc(insn.expression)
            insn_iname_order.extend(ornc.outer_redn_nest)
            if insn_iname_order:
                iname_orders.add(tuple(insn_iname_order))
        elif isinstance(insn, CallInstruction):
            # must be a callable kernel, don't touch.
            pass
        elif isinstance(insn, (BarrierInstruction, NoOpInstruction)):
            pass
        else:
            raise NotImplementedError(type(insn))

    if not iname_orders:
        raise RuntimeError("split_iteration_domain failed by receiving a"
                           " kernel not belonging to the expected grammar or"
                           " kernels.")

    # Merge the per-assignee partial orders into a single total order
    from pytools.graph import CycleError, compute_topological_order

    successors: dict[str, set[str]] = {iname: set() for iname in outer_inames}
    for order in iname_orders:
        for earlier, later in zip(order[:-1], order[1:], strict=True):
            for earlier_iname in earlier:
                for later_iname in later:
                    successors[earlier_iname].add(later_iname)

    try:
        # key= for determinism
        iname_order = compute_topological_order(successors, key=lambda x: x)
    except CycleError as err:
        raise InconsistentInameOrdersError(
            f"inconsistent iname orderings across assignees: {iname_orders}"
            ) from err

    return {iname: i
            for i, iname in enumerate(iname_order)}


def _split_loop_set_across_work_items(
        kernel: LoopKernel,
        callables: CallablesTable,
        loop_set: _LoopSet,
        iname_to_length: Mapping[str, float | int],
        max_device_compute_units: int,
) -> LoopKernel:

    # Could possibly do something fancier that also includes the individual inner
    # loops in the loop set, but then it might be necessary to add fences. Not sure
    # if it's worth it?
    outer_non_redn_inames = loop_set.inames
    for insn_id in loop_set.insns_in_loop_set:
        outer_non_redn_inames &= kernel.id_to_insn[insn_id].within_inames

    outer_redn_inames = loop_set.inames
    for insn_id in loop_set.insns_in_loop_set:
        outer_redn_inames &= kernel.id_to_insn[insn_id].reduction_inames()

    outer_iname_pos: Mapping[str, int]
    all_outer_inames = outer_non_redn_inames | outer_redn_inames
    if all_outer_inames:
        try:
            outer_iname_pos = _get_outer_iname_pos_from_loop_set(
                kernel, loop_set, all_outer_inames)
        except InconsistentInameOrdersError:
            # No consistent merge of the per-assignee orderings exists; fall
            # back to a deterministic order based on iname names
            outer_iname_pos = {iname: i
                for i, iname in enumerate(sorted(all_outer_inames))}
    else:
        outer_iname_pos = {}

    # Prioritize the non-reduction loop with largest loop count. In case of ties,
    # look at the iname position in the assignee and pick the iname indexing over
    # leading axis for the work-group hardware iname
    inames_to_parallelize = sorted(
        outer_non_redn_inames,
        key=lambda iname: (
            iname_to_length[iname],
            -outer_iname_pos[iname]))

    # Add the largest reduction loop if we don't already have 2 non-reduction loops
    # to parallelize over
    if len(inames_to_parallelize) < 2 and outer_redn_inames:
        inames_to_parallelize.insert(0,
            max(
                outer_redn_inames,
                key=lambda iname: (
                    iname_to_length[iname],
                    -outer_iname_pos[iname])))

    vng = kernel.get_var_name_generator()

    if len(inames_to_parallelize) == 0:
        pass
    elif len(inames_to_parallelize) == 1:
        iname, = inames_to_parallelize
        if iname in outer_non_redn_inames:
            ngroups = max_device_compute_units * 4  # '4' to overfill the device
            l_one_size = 4
            l_zero_size = 16

            kernel = split_iname(kernel, iname,
                                 ngroups * l_zero_size * l_one_size)
            kernel = split_iname(kernel, f"{iname}_inner",
                                 l_zero_size, inner_tag="l.0")
            kernel = split_iname(kernel, f"{iname}_inner_outer",
                                 l_one_size, inner_tag="l.1",
                                 outer_tag="g.0")
        else:
            from loopy.match import Id
            from loopy.transform.data import reduction_arg_to_subst_rule
            from loopy.transform.precompute import precompute_for_single_kernel

            ngroups = max_device_compute_units
            wg_size = 32

            iredn_chunk = vng(f"{iname}_chunk")
            iredn_inner = vng(f"{iname}_inner")
            kernel = split_iname(kernel, iname, ngroups * wg_size,
                                 inner_iname=iredn_inner, outer_iname=iredn_chunk)

            iredn_group = vng(f"{iname}_group")
            iredn_thread = vng(f"{iname}_thread")
            kernel = split_iname(kernel, iredn_inner, wg_size,
                                 outer_iname=iredn_group, inner_iname=iredn_thread,
                                 inner_tag="l.0")
            kernel = split_reduction_outward(kernel, iredn_group)
            kernel = split_reduction_outward(kernel, iredn_thread)

            insn_ids = sorted(loop_set.insns_in_loop_set)

            iprcmpt_redn_group = vng(f"iprcmpt_{iredn_group}")

            compute_insns: list[str] = []
            for insn_id in insn_ids:
                subst_rule_name = vng(f"redn_subst_{iname}_{insn_id}")
                kernel = reduction_arg_to_subst_rule(
                    kernel, iredn_group,
                    subst_rule_name=subst_rule_name,
                    insn_match=Id(insn_id))

                temp_name = vng(f"redn_temp_{iname}_{insn_id}")
                compute_insn_id = vng(f"redn_compute_{iname}_{insn_id}")
                kernel = precompute_for_single_kernel(
                    kernel, callables, subst_rule_name, iredn_group,
                    temporary_name=temp_name,
                    temporary_address_space=AddressSpace.GLOBAL,
                    precompute_inames=[iprcmpt_redn_group],
                    default_tag="g.0",
                    # Don't want a separate barrier to be added for each temporary;
                    # instead we will add one below (this is safe because the
                    # instructions inside a reduction-only outer loop can't depend
                    # on each other)
                    add_barrier_for_global_temporary=False,
                    compute_insn_id=compute_insn_id)

                compute_insns.append(compute_insn_id)

            barrier_id = vng(f"redn_barrier_{iname}")
            kernel = add_barrier(
                kernel,
                insn_before=InsnIds(frozenset(compute_insns)),
                insn_after=InsnIds(frozenset(insn_ids)),
                id_based_on=barrier_id,
                synchronization_kind="global",
                mem_kind="global",
                within_inames=frozenset())

    else:
        bigger_loop = inames_to_parallelize[-1]
        smaller_loop = inames_to_parallelize[-2]

        ngroups = max_device_compute_units * 4  # '4' to overfill the device
        l_one_size = 4
        l_zero_size = 16

        kernel = split_iname(kernel, f"{bigger_loop}",
                             l_one_size * ngroups)
        kernel = split_iname(kernel, f"{bigger_loop}_inner",
                             l_one_size, inner_tag="l.1", outer_tag="g.0")
        if smaller_loop in outer_non_redn_inames:
            kernel = split_iname(kernel, smaller_loop,
                                 l_zero_size, inner_tag="l.0")
        else:
            smaller_inner_loop = vng(f"{smaller_loop}_inner")
            kernel = split_iname(kernel, smaller_loop,
                                 l_zero_size, inner_iname=smaller_inner_loop,
                                 inner_tag="l.0")
            kernel = split_reduction_outward(kernel, smaller_inner_loop)

    return kernel


@for_each_kernel
def _split_iteration_domain_across_work_items_for_single_kernel(
    kernel: LoopKernel,
    callables: CallablesTable,
    max_device_compute_units: int,
) -> LoopKernel:

    iname_to_length = {iname: get_iname_length(kernel, iname)
                       for iname in kernel.all_inames()}

    loop_sets = _get_disjoint_loop_sets(kernel)

    for loop_set in loop_sets:
        kernel = _split_loop_set_across_work_items(kernel,
                                                   callables,
                                                   loop_set,
                                                   iname_to_length,
                                                   max_device_compute_units)

    return kernel


def split_iteration_domain_across_work_items(
    t_unit: TranslationUnit,
    max_device_compute_units: int,
) -> TranslationUnit:
    # Need to pass callables table down into per-kernel function due to
    # precompute_for_single_kernel call
    return _split_iteration_domain_across_work_items_for_single_kernel(
        t_unit, t_unit.callables_table, max_device_compute_units)

# }}}


# {{{ add_gbarrier_between_disjoint_loop_sets

@dataclass(frozen=True)
class InsnIds(MatchExpressionBase):
    insn_ids_to_match: frozenset[str]

    @override
    def __call__(self, kernel: LoopKernel, matchable: Matchable):
        return matchable.id in self.insn_ids_to_match


def _get_call_kernel_insn_ids(kernel: LoopKernel) -> tuple[frozenset[str], ...]:
    """
    Returns a sequence of collection of instruction ids where each entry in the
    sequence corresponds to the instructions in a call-kernel to launch.

    In this heuristic we simply draw kernel boundaries such that instruction
    belonging to disjoint loop set pairs are executed in different call kernels.
    """
    loop_sets = _get_disjoint_loop_sets(kernel)

    insn_id_to_loop_set = {
        insn_id: loop_set
        for loop_set in loop_sets
        for insn_id in loop_set.insns_in_loop_set}

    from pytools.graph import compute_topological_order

    loop_set_dep_graph: dict[_LoopSet, set[_LoopSet]] = {
        insn_id_to_loop_set[insn.id]: set()
        for insn in kernel.instructions
    }

    for insn in kernel.instructions:
        insn_loop_set = insn_id_to_loop_set[insn.id]
        for dep_id in insn.depends_on:
            dep_loop_set = insn_id_to_loop_set[dep_id]
            if insn_loop_set != dep_loop_set:
                loop_set_dep_graph[dep_loop_set].add(insn_loop_set)

    # Break ties between ready loop sets using the lexicographically smallest
    # instruction ID in each set. Loop sets are disjoint by construction, so these
    # mins are unique across sets
    toposorted_loop_sets: list[_LoopSet] = compute_topological_order(
        loop_set_dep_graph,
        key=lambda ls: min(ls.insns_in_loop_set))

    return tuple(loop_set.insns_in_loop_set for loop_set in toposorted_loop_sets)


def add_gbarrier_between_disjoint_loop_sets(
        t_unit: TranslationUnit) -> TranslationUnit:
    kernel = t_unit.default_entrypoint
    ing = kernel.get_instruction_id_generator()

    call_kernel_insn_ids = _get_call_kernel_insn_ids(kernel)
    gbarrier_ids: list[str] = []

    for ibarrier, (insns_before, insns_after) in enumerate(
            zip(call_kernel_insn_ids[:-1], call_kernel_insn_ids[1:], strict=True)):
        id_based_on = ing(f"_actx_gbarrier_{ibarrier}")
        kernel = add_barrier(kernel,
                             insn_before=InsnIds(insns_before),
                             insn_after=InsnIds(insns_after),
                             id_based_on=id_based_on,
                             within_inames=frozenset())
        assert id_based_on in kernel.id_to_insn
        gbarrier_ids.append(id_based_on)

    from loopy.match import Id
    for pred_gbarrier, succ_gbarrier in zip(
            gbarrier_ids[:-1], gbarrier_ids[1:], strict=True):
        kernel = add_dependency(kernel, Id(succ_gbarrier), pred_gbarrier)

    return t_unit.with_kernel(kernel)

# }}}


# vim: foldmethod=marker

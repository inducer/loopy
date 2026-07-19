from __future__ import annotations

from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, final, override

import namedisl as nisl
from constantdict import constantdict
from namedisl import DimType

from pymbolic import primitives as prim
from pytools.graph import compute_topological_order

from loopy import for_each_kernel
from loopy.kernel.instruction import (
    HappensAfter,
    InstructionBase,
    MultiAssignmentBase,
)
from loopy.symbolic import (
    LinearSubscript,
    Reduction,
    SubArrayRef,
    WalkMapper,
    aff_from_expr,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from namedisl.core import NamedIslObjectT

    from pymbolic.typing import Expression

    from loopy.kernel import LoopKernel


@final
class AccessType(Enum):
    read = 0
    write = 1


class AccessRelationFinder(WalkMapper[[str, AccessType]]):
    """Collect per-instruction statement-instance-to-cell access relations."""

    kernel: LoopKernel
    _additional_inames: frozenset[str]
    _read_relations: dict[str, dict[str, nisl.Map]]
    _write_relations: dict[str, dict[str, nisl.Map]]

    def __init__(self, kernel: LoopKernel):
        self.kernel = kernel
        self._additional_inames = frozenset()
        self._read_relations = {insn.id: {} for insn in kernel.instructions}
        self._write_relations = {insn.id: {} for insn in kernel.instructions}

        super().__init__()

    def _get_access_relation(
        self,
        domain: nisl.Set,
        subscript: tuple[Expression, ...],
    ) -> nisl.Map:
        instance_names = domain.space.dimtype_to_names[DimType.out]
        cell_names = tuple(f"ax_{axis}" for axis in range(len(subscript)))

        access_set = domain.add_dims(DimType.out, cell_names)
        coordinates = access_set.pw_affs
        for cell_name, index_expr in zip(cell_names, subscript, strict=True):
            index_aff = nisl.make_aff(
                aff_from_expr(
                    access_set.space.as_isl_set_space(),
                    index_expr,
                )
            ).as_pw_aff()

            access_set = access_set & coordinates[cell_name].eq_set(index_aff)

        return access_set.as_map(in_names=instance_names)

    def _insn_writes_var(self, insn_id: str, var: str) -> bool:
        return (
            var in self.kernel.writer_map()
            and insn_id in self.kernel.writer_map()[var]
        )

    def _insn_reads_var(self, insn_id: str, var: str) -> bool:
        return (
            var in self.kernel.reader_map()
            and insn_id in self.kernel.reader_map()[var]
        )

    def _insn_accesses_var(self, insn_id: str, var: str) -> bool:
        return self._insn_reads_var(insn_id, var) or self._insn_writes_var(
            insn_id, var
        )

    def _record_access(
        self,
        insn_id: str,
        var: str,
        subscript: tuple[Expression, ...],
        access_type: AccessType,
    ) -> None:
        if not self._insn_accesses_var(insn_id, var):
            return

        insn = self.kernel.id_to_insn[insn_id]
        domain_inames = insn.within_inames | self._additional_inames
        inames_domain = nisl.make_set(
            self.kernel.get_inames_domain(domain_inames).to_set()
        )
        access_rel = self._get_access_relation(inames_domain, subscript)

        additional_inames = self._additional_inames - insn.within_inames
        if additional_inames:
            access_rel = access_rel.project_out(additional_inames)

        match access_type:
            case AccessType.read:
                previous = self._read_relations[insn_id].get(var)
                self._read_relations[insn_id][var] = (
                    access_rel if previous is None else previous | access_rel
                )
            case AccessType.write:
                previous = self._write_relations[insn_id].get(var)
                self._write_relations[insn_id][var] = (
                    access_rel if previous is None else previous | access_rel
                )
            case _:
                raise ValueError("unknown AccessType")

    @cached_property
    def read_relations(self) -> Mapping[str, Mapping[str, nisl.Map]]:
        return constantdict({
            insn_id: constantdict(relations)
            for insn_id, relations in self._read_relations.items()
        })

    @cached_property
    def write_relations(self) -> Mapping[str, Mapping[str, nisl.Map]]:
        return constantdict({
            insn_id: constantdict(relations)
            for insn_id, relations in self._write_relations.items()
        })

    @override
    def map_subscript(
        self, expr: prim.Subscript, /, insn_id: str, access_type: AccessType
    ) -> None:
        assert isinstance(expr.aggregate, prim.Variable)
        self._record_access(
            insn_id, expr.aggregate.name, expr.index_tuple, access_type
        )

    @override
    def map_linear_subscript(
        self, expr: LinearSubscript, /, insn_id: str, access_type: AccessType
    ) -> None:
        self.rec(expr.index, insn_id, AccessType.read)

        assert isinstance(expr.aggregate, prim.Variable)
        self._record_access(
            insn_id, expr.aggregate.name, (expr.index,), access_type
        )

    @override
    def map_reduction(
        self, expr: Reduction, /, insn_id: str, access_type: AccessType
    ) -> None:
        previous_inames = self._additional_inames
        self._additional_inames |= frozenset(expr.inames)
        try:
            WalkMapper.map_reduction(self, expr, insn_id, access_type)
        finally:
            self._additional_inames = previous_inames

    @override
    def map_sub_array_ref(
        self, expr: SubArrayRef, /, insn_id: str, access_type: AccessType
    ) -> None:
        previous_inames = self._additional_inames
        self._additional_inames |= frozenset(
            iname.name for iname in expr.swept_inames
        )
        try:
            self.rec(expr.subscript, insn_id, access_type)
        finally:
            self._additional_inames = previous_inames


def _suffix_names(
    obj: NamedIslObjectT, suffix: str, dim_type: DimType
) -> NamedIslObjectT:
    return obj.rename_dims(
        (name, name + suffix) for name in obj.space.dimtype_to_names[dim_type]
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

    new_insns: list[InstructionBase] = []
    for i, insn in enumerate(kernel.instructions):
        new_happens_after: dict[str, HappensAfter] = {}

        sources = (insn,) if i == 0 else (insn, kernel.instructions[i - 1])

        after_domain = nisl.make_set(
            kernel.get_inames_domain(insn.within_inames).to_set()
        )

        after_inames = after_domain.space.dimtype_to_names[DimType.out]
        after_domain = _suffix_names(after_domain, "_after", DimType.out)
        for source in sources:
            before_domain = nisl.make_set(
                kernel.get_inames_domain(source.within_inames).to_set()
            )

            before_inames = before_domain.space.dimtype_to_names[DimType.out]
            before_domain = _suffix_names(before_domain, "_before", DimType.out)

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

            if source.id == insn.id:
                ordered_instances = strict_lex
            else:
                ordered_instances = strict_lex | equal_prefix

            instances_rel = ordered_instances.as_map(
                in_names=tuple(f"{name}_after" for name in after_inames)
            )

            new_happens_after[source.id] = HappensAfter(
                instances_rel=instances_rel
            )

        new_insns.append(insn.copy(happens_after=new_happens_after))

    return kernel.copy(instructions=new_insns)


def _relax_strict_happens_after_inner(
    kernel: LoopKernel,
    sink_id: str,
    source_id: str,
    var: str,
    sink_access_type: AccessType,
    incoming_instances_rel: nisl.Map,
    live_access_rel: nisl.Map,
    rel_finder: AccessRelationFinder,
    happens_after: dict[str, HappensAfter],
) -> Mapping[str, HappensAfter]:
    """
    Recursively finds conflicting accesses to *var* by *sink* and *source* to
    determine the minimal required execution order between statement instances
    of *source* and *sink*.

    :arg sink_id: The ID of the statement whose instances will be in the domain
    of the resulting :class:`namedisl.Map`.

    :arg source_id: The ID of the statement whose instances will be in the range
    of the resulting :class:`namedisl.Map`.

    :arg var: The variable for which we are performing data dependence analysis.

    :arg sink_access_type: A :class:`AccessType` describing whether *sink_id*
    reads or writes *var*. This determines how live instances are removed from
    *live_access_rel*.

    :arg incoming_instances_rel: The incoming :class:`namedisl.Map` describing
    how each sink and source instance are related.

    :arg live_access_rel: A :class:`namedisl.Map` describing the set of live
    accesses by *sink_id* to *var*. When conflicts are found, the conflicting
    relation is used to remove elements from this relation.

    :arg rel_finder: A :class:`AccessRelationFinder` with access relations
    constructed before entering this routine.

    :arg happens_after: A mapping from statement IDs to
    :class:`loopy.HappensAfter` recording the dependencies from *sink* to all
    statements in *happens_after*.

    :returns: The updated precise dependencies for *sink_id*.
    """

    def record_conflicts(source_relation: nisl.Map) -> nisl.Map:
        source_relation = _suffix_names(source_relation, "_before", DimType.in_)

        # live_access_rel; source_relation^-1
        conflicts = live_access_rel.apply_range(source_relation.reverse())

        # Only conflicts ordered along this graph path are required.
        required_order = incoming_instances_rel & conflicts
        previous = happens_after.get(source_id)
        if not required_order.is_empty():
            if previous is None:
                combined_order = required_order
            else:
                assert previous.instances_rel is not None
                combined_order = required_order | previous.instances_rel

            happens_after[source_id] = HappensAfter(combined_order)

        # Retire only live accesses supplied by an ordered source instance.
        return live_access_rel & required_order.apply_range(source_relation)

    def normalize_interface_and_compose(
        incoming_relation: nisl.Map, next_edge_relation: nisl.Map
    ) -> nisl.Map:
        incoming_relation = incoming_relation.rename_dims(
            (name, name[: len(name) - len("_before")])
            for name in incoming_relation.space.out_names
        )

        next_edge_relation = next_edge_relation.rename_dims(
            (name, name[: len(name) - len("_after")])
            for name in next_edge_relation.space.in_names
        )

        return incoming_relation.apply_range(next_edge_relation)

    match sink_access_type:
        # Read-after-write
        case AccessType.read:
            if var in rel_finder.write_relations[source_id]:
                source_relation = rel_finder.write_relations[source_id][var]

                caught_accesses = record_conflicts(source_relation)
                live_access_rel = live_access_rel - caught_accesses

        # Write-after-write and write-after-read
        case AccessType.write:
            if var in rel_finder.write_relations[source_id]:
                source_relation = rel_finder.write_relations[source_id][var]

                caught_accesses = record_conflicts(source_relation)
                live_access_rel = live_access_rel - caught_accesses

            # don't update live_access_rel; does not find a "most recent writer"
            if var in rel_finder.read_relations[source_id]:
                source_relation = rel_finder.read_relations[source_id][var]
                _ = record_conflicts(source_relation)

        case _:
            raise ValueError("unknown access type")

    # Continue backward through the strict-order graph.
    if not live_access_rel.is_empty() and (sink_id != source_id):
        source_insn = kernel.id_to_insn[source_id]
        for src_dep_id, src_happens_after in source_insn.happens_after.items():
            if src_dep_id == source_id:
                continue

            if src_happens_after.instances_rel is None:
                raise ValueError(
                    "All `HappensAfter`s must have precise dependencies "
                    "defined to use precise dependency finding machinery."
                )

            outgoing_instances_rel = normalize_interface_and_compose(
                incoming_instances_rel, src_happens_after.instances_rel
            ).coalesce()

            _relax_strict_happens_after_inner(
                kernel,
                sink_id,
                src_dep_id,
                var,
                sink_access_type,
                outgoing_instances_rel,
                live_access_rel,
                rel_finder,
                happens_after,
            )

    return happens_after


@for_each_kernel
def relax_strict_happens_after(kernel: LoopKernel) -> LoopKernel:
    """
    Relaxes an incoming strict execution order imposed on statements in *kernel*
    through dependence analysis.

    :returns: *kernel* with the minimally required execution order on statement
    instances in a program needed to satisfy data dependencies.
    """

    coarse_dependency_graph: dict[str, frozenset[str]] = {}
    for insn in kernel.instructions:
        coarse_dependency_graph[insn.id] = frozenset({
            dep for dep in insn.happens_after if dep != insn.id
        })

    topological_order = compute_topological_order(coarse_dependency_graph)

    rel_finder = AccessRelationFinder(kernel)
    for insn in kernel.instructions:
        if isinstance(insn, MultiAssignmentBase):
            for assignee in insn.assignees:
                rel_finder(assignee, insn.id, AccessType.write)
            rel_finder(insn.expression, insn.id, AccessType.read)
            for pred in insn.predicates:
                rel_finder(pred, insn.id, AccessType.read)

    new_insns: list[InstructionBase] = []
    for sink_id in topological_order:
        new_happens_after: dict[str, HappensAfter] = {}
        old_happens_after = kernel.id_to_insn[sink_id].happens_after
        for sink_access_type, access_relations in (
            (AccessType.read, rel_finder.read_relations[sink_id]),
            (AccessType.write, rel_finder.write_relations[sink_id]),
        ):
            for var, access_relation in access_relations.items():
                access_relation = _suffix_names(
                    access_relation, "_after", DimType.in_
                )
                for source_id, happens_after in old_happens_after.items():
                    if happens_after.instances_rel is None:
                        raise ValueError(
                            "All `HappensAfter`s must have precise dependencies "
                            "defined to use precise dependency finding machinery."
                        )

                    _relax_strict_happens_after_inner(
                        kernel,
                        sink_id,
                        source_id,
                        var,
                        sink_access_type,
                        happens_after.instances_rel,
                        access_relation,
                        rel_finder,
                        new_happens_after,
                    )

        new_insns.append(
            kernel.id_to_insn[sink_id].copy(
                happens_after=constantdict(new_happens_after)
            )
        )

    return kernel.copy(instructions=new_insns)

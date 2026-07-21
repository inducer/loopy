from __future__ import annotations


__copyright__ = """
Copyright (C) 2026 Addison Alvey-Blanco
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

import namedisl as nisl
from constantdict import constantdict
from namedisl import DimType

import islpy as isl

from loopy import KernelState, LoopKernel, for_each_kernel
from loopy.diagnostic import LoopyError
from loopy.schedule import (
    Barrier,
    CallKernel,
    EnterLoop,
    LeaveLoop,
    ReturnFromKernel,
    RunInstruction,
    ScheduleItem,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class _PreciseScheduleRecord:
    timestamp: Sequence[int | str]


@dataclass(frozen=True)
class _StatementRecord(_PreciseScheduleRecord):
    subkernel_idx: int | None


@dataclass(frozen=True)
class _BarrierRecord(_PreciseScheduleRecord):
    barrier: Barrier
    subkernel_idx: int | None = None


@dataclass(frozen=True)
class _PreciseSchedule:
    statements: Mapping[str, _StatementRecord]
    barriers: Sequence[_BarrierRecord]


def _get_timestamp_points_from_linearization(
    kernel: LoopKernel,
) -> _PreciseSchedule:
    if kernel.linearization is None or kernel.state != KernelState.LINEARIZED:
        raise LoopyError(
            "Kernel must be linearized before instance-level analysis."
        )

    def build_timestamp_from_stack(
        stack: list[tuple[int, ScheduleItem]],
    ) -> Sequence[int | str]:
        tstamp: Sequence[int | str] = []
        for frame in stack:
            match frame:
                case (x, EnterLoop(iname=iname)):
                    tstamp.append(x)
                    tstamp.append(iname)

                case (x, CallKernel(_)):
                    tstamp.append(x)

                case _:
                    pass

        return tuple(tstamp)

    def find_most_recent_subkernel_idx(
        stack: list[tuple[int, ScheduleItem]],
    ) -> int | None:
        subkernel_idx = None
        for frame in stack:
            pos, sched_item = frame
            if isinstance(sched_item, CallKernel):
                subkernel_idx = pos
                break
        return subkernel_idx

    stack: list[tuple[int, ScheduleItem]] = []
    stmt_records: dict[str, _StatementRecord] = {}
    bar_records: list[_BarrierRecord] = []
    for i, node in enumerate(kernel.linearization):
        match node:
            case CallKernel(_):
                stack.append((i, node))

            case ReturnFromKernel(_):
                stack.pop()

            case EnterLoop(_):
                stack.append((i, node))

            case LeaveLoop(_):
                stack.pop()

            case RunInstruction(insn_id=stmt_id):
                tstamp = build_timestamp_from_stack(stack)
                tstamp = (*tstamp, i)

                subkernel_idx = find_most_recent_subkernel_idx(stack)

                if subkernel_idx is None:
                    raise LoopyError(
                        f"could not determine subkernel index for {stmt_id}"
                    )

                stmt_records[stmt_id] = _StatementRecord(
                    timestamp=tstamp, subkernel_idx=subkernel_idx
                )

            case Barrier(_):
                tstamp = build_timestamp_from_stack(stack)
                tstamp = (*tstamp, i)

                subkernel_idx = find_most_recent_subkernel_idx(stack)

                bar_records.append(
                    _BarrierRecord(
                        timestamp=tstamp,
                        barrier=node,
                        subkernel_idx=subkernel_idx,
                    )
                )

                if node.originating_insn_id is not None:
                    stmt_records[node.originating_insn_id] = _StatementRecord(
                        timestamp=tstamp,
                        subkernel_idx=subkernel_idx,
                    )

            case _:
                pass

    return _PreciseSchedule(
        statements=constantdict(stmt_records), barriers=tuple(bar_records)
    )


def _build_statement_timestamp_relations(
    kernel: LoopKernel,
    stmt_records: Mapping[str, _StatementRecord],
    timestamp_names: Sequence[str],
) -> Mapping[str, nisl.Map]:
    stmt_relns: dict[str, nisl.Map] = {}
    for stmt_id, record in stmt_records.items():
        stmt = kernel.id_to_insn[stmt_id]

        # pad so that composition interface matches across all statements
        pad = len(timestamp_names) - len(record.timestamp)
        timestamp = list(record.timestamp)
        timestamp.extend([0 for _ in range(pad)])

        ran_str = ", ".join(
            f"{name} = {pos}"
            for name, pos in zip(timestamp_names, timestamp, strict=True)
        )

        dom_str = ", ".join(name for name in stmt.within_inames)

        full_str = dom_str + ", " + ran_str if dom_str else ran_str

        # FIXME: isl -> named conversion
        domain = nisl.make_set(
            kernel.get_inames_domain(stmt.within_inames).to_set()
        ).project_out_except([*stmt.within_inames, *kernel.all_params()])

        if stmt.within_inames:
            reln = nisl.make_set(f"{{[{full_str}]}}").as_map(stmt.within_inames)
        else:
            constraints = " and ".join(
                f"{name} = {pos}"
                for name, pos in zip(timestamp_names, timestamp, strict=True)
            )
            reln = nisl.make_map(
                f"{{ [] -> [{', '.join(timestamp_names)}] : {constraints} }}"
            )

        reln = reln.intersect_domain(domain)

        stmt_relns[stmt_id] = reln

    return stmt_relns


def _build_barrier_timestamp_relations(
    kernel: LoopKernel,
    barrier_records: Sequence[_BarrierRecord],
    timestamp_names: Sequence[str],
) -> Sequence[nisl.Map]:
    barrier_relns: list[nisl.Map] = []
    for record in barrier_records:
        inames = tuple(
            value for value in record.timestamp if isinstance(value, str)
        )

        pad = len(timestamp_names) - len(record.timestamp)
        timestamp = [*record.timestamp, *(0 for _ in range(pad))]

        ran_str = ", ".join(
            f"{name} = {pos}"
            for name, pos in zip(timestamp_names, timestamp, strict=True)
        )
        dom_str = ", ".join(inames)
        full_str = f"{dom_str}, {ran_str}" if dom_str else ran_str

        # FIXME: isl -> named conversion
        domain = nisl.make_set(
            kernel.get_inames_domain(frozenset(inames)).to_set()
        ).project_out_except([*inames, *kernel.all_params()])

        if inames:
            relation = nisl.make_set(f"{{[{full_str}]}}").as_map(inames)
        else:
            constraints = " and ".join(
                f"{name} = {pos}"
                for name, pos in zip(timestamp_names, timestamp, strict=True)
            )
            relation = nisl.make_map(
                f"{{ [] -> [{', '.join(timestamp_names)}] : {constraints} }}"
            )

        barrier_relns.append(relation.intersect_domain(domain))

    return tuple(barrier_relns)


def _build_strict_lexicographic_order(
    timestamp_names: Sequence[str],
) -> nisl.Map:
    later_names = tuple(f"{name}_later" for name in timestamp_names)
    earlier_names = tuple(f"{name}_earlier" for name in timestamp_names)

    joint = nisl.make_set(
        f"{{ [{', '.join([*later_names, *earlier_names])}] }}"
    )
    affs = joint.var_pw_affs

    strict_lex = joint - joint
    equal_prefix = joint

    for later_name, earlier_name in zip(
        later_names, earlier_names, strict=True
    ):
        strict_lex = strict_lex | (
            equal_prefix & affs[later_name].gt_set(affs[earlier_name])
        )
        equal_prefix = equal_prefix & affs[later_name].eq_set(
            affs[earlier_name]
        )

    return strict_lex.as_map(later_names)


def _build_timestamp_relations(
    kernel: LoopKernel,
    prec_sched: _PreciseSchedule,
) -> tuple[Mapping[str, nisl.Map], Sequence[nisl.Map], nisl.Map]:
    max_stmt_tstamp_len = max(
        len(record.timestamp) for record in prec_sched.statements.values()
    )

    max_bar_tstamp_len = -1
    if prec_sched.barriers:
        max_bar_tstamp_len = max(
            len(record.timestamp) for record in prec_sched.barriers
        )

    max_tstamp_len = max(max_stmt_tstamp_len, max_bar_tstamp_len)
    timestamp_names = [f"__ts_{i}" for i in range(max_tstamp_len)]

    stmt_relns = _build_statement_timestamp_relations(
        kernel, prec_sched.statements, timestamp_names
    )

    bar_relns = _build_barrier_timestamp_relations(
        kernel, prec_sched.barriers, timestamp_names
    )

    timestamp_lex = _build_strict_lexicographic_order(timestamp_names)

    return constantdict(stmt_relns), bar_relns, timestamp_lex


def _suffix_dim_names(
    relation: nisl.Map,
    dim_type: DimType,
    suffix: str,
) -> nisl.Map:
    return relation.rename_dims(
        (name, f"{name}{suffix}")
        for name in relation.space.dimtype_to_names[dim_type]
    )


def _timestamp_relation_for_role(
    relation: nisl.Map,
    role: str,
) -> nisl.Map:
    return _suffix_dim_names(relation, DimType.out, f"_{role}")


def _hardware_axis_inames(
    kernel: LoopKernel,
    stmt_id: str,
    include_local_axes: bool,
) -> Mapping[tuple[str, int], str]:
    from loopy.kernel.data import GroupInameTag, LocalInameTag

    result: dict[tuple[str, int], str] = {}
    for iname in kernel.id_to_insn[stmt_id].within_inames:
        tags = kernel.iname_tags_of_type(
            iname, (GroupInameTag, LocalInameTag), max_num=1
        )
        if not tags:
            continue

        (tag,) = tags
        if isinstance(tag, GroupInameTag):
            key = ("group", tag.axis)
        elif include_local_axes:
            key = ("local", tag.axis)
        else:
            continue

        if key in result:
            raise LoopyError(
                f"instruction '{stmt_id}' uses multiple inames for "
                f"hardware axis '{key[0]}.{key[1]}'"
            )
        result[key] = iname

    return constantdict(result)


def _build_hardware_id_relation(
    kernel: LoopKernel,
    stmt_id: str,
    instance_domain: nisl.Set,
    instance_suffix: str,
    include_local_axes: bool,
) -> nisl.Map:
    axis_inames = _hardware_axis_inames(kernel, stmt_id, include_local_axes)
    input_names = instance_domain.space.dimtype_to_names[DimType.out]
    hardware_names = tuple(
        f"__{kind}_{axis}" for kind, axis in sorted(axis_inames)
    )
    constraints = " and ".join(
        f"__{kind}_{axis} = {axis_inames[kind, axis]}{instance_suffix}"
        for kind, axis in sorted(axis_inames)
    )
    constraint_str = f" : {constraints}" if constraints else ""

    relation = nisl.make_map(
        "{ "
        f"[{', '.join(input_names)}] -> "
        f"[{', '.join(hardware_names)}]"
        f"{constraint_str} "
        "}"
    )
    return relation.intersect_domain(instance_domain)


def _build_same_hardware_scope_relation(
    kernel: LoopKernel,
    sink_id: str,
    source_id: str,
    sink_domain: nisl.Set,
    source_domain: nisl.Set,
    include_local_axes: bool,
) -> nisl.Map:
    sink_axes = _hardware_axis_inames(kernel, sink_id, include_local_axes)
    source_axes = _hardware_axis_inames(kernel, source_id, include_local_axes)
    if sink_axes.keys() != source_axes.keys():
        scope = "work-item" if include_local_axes else "work-group"
        raise LoopyError(
            f"cannot compare the {scope} instances of '{sink_id}' and "
            f"'{source_id}': their hardware axes differ"
        )

    sink_hardware = _build_hardware_id_relation(
        kernel, sink_id, sink_domain, "_after", include_local_axes
    )
    source_hardware = _build_hardware_id_relation(
        kernel, source_id, source_domain, "_before", include_local_axes
    )
    return sink_hardware.apply_range(source_hardware.reverse())


def _build_enforced_order(
    kernel: LoopKernel,
    sink_id: str,
    source_id: str,
    prec_sched: _PreciseSchedule,
    stmt_relns: Mapping[str, nisl.Map],
    barrier_relns: Sequence[nisl.Map],
    timestamp_lex: nisl.Map,
) -> nisl.Map:
    sink = _suffix_dim_names(stmt_relns[sink_id], DimType.in_, "_after")
    source = _suffix_dim_names(stmt_relns[source_id], DimType.in_, "_before")

    enforced = (
        _timestamp_relation_for_role(sink, "later")
        .apply_range(timestamp_lex)
        .apply_range(_timestamp_relation_for_role(source, "earlier").reverse())
    )
    enforced = enforced & _build_same_hardware_scope_relation(
        kernel,
        sink_id,
        source_id,
        sink.domain(),
        source.domain(),
        include_local_axes=True,
    )

    sink_record = prec_sched.statements[sink_id]
    source_record = prec_sched.statements[source_id]
    for barrier_idx, (barrier_record, barrier_reln) in enumerate(
        zip(prec_sched.barriers, barrier_relns, strict=True)
    ):
        if barrier_record.barrier.synchronization_kind == "local" and (
            sink_record.subkernel_idx != barrier_record.subkernel_idx
            or source_record.subkernel_idx != barrier_record.subkernel_idx
        ):
            continue

        barrier = _suffix_dim_names(
            barrier_reln, DimType.in_, f"_barrier_{barrier_idx}"
        )
        sink_to_barrier = (
            _timestamp_relation_for_role(sink, "later")
            .apply_range(timestamp_lex)
            .apply_range(
                _timestamp_relation_for_role(barrier, "earlier").reverse()
            )
        )
        barrier_to_source = (
            _timestamp_relation_for_role(barrier, "later")
            .apply_range(timestamp_lex)
            .apply_range(
                _timestamp_relation_for_role(source, "earlier").reverse()
            )
        )
        through_barrier = sink_to_barrier.apply_range(barrier_to_source)

        if barrier_record.barrier.synchronization_kind == "local":
            through_barrier = (
                through_barrier
                & _build_same_hardware_scope_relation(
                    kernel,
                    sink_id,
                    source_id,
                    sink.domain(),
                    source.domain(),
                    include_local_axes=False,
                )
            )

        enforced = enforced | through_barrier

    return enforced.coalesce()


@for_each_kernel
def verify_happens_after_is_enforced(kernel: LoopKernel) -> LoopKernel:
    """
    Verifies that the linearization of *kernel* enforces the instance-level order
    required by the :class:`loopy.HappensAfter`s of each
    statement in *kernel*.

    Assumes *kernel* possesses a valid linearization.
    """
    if kernel.linearization is None or kernel.state != KernelState.LINEARIZED:
        raise LoopyError("Kernel must be linearized before verification.")

    prec_sched = _get_timestamp_points_from_linearization(kernel)
    stmt_relns, barrier_relns, timestamp_lex = _build_timestamp_relations(
        kernel, prec_sched
    )

    for sink in kernel.instructions:
        for source_id, happens_after in sink.happens_after.items():
            if happens_after.instances_rel is None:
                raise LoopyError(
                    "precise happens-after verification requires an "
                    f"instance relation for '{sink.id}' after "
                    f"'{source_id}'"
                )

            required = nisl.make_map(
                happens_after.instances_rel.reset_tuple_id(
                    isl.dim_type.in_
                ).reset_tuple_id(isl.dim_type.out)
            )
            enforced = _build_enforced_order(
                kernel,
                sink.id,
                source_id,
                prec_sched,
                stmt_relns,
                barrier_relns,
                timestamp_lex,
            )
            missing = required - enforced
            if not missing.is_empty():
                raise LoopyError(
                    f"schedule does not enforce '{sink.id}' after "
                    f"'{source_id}': missing order {missing}"
                )

    return kernel

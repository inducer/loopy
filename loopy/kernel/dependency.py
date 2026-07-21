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


from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, final, override

import namedisl as nisl
from constantdict import constantdict
from namedisl import DimType

from pymbolic import primitives as prim
from pytools.graph import compute_topological_order

from loopy import for_each_kernel
from loopy.diagnostic import LoopyError
from loopy.kernel.instruction import (
    HappensAfter,
    InstructionBase,
    MultiAssignmentBase,
)
from loopy.symbolic import (
    LinearSubscript,
    Reduction,
    SubArrayRef,
    SubstitutionRuleExpander,
    WalkMapper,
    aff_from_expr,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from namedisl.core import NamedIslObjectT

    from pymbolic.typing import Expression
    from pytools import UniqueNameGenerator

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
    _name_generator: UniqueNameGenerator
    _cell_names: list[str]
    _storage_variables: frozenset[str]
    _subst_expander: SubstitutionRuleExpander

    def __init__(self, kernel: LoopKernel):
        self.kernel = kernel
        self._additional_inames = frozenset()
        self._read_relations = {stmt.id: {} for stmt in kernel.instructions}
        self._write_relations = {stmt.id: {} for stmt in kernel.instructions}
        self._name_generator = kernel.get_var_name_generator()
        self._cell_names = []
        self._storage_variables = frozenset(kernel.non_iname_variable_names())
        self._subst_expander = SubstitutionRuleExpander(kernel.substitutions)

        super().__init__()

    @override
    def __call__(
        self, expr: Expression, stmt_id: str, access_type: AccessType
    ) -> None:
        self.rec(self._subst_expander(expr), stmt_id, access_type)

    def _get_access_relation(
        self,
        domain: nisl.Set,
        subscript: tuple[Expression, ...],
    ) -> nisl.Map:
        instance_names = domain.space.dimtype_to_names[DimType.out]
        while len(self._cell_names) < len(subscript):
            axis = len(self._cell_names)
            self._cell_names.append(self._name_generator(f"ax_{axis}"))

        cell_names = tuple(self._cell_names[: len(subscript)])

        access_set = domain.add_dims(DimType.out, cell_names)
        coordinates = access_set.var_pw_affs
        for cell_name, index_expr in zip(cell_names, subscript, strict=True):
            index_aff = nisl.make_aff(
                aff_from_expr(
                    access_set.space.as_isl_set_space(),
                    index_expr,
                )
            ).as_pw_aff()

            access_set = access_set & coordinates[cell_name].eq_set(index_aff)

        return access_set.as_map(in_names=instance_names)

    def _record_access(
        self,
        stmt_id: str,
        var: str,
        subscript: tuple[Expression, ...],
        access_type: AccessType,
    ) -> None:
        if var not in self._storage_variables:
            return

        stmt = self.kernel.id_to_insn[stmt_id]
        domain_inames = stmt.within_inames | self._additional_inames
        inames_domain = nisl.make_set(
            self.kernel.get_inames_domain(domain_inames).to_set()
        )
        access_rel = self._get_access_relation(inames_domain, subscript)

        additional_inames = self._additional_inames - stmt.within_inames
        if additional_inames:
            access_rel = access_rel.project_out(additional_inames)

        match access_type:
            case AccessType.read:
                previous = self._read_relations[stmt_id].get(var)
                self._read_relations[stmt_id][var] = (
                    access_rel if previous is None else previous | access_rel
                )
            case AccessType.write:
                previous = self._write_relations[stmt_id].get(var)
                self._write_relations[stmt_id][var] = (
                    access_rel if previous is None else previous | access_rel
                )
            case _:
                raise ValueError("unknown AccessType")

    @cached_property
    def read_relations(self) -> Mapping[str, Mapping[str, nisl.Map]]:
        return constantdict({
            stmt_id: constantdict(relations)
            for stmt_id, relations in self._read_relations.items()
        })

    @cached_property
    def write_relations(self) -> Mapping[str, Mapping[str, nisl.Map]]:
        return constantdict({
            stmt_id: constantdict(relations)
            for stmt_id, relations in self._write_relations.items()
        })

    @override
    def map_variable(
        self, expr: prim.Variable, /, stmt_id: str, access_type: AccessType
    ) -> None:
        self._record_access(stmt_id, expr.name, (), access_type)

    @override
    def map_subscript(
        self, expr: prim.Subscript, /, stmt_id: str, access_type: AccessType
    ) -> None:
        assert isinstance(expr.aggregate, prim.Variable)
        self._record_access(
            stmt_id, expr.aggregate.name, expr.index_tuple, access_type
        )

    @override
    def map_linear_subscript(
        self, expr: LinearSubscript, /, stmt_id: str, access_type: AccessType
    ) -> None:
        self.rec(expr.index, stmt_id, AccessType.read)

        assert isinstance(expr.aggregate, prim.Variable)
        self._record_access(
            stmt_id, expr.aggregate.name, (expr.index,), access_type
        )

    @override
    def map_reduction(
        self, expr: Reduction, /, stmt_id: str, access_type: AccessType
    ) -> None:
        previous_inames = self._additional_inames
        self._additional_inames |= frozenset(expr.inames)
        try:
            WalkMapper.map_reduction(self, expr, stmt_id, access_type)
        finally:
            self._additional_inames = previous_inames

    @override
    def map_sub_array_ref(
        self, expr: SubArrayRef, /, stmt_id: str, access_type: AccessType
    ) -> None:
        previous_inames = self._additional_inames
        self._additional_inames |= frozenset(
            iname.name for iname in expr.swept_inames
        )
        try:
            self.rec(expr.subscript, stmt_id, access_type)
        finally:
            self._additional_inames = previous_inames


def apply_affine_transform_to_happens_afters(
    kernel: LoopKernel, affine_reln: nisl.Map
) -> LoopKernel:
    """
    Applies an affine transformation to all relevant happens-after relations.
    """

    affine_reln = affine_reln.coalesce()
    transformed_inames = frozenset(affine_reln.space.in_names)
    name_generator = kernel.get_var_name_generator()
    for names in affine_reln.space.dimtype_to_names.values():
        name_generator.add_names(names, conflicting_ok=True)

    def build_xform_reln(
        stmt: InstructionBase, suffix: str
    ) -> tuple[nisl.Map, tuple[tuple[str, str], ...]] | None:
        overlap = stmt.within_inames & transformed_inames
        if not overlap:
            return None
        if overlap != transformed_inames:
            raise LoopyError(
                f"statement '{stmt.id}' is within only part of the affine "
                "transformation's input inames"
            )

        # FIXME: Remove conversion once LoopKernel domains use namedisl.Set.
        stmt_domain = nisl.make_set(
            kernel.get_inames_domain(stmt.within_inames).to_set()
        )
        stmt_inames = stmt_domain.space.dimtype_to_names[DimType.out]
        nonxformed_names = tuple(
            name for name in stmt_inames if name not in transformed_inames
        )
        output_proxy_names = tuple(
            name_generator(f"{name}_new_") for name in nonxformed_names
        )

        xform_reln = affine_reln.add_dims(DimType.in_, nonxformed_names)
        xform_reln = xform_reln.add_dims(DimType.out, output_proxy_names)
        xform_reln = xform_reln.equate_dims(tuple(zip(
            nonxformed_names, output_proxy_names, strict=True
        )))
        xform_reln = _suffix_names(xform_reln, suffix, DimType.in_)
        xform_reln = _suffix_names(xform_reln, suffix, DimType.out)

        proxy_renames = tuple(
            (f"{proxy}{suffix}", f"{name}{suffix}")
            for name, proxy in zip(
                nonxformed_names, output_proxy_names, strict=True
            )
        )
        return xform_reln.coalesce(), proxy_renames

    new_stmts: list[InstructionBase] = []
    for sink_stmt in kernel.instructions:
        sink_xform = build_xform_reln(sink_stmt, "_after")
        new_happens_after: dict[str, HappensAfter] = {}

        for src_id, happens_after in sink_stmt.happens_after.items():
            if happens_after.instances_rel is None:
                raise LoopyError(
                    "cannot determine precise happens-after information"
                )

            src_stmt = kernel.id_to_insn[src_id]
            src_xform = build_xform_reln(src_stmt, "_before")
            if sink_xform is None and src_xform is None:
                new_happens_after[src_id] = happens_after
                continue

            # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
            instances_rel = nisl.make_map(
                happens_after.instances_rel
            ).coalesce()
            proxy_renames: list[tuple[str, str]] = []

            if sink_xform is not None:
                sink_xform_reln, sink_proxy_renames = sink_xform
                instances_rel = sink_xform_reln.reverse().apply_range(
                    instances_rel
                ).coalesce()
                proxy_renames.extend(sink_proxy_renames)

            if src_xform is not None:
                if instances_rel.space.dim(DimType.in_) == 0:
                    dummy_name = name_generator("happens_after_dummy")
                    instances_rel = (
                        instances_rel
                        .add_dims(DimType.in_, (dummy_name,))
                        .project_out((dummy_name,))
                    )

                src_xform_reln, src_proxy_renames = src_xform
                instances_rel = instances_rel.apply_range(
                    src_xform_reln
                ).coalesce()
                proxy_renames.extend(src_proxy_renames)

            instances_rel = instances_rel.rename_dims(proxy_renames).coalesce()
            # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
            new_happens_after[src_id] = HappensAfter(instances_rel.as_isl())

        new_stmts.append(
            sink_stmt.copy(happens_after=constantdict(new_happens_after))
        )

    return kernel.copy(instructions=new_stmts)


def has_precise_dependencies(kernel: LoopKernel) -> bool:
    has_precise = False
    has_legacy = False
    for stmt in kernel.instructions:
        for happens_after in stmt.happens_after.values():
            if happens_after.instances_rel is None:
                has_legacy = True
            else:
                has_precise = True

    if has_precise and has_legacy:
        raise LoopyError(
            f"kernel '{kernel.name}' mixes precise and legacy "
            "happens-after dependencies"
        )

    return has_precise


def _suffix_names(
    obj: NamedIslObjectT, suffix: str, dim_type: DimType
) -> NamedIslObjectT:
    return obj.rename_dims(
        (name, name + suffix) for name in obj.space.dimtype_to_names[dim_type]
    )


def _statement_instance_set(
    kernel: LoopKernel, stmt: InstructionBase, suffix: str
) -> nisl.Set:
    # FIXME: Remove conversion once LoopKernel domains use namedisl.Set.
    instance_set = nisl.make_set(
        kernel.get_inames_domain(stmt.within_inames).to_set()
    )
    unused_inames = instance_set.space.out_names - stmt.within_inames
    if unused_inames:
        instance_set = instance_set.project_out(unused_inames)

    return _suffix_names(instance_set, suffix, DimType.out).coalesce()


def _compose_happens_after_relations(
    first: nisl.Map, second: nisl.Map
) -> nisl.Map:
    first = first.coalesce()
    second = second.coalesce()
    first_interface = tuple(
        name.removesuffix("_before")
        for name in first.space.dimtype_to_names[DimType.out]
    )
    second_interface = tuple(
        name.removesuffix("_after")
        for name in second.space.dimtype_to_names[DimType.in_]
    )
    if frozenset(first_interface) != frozenset(second_interface):
        raise LoopyError(
            "cannot compose happens-after relations with different "
            "intermediate instance spaces"
        )

    first = first.rename_dims(zip(
        first.space.dimtype_to_names[DimType.out], first_interface, strict=True
    ))
    second = second.rename_dims(zip(
        second.space.dimtype_to_names[DimType.in_], second_interface, strict=True
    ))
    return first.apply_range(second).coalesce()


def _validate_instance_mapping(
    relation: nisl.Map,
    domain_instances: nisl.Set,
    range_instances: nisl.Set,
    *,
    relation_name: str,
    domain_name: str,
    range_name: str,
) -> nisl.Map:
    relation = relation.coalesce()
    if (
        relation.space.dimtype_to_names[DimType.in_]
        != domain_instances.space.dimtype_to_names[DimType.out]
    ):
        raise LoopyError(
            f"{relation_name} relation has the wrong {domain_name} "
            "instance space"
        )
    if (
        relation.space.dimtype_to_names[DimType.out]
        != range_instances.space.dimtype_to_names[DimType.out]
    ):
        raise LoopyError(
            f"{relation_name} relation has the wrong {range_name} "
            "instance space"
        )
    if not (relation.domain() - domain_instances).is_empty():
        raise LoopyError(
            f"{relation_name} relation contains instances outside the "
            f"{domain_name} domain"
        )
    if not (relation.range() - range_instances).is_empty():
        raise LoopyError(
            f"{relation_name} relation contains instances outside the "
            f"{range_name} domain"
        )

    return relation


def _add_or_union_happens_after(
    happens_after: dict[str, HappensAfter],
    sink_id: str,
    source_id: str,
    instances_rel: nisl.Map,
) -> None:
    instances_rel = instances_rel.coalesce()
    if instances_rel.is_empty():
        return

    previous = happens_after.get(source_id)
    if previous is not None:
        if previous.instances_rel is None:
            raise LoopyError(
                "cannot combine precise and imprecise happens-after "
                f"relations for '{sink_id}' and '{source_id}'"
            )

        # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
        previous_rel = nisl.make_map(previous.instances_rel)
        if (
            previous_rel.space.dimtype_to_names[DimType.in_]
            != instances_rel.space.dimtype_to_names[DimType.in_]
            or previous_rel.space.dimtype_to_names[DimType.out]
            != instances_rel.space.dimtype_to_names[DimType.out]
        ):
            raise LoopyError(
                "cannot union happens-after relations with different "
                "statement instance spaces"
            )
        instances_rel = (previous_rel | instances_rel).coalesce()

    # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
    happens_after[source_id] = HappensAfter(instances_rel.as_isl())


def splice_happens_after_as_consumer(
    kernel: LoopKernel,
    consumer_id: str,
    anchor_id: str,
    consumer_to_anchor: nisl.Map,
) -> LoopKernel:
    """Give *consumer_id* the incoming dependencies of *anchor_id*.

    *consumer_to_anchor* maps consumer instances to the anchor instances whose
    incoming dependencies they inherit. Its input dimensions use the consumer
    inames suffixed with ``"_after"`` and its output dimensions use the anchor
    inames suffixed with ``"_before"``.

    The anchor's self-edge is not inherited. Existing dependencies of the
    consumer are preserved and unioned with inherited dependencies to the same
    source.
    """

    if consumer_id == anchor_id:
        raise LoopyError("consumer and anchor instructions must be distinct")
    if not has_precise_dependencies(kernel):
        raise LoopyError("consumer splicing requires precise dependencies")

    try:
        consumer = kernel.id_to_insn[consumer_id]
        anchor = kernel.id_to_insn[anchor_id]
    except KeyError as err:
        raise LoopyError(f"unknown instruction ID '{err.args[0]}'") from err

    consumer_instances = _statement_instance_set(kernel, consumer, "_after")
    anchor_instances = _statement_instance_set(kernel, anchor, "_before")
    consumer_to_anchor = _validate_instance_mapping(
        consumer_to_anchor,
        consumer_instances,
        anchor_instances,
        relation_name="consumer-to-anchor",
        domain_name="consumer",
        range_name="anchor",
    )

    new_happens_after = dict(consumer.happens_after)
    for source_id, happens_after in anchor.happens_after.items():
        if source_id == anchor_id:
            continue
        if happens_after.instances_rel is None:
            raise LoopyError(
                "cannot inherit an imprecise happens-after relation from "
                f"'{anchor_id}' to '{source_id}'"
            )

        # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
        inherited = _compose_happens_after_relations(
            consumer_to_anchor,
            nisl.make_map(happens_after.instances_rel),
        )
        _add_or_union_happens_after(
            new_happens_after, consumer_id, source_id, inherited
        )

    return kernel.copy(instructions=tuple(
        stmt.copy(happens_after=constantdict(new_happens_after))
        if stmt.id == consumer_id else stmt
        for stmt in kernel.instructions
    ))


def splice_happens_after_as_producer(
    kernel: LoopKernel,
    producer_id: str,
    anchor_id: str,
    anchor_to_producer: nisl.Map,
) -> LoopKernel:
    """Redirect consumers of *anchor_id* to *producer_id*.

    *anchor_to_producer* maps the anchor instances being replaced to the
    producer instances that replace them. Its input dimensions use the anchor
    inames suffixed with ``"_after"`` and its output dimensions use the producer
    inames suffixed with ``"_before"``.

    If the map covers only part of the anchor instance space, dependencies on
    the remaining anchor instances are preserved. Existing dependencies on the
    producer are unioned with the redirected dependencies.
    """

    if producer_id == anchor_id:
        raise LoopyError("producer and anchor instructions must be distinct")
    if not has_precise_dependencies(kernel):
        raise LoopyError("producer splicing requires precise dependencies")

    try:
        producer = kernel.id_to_insn[producer_id]
        anchor = kernel.id_to_insn[anchor_id]
    except KeyError as err:
        raise LoopyError(f"unknown instruction ID '{err.args[0]}'") from err

    anchor_instances = _statement_instance_set(kernel, anchor, "_after")
    producer_instances = _statement_instance_set(kernel, producer, "_before")
    anchor_to_producer = _validate_instance_mapping(
        anchor_to_producer,
        anchor_instances,
        producer_instances,
        relation_name="anchor-to-producer",
        domain_name="anchor",
        range_name="producer",
    )

    mapped_anchor_instances = (
        anchor_to_producer.domain()
        .coalesce()
        .rename_dims(zip(
            anchor_to_producer.space.dimtype_to_names[DimType.in_],
            tuple(
                name.removesuffix("_after") + "_before"
                for name in anchor_to_producer.space.dimtype_to_names[DimType.in_]
            ),
            strict=True,
        ))
    )

    new_stmts: list[InstructionBase] = []
    for sink in kernel.instructions:
        if sink.id in {anchor_id, producer_id}:
            new_stmts.append(sink)
            continue

        happens_after = sink.happens_after.get(anchor_id)
        if happens_after is None:
            new_stmts.append(sink)
            continue
        if happens_after.instances_rel is None:
            raise LoopyError(
                "cannot redirect an imprecise happens-after relation from "
                f"'{sink.id}' to '{anchor_id}'"
            )

        # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
        anchor_order = nisl.make_map(happens_after.instances_rel).coalesce()
        redirected_anchor_order = anchor_order.intersect_range(
            mapped_anchor_instances
        ).coalesce()
        remaining_anchor_order = (
            anchor_order - redirected_anchor_order
        ).coalesce()
        redirected_order = _compose_happens_after_relations(
            redirected_anchor_order, anchor_to_producer
        )

        new_happens_after = dict(sink.happens_after)
        if remaining_anchor_order.is_empty():
            del new_happens_after[anchor_id]
        else:
            # FIXME: Remove conversion once HappensAfter stores namedisl.Map.
            new_happens_after[anchor_id] = HappensAfter(
                remaining_anchor_order.as_isl()
            )

        _add_or_union_happens_after(
            new_happens_after, sink.id, producer_id, redirected_order
        )

        new_stmts.append(
            sink.copy(happens_after=constantdict(new_happens_after))
        )

    return kernel.copy(instructions=tuple(new_stmts))


def splice_happens_after_as_consumer_and_producer(
    kernel: LoopKernel,
    instruction_id: str,
    anchor_id: str,
    instruction_to_anchor: nisl.Map,
    anchor_to_instruction: nisl.Map,
) -> LoopKernel:
    """Splice *instruction_id* across both sides of *anchor_id*.

    The new instruction inherits the anchor's incoming dependencies according
    to *instruction_to_anchor*. Dependencies on the mapped anchor instances are
    redirected to the new instruction according to *anchor_to_instruction*.
    The two relations are supplied independently and need not be inverses.
    """

    kernel = splice_happens_after_as_consumer(
        kernel,
        instruction_id,
        anchor_id,
        instruction_to_anchor,
    )
    return splice_happens_after_as_producer(
        kernel,
        instruction_id,
        anchor_id,
        anchor_to_instruction,
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

    new_stmts: list[InstructionBase] = []
    for i, stmt in enumerate(kernel.instructions):
        new_happens_after: dict[str, HappensAfter] = {}

        sources = (stmt,) if i == 0 else (stmt, kernel.instructions[i - 1])

        after_domain = nisl.make_set(
            kernel.get_inames_domain(stmt.within_inames).to_set()
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
            affs = joint_domain.var_pw_affs

            strict_lex = joint_domain - joint_domain
            equal_prefix = joint_domain
            for iname in shared_order:
                after_aff = affs[f"{iname}_after"]
                before_aff = affs[f"{iname}_before"]

                strict_lex = strict_lex | (
                    equal_prefix & after_aff.gt_set(before_aff)
                )
                equal_prefix = equal_prefix & after_aff.eq_set(before_aff)

            if source.id == stmt.id:
                ordered_instances = strict_lex
            else:
                ordered_instances = strict_lex | equal_prefix

            instances_rel = ordered_instances.as_map(
                in_names=tuple(f"{name}_after" for name in after_inames)
            )

            new_happens_after[source.id] = HappensAfter(
                instances_rel=instances_rel.as_isl()
            )

        new_stmts.append(stmt.copy(happens_after=new_happens_after))

    return kernel.copy(instructions=new_stmts)


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
                # FIXME: remove named conversion
                previous_instances_rel = nisl.make_map(previous.instances_rel)
                combined_order = required_order | previous_instances_rel

            # FIXME: remove unnamed conversion
            happens_after[source_id] = HappensAfter(combined_order.as_isl())

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
        source_stmt = kernel.id_to_insn[source_id]
        for src_dep_id, src_happens_after in source_stmt.happens_after.items():
            if src_dep_id == source_id:
                continue

            if src_happens_after.instances_rel is None:
                raise ValueError(
                    "All `HappensAfter`s must have precise dependencies "
                    "defined to use precise dependency finding machinery."
                )

            # FIXME: removed named conversion
            src_instances_rel = nisl.make_map(src_happens_after.instances_rel)
            outgoing_instances_rel = normalize_interface_and_compose(
                incoming_instances_rel, src_instances_rel
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
    for stmt in kernel.instructions:
        coarse_dependency_graph[stmt.id] = frozenset({
            dep for dep in stmt.happens_after if dep != stmt.id
        })

    topological_order = compute_topological_order(coarse_dependency_graph)

    rel_finder = AccessRelationFinder(kernel)
    for stmt in kernel.instructions:
        if isinstance(stmt, MultiAssignmentBase):
            for assignee in stmt.assignees:
                rel_finder(assignee, stmt.id, AccessType.write)
            rel_finder(stmt.expression, stmt.id, AccessType.read)
            for pred in stmt.predicates:
                rel_finder(pred, stmt.id, AccessType.read)

    new_stmts: list[InstructionBase] = []
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
                        nisl.make_map(happens_after.instances_rel),
                        access_relation,
                        rel_finder,
                        new_happens_after,
                    )

        new_stmts.append(
            kernel.id_to_insn[sink_id].copy(
                happens_after=constantdict(new_happens_after)
            )
        )

    return kernel.copy(instructions=new_stmts)

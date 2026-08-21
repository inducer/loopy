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
    CInstruction,
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
    get_dependencies,
)


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

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
    _constant_names: frozenset[str]
    _storage_variables: frozenset[str]
    _subst_expander: SubstitutionRuleExpander

    def __init__(self, kernel: LoopKernel):
        self.kernel = kernel
        self._additional_inames = frozenset()
        self._read_relations = {stmt.id: {} for stmt in kernel.instructions}
        self._write_relations = {stmt.id: {} for stmt in kernel.instructions}
        self._name_generator = kernel.get_var_name_generator()
        self._cell_names = []
        from loopy.kernel.data import ValueArg
        self._constant_names = frozenset(
            arg.name for arg in kernel.args
            if isinstance(arg, ValueArg)
            and arg.name not in kernel.get_written_variables()
        )
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
        subscript_dependencies = frozenset(
            dependency
            for index in subscript
            for dependency in get_dependencies(index)
        )
        domain = domain.add_dims(
            DimType.param,
            (subscript_dependencies & self._constant_names)
            - domain.space.param_names,
        )
        instance_names = domain.space.dimtype_to_names[DimType.out]
        while len(self._cell_names) < len(subscript):
            axis = len(self._cell_names)
            self._cell_names.append(self._name_generator(f"ax_{axis}"))

        cell_names = tuple(self._cell_names[: len(subscript)])

        access_set = domain.add_dims(DimType.out, cell_names)
        coordinates = access_set.var_pw_affs
        for cell_name, index_expr in zip(cell_names, subscript, strict=True):
            index_aff = aff_from_expr(
                access_set.var_affs,
                index_expr,
            ).as_pw_aff()

            access_set = access_set & coordinates[cell_name].eq_set(index_aff)

        return _set_as_map(access_set, in_names=instance_names)

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
        inames_domain = self.kernel.get_inames_domain(domain_inames)
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


def _set_as_map(
    set_: nisl.Set, in_names: Collection[str]
) -> nisl.Map:
    return set_.as_map(in_names)


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

        stmt_domain = kernel.get_inames_domain(stmt.within_inames)
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

            instances_rel = happens_after.instances_rel.coalesce()
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
            new_happens_after[src_id] = HappensAfter(
                instances_rel,
                variable_name=happens_after.variable_name,
            )

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
    instance_set = kernel.get_inames_domain(stmt.within_inames)
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


def _saturate_cross_relations_with_self_relations(
    kernel: LoopKernel,
) -> LoopKernel:
    self_relations: dict[str, nisl.Map] = {}
    for stmt in kernel.instructions:
        happens_after = stmt.happens_after.get(stmt.id)
        if happens_after is None:
            continue
        if happens_after.instances_rel is None:
            raise LoopyError(
                "self-relation saturation requires precise dependencies"
            )

        self_relation = happens_after.instances_rel.coalesce()
        if not (
            _compose_happens_after_relations(
                self_relation, self_relation
            ) - self_relation
        ).is_empty():
            raise LoopyError(
                f"self happens-after relation for '{stmt.id}' is not "
                "transitive"
            )

        self_relations[stmt.id] = self_relation

    new_stmts: list[InstructionBase] = []
    for stmt in kernel.instructions:
        new_happens_after: dict[str, HappensAfter] = {}
        for source_id, happens_after in stmt.happens_after.items():
            if happens_after.instances_rel is None:
                raise LoopyError(
                    "self-relation saturation requires precise dependencies"
                )

            relation = happens_after.instances_rel.coalesce()
            if source_id != stmt.id:
                self_relation = self_relations.get(stmt.id)
                if self_relation is not None:
                    relation = (
                        relation
                        | _compose_happens_after_relations(
                            self_relation, relation
                        )
                    ).coalesce()

                self_relation = self_relations.get(source_id)
                if self_relation is not None:
                    relation = (
                        relation
                        | _compose_happens_after_relations(
                            relation, self_relation
                        )
                    ).coalesce()

            new_happens_after[source_id] = HappensAfter(
                relation,
                variable_name=happens_after.variable_name,
            )

        new_stmts.append(
            stmt.copy(happens_after=constantdict(new_happens_after))
        )

    return kernel.copy(instructions=tuple(new_stmts))


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
    variable_name: str | None = None,
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

        previous_rel = previous.instances_rel
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
        if variable_name != previous.variable_name:
            variable_name = None

    happens_after[source_id] = HappensAfter(
        instances_rel,
        variable_name=variable_name,
    )


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

        inherited = _compose_happens_after_relations(
            consumer_to_anchor,
            happens_after.instances_rel,
        )
        _add_or_union_happens_after(
            new_happens_after,
            consumer_id,
            source_id,
            inherited,
            variable_name=happens_after.variable_name,
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

        anchor_order = happens_after.instances_rel.coalesce()
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
            new_happens_after[anchor_id] = HappensAfter(
                remaining_anchor_order,
                variable_name=happens_after.variable_name,
            )

        _add_or_union_happens_after(
            new_happens_after,
            sink.id,
            producer_id,
            redirected_order,
            variable_name=happens_after.variable_name,
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

        after_domain = kernel.get_inames_domain(stmt.within_inames)

        after_inames = after_domain.space.dimtype_to_names[DimType.out]
        after_domain = _suffix_names(after_domain, "_after", DimType.out)
        for source in sources:
            before_domain = kernel.get_inames_domain(source.within_inames)

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

            instances_rel = _set_as_map(
                ordered_instances,
                in_names=tuple(f"{name}_after" for name in after_inames)
            )

            new_happens_after[source.id] = HappensAfter(
                instances_rel=instances_rel
            )

        new_stmts.append(stmt.copy(happens_after=new_happens_after))

    return kernel.copy(instructions=new_stmts)


def _compute_reachable_happens_after(
    kernel: LoopKernel,
    topological_order: list[str],
) -> dict[str, dict[str, nisl.Map]]:
    """Return every nonempty fine-grained branch of the supplied order."""
    result: dict[str, dict[str, nisl.Map]] = {}

    for sink_id in topological_order:
        reachable: dict[str, nisl.Map] = {}
        self_happens_after = kernel.id_to_insn[sink_id].happens_after.get(sink_id)
        if self_happens_after is not None:
            assert self_happens_after.instances_rel is not None
            if not self_happens_after.instances_rel.is_empty():
                reachable[sink_id] = self_happens_after.instances_rel

        for intermediate_id in topological_order:
            if intermediate_id == sink_id:
                sink_to_intermediate = None
            else:
                sink_to_intermediate = reachable.get(intermediate_id)
                if sink_to_intermediate is None:
                    continue

            intermediate = kernel.id_to_insn[intermediate_id]
            for source_id, happens_after in intermediate.happens_after.items():
                if source_id == intermediate_id:
                    continue

                assert happens_after.instances_rel is not None
                edge_relation = happens_after.instances_rel
                if edge_relation.is_empty():
                    continue

                if sink_to_intermediate is None:
                    path_relation = edge_relation
                else:
                    path_relation = _compose_happens_after_relations(
                        sink_to_intermediate, edge_relation
                    )
                if path_relation.is_empty():
                    continue

                previous = reachable.get(source_id)
                reachable[source_id] = (
                    path_relation
                    if previous is None
                    else (previous | path_relation).coalesce()
                )

        result[sink_id] = reachable

    return result


def _find_conflicting_access_candidates(
    sink_access_relation: nisl.Map,
    sink_to_source: nisl.Map,
    source_access_relation: nisl.Map,
) -> nisl.Map:
    """Return ``(sink instance, cell) -> source instance`` conflicts."""
    source_access_relation = _suffix_names(
        source_access_relation, "_before", DimType.in_
    )
    sink_names = sink_to_source.space.in_names
    cell_names = sink_access_relation.space.out_names
    candidate_set = (
        sink_access_relation.as_set()
        & sink_to_source.as_set()
        & source_access_relation.as_set()
    )
    return _set_as_map(
        candidate_set, in_names=(*sink_names, *cell_names)
    ).coalesce()


def _discard_candidates_preceding_writers(
    candidates: Mapping[str, nisl.Map],
    writer_candidates: Mapping[str, nisl.Map],
    reachable_order: Mapping[str, Mapping[str, nisl.Map]],
) -> dict[str, nisl.Map]:
    """Remove candidates with a same-cell writer ordered after them."""
    result: dict[str, nisl.Map] = {}
    for candidate_id, candidate_relation in candidates.items():
        dominated = candidate_relation - candidate_relation
        for writer_id, writer_relation in writer_candidates.items():
            writer_to_candidate = reachable_order[writer_id].get(candidate_id)
            if writer_to_candidate is None:
                continue

            dominated = dominated | (
                candidate_relation
                & _compose_happens_after_relations(
                    writer_relation, writer_to_candidate
                )
            )

        remaining = (candidate_relation - dominated).coalesce()
        if not remaining.is_empty():
            result[candidate_id] = remaining

    return result


def _record_candidate_order(
    candidates: Mapping[str, nisl.Map],
    sink_names: Collection[str],
    cell_names: Collection[str],
    var: str,
    happens_after: dict[str, HappensAfter],
) -> None:
    for source_id, candidate_relation in candidates.items():
        required_order = _set_as_map(
            candidate_relation.as_set().project_out(cell_names),
            in_names=sink_names,
        ).coalesce()
        previous = happens_after.get(source_id)
        if previous is None:
            combined_order = required_order
        else:
            assert previous.instances_rel is not None
            combined_order = (
                required_order | previous.instances_rel
            ).coalesce()

        variable_name = (
            var
            if previous is None or previous.variable_name == var
            else None
        )
        happens_after[source_id] = HappensAfter(
            combined_order,
            variable_name=variable_name,
        )


@for_each_kernel
def relax_strict_happens_after(kernel: LoopKernel) -> LoopKernel:
    """
    Relaxes an incoming strict execution order imposed on statements in *kernel*
    through dependence analysis.

    :returns: *kernel* with the minimally required execution order on statement
    instances in a program needed to satisfy data dependencies.
    """

    for stmt in kernel.instructions:
        if isinstance(stmt, CInstruction):
            raise LoopyError(
                "precise dependency relaxation does not support "
                f"CInstruction '{stmt.id}'"
            )

    for temporary in kernel.temporary_variables.values():
        if temporary.base_storage is not None:
            raise LoopyError(
                "precise dependency relaxation does not support temporary "
                f"'{temporary.name}' with base_storage"
            )

    kernel = _saturate_cross_relations_with_self_relations(kernel)

    coarse_dependency_graph: dict[str, frozenset[str]] = {}
    for stmt in kernel.instructions:
        dependencies: set[str] = set()
        for source_id, happens_after in stmt.happens_after.items():
            if source_id == stmt.id:
                continue

            assert happens_after.instances_rel is not None
            relation = happens_after.instances_rel
            if not relation.is_empty():
                dependencies.add(source_id)

        coarse_dependency_graph[stmt.id] = frozenset(dependencies)

    topological_order = compute_topological_order(coarse_dependency_graph)
    reachable_order = _compute_reachable_happens_after(
        kernel, topological_order
    )

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
        for sink_access_type, access_relations in (
            (AccessType.read, rel_finder.read_relations[sink_id]),
            (AccessType.write, rel_finder.write_relations[sink_id]),
        ):
            for var, access_relation in access_relations.items():
                access_relation = _suffix_names(
                    access_relation, "_after", DimType.in_
                )
                writer_candidates: dict[str, nisl.Map] = {}
                reader_candidates: dict[str, nisl.Map] = {}
                for source_id, sink_to_source in reachable_order[sink_id].items():
                    source_writes = rel_finder.write_relations[source_id]
                    if var in source_writes:
                        candidates = _find_conflicting_access_candidates(
                            access_relation,
                            sink_to_source,
                            source_writes[var],
                        )
                        if not candidates.is_empty():
                            writer_candidates[source_id] = candidates

                    if sink_access_type == AccessType.write:
                        source_reads = rel_finder.read_relations[source_id]
                        if var in source_reads:
                            candidates = _find_conflicting_access_candidates(
                                access_relation,
                                sink_to_source,
                                source_reads[var],
                            )
                            if not candidates.is_empty():
                                reader_candidates[source_id] = candidates

                most_recent_writers = _discard_candidates_preceding_writers(
                    writer_candidates,
                    writer_candidates,
                    reachable_order,
                )
                sink_names = access_relation.space.in_names
                cell_names = access_relation.space.out_names
                _record_candidate_order(
                    most_recent_writers,
                    sink_names,
                    cell_names,
                    var,
                    new_happens_after,
                )

                if sink_access_type == AccessType.write:
                    live_readers = _discard_candidates_preceding_writers(
                        reader_candidates,
                        writer_candidates,
                        reachable_order,
                    )
                    _record_candidate_order(
                        live_readers,
                        sink_names,
                        cell_names,
                        var,
                        new_happens_after,
                    )

        new_stmts.append(
            kernel.id_to_insn[sink_id].copy(
                happens_after=constantdict(new_happens_after)
            )
        )

    return kernel.copy(instructions=new_stmts)

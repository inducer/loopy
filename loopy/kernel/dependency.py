from __future__ import annotations


__copyright__ = "Copyright (C) 2025 Addison Alvey-Blanco"

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

import namedisl as nisl

import islpy as isl

from loopy import HappensAfter, LoopKernel, for_each_kernel
from loopy.kernel.instruction import (
    InstructionBase,
    VariableSpecificHappensAfter,
)
from loopy.transform.dependency import AccessMapFinder


def _to_isl_map(map_: nisl.BasicMap | nisl.Map) -> isl.Map:
    raw_map = map_._reconstruct_isl_object()
    if isinstance(raw_map, isl.BasicMap):
        return isl.Map.from_basic_map(raw_map)
    return raw_map


def _namespace_name(role: str, identifier: str, name: str) -> str:
    return f"{role}${identifier}${name}"


def _base_name(name: str) -> str:
    return name.rsplit("$", 1)[-1]


def _namespace_set(set_: isl.BasicSet | isl.Set, role: str, insn_id: str) -> nisl.Set:
    obj = isl.Set.from_basic_set(set_) if isinstance(set_, isl.BasicSet) else set_
    for idim in range(obj.dim(isl.dim_type.set)):
        dim_name = obj.get_dim_name(isl.dim_type.set, idim)
        assert dim_name is not None
        obj = obj.set_dim_name(
            isl.dim_type.set,
            idim,
            _namespace_name(role, insn_id, dim_name),
        )
    return nisl.make_set(obj)


def _namespace_relation_map(
        map_: isl.Map,
        input_role: str,
        input_insn_id: str,
        output_insn_id: str,
    ) -> nisl.Map:
    for idim in range(map_.dim(isl.dim_type.in_)):
        dim_name = map_.get_dim_name(isl.dim_type.in_, idim)
        assert dim_name is not None
        map_ = map_.set_dim_name(
            isl.dim_type.in_,
            idim,
            _namespace_name(input_role, input_insn_id, dim_name),
        )

    for idim in range(map_.dim(isl.dim_type.out)):
        dim_name = map_.get_dim_name(isl.dim_type.out, idim)
        assert dim_name is not None
        map_ = map_.set_dim_name(
            isl.dim_type.out,
            idim,
            _namespace_name("dst", output_insn_id, dim_name.removesuffix("'")),
        )

    return nisl.make_map(map_)


def _namespace_access_map(
        map_: isl.Map,
        variable_name: str,
        input_role: str,
        insn_id: str,
    ) -> nisl.Map:
    for idim in range(map_.dim(isl.dim_type.in_)):
        dim_name = map_.get_dim_name(isl.dim_type.in_, idim)
        assert dim_name is not None
        map_ = map_.set_dim_name(
            isl.dim_type.in_,
            idim,
            _namespace_name(input_role, insn_id, dim_name),
        )

    for idim in range(map_.dim(isl.dim_type.out)):
        map_ = map_.set_dim_name(
            isl.dim_type.out,
            idim,
            f"var${variable_name}${idim}",
        )

    return nisl.make_map(map_)


def _dependency_map_to_isl(map_: nisl.BasicMap | nisl.Map) -> isl.Map:
    raw_map = _to_isl_map(map_)

    for idim in range(raw_map.dim(isl.dim_type.in_)):
        dim_name = raw_map.get_dim_name(isl.dim_type.in_, idim)
        assert dim_name is not None
        raw_map = raw_map.set_dim_name(isl.dim_type.in_, idim, _base_name(dim_name))

    for idim in range(raw_map.dim(isl.dim_type.out)):
        dim_name = raw_map.get_dim_name(isl.dim_type.out, idim)
        assert dim_name is not None
        raw_map = raw_map.set_dim_name(
            isl.dim_type.out,
            idim,
            _base_name(dim_name) + "'",
        )

    return raw_map


def _add_lexicographic_happens_after_inner(
        knl: LoopKernel,
        after_insn: InstructionBase,
        before_insn: InstructionBase,
    ) -> nisl.BasicMap | nisl.Map:
    domain_before = nisl.make_set(knl.get_inames_domain(before_insn.within_inames))
    domain_after = nisl.make_set(knl.get_inames_domain(after_insn.within_inames))
    renamed_domain_before = domain_before.rename_dims({
        name: name + "'"
        for name in domain_before.ordered_dim_names(isl.dim_type.out)
    })

    happens_after = nisl.make_map_from_domain_and_range(
        domain_after,
        renamed_domain_before,
    )
    assert isinstance(happens_after, nisl.BasicMap | nisl.Map)

    shared_inames = before_insn.within_inames & after_insn.within_inames

    shared_inames_order_before = [
        iname for iname in domain_before.ordered_dim_names(isl.dim_type.out)
        if iname in shared_inames
    ]

    shared_inames_order_after = [
        iname for iname in domain_after.ordered_dim_names(isl.dim_type.out)
        if iname in shared_inames
    ]

    assert shared_inames_order_after == shared_inames_order_before
    shared_inames_order = list(shared_inames_order_after)

    affs_in = isl.affs_from_space(happens_after.domain().get_space())
    affs_out = isl.affs_from_space(happens_after.range().get_space())

    lex_map = isl.Map.empty(happens_after.get_space())
    for iinnermost, innermost_iname in enumerate(shared_inames_order):
        innermost_map = affs_in[innermost_iname].gt_map(
            affs_out[innermost_iname + "'"]
        )

        for outer_iname in shared_inames_order[:iinnermost]:
            innermost_map = innermost_map & (
                affs_in[outer_iname].eq_map(
                    affs_out[outer_iname + "'"]
                )
            )

        if before_insn != after_insn:
            innermost_map = innermost_map | (
                affs_in[shared_inames_order[iinnermost]].eq_map(
                    affs_out[shared_inames_order[iinnermost] + "'"]
                )
            )

        lex_map = lex_map | innermost_map

    return happens_after & nisl.make_map(lex_map)


@for_each_kernel
def add_lexicographic_happens_after(knl: LoopKernel) -> LoopKernel:
    """
    Impose a sequential, top-down execution order to instructions in a program.
    It is expected that this strict order will be relaxed with
    :func:`reduce_strict_ordering_with_dependencies` using data dependencies.
    """

    rmap = knl.reader_map()
    wmap_r: dict[str, set[str]] = {}
    for var, insns in knl.writer_map().items():
        for insn in insns:
            wmap_r.setdefault(insn, set())
            wmap_r[insn].add(var)

    new_insns = []
    for iafter, after_insn in enumerate(knl.instructions):
        assert after_insn.id is not None

        new_happens_after = {}

        # check for self dependencies
        for var in wmap_r[after_insn.id]:
            if rmap.get(var) and after_insn.id in rmap[var]:
                self_happens_after = _add_lexicographic_happens_after_inner(
                    knl, after_insn, after_insn
                )
                new_happens_after[after_insn.id] = HappensAfter(
                    _to_isl_map(self_happens_after)
                )

        if iafter != 0:
            before_insn = knl.instructions[iafter - 1]
            happens_after = _add_lexicographic_happens_after_inner(
                knl, after_insn, before_insn
            )
            new_happens_after[before_insn.id] = HappensAfter(
                _to_isl_map(happens_after)
            )

        new_insns.append(after_insn.copy(happens_after=new_happens_after))

    return knl.copy(instructions=new_insns)


@for_each_kernel
def reduce_strict_ordering(knl: LoopKernel) -> LoopKernel:
    def narrow_dependencies(
            after: InstructionBase,
            before: InstructionBase,
            remaining_instances: nisl.BasicSet | nisl.Set,
            happens_afters: dict[str, VariableSpecificHappensAfter] | None = None,
            happens_after_map: nisl.BasicMap | nisl.Map | None = None,
        ) -> dict[str, VariableSpecificHappensAfter]:
        # FIXME: can we get rid of all the "assert x is not None" stuff?

        assert isinstance(after.id, str)
        assert isinstance(before.id, str)
        if happens_afters is None:
            happens_afters = {}

        if remaining_instances.is_empty():
            return happens_afters

        for insn, happens_after in before.happens_after.items():
            if happens_after_map is None:
                assert happens_after.instances_rel is not None
                happens_after_map = _namespace_relation_map(
                    happens_after.instances_rel,
                    "src",
                    after.id,
                    insn,
                )
            else:
                assert happens_after.instances_rel is not None
                happens_after_map = happens_after_map.apply_range(
                    _namespace_relation_map(
                        happens_after.instances_rel,
                        "dst",
                        before.id,
                        insn,
                    ))

            source_vars = access_mapper.get_accessed_variables(after.id)
            assert source_vars is not None
            common_vars = wmap_r[insn] & source_vars
            for var in common_vars:
                write_map = access_mapper.get_map(insn, var)
                source_map = access_mapper.get_map(after.id, var)

                assert write_map is not None
                assert source_map is not None

                source_to_writer = _namespace_access_map(
                    source_map,
                    var,
                    "src",
                    after.id,
                ).apply_range(
                    _namespace_access_map(
                        write_map,
                        var,
                        "dst",
                        insn,
                    ).reverse()
                )
                assert happens_after_map is not None
                dependency_map = source_to_writer & happens_after_map
                remaining_instances = remaining_instances - dependency_map.domain()
                if not dependency_map.is_empty():
                    happens_after_obj = VariableSpecificHappensAfter(
                        _dependency_map_to_isl(dependency_map), var
                    )

                    happens_afters = happens_afters | {insn: happens_after_obj}

            if insn != after.id:
                happens_afters = happens_afters | narrow_dependencies(
                    after,
                    knl.id_to_insn[insn],
                    remaining_instances,
                    happens_afters,
                    happens_after_map,
                )

        return happens_afters

    access_mapper = AccessMapFinder(knl)
    for insn in knl.instructions:
        access_mapper(insn.expression, insn.id)
        access_mapper(insn.assignee, insn.id)

    wmap_r: dict[str, set[str]] = {}
    for var, insns in knl.writer_map().items():
        for insn in insns:
            wmap_r.setdefault(insn, set())
            wmap_r[insn].add(var)

    new_insns = []
    for insn in knl.instructions[::-1]:
        new_insns.append(
            insn.copy(happens_after=narrow_dependencies(
                after=insn,
                before=insn,
                remaining_instances=_namespace_set(
                    knl.get_inames_domain(insn.within_inames)
                , "src", insn.id)))
        )

    return knl.copy(instructions=new_insns[::-1])

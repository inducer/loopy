from __future__ import annotations


__copyright__ = """
Copyright (C) 2016 Matt Wala
Copyright (C) 2020 University of Illinois Board of Trustees
Copyright (C) 2022 Kaushik Kulkarni
"""

__doc__ = """
.. autofunction:: get_block_boundaries
.. autofunction:: temporaries_read_in_subkernel
.. autofunction:: args_read_in_subkernel
.. autofunction:: args_written_in_subkernel
.. autofunction:: supporting_temporary_names

.. autoclass:: KernelArgInfo
.. autoclass:: SubKernelArgInfo

.. autofunction:: get_kernel_arg_info
.. autofunction:: get_subkernel_arg_info

.. autofunction:: get_return_from_kernel_mapping

.. autoclass:: AccessMapDescriptor
.. autoclass:: WriteRaceChecker

.. autoclass:: LoopNestTree
.. autoclass:: LoopTree

.. autofunction:: separate_loop_nest
.. autofunction:: get_partial_loop_nest_tree
.. autofunction:: get_loop_tree

References
^^^^^^^^^^

.. class:: InameStrSet

    See :class:`loopy.typing.InameStrSet`
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

import contextlib
import enum
import itertools
from dataclasses import dataclass
from functools import cached_property, reduce
from typing import TYPE_CHECKING, TypeAlias

from constantdict import constantdict

import islpy as isl
from pytools import fset_union, memoize_method, memoize_on_first_arg

from loopy.diagnostic import LoopyError
from loopy.kernel.data import AddressSpace, ArrayArg, TemporaryVariable
from loopy.schedule.tree import Tree
from loopy.typing import InameStr, InameStrSet, not_none


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence, Set

    from loopy.kernel import LoopKernel
    from loopy.schedule import ScheduleItem


# {{{ block boundary finder

def get_block_boundaries(schedule: Sequence[ScheduleItem]) -> Mapping[int, int]:
    r"""
    Return a dictionary mapping indices of
    :class:`loopy.schedule.BeginBlockItem`\ s to
    :class:`loopy.schedule.EndBlockItem`\ s and vice versa.
    """
    from loopy.schedule import BeginBlockItem, EndBlockItem
    block_bounds = {}
    active_blocks = []
    for idx, sched_item in enumerate(schedule):
        if isinstance(sched_item, BeginBlockItem):
            active_blocks.append(idx)
        elif isinstance(sched_item, EndBlockItem):
            start = active_blocks.pop()
            block_bounds[start] = idx
            block_bounds[idx] = start
    return block_bounds

# }}}


# {{{ subkernel tools

def temporaries_read_in_subkernel(
        kernel: LoopKernel, subkernel_name: str) -> frozenset[str]:
    from loopy.kernel.tools import get_subkernel_to_insn_id_map
    insn_ids = get_subkernel_to_insn_id_map(kernel)[subkernel_name]
    inames = frozenset().union(*(kernel.insn_inames(insn_id)
                                 for insn_id in insn_ids))
    domain_idxs = {kernel.get_home_domain_index(iname) for iname in inames}
    params = fset_union(
        kernel.domains[dom_idx].get_var_names_not_none(isl.dim_type.param)
        for dom_idx in domain_idxs)

    return (frozenset(tv
                for insn_id in insn_ids
                for tv in kernel.id_to_insn[insn_id].read_dependency_names()
                if tv in kernel.temporary_variables)
            | (params & frozenset(kernel.temporary_variables)))


def temporaries_written_in_subkernel(
        kernel: LoopKernel, subkernel_name: str) -> frozenset[str]:
    from loopy.kernel.tools import get_subkernel_to_insn_id_map
    insn_ids = get_subkernel_to_insn_id_map(kernel)[subkernel_name]
    return frozenset(tv
                for insn_id in insn_ids
                for tv in kernel.id_to_insn[insn_id].assignee_var_names()
                if tv in kernel.temporary_variables)


def args_read_in_subkernel(
        kernel: LoopKernel, subkernel_name: str) -> frozenset[str]:
    from loopy.kernel.tools import get_subkernel_to_insn_id_map
    insn_ids = get_subkernel_to_insn_id_map(kernel)[subkernel_name]
    inames = frozenset().union(*(kernel.insn_inames(insn_id)
                                 for insn_id in insn_ids))
    domain_idxs = {kernel.get_home_domain_index(iname) for iname in inames}
    params = frozenset().union(*(
        kernel.domains[dom_idx].get_var_names_not_none(isl.dim_type.param)
        for dom_idx in domain_idxs))
    return (frozenset(arg
                for insn_id in insn_ids
                for arg in kernel.id_to_insn[insn_id].read_dependency_names()
                if arg in kernel.arg_dict)
            | (params & frozenset(kernel.arg_dict)))


def args_written_in_subkernel(
        kernel: LoopKernel, subkernel_name: str) -> frozenset[str]:
    from loopy.kernel.tools import get_subkernel_to_insn_id_map
    insn_ids = get_subkernel_to_insn_id_map(kernel)[subkernel_name]
    return frozenset(arg
                for insn_id in insn_ids
                for arg in kernel.id_to_insn[insn_id].assignee_var_names()
                if arg in kernel.arg_dict)


def supporting_temporary_names(
        kernel: LoopKernel, tv_names: frozenset[str]) -> frozenset[str]:
    result: set[str] = set()

    for name in tv_names:
        tv = kernel.temporary_variables[name]
        for supp_name in tv.supporting_names():
            if supp_name in kernel.temporary_variables:
                result.add(supp_name)

    return frozenset(result)


def _get_temporaries_accessed_in_schedule(
            kernel: LoopKernel,
            sched_idx_lower_bound: int,
            sched_idx_upper_bound: int
        ) -> frozenset[str]:
    from loopy.schedule import CallKernel, EnterLoop, LeaveLoop

    linearization = kernel.linearization
    assert linearization is not None

    temporaries: frozenset[str] = frozenset()
    for sched_index in range(sched_idx_lower_bound, sched_idx_upper_bound):
        sched_item = linearization[sched_index]
        if isinstance(sched_item, CallKernel):
            temporaries = (
                temporaries_written_in_subkernel(kernel, sched_item.kernel_name)
                | temporaries_read_in_subkernel(
                    kernel, sched_item.kernel_name
                )
                | (temporaries)
            )
        elif isinstance(sched_item, (EnterLoop, LeaveLoop)):
            # ignore further outside-kernel loops
            pass

        else:
            raise NotImplementedError("kernel with non-CallKernel outermost")

    return temporaries


def _map_to_base_storage(kernel: LoopKernel, tv_names: Set[str]) -> Set[str]:
    result: set[str] = set()
    for tv_name in tv_names:
        while True:
            tv = kernel.temporary_variables[tv_name]
            if tv.base_storage is not None:
                tv_name = tv.base_storage
            else:
                break

        result.add(tv_name)

    return result


@memoize_on_first_arg
def get_sched_index_to_first_and_last_used(
        kernel: LoopKernel
    ) -> tuple[Mapping[int, Set[str]], Mapping[int, Set[str]]]:
    """
    Returns the tuple (first_used, last_used), where first_used is
    a dict such that first_used[sched_index] is the set of all global temporary
    variable names first used at sched_index.

    Likewise, last_used[sched_index] is the set of all global temporary
    variable names last used at sched_index.
    """
    from loopy.kernel.data import AddressSpace
    from loopy.schedule import CallKernel, EnterLoop, Barrier

    assert kernel.linearization is not None

    global_temporaries = frozenset(
        tv.name for tv in kernel.temporary_variables.values()
        if tv.address_space == AddressSpace.GLOBAL
    )

    # Collapse into blocks
    block_boundaries = get_block_boundaries(kernel.linearization)

    tvs_accessed_at: dict[int, frozenset[str]] = {}
    sched_index = 0
    while sched_index < len(kernel.linearization):
        sched_item = kernel.linearization[sched_index]
        if isinstance(sched_item, CallKernel):
            block_end = block_boundaries[sched_index]
            tvs_accessed_at[sched_index] = (
                temporaries_written_in_subkernel(kernel, sched_item.kernel_name)
                | temporaries_read_in_subkernel(
                    kernel, sched_item.kernel_name
                )
            ) & global_temporaries

            sched_index = block_end + 1

        elif isinstance(sched_item, EnterLoop):
            block_end = block_boundaries[sched_index]
            tvs_accessed_at[sched_index] = _get_temporaries_accessed_in_schedule(
                kernel, sched_index, block_end+1
            ) & global_temporaries

            sched_index = block_end + 1

        elif isinstance(sched_item, Barrier):
            sched_index += 1
        else:
            raise ValueError(
                    f"unexpected schedule item at outermost level: {type(sched_item)}")

    storage_vars_accessed_at = {
        sched_index: _map_to_base_storage(kernel, accessed)
        for sched_index, accessed in tvs_accessed_at.items()
    }
    del tvs_accessed_at

    # forward pass for first_accesses
    first_accesses: dict[int, Set[str]] = {}
    seen_storage_vars: set[str] = set()
    for sched_index in range(0, len(kernel.linearization)):
        accessed = storage_vars_accessed_at.get(sched_index, set())
        new_storage_vars = accessed - seen_storage_vars
        seen_storage_vars.update(accessed)

        if new_storage_vars:
            first_accesses[sched_index] = new_storage_vars

    # backward pass for last_accesses
    last_accesses: dict[int, Set[str]] = {}
    seen_storage_vars = set()
    for sched_index in range(len(kernel.linearization)-1, -1, -1):
        accessed = storage_vars_accessed_at.get(sched_index, set())
        new_storage_vars = accessed - seen_storage_vars
        seen_storage_vars.update(accessed)

        if new_storage_vars:
            last_accesses[sched_index] = new_storage_vars

    return (first_accesses, last_accesses)

# }}}


# {{{ argument lists

@dataclass(frozen=True)
class KernelArgInfo:
    """
    .. autoattribute:: passed_arg_names
    .. autoattribute:: written_names
    """

    passed_arg_names: Sequence[str]
    written_names: frozenset[str]

    @property
    def passed_names(self) -> Sequence[str]:
        return self.passed_arg_names


@dataclass(frozen=True)
class SubKernelArgInfo(KernelArgInfo):
    """Inherits from :class:`KernelArgInfo`.

    .. autoattribute:: passed_inames
    .. autoattribute:: passed_temporaries
    """

    passed_inames: Sequence[str]
    passed_temporaries: Sequence[str]

    @property
    def passed_names(self) -> Sequence[str]:
        return (list(self.passed_arg_names)
                + list(self.passed_inames)
                + list(self.passed_temporaries))


def _should_temp_var_be_passed(tv: TemporaryVariable) -> bool:
    return tv.address_space == AddressSpace.GLOBAL and tv.initializer is None


class _SupportingNameTracker:
    def __init__(self, kernel: LoopKernel):
        self.kernel = kernel
        self.name_to_main_name: dict[str, str] = {}

    def add_supporting_names_for(self, name):
        var_descr = self.kernel.get_var_descriptor(name)
        for supp_name in var_descr.supporting_names():
            self.name_to_main_name[supp_name] = (
                    self.name_to_main_name.get(supp_name, frozenset())
                    | {name})

    def get_additional_args_and_tvs(
            self, already_passed: set[str]
            ) -> tuple[list[str], list[str]]:
        additional_args = []
        additional_temporaries = []

        for supporting_name in sorted(frozenset(self.name_to_main_name)):
            if supporting_name not in already_passed:
                already_passed.add(supporting_name)
                var_descr = self.kernel.get_var_descriptor(supporting_name)
                if isinstance(var_descr, TemporaryVariable):
                    if _should_temp_var_be_passed(var_descr):
                        additional_temporaries.append(supporting_name)
                else:
                    additional_args.append(supporting_name)

        return additional_args, additional_temporaries


def _process_args_for_arg_info(
        kernel: LoopKernel, args_read: set[str], args_written: set[str],
        supp_name_tracker: _SupportingNameTracker, used_only: bool,
        ) -> list[str]:

    args_expected: set[str] = set()

    passed_arg_names = []
    for arg in kernel.args:
        if used_only and not (arg.name in args_read or arg.name in args_written):
            continue

        with contextlib.suppress(KeyError):
            args_expected.remove(arg.name)

        # Disregard the original array if it had a sep-tagged axis.
        if isinstance(arg, ArrayArg):
            if not arg._separation_info:
                passed_arg_names.append(arg.name)
                supp_name_tracker.add_supporting_names_for(arg.name)
            else:
                for sep_name in sorted(arg._separation_info.subarray_names.values()):
                    # Separated arrays occur later in the argument list.
                    # Mark them as accessed if the original array was,
                    # we'll stumble on them when it is their turn.
                    # Add them to args_expected to ensure they're not missed.
                    if arg.name in args_read:
                        args_read.add(sep_name)
                        args_expected.add(sep_name)
                    if arg.name in args_written:
                        args_written.add(sep_name)
                        args_expected.add(sep_name)

        else:
            passed_arg_names.append(arg.name)
            supp_name_tracker.add_supporting_names_for(arg.name)

    assert not args_expected

    return passed_arg_names


def get_kernel_arg_info(kernel: LoopKernel) -> KernelArgInfo:
    args_written = set(kernel.arg_dict) & kernel.get_written_variables()

    supp_name_tracker = _SupportingNameTracker(kernel)

    passed_arg_names = _process_args_for_arg_info(kernel,
            args_read=set(), args_written=args_written,
            supp_name_tracker=supp_name_tracker,
            used_only=False)

    additional_args, additional_temporaries = \
            supp_name_tracker.get_additional_args_and_tvs(
                    already_passed=(
                        set(passed_arg_names)))

    assert not additional_temporaries

    return KernelArgInfo(
            passed_arg_names=passed_arg_names + additional_args,
            written_names=frozenset(args_written))


def get_subkernel_arg_info(
        kernel: LoopKernel, subkernel_name: str) -> SubKernelArgInfo:
    assert kernel.linearization is not None

    args_read = set(args_read_in_subkernel(kernel, subkernel_name))
    args_written = set(args_written_in_subkernel(kernel, subkernel_name))

    tvs_read = temporaries_read_in_subkernel(kernel, subkernel_name)
    tvs_written = set(temporaries_written_in_subkernel(kernel, subkernel_name))

    supp_name_tracker = _SupportingNameTracker(kernel)

    passed_arg_names = _process_args_for_arg_info(kernel,
            args_read=args_read, args_written=args_written,
            supp_name_tracker=supp_name_tracker,
            used_only=True)

    passed_temporaries: list[str] = []
    for tv_name in sorted(tvs_read | tvs_written):
        supp_name_tracker.add_supporting_names_for(tv_name)
        tv = kernel.temporary_variables[tv_name]

        if _should_temp_var_be_passed(tv):
            if tv.base_storage:
                if tv_name in tvs_written and tv_name in tvs_written:
                    tvs_written.add(tv.base_storage)
            else:
                passed_temporaries.append(tv.name)

    additional_args, additional_temporaries = \
            supp_name_tracker.get_additional_args_and_tvs(
                    already_passed=(
                        set(passed_arg_names) | set(passed_temporaries)))

    from loopy.kernel.tools import get_subkernel_extra_inames

    return SubKernelArgInfo(
            passed_arg_names=passed_arg_names + additional_args,
            passed_inames=sorted(get_subkernel_extra_inames(kernel)[subkernel_name]),
            passed_temporaries=passed_temporaries + additional_temporaries,
            written_names=frozenset(args_written | tvs_written))

# }}}


# {{{ get_return_from_kernel_mapping

def get_return_from_kernel_mapping(kernel: LoopKernel) -> Mapping[int, int | None]:
    """
    Returns a mapping from schedule index of every schedule item (S) in
    *kernel* to the schedule index of :class:`loopy.schedule.ReturnFromKernel`
    of the active sub-kernel at 'S'.
    """
    from loopy.kernel import LoopKernel
    from loopy.schedule import (
        Barrier,
        CallKernel,
        EnterLoop,
        LeaveLoop,
        ReturnFromKernel,
        RunInstruction,
    )
    assert isinstance(kernel, LoopKernel)
    assert isinstance(kernel.linearization, list)
    return_from_kernel_idxs: dict[int, int | None] = {}
    current_return_from_kernel: int | None = None
    for sched_idx, sched_item in list(enumerate(kernel.linearization))[::-1]:
        if isinstance(sched_item, CallKernel):
            return_from_kernel_idxs[sched_idx] = current_return_from_kernel
            current_return_from_kernel = None
        elif isinstance(sched_item, ReturnFromKernel):
            assert current_return_from_kernel is None
            current_return_from_kernel = sched_idx
            return_from_kernel_idxs[sched_idx] = current_return_from_kernel
        elif isinstance(sched_item, (RunInstruction, EnterLoop, LeaveLoop,
                                     Barrier)):
            return_from_kernel_idxs[sched_idx] = current_return_from_kernel
        else:
            raise NotImplementedError(type(sched_item))

    return return_from_kernel_idxs

# }}}


# {{{ check for write races in accesses

def _check_for_access_races(map_a, insn_a, map_b, insn_b, knl, callables_table,
                            address_space):
    """
    Returns *True* if the execution instances of *insn_a* and *insn_b*, accessing
    the same variable via access maps *map_a* and *map_b*, result in an access race.

    :arg address_space: An instance of :class:`loopy.kernel.data.AddressSpace`
        of the variable whose accesses are being checked for a race.

    .. note::

        The accesses ``map_a``, ``map_b`` lead to write races iff there exists 2
        *unequal* global ids that access the same address.
    """
    import pymbolic.primitives as p

    from loopy.kernel.data import (
        AddressSpace,
        HardwareConcurrentTag,
        filter_iname_tags_by_type,
    )
    from loopy.kernel.tools import get_hw_axis_base_for_codegen
    from loopy.symbolic import aff_from_expr, aff_to_expr, isl_set_from_expr

    assert address_space in [AddressSpace.LOCAL, AddressSpace.GLOBAL]

    gsize, lsize = knl.get_grid_size_upper_bounds(callables_table,
                                                  return_dict=True)

    # {{{ Step 1: Preprocess the maps

    # Step 1.1: Project out inames which are also map's dims, but does not form the
    #           insn's within_inames
    # Step 1.2: Perform any offsetting required to the hw axes iname terms
    # Step 1.3: Project out sequential inames in the access maps
    # Step 1.4: Rename the dims with their iname tags i.e. (g.i or l.i)
    # Step 1.5: Name the ith output dims as _lp_dim{i}

    updated_maps = []

    for (map_, insn) in [
            (map_a, insn_a),
            (map_b, insn_b)]:
        dims_not_to_project_out = ({iname
                                    for iname in insn.within_inames
                                    if knl.iname_tags_of_type(
                                        iname, HardwareConcurrentTag)}
                                   | knl.all_params())
        map_ = map_.project_out_except(sorted(dims_not_to_project_out),
                                       [isl.dim_type.in_,
                                        isl.dim_type.param,
                                        isl.dim_type.div,
                                        isl.dim_type.cst])

        for name, (dt, pos) in map_.get_var_dict().items():
            if dt == isl.dim_type.in_:
                tag, = filter_iname_tags_by_type(knl.inames[name].tags,
                                                 HardwareConcurrentTag)

                iname_lower_bound = get_hw_axis_base_for_codegen(knl, name)

                if not iname_lower_bound.plain_is_zero():
                    # Hardware inames with nonzero base have an offset applied in
                    # code generation:
                    # https://github.com/inducer/loopy/blob/4e0b1c7635afe1473c8636377f8e7ef6d78dfd46/loopy/codegen/loop.py#L293-L297
                    # https://github.com/inducer/loopy/issues/600#issuecomment-1104066735

                    map_ = map_.add_dims(isl.dim_type.out, 1)
                    map_ = map_.move_dims(
                        isl.dim_type.in_, pos+1,
                        isl.dim_type.out, map_.dim(isl.dim_type.out)-1,
                        1
                    )
                    map_ = map_.set_dim_name(isl.dim_type.in_, pos+1, name+"'")

                    lbound_offset_expr_aff = aff_from_expr(
                        map_.domain().space,
                        (p.Variable(name+"'")
                         + aff_to_expr(iname_lower_bound)
                         - p.Variable(name))
                    )
                    lbound_offset_as_domain = lbound_offset_expr_aff.zero_basic_set()
                    map_ = map_.intersect_domain(lbound_offset_as_domain)

                    map_ = map_.project_out(dt, pos, 1)
                    assert map_.get_dim_name(dt, pos) == name+"'"
                    map_ = map_.set_dim_name(dt, pos, name)

                map_ = map_.set_dim_name(dt, pos, str(tag))

        for i_l in lsize:
            if f"l.{i_l}" not in map_.get_var_dict():
                ndim = map_.dim(isl.dim_type.in_)
                map_ = map_.add_dims(isl.dim_type.in_, 1)
                map_ = map_.set_dim_name(isl.dim_type.in_, ndim, f"l.{i_l}")

        for i_g in gsize:
            if f"g.{i_g}" not in map_.get_var_dict():
                ndim = map_.dim(isl.dim_type.in_)
                map_ = map_.add_dims(isl.dim_type.in_, 1)
                map_ = map_.set_dim_name(isl.dim_type.in_, ndim, f"g.{i_g}")

        for pos in range(map_.dim(isl.dim_type.out)):
            map_ = map_.set_dim_name(isl.dim_type.out, pos, f"_lp_dim{pos}")

        updated_maps.append(map_)

    map_a, map_b = updated_maps

    # }}}

    # {{{ Step 2: rename all lid's, gid's in map_a to lid.A, gid.A

    for name, (dt, pos) in map_a.get_var_dict().items():
        if dt == isl.dim_type.in_:
            map_a = map_a.set_dim_name(dt, pos, name+".A")

    # }}}

    # {{{ Step 3: rename all lid's, gid's in map_b to lid.B, gid.B

    for name, (dt, pos) in map_b.get_var_dict().items():
        if dt == isl.dim_type.in_:
            map_b = map_b.set_dim_name(dt, pos, name+".B")

    # }}}

    # {{{ Step 4: make map_a, map_b ISL sets

    map_a, map_b = isl.align_two(map_a, map_b)
    map_a = map_a.move_dims(isl.dim_type.in_, map_a.dim(isl.dim_type.in_),
                            isl.dim_type.out, 0, map_a.dim(isl.dim_type.out))

    map_b = map_b.move_dims(isl.dim_type.in_, map_b.dim(isl.dim_type.in_),
                            isl.dim_type.out, 0, map_b.dim(isl.dim_type.out))
    set_a = map_a.domain()
    set_b = map_b.domain()

    # }}}

    assert set_a.get_space() == set_b.get_space()

    # {{{ Step 5: create the set any(l.i.A != l.i.B) OR any(g.i.A != g.i.B)

    space = set_a.space
    unequal_local_id_set = isl.Set.empty(set_a.get_space())
    unequal_group_id_set = isl.Set.empty(set_a.get_space())
    equal_group_id_set = isl.BasicSet.universe(set_a.get_space())

    for i_l in lsize:
        lid_a = p.Variable(f"l.{i_l}.A")
        lid_b = p.Variable(f"l.{i_l}.B")
        unequal_local_id_set |= (isl_set_from_expr(space,
                                                   p.Comparison(lid_a, "!=", lid_b))
                                 )

    for i_g in gsize:
        gid_a = p.Variable(f"g.{i_g}.A")
        gid_b = p.Variable(f"g.{i_g}.B")
        unequal_group_id_set |= (isl_set_from_expr(space,
                                                   p.Comparison(gid_a, "!=", gid_b))
                                 )
        equal_group_id_set &= (isl_set_from_expr(space,
                                                 p.Comparison(gid_a, "==", gid_b))
                               )

    # }}}

    if address_space == AddressSpace.GLOBAL:
        return not (set_a
                    & set_b
                    & (unequal_local_id_set
                       | unequal_group_id_set)
                    ).is_empty()
    else:
        return not (set_a
                    & set_b
                    & unequal_local_id_set
                    & equal_group_id_set).is_empty()


class AccessMapDescriptor(enum.Enum):
    """
    Special access map values.

    :attr DOES_NOT_ACCESS: Describes an unaccessed variable.
    :attr NON_AFFINE_ACCESS: Describes a non-quasi-affine access into an array.
    """
    DOES_NOT_ACCESS = enum.auto()
    NON_AFFINE_ACCESS = enum.auto()


class WriteRaceChecker:
    """Used for checking for overlap between access ranges of instructions."""

    def __init__(self, kernel, callables_table):
        self.kernel = kernel
        self.callables_table = callables_table

    @cached_property
    def vars(self):
        return (self.kernel.get_written_variables()
                | self.kernel.get_read_variables())

    @memoize_method
    def _get_access_maps(self, insn_id, access_dir):
        from collections import defaultdict

        from loopy.symbolic import BatchedAccessMapMapper

        insn = self.kernel.id_to_insn[insn_id]

        exprs = list(insn.assignees)
        if access_dir == "any":
            exprs.append(insn.expression)
            exprs.extend(insn.predicates)

        access_maps = defaultdict(lambda: AccessMapDescriptor.DOES_NOT_ACCESS)

        arm = BatchedAccessMapMapper(self.kernel, self.vars, overestimate=True)

        for expr in exprs:
            arm(expr, insn.within_inames)

        for name in arm.access_maps:
            if arm.bad_subscripts[name]:
                access_maps[name] = AccessMapDescriptor.NON_AFFINE_ACCESS
                continue
            access_maps[name] = arm.access_maps[name][insn.within_inames]

        return access_maps

    def _get_access_map_for_var(self, insn_id, access_dir, var_name):
        assert access_dir in ["w", "any"]

        insn = self.kernel.id_to_insn[insn_id]
        # Access range checks only apply to assignment-style instructions. For
        # non-assignments, we rely on read/write dependency information.
        from loopy.kernel.instruction import MultiAssignmentBase
        if not isinstance(insn, MultiAssignmentBase):
            if access_dir == "any":
                return var_name in insn.dependency_names()
            else:
                return var_name in insn.write_dependency_names()

        return self._get_access_maps(insn_id, access_dir)[var_name]

    def do_accesses_result_in_races(self, insn1, insn1_dir, insn2, insn2_dir,
                                    var_name):
        """Determine whether the access maps to *var_name* in the two given
        instructions result in write races owing to concurrent iname tags. This
        determination is made 'conservatively', i.e. if precise information is
        unavailable (for ex. if one of the instructions accesses *var_name* via
        indirection), it is concluded that the ranges overlap.

        :arg insn1_dir: either ``"w"`` or ``"any"``, to indicate which
            type of access is desired--writing or any
        :arg insn2_dir: either ``"w"`` or ``"any"``
        :returns: a :class:`bool`
        """

        insn1_amap = self._get_access_map_for_var(insn1, insn1_dir, var_name)
        insn2_amap = self._get_access_map_for_var(insn2, insn2_dir, var_name)

        if (insn1_amap is AccessMapDescriptor.DOES_NOT_ACCESS
                or insn2_amap is AccessMapDescriptor.DOES_NOT_ACCESS):
            return False
        if (insn1_amap is AccessMapDescriptor.NON_AFFINE_ACCESS
                or insn2_amap is AccessMapDescriptor.NON_AFFINE_ACCESS):
            return True

        return _check_for_access_races(insn1_amap, self.kernel.id_to_insn[insn1],
                                       insn2_amap, self.kernel.id_to_insn[insn2],
                                       self.kernel, self.callables_table,
                                       (self.kernel
                                        .get_var_descriptor(var_name)
                                        .address_space))

# }}}


LoopNestTree: TypeAlias = Tree[InameStrSet]
LoopTree: TypeAlias = Tree[InameStr]


class V2SchedulerNotImplementedError(LoopyError):
    pass


def separate_loop_nest(
            tree: LoopNestTree,
            loop_nests: Collection[InameStrSet],
            inames_to_separate: InameStrSet
        ) -> tuple[LoopNestTree, InameStrSet, InameStrSet | None]:
    """
    Returns a copy of *tree* that has *inames_to_separate* occur in
    nodes that are not shared with other inames.
    Returns a version of the loop nest tree *tree* so that every node in the tree is
    either a subset of *outermost_inames* or has an empty intersection with
    *outermost_inames*.

    This routine modifies at most one node of the tree.
    All its ancestors must satisfy `ancestor <= outermost_inames`.
    For the first node not  satisfying this relationship,
    if `node & outermost_inames` is empty, no modification is made.
    Otherwise, if ``node & outermost_inames < node``, that node is split
    so as to separate *outermost_inames* in their own node.

    :arg loop_nests: A collection of nodes in *tree* that cover
        *inames_to_separate*.

    :returns: a :class:`tuple` ``(new_tree, outer_loop_nest, inner_loop_nest)``,
        where outer_loop_nest is the identifier for the new outer and inner
        loop nests so that *inames_to_separate* is a valid nesting.

    .. note::

        We could compute *loop_nests* within this routine's implementation, but
        computing would be expensive and hence we ask the caller for this info.

    Example::
       *tree*: frozenset()
               └── frozenset({'j', 'i'})
                   └── frozenset({'k', 'l'})

       *inames_to_separate*: frozenset({'k', 'i', 'j'})
       *loop_nests*: {frozenset({'j', 'i'}), frozenset({'k', 'l'})}

       Returns:

       *new_tree*: frozenset()
                   └── frozenset({'j', 'i'})
                       └── frozenset({'k'})
                           └── frozenset({'l'})

       *outer_loop_nest*: frozenset({'k'})
       *inner_loop_nest*: frozenset({'l'})
    """
    assert all(isinstance(loop_nest, frozenset) for loop_nest in loop_nests)

    # annotation to avoid https://github.com/python/mypy/issues/17693
    emptyset: InameStrSet = frozenset()

    assert inames_to_separate <= reduce(frozenset.union, loop_nests, emptyset)

    # {{{ sanity check to ensure the loop nest *inames_to_separate* is possible

    loop_nests = sorted(loop_nests, key=lambda nest: tree.depth(nest))

    for outer, inner in itertools.pairwise(loop_nests):
        if outer != tree.parent(inner):
            raise LoopyError(f"Cannot schedule loop nest {inames_to_separate} "
                             f" in the nesting tree:\n{tree}")

    assert tree.depth(loop_nests[0]) == 0

    # }}}

    innermost_node = loop_nests[-1]
    # separate variable to avoid https://github.com/python/mypy/issues/17694
    outerer_loops = reduce(frozenset.union, loop_nests[:-1], emptyset)
    new_outer_node = inames_to_separate - outerer_loops
    new_inner_node = innermost_node - inames_to_separate

    if new_outer_node == innermost_node:
        # such a loop nesting already exists => do nothing
        return tree, new_outer_node, None

    # add the outer loop to our loop nest tree
    tree = tree.add_node(new_outer_node,
                         parent=not_none(tree.parent(innermost_node)))

    # rename the old loop to the inner loop
    tree = tree.replace_node(innermost_node,
                            new_node=new_inner_node)

    # set the parent of inner loop to be the outer loop
    tree = tree.move_node(new_inner_node, new_parent=new_outer_node)

    return tree, new_outer_node, new_inner_node


def _add_inner_loops(tree, outer_loop_nest, inner_loop_nest):
    """
    Returns a copy of *tree* that nests *inner_loop_nest* inside *outer_loop_nest*.
    """
    # add the outer loop to our loop nest tree
    return tree.add_node(inner_loop_nest, parent=outer_loop_nest)


def _order_loop_nests(
            loop_nest_tree: LoopNestTree,
            strict_priorities: frozenset[tuple[InameStr, ...]],
            relaxed_priorities: frozenset[tuple[InameStr, ...]],
            iname_to_tree_node_id: Mapping[InameStr, InameStrSet],
          ) -> LoopTree:
    """
    Returns a loop nest where all nodes in the tree are instances of
    :class:`str` denoting inames. Unlike *loop_nest_tree* which corresponds to
    multiple loop nesting, this routine returns a unique loop nest that is
    obtained after constraining *loop_nest_tree* with the constraints enforced
    by *priorities*.

    :arg strict_priorities: Expresses strict nesting constraints using the same
        data structure as :attr:`loopy.LoopKernel.loop_priority`.
        These priorities are imposed strictly i.e. if these conditions cannot be met a
        :class:`loopy.diagnostic.LoopyError` is raised.

    :arg relaxed_priorities: Expresses strict nesting constraints using the same
        data structure as :attr:`loopy.LoopKernel.loop_priority`.
        These nesting constraints are treated as optional.

    :arg iname_to_tree_node_id: A mapping from iname to the loop nesting its a
        part of.
    """
    from warnings import warn

    from pytools.graph import compute_topological_order as toposort

    loop_nests = set(iname_to_tree_node_id.values())

    # nesting_constraints: A mapping from the loop nest level to the nesting
    # constraints applicable to it.
    # Each nesting constraint is represented as a DAG. In the DAG, if there
    # exists an edge from from iname 'i' -> iname 'j' => 'j' should be nested
    # inside 'i'.
    iname_to_nesting_constraints: dict[InameStrSet, dict[InameStr, InameStrSet]] = {
        loop_nest: {iname: frozenset() for iname in loop_nest}
        for loop_nest in loop_nests}

    # The plan here is populate DAGs in *nesting_constraints* and then perform a
    # toposort for each loop nest.

    def _update_nesting_constraints(
                priorities: frozenset[tuple[InameStr, ...]],
                cannot_satisfy_callback: Callable[[str], None]
            ) -> None:
        """
        Records *priorities* in *nesting_constraints* and calls
        *cannot_satisfy_callback* with an appropriate error message if the
        priorities cannot be met.
        """
        for priority in priorities:
            for outer_iname, inner_iname in itertools.pairwise(priority):
                if inner_iname not in iname_to_tree_node_id:
                    cannot_satisfy_callback(f"Cannot enforce the constraint:"
                                            f" {inner_iname} to be nested within"
                                            f" {outer_iname}, as {inner_iname}"
                                            f" is either a parallel loop or"
                                            f" not an iname.")
                    continue

                if outer_iname not in iname_to_tree_node_id:
                    cannot_satisfy_callback(f"Cannot enforce the constraint:"
                                            f" {inner_iname} to be nested within"
                                            f" {outer_iname}, as {outer_iname}"
                                            f" is either a parallel loop or"
                                            f" not an iname.")
                    continue

                inner_iname_nest = iname_to_tree_node_id[inner_iname]
                outer_iname_nest = iname_to_tree_node_id[outer_iname]

                if inner_iname_nest == outer_iname_nest:
                    iname_to_nesting_constraints[
                        inner_iname_nest][outer_iname] |= {inner_iname}
                else:
                    ancestors_of_inner_iname = (loop_nest_tree
                                                .ancestors(inner_iname_nest))
                    ancestors_of_outer_iname = (loop_nest_tree
                                                .ancestors(outer_iname_nest))
                    if any(outer_iname in ancestor
                           for ancestor in ancestors_of_inner_iname):
                        # nesting constraint already satisfied => do nothing
                        pass
                    elif any(inner_iname in ancestor
                             for ancestor in ancestors_of_outer_iname):
                        cannot_satisfy_callback("Cannot satisfy constraint that"
                                                f" iname '{inner_iname}' must be"
                                                f" nested within '{outer_iname}''.")
                    else:
                        # inner iname and outer iname are indirect family members
                        # => must be realized via dependencies in the linearization
                        # phase, not implemented in v2-scheduler yet.
                        raise V2SchedulerNotImplementedError("cannot"
                                " schedule kernels with priority dependencies"
                                " between sibling loop nests")

    def _raise_loopy_err(x):
        raise LoopyError(x)

    # record strict priorities
    _update_nesting_constraints(strict_priorities, _raise_loopy_err)
    # record relaxed priorities
    _update_nesting_constraints(relaxed_priorities, warn)

    # ordered_loop_nests: A mapping from the unordered loop nests to their
    # ordered counterparts. For example. If we had only one loop nest
    # `frozenset({"i", "j", "k"})`, and the prioirities said added the
    # constraint that "i" must be nested within "k", then `ordered_loop_nests`
    # would be: `{frozenset({"i", "j", "k"}): ["j", "k", "i"]}` i.e. the loop
    # nests would now have an order.
    ordered_loop_nests = {
        unordered_nest: toposort(flow, key=lambda x: x)
        for unordered_nest, flow in iname_to_nesting_constraints.items()}

    # {{{ combine 'loop_nest_tree' along with 'ordered_loop_nest_tree'

    assert loop_nest_tree.root == frozenset()

    new_tree = Tree.from_root("")

    old_to_new_parent = {}

    old_to_new_parent[loop_nest_tree.root] = ""

    # traversing 'tree' in an BFS fashion to create 'new_tree'
    queue = list(loop_nest_tree.children(loop_nest_tree.root))

    while queue:
        current_nest = queue.pop(0)

        ordered_nest = ordered_loop_nests[current_nest]
        new_tree = new_tree.add_node(ordered_nest[0],
                                     parent=old_to_new_parent[not_none(loop_nest_tree
                                                              .parent(current_nest))])
        for new_parent, new_child in itertools.pairwise(ordered_nest):
            new_tree = new_tree.add_node(node=new_child, parent=new_parent)

        old_to_new_parent[current_nest] = ordered_nest[-1]

        queue.extend(loop_nest_tree.children(current_nest))

    # }}}

    return new_tree


@memoize_on_first_arg
def _get_parallel_inames(kernel: LoopKernel) -> Set[str]:
    from loopy.kernel.data import ConcurrentTag, IlpBaseTag, VectorizeTag

    concurrent_inames = {iname for iname in kernel.all_inames()
                         if kernel.iname_tags_of_type(iname, ConcurrentTag)}
    ilp_inames = {iname for iname in kernel.all_inames()
                  if kernel.iname_tags_of_type(iname, IlpBaseTag)}
    vec_inames = {iname for iname in kernel.all_inames()
                  if kernel.iname_tags_of_type(iname, VectorizeTag)}
    return (concurrent_inames - ilp_inames - vec_inames)


def get_partial_loop_nest_tree(kernel: LoopKernel) -> LoopNestTree:
    """
    Returns a tree representing the *kernel*'s loop nests.

    Each node of the returned tree has a :class:`frozenset` of inames.
    All the inames in the identifier of a parent node of a loop nest in the
    tree must be nested outside all the iname in identifier of the loop nest.

    .. note::

        This routine only takes into account the nesting dependency
        constraints of :attr:`loopy.InstructionBase.within_inames` of all the
        *kernel*'s instructions and the iname tags. This routine does *NOT*
        include the nesting constraints imposed by the dependencies between the
        instructions and the dependencies imposed by the kernel's domain tree.
    """
    from loopy.kernel.data import IlpBaseTag

    # figuring the possible loop nestings minus the concurrent_inames as they
    # are never realized as actual loops
    insn_iname_sets = {
        insn.within_inames - _get_parallel_inames(kernel)
        for insn in kernel.instructions}

    root: InameStrSet = frozenset()
    tree = Tree[InameStrSet].from_root(root)

    # mapping from iname to the innermost loop nest they are part of in *tree*.
    iname_to_tree_node_id: dict[InameStr, InameStrSet] = {}

    # if there were any loop with no inames, those have been already account
    # for as the root.
    insn_iname_sets = insn_iname_sets - {root}

    for iname_set in insn_iname_sets:
        not_seen_inames = frozenset(iname for iname in iname_set
                                    if iname not in iname_to_tree_node_id)
        seen_inames = iname_set - not_seen_inames

        all_nests = {iname_to_tree_node_id[iname] for iname in seen_inames}

        tree, outer_loop, inner_loop = separate_loop_nest(tree,
                                                           (all_nests
                                                            | {frozenset()}),
                                                           seen_inames)
        if not_seen_inames:
            # make '_not_seen_inames' nest inside the seen ones.
            # example: if there is already a loop nesting "i,j,k"
            # and the current iname chain is "i,j,l". Only way this is possible
            # is if "l" is nested within "i,j"-loops.
            tree = _add_inner_loops(tree, outer_loop, not_seen_inames)

        # {{{ update iname to node id

        for iname in outer_loop:
            iname_to_tree_node_id[iname] = outer_loop

        if inner_loop is not None:
            for iname in inner_loop:
                iname_to_tree_node_id[iname] = inner_loop

        for iname in not_seen_inames:
            iname_to_tree_node_id[iname] = not_seen_inames

        # }}}

    # {{{ make ILP tagged inames innermost

    ilp_inames = {iname for iname in kernel.all_inames()
                  if kernel.iname_tags_of_type(iname, IlpBaseTag)}

    for iname_set in insn_iname_sets:
        for ilp_iname in (ilp_inames & insn_iname_sets):
            # pull out other loops so that ilp_iname is the innermost
            all_nests = {iname_to_tree_node_id[iname] for iname in seen_inames}
            tree, outer_loop, inner_loop = separate_loop_nest(tree,
                                                               (all_nests
                                                                | {frozenset()}),
                                                               (iname_set
                                                                - {ilp_iname}))

            for iname in outer_loop:
                iname_to_tree_node_id[iname] = outer_loop

            if inner_loop is not None:
                for iname in inner_loop:
                    iname_to_tree_node_id[iname] = inner_loop

    # }}}

    return tree


def _get_iname_to_tree_node_id_from_partial_loop_nest_tree(
            tree: LoopNestTree,
        ) -> Mapping[InameStr, InameStrSet]:
    """
    Returns the mapping from the iname to the *tree*'s node that it was a part
    of.

    :arg tree: A partial loop nest tree.
    """
    iname_to_tree_node_id = {}
    for node in tree.nodes():
        assert isinstance(node, frozenset)
        for iname in node:
            iname_to_tree_node_id[iname] = node

    return constantdict(iname_to_tree_node_id)


def get_loop_tree(kernel: LoopKernel) -> LoopTree:
    """
    Returns a tree representing the loop nesting for *kernel*. A parent node in
    the tree is always nested outside all its children.

    .. note::

        Multiple loop nestings might exist for *kernel*, but this routine returns
        one valid loop nesting.
    """
    from islpy import dim_type

    tree = get_partial_loop_nest_tree(kernel)
    iname_to_tree_node_id = (
        _get_iname_to_tree_node_id_from_partial_loop_nest_tree(tree))

    strict_loop_priorities: frozenset[tuple[InameStr, ...]] = frozenset()

    # {{{ impose constraints by the domain tree

    loop_inames = fset_union(
            insn.within_inames for insn in kernel.instructions)
    loop_inames = loop_inames - _get_parallel_inames(kernel)

    for dom in kernel.domains:
        for outer_iname in set(dom.get_var_names(dim_type.param)):
            if outer_iname not in loop_inames:
                continue

            for inner_iname in dom.get_var_names(dim_type.set):
                if inner_iname not in loop_inames:
                    continue

                # either outer_iname and inner_iname should belong to the same
                # loop nest level or outer should be strictly outside inner
                # iname
                inner_iname_nest = iname_to_tree_node_id[inner_iname]
                outer_iname_nest = iname_to_tree_node_id[outer_iname]

                if inner_iname_nest == outer_iname_nest:
                    strict_loop_priorities |= {(outer_iname, inner_iname)}
                else:
                    ancestors_of_inner_iname = tree.ancestors(inner_iname_nest)
                    if outer_iname_nest not in ancestors_of_inner_iname:
                        raise LoopyError(f"Loop '{outer_iname}' cannot be nested"
                                         f" outside '{inner_iname}'.")

    # }}}

    return _order_loop_nests(tree,
                             strict_loop_priorities,
                             kernel.loop_priority,
                             iname_to_tree_node_id)

# vim: fdm=marker

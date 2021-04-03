__copyright__ = "Copyright (C) 2016 Matt Wala"

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

from loopy.kernel.data import AddressSpace
from pytools import memoize_method
import islpy as isl
import enum


# {{{ block boundary finder

def get_block_boundaries(schedule):
    """
    Return a dictionary mapping indices of
    :class:`loopy.schedule.BlockBeginItem`s to
    :class:`loopy.schedule.BlockEndItem`s and vice versa.
    """
    from loopy.schedule import (BeginBlockItem, EndBlockItem)
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

def temporaries_read_in_subkernel(kernel, subkernel):
    from loopy.kernel.tools import get_subkernel_to_insn_id_map
    insn_ids = get_subkernel_to_insn_id_map(kernel)[subkernel]
    inames = frozenset().union(*(kernel.insn_inames(insn_id)
                                 for insn_id in insn_ids))
    params = frozenset().union(*(
        kernel.domains[dom_idx].get_var_names(isl.dim_type.param)
        for dom_idx in kernel.get_leaf_domain_indices(inames)))

    return (frozenset(tv
                      for insn_id in insn_ids
                      for tv in kernel.id_to_insn[insn_id].read_dependency_names()
                      if tv in kernel.temporary_variables)
            | (params & frozenset(kernel.temporary_variables)))


def temporaries_written_in_subkernel(kernel, subkernel):
    from loopy.kernel.tools import get_subkernel_to_insn_id_map
    insn_ids = get_subkernel_to_insn_id_map(kernel)[subkernel]
    return frozenset(tv
            for insn_id in insn_ids
            for tv in kernel.id_to_insn[insn_id].write_dependency_names()
            if tv in kernel.temporary_variables)


def args_read_in_subkernel(kernel, subkernel):
    from loopy.kernel.tools import get_subkernel_to_insn_id_map
    insn_ids = get_subkernel_to_insn_id_map(kernel)[subkernel]
    inames = frozenset().union(*(kernel.insn_inames(insn_id)
                                 for insn_id in insn_ids))
    params = frozenset().union(*(
        kernel.domains[dom_idx].get_var_names(isl.dim_type.param)
        for dom_idx in kernel.get_leaf_domain_indices(inames)))

    return (frozenset(arg
                      for insn_id in insn_ids
                      for arg in kernel.id_to_insn[insn_id].read_dependency_names()
                      if arg in kernel.arg_dict)
            | (params & frozenset(kernel.arg_dict)))


def args_written_in_subkernel(kernel, subkernel):
    from loopy.kernel.tools import get_subkernel_to_insn_id_map
    insn_ids = get_subkernel_to_insn_id_map(kernel)[subkernel]
    return frozenset(arg
                     for insn_id in insn_ids
                     for arg in kernel.id_to_insn[insn_id].write_dependency_names()
                     if arg in kernel.arg_dict)


def get_callkernel_dependencies(kernel, subkernel):
    """
    Returns variable names referenced by :class:`~loopy.schedule.CallKernel`
    named *subkernel*.
    """
    return (temporaries_read_in_subkernel(kernel, subkernel)
            | temporaries_written_in_subkernel(kernel, subkernel)
            | args_read_in_subkernel(kernel, subkernel)
            | args_written_in_subkernel(kernel, subkernel))

# }}}


# {{{ add extra args to schedule

def add_extra_args_to_schedule(kernel):
    """
    Fill the `extra_args` fields in all the :class:`loopy.schedule.CallKernel`
    instructions in the schedule with global temporaries.
    """
    new_schedule = []
    from loopy.schedule import CallKernel

    for sched_item in kernel.linearization:
        if isinstance(sched_item, CallKernel):
            subkernel = sched_item.kernel_name

            used_temporaries = (
                    temporaries_read_in_subkernel(kernel, subkernel)
                    | temporaries_written_in_subkernel(kernel, subkernel))

            more_args = {tv
                    for tv in used_temporaries
                    if
                    kernel.temporary_variables[tv].address_space
                    == AddressSpace.GLOBAL
                    and
                    kernel.temporary_variables[tv].initializer is None
                    and
                    tv not in sched_item.extra_args}

            new_schedule.append(sched_item.copy(
                    extra_args=sched_item.extra_args + sorted(more_args)))
        else:
            new_schedule.append(sched_item)

    return kernel.copy(linearization=new_schedule)

# }}}


# {{{ get_return_from_kernel_mapping

def get_return_from_kernel_mapping(kernel):
    """
    Returns a mapping from schedule index of every schedule item (S) in
    *kernel* to the schedule index of :class:`loopy.schedule.ReturnFromKernel`
    of the active sub-kernel at 'S'.
    """
    from loopy.kernel import LoopKernel
    from loopy.schedule import (RunInstruction, EnterLoop, LeaveLoop,
                                CallKernel, ReturnFromKernel, Barrier)
    assert isinstance(kernel, LoopKernel)
    assert isinstance(kernel.linearization, list)
    return_from_kernel_idxs = {}
    current_return_from_kernel = None
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

def _check_for_access_races(map_a, insn_a, map_b, insn_b, knl, callables_table):
    """
    Returns *True* if the execution instances of *insn_a* and *insn_b*, accessing
    the same variable via access maps *map_a* and *map_b*, result in an access race.

    .. note::

        The accesses ``map_a``, ``map_b`` lead to write races iff there exists 2
        *unequal* global ids that access the same address.
    """
    import pymbolic.primitives as p
    from loopy.symbolic import isl_set_from_expr
    from loopy.kernel.data import (filter_iname_tags_by_type,
                                   HardwareConcurrentTag)

    gsize, lsize = knl.get_grid_size_upper_bounds(callables_table,
                                                  return_dict=True)

    # {{{ Step 1: Preprocess the maps

    # Step 1.1: Project out inames which are also map's dims, but does not form the
    #           insn's within_inames
    # Step 1.2: Project out sequential inames in the access maps
    # Step 1.3: Rename the dims with their iname tags i.e. (g.i or l.i)
    # Step 1.4: Name the ith output dims as _lp_dim{i}

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
    unequal_global_id_set = isl.Set.empty(set_a.get_space())

    for i_l in lsize:
        lid_a = p.Variable(f"l.{i_l}.A")
        lid_b = p.Variable(f"l.{i_l}.B")
        unequal_global_id_set |= (isl_set_from_expr(space,
                                                    p.Comparison(lid_a, "!=", lid_b))
                                  )

    for i_g in gsize:
        gid_a = p.Variable(f"g.{i_g}.A")
        gid_b = p.Variable(f"g.{i_g}.B")
        unequal_global_id_set |= (isl_set_from_expr(space,
                                                    p.Comparison(gid_a, "!=", gid_b))
                                  )

    # }}}

    return not (set_a & set_b & unequal_global_id_set).is_empty()


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

    @property
    @memoize_method
    def vars(self):
        return (self.kernel.get_written_variables()
                | self.kernel.get_read_variables())

    @memoize_method
    def _get_access_maps(self, insn_id, access_dir):
        from loopy.symbolic import BatchedAccessMapMapper
        from collections import defaultdict

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
                                       self.kernel, self.callables_table)

# }}}

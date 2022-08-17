"""
.. currentmodule:: loopy

.. autofunction:: distribute_loops

.. autoclass:: IllegalLoopDistributionError
"""

__copyright__ = """
Copyright (C) 2022 Kaushik Kulkarni
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

import islpy as isl

from dataclasses import dataclass
from functools import reduce
import loopy.match as match
from loopy.diagnostic import LoopyError
from itertools import combinations
from more_itertools import split_when
from loopy.transform.iname import remove_any_newly_unused_inames


class IllegalLoopDistributionError(LoopyError):
    """
    Raised if the call to :func:`distribute_loops` wouldn't
    preserve dependencies in the transformed kernel.
    """


def _get_rev_dep_dag(kernel, insn_ids):
    rev_dep_graph = {insn_id: set() for insn_id in insn_ids}

    for insn_id in insn_ids:
        for dep in (kernel.id_to_insn[insn_id].depends_on
                    & insn_ids):
            rev_dep_graph[dep].add(insn_id)

    return rev_dep_graph


def _union_amaps(isl_maps):
    def union_map1_map2(map1, map2):
        if map1.space != map2.space:
            map1, map2 = isl.align_two(map1, map2)
        return map1 | map2

    return reduce(union_map1_map2,
                  isl_maps[1:],
                  isl_maps[0])


def _is_loop_distribution_sound(kernel,
                                iname,
                                insns_before,
                                insns_after,
                                outer_inames):
    from loopy.kernel.tools import get_insn_access_map
    id_to_insn = kernel.id_to_insn

    insns_before = frozenset({insn_id
                              for insn_id in insns_before
                              if iname in id_to_insn[insn_id].dependency_names()})
    insns_after = frozenset({insn_id
                              for insn_id in insns_after
                              if iname in id_to_insn[insn_id].dependency_names()})

    vars_written_in_insns_before = reduce(frozenset.union,
                                          (id_to_insn[insn_id].assignee_var_names()
                                           for insn_id in insns_before),
                                          frozenset())
    vars_written_in_insns_after = reduce(frozenset.union,
                                         (id_to_insn[insn_id].assignee_var_names()
                                          for insn_id in insns_after),
                                         frozenset())
    vars_read_in_insns_before = reduce(frozenset.union,
                                       (id_to_insn[insn_id].read_dependency_names()
                                        for insn_id in insns_before),
                                       frozenset())
    vars_read_in_insns_after = reduce(frozenset.union,
                                      (id_to_insn[insn_id].read_dependency_names()
                                       for insn_id in insns_after),
                                      frozenset())

    # dep_inducing_vars: dependency inducing variables
    dep_inducing_vars = (
        # WAW
        (vars_written_in_insns_before & vars_written_in_insns_after)
        # RAW
        | (vars_written_in_insns_before & vars_read_in_insns_after)
        # WAR
        | (vars_read_in_insns_before & vars_written_in_insns_after))

    # {{{ process_amap

    def process_amap(amap):
        for i, outer_iname in enumerate(sorted(outer_inames)):
            dt, pos = amap.get_var_dict()[outer_iname]
            assert dt == isl.dim_type.in_
            amap = amap.move_dims(isl.dim_type.param, i,
                                  dt, pos, 1)

        amap = amap.project_out_except([iname], [isl.dim_type.in_])
        assert amap.dim(isl.dim_type.in_) == 1
        assert amap.get_var_names(isl.dim_type.in_) == [iname]
        return amap

    # }}}

    for var in dep_inducing_vars:
        # There is some issue regarding the alignment of the polyhedra spaces
        # here.
        amaps_pred = [
            process_amap(get_insn_access_map(kernel, insn_id, var))
            for insn_id in insns_before
            if var in id_to_insn[insn_id].dependency_names()]

        amaps_succ = [
            process_amap(get_insn_access_map(kernel, insn_id, var))
            for insn_id in insns_after
            if var in id_to_insn[insn_id].dependency_names()]

        amap_pred = _union_amaps(amaps_pred)
        amap_succ = _union_amaps(amaps_succ)

        assert amap_pred.dim(isl.dim_type.in_) == 1
        assert amap_succ.dim(isl.dim_type.in_) == 1
        assert amap_pred.dim(isl.dim_type.out) == amap_succ.dim(isl.dim_type.out)

        amap_pred = amap_pred.set_dim_name(isl.dim_type.in_, 0, f"{iname}_pred")
        amap_succ = amap_succ.set_dim_name(isl.dim_type.in_, 0, f"{iname}_succ")

        for i in range(amap_pred.dim(isl.dim_type.out)):
            amap_pred = amap_pred.set_dim_name(isl.dim_type.out, i, f"out_{i}")
            amap_succ = amap_succ.set_dim_name(isl.dim_type.out, i, f"out_{i}")

        amap_pred, amap_succ = isl.align_two(amap_pred, amap_succ)
        ipred_gt_isucc = (isl.BasicMap
                        .universe(amap_pred.space)
                        .add_constraint(isl.Constraint
                                        .ineq_from_names(amap_pred.space,
                                                         {f"{iname}_pred": 1,
                                                          f"{iname}_succ": -1,
                                                          1: -1,
                                                          })))

        if not (amap_pred & amap_succ & ipred_gt_isucc).range().is_empty():
            return False

    return True


# {{{ _get_loop_nest_aware_statement_toposort

@dataclass(frozen=True)
class ScheduleItem:
    pass


@dataclass(frozen=True)
class BeginLoop(ScheduleItem):
    iname: str


@dataclass(frozen=True)
class EndLoop(ScheduleItem):
    iname: str


@dataclass(frozen=True)
class Statement(ScheduleItem):
    insn_id: str


def _get_loop_nest_aware_statement_toposort(kernel, insn_ids,
                                            insns_to_distribute,
                                            outer_inames):
    from loopy.schedule.tools import get_loop_nest_tree
    from pytools.graph import compute_topological_order_with_dynamic_key

    id_to_insn = kernel.id_to_insn
    all_inames = reduce(frozenset.union,
                        [id_to_insn[insn_id].within_inames
                         for insn_id in insn_ids],
                        frozenset()
                        ) - outer_inames

    dag = {**{BeginLoop(iname): set()
              for iname in all_inames},
           **{EndLoop(iname): set()
              for iname in all_inames},
           **{Statement(insn): set()
              for insn in insn_ids}
           }

    for insn_id in insn_ids:
        insn = id_to_insn[insn_id]
        for dep in (insn.depends_on & insn_ids):

            inames_dep = id_to_insn[dep].within_inames
            inames_insn = id_to_insn[insn_id].within_inames

            # {{{ register deps on loop entry/leave because of insn. deps

            if inames_dep < inames_insn:
                for iname in inames_insn - inames_dep:
                    dag[Statement(dep)].add(BeginLoop(iname))
            elif inames_insn < inames_dep:
                for iname in inames_dep - inames_insn:
                    dag[EndLoop(iname)].add(Statement(insn_id))
            elif inames_dep != inames_insn:
                for iname_dep in inames_dep - (inames_dep & inames_insn):
                    for iname_insn in inames_insn - (inames_dep & inames_insn):
                        dag[EndLoop(iname_dep)].add(BeginLoop(iname_insn))
            else:
                dag[Statement(dep)].add(Statement(insn_id))

            # }}}

        for iname in insn.within_inames - outer_inames:
            dag[BeginLoop(iname)].add(Statement(insn_id))
            dag[Statement(insn_id)].add(EndLoop(iname))

    # TODO: [perf] it would be better to only look at the
    # statements inner to 'outer_inames'.
    loop_nest_tree = get_loop_nest_tree(kernel)

    def trigger_key_update(state):
        recent_instructions = (x.insn_id
                               for x in reversed(state.scheduled_nodes)
                               if isinstance(x, Statement))

        try:
            last_insn = next(recent_instructions)
        except StopIteration:
            # no instructions scheduled yet.
            return False
        else:
            try:
                second_to_last_insn = next(recent_instructions)
            except StopIteration:
                return last_insn in insns_to_distribute
            else:
                return ((second_to_last_insn in insns_to_distribute)
                        ^ (last_insn in insns_to_distribute))

    def key(x, prev_node_in_distributed_insns):
        if isinstance(x, Statement):
            iname = max(id_to_insn[x.insn_id].within_inames,
                        key=loop_nest_tree.depth,
                        default="")
            loop_nest = tuple(
                sorted(loop_nest_tree.ancestors(iname),
                       key=loop_nest_tree.depth)) + (iname,)

            return (loop_nest,
                    ((x.insn_id in insns_to_distribute)
                     ^ prev_node_in_distributed_insns),
                    x.insn_id)
        elif isinstance(x, (BeginLoop, EndLoop)):
            loop_nest = tuple(
                sorted(loop_nest_tree.ancestors(x.iname),
                       key=loop_nest_tree.depth)) + (x.iname,)
            return (loop_nest,)
        else:
            raise NotImplementedError(x)

    def get_key(state):
        if not any(isinstance(sched_node, Statement)
                   for sched_node in state.scheduled_nodes):
            return lambda x: key(x, False)
        else:
            last_insn = next(x.insn_id
                             for x in reversed(state.scheduled_nodes)
                             if isinstance(x, Statement))
            return lambda x: key(x,
                                 last_insn in insns_to_distribute)

    toposorted_dag = compute_topological_order_with_dynamic_key(
        dag, trigger_key_update=trigger_key_update, get_key=get_key)

    return [x.insn_id for x in toposorted_dag if isinstance(x, Statement)]

# }}}


@remove_any_newly_unused_inames
def distribute_loops(kernel,
                     insn_match,
                     outer_inames,
                     check_soundness: bool = __debug__,
                     ):
    r"""
    Returns a copy of *kernel* with instructions in *insn_match* having all
    inames expect *outer_inames* duplicated. This is a dependency preserving
    transformation which might require duplicating inames in instructions not
    matching *insn_match*.

    :arg insn_match: A match expression as understood by
        :func:`loopy.match.parse_match`.
    :arg outer_loops: A set of inames under which the loop distribution is to
        be performed.
    :arg check_soundness: If *True*, raises an error if any duplication of the
       inames would violate dependencies. If *False*, skips these checks and
       could potentially return a kernel transformed with a
       dependency-violating transformation.

    .. note::

        - This assumes that the generated code would iterate over a loop in an
          ascending order of the value taken by the loop-induction variable.
        - This transformation is undefined for *kernel* that cannot be
          linearized, as they don't represent a computable expression.
    """
    from loopy.transform.iname import duplicate_inames

    within = match.parse_match(insn_match)
    id_to_insn = kernel.id_to_insn

    insns_to_distribute = {insn.id
                           for insn in kernel.instructions
                           if within(kernel, insn)}

    insns_in_outer_inames = frozenset(insn.id
                                      for insn in kernel.instructions
                                      if insn.within_inames >= outer_inames)

    if not (insns_to_distribute <= insns_in_outer_inames):
        raise LoopyError(f"Instructions {insns_to_distribute-insns_in_outer_inames} "
                         " to be distributed do not nest under outer_inames.")

    toposorted_insns = _get_loop_nest_aware_statement_toposort(kernel,
                                                               insns_in_outer_inames,
                                                               insns_to_distribute,
                                                               outer_inames)

    insn_partitions = tuple(split_when(
        toposorted_insns,
        lambda x, y: ((x in insns_to_distribute) ^ (y in insns_to_distribute))))

    # {{{ compute which inames must be duplicate to perform the distribution.

    partition_idx_to_inner_inames = [
        reduce(frozenset.union,
               (((id_to_insn[insn_id].within_inames
                  | id_to_insn[insn_id].reduction_inames())
                 - outer_inames)
                for insn_id in insn_partition),
               frozenset())
        for insn_partition in insn_partitions
    ]

    inner_iname_to_partition_indices = {}

    for ipartition, inner_inames in enumerate(partition_idx_to_inner_inames):
        for inner_iname in inner_inames:
            (inner_iname_to_partition_indices
             .setdefault(inner_iname, set())
             .add(ipartition))

    inames_to_duplicate = {
        inner_iname
        for inner_iname, ipartitions in inner_iname_to_partition_indices.items()
        if len(ipartitions) > 1
    }
    iname_to_duplicate_tags = {iname: kernel.inames[iname].tags
                               for iname in inames_to_duplicate}

    del partition_idx_to_inner_inames

    # }}}

    # {{{ check soundness

    if check_soundness:
        for inner_iname in inames_to_duplicate:
            ipartitions = sorted(inner_iname_to_partition_indices[inner_iname])

            for ipart_pred, ipart_succ in combinations(ipartitions, 2):
                if not _is_loop_distribution_sound(kernel,
                                                   inner_iname,
                                                   insn_partitions[ipart_pred],
                                                   insn_partitions[ipart_succ],
                                                   outer_inames):
                    raise IllegalLoopDistributionError(
                        f"Distributing the loops '{inames_to_duplicate}'"
                        " would violate memory dependencies between the instructions"
                        f" '{insn_partitions[ipart_pred]}' and "
                        f" '{insn_partitions[ipart_succ]}'.")

    # }}}

    for insn_partition in insn_partitions:
        within_partition = match.Or(tuple(match.Id(insn_id)
                                          for insn_id in insn_partition))
        inames_in_partition = reduce(frozenset.union,
                                     ((id_to_insn[insn_id].within_inames
                                       | id_to_insn[insn_id].reduction_inames())
                                      for insn_id in insn_partition),
                                     frozenset())
        inames_to_duplicate_in_partition = inames_to_duplicate & inames_in_partition
        if not inames_to_duplicate_in_partition:
            continue

        kernel = duplicate_inames(kernel,
                                  inames_to_duplicate_in_partition,
                                  within=within_partition,
                                  tags=iname_to_duplicate_tags)

    return kernel

# vim: fdm=marker

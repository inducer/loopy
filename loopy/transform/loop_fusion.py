__copyright__ = """
Copyright (C) 2021 Kaushik Kulkarni
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

from loopy.diagnostic import LoopyError
from loopy.kernel import LoopKernel
from typing import FrozenSet, Mapping, Tuple, Dict, Set
from functools import reduce
from dataclasses import dataclass

__doc__ = """
.. autofunction:: rename_inames_in_batch
.. autofunction:: get_kennedy_unweighted_fusion_candidates
"""


# {{{ Loop Depenendence graph class + builder


@dataclass(frozen=True, eq=True)
class LoopDependenceGraph:
    """
    .. attribute:: successors

        A mapping from iname (``i``) to the collection of inames that can be
        scheduled only after the loop corresponding to ``i`` has been exited.

    .. attribute:: predecessors

        A mapping from iname (``i``) to the collection of inames that must have
        been exited before entering ``i``.

    .. attribute:: is_infusible

        A mapping from the edges in the loop dependence graph to their
        fusibility crierion. An edge in this mapping is represented by a pair
        of inames``(iname_i, iname_j)`` such that the edge ``iname_i ->
        iname_j`` is present in the graph.

    .. note::

        Both :attr:`successors` and :attr:`predecessors` are maintained to
        reduce the complexity of graph primitive operations (like remove node,
        add edge, etc.).
    """
    successors: Mapping[str, FrozenSet[str]]
    predecessors: Mapping[str, FrozenSet[str]]
    is_infusible: Mapping[Tuple[str, str], bool]

    @classmethod
    def new(cls, successors, is_infusible):
        predecessors = {node: set()
                        for node in successors}
        for node, succs in successors.items():
            for succ in succs:
                predecessors[succ].add(node)

        predecessors = {node: frozenset(preds)
                        for node, preds in predecessors.items()}
        successors = {node: frozenset(succs)
                      for node, succs in successors.items()}

        return LoopDependenceGraph(successors, predecessors, is_infusible)

    def is_empty(self):
        """
        Returns *True* only if the loop dependence graph contains no nodes.
        """
        return (len(self.successors) == 0)

    def get_loops_with_no_predecessors(self):
        return {loop
                for loop, preds in self.predecessors.items()
                if len(preds) == 0}

    def remove_nodes(self, nodes_to_remove):
        """
        Returns a copy of *self* after removing *nodes_to_remove* in the graph.
        This routine adds necessary edges after removing *nodes_to_remove* to
        conserve the scheduling constraints present in the graph.
        """
        # {{{ Step 1. Remove the nodes

        new_successors = {node: succs
                          for node, succs in self.successors.items()
                          if node not in nodes_to_remove}
        new_predecessors = {node: preds
                            for node, preds in self.predecessors.items()
                            if node not in nodes_to_remove}

        new_is_infusible = {(from_, to): v
                            for (from_, to), v in self.is_infusible.items()
                            if (from_ not in nodes_to_remove
                                and to not in nodes_to_remove)}

        # }}}

        # {{{ Step 2. Propagate dependencies

        # For every Node 'R' to be removed and every pair (S, P) such that
        # 1. there exists an edge 'P' -> 'R' in the original graph, and,
        # 2. there exits an edge 'R' -> 'S' in the original graph.
        # add the edge 'P' -> 'S' in the new graph.

        for node_to_remove in nodes_to_remove:
            for succ in (self.successors[node_to_remove]
                         - nodes_to_remove):
                new_predecessors[succ] = (new_predecessors[succ]
                                          - frozenset([node_to_remove]))

            for pred in (self.predecessors[node_to_remove]
                         - nodes_to_remove):
                new_successors[pred] = (new_successors[pred]
                                        - frozenset([node_to_remove]))

        # }}}

        return LoopDependenceGraph(new_successors,
                                   new_predecessors,
                                   new_is_infusible)


@dataclass
class LoopDependenceGraphBuilder:
    _dag: Dict[str, Set[str]]
    _is_infusible: Mapping[Tuple[str, str], bool]

    @classmethod
    def new(cls, candidates):
        return LoopDependenceGraphBuilder({iname: set()
                                           for iname in candidates},
                                          {})

    def add_edge(self, from_: str, to: str, is_infusible: bool):
        self._dag[from_].add(to)
        self._is_infusible[(from_, to)] = (is_infusible
                                           or self._is_infusible.get((from_, to),
                                                                     False))

    def done(self):
        """
        Returns the built :class:`LoopDependenceGraph`.
        """
        return LoopDependenceGraph.new(self._dag, self._is_infusible)

# }}}


def _remove_irrelevant_insns_from_statement_dag(kernel,
                                                insn_to_predecessors,
                                                insn_to_successors,
                                                candidates):
    """
    Removes instructions from the statement DAG represented by
    *insn_to_predecessors*, *insn_to_successors* that are not nested in
    *candidates*.

    Returns a new statement DAG ``new_predecessors, new_successors`` , where
    edges are added between the remaining nodes of the statement DAG to
    preserve the dependencies in the original DAG.
    """
    # {{{ input validation

    assert set(insn_to_predecessors) == set(insn_to_successors)
    assert all(isinstance(val, frozenset) for val in insn_to_predecessors.values())
    assert all(isinstance(val, frozenset) for val in insn_to_successors.values())

    # }}}

    insns_to_remove = {insn
                       for insn in insn_to_successors
                       if len(kernel.id_to_insn[insn].within_inames
                              & candidates) == 0}

    new_predecessors = insn_to_predecessors.copy()
    new_successors = insn_to_successors.copy()
    infusible_edges_in_statement_dag = set()

    for insn_to_remove in insns_to_remove:
        for pred in new_predecessors[insn_to_remove]:
            new_successors[pred] = ((new_successors[pred]
                                     - frozenset([insn_to_remove]))
                                    | new_successors[insn_to_remove])

        for succ in new_successors[insn_to_remove]:
            new_predecessors[succ] = ((new_predecessors[succ]
                                       - frozenset([insn_to_remove]))
                                      | new_predecessors[insn_to_remove])

        for pred in new_predecessors[insn_to_remove]:
            for succ in new_successors[insn_to_remove]:
                # now mark the edge from pred -> succ infusible iff both 'pred' and
                # 'succ' are *not* in insns_to_remove
                if ((pred not in insns_to_remove) and (succ not in insns_to_remove)):
                    infusible_edges_in_statement_dag.add((pred, succ))

        del new_predecessors[insn_to_remove]
        del new_successors[insn_to_remove]

    return (new_predecessors,
            new_successors,
            infusible_edges_in_statement_dag)


def _compute_isinfusible_via_access_map(kernel,
                                        insn_pred, candidate_pred,
                                        insn_succ, candidate_succ,
                                        outer_inames,
                                        var):
    """
    Returns *True* if the inames *candidate_pred* and *candidate_succ* are fused then
    that might lead to a loop carried dependency for *var*.
    """
    import islpy as isl
    from loopy.kernel.tools import get_insn_access_map
    import pymbolic.primitives as prim
    from loopy.symbolic import isl_set_from_expr
    from loopy.diagnostic import UnableToDetermineAccessRangeError

    inner_inames_pred = (kernel.insn_inames(insn_pred)
                      - (frozenset([candidate_pred])
                         | outer_inames))

    inner_inames_succ = (kernel.insn_inames(insn_succ)
                      - (frozenset([candidate_succ])
                         | outer_inames))

    try:
        amap_pred = get_insn_access_map(kernel, insn_pred, var, inner_inames_pred)
        amap_succ = get_insn_access_map(kernel, insn_succ, var, inner_inames_succ)
    except UnableToDetermineAccessRangeError:
        # either predecessors or successors has a non-affine access i.e.
        # fallback to the safer option => infusible
        return True

    # since both ranges denote the same variable they must be subscripted with
    # the same number of indices.
    assert amap_pred.dim(isl.dim_type.out) == amap_succ.dim(isl.dim_type.out)

    ndim = amap_pred.dim(isl.dim_type.out)

    # {{{ set the out dim names as `amap_a_dim0`, `amap_a_dim1`, ...

    for idim in range(ndim):
        amap_pred = amap_pred.set_dim_name(isl.dim_type.out,
                                     idim,
                                     f"_lpy_amap_a_dim{idim}")
        amap_succ = amap_succ.set_dim_name(isl.dim_type.out,
                                     idim,
                                     f"_lpy_amap_b_dim{idim}")

    # }}}

    # {{{ amap_pred -> set_pred, amap_succ -> set_succ

    amap_pred = amap_pred.move_dims(isl.dim_type.in_,
                                    amap_pred.dim(isl.dim_type.in_),
                                    isl.dim_type.out,
                                    0, amap_pred.dim(isl.dim_type.out))

    amap_succ = amap_succ.move_dims(isl.dim_type.in_,
                                    amap_succ.dim(isl.dim_type.in_),
                                    isl.dim_type.out,
                                    0, amap_succ.dim(isl.dim_type.out))

    set_pred, set_succ = amap_pred.domain(), amap_succ.domain()
    set_pred, set_succ = isl.align_two(set_pred, set_succ)

    # }}}

    # {{{ build the bset, both accesses access the same element

    accesses_same_index_set = isl.BasicSet.universe(set_pred.space)
    for idim in range(ndim):
        cnstrnt = isl.Constraint.eq_from_names(set_pred.space,
                                               {f"_lpy_amap_a_dim{idim}": 1,
                                                f"_lpy_amap_b_dim{idim}": -1})
        accesses_same_index_set = accesses_same_index_set.add_constraint(cnstrnt)

    # }}}

    candidates_not_equal = isl_set_from_expr(set_pred.space,
                                             prim.Comparison(
                                                 prim.Variable(candidate_pred),
                                                 ">",
                                                 prim.Variable(candidate_succ)))
    return (not (set_pred
                 & set_succ
                 & accesses_same_index_set
                 & candidates_not_equal).is_empty())


def _preprocess_deps(kernel, deps, candidates, outer_inames):
    all_deps = set()

    for dep in deps:
        if kernel.id_to_insn[dep].within_inames == outer_inames:
            all_deps.add(dep)
        elif kernel.id_to_insn[dep].within_inames & candidates:
            all_deps.add(dep)
        else:
            deps |= reduce(frozenset.intersection,
                           (kernel.iname_to_insns()[iname]
                            for iname in kernel.id_to_insn[dep].within_inames),
                           frozenset(kernel.id_to_insn))

    return deps


def _build_ldg(kernel: LoopKernel,
               candidates: FrozenSet[str],
               outer_inames: FrozenSet[str]):
    """
    Returns an instance of :class:`LoopDependenceGraph` needed while fusing
    *candidates*. Invoked as a helper function in
    :func:`get_kennedy_unweighted_fusion_candidates`.
    """

    from pytools.graph import compute_topological_order

    insns = reduce(frozenset.intersection,
                   (frozenset(kernel.iname_to_insns()[iname])
                    for iname in outer_inames),
                   frozenset(kernel.id_to_insn))
    predecessors = {insn: (_preprocess_deps(kernel,
                                            kernel.id_to_insn[insn].depends_on,
                                            candidates=candidates,
                                            outer_inames=outer_inames)
                           & insns)
                    for insn in insns}
    successors = {insn: frozenset() for insn in insns}

    for insn, preds in predecessors.items():
        for pred in preds:
            successors[pred] |= frozenset([insn])

    predecessors, successors, infusible_edges = (
        _remove_irrelevant_insns_from_statement_dag(kernel,
                                                     predecessors,
                                                     successors,
                                                     candidates))

    builder = LoopDependenceGraphBuilder.new(candidates)

    # Interpret the statement DAG as LDG
    for pred, succs in successors.items():
        for succ in succs:
            succ_candidate, = kernel.id_to_insn[succ].within_inames & candidates
            pred_candidate, = kernel.id_to_insn[pred].within_inames & candidates
            builder.add_edge(pred_candidate, succ_candidate,
                             (pred, succ) in infusible_edges)

    # {{{ add infusible edges to the LDG depending on memory deps.

    all_candidate_insns = reduce(frozenset.union,
                                 (kernel.iname_to_insns()[iname]
                                  for iname in candidates),
                                 frozenset())

    dep_inducing_vars = reduce(frozenset.union,
                               (frozenset(kernel
                                          .id_to_insn[insn]
                                          .assignee_var_names())
                                for insn in all_candidate_insns),
                               frozenset())
    wmap = kernel.writer_map()
    rmap = kernel.reader_map()

    topo_order = {el: i
                  for i, el in enumerate(compute_topological_order(successors))}

    for var in dep_inducing_vars:
        for writer_id in (wmap.get(var, frozenset())
                          & all_candidate_insns):
            for access_id in ((rmap.get(var, frozenset())
                               | wmap.get(var, frozenset()))
                              & all_candidate_insns):
                if writer_id == access_id:
                    # no need to add self dependence
                    continue

                pred, succ = sorted([writer_id, access_id], key=topo_order.get)
                succ_candidate, = kernel.id_to_insn[succ].within_inames & candidates
                pred_candidate, = kernel.id_to_insn[pred].within_inames & candidates

                is_infusible = _compute_isinfusible_via_access_map(kernel,
                                                                   pred,
                                                                   pred_candidate,
                                                                   succ,
                                                                   succ_candidate,
                                                                   outer_inames,
                                                                   var)

                builder.add_edge(pred_candidate, succ_candidate, is_infusible)

    # }}}

    return builder.done()


def _fuse_sequential_loops_with_outer_loops(kernel: LoopKernel,
                                            candidates: FrozenSet[str],
                                            outer_inames: FrozenSet[str],
                                            name_gen, prefix):
    ldg = _build_ldg(kernel, candidates, outer_inames)
    fused_chunks = {}

    while not ldg.is_empty():

        # sorting to have a deterministic order.
        queue = sorted(ldg.get_loops_with_no_predecessors())
        loops_to_be_fused = set()
        non_fusible_loops = set()
        while queue:
            next_loop_in_queue = queue[0]
            queue = queue[1:]
            if not (ldg.predecessors[next_loop_in_queue] <= loops_to_be_fused):
                # this loop still needs some other loops to be scheduled
                # before we can reach this.
                # Bye bye 'next_loop_in_queue' :'( , see you when all your
                # predecessors have been scheduled.
                continue

            if next_loop_in_queue in non_fusible_loops:
                # had an non-fusible edge with an already schedule loop.
                # Sorry 'next_loop_in_queue', until next time :'(.
                continue

            loops_to_be_fused.add(next_loop_in_queue)

            for succ in ldg.successors[next_loop_in_queue]:
                if ldg.is_infusible.get((next_loop_in_queue, succ), False):
                    non_fusible_loops.add(succ)
                else:
                    queue.append(succ)

        ldg = ldg.remove_nodes(loops_to_be_fused)
        fused_chunks[name_gen(prefix)] = loops_to_be_fused

    assert reduce(frozenset.union, fused_chunks.values(), frozenset()) == candidates
    assert sum(len(val) for val in fused_chunks.values()) == len(candidates)

    return fused_chunks


def get_kennedy_unweighted_fusion_candidates(kernel: LoopKernel,
                                             candidates: FrozenSet[str],
                                             prefix="ifused"):
    """
    Returns the fusion candidates mapping that could be fed to
    :func:`rename_inames_in_batch` similar to Ken Kennedy's Unweighted
    Loop-Fusion Algorithm.

    .. attribute:: prefix

        Prefix for the fused inames.
    """
    from loopy.kernel.data import ConcurrentTag
    from loopy.schedule.tools import (
        _get_partial_loop_nest_tree,
        _get_iname_to_tree_node_id_from_partial_loop_nest_tree)

    assert isinstance(kernel, LoopKernel)
    assert isinstance(candidates, frozenset)

    vng = kernel.get_var_name_generator()
    fused_chunks = {}

    # {{{ handle concurrent inames

    # filter out concurrent loops.
    all_concurrent_tags = reduce(frozenset.union,
                                 (kernel.inames[iname].tags_of_type(ConcurrentTag)
                                  for iname in candidates),
                                 frozenset())

    concurrent_tag_to_inames = {tag: set()
                                for tag in all_concurrent_tags}

    for iname in candidates:
        if kernel.inames[iname].tags_of_type(ConcurrentTag):
            # since ConcurrentTag is a UniqueTag there must be exactly one of
            # it.
            tag, = kernel.tags_of_type(ConcurrentTag)
            concurrent_tag_to_inames[tag].add(iname)

    for inames in concurrent_tag_to_inames.values():
        fused_chunks[vng(prefix)] = inames
        candidates = candidates - inames

    # }}}

    tree = _get_partial_loop_nest_tree(kernel)
    iname_to_tree_node_id = (
        _get_iname_to_tree_node_id_from_partial_loop_nest_tree(tree))

    # {{{ sanitary checks

    _nest_tree_id_to_candidate = {}

    for iname in candidates:
        loop_nest_tree_node_id = iname_to_tree_node_id[iname]
        if loop_nest_tree_node_id not in _nest_tree_id_to_candidate:
            _nest_tree_id_to_candidate[loop_nest_tree_node_id] = iname
        else:
            conflict_iname = _nest_tree_id_to_candidate[loop_nest_tree_node_id]
            raise LoopyError(f"'{iname}' and '{conflict_iname}' "
                             "cannot fused be fused as they can be nested "
                             "within one another.")

    for iname in candidates:
        outer_loops = reduce(frozenset.union,
                             tree.ancestors(iname_to_tree_node_id[iname]),
                             frozenset())
        if outer_loops & candidates:
            raise LoopyError(f"Cannot fuse '{iname}' with"
                             f" '{outer_loops & candidates}' as they"
                             " maybe nesting within one another.")

    del _nest_tree_id_to_candidate

    # }}}

    # just_outer_loop_nest: mapping from loop nest to the candidates they
    # contain
    just_outer_loop_nest = {tree.parent(iname_to_tree_node_id[iname]): set()
                            for iname in candidates}

    for iname in candidates:
        just_outer_loop_nest[tree.parent(iname_to_tree_node_id[iname])].add(iname)

    for outer_inames, inames in just_outer_loop_nest.items():
        fused_chunks.update(_fuse_sequential_loops_with_outer_loops(kernel,
                                                                    inames,
                                                                    outer_inames,
                                                                    vng, prefix))

    return fused_chunks


def rename_inames_in_batch(kernel, batches: Mapping[str, FrozenSet[str]]):
    """
    Returns a copy of *kernel* with inames renamed according to *batches*.

    :arg kernel: An instance of :class:`loopy.LoopKernel`.
    :arg batches: A mapping from ``new_iname`` to a :class:`frozenset` of
        inames that are to be renamed to ``new_iname``.
    """
    from loopy.transform.iname import rename_iname
    for new_iname, candidates in batches.items():
        for iname in candidates:
            kernel = rename_iname(kernel, iname, new_iname, existing_ok=True)

    return kernel

# vim: foldmethod=marker

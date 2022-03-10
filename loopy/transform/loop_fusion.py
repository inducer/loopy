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
from loopy.symbolic import RuleAwareIdentityMapper
from loopy.kernel import LoopKernel
from typing import FrozenSet, Mapping, Tuple, Dict, Set, Callable, Optional
from functools import reduce
from dataclasses import dataclass
from pytools import memoize_on_first_arg

__doc__ = """
.. autofunction:: rename_inames_in_batch
.. autofunction:: get_kennedy_unweighted_fusion_candidates
"""


# {{{ Loop Dependence graph class + builder


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
    """
    A mutable type to act as a helper to instantiate a
    :class:`LoopDependenceGraphBuilder`.
    """
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


# {{{ _build_ldg

@dataclass(frozen=True, eq=True, repr=True)
class PreLDGNode:
    """
    A node in the graph representing the dependencies before building
    :class:`LoopDependenceGraph`.
    """


@dataclass(frozen=True, eq=True, repr=True)
class CandidateLoop(PreLDGNode):
    iname: str


@dataclass(frozen=True, eq=True, repr=True)
class NonCandidateLoop(PreLDGNode):
    loop_nest: FrozenSet[str]


@dataclass(frozen=True, eq=True, repr=True)
class OuterLoopNestStatement(PreLDGNode):
    insn_id: str


def _remove_non_candidate_pre_ldg_nodes(kernel,
                                        predecessors: Mapping[PreLDGNode,
                                                              PreLDGNode],
                                        successors: Mapping[PreLDGNode,
                                                            PreLDGNode],
                                        candidates: FrozenSet[str]):
    """
    Returns ``(new_successors, new_predecessors, inufusible_edge)`` where
    ``(new_successors, new_predecessors)`` is the graph describing the
    dependencies between the *candidates* loops that has been obtained by
    removing instances of :class:`NonCandidateLoop` and
    :class:`OuterLoopNestStatement` from the graph described by *predecessors*,
    *succcessors*.

    New dependency edges are added in the new graph to preserve the transitive
    dependencies that exists in the original graph.
    """
    # {{{ input validation

    assert set(predecessors) == set(successors)
    assert all(isinstance(val, frozenset) for val in predecessors.values())
    assert all(isinstance(val, frozenset) for val in successors.values())

    # }}}

    nodes_to_remove = {node
                       for node in predecessors
                       if isinstance(node, (NonCandidateLoop,
                                            OuterLoopNestStatement))
                       }
    new_predecessors = predecessors.copy()
    new_successors = successors.copy()
    infusible_edges_in_statement_dag = set()

    for node_to_remove in nodes_to_remove:
        for pred in new_predecessors[node_to_remove]:
            new_successors[pred] = ((new_successors[pred]
                                     - frozenset([node_to_remove]))
                                    | new_successors[node_to_remove])

        for succ in new_successors[node_to_remove]:
            new_predecessors[succ] = ((new_predecessors[succ]
                                       - frozenset([node_to_remove]))
                                      | new_predecessors[node_to_remove])

        for pred in new_predecessors[node_to_remove]:
            for succ in new_successors[node_to_remove]:
                # now mark the edge from pred -> succ infusible iff both 'pred' and
                # 'succ' are *not* in insns_to_remove
                if ((pred not in nodes_to_remove) and (succ not in nodes_to_remove)):
                    assert isinstance(pred, CandidateLoop)
                    assert isinstance(succ, CandidateLoop)
                    infusible_edges_in_statement_dag.add((pred.iname, succ.iname))

        del new_predecessors[node_to_remove]
        del new_successors[node_to_remove]

    return ({key.iname: frozenset({n.iname for n in value})
             for key, value in new_predecessors.items()},
            {key.iname: frozenset({n.iname for n in value})
             for key, value in new_successors.items()},
            infusible_edges_in_statement_dag)


def _get_ldg_nodes_from_loopy_insn(kernel, insn, candidates, non_candidates,
                                   just_outer_loop_nest):
    """
    Helper used in :func:`_build_ldg`.

    :arg just_outer_inames: A :class:`frozenset` of the loop nest that appears
        just outer to the *candidates* in the partial loop nest tree.
    """
    if (insn.within_inames | insn.reduction_inames()) & candidates:
        # => the statement containing
        return [CandidateLoop(candidate)
                for candidate in ((insn.within_inames
                                   | insn.reduction_inames())
                                  & candidates)]
    elif {loop_nest
          for loop_nest in non_candidates
          if (loop_nest & insn.within_inames)}:
        non_candidate, = {loop_nest
                          for loop_nest in non_candidates
                          if (loop_nest & insn.within_inames)}

        return [NonCandidateLoop(non_candidate)]
    else:
        assert ((insn.within_inames & just_outer_loop_nest)
                or (insn.within_inames == just_outer_loop_nest))
        return [OuterLoopNestStatement(insn.id)]


@memoize_on_first_arg
def _compute_isinfusible_via_access_map(kernel,
                                        insn_pred, candidate_pred,
                                        insn_succ, candidate_succ,
                                        outer_inames,
                                        var):
    """
    Returns *True* if the inames *candidate_pred* and *candidate_succ* are fused then
    that might lead to a loop carried dependency for *var*.

    Helper used in :func:`_build_ldg`.
    """
    import islpy as isl
    from loopy.kernel.tools import get_insn_access_map
    import pymbolic.primitives as prim
    from loopy.symbolic import isl_set_from_expr
    from loopy.diagnostic import UnableToDetermineAccessRangeError

    try:
        amap_pred = get_insn_access_map(kernel, insn_pred, var)
        amap_succ = get_insn_access_map(kernel, insn_succ, var)
    except UnableToDetermineAccessRangeError:
        # either predecessors or successors has a non-affine access i.e.
        # fallback to the safer option => infusible
        return True

    amap_pred = amap_pred.project_out_except(outer_inames | {candidate_pred},
                                             [isl.dim_type.param,
                                              isl.dim_type.in_])
    amap_succ = amap_succ.project_out_except(outer_inames | {candidate_succ},
                                             [isl.dim_type.param,
                                              isl.dim_type.in_])

    for outer_iname in sorted(outer_inames):
        amap_pred = amap_pred.move_dims(dst_type=isl.dim_type.param,
                                        dst_pos=amap_pred.dim(isl.dim_type.param),
                                        src_type=isl.dim_type.in_,
                                        src_pos=amap_pred.get_var_dict()[
                                            outer_iname][1],
                                        n=1)
        amap_succ = amap_succ.move_dims(dst_type=isl.dim_type.param,
                                        dst_pos=amap_succ.dim(isl.dim_type.param),
                                        src_type=isl.dim_type.in_,
                                        src_pos=amap_succ.get_var_dict()[
                                            outer_iname][1],
                                        n=1)

    # since both ranges denote the same variable they must be subscripted with
    # the same number of indices.
    assert amap_pred.dim(isl.dim_type.out) == amap_succ.dim(isl.dim_type.out)
    assert amap_pred.dim(isl.dim_type.in_) == 1
    assert amap_succ.dim(isl.dim_type.in_) == 1

    if amap_pred == amap_succ:
        return False

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
    result = (not (set_pred
                 & set_succ
                 & accesses_same_index_set & candidates_not_equal).is_empty())

    return result


def _build_ldg(kernel: LoopKernel,
               candidates: FrozenSet[str],
               outer_inames: FrozenSet[str]):
    """
    Returns an instance of :class:`LoopDependenceGraph` needed while fusing
    *candidates*. Invoked as a helper function in
    :func:`get_kennedy_unweighted_fusion_candidates`.
    """

    from pytools.graph import compute_topological_order

    loop_nest_tree = _get_partial_loop_nest_tree_for_fusion(kernel)

    non_candidate_loop_nests = {
        child_loop_nest
        for child_loop_nest in loop_nest_tree.children(outer_inames)
        if len(child_loop_nest & candidates) == 0}

    insns = reduce(frozenset.intersection,
                   (frozenset(kernel.iname_to_insns()[iname])
                    for iname in outer_inames),
                   frozenset(kernel.id_to_insn))
    predecessors = {}
    successors = {}

    for insn in insns:
        for successor in _get_ldg_nodes_from_loopy_insn(kernel,
                                                        kernel.id_to_insn[insn],
                                                        candidates,
                                                        non_candidate_loop_nests,
                                                        outer_inames):
            predecessors.setdefault(successor, set())
            successors.setdefault(successor, set())
            for dep in kernel.id_to_insn[insn].depends_on:
                if ((kernel.id_to_insn[dep].within_inames & outer_inames)
                        != outer_inames):
                    # this is not an instruction in 'outer_inames' => bogus dep.
                    continue
                for predecessor in _get_ldg_nodes_from_loopy_insn(
                        kernel,
                        kernel.id_to_insn[dep],
                        candidates,
                        non_candidate_loop_nests,
                        outer_inames):
                    if predecessor != successor:
                        predecessors.setdefault(successor, set()).add(predecessor)
                        successors.setdefault(predecessor, set()).add(successor)

    predecessors, successors, infusible_edges = (
        _remove_non_candidate_pre_ldg_nodes(
            kernel,
            {key: frozenset(value)
             for key, value in predecessors.items()},
            {key: frozenset(value)
             for key, value in successors.items()},
            candidates))
    del predecessors

    builder = LoopDependenceGraphBuilder.new(candidates)

    # Interpret the statement DAG as LDG
    for pred, succs in successors.items():
        for succ in succs:
            builder.add_edge(pred, succ,
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

                writer_candidate, = (kernel.id_to_insn[writer_id].within_inames
                                     & candidates)
                access_candidate, = (kernel.id_to_insn[access_id].within_inames
                                     & candidates)
                (pred_candidate, pred), (succ_candidate, succ) = sorted(
                    [(writer_candidate, writer_id),
                     (access_candidate, access_id)],
                    key=lambda x: topo_order[x[0]])

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

# }}}


def _fuse_sequential_loops_with_outer_loops(kernel: LoopKernel,
                                            candidates: FrozenSet[str],
                                            outer_inames: FrozenSet[str],
                                            name_gen, prefix, force_infusible):
    from collections import deque
    ldg = _build_ldg(kernel, candidates, outer_inames)

    fused_chunks = {}

    while not ldg.is_empty():

        # sorting to have a deterministic order.
        # prefer 'deque' over list, as popping elements off the queue would be
        # O(1).

        loops_with_no_preds = sorted(ldg.get_loops_with_no_predecessors())

        queue = deque([loops_with_no_preds[0]])
        for node in loops_with_no_preds[1:]:
            if not force_infusible(node, loops_with_no_preds[0]):
                queue.append(node)

        loops_to_be_fused = set()
        non_fusible_loops = set()
        while queue:
            next_loop_in_queue = queue.popleft()

            if next_loop_in_queue in non_fusible_loops:
                # had an non-fusible edge with an already schedule loop.
                # Sorry 'next_loop_in_queue', until next time :'(.
                continue

            if next_loop_in_queue in loops_to_be_fused:
                # already fused, no need to fuse again ;)
                continue

            if not (ldg.predecessors[next_loop_in_queue] <= loops_to_be_fused):
                # this loop still needs some other loops to be scheduled
                # before we can reach this.
                # Bye bye 'next_loop_in_queue' :'( , see you when all your
                # predecessors have been scheduled.
                continue

            loops_to_be_fused.add(next_loop_in_queue)

            for succ in ldg.successors[next_loop_in_queue]:
                if (ldg.is_infusible.get((next_loop_in_queue, succ), False)
                        or force_infusible(next_loop_in_queue, succ)):
                    non_fusible_loops.add(succ)
                else:
                    queue.append(succ)

        ldg = ldg.remove_nodes(loops_to_be_fused)
        fused_chunks[name_gen(prefix)] = loops_to_be_fused

    assert reduce(frozenset.union, fused_chunks.values(), frozenset()) == candidates
    assert sum(len(val) for val in fused_chunks.values()) == len(candidates)

    return fused_chunks


class ReductionLoopInserter(RuleAwareIdentityMapper):
    """
    Main mapper used by :func:`_add_reduction_loops_in_partial_loop_nest_tree`.
    """
    def __init__(self, rule_mapping_context, tree):
        super().__init__(rule_mapping_context)
        self.tree = tree
        from loopy.schedule.tools import (
            _get_iname_to_tree_node_id_from_partial_loop_nest_tree)
        self.iname_to_tree_node_id = (
            _get_iname_to_tree_node_id_from_partial_loop_nest_tree(tree))

    def map_reduction(self, expr, expn_state, *, outer_redn_inames=frozenset()):
        redn_inames = frozenset(expr.inames)
        iname_chain = (expn_state.instruction.within_inames
                       | outer_redn_inames
                       | redn_inames)
        not_seen_inames = frozenset(iname for iname in iname_chain
                                    if iname not in self.iname_to_tree_node_id)
        seen_inames = iname_chain - not_seen_inames

        # {{{ verbatim copied from loopy/schedule/tools.py

        from loopy.schedule.tools import (_pull_out_loop_nest,
                                          _add_inner_loops)

        all_nests = {self.iname_to_tree_node_id[iname]
                     for iname in seen_inames}

        self.tree, outer_loop, inner_loop = _pull_out_loop_nest(self.tree,
                                                                (all_nests
                                                                 | {frozenset()}),
                                                                seen_inames)
        if not_seen_inames:
            # make '_not_seen_inames' nest inside the seen ones.
            # example: if there is already a loop nesting "i,j,k"
            # and the current iname chain is "i,j,l". Only way this is possible
            # is if "l" is nested within "i,j"-loops.
            self.tree = _add_inner_loops(self.tree, outer_loop, not_seen_inames)

        # {{{ update iname to node id

        for iname in outer_loop:
            self.iname_to_tree_node_id = self.iname_to_tree_node_id.set(iname,
                                                                        outer_loop)

        if inner_loop is not None:
            for iname in inner_loop:
                self.iname_to_tree_node_id = self.iname_to_tree_node_id.set(
                    iname, inner_loop)

        for iname in not_seen_inames:
            self.iname_to_tree_node_id = self.iname_to_tree_node_id.set(
                iname, not_seen_inames)

        # }}}

        # }}}

        assert not (outer_redn_inames & redn_inames)
        return super().map_reduction(
            expr,
            expn_state,
            outer_redn_inames=(outer_redn_inames | redn_inames))


def _add_reduction_loops_in_partial_loop_nest_tree(kernel, tree):
    """
    Returns a partial loop nest tree with the loop nests corresponding to the
    reduction inames added to *tree*.
    """
    from loopy.symbolic import SubstitutionRuleMappingContext
    rule_mapping_context = SubstitutionRuleMappingContext(
        kernel.substitutions, kernel.get_var_name_generator())
    reduction_loop_inserter = ReductionLoopInserter(rule_mapping_context, tree)

    def does_insn_have_reduce(kernel, insn, *args):
        return bool(insn.reduction_inames())

    reduction_loop_inserter.map_kernel(kernel,
                                       within=does_insn_have_reduce,
                                       map_tvs=False, map_args=False)
    return reduction_loop_inserter.tree


def _get_partial_loop_nest_tree_for_fusion(kernel):
    from loopy.schedule.tools import _get_partial_loop_nest_tree
    tree = _get_partial_loop_nest_tree(kernel)
    tree = _add_reduction_loops_in_partial_loop_nest_tree(kernel, tree)
    return tree


def get_kennedy_unweighted_fusion_candidates(
        kernel: LoopKernel,
        candidates: FrozenSet[str],
        *,
        force_infusible: Optional[Callable[[str, str], bool]] = None,
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
        _get_iname_to_tree_node_id_from_partial_loop_nest_tree)
    from collections.abc import Collection
    assert not isinstance(candidates, str)
    assert isinstance(candidates, Collection)
    assert isinstance(kernel, LoopKernel)

    candidates = frozenset(candidates)
    vng = kernel.get_var_name_generator()
    fused_chunks = {}

    if force_infusible is None:
        force_infusible = lambda x, y: False  # noqa: E731

    # {{{ implementation scope

    # All of the candidates must be either "pure" reduction loops or
    # pure-within_inames loops.
    # Reason: otherwise _compute_isinfusible_via_access_map might result in
    # spurious results.
    # One option is to simply perform 'realize_reduction' before implementing
    # this algorithm, but that seems like an unnecessary cost to pay.
    if any(candidates & insn.reduction_inames()
           for insn in kernel.instructions):
        if any(candidates & insn.within_inames
               for insn in kernel.instructions):
            raise NotImplementedError("Some candidates are reduction"
                                      " inames while some of them are not. Such"
                                      " cases are not yet supported.")

    # }}}

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

    tree = _get_partial_loop_nest_tree_for_fusion(kernel)

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
                                                                    vng,
                                                                    prefix,
                                                                    force_infusible
                                                                    ))

    return fused_chunks


def rename_inames_in_batch(kernel, batches: Mapping[str, FrozenSet[str]]):
    """
    Returns a copy of *kernel* with inames renamed according to *batches*.

    :arg kernel: An instance of :class:`loopy.LoopKernel`.
    :arg batches: A mapping from ``new_iname`` to a :class:`frozenset` of
        inames that are to be renamed to ``new_iname``.
    """
    from loopy.transform.iname import rename_inames, remove_unused_inames
    for new_iname, candidates in batches.items():
        # pylint:disable=unexpected-keyword-arg
        kernel = rename_inames(
            kernel, candidates, new_iname,
            remove_newly_unused_inames=False
        )

    return remove_unused_inames(kernel, reduce(frozenset.union,
                                               batches.values(),
                                               frozenset()))

# vim: foldmethod=marker

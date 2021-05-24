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
from loopy.diagnostic import LoopyError
from loopy.tools import Tree
from functools import reduce
from pytools import memoize_on_first_arg
from pyrsistent import pmap


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
    from loopy.schedule.tree import Schedule, get_insns_in_function

    if isinstance(kernel.schedule, Schedule):
        insn_ids = get_insns_in_function(kernel, subkernel)
    else:
        assert isinstance(kernel.schedule, list)
        insn_ids = get_subkernel_to_insn_id_map(kernel)[subkernel]

    return frozenset(tv
            for insn_id in insn_ids
            for tv in kernel.id_to_insn[insn_id].read_dependency_names()
            if tv in kernel.temporary_variables)


def temporaries_written_in_subkernel(kernel, subkernel):
    from loopy.kernel.tools import get_subkernel_to_insn_id_map
    from loopy.schedule.tree import Schedule, get_insns_in_function

    if isinstance(kernel.schedule, Schedule):
        insn_ids = get_insns_in_function(kernel, subkernel)
    else:
        assert isinstance(kernel.schedule, list)
        insn_ids = get_subkernel_to_insn_id_map(kernel)[subkernel]

    return frozenset(tv
            for insn_id in insn_ids
            for tv in kernel.id_to_insn[insn_id].write_dependency_names()
            if tv in kernel.temporary_variables)

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


def _pull_out_loop_nest(tree, loop_nests, inames_to_pull_out):
    """
    Returns a copy of *tree* that realizes *inames_to_pull_out* as loop
    nesting.

    :arg tree: A :class:`loopy.tools.Tree`, where each node is
        :class:`frozenset` of inames representing a loop nest. For example a
        tree might look like:

    :arg loop_nests: A collection of nodes in *tree* that cover
        *inames_to_pull_out*.

    :returns: a :class:`tuple` ``(new_tree, outer_loop_nest, inner_loop_nest)``,
        where outer_loop_nest is the identifier for the new outer and inner
        loop nests so that *inames_to_pull_out* is a valid nesting.

    .. note::

        We could compute *loop_nests* within this routine's implementation, but
        computing would be expensive and hence we ask the caller for this info.

    Example::
       *tree*: frozenset()
               └── frozenset({'j', 'i'})
                   └── frozenset({'k', 'l'})

       *inames_to_pull_out*: frozenset({'k', 'i', 'j'})
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
    assert inames_to_pull_out <= reduce(frozenset.union, loop_nests, frozenset())

    # {{{ sanity check to ensure the loop nest *inames_to_pull_out* is possible

    loop_nests = sorted(loop_nests, key=lambda nest: tree.depth(nest))

    for outer, inner in zip(loop_nests[:-1], loop_nests[1:]):
        if outer != tree.parent(inner):
            raise LoopyError(f"Cannot schedule loop nest {inames_to_pull_out} "
                             f" in the nesting tree:\n{tree}")

    assert tree.depth(loop_nests[0]) == 0

    # }}}

    innermost_loop_nest = loop_nests[-1]
    new_outer_loop_nest = inames_to_pull_out - reduce(frozenset.union,
                                                      loop_nests[:-1],
                                                      frozenset())
    new_inner_loop_nest = innermost_loop_nest - inames_to_pull_out

    if new_outer_loop_nest == innermost_loop_nest:
        # such a loop nesting already exists => do nothing
        return tree, new_outer_loop_nest, None

    # add the outer loop to our loop nest tree
    tree = tree.add_node(new_outer_loop_nest,
                         parent=tree.parent(innermost_loop_nest))

    # rename the old loop to the inner loop
    tree = tree.rename_node(innermost_loop_nest,
                            new_id=new_inner_loop_nest)

    # set the parent of inner loop to be the outer loop
    tree = tree.move_node(new_inner_loop_nest, new_parent=new_outer_loop_nest)

    return tree, new_outer_loop_nest, new_inner_loop_nest


def _add_inner_loops(tree, outer_loop_nest, inner_loop_nest):
    """
    Returns a copy of *tree* that nests *inner_loop_nest* inside *outer_loop_nest*.
    """
    # add the outer loop to our loop nest tree
    return tree.add_node(inner_loop_nest, parent=outer_loop_nest)


def _order_loop_nests(loop_nest_tree,
                      strict_priorities,
                      relaxed_priorities,
                      iname_to_tree_node_id):
    """
    Returns a loop nest where all nodes in the tree are instances of
    :class:`str` denoting inames. Unlike *loop_nest_tree* which corresponds to
    multiple loop nesting, this routine returns a unique loop nest that is
    obtained after constraining *loop_nest_tree* with the constraints enforced
    by *priorities*.

    :arg strict_priorities: Expresses strict nesting constraints similar to
        :attr:`loopy.LoopKernel.loop_priorities`. These priorities are imposed
        strictly i.e. if these conditions cannot be met a
        :class:`loopy.diagnostic.LoopyError` is raised.

    :arg relaxed_priorities: Expresses strict nesting constraints similar to
        :attr:`loopy.LoopKernel.loop_priorities`. These nesting constraints are
        treated as options.

    :arg iname_to_tree_node_id: A mapping from iname to the loop nesting its a
        part of.
    """
    from pytools.graph import compute_topological_order as toposort
    from warnings import warn

    loop_nests = set(iname_to_tree_node_id.values())

    # flow_requirements: A mapping from the loop nest level to the nesting
    # constraints applicable to it.
    # Each nesting constraint is represented as a DAG. In the DAG, if there
    # exists an edge from from iname 'i' -> iname 'j' => 'j' should be nested
    # inside 'i'.
    flow_requirements = {loop_nest: {iname: frozenset()
                                     for iname in loop_nest}
                         for loop_nest in loop_nests}

    # The plan here is populate DAGs in *flow_requirements* and then perform a
    # toposort for each loop nest.

    def _update_flow_requirements(priorities, cannot_satisfy_callback):
        """
        Records *priorities* in *flow_requirements* and calls
        *cannot_satisfy_callback* with an appropriate error message if the
        priorities cannot be met.
        """
        for priority in priorities:
            for outer_iname, inner_iname in zip(priority[:-1], priority[1:]):
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
                    flow_requirements[inner_iname_nest][outer_iname] |= {inner_iname}
                else:
                    ancestors_of_inner_iname = (loop_nest_tree
                                                .ancestors(inner_iname_nest))
                    ancestors_of_outer_iname = (loop_nest_tree
                                                .ancestors(outer_iname_nest))
                    if outer_iname in ancestors_of_inner_iname:
                        # nesting constraint already satisfied => do nothing
                        pass
                    elif inner_iname in ancestors_of_outer_iname:
                        cannot_satisfy_callback("Cannot satisfy constraint that"
                                                f" iname '{inner_iname}' must be"
                                                f" nested within '{outer_iname}''.")
                    else:
                        # inner iname and outer iname are indirect family members
                        # => must be realized via dependencies in the linearization
                        # phase, not implemented in v2-scheduler yet.
                        from loopy.schedule import V2SchedulerNotImplementedException
                        raise V2SchedulerNotImplementedException("cannot"
                                " schedule kernels with priority dependencies"
                                " between sibling loop nests")

    def _raise_loopy_err(x):
        raise LoopyError(x)

    # record strict priorities
    _update_flow_requirements(strict_priorities, _raise_loopy_err)
    # record relaxed priorities
    _update_flow_requirements(relaxed_priorities, warn)

    # ordered_loop_nests: A mapping from the unordered loop nests to their
    # ordered couterparts. For example. If we had only one loop nest
    # `frozenset({"i", "j", "k"})`, and the prioirities said added the
    # constraint that "i" must be nested within "k", then `ordered_loop_nests`
    # would be: `{frozenset({"i", "j", "k"}): ["j", "k", "i"]}` i.e. the loop
    # nests would now have an order.
    ordered_loop_nests = {unordered_nest: toposort(flow,
                                                   key=lambda x: x)
                          for unordered_nest, flow in flow_requirements.items()}

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
                                     parent=old_to_new_parent[loop_nest_tree
                                                              .parent(current_nest)])
        for new_parent, new_child in zip(ordered_nest[:-1], ordered_nest[1:]):
            new_tree = new_tree.add_node(node=new_child, parent=new_parent)

        old_to_new_parent[current_nest] = ordered_nest[-1]

        queue.extend(list(loop_nest_tree.children(current_nest)))

    # }}}

    return new_tree


@memoize_on_first_arg
def _get_parallel_inames(kernel):
    from loopy.kernel.data import ConcurrentTag, IlpBaseTag, VectorizeTag

    concurrent_inames = {iname for iname in kernel.all_inames()
                         if kernel.iname_tags_of_type(iname, ConcurrentTag)}
    ilp_inames = {iname for iname in kernel.all_inames()
                  if kernel.iname_tags_of_type(iname, IlpBaseTag)}
    vec_inames = {iname for iname in kernel.all_inames()
                  if kernel.iname_tags_of_type(iname, VectorizeTag)}
    return (concurrent_inames - ilp_inames - vec_inames)


def _get_partial_loop_nest_tree(kernel):
    """
    Returns :class:`loopy.Tree` representing the *kernel*'s loop-nests.

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
    iname_chains = {insn.within_inames - _get_parallel_inames(kernel)
                     for insn in kernel.instructions}

    root = frozenset()
    tree = Tree.from_root(root)

    # mapping from iname to the innermost loop nest they are part of in *tree*.
    iname_to_tree_node_id = {}

    # if there were any loop with no inames, those have been already account
    # for as the root.
    iname_chains = iname_chains - {root}

    for iname_chain in iname_chains:
        not_seen_inames = frozenset(iname for iname in iname_chain
                                    if iname not in iname_to_tree_node_id)
        seen_inames = iname_chain - not_seen_inames

        all_nests = {iname_to_tree_node_id[iname] for iname in seen_inames}

        tree, outer_loop, inner_loop = _pull_out_loop_nest(tree,
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

    for iname_chain in iname_chains:
        for ilp_iname in (ilp_inames & iname_chains):
            # pull out other loops so that ilp_iname is the innermost
            all_nests = {iname_to_tree_node_id[iname] for iname in seen_inames}
            tree, outer_loop, inner_loop = _pull_out_loop_nest(tree,
                                                               (all_nests
                                                                | {frozenset()}),
                                                               (iname_chain
                                                                - {ilp_iname}))

            for iname in outer_loop:
                iname_to_tree_node_id[iname] = outer_loop

            if inner_loop is not None:
                for iname in inner_loop:
                    iname_to_tree_node_id[iname] = inner_loop

    # }}}

    return tree


def _get_iname_to_tree_node_id_from_partial_loop_nest_tree(tree):
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

    return pmap(iname_to_tree_node_id)


def get_loop_nest_tree(kernel):
    """
    Returns ```tree``` (an instance of :class:`Tree`) representing the loop
    nesting for *kernel*. Each node of ``tree`` is an instance of :class:`str`
    corresponding to the inames of *kernel* that are realized as concrete
    ``for-loops``. A parent node in `tree` is always nested outside all its
    children.

    .. note::

        Multiple loop nestings might exist for *kernel*, but this routine returns
        one valid loop nesting.
    """
    from islpy import dim_type

    tree = _get_partial_loop_nest_tree(kernel)
    iname_to_tree_node_id = (
        _get_iname_to_tree_node_id_from_partial_loop_nest_tree(tree))

    strict_loop_priorities = frozenset()

    # {{{ impose constraints by the domain tree

    loop_inames = (reduce(frozenset.union,
                          (insn.within_inames
                           for insn in kernel.instructions),
                          frozenset())
                   - _get_parallel_inames(kernel))

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

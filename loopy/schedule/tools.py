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
from treelib import Tree
from collections import defaultdict
from functools import reduce


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
    return frozenset(tv
            for insn_id in insn_ids
            for tv in kernel.id_to_insn[insn_id].read_dependency_names()
            if tv in kernel.temporary_variables)


def temporaries_written_in_subkernel(kernel, subkernel):
    from loopy.kernel.tools import get_subkernel_to_insn_id_map
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

    for sched_item in kernel.schedule:
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

    return kernel.copy(schedule=new_schedule)

# }}}


class _not_seen:  # noqa: N801
    pass


def pull_out_loop_nest(tree, loop_nests, inames_to_pull_out):
    """
    Updates *tree* to make *inames_to_pull_out* a loop nesting level in
    *loop_nests*

    :returns: a :class:`tuple` ``(outer_loop_nest, inner_loop_nest)``, where
        outer_loop_nest is the identifier for the new outer and inner loop
        nests so that *inames_to_pull_out* is a valid nesting.
    """
    assert all(isinstance(loop_nest, frozenset) for loop_nest in loop_nests)
    assert inames_to_pull_out <= reduce(frozenset.union, loop_nests, frozenset())

    # {{{ sanity check to ensure the loop nest *inames_to_pull_out* is possible

    loop_nests = sorted(loop_nests, key=lambda nest: tree.depth(nest))

    for outer, inner in zip(loop_nests[:-1], loop_nests[1:]):
        if outer != tree.parent(inner).identifier:
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
        return new_outer_loop_nest, None

    # add the outer loop to our loop nest tree
    tree.create_node(identifier=new_outer_loop_nest,
                     parent=tree.parent(innermost_loop_nest).identifier)

    # rename the old loop to the inner loop
    tree.update_node(innermost_loop_nest,
                     identifier=new_inner_loop_nest,
                     tag=new_inner_loop_nest)

    # set the parent of inner loop to be the outer loop
    tree.move_node(new_inner_loop_nest, new_outer_loop_nest)

    return new_outer_loop_nest, new_inner_loop_nest


def add_inner_loops(tree, outer_loop_nest, inner_loop_nest):
    """
    Update *tree* to nest *inner_loop_nest* inside *outer_loop_nest*.
    """
    # add the outer loop to our loop nest tree
    tree.create_node(identifier=inner_loop_nest, parent=outer_loop_nest)


def get_loop_nest_tree(kernel):
    """
    Returns an instance of :class:`treelib.Tree` denoting the kernel's loop
    nestings.

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
    from islpy import dim_type
    from loopy.kernel.data import ConcurrentTag, IlpBaseTag

    concurrent_inames = {iname for iname in kernel.all_inames()
                         if kernel.iname_tags_of_type(iname, ConcurrentTag)}
    ilp_inames = {iname for iname in kernel.all_inames()
                  if kernel.iname_tags_of_type(iname, IlpBaseTag)}

    # figuring the possible loop nestings minus the concurrent_inames as they
    # are never realized as actual loops
    iname_chains = {insn.within_inames - concurrent_inames
                     for insn in kernel.instructions}

    tree = Tree()
    root = frozenset()

    # mapping from iname to the innermost loop nest they are part of in *tree*.
    iname_to_tree_node_id = defaultdict(lambda: _not_seen)

    tree.create_node(identifier=root)

    # if there were any loop with no inames, those have been already account
    # for as the root.
    iname_chains = iname_chains - {root}

    for iname_chain in iname_chains:
        not_seen_inames = frozenset(iname for iname in iname_chain
                                    if iname_to_tree_node_id[iname] is _not_seen)
        seen_inames = iname_chain - not_seen_inames

        all_nests = {iname_to_tree_node_id[iname] for iname in seen_inames}

        outer_loop, inner_loop = pull_out_loop_nest(tree,
                                                    (all_nests | {frozenset()}),
                                                    seen_inames)
        if not_seen_inames:
            add_inner_loops(tree, outer_loop, not_seen_inames)

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

    for iname_chain in iname_chains:
        for ilp_iname in (ilp_inames & iname_chains):
            # pull out other loops so that ilp_iname is the innermost
            all_nests = {iname_to_tree_node_id[iname] for iname in seen_inames}
            outer_loop, inner_loop = pull_out_loop_nest(tree,
                                                        (all_nests | {frozenset()}),
                                                        iname_chain - {ilp_iname})

            for iname in outer_loop:
                iname_to_tree_node_id[iname] = outer_loop

            if inner_loop is not None:
                for iname in inner_loop:
                    iname_to_tree_node_id[iname] = inner_loop

    # }}}

    loop_priorities = kernel.loop_priority.copy()

    # {{{ impose constraints by the domain tree

    all_inames = kernel.all_inames()

    for dom_idx, dom in enumerate(kernel.domains):
        for outer_iname in dom.get_var_names(dim_type.param):
            if outer_iname not in all_inames:
                continue

            for inner_iname in dom.get_var_names(dim_type.set):
                # either outer_iname and inner_iname should belong to the same
                # loop nest level or outer should be strictly outside inner
                # iname

                inner_iname_nest = iname_to_tree_node_id[inner_iname]
                outer_iname_nest = iname_to_tree_node_id[outer_iname]

                if inner_iname_nest == outer_iname_nest:
                    loop_priorities |= {(outer_iname, inner_iname)}
                else:
                    ancestors_of_inner_iname = {
                        tree.ancestor(inner_iname_nest, k)
                        for k in range(tree.depth(inner_iname_nest))}
                    if outer_iname_nest not in ancestors_of_inner_iname:
                        raise LoopyError(f"Loop '{outer_iname}' cannot be nested"
                                         f" outside '{inner_iname}'.")

    # }}}

    if loop_priorities:
        raise NotImplementedError

    # {{{ just choose one of the possible loop nestings

    # Either all of these loop nestings would be valid or all would invalid =>
    # we aren't marking any schedulable kernel as unschedulable.

    new_tree = Tree()

    old_to_new_parent = {}

    new_tree.create_node(identifier="")
    old_to_new_parent[root] = ""

    # traversing 'tree' in an BFS fashion to create 'new_tree'
    queue = [node.identifier for node in tree.children(root)]

    while queue:
        current_node = queue.pop(0)

        sorted_inames = sorted(current_node)
        new_tree.create_node(identifier=sorted_inames[0],
                             parent=old_to_new_parent[tree.parent(current_node)
                                                      .identifier])
        for new_parent, new_child in zip(sorted_inames[:-1], sorted_inames[1:]):
            new_tree.create_node(identifier=new_child, parent=new_parent)

        old_to_new_parent[current_node] = sorted_inames[-1]

        queue.extend([child.identifier for child in tree.children(current_node)])

    # }}}

    return new_tree

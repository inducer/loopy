# mypy: disallow-untyped-defs

from __future__ import annotations


__copyright__ = """
Copyright (C) 2022 Kaushik Kulkarni
Copyright (C) 2022-24 University of Illinois Board of Trustees
"""


__doc__ = """
.. autoclass:: NodeT
.. autoclass:: Tree
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

import operator
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property, reduce
from typing import Generic, TypeVar

from immutabledict import immutabledict

from pytools import memoize_method


# {{{ tree data structure

NodeT = TypeVar("NodeT", bound=Hashable)


# Not frozen when optimizations are enabled because it is slower.
# Tree objects are immutable, and offer no way to mutate the tree.
@dataclass(frozen=__debug__)  # type: ignore[literal-required]
class Tree(Generic[NodeT]):
    """
    An immutable tree containing nodes of type :class:`NodeT`.

    .. automethod:: ancestors
    .. automethod:: parent
    .. automethod:: children
    .. automethod:: add_node
    .. automethod:: depth
    .. automethod:: replace_node
    .. automethod:: move_node

    .. automethod:: __contains__

    .. note::

       Almost all the operations are implemented recursively. NOT suitable for
       deep trees. At the very least if the Python implementation is CPython
       this allocates a new stack frame for each iteration of the operation.
    """

    _parent_to_children: immutabledict[NodeT, tuple[NodeT, ...]]
    _child_to_parent: immutabledict[NodeT, NodeT | None]

    @staticmethod
    def from_root(root: NodeT) -> Tree[NodeT]:
        return Tree(immutabledict({root: ()}),
                    immutabledict({root: None}))

    @cached_property
    def root(self) -> NodeT:
        guess = set(self._child_to_parent).pop()
        parent_of_guess = self.parent(guess)
        while parent_of_guess is not None:
            guess = parent_of_guess
            parent_of_guess = self.parent(guess)

        return guess

    @memoize_method
    def ancestors(self, node: NodeT) -> tuple[NodeT, ...]:
        """
        Returns a :class:`tuple` of nodes that are ancestors of *node*.
        """
        parent = self.parent(node)
        if parent is None:
            # => root
            return ()

        return (parent, *self.ancestors(parent))

    def parent(self, node: NodeT) -> NodeT | None:
        """
        Returns the parent of *node*.
        """
        return self._child_to_parent[node]

    def children(self, node: NodeT) -> tuple[NodeT, ...]:
        """
        Returns the children of *node*.
        """
        return self._parent_to_children[node]

    @memoize_method
    def depth(self, node: NodeT) -> int:
        """
        Returns the depth of *node*, with the root having depth 0.
        """
        parent_of_node = self.parent(node)
        if parent_of_node is None:
            return 0

        return 1 + self.depth(parent_of_node)

    def is_root(self, node: NodeT) -> bool:
        """Return *True* if *node* is the root of the tree."""
        return self.parent(node) is None

    def is_leaf(self, node: NodeT) -> bool:
        """Return *True* if *node* has no children."""
        return len(self.children(node)) == 0

    def __contains__(self, node: NodeT) -> bool:
        """Return *True* if *node* is a node in the tree."""
        return node in self._child_to_parent

    def add_node(self, node: NodeT, parent: NodeT) -> Tree[NodeT]:
        """
        Returns a :class:`Tree` with added node *node* having a parent
        *parent*.
        """
        if node in self:
            raise ValueError(f"'{node}' already present in tree.")

        siblings = self._parent_to_children[parent]

        _parent_to_children_mut = self._parent_to_children.mutate()
        _parent_to_children_mut[parent] = (*siblings, node)
        _parent_to_children_mut[node] = ()

        return Tree(_parent_to_children_mut.finish(),
                    self._child_to_parent.set(node, parent))

    def replace_node(self, node: NodeT, new_node: NodeT) -> Tree[NodeT]:
        """
        Returns a copy of *self* with *node* replaced with *new_node*.
        """
        if node not in self:
            raise ValueError(f"'{node}' not present in tree.")

        if new_node in self:
            raise ValueError(f"cannot replace with '{new_node}', as its already a part"
                             " of the tree.")

        parent = self.parent(node)
        children = self.children(node)

        # {{{ update child to parent

        child_to_parent_mut = dict(self._child_to_parent)
        del child_to_parent_mut[node]
        child_to_parent_mut[new_node] = parent

        for child in children:
            child_to_parent_mut[child] = new_node

        # }}}

        # {{{ update parent_to_children

        parent_to_children_mut = dict(self._parent_to_children)
        del parent_to_children_mut[node]
        parent_to_children_mut[new_node] = children

        if parent is not None:
            # update the child's name in the parent's children
            parent_to_children_mut[parent] = (
                            *(frozenset(self.children(parent)) - frozenset([node])),
                            new_node,)

        # }}}

        return Tree(immutabledict(parent_to_children_mut),
                    immutabledict(child_to_parent_mut))

    def move_node(self, node: NodeT, new_parent: NodeT | None) -> Tree[NodeT]:
        """
        Returns a copy of *self* with node *node* as a child of *new_parent*.
        """
        if node not in self:
            raise ValueError(f"'{node}' not a part of the tree => cannot move.")

        if self.is_root(node):
            if new_parent is None:
                return self
            else:
                raise ValueError("Moving root not allowed.")

        if new_parent is None:
            raise ValueError("Making multiple roots not allowed")

        if new_parent not in self:
            raise ValueError(f"Cannot move to '{new_parent}' as it's not in tree.")

        parent = self.parent(node)
        assert parent is not None  # parent=root handled as a special case
        siblings = self.children(parent)
        parents_new_children = tuple(frozenset(siblings) - frozenset([node]))
        new_parents_children = (*self.children(new_parent), node)

        _parent_to_children_mut = self._parent_to_children.mutate()
        _parent_to_children_mut[parent] = parents_new_children
        _parent_to_children_mut[new_parent] = new_parents_children

        return Tree(_parent_to_children_mut.finish(),
                    self._child_to_parent.set(node, new_parent))

    def __str__(self) -> str:
        """
        Stringifies the tree by using the box-drawing unicode characters.

        .. doctest::

            >>> from loopy.schedule.tree import Tree
            >>> tree = (Tree.from_root("Root")
            ...         .add_node("A", "Root")
            ...         .add_node("B", "Root")
            ...         .add_node("D", "B")
            ...         .add_node("E", "B")
            ...         .add_node("C", "A"))

            >>> print(tree)
            Root
            ├── A
            │   └── C
            └── B
                ├── D
                └── E
        """
        def rec(node: NodeT) -> list[str]:
            children_result = [rec(c) for c in self.children(node)]

            def post_process_non_last_child(children: Sequence[str]) -> list[str]:
                return ["├── " + children[0]] + [f"│   {c}" for c in children[1:]]

            def post_process_last_child(children: Sequence[str]) -> list[str]:
                return ["└── " + children[0]] + [f"    {c}" for c in children[1:]]

            children_result = ([post_process_non_last_child(c)
                                for c in children_result[:-1]]
                            + [post_process_last_child(c)
                                for c in children_result[-1:]])
            return [str(node), *reduce(operator.iadd, children_result, [])]

        return "\n".join(rec(self.root))

    def nodes(self) -> Iterator[NodeT]:
        return iter(self._child_to_parent.keys())

# }}}

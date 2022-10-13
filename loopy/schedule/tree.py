__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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

from immutables import Map
from typing import Generic, Hashable, Tuple, TypeVar, Iterator, Optional, List
from dataclasses import dataclass

# {{{ tree data structure

NodeT = TypeVar("NodeT", bound=Hashable)


@dataclass(frozen=True)
class Tree(Generic[NodeT]):
    """
    An immutable n-ary tree containing nodes of type :class:`NodeT`.

    .. automethod:: ancestors
    .. automethod:: parent
    .. automethod:: children
    .. automethod:: add_node
    .. automethod:: depth
    .. automethod:: replace_node
    .. automethod:: move_node

    .. note::

       Almost all the operations are implemented recursively. NOT suitable for
       deep trees. At the very least if the Python implementation is CPython
       this allocates a new stack frame for each iteration of the operation.
    """
    _parent_to_children: Map[NodeT, Tuple[NodeT, ...]]
    _child_to_parent: Map[NodeT, Optional[NodeT]]

    @staticmethod
    def from_root(root: NodeT) -> "Tree[NodeT]":
        return Tree(Map({root: tuple()}),
                    Map({root: None}))

    @property
    def root(self) -> NodeT:
        guess = set(self._child_to_parent).pop()
        parent_of_guess = self.parent(guess)
        while parent_of_guess is not None:
            guess = parent_of_guess
            parent_of_guess = self.parent(guess)

        return guess

    def ancestors(self, node: NodeT) -> Tuple[NodeT, ...]:
        """
        Returns a :class:`tuple` of nodes that are ancestors of *node*.
        """
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not in tree.")

        if self.is_root(node):
            # => root
            return tuple()

        parent = self._child_to_parent[node]
        assert parent is not None

        return (parent,) + self.ancestors(parent)

    def parent(self, node: NodeT) -> Optional[NodeT]:
        """
        Returns the parent of *node*.
        """
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not in tree.")

        return self._child_to_parent[node]

    def children(self, node: NodeT) -> Tuple[NodeT, ...]:
        """
        Returns the children of *node*.
        """
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not in tree.")

        return self._parent_to_children[node]

    def depth(self, node: NodeT) -> int:
        """
        Returns the depth of *node*.
        """
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not in tree.")

        if self.is_root(node):
            # => None
            return 0

        parent_of_node = self.parent(node)
        assert parent_of_node is not None

        return 1 + self.depth(parent_of_node)

    def is_root(self, node: NodeT) -> bool:
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not in tree.")

        return self.parent(node) is None

    def is_leaf(self, node: NodeT) -> bool:
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not in tree.")

        return len(self.children(node)) == 0

    def is_a_node(self, node: NodeT) -> bool:
        return node in self._child_to_parent

    def add_node(self, node: NodeT, parent: NodeT) -> "Tree[NodeT]":
        """
        Returns a :class:`Tree` with added node *node* having a parent
        *parent*.
        """
        if self.is_a_node(node):
            raise ValueError(f"'{node}' already present in tree.")

        siblings = self._parent_to_children[parent]

        return Tree((self._parent_to_children
                     .set(parent, siblings + (node,))
                     .set(node, tuple())),
                    self._child_to_parent.set(node, parent))

    def replace_node(self, node: NodeT, new_id: NodeT) -> "Tree[NodeT]":
        """
        Returns a copy of *self* with *node* replaced with *new_id*.
        """
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not present in tree.")

        if self.is_a_node(new_id):
            raise ValueError(f"cannot rename to '{new_id}', as its already a part"
                             " of the tree.")

        parent = self.parent(node)
        children = self.children(node)

        # {{{ update child to parent

        new_child_to_parent = (self._child_to_parent.delete(node)
                               .set(new_id, parent))

        for child in children:
            new_child_to_parent = (new_child_to_parent
                                   .set(child, new_id))

        # }}}

        # {{{ update parent_to_children

        new_parent_to_children = (self._parent_to_children
                                  .delete(node)
                                  .set(new_id, self.children(node)))

        if parent is not None:
            # update the child's name in the parent's children
            new_parent_to_children = (new_parent_to_children
                                      .delete(parent)
                                      .set(parent, tuple(
                                                    frozenset(self.children(parent))
                                                    - frozenset([node]))
                                                    + (new_id,)))

        # }}}

        return Tree(new_parent_to_children,
                    new_child_to_parent)

    def move_node(self, node: NodeT, new_parent: Optional[NodeT]) -> "Tree[NodeT]":
        """
        Returns a copy of *self* with node *node* as a child of *new_parent*.
        """
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not a part of the tree => cannot move.")

        if self.is_root(node):
            if new_parent is None:
                return self
            else:
                raise ValueError("Moving root not allowed.")

        if new_parent is None:
            raise ValueError("Making multiple roots not allowed")

        if not self.is_a_node(new_parent):
            raise ValueError(f"Cannot move to '{new_parent}' as it's not in tree.")

        parent = self.parent(node)
        assert parent is not None  # parent=root handled as a special case
        siblings = self.children(parent)
        parents_new_children = tuple(frozenset(siblings) - frozenset([node]))
        new_parents_children = self.children(new_parent) + (node,)

        new_child_to_parent = self._child_to_parent.set(node, new_parent)
        new_parent_to_children = (self._parent_to_children
                                  .set(parent, parents_new_children)
                                  .set(new_parent, new_parents_children))

        return Tree(new_parent_to_children,
                    new_child_to_parent)

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
        def rec(node: NodeT) -> List[str]:
            children_result = [rec(c) for c in self.children(node)]

            def post_process_non_last_child(child):
                return ["├── " + child[0]] + [f"│   {c}" for c in child[1:]]

            def post_process_last_child(child):
                return ["└── " + child[0]] + [f"    {c}" for c in child[1:]]

            children_result = ([post_process_non_last_child(c)
                                for c in children_result[:-1]]
                            + [post_process_last_child(c)
                                for c in children_result[-1:]])
            return [str(node)] + sum(children_result, start=[])

        return "\n".join(rec(self.root))

    def nodes(self) -> Iterator[NodeT]:
        return iter(self._child_to_parent.keys())

# }}}

# {{{ tree data structure

T = TypeVar("T")


@dataclass(frozen=True)
class Tree(Generic[T]):
    """
    An immutable tree implementation.
    .. automethod:: ancestors
    .. automethod:: parent
    .. automethod:: children
    .. automethod:: create_node
    .. automethod:: depth
    .. automethod:: rename_node
    .. automethod:: move_node
    .. note::
       Almost all the operations are implemented recursively. NOT suitable for
       deep trees. At the very least if the Python implementation is CPython
       this allocates a new stack frame for each iteration of the operation.
    """
    _parent_to_children: Map[T, FrozenSet[T]]
    _child_to_parent: Map[T, OptionalT[T]]

    @staticmethod
    def from_root(root: T):
        return Tree(Map({root: frozenset()}),
                    Map({root: None}))

    @property
    def root(self) -> T:
        guess = set(self._child_to_parent).pop()
        parent_of_guess = self.parent(guess)
        while parent_of_guess is not None:
            guess = parent_of_guess
            parent_of_guess = self.parent(guess)

        return guess

    def ancestors(self, node: T) -> FrozenSet[T]:
        """
        Returns a :class:`frozenset` of nodes that are ancestors of *node*.
        """
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not in tree.")

        if self.is_root(node):
            # => root
            return frozenset()

        parent = self._child_to_parent[node]
        assert parent is not None

        return frozenset([parent]) | self.ancestors(parent)

    def parent(self, node: T) -> OptionalT[T]:
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not in tree.")

        return self._child_to_parent[node]

    def children(self, node: T) -> FrozenSet[T]:
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not in tree.")

        return self._parent_to_children[node]

    def depth(self, node: T) -> int:
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not in tree.")

        if self.is_root(node):
            # => None
            return 0

        parent_of_node = self.parent(node)
        assert parent_of_node is not None

        return 1 + self.depth(parent_of_node)

    def is_root(self, node: T) -> bool:
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not in tree.")

        return self.parent(node) is None

    def is_leaf(self, node: T) -> bool:
        if not self.is_a_node(node):
            raise ValueError(f"'{node}' not in tree.")

        return len(self.children(node)) == 0

    def is_a_node(self, node: T) -> bool:
        return node in self._child_to_parent

    def add_node(self, node: T, parent: T) -> "Tree[T]":
        """
        Returns a :class:`Tree` with added node *node* having a parent
        *parent*.
        """
        if self.is_a_node(node):
            raise ValueError(f"'{node}' already present in tree.")

        siblings = self._parent_to_children[parent]

        return Tree((self._parent_to_children
                     .set(parent, siblings | frozenset([node]))
                     .set(node, frozenset())),
                    self._child_to_parent.set(node, parent))

    def rename_node(self, node: T, new_id: T) -> "Tree[T]":
        """
        Returns a copy of *self* with *node* renamed to *new_id*.
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
                                      .set(parent, ((self.children(parent)
                                                    - frozenset([node]))
                                                    | frozenset([new_id]))))

        # }}}

        return Tree(new_parent_to_children,
                    new_child_to_parent)

    def move_node(self, node: T, new_parent: OptionalT[T]) -> "Tree[T]":
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
        parents_new_children = siblings - frozenset([node])
        new_parents_children = self.children(new_parent) | frozenset([node])

        new_child_to_parent = self._child_to_parent.set(node, new_parent)
        new_parent_to_children = (self._parent_to_children
                                  .set(parent, parents_new_children)
                                  .set(new_parent, new_parents_children))

        return Tree(new_parent_to_children,
                    new_child_to_parent)

    def __str__(self) -> str:
        """
        Stringifies the tree by using the box-drawing unicode characters.
        ::
            >>> from loopy.tools import Tree
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
        def rec(node):
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

    def nodes(self) -> Iterator[T]:
        return iter(self._child_to_parent.keys())

# }}}

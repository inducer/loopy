__copyright__ = "Copyright (C) 2022 University of Illinois Board of Trustees"

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

from pyopencl.tools import (  # noqa: F401
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

from loopy.schedule.tree import Tree


def test_tree_simple():
    tree = Tree.from_root("")

    tree = tree.add_node("bar", parent="")
    tree = tree.add_node("baz", parent="bar")

    assert tree.depth("") == 0
    assert tree.depth("bar") == 1
    assert tree.depth("baz") == 2

    assert "" in tree
    assert "bar" in tree
    assert "baz" in tree
    assert "foo" not in tree

    tree = tree.replace_node("bar", "foo")
    assert "bar" not in tree
    assert "foo" in tree

    tree = tree.move_node("baz", new_parent="")
    assert tree.depth("baz") == 1

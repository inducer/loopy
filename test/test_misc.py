from __future__ import division, absolute_import, print_function

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

import six  # noqa
import pytest
from six.moves import range

import sys

import logging
logger = logging.getLogger(__name__)


def test_compute_sccs():
    from loopy.tools import compute_sccs
    import random

    rng = random.Random(0)

    def generate_random_graph(nnodes):
        graph = dict((i, set()) for i in range(nnodes))
        for i in range(nnodes):
            for j in range(nnodes):
                # Edge probability 2/n: Generates decently interesting inputs.
                if rng.randint(0, nnodes - 1) <= 1:
                    graph[i].add(j)
        return graph

    def verify_sccs(graph, sccs):
        visited = set()

        def visit(node):
            if node in visited:
                return []
            else:
                visited.add(node)
                result = []
                for child in graph[node]:
                    result = result + visit(child)
                return result + [node]

        for scc in sccs:
            scc = set(scc)
            assert not scc & visited
            # Check that starting from each element of the SCC results
            # in the same set of reachable nodes.
            for scc_root in scc:
                visited.difference_update(scc)
                result = visit(scc_root)
                assert set(result) == scc, (set(result), scc)

    for nnodes in range(10, 20):
        for i in range(40):
            graph = generate_random_graph(nnodes)
            verify_sccs(graph, compute_sccs(graph))


def test_SetTrie():
    from loopy.kernel.tools import SetTrie

    s = SetTrie()
    s.add_or_update(set([1, 2, 3]))
    s.add_or_update(set([4, 2, 1]))
    s.add_or_update(set([1, 5]))

    result = []
    s.descend(lambda prefix: result.extend(prefix))
    assert result == [1, 2, 3, 4, 5]

    with pytest.raises(ValueError):
        s.add_or_update(set([1, 4]))


class PickleDetector(object):
    """Contains a class attribute which flags if any instance was unpickled.
    """

    @classmethod
    def reset(cls):
        cls.instance_unpickled = False

    def __getstate__(self):
        return {"state": self.state}

    def __setstate__(self, state):
        self.__class__.instance_unpickled = True
        self.state = state["state"]


class PickleDetectorForLazilyUnpicklingDict(PickleDetector):
    instance_unpickled = False

    def __init__(self):
        self.state = None


def test_LazilyUnpicklingDict():
    from loopy.tools import LazilyUnpicklingDict

    cls = PickleDetectorForLazilyUnpicklingDict
    mapping = LazilyUnpicklingDict({0: cls()})

    assert not cls.instance_unpickled

    from pickle import loads, dumps

    pickled_mapping = dumps(mapping)

    # {{{ test lazy loading

    mapping = loads(pickled_mapping)
    assert not cls.instance_unpickled
    list(mapping.keys())
    assert not cls.instance_unpickled
    assert isinstance(mapping[0], cls)
    assert cls.instance_unpickled

    # }}}

    # {{{ conversion

    cls.reset()
    mapping = loads(pickled_mapping)
    dict(mapping)
    assert cls.instance_unpickled

    # }}}

    # {{{ test multi round trip

    mapping = loads(dumps(loads(pickled_mapping)))
    assert isinstance(mapping[0], cls)

    # }}}

    # {{{ test empty map

    mapping = LazilyUnpicklingDict({})
    mapping = loads(dumps(mapping))
    assert len(mapping) == 0

    # }}}


class PickleDetectorForLazilyUnpicklingList(PickleDetector):
    instance_unpickled = False

    def __init__(self):
        self.state = None


def test_LazilyUnpicklingList():
    from loopy.tools import LazilyUnpicklingList

    cls = PickleDetectorForLazilyUnpicklingList
    lst = LazilyUnpicklingList([cls()])
    assert not cls.instance_unpickled

    from pickle import loads, dumps
    pickled_lst = dumps(lst)

    # {{{ test lazy loading

    lst = loads(pickled_lst)
    assert not cls.instance_unpickled
    assert isinstance(lst[0], cls)
    assert cls.instance_unpickled

    # }}}

    # {{{ conversion

    cls.reset()
    lst = loads(pickled_lst)
    list(lst)
    assert cls.instance_unpickled

    # }}}

    # {{{ test multi round trip

    lst = loads(dumps(loads(dumps(lst))))
    assert isinstance(lst[0], cls)

    # }}}

    # {{{ test empty list

    lst = LazilyUnpicklingList([])
    lst = loads(dumps(lst))
    assert len(lst) == 0

    # }}}


class PickleDetectorForLazilyUnpicklingListWithEqAndPersistentHashing(
        PickleDetector):
    instance_unpickled = False

    def __init__(self, comparison_key):
        self.state = comparison_key

    def __repr__(self):
        return repr(self.state)

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.rec(key_hash, repr(self))


def test_LazilyUnpicklingListWithEqAndPersistentHashing():
    from loopy.tools import LazilyUnpicklingListWithEqAndPersistentHashing

    cls = PickleDetectorForLazilyUnpicklingListWithEqAndPersistentHashing
    from pickle import loads, dumps

    # {{{ test comparison of a pair of lazy lists

    lst0 = LazilyUnpicklingListWithEqAndPersistentHashing(
            [cls(0), cls(1)],
            eq_key_getter=repr,
            persistent_hash_key_getter=repr)
    lst1 = LazilyUnpicklingListWithEqAndPersistentHashing(
            [cls(0), cls(1)],
            eq_key_getter=repr,
            persistent_hash_key_getter=repr)

    assert not cls.instance_unpickled

    assert lst0 == lst1
    assert not cls.instance_unpickled

    lst0 = loads(dumps(lst0))
    lst1 = loads(dumps(lst1))

    assert lst0 == lst1
    assert not cls.instance_unpickled

    lst0.append(cls(3))
    lst1.append(cls(2))

    assert lst0 != lst1

    # }}}

    # {{{ comparison with plain lists

    lst = [cls(0), cls(1), cls(3)]

    assert lst == lst0
    assert lst0 == lst
    assert not cls.instance_unpickled

    # }}}

    # {{{ persistent hashing

    from loopy.tools import LoopyKeyBuilder
    kb = LoopyKeyBuilder()

    assert kb(lst0) == kb(lst)
    assert not cls.instance_unpickled

    # }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: foldmethod=marker

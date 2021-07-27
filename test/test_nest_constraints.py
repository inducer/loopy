__copyright__ = "Copyright (C) 2021 James Stevens"

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

import sys
import loopy as lp
import numpy as np
import pyopencl as cl
from loopy import preprocess_kernel, get_one_linearized_kernel

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

__all__ = [
        "pytest_generate_tests",
        "cl"  # "cl.create_some_context"
        ]


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


# {{{ Helper functions

def _process_and_linearize(prog, knl_name="loopy_kernel"):
    # Return linearized kernel
    proc_prog = preprocess_kernel(prog)
    lin_prog = get_one_linearized_kernel(
        proc_prog[knl_name], proc_prog.callables_table)
    return lin_prog


def _linearize_and_get_nestings(prog, knl_name="loopy_kernel"):
    from loopy.transform.iname import get_iname_nestings
    lin_knl = _process_and_linearize(prog, knl_name)
    return get_iname_nestings(lin_knl.linearization)

# }}}


# {{{ test_loop_constraint_string_parsing

def test_loop_constraint_string_parsing():
    ref_knl = lp.make_kernel(
            "{ [g,h,i,j,k,xx]: 0<=g,h,i,j,k,xx<n }",
            "out[g,h,i,j,k,xx] = 2*a[g,h,i,j,k,xx]",
            assumptions="n >= 1",
            )

    try:
        lp.constrain_loop_nesting(ref_knl, "{g,h,k},{j,i}")
        raise AssertionError()
    except ValueError as e:
        assert "Unrecognized character(s)" in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, "{g,h,i,k},{j}")
        raise AssertionError()
    except ValueError as e:
        assert "Unrecognized character(s)" in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, "{g,{h,i,k}")
        raise AssertionError()
    except ValueError as e:
        assert "Unrecognized character(s)" in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, "{g,~h,i,k}")
        raise AssertionError()
    except ValueError as e:
        assert "Unrecognized character(s)" in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, "{g,#h,i,k}")
        raise AssertionError()
    except ValueError as e:
        assert "Unrecognized character(s)" in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, ("{g,{h}", "i,k"))
        raise AssertionError()
    except ValueError as e:
        assert "Unrecognized character(s)" in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, ("{g,~h}", "i,k"))
        raise AssertionError()
    except ValueError as e:
        assert "Unrecognized character(s)" in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, ("k", "~{g,h}", "{g,h}"))
        raise AssertionError()
    except ValueError as e:
        assert "Complement (~) not allowed" in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, ("k", "{i,j,k}", "{g,h}"))
        raise AssertionError()
    except ValueError as e:
        assert "contains cycle" in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, must_not_nest=("~j,i", "{j,i}"))
        raise AssertionError()
    except ValueError as e:
        assert ("Complements of sets containing multiple inames "
            "must enclose inames in braces") in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, must_nest=("k", "{h}", "{j,i,}"))
        raise AssertionError()
    except ValueError as e:
        assert ("Found 2 inames but expected 3") in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, must_nest=("k", "{h}", "{j, x x, i}"))
        raise AssertionError()
    except ValueError as e:
        assert ("Found 4 inames but expected 3") in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, must_nest="{h}}")
        raise AssertionError()
    except ValueError as e:
        assert (
            "Unrecognized character(s) ['{', '}', '}'] in nest string {h}}"
            ) in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, must_nest="{h i j,,}")
        raise AssertionError()
    except ValueError as e:
        assert(
            "Unrecognized character(s) [\'{\', \'}\'] in nest string {h i j,,}"
            ) in str(e)

    try:
        lp.constrain_loop_nesting(ref_knl, must_nest=("{h}}", "i"))
        raise AssertionError()
    except ValueError as e:
        assert (
            "Unrecognized character(s) [\'}\'] in nest string h}"
            ) in str(e)

    # Valid syntax
    lp.constrain_loop_nesting(ref_knl, must_not_nest=("~{j,i}", "{j,i}"))
    lp.constrain_loop_nesting(ref_knl, must_not_nest=("{h}", "{j,i}"))
    lp.constrain_loop_nesting(ref_knl, must_not_nest=("h", "{j,i}"))
    lp.constrain_loop_nesting(ref_knl, must_nest=("k", "{h}", "{j,i}"))
    lp.constrain_loop_nesting(ref_knl, must_nest=("k", "h", "{j,i}"))
    lp.constrain_loop_nesting(ref_knl, must_not_nest="~j,j")
    lp.constrain_loop_nesting(ref_knl, must_nest="k,h,j")

    # Handling spaces
    knl = lp.constrain_loop_nesting(
        ref_knl, must_nest=("k", "{h }", " { j , i } "))["loopy_kernel"]
    assert list(knl.loop_nest_constraints.must_nest)[0][0].inames == set("k")
    assert list(knl.loop_nest_constraints.must_nest)[0][1].inames == set("h")
    assert list(knl.loop_nest_constraints.must_nest)[0][2].inames == set(["j", "i"])

    try:
        knl = lp.constrain_loop_nesting(ref_knl, ("j", "{}"))
        raise AssertionError()
    except ValueError as e:
        assert "Found 0 inames" in str(e)

    try:
        knl = lp.constrain_loop_nesting(ref_knl, ("j", ""))
        raise AssertionError()
    except ValueError as e:
        assert "Found 0 inames" in str(e)

# }}}


# {{{ test_loop_nest_constraints_satisfied

def test_loop_nest_constraints_satisfied():
    from loopy.transform.iname import (
        process_loop_nest_specification,
        loop_nest_constraints_satisfied,
    )

    all_inames = frozenset(["g", "h", "i", "j", "k"])

    must_nest_constraints = [
        process_loop_nest_specification(
            nesting=("{g,h}", "{i,j,k}"),
            complement_sets_allowed=True),
        ]
    must_not_nest_constraints = [
        process_loop_nest_specification(
            nesting="k,~k",
            complement_sets_allowed=True),
        ]

    loop_nests = set([("g", "h", "i", "j", "k"), ])
    valid = loop_nest_constraints_satisfied(
        loop_nests, must_nest_constraints, must_not_nest_constraints, all_inames)
    assert valid

    loop_nests = set([("g", "i", "h", "j", "k"), ])
    valid = loop_nest_constraints_satisfied(
        loop_nests, must_nest_constraints, must_not_nest_constraints, all_inames)
    assert not valid

    loop_nests = set([("g", "h", "i", "k", "j"), ])
    valid = loop_nest_constraints_satisfied(
        loop_nests, must_nest_constraints, must_not_nest_constraints, all_inames)
    assert not valid

    # now j, k must be innermost
    must_not_nest_constraints = [
        process_loop_nest_specification(("{k,j}", "~{k,j}")),
        ]
    loop_nests = set([("g", "i", "h", "j", "k"), ])
    valid = loop_nest_constraints_satisfied(
        loop_nests, None, must_not_nest_constraints, all_inames)
    assert valid

    loop_nests = set([("g", "h", "i", "k", "j"), ])
    valid = loop_nest_constraints_satisfied(
        loop_nests, None, must_not_nest_constraints, all_inames)
    assert valid

    loop_nests = set([("g", "i", "j", "h", "k"), ])
    valid = loop_nest_constraints_satisfied(
        loop_nests, None, must_not_nest_constraints, all_inames)
    assert not valid

    loop_nests = set([("g", "h", "j", "k", "i"), ])
    valid = loop_nest_constraints_satisfied(
        loop_nests, None, must_not_nest_constraints, all_inames)
    assert not valid

    loop_nests = set([("j", "k"), ])
    valid = loop_nest_constraints_satisfied(
        loop_nests, None, must_not_nest_constraints, all_inames)
    assert valid

    loop_nests = set([("g", "k"), ])  # j not present
    valid = loop_nest_constraints_satisfied(
        loop_nests, None, must_not_nest_constraints, all_inames)
    assert valid

    loop_nests = set([("g", "i"), ])  # j, k not present
    valid = loop_nest_constraints_satisfied(
        loop_nests, None, must_not_nest_constraints, all_inames)
    assert valid

    loop_nests = set([("k",), ])  # only k present
    valid = loop_nest_constraints_satisfied(
        loop_nests, None, must_not_nest_constraints, all_inames)
    assert valid

    loop_nests = set([("i",), ])
    valid = loop_nest_constraints_satisfied(
        loop_nests, None, must_not_nest_constraints, all_inames)
    assert valid

# }}}


# {{{ test_adding_multiple_nest_constraints_to_knl

def test_adding_multiple_nest_constraints_to_knl():
    ref_knl = lp.make_kernel(
            "{ [g,h,i,j,k,x,y,z]: 0<=g,h,i,j,k,x,y,z<n }",
            """
            out[g,h,i,j,k] = 2*a[g,h,i,j,k]
            for x,y
                out2[x,y] = 2*a2[x,y]
                for z
                    out3[x,y,z] = 2*a3[x,y,z]
                end
            end
            """,
            assumptions="n >= 1",
            )
    ref_knl = lp.add_and_infer_dtypes(ref_knl, {"a,a2,a3": np.dtype(np.float32)})
    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl, must_not_nest=("{k,i}", "~{k,i}"))
    knl = lp.constrain_loop_nesting(
        knl, must_nest=("g", "h,i"))
    knl = lp.constrain_loop_nesting(
        knl, must_nest=("g", "j", "k"))
    knl = lp.constrain_loop_nesting(
        knl, must_nest=("g", "j", "h"))
    knl = lp.constrain_loop_nesting(
        knl, must_nest=("i", "k"))
    knl = lp.constrain_loop_nesting(
        knl, must_nest=("x", "y"))

    must_nest_knl = knl["loopy_kernel"].loop_nest_constraints.must_nest
    from loopy.transform.iname import UnexpandedInameSet
    must_nest_expected = set([
        (UnexpandedInameSet(set(["g"], )), UnexpandedInameSet(set(["h", "i"], ))),
        (UnexpandedInameSet(set(["g"], )), UnexpandedInameSet(set(["j"], )),
            UnexpandedInameSet(set(["k"], ))),
        (UnexpandedInameSet(set(["g"], )), UnexpandedInameSet(set(["j"], )),
            UnexpandedInameSet(set(["h"], ))),
        (UnexpandedInameSet(set(["i"], )), UnexpandedInameSet(set(["k"], ))),
        (UnexpandedInameSet(set(["x"], )), UnexpandedInameSet(set(["y"], ))),
        ])
    assert must_nest_knl == must_nest_expected

    must_not_nest_knl = knl["loopy_kernel"].loop_nest_constraints.must_not_nest
    must_not_nest_expected = set([
        (UnexpandedInameSet(set(["k", "i"], )), UnexpandedInameSet(set(["k", "i"], ),
            complement=True)),
        ])
    assert must_not_nest_knl == must_not_nest_expected

# }}}


# {{{ test_incompatible_nest_constraints

def test_incompatible_nest_constraints():
    ref_knl = lp.make_kernel(
            "{ [g,h,i,j,k,x,y,z]: 0<=g,h,i,j,k,x,y,z<n }",
            """
            out[g,h,i,j,k] = 2*a[g,h,i,j,k]
            for x,y
                out2[x,y] = 2*a2[x,y]
                for z
                    out3[x,y,z] = 2*a3[x,y,z]
                end
            end
            """,
            assumptions="n >= 1",
            )
    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl, must_not_nest=("{k,i}", "~{k,i}"))

    try:
        knl = lp.constrain_loop_nesting(
            knl, must_nest=("k", "h"))  # (should fail)
        raise AssertionError()
    except ValueError as e:
        assert "Nest constraint conflict detected" in str(e)

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl, must_nest=("g", "j", "k"))

    try:
        knl = lp.constrain_loop_nesting(
            knl, must_nest=("j", "g"))  # (should fail)
        raise AssertionError()
    except ValueError as e:
        assert "Nest constraint cycle detected" in str(e)

# }}}


# {{{ test_vec_innermost:

def test_vec_innermost():

    def is_innermost(iname, lin_items):
        from loopy.schedule import (EnterLoop, LeaveLoop)

        # find EnterLoop(iname) in linearization
        enter_iname_idx = None
        for i, lin_item in enumerate(lin_items):
            if isinstance(lin_item, EnterLoop) and (
                    lin_item.iname == iname):
                enter_iname_idx = i
                break
        else:
            # iname not found
            return False

        # now go through remaining linearization items after EnterLoop(iname)
        for lin_item in lin_items[enter_iname_idx+1:]:
            if isinstance(lin_item, LeaveLoop):
                # Break as soon as we find a LeaveLoop
                # If this happens before we find an EnterLoop, iname is innermost
                break
            elif isinstance(lin_item, EnterLoop):
                # we found an EnterLoop inside iname
                return False

        return True

    ref_knl = lp.make_kernel(
            "{ [g,h,i,j,k]: 0<=g,h,i,j,k<n }",
            """
            out[g,h,i,j,k] = 2*a[g,h,i,j,k]
            """,
            assumptions="n >= 1",
            )
    ref_knl = lp.add_and_infer_dtypes(ref_knl, {"a": np.dtype(np.float32)})

    knl = ref_knl
    knl = lp.tag_inames(knl, {"h": "vec"})
    lin_knl = _process_and_linearize(knl)
    assert is_innermost("h", lin_knl.linearization)

    knl = ref_knl
    knl = lp.tag_inames(knl, {"h": "vec", "g": "l.1", "i": "l.0"})
    lin_knl = _process_and_linearize(knl)
    assert is_innermost("h", lin_knl.linearization)

    knl = ref_knl
    knl = lp.tag_inames(
        knl, {"h": "vec", "g": "l.1", "i": "l.0", "k": "unr"})
    lin_knl = _process_and_linearize(knl)
    assert is_innermost("h", lin_knl.linearization)

    knl = ref_knl
    knl = lp.tag_inames(knl, {"h": "vec"})
    knl = lp.constrain_loop_nesting(knl, must_nest=("k", "i"))
    lin_knl = _process_and_linearize(knl)
    assert is_innermost("h", lin_knl.linearization)
    lp.set_caching_enabled(True)

    # try adding a must_nest constraint that conflicts with a vec tag
    knl = ref_knl
    knl = lp.tag_inames(knl, {"h": "vec"})
    try:
        lp.constrain_loop_nesting(knl, must_nest=("{g,h,i,j}", "{k}"))
        raise AssertionError()
    except ValueError as e:
        assert (
            "iname h tagged with ConcurrentTag, "
            "cannot use iname in must-nest constraint" in str(e))

    # try adding a vec tag that conflicts with a must_nest constraint
    # TODO uncomment after implemented in tag_inames
    """
    knl = ref_knl
    knl = lp.constrain_loop_nesting(knl, must_nest=("{g,h,i,j}", "{k}"))
    try:
        lp.tag_inames(knl, {"h": "vec"})
        raise AssertionError()
    except ValueError as e:
        assert (
            "cannot tag 'h' as concurrent--iname involved "
            "in must-nest constraint" in str(e))
    """

    # TODO try adding a vec tag forcing h to be innermost, but
    # also add a must-not-nest constraint preventing h
    # from nesting inside j

# }}}


# {{{ test_linearization_with_nesting_constraints

def test_linearization_with_nesting_constraints():

    def loop_order(lin_items):
        from loopy.schedule import EnterLoop
        order = []
        for lin_item in lin_items:
            if isinstance(lin_item, EnterLoop):
                order.append(lin_item.iname)
        return order

    ref_knl = lp.make_kernel(
            "{ [g,h,i,j,k]: 0<=g,h,i,j,k<n }",
            """
            out[g,h,i,j,k] = 2*a[g,h,i,j,k]
            """,
            assumptions="n >= 1",
            )
    ref_knl = lp.add_and_infer_dtypes(ref_knl, {"a": np.dtype(np.float32)})

    # must_nest constraints
    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "j", "h", "k", "g"),
        )
    lin_knl = _process_and_linearize(knl)
    assert loop_order(lin_knl.linearization) == ["i", "j", "h", "k", "g"]

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("k", "{g, h, i, j}"),
        )
    lin_knl = _process_and_linearize(knl)
    assert loop_order(lin_knl.linearization)[0] == "k"

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("{g, h, i, j}", "k"),
        )
    lin_knl = _process_and_linearize(knl)
    assert loop_order(lin_knl.linearization)[-1] == "k"

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("{g, h, i}", "{j, k}"),
        )
    lin_knl = _process_and_linearize(knl)
    assert set(loop_order(lin_knl.linearization)[-2:]) == set(["j", "k"])

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("{g, h, i}", "{j, k}"),
        )
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "{g, h}"),
        )
    lin_knl = _process_and_linearize(knl)
    assert set(loop_order(lin_knl.linearization)[3:]) == set(["j", "k"])
    assert set(loop_order(lin_knl.linearization)[1:3]) == set(["g", "h"])
    assert loop_order(lin_knl.linearization)[0] == "i"

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "{g, h}", "{j, k}"),
        )
    lin_knl = _process_and_linearize(knl)
    assert set(loop_order(lin_knl.linearization)[3:]) == set(["j", "k"])
    assert set(loop_order(lin_knl.linearization)[1:3]) == set(["g", "h"])
    assert loop_order(lin_knl.linearization)[0] == "i"

    # must_not_nest constraints

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_not_nest=("~k", "k"),
        )
    lin_knl = _process_and_linearize(knl)
    assert loop_order(lin_knl.linearization)[0] == "k"

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_not_nest=("k", "~k"),
        )
    lin_knl = _process_and_linearize(knl)
    assert loop_order(lin_knl.linearization)[-1] == "k"

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_not_nest=("{j, k}", "~{j, k}"),
        )
    lin_knl = _process_and_linearize(knl)
    assert set(loop_order(lin_knl.linearization)[-2:]) == set(["j", "k"])

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_not_nest=("{j, k}", "~{j, k}"),
        )
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "{g, h}"),
        )
    lin_knl = _process_and_linearize(knl)
    assert set(loop_order(lin_knl.linearization)[3:]) == set(["j", "k"])
    assert set(loop_order(lin_knl.linearization)[1:3]) == set(["g", "h"])
    assert loop_order(lin_knl.linearization)[0] == "i"

    # must_nest + must_not_nest
    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("{g, h, i}", "{j, k}"),
        must_not_nest=("i", "{g, h}"),
        )
    lin_knl = _process_and_linearize(knl)
    assert set(loop_order(lin_knl.linearization)[3:]) == set(["j", "k"])
    assert set(loop_order(lin_knl.linearization)[0:2]) == set(["g", "h"])
    assert loop_order(lin_knl.linearization)[2] == "i"

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_not_nest=("i", "~i"),
        )
    lin_knl = _process_and_linearize(knl)
    assert loop_order(lin_knl.linearization)[-1] == "i"

    # contradictory must_not_nest

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_not_nest=("~k", "k"),
        )
    knl = lp.constrain_loop_nesting(
        knl,
        must_not_nest=("k", "h"),
        )

    try:
        proc_prog = preprocess_kernel(knl)
        get_one_linearized_kernel(
            proc_prog["loopy_kernel"],
            proc_prog.callables_table,
            debug_args={"interactive": False},
            )
        raise AssertionError()
    except RuntimeError as e:
        assert "no valid schedules found" in str(e)

# }}}


# {{{ test constraint updating during transformation


# {{{ test_constraint_updating_split_iname

def test_constraint_updating_split_iname():

    ref_knl = lp.make_kernel(
            "{ [g,h,i,j,k]: 0<=g,h,i,j,k<n }",
            """
            out[g,h,i,j,k] = 2*a[g,h,i,j,k]  {id=insn}
            """,
            assumptions="n >= 1",
            )
    ref_knl = lp.add_and_infer_dtypes(ref_knl, {"a": np.dtype(np.float32)})

    # Test split_iname where 'within'=None

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("k", "{g, h, i, j}"),
        )
    knl = lp.split_iname(knl, "j", 4)
    loop_nesting = _linearize_and_get_nestings(knl)[0]  # only one nesting
    assert loop_nesting[0] == "k"

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("{g, h, i, j}", "k"),
        )
    knl = lp.split_iname(knl, "j", 4)
    loop_nesting = _linearize_and_get_nestings(knl)[0]  # only one nesting
    assert loop_nesting[-1] == "k"

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        #must_nest=("{g, h, i}", "{j, k}"),
        must_not_nest=("{j, k}", "~{j, k}"),
        )
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "{g, h}"),
        )
    knl = lp.split_iname(knl, "g", 4)
    knl = lp.split_iname(knl, "j", 4)
    loop_nesting = _linearize_and_get_nestings(knl)[0]  # only one nesting
    assert loop_nesting[0] == "i"
    assert set(loop_nesting[1:4]) == set(["g_outer", "g_inner", "h"])
    assert set(loop_nesting[4:]) == set(["j_outer", "j_inner", "k"])

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "{g, h, j, k}"),
        must_not_nest=("h", "g"),
        )
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("{g, h, i}", "{j, k}"),
        )
    knl = lp.split_iname(knl, "g", 4)
    loop_nesting = _linearize_and_get_nestings(knl)[0]  # only one nesting
    assert loop_nesting[0] == "i"
    assert loop_nesting[1:4] == ("g_outer", "g_inner", "h")
    assert set(loop_nesting[4:]) == set(["j", "k"])

    # Testing split_iname with 'within'

    ref_knl = lp.make_kernel(
            "{ [g,h,i,j,k]: 0<=g,h,i,j,k<n }",
            """
            out1[g,h,i,j,k] = 2*a[g,h,i,j,k]  {id=insn1}
            out2[i,j,k] = 2+i+j+k  {id=insn2}
            """,
            assumptions="n >= 1",
            )
    ref_knl = lp.add_and_infer_dtypes(
        ref_knl, {"a": np.dtype(np.float32)}, {"out2": np.dtype(np.float32)})
    ref_knl = lp.constrain_loop_nesting(
        ref_knl,
        must_nest=("k", "i", "j"),
        must_not_nest=("g", "{j,h}"),
        )

    knl = ref_knl
    knl = lp.split_iname(knl, "j", 4, within="id:insn1")
    loop_nestings = _linearize_and_get_nestings(knl)
    assert ("k", "i", "j_outer", "j_inner", "h", "g") in loop_nestings
    assert ("k", "i", "j") in loop_nestings
    assert len(loop_nestings) == 2

    knl = ref_knl
    knl = lp.split_iname(knl, "j", 4, within="id:insn2")
    loop_nestings = _linearize_and_get_nestings(knl)
    assert ("k", "i", "j", "h", "g") in loop_nestings
    assert ("k", "i", "j_outer", "j_inner") in loop_nestings
    assert len(loop_nestings) == 2

# }}}


# {{{ test_constraint_updating_duplicate_inames

def test_constraint_updating_duplicate_inames():

    ref_knl = lp.make_kernel(
            "{ [g,h,i,j,k]: 0<=g,h,i,j,k<n }",
            """
            out[g,h,i,j,k] = 2*a[g,h,i,j,k]  {id=insn}
            out0[g,h,i,j,k] = 2*a0[g,h,i,j,k]  {id=insn0,dep=insn}
            """,
            assumptions="n >= 1",
            )
    ref_knl = lp.add_and_infer_dtypes(
        ref_knl,
        {"a": np.dtype(np.float32), "a0": np.dtype(np.float32)})

    # duplicate within insn0

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "{g, h, j, k}"),
        )
    knl = lp.duplicate_inames(
        knl,
        inames=["g", "h"],
        within="id:insn0",
        new_inames=["gg", "hh"])

    must_nest_graph_exp = dict([
        (iname, set()) for iname in ["g", "h", "j", "k", "gg", "hh"]])
    must_nest_graph_exp["i"] = set(["g", "h", "j", "k", "gg", "hh"])

    assert knl[
        "loopy_kernel"].loop_nest_constraints.must_nest_graph == must_nest_graph_exp

    nesting_for_insn, nesting_for_insn0 = _linearize_and_get_nestings(knl)

    # i must be outermost
    assert nesting_for_insn[0] == nesting_for_insn0[0] == "i"
    # j and k are shared between both insns, must come next
    assert (
        set(nesting_for_insn[1:3]) ==
        set(nesting_for_insn0[1:3]) ==
        set(["j", "k"]))
    # g,h and gg,hh should come after that
    assert set(nesting_for_insn[3:]) == set(["g", "h"])
    assert set(nesting_for_insn0[3:]) == set(["gg", "hh"])  # new names

    # duplicate within BOTH insns

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "{g, h, j, k}"),
        )
    knl = lp.duplicate_inames(
        knl,
        inames=["g", "h"],
        within="id:insn0 or id:insn",
        new_inames=["gg", "hh"])

    must_nest_graph_exp = dict([
        (iname, set()) for iname in ["j", "k", "gg", "hh"]])
    must_nest_graph_exp["i"] = set(["j", "k", "gg", "hh"])

    assert knl[
        "loopy_kernel"].loop_nest_constraints.must_nest_graph == must_nest_graph_exp

    loop_nestings = _linearize_and_get_nestings(knl)
    assert len(loop_nestings) == 1
    loop_nesting = loop_nestings[0]

    # i must be outermost
    assert loop_nesting[0] == loop_nesting[0] == "i"
    # j and k are shared between both insns, must come next
    assert set(loop_nesting[1:]) == set(["j", "k", "gg", "hh"])

    # duplicate within insn0 with must-not-nest

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_not_nest=("~{i}", "i"),
        )
    knl = lp.duplicate_inames(
        knl,
        inames=["g", "h"],
        within="id:insn0",
        new_inames=["gg", "hh"])
    nesting_for_insn, nesting_for_insn0 = _linearize_and_get_nestings(knl)

    assert nesting_for_insn[0] == nesting_for_insn0[0] == "i"

# }}}


# {{{ test_constraint_updating_rename_iname

def test_constraint_updating_rename_iname():

    ref_knl = lp.make_kernel(
            "{ [g,h,i,j,k]: 0<=g,h,i,j,k<n }",
            """
            out[g,h,i,j,k] = 2*a[g,h,i,j,k]  {id=insn}
            """,
            assumptions="n >= 1",
            )
    ref_knl = lp.add_and_infer_dtypes(ref_knl, {"a": np.dtype(np.float32)})

    # Test rename_iname (+ remove_unused_inames) where new iname does not exist
    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "{g, h, j, k}"),
        must_not_nest=("h", "g"),
        )
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("{g, h, i}", "{j, k}"),
        )
    knl = lp.rename_iname(knl, "g", "g_new")
    knl = lp.rename_iname(knl, "h", "h_new")
    knl = lp.rename_iname(knl, "i", "i_new")
    loop_nesting = _linearize_and_get_nestings(knl)[0]  # only one nesting
    assert loop_nesting[0] == "i_new"
    assert loop_nesting[1:3] == ("g_new", "h_new")
    assert set(loop_nesting[3:]) == set(["j", "k"])

    # TODO Test rename_iname where new iname DOES not exist (once implemented)

# }}}


# {{{ test_constraint_handling_tag_inames

def test_constraint_handling_tag_inames():

    ref_knl = lp.make_kernel(
            "{ [g,h,i,j,k]: 0<=g,h,i,j,k<n }",
            """
            out[g,h,i,j,k] = 2*a[g,h,i,j,k]  {id=insn}
            """,
            assumptions="n >= 1",
            )
    ref_knl = lp.add_and_infer_dtypes(ref_knl, {"a": np.dtype(np.float32)})

    # should error when constrained inames are tagged as concurrent
    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "{j, k}"),
        must_not_nest=("h", "g"),
        )
    try:
        lp.tag_inames(knl, {"i": "l.0"})
        raise AssertionError()
    except ValueError as e:
        assert (
            "cannot tag 'i' as concurrent--iname involved in must-nest constraint"
            in str(e))

    # Need to test anything else here...?

# }}}


# {{{ test_constraint_updating_join_inames

def test_constraint_updating_join_inames():

    ref_knl = lp.make_kernel(
            "{ [g,h,i,j,k]: 0<=g,h,i,j,k<1024 }",
            """
            out[g,h,i,j,k] = 2*a[g,h,i,j,k]  {id=insn}
            """,
            )
    ref_knl = lp.add_and_infer_dtypes(ref_knl, {"a": np.dtype(np.float32)})

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "{g, h, j, k}"),
        must_not_nest=("h", "g"),
        )
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("{g, h, i}", "{j, k}"),
        )
    knl = lp.join_inames(knl, inames=["g", "h"], new_iname="gh")
    loop_nesting = _linearize_and_get_nestings(knl)[0]  # only one nesting
    assert loop_nesting[0] == "i"
    assert loop_nesting[1] == "gh"
    assert set(loop_nesting[2:]) == set(["j", "k"])

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "{g, h, j, k}"),
        must_not_nest=("h", "g"),
        )
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("{g, h, i}", "{j, k}"),
        )
    knl = lp.join_inames(knl, inames=["j", "k"], new_iname="jk")
    # TODO figure out reason for jk key error
    loop_nesting = _linearize_and_get_nestings(knl)[0]  # only one nesting
    assert loop_nesting[0] == "i"
    assert loop_nesting[1:3] == ("g", "h")
    assert loop_nesting[3] == "jk"

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("h", "i", "g", "{j, k}"),
        )
    knl = lp.join_inames(knl, inames=["i", "g"], new_iname="ig")
    loop_nesting = _linearize_and_get_nestings(knl)[0]  # only one nesting
    assert loop_nesting[0] == "h"
    assert loop_nesting[1] == "ig"
    assert set(loop_nesting[2:4]) == set(["j", "k"])

    # test cycle detection
    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "{g, h}", "{j, k}"),
        )
    try:
        lp.join_inames(knl, inames=["i", "k"], new_iname="ik")
        raise AssertionError()
    except ValueError as e:
        assert "cycle" in str(e)

    # test implied nesting that creates constraint violation
    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_not_nest=("i", "k"),
        )
    try:
        lp.join_inames(knl, inames=["i", "k"], new_iname="ik")
        raise AssertionError()
    except ValueError as e:
        assert "Implied nestings violate existing must-not-nest" in str(e)

# }}}


# {{{ test_iname_coalescing_in_loop_nest_constraints

def test_iname_coalescing_in_loop_nest_constraints():

    def get_sets_of_inames(iname_sets_tuple, iname_universe):
        # convert UnexpandedInameSets to sets
        sets_of_inames = []
        for iname_set in iname_sets_tuple:
            sets_of_inames.append(
                iname_set.get_inames_represented(iname_universe))
        return sets_of_inames

    ref_knl = lp.make_kernel(
            "{ [g,h,i,j,k]: 0<=g,h,i,j,k<1024 }",
            """
            out[g,h,i,j,k] = 2*a[g,h,i,j,k]  {id=insn}
            """,
            )
    # (join_inames errors if domain bound is variable)

    ref_knl = lp.add_and_infer_dtypes(ref_knl, {"a": np.dtype(np.float32)})

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "g", "h", "j", "k"),
        )
    knl = lp.join_inames(knl, inames=["g", "h"], new_iname="gh")
    knl = knl["loopy_kernel"]
    new_must_nest = get_sets_of_inames(
        list(knl.loop_nest_constraints.must_nest)[0], knl.all_inames())
    expected_must_nest = [
        set(["i", ]), set(["gh", ]), set(["j", ]), set(["k", ])]
    assert new_must_nest == expected_must_nest

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("{i, g}", "h", "j", "k"),
        )
    knl = lp.join_inames(knl, inames=["g", "h"], new_iname="gh")
    knl = knl["loopy_kernel"]
    new_must_nest = get_sets_of_inames(
        list(knl.loop_nest_constraints.must_nest)[0], knl.all_inames())
    expected_must_nest = [
        set(["i", ]), set(["gh", ]), set(["j", ]), set(["k", ])]
    assert new_must_nest == expected_must_nest

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "g", "{h, j}", "k"),
        )
    knl = lp.join_inames(knl, inames=["g", "h"], new_iname="gh")
    knl = knl["loopy_kernel"]
    new_must_nest = get_sets_of_inames(
        list(knl.loop_nest_constraints.must_nest)[0], knl.all_inames())
    expected_must_nest = [
        set(["i", ]), set(["gh", ]), set(["j", ]), set(["k", ])]
    assert new_must_nest == expected_must_nest

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "g", "{h, j, k}"),
        )
    knl = lp.join_inames(knl, inames=["g", "h"], new_iname="gh")
    knl = knl["loopy_kernel"]
    new_must_nest = get_sets_of_inames(
        list(knl.loop_nest_constraints.must_nest)[0], knl.all_inames())
    expected_must_nest = [
        set(["i", ]), set(["gh", ]), set(["j", "k"])]
    assert new_must_nest == expected_must_nest

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "{g, h}", "j", "k"),
        )
    knl = lp.join_inames(knl, inames=["g", "h"], new_iname="gh")
    knl = knl["loopy_kernel"]
    new_must_nest = get_sets_of_inames(
        list(knl.loop_nest_constraints.must_nest)[0], knl.all_inames())
    expected_must_nest = [
        set(["i", ]), set(["gh", ]), set(["j", ]), set(["k", ])]
    assert new_must_nest == expected_must_nest

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("{i, g}", "{h, j, k}"),
        )
    knl = lp.join_inames(knl, inames=["g", "h"], new_iname="gh")
    knl = knl["loopy_kernel"]
    new_must_nest = get_sets_of_inames(
        list(knl.loop_nest_constraints.must_nest)[0], knl.all_inames())
    expected_must_nest = [
        set(["i", ]), set(["gh", ]), set(["j", "k"])]
    assert new_must_nest == expected_must_nest

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("i", "g", "j", "h", "k"),
        )
    try:
        knl = lp.join_inames(knl, inames=["g", "h"], new_iname="gh")
        raise AssertionError()
    except ValueError as e:
        assert "contains cycle" in str(e)

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("{i, g}", "j", "{h, k}"),
        )
    try:
        knl = lp.join_inames(knl, inames=["g", "h"], new_iname="gh")
        raise AssertionError()
    except ValueError as e:
        assert "contains cycle" in str(e)

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_nest=("{i, h}", "j", "{g, k}"),
        )
    try:
        knl = lp.join_inames(knl, inames=["g", "h"], new_iname="gh")
        raise AssertionError()
    except ValueError as e:
        assert "nestings violate existing must-nest" in str(e)

    knl = ref_knl
    knl = lp.constrain_loop_nesting(
        knl,
        must_not_nest=("g", "h"),
        )
    try:
        knl = lp.join_inames(knl, inames=["g", "h"], new_iname="gh")
        raise AssertionError()
    except ValueError as e:
        assert "nestings violate existing must-not-nest" in str(e)

# }}}

# TODO make standalone test for constraint updating functions that
# doesn't bother with transforms/linearization

# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

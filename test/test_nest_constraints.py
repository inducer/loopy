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
    prog = lp.constrain_loop_nesting(ref_knl, must_nest=("k", "{h }", " { j , i } "))
    knl = prog["loopy_kernel"]
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

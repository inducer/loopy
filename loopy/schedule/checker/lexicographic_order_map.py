# coding: utf-8
__copyright__ = "Copyright (C) 2019 James Stevens"

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

import islpy as isl


def get_statement_ordering_map(
        sched_before, sched_after, lex_map, before_marker="'"):
    """Return a statement ordering represented as a map from each statement
        instance to all statement instances occurring later.

    :arg sched_before: An :class:`islpy.Map` representing a schedule
        as a mapping from statement instances (for one particular statement)
        to lexicographic time. The statement represented will typically
        be the dependee in a dependency relationship.

    :arg sched_after: An :class:`islpy.Map` representing a schedule
        as a mapping from statement instances (for one particular statement)
        to lexicographic time. The statement represented will typically
        be the depender in a dependency relationship.

    :arg lex_map: An :class:`islpy.Map` representing a lexicographic
        ordering as a mapping from each point in lexicographic time
        to every point that occurs later in lexicographic time. E.g.::

            {[i0', i1', i2', ...] -> [i0, i1, i2, ...] :
                i0' < i0 or (i0' = i0 and i1' < i1)
                or (i0' = i0 and i1' = i1 and i2' < i2) ...}

    :arg before_marker: A :class:`str` to be appended to the names of the
        map dimensions representing the 'before' statement in the
        'happens before' relationship.

    :returns: An :class:`islpy.Map` representing the statement odering as
        a mapping from each statement instance to all statement instances
        occurring later. I.e., we compose relations B, L, and A as
        B ∘ L ∘ A^-1, where B is `sched_before`, A is `sched_after`,
        and L is `lex_map`.

    """

    # Perform the composition of relations
    sio = sched_before.apply_range(
        lex_map).apply_range(sched_after.reverse())

    # Append marker to in_ dims
    from loopy.schedule.checker.utils import (
        append_marker_to_isl_map_var_names,
    )
    return append_marker_to_isl_map_var_names(
        sio, isl.dim_type.in_, before_marker)


def get_lex_order_set(
        before_names, after_names,
        before_names_concurrent=[],
        after_names_concurrent=[],
        islvars=None,
        ):
    """Return an :class:`islpy.Set` representing a lexicographic ordering
        with the number of dimensions provided in `before_names`
        (equal to the number of dimensions in `after_names`).

    :arg before_names: A list of :class:`str` variable names to be used
        to describe lexicographic space dimensions for a point in a lexicographic
        ordering that occurs before another point, which will be represented using
        `after_names`. (see example below)

    :arg after_names: A list of :class:`str` variable names to be used
        to describe lexicographic space dimensions for a point in a lexicographic
        ordering that occurs after another point, which will be represented using
        `before_names`. (see example below)

    :arg islvars: A dictionary mapping variable names in `before_names` and
        `after_names` to :class:`islpy.PwAff` instances that represent each
        of the variables (islvars may be produced by `islpy.make_zero_and_vars`).
        The key '0' is also include and represents a :class:`islpy.PwAff` zero
        constant. This dictionary defines the space to be used for the set. If no
        value is passed, the dictionary will be made using `before_names`
        and `after_names`.

    :returns: An :class:`islpy.Set` representing a big-endian lexicographic ordering
        with the number of dimensions provided in `before_names`. The set
        has one dimension for each name in *both* `before_names` and
        `after_names`, and contains all points which meet a 'happens before'
        constraint defining the lexicographic ordering. E.g., if
        `before_names = [i0', i1', i2']` and `after_names = [i0, i1, i2]`,
        return the set containing all points in a 3-dimensional, big-endian
        lexicographic ordering such that point
        `[i0', i1', i2']` happens before `[i0, i1, i2]`. I.e., return::

            {[i0', i1', i2', i0, i1, i2] :
                i0' < i0 or (i0' = i0 and i1' < i1)
                or (i0' = i0 and i1' = i1 and i2' < i2)}

    """
    # TODO update doc

    from loopy.schedule.checker.utils import (
        create_elementwise_comparison_conjunction_set,
    )

    # If no islvars passed, make them using the names provided
    # (make sure to pass var names in desired order of space dims)
    if islvars is None:
        islvars = isl.make_zero_and_vars(
            before_names+before_names_concurrent+after_names+after_names_concurrent,
            [])

    # Initialize set with constraint i0' < i0
    lex_order_set = islvars[before_names[0]].lt_set(islvars[after_names[0]])

    # For each dim d, starting with d=1, equality_conj_set will be constrained
    # by d equalities, e.g., (i0' = i0 and i1' = i1 and ... i(d-1)' = i(d-1)).
    equality_conj_set = islvars[0].eq_set(islvars[0])  # initialize to 'true'

    for i in range(1, len(before_names)):

        # Add the next equality constraint to equality_conj_set
        equality_conj_set = equality_conj_set & \
            islvars[before_names[i-1]].eq_set(islvars[after_names[i-1]])

        # Create a set constrained by adding a less-than constraint for this dim,
        # e.g., (i1' < i1), to the current equality conjunction set.
        # For each dim d, starting with d=1, this full conjunction will have
        # d equalities and one inequality, e.g.,
        # (i0' = i0 and i1' = i1 and ... i(d-1)' = i(d-1) and id' < id)
        full_conj_set = islvars[before_names[i]].lt_set(
            islvars[after_names[i]]) & equality_conj_set

        # Union this new constraint with the current lex_order_set
        lex_order_set = lex_order_set | full_conj_set

    lex_order_set = lex_order_set & \
        create_elementwise_comparison_conjunction_set(
            before_names_concurrent, after_names_concurrent,
            islvars, op="eq",
            )

    return lex_order_set


def create_lex_order_map(
        n_dims=None,
        before_names=None,
        after_names=None,
        after_names_concurrent=[],
        ):
    """Return a map from each point in a lexicographic ordering to every
        point that occurs later in the lexicographic ordering.

    :arg n_dims: An :class:`int` representing the number of dimensions
        in the lexicographic ordering. If not provided, `n_dims` will be
        set to length of `after_names`.

    :arg before_names: A list of :class:`str` variable names to be used
        to describe lexicographic space dimensions for a point in a lexicographic
        ordering that occurs before another point, which will be represented using
        `after_names`. (see example below)

    :arg after_names: A list of :class:`str` variable names to be used
        to describe lexicographic space dimensions for a point in a lexicographic
        ordering that occurs after another point, which will be represented using
        `before_names`. (see example below)

    :returns: An :class:`islpy.Map` representing a lexicographic
        ordering as a mapping from each point in lexicographic time
        to every point that occurs later in lexicographic time.
        E.g., if `before_names = [i0', i1', i2']` and
        `after_names = [i0, i1, i2]`, return the map::

            {[i0', i1', i2'] -> [i0, i1, i2] :
                i0' < i0 or (i0' = i0 and i1' < i1)
                or (i0' = i0 and i1' = i1 and i2' < i2)}

    """
    # TODO update doc

    from loopy.schedule.checker.utils import append_marker_to_strings

    if after_names is None:
        after_names = ["i%s" % (i) for i in range(n_dims)]
    if before_names is None:
        before_names = append_marker_to_strings(after_names, marker="'")
    if n_dims is None:
        n_dims = len(after_names)
    before_names_concurrent = append_marker_to_strings(
        after_names_concurrent, marker="'")

    assert len(before_names) == len(after_names) == n_dims
    dim_type = isl.dim_type

    # First, get a set representing the lexicographic ordering.
    lex_order_set = get_lex_order_set(
        before_names, after_names,
        before_names_concurrent, after_names_concurrent,
        )

    # Now convert that set to a map.
    lex_map = isl.Map.from_domain(lex_order_set)
    return lex_map.move_dims(
        dim_type.out, 0, dim_type.in_,
        len(before_names) + len(before_names_concurrent),
        len(after_names) + len(after_names_concurrent))

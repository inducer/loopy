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
        sched_before, sched_after, lex_map, before_mark):
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

    :arg before_mark: A :class:`str` to be appended to the names of the
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

    # Append mark to in_ dims
    from loopy.schedule.checker.utils import (
        append_mark_to_isl_map_var_names,
    )
    return append_mark_to_isl_map_var_names(
        sio, isl.dim_type.in_, before_mark)


def _create_lex_order_set(
        dim_names,
        in_dim_mark,
        islvars=None,
        ):
    """Return an :class:`islpy.Set` representing a lexicographic ordering
    over a space with the number of dimensions provided in `dim_names`
    (the set itself will have twice this many dimensions in order to
    represent the ordering as before-after pairs of points).

    :arg dim_names: A list of :class:`str` variable names to be used
        to describe lexicographic space dimensions for a point in a lexicographic
        ordering. (see example below)

    :arg in_dim_mark: A :class:`str` to be appended to dimension names to
        distinguish corresponding dimensions in before-after pairs of points.
        (see example below)

    :arg islvars: A dictionary mapping variable names in `dim_names` to
        :class:`islpy.PwAff` instances that represent each of the variables
        (islvars may be produced by `islpy.make_zero_and_vars`).
        The key '0' is also include and represents a :class:`islpy.PwAff` zero
        constant. This dictionary defines the space to be used for the set and
        must also include versions of `dim_names` with the `in_dim_mark`
        appended. If no value is passed, the dictionary will be made using
        `dim_names` and `dim_names` with the `in_dim_mark` appended.

    :returns: An :class:`islpy.Set` representing a big-endian lexicographic
        ordering with the number of dimensions provided in `dim_names`. The set
        has two dimensions for each name in `dim_names`, one identified by the
        given name and another identified by the same name with `in_dim_mark`
        appended. The set contains all points which meet a 'happens before'
        constraint defining the lexicographic ordering. E.g., if
        `dim_names = [i0, i1, i2]` and `in_dim_mark="'"`,
        return the set containing all points in a 3-dimensional, big-endian
        lexicographic ordering such that point
        `[i0', i1', i2']` happens before `[i0, i1, i2]`. I.e., return::

            {[i0', i1', i2', i0, i1, i2] :
                i0' < i0 or (i0' = i0 and i1' < i1)
                or (i0' = i0 and i1' = i1 and i2' < i2)}

    """

    from loopy.schedule.checker.utils import (
        append_mark_to_strings,
    )

    in_dim_names = append_mark_to_strings(dim_names, mark=in_dim_mark)

    # If no islvars passed, make them using the names provided
    # (make sure to pass var names in desired order of space dims)
    if islvars is None:
        islvars = isl.make_zero_and_vars(
            in_dim_names+dim_names,
            [])

    # Initialize set with constraint i0' < i0
    lex_order_set = islvars[in_dim_names[0]].lt_set(islvars[dim_names[0]])

    # For each dim d, starting with d=1, equality_conj_set will be constrained
    # by d equalities, e.g., (i0' = i0 and i1' = i1 and ... i(d-1)' = i(d-1)).
    equality_conj_set = islvars[0].eq_set(islvars[0])  # initialize to 'true'

    for i in range(1, len(in_dim_names)):

        # Add the next equality constraint to equality_conj_set
        equality_conj_set = equality_conj_set & \
            islvars[in_dim_names[i-1]].eq_set(islvars[dim_names[i-1]])

        # Create a set constrained by adding a less-than constraint for this dim,
        # e.g., (i1' < i1), to the current equality conjunction set.
        # For each dim d, starting with d=1, this full conjunction will have
        # d equalities and one inequality, e.g.,
        # (i0' = i0 and i1' = i1 and ... i(d-1)' = i(d-1) and id' < id)
        full_conj_set = islvars[in_dim_names[i]].lt_set(
            islvars[dim_names[i]]) & equality_conj_set

        # Union this new constraint with the current lex_order_set
        lex_order_set = lex_order_set | full_conj_set

    return lex_order_set


def create_lex_order_map(
        dim_names,
        in_dim_mark,
        ):
    """Return a map from each point in a lexicographic ordering to every
    point that occurs later in the lexicographic ordering.

    :arg dim_names: A list of :class:`str` variable names for the
        lexicographic space dimensions.

    :arg in_dim_mark: A :class:`str` to be appended to `dim_names` to create
        the names for the input dimensions of the map, thereby distinguishing
        them from the corresponding output dimensions in before-after pairs of
        points. (see example below)

    :returns: An :class:`islpy.Map` representing a lexicographic
        ordering as a mapping from each point in lexicographic time
        to every point that occurs later in lexicographic time.
        E.g., if `dim_names = [i0, i1, i2]` and `in_dim_mark = "'"`,
        return the map::

            {[i0', i1', i2'] -> [i0, i1, i2] :
                i0' < i0 or (i0' = i0 and i1' < i1)
                or (i0' = i0 and i1' = i1 and i2' < i2)}

    """

    n_dims = len(dim_names)
    dim_type = isl.dim_type

    # First, get a set representing the lexicographic ordering.
    lex_order_set = _create_lex_order_set(
        dim_names,
        in_dim_mark=in_dim_mark,
        )

    # Now convert that set to a map.
    lex_map = isl.Map.from_domain(lex_order_set)
    return lex_map.move_dims(
        dim_type.out, 0, dim_type.in_,
        n_dims, n_dims)

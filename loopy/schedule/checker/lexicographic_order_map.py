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
        sched_map_before, sched_map_after, lex_map, before_marker="'"):
    """Return a mapping that maps each statement instance to
        all statement instances occuring later.

    :arg sched_map_before: An :class:`islpy.Map` representing instruction
        instance order for the dependee as a mapping from each statement
        instance to a point in the lexicographic ordering.

    :arg sched_map_after: An :class:`islpy.Map` representing instruction
        instance order for the depender as a mapping from each statement
        instance to a point in the lexicographic ordering.

    :arg lex_map: An :class:`islpy.Map` representing a lexicographic
        ordering as a mapping from each point in lexicographic time
        to every point that occurs later in lexicographic time. E.g.::

            {[i0', i1', i2', ...] -> [i0, i1, i2, ...] :
                i0' < i0 or (i0' = i0 and i1' < i1)
                or (i0' = i0 and i1' = i1 and i2' < i2) ...}

    :returns: An :class:`islpy.Map` representing the lex schedule as
        a mapping from each statement instance to all statement instances
        occuring later. I.e., we compose relations B, L, and A as
        B ∘ L ∘ A^-1, where B is sched_map_before, A is sched_map_after,
        and L is the lexicographic ordering map.

    """

    sio = sched_map_before.apply_range(
        lex_map).apply_range(sched_map_after.reverse())
    # append marker to in names
    from loopy.schedule.checker.utils import (
        append_marker_to_isl_map_var_names,
    )
    return append_marker_to_isl_map_var_names(
        sio, isl.dim_type.in_, before_marker)


def get_lex_order_constraint(before_names, after_names, islvars=None):
    """Return a constraint represented as an :class:`islpy.Set`
        defining a 'happens before' relationship in a lexicographic
        ordering.

    :arg before_names: A list of :class:`str` variable names representing
        the lexicographic space dimensions for a point in lexicographic
        time that occurs before. (see example below)

    :arg after_names: A list of :class:`str` variable names representing
        the lexicographic space dimensions for a point in lexicographic
        time that occurs after. (see example below)

    :arg islvars: A dictionary from variable names to :class:`islpy.PwAff`
        instances that represent each of the variables
        (islvars may be produced by `islpy.make_zero_and_vars`). The key
        '0' is also include and represents a :class:`islpy.PwAff` zero constant.
        This dictionary defines the space to be used for the set. If no
        value is passed, the dictionary will be made using ``before_names``
        and ``after_names``.

    :returns: An :class:`islpy.Set` representing a constraint that enforces a
        lexicographic ordering. E.g., if ``before_names = [i0', i1', i2']`` and
        ``after_names = [i0, i1, i2]``, return the set::

            {[i0', i1', i2', i0, i1, i2] :
                i0' < i0 or (i0' = i0 and i1' < i1)
                or (i0' = i0 and i1' = i1 and i2' < i2)}

    """

    # If no islvars passed, make them using the names provided
    if islvars is None:
        islvars = isl.make_zero_and_vars(before_names+after_names, [])

    # Initialize constraint with i0' < i0
    lex_order_constraint = islvars[before_names[0]].lt_set(islvars[after_names[0]])

    # Initialize conjunction constraint with True.
    # For each dim d, starting with d=1, this conjunction will have d equalities,
    # e.g., (i0' = i0 and i1' = i1 and ... i(d-1)' = i(d-1))
    equality_constraint_conj = islvars[0].eq_set(islvars[0])

    for i in range(1, len(before_names)):

        # Add the next equality constraint to equality_constraint_conj
        equality_constraint_conj = equality_constraint_conj & \
            islvars[before_names[i-1]].eq_set(islvars[after_names[i-1]])

        # Create a conjunction constraint by combining a less-than
        # constraint for this dim, e.g., (i1' < i1), with the current
        # equality constraint conjunction.
        # For each dim d, starting with d=1, this conjunction will have d equalities,
        # and one inequality,
        # e.g., (i0' = i0 and i1' = i1 and ... i(d-1)' = i(d-1) and id' < id)
        full_conj_constraint = islvars[before_names[i]].lt_set(
            islvars[after_names[i]]) & equality_constraint_conj

        # Union this new constraint with the current lex_order_constraint
        lex_order_constraint = lex_order_constraint | full_conj_constraint

    return lex_order_constraint


def create_lex_order_map(
        n_dims=None,
        before_names=None,
        after_names=None,
        ):
    """Return a mapping that maps each point in a lexicographic
        ordering to every point that occurs later in lexicographic
        time.

    :arg n_dims: An :class:`int` representing the number of dimensions
        in the lexicographic ordering.

    :arg before_names: A list of :class:`str` variable names representing
        the lexicographic space dimensions for a point in lexicographic
        time that occurs before. (see example below)

    :arg after_names: A list of :class:`str` variable names representing
        the lexicographic space dimensions for a point in lexicographic
        time that occurs after. (see example below)

    :returns: An :class:`islpy.Map` representing a lexicographic
        ordering as a mapping from each point in lexicographic time
        to every point that occurs later in lexicographic time.
        E.g., if ``before_names = [i0', i1', i2']`` and
        ``after_names = [i0, i1, i2]``, return the map::

            {[i0', i1', i2'] -> [i0, i1, i2] :
                i0' < i0 or (i0' = i0 and i1' < i1)
                or (i0' = i0 and i1' = i1 and i2' < i2)}

    """

    if after_names is None:
        after_names = ["i%s" % (i) for i in range(n_dims)]
    if before_names is None:
        from loopy.schedule.checker.utils import (
            append_marker_to_strings,
        )
        before_names = append_marker_to_strings(after_names, marker="'")
    if n_dims is None:
        n_dims = len(after_names)

    assert len(before_names) == len(after_names) == n_dims
    dim_type = isl.dim_type

    lex_order_constraint = get_lex_order_constraint(before_names, after_names)

    lex_map = isl.Map.from_domain(lex_order_constraint)
    lex_map = lex_map.move_dims(
        dim_type.out, 0, dim_type.in_,
        len(before_names), len(after_names))

    return lex_map

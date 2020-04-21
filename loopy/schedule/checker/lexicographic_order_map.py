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
        occuring later. I.e., we compose B -> L -> A^-1, where B
        is sched_map_before, A is sched_map_after, and L is the
        lexicographic ordering map.

    """

    sio = sched_map_before.apply_range(
        lex_map).apply_range(sched_map_after.reverse())
    # append marker to in names
    from loopy.schedule.checker.utils import (
        append_marker_to_isl_map_var_names,
    )
    return append_marker_to_isl_map_var_names(
        sio, isl.dim_type.in_, before_marker)


def get_lex_order_constraint(islvars, before_names, after_names):
    """Return a constraint represented as an :class:`islpy.Set`
        defining a 'happens before' relationship in a lexicographic
        ordering.

    :arg islvars: A dictionary from variable names to :class:`islpy.PwAff`
        instances that represent each of the variables
        (islvars may be produced by `islpy.make_zero_and_vars`). The key
        '0' is also include and represents a :class:`islpy.PwAff` zero constant.
        This dictionary defines the space to be used for the set.

    :arg before_names: A list of :class:`str` variable names representing
        the lexicographic space dimensions for a point in lexicographic
        time that occurs before. (see example below)

    :arg after_names: A list of :class:`str` variable names representing
        the lexicographic space dimensions for a point in lexicographic
        time that occurs after. (see example below)

    :returns: An :class:`islpy.Set` representing a constraint that enforces a
        lexicographic ordering. E.g., if ``before_names = [i0', i1', i2']`` and
        ``after_names = [i0, i1, i2]``, return the set::

            {[i0', i1', i2', i0, i1, i2] :
                i0' < i0 or (i0' = i0 and i1' < i1)
                or (i0' = i0 and i1' = i1 and i2' < i2)}

    """

    lex_order_constraint = islvars[before_names[0]].lt_set(islvars[after_names[0]])
    for i in range(1, len(before_names)):
        lex_order_constraint_conj = islvars[before_names[i]].lt_set(
            islvars[after_names[i]])
        for j in range(i):
            lex_order_constraint_conj = lex_order_constraint_conj & \
                islvars[before_names[j]].eq_set(islvars[after_names[j]])
        lex_order_constraint = lex_order_constraint | lex_order_constraint_conj
    return lex_order_constraint


def create_lex_order_map(
        n_dims,
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

    if before_names is None:
        before_names = ["i%s" % (i) for i in range(n_dims)]
    if after_names is None:
        from loopy.schedule.checker.utils import (
            append_marker_to_strings,
        )
        after_names = append_marker_to_strings(before_names, marker="_")

    assert len(before_names) == len(after_names) == n_dims
    dim_type = isl.dim_type

    islvars = isl.make_zero_and_vars(
            before_names+after_names,
            [])

    lex_order_constraint = get_lex_order_constraint(
        islvars, before_names, after_names)

    lex_map = isl.Map.from_domain(lex_order_constraint)
    lex_map = lex_map.move_dims(
        dim_type.out, 0, dim_type.in_,
        len(before_names), len(after_names))

    return lex_map

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
from dataclasses import dataclass
from loopy.schedule.checker.utils import (
    add_and_name_isl_dims,
    add_eq_isl_constraint_from_names,
    append_mark_to_isl_map_var_names,
    move_dims_by_name,
    remove_dims_by_name,
    prettier_map_string,  # noqa
)
from loopy.isl_helpers import (
    find_and_rename_dims,
)
dim_type = isl.dim_type


# {{{ Constants

__doc__ = """

.. data:: LIN_CHECK_IDENTIFIER_PREFIX

    The :class:`str` prefix for identifiers involved in linearization
    checking.

.. data:: LEX_VAR_PREFIX

    The :class:`str` prefix for the variables representing the
    dimensions in the lexicographic ordering used in a pairwise schedule. E.g.,
    a prefix of ``_lp_linchk_lex`` might yield lexicographic dimension
    variables ``_lp_linchk_lex0``, ``_lp_linchk_lex1``, ``_lp_linchk_lex2``.
    Cf.  :ref:`reserved-identifiers`.

.. data:: STATEMENT_VAR_NAME

    The :class:`str` name for the statement-identifying dimension of maps
    representing schedules and statement instance orderings.

.. data:: LTAG_VAR_NAME

    An array of :class:`str` names for map dimensions carrying values for local
    (intra work-group) thread identifiers in maps representing schedules and
    statement instance orderings.

.. data:: GTAG_VAR_NAME

    An array of :class:`str` names for map dimensions carrying values for group
    identifiers in maps representing schedules and statement instance orderings.

.. data:: BEFORE_MARK

    The :class:`str` identifier to be appended to input dimension names in
    maps representing schedules and statement instance orderings.

"""

LIN_CHECK_IDENTIFIER_PREFIX = "_lp_linchk_"
#LEX_VAR_PREFIX = "%slex" % (LIN_CHECK_IDENTIFIER_PREFIX)
LEX_VAR_PREFIX = "lx"  # TODO change back
STATEMENT_VAR_NAME = "%sstmt" % (LIN_CHECK_IDENTIFIER_PREFIX)
LTAG_VAR_NAMES = []
GTAG_VAR_NAMES = []
for par_level in [0, 1, 2]:
    LTAG_VAR_NAMES.append("%slid%d" % (LIN_CHECK_IDENTIFIER_PREFIX, par_level))
    GTAG_VAR_NAMES.append("%sgid%d" % (LIN_CHECK_IDENTIFIER_PREFIX, par_level))
BEFORE_MARK = "'"

# }}}


# {{{ Helper Functions

# {{{ _pad_tuple_with_zeros

def _pad_tuple_with_zeros(tup, desired_length):
    return tup[:] + tuple([0]*(desired_length-len(tup)))

# }}}


# {{{ _simplify_lex_dims

def _simplify_lex_dims(tup0, tup1):
    """Simplify a pair of lex tuples in order to reduce the complexity of
    resulting maps. Remove lex tuple dimensions with matching integer values
    since these do not provide information on relative ordering. Once a
    dimension is found where both tuples have non-matching integer values,
    remove any faster-updating lex dimensions since they are not necessary
    to specify a relative ordering.
    """

    new_tup0 = []
    new_tup1 = []

    # Loop over dims from slowest updating to fastest
    for d0, d1 in zip(tup0, tup1):
        if isinstance(d0, int) and isinstance(d1, int):

            # Both vals are ints for this dim
            if d0 == d1:
                # Do not keep this dim
                continue
            elif d0 > d1:
                # These ints inform us about the relative ordering of
                # two statements. While their values may be larger than 1 in
                # the lexicographic ordering describing a larger set of
                # statements, in a pairwise schedule, only ints 0 and 1 are
                # necessary to specify relative order. To keep the pairwise
                # schedules as simple and comprehensible as possible, use only
                # integers 0 and 1 to specify this relative ordering.
                # (doesn't take much extra time since we are already going
                # through these to remove unnecessary lex tuple dims)
                new_tup0.append(1)
                new_tup1.append(0)

                # No further dims needed to fully specify ordering
                break
            else:  # d1 > d0
                new_tup0.append(0)
                new_tup1.append(1)

                # No further dims needed to fully specify ordering
                break
        else:
            # Keep this dim without modifying
            new_tup0.append(d0)
            new_tup1.append(d1)

    if len(new_tup0) == 0:
        # Statements map to the exact same point(s) in the lex ordering,
        # which is okay, but to represent this, our lex tuple cannot be empty.
        return (0, ), (0, )
    else:
        return tuple(new_tup0), tuple(new_tup1)

# }}}

# }}}


# {{{ class SpecialLexPointWRTLoop

class SpecialLexPointWRTLoop:
    """Strings identifying a particular point or set of points in a
    lexicographic ordering of statements, specified relative to a loop.

    .. attribute:: PRE
        A :class:`str` indicating the last lexicographic point that
        precedes the loop.

    .. attribute:: FIRST
        A :class:`str` indicating the first lexicographic point in the
        first loop iteration (i.e., with the iname set to its min. val).

    .. attribute:: TOP
        A :class:`str` indicating the first lexicographic point in
        an arbitrary loop iteration.

    .. attribute:: BOTTOM
        A :class:`str` indicating the last lexicographic point in
        an arbitrary loop iteration.

    .. attribute:: LAST
        A :class:`str` indicating the last lexicographic point in the
        last loop iteration (i.e., with the iname set to its max val).

    .. attribute:: POST
        A :class:`str` indicating the first lexicographic point that
        follows the loop.
    """

    PRE = "pre"
    FIRST = "first"
    TOP = "top"
    BOTTOM = "bottom"
    LAST = "last"
    POST = "post"

# }}}


# {{{ class StatementOrdering

@dataclass
class StatementOrdering:
    r"""A container for the three statement instance orderings (described
    below) used to formalize the ordering of statement instances for a pair of
    statements.

    Also included (mostly for testing and debugging) are the
    intra-thread pairwise schedule (`pwsched_intra_thread`), intra-group
    pairwise schedule (`pwsched_intra_group`), and global pairwise schedule
    (`pwsched_global`), each containing a pair of mappings from statement
    instances to points in a lexicographic ordering, one for each statement.
    Each SIO is created by composing the two mappings in the corresponding
    pairwise schedule with an associated mapping defining the ordering of
    points in the lexicographical space (not included).
    """

    sio_intra_thread: isl.Map
    sio_intra_group: isl.Map
    sio_global: isl.Map
    pwsched_intra_thread: tuple
    pwsched_intra_group: tuple
    pwsched_global: tuple

# }}}


# {{{ _gather_blex_ordering_info

# {{{ Helper functions

def _assert_exact_closure(mapping):
    closure_test, closure_exact = mapping.transitive_closure()
    assert closure_exact
    assert closure_test == mapping


def _add_one_blex_tuple(
        all_blex_points, blex_tuple, all_seq_blex_dim_names,
        conc_inames, knl):
    """Create the (bounded) set of blex points represented by *blex_tuple* and
    add it to *all_blex_points*.
    """

    # blex_tuple: (int, iname, int, iname, int, ...)
    # - Contains 1 initial dim plus 2 dims for each sequential loop surrounding
    # the *current* linearization item
    # - Will need padding with zeros for any trailing blex dims
    # - blex_tuple[1::2] is a subset of all sequential inames

    # {{{ Get inames domain for current inames

    # (need to account for concurrent inames here rather than adding them on
    # to blex map at the end because a sequential iname domain may depend on a
    # concurrent iname domain)

    # Get set of inames nested outside (including this iname)
    all_within_inames = set(blex_tuple[1::2]) | conc_inames

    dom = knl.get_inames_domain(
        all_within_inames).project_out_except(
            all_within_inames, [dim_type.set])

    # }}}

    # {{{ Prepare for union between dom and all_blex_points

    # Rename sequential iname dims in dom to corresponding blex dim names
    dom = find_and_rename_dims(
        dom, dim_type.set,
        dict(zip(blex_tuple[1::2], all_seq_blex_dim_names[1::2])))

    # Move concurrent inames in dom to params
    dom = move_dims_by_name(
        dom, dim_type.param, dom.n_param(),
        dim_type.set, conc_inames)

    # Add any new params found in dom to all_blex_points prior to aligning dom
    # with all_blex_points
    missing_params = set(
        dom.get_var_names(dim_type.param)  # needed params
        ) - set(all_blex_points.get_var_names(dim_type.param))  # current params
    all_blex_points = add_and_name_isl_dims(
        all_blex_points, dim_type.param, missing_params)

    # Add missing blex dims to dom and align it with all_blex_points
    dom = isl.align_spaces(dom, all_blex_points)

    # Set values for non-iname (integer) blex dims in dom
    for blex_dim_name, blex_val in zip(all_seq_blex_dim_names[::2], blex_tuple[::2]):
        dom = add_eq_isl_constraint_from_names(dom, blex_dim_name, blex_val)
    # Set values for any unused (rightmost, fastest-updating) dom blex dims to zero
    for blex_dim_name in all_seq_blex_dim_names[len(blex_tuple):]:
        dom = add_eq_isl_constraint_from_names(dom, blex_dim_name, 0)

    # }}}

    # Add this blex set to full set of blex points
    return all_blex_points | dom

# }}}


def _gather_blex_ordering_info(
        knl,
        sync_kind,
        lin_items, seq_loops_with_barriers,
        max_seq_loop_depth,
        conc_inames, loop_bounds,
        all_stmt_ids,
        all_conc_lex_dim_names, gid_lex_dim_names,
        conc_iname_constraint_dicts,
        perform_closure_checks=False,
        ):
    # TODO some of these params might be redundant
    """For the given sync_kind ("local" or "global"), create a mapping from
    statement instances to blex space (dict), as well as a mapping
    defining the blex ordering (isl map from blex space -> blex space)

    Note that, unlike in the intra-thread case, there will be a single
    blex ordering map defining the blex ordering for all statement pairs,
    rather than separate (smaller) lex ordering maps for each pair
    """
    from loopy.schedule import (EnterLoop, LeaveLoop, Barrier, RunInstruction)
    from loopy.schedule.checker.lexicographic_order_map import (
        create_lex_order_map,
    )
    from loopy.schedule.checker.utils import (
        add_and_name_isl_dims,
        append_mark_to_strings,
        add_eq_isl_constraint_from_names,
    )
    slex = SpecialLexPointWRTLoop

    # {{{ First, create map from stmt instances to blex space.

    # At the same time,
    # - Gather information necessary to create the blex ordering map, i.e., for
    # each loop, gather the 6 lex order tuples defined above in
    # SpecialLexPointWRTLoop that will be required to create sub-maps which
    # will be *excluded* (subtracted) from a standard lexicographic ordering in
    # order to create the blex ordering
    # - Create all_blex_points, a set containing *all* blex points, which will
    # be used later to impose bounds on the full blex map and any blex maps to
    # be subtracted from it

    # {{{ Create the initial (pre-subtraction) blex order map, initially w/o bounds

    # Determine the number of blex dims we will need
    n_seq_blex_dims = max_seq_loop_depth*2 + 1

    # Create names for the blex dimensions for sequential loops
    seq_blex_dim_names = [
        LEX_VAR_PREFIX+str(i) for i in range(n_seq_blex_dims)]
    seq_blex_dim_names_prime = append_mark_to_strings(
        seq_blex_dim_names, mark=BEFORE_MARK)

    # Begin with the blex order map created as a standard lexicographical order
    # (bounds will be applied later by intersecting this with map containing
    # all blex points)
    blex_order_map = create_lex_order_map(
        dim_names=seq_blex_dim_names,
        in_dim_mark=BEFORE_MARK)

    # }}}

    # {{{  Create a template set for the space of all blex points

    # Create set of all blex points by starting with (0, 0, 0, ...)
    # and then unioning this with each new set of blex points we find
    all_blex_points = isl.align_spaces(
        isl.Map("[ ] -> { [ ] -> [ ] }"), blex_order_map).range()
    for var_name in seq_blex_dim_names:
        all_blex_points = add_eq_isl_constraint_from_names(
                all_blex_points, var_name, 0)
    # Add concurrent inames as params
    # (iname domains found in the pass below may depend on concurrent inames)
    all_blex_points = add_and_name_isl_dims(
        all_blex_points, dim_type.param, conc_inames)

    # }}}

    stmt_inst_to_blex = {}  # Map stmt instances to blex space
    iname_to_blex_dim = {}  # Map from inames to corresponding blex space dim
    blex_exclusion_info = {}  # Info for creating maps to exclude from blex order
    next_blex_tuple = [0]  # Next tuple of points in blex order

    for lin_item in lin_items:
        if isinstance(lin_item, EnterLoop):
            enter_iname = lin_item.iname
            if enter_iname in seq_loops_with_barriers:
                pre_loop_blex_pt = next_blex_tuple[:]

                # Increment next_blex_tuple[-1] for statements in the section
                # of code between this EnterLoop and the matching LeaveLoop.
                next_blex_tuple[-1] += 1

                # Upon entering a loop, add one blex dimension for the loop
                # iteration, add second blex dim to enumerate sections of
                # code within new loop
                next_blex_tuple.append(enter_iname)
                next_blex_tuple.append(0)

                # Store 3 tuples that will later be used to create mappings
                # between blex points that will be subtracted from the full
                # blex order map

                first_iter_blex_pt = next_blex_tuple[:]
                first_iter_blex_pt[-2] = enter_iname
                blex_exclusion_info[enter_iname] = {
                    slex.PRE: tuple(pre_loop_blex_pt),
                    slex.TOP: tuple(next_blex_tuple),
                    slex.FIRST: tuple(first_iter_blex_pt),
                    }
                # (copy these three blex points when creating dict because
                # the lists will continue to be updated)

                # {{{ Create the blex set for this point, add it to all_blex_points

                all_blex_points = _add_one_blex_tuple(
                    all_blex_points, next_blex_tuple,
                    seq_blex_dim_names, conc_inames, knl)

                # }}}

        elif isinstance(lin_item, LeaveLoop):
            leave_iname = lin_item.iname
            if leave_iname in seq_loops_with_barriers:

                curr_blex_dim_ct = len(next_blex_tuple)

                # Record the blex dim for this loop iname
                iname_to_blex_dim[leave_iname] = curr_blex_dim_ct-2

                # Update next blex pt
                pre_end_loop_blex_pt = next_blex_tuple[:]
                # Upon leaving a loop:
                # - Pop lex dim for enumerating code sections within this loop
                # - Pop lex dim for the loop iteration
                # - Increment lex dim val enumerating items in current section
                next_blex_tuple.pop()
                next_blex_tuple.pop()
                next_blex_tuple[-1] += 1

                # Store 3 tuples that will later be used to create mappings
                # between blex points that will be subtracted from the full
                # blex order map

                # TODO some of this storage may be unnecessary now that loop
                # bounds are found elsewhere... clean this up

                last_iter_blex_pt = pre_end_loop_blex_pt[:]
                last_iter_blex_pt[-2] = leave_iname
                blex_exclusion_info[leave_iname][slex.BOTTOM] = tuple(
                    pre_end_loop_blex_pt)
                blex_exclusion_info[leave_iname][slex.LAST] = tuple(
                    last_iter_blex_pt)
                blex_exclusion_info[leave_iname][slex.POST] = tuple(
                    next_blex_tuple)
                # (copy these three blex points when creating dict because
                # the lists will continue to be updated)

                # {{{ Create the blex set for this point, add it to all_blex_points

                all_blex_points = _add_one_blex_tuple(
                    all_blex_points, next_blex_tuple,
                    seq_blex_dim_names, conc_inames, knl)

                # }}}

        elif isinstance(lin_item, RunInstruction):
            # Add stmt->blex pair to stmt_inst_to_blex
            stmt_inst_to_blex[lin_item.insn_id] = tuple(next_blex_tuple)

            # (Don't increment blex dim val)

        elif isinstance(lin_item, Barrier):
            # Increment blex dim val if the sync scope matches
            if lin_item.synchronization_kind == sync_kind:
                next_blex_tuple[-1] += 1

                # {{{ Create the blex set for this point, add it to all_blex_points

                all_blex_points = _add_one_blex_tuple(
                    all_blex_points, next_blex_tuple,
                    seq_blex_dim_names, conc_inames, knl)

                # }}}

            lp_stmt_id = lin_item.originating_insn_id

            if lp_stmt_id is None:
                # Barriers without stmt ids were inserted as a result of a
                # dependency. They don't themselves have dependencies.
                # Don't map this barrier to a blex tuple.
                continue

            # This barrier has a stmt id.
            # If it was included in listed stmts, process it.
            # Otherwise, there's nothing left to do (we've already
            # incremented next_blex_tuple if necessary, and this barrier
            # does not need to be assigned to a designated point in blex
            # time)
            if lp_stmt_id in all_stmt_ids:

                # Assign a blex point to this barrier just as we would for an
                # assignment stmt
                stmt_inst_to_blex[lp_stmt_id] = tuple(next_blex_tuple)

                # If sync scope matches, give this barrier its *own* point in
                # lex time by updating blex tuple after barrier.
                if lin_item.synchronization_kind == sync_kind:
                    next_blex_tuple[-1] += 1

                    # {{{ Create the blex set for this point, add it to
                    # all_blex_points

                    all_blex_points = _add_one_blex_tuple(
                        all_blex_points, next_blex_tuple,
                        seq_blex_dim_names, conc_inames, knl)

                    # }}}
        else:
            from loopy.schedule import (CallKernel, ReturnFromKernel)
            # No action needed for these types of linearization item
            assert isinstance(
                lin_item, (CallKernel, ReturnFromKernel))
            pass

    # At this point, some blex tuples may have more dimensions than others;
    # the missing dims are the fastest-updating dims, and their values should
    # be zero. Add them.
    for stmt, tup in stmt_inst_to_blex.items():
        stmt_inst_to_blex[stmt] = _pad_tuple_with_zeros(tup, n_seq_blex_dims)

    # }}}

    # {{{ Second, create the blex order map

    # {{{ Bound the (pre-subtraction) blex order map

    conc_iname_to_iname_prime = {
        conc_iname: conc_iname+BEFORE_MARK for conc_iname in conc_inames}
    all_blex_points_prime = append_mark_to_isl_map_var_names(
        all_blex_points, dim_type.set, BEFORE_MARK)
    all_blex_points_prime = find_and_rename_dims(
        all_blex_points_prime, dim_type.param, conc_iname_to_iname_prime,
        )
    blex_order_map = blex_order_map.intersect_domain(
        all_blex_points_prime).intersect_range(all_blex_points)

    # }}}

    # {{{ Subtract unwanted pairs from happens-before blex map

    # Create mapping (dict) from iname to corresponding blex dim name
    # TODO rename to "seq_..."
    seq_iname_to_blex_var = {}
    for iname, dim in iname_to_blex_dim.items():
        seq_iname_to_blex_var[iname] = seq_blex_dim_names[dim]
        seq_iname_to_blex_var[iname+BEFORE_MARK] = seq_blex_dim_names_prime[dim]

    # {{{ Get a template map matching blex_order_map.space that will serve as
    # the starting point when creating the maps to subtract from blex_order_map

    # This template includes concurrent inames as params, both marked
    # ('before') and unmarked ('after').
    # Note that this template cannot be created until *after* the intersection
    # of blex_order_map with all_blex_points above, otherwise the template will
    # be missing necessary parameters.
    blex_map_template = isl.align_spaces(
        isl.Map("[ ] -> { [ ] -> [ ] }"), blex_order_map)
    blex_set_template = blex_map_template.range()

    # }}}

    # {{{ _pad_tuples_and_assign_integer_vals_to_map_template() helper

    seq_blex_in_out_dim_names = seq_blex_dim_names_prime + seq_blex_dim_names

    def _pad_tuples_and_assign_integer_vals_to_map_template(
            in_tuple, out_tuple):
        # External variables read (not written):
        # n_seq_blex_dims, seq_blex_in_out_dim_names, blex_map_template

        # Pad the tuples
        in_tuple_padded = _pad_tuple_with_zeros(in_tuple, n_seq_blex_dims)
        out_tuple_padded = _pad_tuple_with_zeros(out_tuple, n_seq_blex_dims)

        # Assign map values for ints only
        map_with_int_vals_assigned = blex_map_template
        for dim_name, val in zip(
                seq_blex_in_out_dim_names,
                in_tuple_padded+out_tuple_padded):
            if isinstance(val, int):
                map_with_int_vals_assigned = add_eq_isl_constraint_from_names(
                    map_with_int_vals_assigned, dim_name, val)

        return map_with_int_vals_assigned

    # }}}

    # {{{ Create blex map to subtract for each iname in blex_exclusion_info

    maps_to_subtract = []
    for iname, key_lex_tuples in blex_exclusion_info.items():

        # {{{ Create blex map to subtract for one iname

        """Create the maps that must be subtracted from the
        initial blex order map for this particular loop using the 6 blex
        tuples in key_lex_tuples:
        PRE->FIRST, BOTTOM(iname')->TOP(iname'+1), LAST->POST

        """

        # {{{ Create PRE->FIRST, BOTTOM(iname')->TOP(iname'+1), LAST->POST
        # initially without iname domain bounds.

        # We know which blex dims correspond to inames due to their
        # position in blex tuples (int, iname, int, iname, int, ...), and their
        # iname domain bounds will be set later by intersecting the subtraction
        # map with the (bounded) full blex map.

        # Perform the following:
        # - For map domains/ranges corresponding to the PRE, BOTTOM, TOP, and
        # POST sets, leave the blex dims corresponding to inames unbounded and
        # set the values for blex dims that will be ints, i.e., the
        # even-indexed (intra-loop-section) blex dims and any trailing zeros.
        # - For map domains/ranges corresponding to the FIRST and LAST sets,
        # set the map dimension corresponding to this iname using
        # loop_bounds[iname][0] and loop_bounds[iname][1].
        # - For the BOTTOM->TOP map, add constraint iname = iname' + 1

        # {{{ Create PRE->FIRST map

        # PRE dim vals should all be inames (bounded later) or ints (assign now).
        # FIRST dim values will be inames, ints, or one of our lexmin bounds.

        # Create PRE->FIRST map and assign int (non-iname) dim values.
        first_tuple = key_lex_tuples[slex.FIRST]
        pre_to_first_map = _pad_tuples_and_assign_integer_vals_to_map_template(
            key_lex_tuples[slex.PRE], first_tuple)

        pre_to_first_map = _pad_tuples_and_assign_integer_vals_to_map_template(
            key_lex_tuples[slex.PRE], first_tuple)

        # Get the set representing the value of the iname on the first
        # iteration of the loop
        loop_min_bound = loop_bounds[iname][0]
        # (in loop_bounds sets, concurrent inames are params)

        # Prepare the loop_min_bound set for intersection with the range of
        # pre_to_first_map by renaming iname dims to blex dims and aligning
        # spaces
        loop_min_bound = find_and_rename_dims(
            loop_min_bound, dim_type.set,
            {k: seq_iname_to_blex_var[k] for k in first_tuple[1::2]})
        # Align with blex space (adds needed dims)
        loop_first_set = isl.align_spaces(loop_min_bound, blex_set_template)

        # Finish making PRE->FIRST pair by intersecting this with the range of
        # our pre_to_first_map
        pre_to_first_map = pre_to_first_map.intersect_range(loop_first_set)

        # NOTE: We will add a condition to fix iteration values for
        # *surrounding* sequential loops (j = j') after combining the three
        # maps (PRE-FIRST, BOTTOM->TOP, LAST->POST) below

        # }}}

        # {{{ Create BOTTOM->TOP map

        # BOTTOM/TOP dim vals should all be inames (bounded later) or ints
        # (assign now).

        # Create BOTTOM->TOP map and assign int (non-iname) dim values
        bottom_to_top_map = _pad_tuples_and_assign_integer_vals_to_map_template(
            key_lex_tuples[slex.BOTTOM], key_lex_tuples[slex.TOP])

        bottom_to_top_map = _pad_tuples_and_assign_integer_vals_to_map_template(
            key_lex_tuples[slex.BOTTOM], key_lex_tuples[slex.TOP])

        # Add constraint iname = iname' + 1
        blex_var_for_iname = seq_iname_to_blex_var[iname]
        bottom_to_top_map = bottom_to_top_map.add_constraint(
            isl.Constraint.eq_from_names(
                bottom_to_top_map.space,
                {1: 1, blex_var_for_iname + BEFORE_MARK: 1, blex_var_for_iname: -1}))

        # }}}

        # {{{ LAST->POST

        # POST dim vals should all be inames (bounded later) or ints (assign now).
        # LAST dim values will be inames, ints, or one of our lexmax bounds.

        # Create LAST->POST map and assign int (non-iname) dim values.
        last_tuple = key_lex_tuples[slex.LAST]
        last_to_post_map = _pad_tuples_and_assign_integer_vals_to_map_template(
            last_tuple, key_lex_tuples[slex.POST])

        last_to_post_map = _pad_tuples_and_assign_integer_vals_to_map_template(
            last_tuple, key_lex_tuples[slex.POST])

        # Get the set representing the value of the iname on the last
        # iteration of the loop
        loop_max_bound = loop_bounds[iname][1]

        # {{{ Prepare the loop_max_bound set for intersection with the domain of
        # last_to_post_map by renaming iname dims to blex dims and aligning
        # spaces
        loop_max_bound = find_and_rename_dims(
            loop_max_bound, dim_type.set,
            {k: seq_iname_to_blex_var[k] for k in last_tuple[1::2]})

        # There may be concurrent inames in the dim_type.param dimensions of
        # the loop_max_bound, and we need to append the BEFORE_MARK to those
        # inames to ensure that they are distinguished from the corresponding
        # non-marked 'after' (concurrent) inames.
        # (While the other dims in loop_max_bound also correspond to 'before'
        # dimensions of last_to_post_map, which carry the 'before' mark, we do
        # not need to append the mark to them in loop_max_bound because calling
        # last_to_post_map.intersect_domain(loop_last_set) below will match the
        # space.in_ dims by position rather than name)
        loop_max_bound = find_and_rename_dims(
            loop_max_bound, dim_type.param, conc_iname_to_iname_prime)

        # Align with blex space (adds needed dims)
        loop_last_set = isl.align_spaces(loop_max_bound, blex_set_template)

        # }}}

        # Make LAST->POST pair by intersecting this with the range of our map
        # Finish making LAST->POST pair by intersecting this with the range of
        # our last_to_post_map
        last_to_post_map = last_to_post_map.intersect_domain(loop_last_set)

        # }}}

        map_to_subtract = pre_to_first_map | bottom_to_top_map | last_to_post_map

        # Add condition to fix iter value for *surrounding* sequential loops (j = j')
        # (odd indices in key_lex_tuples[PRE] contain the sounding inames)
        for seq_surrounding_iname in key_lex_tuples[slex.PRE][1::2]:
            s_blex_var = seq_iname_to_blex_var[seq_surrounding_iname]
            map_to_subtract = add_eq_isl_constraint_from_names(
                map_to_subtract, s_blex_var, s_blex_var+BEFORE_MARK)

        # Bound the blex dims by intersecting with the full blex map, which
        # contains all the bound constraints
        map_to_subtract &= blex_order_map
        print("CONSTRAINED MAP_TO_SUBTRACT FOR LOOP", iname)
        print(prettier_map_string(map_to_subtract))

        # }}}

        maps_to_subtract.append(map_to_subtract)

    # }}}

    # {{{ Subtract transitive closure of union of blex maps to subtract

    if maps_to_subtract:

        # Get union of maps
        map_to_subtract = maps_to_subtract[0]
        for other_map in maps_to_subtract[1:]:
            map_to_subtract |= other_map

        # Get transitive closure of maps
        map_to_subtract_closure, closure_exact = map_to_subtract.transitive_closure()

        assert closure_exact  # TODO warn instead?

        # {{{ Check assumptions about map transitivity

        if perform_closure_checks:

            # Make sure map_to_subtract_closure is subset of blex_order_map
            assert map_to_subtract <= blex_order_map
            assert map_to_subtract_closure <= blex_order_map

            # Make sure blex_order_map and map_to_subtract are closures
            _assert_exact_closure(blex_order_map)
            _assert_exact_closure(map_to_subtract_closure)

        # }}}

        # Subtract closure from blex order map
        blex_order_map -= map_to_subtract_closure

        # {{{ Check assumptions about map transitivity

        # Make sure blex_order_map is closure after subtraction
        if perform_closure_checks:
            _assert_exact_closure(blex_order_map)

        # }}}

    # }}}

    # Add LID/GID dims to blex order map:

    # At this point, all concurrent inames should be params in blex order map.
    # Rename them to the corresponding concurrent lex dim name and move them to
    # in_/out dims.

    # NOTE:
    # Even though all parallel thread dims are active throughout the
    # whole kernel, they may be assigned (tagged) to one iname for some
    # subset of statements and another iname for a different subset of
    # statements (e.g., tiled, parallel matmul).
    # There could, e.g., be *multiple* inames that correspond to LID0, and each
    # of these inames could be involved in defining the domain set for other
    # inames. We don't want to lose any of this information. For this reason,
    # we first creat the LID/GID dims, then set each one equal to *all*
    # corresponding concurrent inames (which are in param dims), and then
    # remove the (param) iname dims.

    # Add conc lex dim names to both in_ and out dims
    blex_order_map = add_and_name_isl_dims(
        blex_order_map, dim_type.in_,
        [v+BEFORE_MARK for v in all_conc_lex_dim_names])
    blex_order_map = add_and_name_isl_dims(
        blex_order_map, dim_type.out, all_conc_lex_dim_names)

    # Set each of the new conc lex dims equal to *all* corresponding inames
    # (here, conc_iname_constraint_dicts includes primed inames)
    for constraint_dict in conc_iname_constraint_dicts:
        blex_order_map = blex_order_map.add_constraint(
            isl.Constraint.eq_from_names(blex_order_map.space, constraint_dict))

    # Now remove conc inames from params
    blex_order_map = remove_dims_by_name(
        blex_order_map, dim_type.param,
        conc_inames | set([v+BEFORE_MARK for v in conc_inames]))

    if sync_kind == "local":
        # For intra-group case, constrain GID 'before' to equal GID 'after'

        # (in the current implementation, all gid_lex_dim_names should be
        # present in blex_order_map)
        for var_name in gid_lex_dim_names:
            blex_order_map = add_eq_isl_constraint_from_names(
                    blex_order_map, var_name, var_name+BEFORE_MARK)

    # (if sync_kind == "global", don't need constraints on LID/GID vars)

    # }}}

    # }}}

    return (
        stmt_inst_to_blex,  # map stmt instances to blex space
        blex_order_map,
        seq_blex_dim_names,
        )

# }}}


# {{{ get_pairwise_statement_orderings_inner

def get_pairwise_statement_orderings_inner(
        knl,
        lin_items,
        stmt_id_pairs,
        ilp_and_vec_inames=frozenset(),
        perform_closure_checks=False,
        ):
    r"""For each statement pair in a subset of all statement pairs found in a
    linearized kernel, determine the (relative) order in which the statement
    instances are executed. For each pair, represent this relative ordering
    using three ``statement instance orderings`` (SIOs):

    - The intra-thread SIO: A :class:`islpy.Map` from each instance of the
      first statement to all instances of the second statement that occur
      later, such that both statement instances in each before-after pair are
      executed within the same work-item (thread).

    - The intra-group SIO: A :class:`islpy.Map` from each instance of the first
      statement to all instances of the second statement that occur later, such
      that both statement instances in each before-after pair are executed
      within the same work-group (though potentially by different work-items).

    - The global SIO: A :class:`islpy.Map` from each instance of the first
      statement to all instances of the second statement that occur later, even
      if the two statement instances in a given before-after pair are executed
      within different work-groups.

    :arg knl: A preprocessed :class:`loopy.kernel.LoopKernel` containing the
        linearization items that will be used to create the SIOs. This
        kernel will be used to get the domains associated with the inames
        used in the statements, and to determine which inames have been
        tagged with parallel tags.

    :arg lin_items: A list of :class:`loopy.schedule.ScheduleItem`
        (to be renamed to `loopy.schedule.LinearizationItem`) containing
        all linearization items for which SIOs will be
        created. To allow usage of this routine during linearization, a
        truncated (i.e. partial) linearization may be passed through this
        argument

    :arg stmt_id_pairs: A list containing pairs of statement identifiers.

    :arg ilp_and_vec_inames: A set of inames that will be ignored when
        determining the relative ordering of statements. This will typically
        contain concurrent inames tagged with the ``vec`` or ``ilp`` array
        access tags.

    :returns: A dictionary mapping each two-tuple of statement identifiers
        provided in `stmt_id_pairs` to a :class:`StatementOrdering`, which
        contains the three SIOs described above.
    """

    from loopy.schedule import (EnterLoop, LeaveLoop, Barrier, RunInstruction)
    from loopy.kernel.data import (LocalInameTag, GroupInameTag)
    from loopy.schedule.checker.lexicographic_order_map import (
        create_lex_order_map,
        get_statement_ordering_map,
    )
    from loopy.schedule.checker.utils import (
        add_and_name_isl_dims,
        append_mark_to_strings,
        sorted_union_of_names_in_isl_sets,
        create_symbolic_map_from_tuples,
        insert_and_name_isl_dims,
        partition_inames_by_concurrency,
    )

    all_stmt_ids = set().union(*stmt_id_pairs)
    conc_inames = partition_inames_by_concurrency(knl)[0]

    # {{{ Intra-thread lex order creation

    # First, use one pass through lin_items to generate an *intra-thread*
    # lexicographic ordering describing the relative order of all statements
    # represented by all_stmt_ids

    # For each statement, map the stmt_id to a tuple representing points
    # in the intra-thread lexicographic ordering containing items of :class:`int` or
    # :class:`str` :mod:`loopy` inames
    stmt_inst_to_lex_intra_thread = {}

    # Keep track of the next tuple of points in our lexicographic
    # ordering, initially this as a 1-d point with value 0
    next_lex_tuple = [0]

    # While we're passing through, determine which loops contain barriers,
    # this information will be used later when creating *intra-group* and
    # *global* lexicographic orderings
    seq_loops_with_barriers = {"local": set(), "global": set()}
    max_depth_of_barrier_loop = {"local": 0, "global": 0}
    current_seq_inames = []

    # While we're passing through, also determine the values of the active
    # inames on the first and last iteration of each loop that contains
    # barriers (dom.lexmin/lexmax).
    # This information will be used later when creating *intra-group* and
    # *global* lexicographic orderings
    loop_bounds = {}

    for lin_item in lin_items:
        if isinstance(lin_item, EnterLoop):
            iname = lin_item.iname

            if iname in ilp_and_vec_inames:
                continue

            current_seq_inames.append(iname)

            # Increment next_lex_tuple[-1] for statements in the section
            # of code between this EnterLoop and the matching LeaveLoop.
            # (not technically necessary if no statement was added in the
            # previous section; gratuitous incrementing is counteracted
            # in the simplification step below)
            next_lex_tuple[-1] += 1

            # Upon entering a loop, add one lex dimension for the loop iteration,
            # add second lex dim to enumerate sections of code within new loop
            next_lex_tuple.append(iname)
            next_lex_tuple.append(0)

        elif isinstance(lin_item, LeaveLoop):
            iname = lin_item.iname

            if iname in ilp_and_vec_inames:
                continue

            current_seq_inames.pop()

            # Upon leaving a loop:
            # - Pop lex dim for enumerating code sections within this loop
            # - Pop lex dim for the loop iteration
            # - Increment lex dim val enumerating items in current section of code
            next_lex_tuple.pop()
            next_lex_tuple.pop()
            next_lex_tuple[-1] += 1

            # (not technically necessary if no statement was added in the
            # previous section; gratuitous incrementing is counteracted
            # in the simplification step below)

        elif isinstance(lin_item, RunInstruction):
            lp_stmt_id = lin_item.insn_id

            # Only process listed stmts, otherwise ignore
            if lp_stmt_id in all_stmt_ids:
                # Add item to stmt_inst_to_lex_intra_thread
                stmt_inst_to_lex_intra_thread[lp_stmt_id] = tuple(next_lex_tuple)

                # Increment lex dim val enumerating items in current section of code
                next_lex_tuple[-1] += 1

        elif isinstance(lin_item, Barrier):
            lp_stmt_id = lin_item.originating_insn_id
            sync_kind = lin_item.synchronization_kind
            seq_loops_with_barriers[sync_kind] |= set(current_seq_inames)
            max_depth_of_barrier_loop[sync_kind] = max(
                len(current_seq_inames), max_depth_of_barrier_loop[sync_kind])

            # {{{ Store bounds for loops containing barriers

            # Only compute the bounds we haven't already stored; bounds finding
            # will only happen once for each barrier-containing loop
            for depth, iname in enumerate(current_seq_inames):

                # If we haven't already stored bounds for this iname, do so
                if iname not in loop_bounds:

                    # Get set of inames that might be involved in this bound
                    # (this iname plus any nested outside this iname, including
                    # concurrent inames)
                    seq_surrounding_inames = set(current_seq_inames[:depth])
                    all_surrounding_inames = seq_surrounding_inames | conc_inames

                    # Get inames domain
                    inames_involved_in_bound = all_surrounding_inames | {iname}
                    dom = knl.get_inames_domain(
                        inames_involved_in_bound).project_out_except(
                            inames_involved_in_bound, [dim_type.set])

                    # {{{ Move domain dims for surrounding inames to parameters

                    dom = move_dims_by_name(
                        dom, dim_type.param, dom.n_param(),
                        dim_type.set, all_surrounding_inames)

                    # }}}

                    lmin = dom.lexmin()
                    lmax = dom.lexmax()

                    # Now move non-concurrent param inames back to set dim
                    lmin = move_dims_by_name(
                        lmin, dim_type.set, 0,
                        dim_type.param, seq_surrounding_inames)
                    lmax = move_dims_by_name(
                        lmax, dim_type.set, 0,
                        dim_type.param, seq_surrounding_inames)

                    loop_bounds[iname] = (lmin, lmax)

            # }}}

            if lp_stmt_id is None:
                # Barriers without stmt ids were inserted as a result of a
                # dependency. They don't themselves have dependencies. Ignore them.

                # FIXME: It's possible that we could record metadata about them
                # (e.g. what dependency produced them) and verify that they're
                # adequately protecting all statement instance pairs.

                continue

            # If barrier was identified in listed stmts, process it
            if lp_stmt_id in all_stmt_ids:
                # Add item to stmt_inst_to_lex_intra_thread
                stmt_inst_to_lex_intra_thread[lp_stmt_id] = tuple(next_lex_tuple)

                # Increment lex dim val enumerating items in current section of code
                next_lex_tuple[-1] += 1

        else:
            from loopy.schedule import (CallKernel, ReturnFromKernel)
            # No action needed for these types of linearization item
            assert isinstance(
                lin_item, (CallKernel, ReturnFromKernel))
            pass

    # }}}

    # {{{ Create lex dim names representing parallel axes

    # Create lex dim names representing lid/gid axes.
    # At the same time, create the dicts that will be used later to create map
    # constraints that match each parallel iname to the corresponding lex dim
    # name in schedules, i.e., i = lid0, j = lid1, etc.
    # TODO some of these vars may be redundant:
    lid_lex_dim_names = set()
    gid_lex_dim_names = set()

    # Dicts that will be used to create constraints i = lid0, j = lid1, etc.
    # (for efficiency, create these dicts one time per concurrent iname here,
    # rather than recreating the dicts multiple times later)
    conc_iname_constraint_dicts = {}
    conc_iname_constraint_dicts_prime = {}

    # Even though all parallel thread dims are active throughout the
    # whole kernel, they may be assigned (tagged) to one iname for some
    # subset of statements and another iname for a different subset of
    # statements (e.g., tiled, paralle. matmul).
    for iname in knl.all_inames():
        conc_tag = knl.iname_tags_of_type(iname, (LocalInameTag, GroupInameTag))
        if conc_tag:
            assert len(conc_tag) == 1  # (should always be true)
            conc_tag = conc_tag.pop()
            if isinstance(conc_tag, LocalInameTag):
                tag_var = LTAG_VAR_NAMES[conc_tag.axis]
                lid_lex_dim_names.add(tag_var)
            else:  # Must be GroupInameTag
                tag_var = GTAG_VAR_NAMES[conc_tag.axis]
                gid_lex_dim_names.add(tag_var)

            tag_var_prime = tag_var+BEFORE_MARK
            iname_prime = iname+BEFORE_MARK
            conc_iname_constraint_dicts[iname] = {1: 0, iname: 1, tag_var: -1}
            conc_iname_constraint_dicts_prime[iname_prime] = {
                1: 0, iname_prime: 1, tag_var_prime: -1}

    # Sort for consistent dimension ordering
    lid_lex_dim_names = sorted(lid_lex_dim_names)
    gid_lex_dim_names = sorted(gid_lex_dim_names)
    # TODO remove redundancy have one definitive list for these
    # (just make separate 1-d lists for everything?)

    # }}}

    # {{{ Intra-group and global blex ("barrier-lex") order creation

    # (may be combined with pass above in future)

    """In blex space, we order barrier-delimited sections of code.
    Each statement instance within a single barrier-delimited section will
    map to the same blex point. The resulting statement instance ordering
    (SIO) will map each statement to all statements that occur in a later
    barrier-delimited section.

    To achieve this, we will first create a map from statement instances to
    lexicographic space almost as we did above in the intra-thread case,
    though we will not increment the fastest-updating lex dim with each
    statement, and we will increment it with each barrier encountered. To
    denote these differences, we refer to this space as 'blex' space.

    The resulting pairwise schedule, if composed with a map defining a
    standard lexicographic ordering to create an SIO, would include a number
    of unwanted 'before->after' pairs of statement instances, so before
    creating the SIO, we will subtract unwanted pairs from a standard
    lex order map, yielding the 'blex' order map.
    """

    # {{{ Create blex order maps and blex tuples defining statement ordering (x2)

    all_conc_lex_dim_names = lid_lex_dim_names + gid_lex_dim_names
    all_conc_iname_constraint_dicts = list(
        conc_iname_constraint_dicts.values()
        ) + list(conc_iname_constraint_dicts_prime.values())

    # Get the blex schedule blueprint (dict will become a map below) and
    # blex order map w.r.t. local and global barriers
    (stmt_inst_to_lblex,
     lblex_order_map,
     seq_lblex_dim_names) = _gather_blex_ordering_info(
        knl,
        "local",
        lin_items, seq_loops_with_barriers["local"],
        max_depth_of_barrier_loop["local"],
        conc_inames, loop_bounds,
        all_stmt_ids,
        all_conc_lex_dim_names, gid_lex_dim_names,
        all_conc_iname_constraint_dicts,
        perform_closure_checks=perform_closure_checks,
        )
    (stmt_inst_to_gblex,
     gblex_order_map,
     seq_gblex_dim_names) = _gather_blex_ordering_info(
        knl,
        "global",
        lin_items, seq_loops_with_barriers["global"],
        max_depth_of_barrier_loop["global"],
        conc_inames, loop_bounds,
        all_stmt_ids,
        all_conc_lex_dim_names, gid_lex_dim_names,
        all_conc_iname_constraint_dicts,
        perform_closure_checks=perform_closure_checks,
        )

    # }}}

    # }}}  end intra-group and global blex order creation

    # {{{ Create pairwise schedules (ISL maps) for each stmt pair

    # {{{ _get_map_for_stmt()

    def _get_map_for_stmt(
            stmt_id, lex_points, int_sid, lex_dim_names):

        # Get inames domain for statement instance (a BasicSet)
        within_inames = knl.id_to_insn[stmt_id].within_inames
        dom = knl.get_inames_domain(
            within_inames).project_out_except(within_inames, [dim_type.set])

        # Create map space (an isl space in current implementation)
        # {('statement', <inames used in statement domain>) ->
        #  (lexicographic ordering dims)}
        dom_inames_ordered = sorted_union_of_names_in_isl_sets([dom])

        in_names_sched = [STATEMENT_VAR_NAME] + dom_inames_ordered[:]
        sched_space = isl.Space.create_from_names(
            isl.DEFAULT_CONTEXT,
            in_=in_names_sched,
            out=lex_dim_names,
            params=[],
            )

        # Insert 'statement' dim into domain so that its space allows
        # for intersection with sched map later
        dom_to_intersect = insert_and_name_isl_dims(
                dom, dim_type.set, [STATEMENT_VAR_NAME], 0)

        # Each map will map statement instances -> lex time.
        # At this point, statement instance tuples consist of single int.
        # Add all inames from domains to each map domain tuple.
        tuple_pair = [(
            (int_sid, ) + tuple(dom_inames_ordered),
            lex_points
            )]

        # Note that lex_points may have fewer dims than the out-dim of sched_space
        # if sched_space includes concurrent LID/GID dims. This is okay because
        # the following symbolic map creation step, when assigning dim values,
        # zips the space dims with the lex tuple, and any leftover LID/GID dims
        # will not be assigned a value yet, which is what we want.

        # Create map
        sched_map = create_symbolic_map_from_tuples(
            tuple_pairs_with_domains=zip(tuple_pair, [dom_to_intersect, ]),
            space=sched_space,
            )

        # Set inames equal to relevant GID/LID var names
        for iname, constraint_dict in conc_iname_constraint_dicts.items():
            # Even though all parallel thread dims are active throughout the
            # whole kernel, they may be assigned (tagged) to one iname for some
            # subset of statements and another iname for a different subset of
            # statements (e.g., tiled, paralle. matmul).
            # So before adding each parallel iname constraint, make sure the
            # iname applies to this statement:
            if iname in dom_inames_ordered:
                sched_map = sched_map.add_constraint(
                    isl.Constraint.eq_from_names(sched_map.space, constraint_dict))

        return sched_map

    # }}}

    pairwise_sios = {}

    for stmt_ids in stmt_id_pairs:
        # Determine integer IDs that will represent each statement in mapping
        # (dependency map creation assumes sid_before=0 and sid_after=1, unless
        # before and after refer to same stmt, in which case
        # sid_before=sid_after=0)
        int_sids = [0, 0] if stmt_ids[0] == stmt_ids[1] else [0, 1]

        # {{{  Create SIO for intra-thread case (lid0' == lid0, gid0' == gid0, etc)

        # Simplify tuples to the extent possible ------------------------------------

        lex_tuples = [stmt_inst_to_lex_intra_thread[stmt_id] for stmt_id in stmt_ids]

        # At this point, one of the lex tuples may have more dimensions than
        # another; the missing dims are the fastest-updating dims, and their
        # values should be zero. Add them.
        max_lex_dims = max([len(lex_tuple) for lex_tuple in lex_tuples])
        lex_tuples_padded = [
            _pad_tuple_with_zeros(lex_tuple, max_lex_dims)
            for lex_tuple in lex_tuples]

        # Now generate maps from the blueprint --------------------------------------

        lex_tuples_simplified = _simplify_lex_dims(*lex_tuples_padded)

        # Create names for the output dimensions for sequential loops
        seq_lex_dim_names = [
            LEX_VAR_PREFIX+str(i) for i in range(len(lex_tuples_simplified[0]))]

        intra_thread_sched_maps = [
            _get_map_for_stmt(
                stmt_id, lex_tuple, int_sid,
                seq_lex_dim_names+all_conc_lex_dim_names)
            for stmt_id, lex_tuple, int_sid
            in zip(stmt_ids, lex_tuples_simplified, int_sids)
            ]

        # Create pairwise lex order map (pairwise only in the intra-thread case)
        lex_order_map = create_lex_order_map(
            dim_names=seq_lex_dim_names,
            in_dim_mark=BEFORE_MARK,
            )

        # Add lid/gid dims to lex order map
        lex_order_map = add_and_name_isl_dims(
            lex_order_map, dim_type.out, all_conc_lex_dim_names)
        lex_order_map = add_and_name_isl_dims(
            lex_order_map, dim_type.in_,
            append_mark_to_strings(all_conc_lex_dim_names, mark=BEFORE_MARK))
        # Constrain lid/gid vars to be equal (this is the intra-thread case)
        for var_name in all_conc_lex_dim_names:
            lex_order_map = add_eq_isl_constraint_from_names(
                lex_order_map, var_name, var_name+BEFORE_MARK)

        # Create statement instance ordering,
        # maps each statement instance to all statement instances occurring later
        sio_intra_thread = get_statement_ordering_map(
            *intra_thread_sched_maps,  # note, func accepts exactly two maps
            lex_order_map,
            before_mark=BEFORE_MARK,
            )

        # }}}

        # {{{  Create SIOs for intra-group case (gid0' == gid0, etc) and global case

        def _get_sched_maps_and_sio_for_conc_exec(
                stmt_inst_to_blex, blex_order_map, seq_blex_dim_names):
            # (Vars from outside func used here:
            # stmt_ids, int_sids, all_conc_lex_dim_names)

            # Use *unsimplified* lex tuples w/ blex map, which are already padded
            blex_tuples_padded = [stmt_inst_to_blex[stmt_id] for stmt_id in stmt_ids]

            sched_maps = [
                _get_map_for_stmt(
                    stmt_id, blex_tuple, int_sid,
                    seq_blex_dim_names+all_conc_lex_dim_names)  # all par names
                for stmt_id, blex_tuple, int_sid
                in zip(stmt_ids, blex_tuples_padded, int_sids)
                ]

            # Note that for the intra-group case, we already constrained GID
            # 'before' to equal GID 'after' earlier in _gather_blex_ordering_info()

            # Create statement instance ordering
            sio = get_statement_ordering_map(
                *sched_maps,  # note, func accepts exactly two maps
                blex_order_map,
                before_mark=BEFORE_MARK,
                )

            return sched_maps, sio

        pwsched_intra_group, sio_intra_group = _get_sched_maps_and_sio_for_conc_exec(
            stmt_inst_to_lblex, lblex_order_map, seq_lblex_dim_names)
        pwsched_global, sio_global = _get_sched_maps_and_sio_for_conc_exec(
            stmt_inst_to_gblex, gblex_order_map, seq_gblex_dim_names)

        # }}}

        # Store sched maps along with SIOs
        pairwise_sios[tuple(stmt_ids)] = StatementOrdering(
            sio_intra_thread=sio_intra_thread,
            sio_intra_group=sio_intra_group,
            sio_global=sio_global,
            pwsched_intra_thread=tuple(intra_thread_sched_maps),
            pwsched_intra_group=tuple(pwsched_intra_group),
            pwsched_global=tuple(pwsched_global),
            )

    # }}}

    return pairwise_sios

# }}}

# vim: foldmethod=marker

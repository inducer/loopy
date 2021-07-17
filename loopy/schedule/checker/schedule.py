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
dt = isl.dim_type.set


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
LEX_VAR_PREFIX = "%slex" % (LIN_CHECK_IDENTIFIER_PREFIX)
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


# {{{ get_pairwise_statement_orderings_inner

def get_pairwise_statement_orderings_inner(
        knl,
        lin_items,
        stmt_id_pairs,
        loops_to_ignore=frozenset(),
        ):
    r"""For each statement pair in a subset of all statement pairs found in a
    linearized kernel, determine the (relative) order in which the statement
    instances are executed. For each pair, represent this relative ordering as
    a ``statement instance ordering`` (SIO): a map from each instance of the
    first statement to all instances of the second statement that occur
    later.

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

    :arg loops_to_ignore: A set of inames that will be ignored when
        determining the relative ordering of statements. This will typically
        contain concurrent inames tagged with the ``vec`` or ``ilp`` array
        access tags.

    :returns: A dictionary mapping each two-tuple of statement identifiers
        provided in `stmt_id_pairs` to a :class:`collections.namedtuple`
        containing the intra-thread SIO (`sio_intra_thread`), intra-group SIO
        (`sio_intra_group`), and global SIO (`sio_global`), each realized
        as an :class:`islpy.Map` from each instance of the first
        statement to all instances of the second statement that occur later,
        as well as the intra-thread pairwise schedule (`pwsched_intra_thread`),
        intra-group pairwise schedule (`pwsched_intra_group`), and the global
        pairwise schedule (`pwsched_global`), each containing a pair of
        mappings from statement instances to points in a lexicographic
        ordering, one for each statement. Note that a pairwise schedule
        alone cannot be used to reproduce the corresponding SIO without the
        corresponding (unique) lexicographic order map, which is not returned.
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
        add_eq_isl_constraint_from_names,
        sorted_union_of_names_in_isl_sets,
        create_symbolic_map_from_tuples,
        insert_and_name_isl_dims,
    )
    slex = SpecialLexPointWRTLoop

    all_stmt_ids = set().union(*stmt_id_pairs)

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
    loops_with_barriers = {"local": set(), "global": set()}
    current_inames = set()

    for lin_item in lin_items:
        if isinstance(lin_item, EnterLoop):
            iname = lin_item.iname
            current_inames.add(iname)

            if iname in loops_to_ignore:
                continue

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
            current_inames.remove(iname)

            if iname in loops_to_ignore:
                continue

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
            loops_with_barriers[lin_item.synchronization_kind] |= current_inames

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
    lid_lex_dim_names = set()
    gid_lex_dim_names = set()
    par_iname_constraint_dicts = {}
    for iname in knl.all_inames():
        ltag = knl.iname_tags_of_type(iname, LocalInameTag)
        if ltag:
            assert len(ltag) == 1  # (should always be true)
            ltag_var = LTAG_VAR_NAMES[ltag.pop().axis]
            lid_lex_dim_names.add(ltag_var)
            par_iname_constraint_dicts[iname] = {1: 0, iname: 1, ltag_var: -1}

            continue  # Shouldn't be any GroupInameTags

        gtag = knl.iname_tags_of_type(iname, GroupInameTag)
        if gtag:
            assert len(gtag) == 1  # (should always be true)
            gtag_var = GTAG_VAR_NAMES[gtag.pop().axis]
            gid_lex_dim_names.add(gtag_var)
            par_iname_constraint_dicts[iname] = {1: 0, iname: 1, gtag_var: -1}

    # Sort for consistent dimension ordering
    lid_lex_dim_names = sorted(lid_lex_dim_names)
    gid_lex_dim_names = sorted(gid_lex_dim_names)

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

    # {{{ Get upper and lower bound for each loop that contains a barrier

    iname_bounds_pwaff = {}
    for iname in loops_with_barriers["local"] | loops_with_barriers["global"]:
        bounds = knl.get_iname_bounds(iname)
        iname_bounds_pwaff[iname] = (
            bounds.lower_bound_pw_aff, bounds.upper_bound_pw_aff)

    # }}}

    # {{{ Create blex order maps and blex tuples defining statement ordering (x2)

    all_par_lex_dim_names = lid_lex_dim_names + gid_lex_dim_names

    # {{{ _gather_blex_ordering_info(sync_kind): gather blex info for sync_kind

    def _gather_blex_ordering_info(sync_kind):
        """For the given sync_kind ("local" or "global"), create a mapping from
        statement instances to blex space (dict), as well as a mapping
        defining the blex ordering (isl map from blex space -> blex space)

        Note that, unlike in the intra-thread case, there will be a single
        blex ordering map defining the blex ordering for all statement pairs,
        rather than separate (smaller) lex ordering maps for each pair
        """

        # {{{ First, create map from stmt instances to blex space.

        # At the same time, gather information necessary to create the
        # blex ordering map, i.e., for each loop, gather the 6 lex order tuples
        # defined above in SpecialLexPointWRTLoop that will be required to
        # create sub-maps which will be *excluded* (subtracted) from a standard
        # lexicographic ordering in order to create the blex ordering

        stmt_inst_to_blex = {}  # Map stmt instances to blex space
        iname_to_blex_dim = {}  # Map from inames to corresponding blex space dim
        blex_exclusion_info = {}  # Info for creating maps to exclude from blex order
        blex_order_map_params = set()  # Params needed in blex order map
        n_seq_blex_dims = 1  # Num dims representing sequential order in blex space
        next_blex_tuple = [0]  # Next tuple of points in blex order

        for lin_item in lin_items:
            if isinstance(lin_item, EnterLoop):
                enter_iname = lin_item.iname
                if enter_iname in loops_with_barriers[sync_kind] - loops_to_ignore:
                    pre_loop_blex_pt = next_blex_tuple[:]

                    # Increment next_blex_tuple[-1] for statements in the section
                    # of code between this EnterLoop and the matching LeaveLoop.
                    next_blex_tuple[-1] += 1

                    # Upon entering a loop, add one blex dimension for the loop
                    # iteration, add second blex dim to enumerate sections of
                    # code within new loop
                    next_blex_tuple.append(enter_iname)
                    next_blex_tuple.append(0)

                    # Store 3 tuples that will be used later to create pairs
                    # that will later be subtracted from the blex order map
                    lbound = iname_bounds_pwaff[enter_iname][0]
                    first_iter_blex_pt = next_blex_tuple[:]
                    first_iter_blex_pt[-2] = lbound
                    blex_exclusion_info[enter_iname] = {
                        slex.PRE: tuple(pre_loop_blex_pt),
                        slex.TOP: tuple(next_blex_tuple),
                        slex.FIRST: tuple(first_iter_blex_pt),
                        }
                    # (make sure ^these are copies)

                    # Store any new params found
                    blex_order_map_params |= set(lbound.get_var_names(dt.param))

            elif isinstance(lin_item, LeaveLoop):
                leave_iname = lin_item.iname
                if leave_iname in loops_with_barriers[sync_kind] - loops_to_ignore:

                    # Update max blex dims
                    n_seq_blex_dims = max(n_seq_blex_dims, len(next_blex_tuple))

                    # Record the blex dim for this loop iname
                    iname_to_blex_dim[leave_iname] = len(next_blex_tuple)-2

                    # Update next blex pt
                    pre_end_loop_blex_pt = next_blex_tuple[:]
                    # Upon leaving a loop:
                    # - Pop lex dim for enumerating code sections within this loop
                    # - Pop lex dim for the loop iteration
                    # - Increment lex dim val enumerating items in current section
                    next_blex_tuple.pop()
                    next_blex_tuple.pop()
                    next_blex_tuple[-1] += 1

                    # Store 3 tuples that will be used later to create pairs
                    # that will later be subtracted from the blex order map
                    ubound = iname_bounds_pwaff[leave_iname][1]
                    last_iter_blex_pt = pre_end_loop_blex_pt[:]
                    last_iter_blex_pt[-2] = ubound
                    blex_exclusion_info[leave_iname][slex.BOTTOM] = tuple(
                        pre_end_loop_blex_pt)
                    blex_exclusion_info[leave_iname][slex.LAST] = tuple(
                        last_iter_blex_pt)
                    blex_exclusion_info[leave_iname][slex.POST] = tuple(
                        next_blex_tuple)
                    # (make sure ^these are copies)

                    # Store any new params found
                    blex_order_map_params |= set(ubound.get_var_names(dt.param))

            elif isinstance(lin_item, RunInstruction):
                # Add stmt->blex pair to stmt_inst_to_blex
                stmt_inst_to_blex[lin_item.insn_id] = tuple(next_blex_tuple)

                # (Don't increment blex dim val)

            elif isinstance(lin_item, Barrier):
                # Increment blex dim val if the sync scope matches
                if lin_item.synchronization_kind == sync_kind:
                    next_blex_tuple[-1] += 1

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
                    # If sync scope matches, give this barrier its own point in
                    # lex time and update blex tuple after barrier.
                    # Otherwise, add stmt->blex pair to stmt_inst_to_blex, but
                    # don't update the blex tuple (just like with any other
                    # stmt)
                    if lin_item.synchronization_kind == sync_kind:
                        stmt_inst_to_blex[lp_stmt_id] = tuple(next_blex_tuple)
                        next_blex_tuple[-1] += 1
                    else:
                        stmt_inst_to_blex[lp_stmt_id] = tuple(next_blex_tuple)
            else:
                from loopy.schedule import (CallKernel, ReturnFromKernel)
                # No action needed for these types of linearization item
                assert isinstance(
                    lin_item, (CallKernel, ReturnFromKernel))
                pass

        blex_order_map_params = sorted(blex_order_map_params)

        # At this point, some blex tuples may have more dimensions than others;
        # the missing dims are the fastest-updating dims, and their values should
        # be zero. Add them.
        for stmt, tup in stmt_inst_to_blex.items():
            stmt_inst_to_blex[stmt] = _pad_tuple_with_zeros(tup, n_seq_blex_dims)

        # }}}

        # {{{ Second, create the blex order map

        # {{{ Create the initial (pre-subtraction) blex order map

        # Create names for the blex dimensions for sequential loops
        seq_blex_dim_names = [
            LEX_VAR_PREFIX+str(i) for i in range(n_seq_blex_dims)]
        seq_blex_dim_names_prime = append_mark_to_strings(
            seq_blex_dim_names, mark=BEFORE_MARK)

        # Begin with the blex order map created as a standard lexicographical order
        blex_order_map = create_lex_order_map(
            dim_names=seq_blex_dim_names,
            in_dim_mark=BEFORE_MARK,
            )

        # Add LID/GID dims to blex order map
        blex_order_map = add_and_name_isl_dims(
            blex_order_map, dt.out, all_par_lex_dim_names)
        blex_order_map = add_and_name_isl_dims(
            blex_order_map, dt.in_,
            append_mark_to_strings(all_par_lex_dim_names, mark=BEFORE_MARK))
        if sync_kind == "local":
            # For intra-group case, constrain GID 'before' to equal GID 'after'
            for var_name in gid_lex_dim_names:
                blex_order_map = add_eq_isl_constraint_from_names(
                        blex_order_map, var_name, var_name+BEFORE_MARK)
        # (if sync_kind == "global", don't need constraints on LID/GID vars)

        # }}}

        # {{{ Subtract unwanted pairs from happens-before blex map

        # Create map from iname to corresponding blex dim name
        iname_to_blex_var = {}
        for iname, dim in iname_to_blex_dim.items():
            iname_to_blex_var[iname] = seq_blex_dim_names[dim]
            iname_to_blex_var[iname+BEFORE_MARK] = seq_blex_dim_names_prime[dim]

        # Add bounds params needed in blex map
        blex_order_map = add_and_name_isl_dims(
            blex_order_map, dt.param, blex_order_map_params)

        # Get a set representing blex_order_map space
        n_blex_dims = n_seq_blex_dims + len(all_par_lex_dim_names)
        blex_set_template = isl.align_spaces(
            isl.Map("[ ] -> { [ ] -> [ ] }"), blex_order_map
            ).move_dims(
            dt.in_, n_blex_dims, dt.out, 0, n_blex_dims
            ).domain()
        blex_set_affs = isl.affs_from_space(blex_set_template.space)

        # {{{ _create_excluded_map_for_iname

        def _create_excluded_map_for_iname(iname, key_lex_tuples):
            """Create the blex->blex pairs that must be subtracted from the
            initial blex order map for this particular loop using the 6 blex
            tuples in the key_lex_tuples:
            PRE->FIRST, BOTTOM(iname')->TOP(iname'+1), LAST->POST
            """

            # Note:
            # only key_lex_tuples[slex.FIRST] & key_lex_tuples[slex.LAST] are pwaffs

            # {{{ _create_blex_set_from_tuple_pair

            def _create_blex_set_from_tuple_pair(before, after, wrap_cond=False):
                """Given a before->after tuple pair in the key_lex_tuples, which may
                have dim vals described by ints, strings (inames), and pwaffs,
                create an ISL set in blex space that can be converted into
                the ISL map to be subtracted
                """
                # (Vars from outside func used here:
                # iname, blex_set_affs, blex_set_template, iname_to_blex_var,
                # n_seq_blex_dims, seq_blex_dim_names,
                # seq_blex_dim_names_prime)

                # Start with a set representing blex_order_map space
                blex_set = blex_set_template.copy()

                # Add marks to inames in the 'before' tuple
                # (all strings should be inames)
                before_prime = tuple(
                    v+BEFORE_MARK if isinstance(v, str) else v for v in before)
                before_padded = _pad_tuple_with_zeros(before_prime, n_seq_blex_dims)
                after_padded = _pad_tuple_with_zeros(after, n_seq_blex_dims)

                # Assign vals in the tuple to dims in the ISL set
                for dim_name, dim_val in zip(
                        seq_blex_dim_names_prime+seq_blex_dim_names,
                        before_padded+after_padded):

                    if isinstance(dim_val, int):
                        # Set idx to int val
                        blex_set &= blex_set_affs[dim_name].eq_set(
                            blex_set_affs[0]+dim_val)
                    elif isinstance(dim_val, str):
                        # This is an iname, set idx to corresponding blex var
                        blex_set &= blex_set_affs[dim_name].eq_set(
                            blex_set_affs[iname_to_blex_var[dim_val]])
                    else:
                        # This is a pwaff iname bound, align and intersect
                        assert isinstance(dim_val, isl.PwAff)
                        pwaff_aligned = isl.align_spaces(dim_val, blex_set_affs[0])
                        # (doesn't matter which blex_set_affs item we align to^)
                        blex_set &= blex_set_affs[dim_name].eq_set(pwaff_aligned)

                if wrap_cond:
                    # This is the BOTTOM->TOP pair, add condition i = i' + 1
                    blex_set &= blex_set_affs[iname_to_blex_var[iname]].eq_set(
                        blex_set_affs[iname_to_blex_var[iname+BEFORE_MARK]] + 1)

                return blex_set

            # }}} end _create_blex_set_from_tuple_pair()

            # Create pairs to be subtracted
            # (set will be converted to map)

            # Enter loop case: PRE->FIRST
            full_blex_set = _create_blex_set_from_tuple_pair(
                key_lex_tuples[slex.PRE], key_lex_tuples[slex.FIRST])
            # Wrap loop case: BOTTOM(iname')->TOP(iname'+1)
            full_blex_set |= _create_blex_set_from_tuple_pair(
                key_lex_tuples[slex.BOTTOM], key_lex_tuples[slex.TOP],
                wrap_cond=True)
            # Leave loop case: LAST->POST
            full_blex_set |= _create_blex_set_from_tuple_pair(
                key_lex_tuples[slex.LAST], key_lex_tuples[slex.POST])

            # Add condition to fix iteration value for *surrounding* loops (j = j')
            for surrounding_iname in key_lex_tuples[slex.PRE][1::2]:
                s_blex_var = iname_to_blex_var[surrounding_iname]
                full_blex_set &= blex_set_affs[s_blex_var].eq_set(
                    blex_set_affs[s_blex_var+BEFORE_MARK])

            # Convert blex set back to map
            return isl.Map.from_domain(full_blex_set).move_dims(
                dt.out, 0, dt.in_, n_blex_dims, n_blex_dims)

        # }}} end _create_excluded_map_for_iname()

        # Create map to subtract for each iname
        maps_to_subtract = []
        for iname, subdict in blex_exclusion_info.items():
            maps_to_subtract.append(_create_excluded_map_for_iname(iname, subdict))

        if maps_to_subtract:

            # Get union of maps
            map_to_subtract = maps_to_subtract[0]
            for other_map in maps_to_subtract[1:]:
                map_to_subtract |= other_map

            # Get transitive closure of maps
            map_to_subtract, closure_exact = map_to_subtract.transitive_closure()
            assert closure_exact  # TODO warn instead?

            # Subtract closure from blex order map
            blex_order_map = blex_order_map - map_to_subtract

        # }}}

        # }}}

        return (
            stmt_inst_to_blex,  # map stmt instances to blex space
            blex_order_map,
            seq_blex_dim_names,
            )

    # }}} end _gather_blex_ordering_info(sync_kind)

    # Get the blex schedule blueprint (dict will become a map below) and
    # blex order map w.r.t. local and global barriers
    (stmt_inst_to_lblex,
     lblex_order_map,
     seq_lblex_dim_names) = _gather_blex_ordering_info("local")
    (stmt_inst_to_gblex,
     gblex_order_map,
     seq_gblex_dim_names) = _gather_blex_ordering_info("global")

    # }}}

    # }}}  end intra-group and global blex order creation

    # {{{ Create pairwise schedules (ISL maps) for each stmt pair

    # {{{ _get_map_for_stmt()

    def _get_map_for_stmt(
            stmt_id, lex_points, int_sid, lex_dim_names):

        # Get inames domain for statement instance (a BasicSet)
        within_inames = knl.id_to_insn[stmt_id].within_inames
        dom = knl.get_inames_domain(
            within_inames).project_out_except(within_inames, [dt.set])

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
                dom, dt.set, [STATEMENT_VAR_NAME], 0)

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

        # Set inames equal to relevant gid/lid var names
        for iname, constraint_dict in par_iname_constraint_dicts.items():
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
    from collections import namedtuple
    StatementOrdering = namedtuple(
        "StatementOrdering",
        [
            "sio_intra_thread", "pwsched_intra_thread",
            "sio_intra_group", "pwsched_intra_group",
            "sio_global", "pwsched_global",
        ])
    # ("sio" = statement instance ordering; "pwsched" = pairwise schedule)

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
                seq_lex_dim_names+all_par_lex_dim_names)
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
            lex_order_map, dt.out, all_par_lex_dim_names)
        lex_order_map = add_and_name_isl_dims(
            lex_order_map, dt.in_,
            append_mark_to_strings(all_par_lex_dim_names, mark=BEFORE_MARK))
        # Constrain lid/gid vars to be equal
        for var_name in all_par_lex_dim_names:
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

        def _get_sched_maps_and_sio(
                stmt_inst_to_blex, blex_order_map, seq_blex_dim_names):
            # (Vars from outside func used here:
            # stmt_ids, int_sids, all_par_lex_dim_names)

            # Use *unsimplified* lex tuples w/ blex map, which are already padded
            blex_tuples_padded = [stmt_inst_to_blex[stmt_id] for stmt_id in stmt_ids]

            par_sched_maps = [
                _get_map_for_stmt(
                    stmt_id, blex_tuple, int_sid,
                    seq_blex_dim_names+all_par_lex_dim_names)  # all par names
                for stmt_id, blex_tuple, int_sid
                in zip(stmt_ids, blex_tuples_padded, int_sids)
                ]

            # Create statement instance ordering
            sio_par = get_statement_ordering_map(
                *par_sched_maps,  # note, func accepts exactly two maps
                blex_order_map,
                before_mark=BEFORE_MARK,
                )

            return par_sched_maps, sio_par

        pwsched_intra_group, sio_intra_group = _get_sched_maps_and_sio(
            stmt_inst_to_lblex, lblex_order_map, seq_lblex_dim_names)
        pwsched_global, sio_global = _get_sched_maps_and_sio(
            stmt_inst_to_gblex, gblex_order_map, seq_gblex_dim_names)

        # }}}

        # Store sched maps along with SIOs
        pairwise_sios[tuple(stmt_ids)] = StatementOrdering(
            sio_intra_thread=sio_intra_thread,
            pwsched_intra_thread=tuple(intra_thread_sched_maps),
            sio_intra_group=sio_intra_group,
            pwsched_intra_group=tuple(pwsched_intra_group),
            sio_global=sio_global,
            pwsched_global=tuple(pwsched_global),
            )

    # }}}

    return pairwise_sios

# }}}

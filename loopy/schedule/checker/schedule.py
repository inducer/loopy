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

__doc__ = """

.. data:: LIN_CHECK_IDENTIFIER_PREFIX

    The prefix for identifiers involved in linearization checking.

.. data:: LEX_VAR_PREFIX

    E.g., a prefix of ``_lp_linchk_lex`` might yield lexicographic dimension
    variables ``_lp_linchk_lex0``, ``_lp_linchk_lex1``, ``_lp_linchk_lex2``. Cf.
    :ref:`reserved-identifiers`.

.. data:: STATEMENT_VAR_NAME

    Set the :class:`str` specifying the prefix to be used for the variables
    representing the dimensions in the lexicographic ordering used in a
    pairwise schedule.

"""

LIN_CHECK_IDENTIFIER_PREFIX = "_lp_linchk_"
LEX_VAR_PREFIX = "%slex" % (LIN_CHECK_IDENTIFIER_PREFIX)
STATEMENT_VAR_NAME = "%sstmt" % (LIN_CHECK_IDENTIFIER_PREFIX)
# TODO document:
GTAG_VAR_NAMES = []
LTAG_VAR_NAMES = []
for par_level in [0, 1, 2]:
    GTAG_VAR_NAMES.append("%sgid%d" % (LIN_CHECK_IDENTIFIER_PREFIX, par_level))
    LTAG_VAR_NAMES.append("%slid%d" % (LIN_CHECK_IDENTIFIER_PREFIX, par_level))


def _pad_tuple_with_zeros(tup, desired_length):
    return tup[:] + tuple([0]*(desired_length-len(tup)))


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


def generate_pairwise_schedules(
        knl,
        linearization_items,
        insn_id_pairs,
        loops_to_ignore=set(),
        ):
    r"""For each statement pair in a subset of all statement pairs found in a
    linearized kernel, determine the (relative) order in which the statement
    instances are executed. For each pair, describe this relative ordering with
    a pair of mappings from statement instances to points in a single
    lexicographic ordering (a ``pairwise schedule'').

    :arg knl: A preprocessed :class:`loopy.kernel.LoopKernel` containing the
        linearization items that will be used to create a schedule. This
        kernel will be used to get the domains associated with the inames
        used in the statements.

    :arg linearization_items: A list of :class:`loopy.schedule.ScheduleItem`
        (to be renamed to `loopy.schedule.LinearizationItem`) containing
        all linearization items for which pairwise schedules will be
        created. To allow usage of this routine during linearization, a
        truncated (i.e. partial) linearization may be passed through this
        argument.

    :arg insn_id_pairs: A list containing pairs of instruction identifiers.

    :arg loops_to_ignore: A set of inames that will be ignored when
        determining the relative ordering of statements. This will typically
        contain concurrent inames tagged with the ``vec`` or ``ilp`` array
        access tags.

    :returns: A dictionary mapping each two-tuple of instruction identifiers
        provided in `insn_id_pairs` to a corresponding two-tuple containing two
        :class:`islpy.Map`\ s representing a pairwise schedule as two
        mappings from statement instances to lexicographic time, one for
        each of the two statements.
    """
    # TODO update doc

    from loopy.schedule import (EnterLoop, LeaveLoop, Barrier, RunInstruction)
    from loopy.kernel.data import (LocalIndexTag, GroupIndexTag)

    all_insn_ids = set().union(*insn_id_pairs)

    # First, use one pass through linearization_items to generate a lexicographic
    # ordering describing the relative order of *all* statements represented by
    # all_insn_ids

    # For each statement, map the insn_id to a tuple representing points
    # in the lexicographic ordering containing items of :class:`int` or
    # :class:`str` :mod:`loopy` inames.
    stmt_instances = {}

    # Keep track of the next tuple of points in our lexicographic
    # ordering, initially this as a 1-d point with value 0
    next_insn_lex_tuple = [0]

    for linearization_item in linearization_items:
        if isinstance(linearization_item, EnterLoop):
            iname = linearization_item.iname
            if iname in loops_to_ignore:
                continue

            # Increment next_insn_lex_tuple[-1] for statements in the section
            # of code after this EnterLoop.
            # (not technically necessary if no statement was added in the
            # previous section; gratuitous incrementing is counteracted
            # in the simplification step below)
            next_insn_lex_tuple[-1] += 1

            # Upon entering a loop, add one lex dimension for the loop variable,
            # add second lex dim to enumerate sections of code within new loop
            next_insn_lex_tuple.append(iname)
            next_insn_lex_tuple.append(0)

        elif isinstance(linearization_item, LeaveLoop):
            if linearization_item.iname in loops_to_ignore:
                continue

            # Upon leaving a loop,
            # pop lex dimension for enumerating code sections within this loop, and
            # pop lex dimension for the loop variable, and
            # increment lex dim val enumerating items in current section of code
            next_insn_lex_tuple.pop()
            next_insn_lex_tuple.pop()

            # Increment next_insn_lex_tuple[-1] for statements in the section
            # of code after this LeaveLoop.
            # (not technically necessary if no statement was added in the
            # previous section; gratuitous incrementing is counteracted
            # in the simplification step below)
            next_insn_lex_tuple[-1] += 1

        elif isinstance(linearization_item, (RunInstruction, Barrier)):
            from loopy.schedule.checker.utils import (
                get_insn_id_from_linearization_item,
            )
            lp_insn_id = get_insn_id_from_linearization_item(linearization_item)

            if lp_insn_id is None:
                assert isinstance(linearization_item, Barrier)

                # Barriers without insn ids were inserted as a result of a
                # dependency. They don't themselves have dependencies. Ignore them.

                # FIXME: It's possible that we could record metadata about them
                # (e.g. what dependency produced them) and verify that they're
                # adequately protecting all statement instance pairs.

                continue

            # Only process listed insns, otherwise ignore
            if lp_insn_id in all_insn_ids:
                # Add item to stmt_instances
                stmt_instances[lp_insn_id] = tuple(next_insn_lex_tuple)

                # Increment lex dim val enumerating items in current section of code
                next_insn_lex_tuple[-1] += 1

        else:
            from loopy.schedule import (CallKernel, ReturnFromKernel)
            # No action needed for these types of linearization item
            assert isinstance(
                linearization_item, (CallKernel, ReturnFromKernel))
            pass

        # To save time, stop when we've found all statements
        if len(stmt_instances.keys()) == len(all_insn_ids):
            break

    # Second, create pairwise schedules for each individual pair of insns

    # Get dim names representing local/group axes for this kernel,
    # and get the dictionary that will be used later to create a
    # constraint requiring {par inames == par axes} in sched
    l_axes_used = set()
    g_axes_used = set()
    par_iname_constraint_dicts = []
    for iname in knl.all_inames():
        ltag = knl.iname_tags_of_type(iname, LocalIndexTag)
        if ltag:
            # assert len(ltag) == 1  # (should always be true)
            ltag_var = LTAG_VAR_NAMES[ltag.pop().axis]
            l_axes_used.add(ltag_var)
            # Represent constraint 'iname = ltag_var' in par_iname_constraint_dicts:
            par_iname_constraint_dicts.append({1: 0, iname: 1, ltag_var: -1})
            continue
        gtag = knl.iname_tags_of_type(iname, GroupIndexTag)
        if gtag:
            # assert len(gtag) == 1  # (should always be true)
            gtag_var = GTAG_VAR_NAMES[gtag.pop().axis]
            g_axes_used.add(gtag_var)
            # Represent constraint 'iname = gtag_var' in par_iname_constraint_dicts:
            par_iname_constraint_dicts.append({1: 0, iname: 1, gtag_var: -1})
            continue
    conc_lex_dim_names = sorted(l_axes_used) + sorted(g_axes_used)

    from loopy.schedule.checker.utils import (
        sorted_union_of_names_in_isl_sets,
        create_symbolic_map_from_tuples,
        add_dims_to_isl_set,
    )

    def _get_map_for_stmt(
            insn_id, lex_points, int_sid, seq_lex_dim_names, conc_lex_dim_names):

        # Get inames domain for statement instance (a BasicSet)
        dom = knl.get_inames_domain(
            knl.id_to_insn[insn_id].within_inames)

        # Create map space (an isl space in current implementation)
        # {('statement', <inames used in statement domain>) ->
        #  (lexicographic ordering dims)}
        dom_inames_ordered = sorted_union_of_names_in_isl_sets([dom])

        in_names_sched = [STATEMENT_VAR_NAME] + dom_inames_ordered[:]
        sched_space = isl.Space.create_from_names(
            isl.DEFAULT_CONTEXT,
            in_=in_names_sched,
            out=seq_lex_dim_names+conc_lex_dim_names,
            params=[],
            )

        # Insert 'statement' dim into domain so that its space allows
        # for intersection with sched map later
        dom_to_intersect = add_dims_to_isl_set(
                dom, isl.dim_type.set, [STATEMENT_VAR_NAME], 0)

        # Each map will map statement instances -> lex time.
        # Right now, statement instance tuples consist of single int.
        # Add all inames from domains to each map domain tuple.
        tuple_pair = [(
            (int_sid, ) + tuple(dom_inames_ordered),
            lex_points
            )]

        # Create map
        sched_map = create_symbolic_map_from_tuples(
            tuple_pairs_with_domains=zip(tuple_pair, [dom_to_intersect, ]),
            space=sched_space,
            )

        # Set inames equal to relevant gid/lid var names
        for constraint_dict in par_iname_constraint_dicts:
            sched_map = sched_map.add_constraint(
                isl.Constraint.eq_from_names(sched_map.space, constraint_dict))

        return sched_map

    from loopy.schedule.checker.lexicographic_order_map import (
        create_lex_order_map,
    )

    pairwise_schedules = {}
    for insn_ids in insn_id_pairs:
        lex_tuples = [stmt_instances[insn_id] for insn_id in insn_ids]

        # Simplify tuples to the extent possible ------------------------------------

        # At this point, one of the lex tuples may have more dimensions than another;
        # the missing dims are the fastest-updating dims, and their values should
        # be zero. Add them.
        max_lex_dims = max([len(lex_tuple) for lex_tuple in lex_tuples])
        lex_tuples_padded = [
            _pad_tuple_with_zeros(lex_tuple, max_lex_dims)
            for lex_tuple in lex_tuples]

        lex_tuples_simplified = _simplify_lex_dims(*lex_tuples_padded)

        # Now generate maps from the blueprint --------------------------------------

        # Create names for the output dimensions for sequential loops
        seq_lex_dim_names = [
            LEX_VAR_PREFIX+str(i) for i in range(len(lex_tuples_simplified[0]))]

        # Determine integer IDs that will represent each statement in mapping
        # (dependency map creation assumes sid_before=0 and sid_after=1, unless
        # before and after refer to same stmt, in which case sid_before=sid_after=0)
        int_sids = [0, 0] if insn_ids[0] == insn_ids[1] else [0, 1]

        sched_maps = [
            _get_map_for_stmt(
                insn_id, lex_tuple, int_sid, seq_lex_dim_names, conc_lex_dim_names)
            for insn_id, lex_tuple, int_sid
            in zip(insn_ids, lex_tuples_simplified, int_sids)
            ]

        # TODO (moved func below up here to avoid passing extra info around)
        # Benefit (e.g.): don't want to examine the schedule tuple in separate func
        # below to re-determine which parallel
        # dims are used. (could simplify everything by always using all dims, which
        # would make maps more complex than necessary)
        lex_order_map = create_lex_order_map(
            after_names=seq_lex_dim_names,
            after_names_concurrent=conc_lex_dim_names,
            )

        pairwise_schedules[tuple(insn_ids)] = (tuple(sched_maps), lex_order_map)

    return pairwise_schedules

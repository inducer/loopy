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

    E.g., a prefix of "_lp_linchk_lex" might yield lexicographic dimension
    variables "_lp_linchk_lex0", "_lp_linchk_lex1", "_lp_linchk_lex2". Cf.
    :ref:`reserved-identifiers`.

.. data:: STATEMENT_VAR_NAME

    Set the :class:`str` specifying the prefix to be used for the variables
    representing the dimensions in the lexicographic ordering used in a
    pairwise schedule.

"""

LIN_CHECK_IDENTIFIER_PREFIX = "_lp_linchk_"
LEX_VAR_PREFIX = "%sl" % (LIN_CHECK_IDENTIFIER_PREFIX)
STATEMENT_VAR_NAME = "%sstatement" % (LIN_CHECK_IDENTIFIER_PREFIX)


def generate_pairwise_schedules(
        knl,
        linearization_items,
        insn_id_pairs,
        loops_to_ignore=set(),
        ):
    r"""Given a pair of statements in a linearized kernel, determine
    the (relative) order in which the instances are executed,
    by creating a mapping from statement instances to points in a single
    lexicographic ordering. Create a pair of :class:`islpy.Map`\ s
    representing a pairwise schedule as two mappings from statement instances
    to lexicographic time.

    :arg knl: A :class:`loopy.kernel.LoopKernel` containing the
        linearization items that will be described by the schedule. This
        kernel will be used to get the domains associated with the inames
        used in the statements.

    :arg linearization_items: A list of :class:`loopy.schedule.ScheduleItem`
        (to be renamed to `loopy.schedule.LinearizationItem`) including the
        two linearization items whose relative order will be described by the
        schedule. This list may be a *partial* linearization for a kernel since
        this function may be used during the linearization process.

    :arg before_insn_id: A :class:`str` instruction id specifying
        stmt_instance_set_before in this pair of instructions.

    :arg after_insn_id: A :class:`str` instruction id specifying
        stmt_instance_set_after in this pair of instructions.

    :returns: A two-tuple containing two :class:`islpy.Map`\ s
        representing a pairwise schedule as two mappings
        from statement instances to lexicographic time, one for
        each of the two statements.
    """

    # TODO
    # update documentation

    all_insn_ids = set().union(*insn_id_pairs)

    # For each statement, map the insn_id to a tuple representing points
    # in the lexicographic ordering containing items of :class:`int` or
    # :class:`str` :mod:`loopy` inames.
    stmt_instances = {}

    from loopy.schedule import (EnterLoop, LeaveLoop, Barrier, RunInstruction)

    # go through linearization_items and generate pairwise sub-schedule

    # keep track of the next tuple of points in our lexicographic
    # ordering, initially this as a 1-d point with value 0
    next_insn_lex_tuple = [0]
    stmt_added_since_prev_block_at_tier = [False]
    for linearization_item in linearization_items:
        if isinstance(linearization_item, EnterLoop):
            iname = linearization_item.iname
            if iname in loops_to_ignore:
                continue

            # We could always increment next_insn_lex_tuple[-1] here since
            # this new section of code comes after the previous section
            # (statements since last opened/closed loop), but if we have
            # not added any statements within the previous section yet, we
            # don't have to (effectively ignoring that section of code).
            if stmt_added_since_prev_block_at_tier[-1]:
                next_insn_lex_tuple[-1] = next_insn_lex_tuple[-1]+1
                stmt_added_since_prev_block_at_tier[-1] = False

            # upon entering a loop, we enter a new (deeper) tier,
            # add one lex dimension for the loop variable,
            # add second lex dim to enumerate code blocks within new loop, and
            # append a dim to stmt_added_since_prev_block_at_tier to represent
            # new tier
            next_insn_lex_tuple.append(iname)
            next_insn_lex_tuple.append(0)
            stmt_added_since_prev_block_at_tier.append(False)
        elif isinstance(linearization_item, LeaveLoop):
            if linearization_item.iname in loops_to_ignore:
                continue
            # upon leaving a loop,
            # pop lex dimension for enumerating code blocks within this loop, and
            # pop lex dimension for the loop variable, and
            # increment lex dim val enumerating items in current code block
            next_insn_lex_tuple.pop()
            next_insn_lex_tuple.pop()

            # We could always increment next_insn_lex_tuple[-1] here since
            # this new block of code comes after the previous block (all
            # statements since last opened/closed loop), but if we have not
            # added any statements within the previous section yet, we
            # don't have to (effectively ignoring that section of code).
            # TODO since we're getting rid of unnecessary dims later, maybe don't need this?
            stmt_added_since_prev_block_at_tier.pop()
            if stmt_added_since_prev_block_at_tier[-1]:
                next_insn_lex_tuple[-1] = next_insn_lex_tuple[-1]+1
                stmt_added_since_prev_block_at_tier[-1] = False
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

            # only process listed insns, otherwise ignore
            if lp_insn_id in all_insn_ids:
                # add item
                stmt_instances[lp_insn_id] = tuple(next_insn_lex_tuple[:])

                # increment lex dim val enumerating items in current code block
                next_insn_lex_tuple[-1] = next_insn_lex_tuple[-1] + 1

                # all current (nested) blocks now contain a statement
                stmt_added_since_prev_block_at_tier = [True]*len(
                    stmt_added_since_prev_block_at_tier)
        else:
            from loopy.schedule import (CallKernel, ReturnFromKernel)
            # no action needed for these types of linearization item
            assert isinstance(
                linearization_item, (CallKernel, ReturnFromKernel))
            pass

        # to save time, stop when we've created all statements
        if len(stmt_instances.keys()) == all_insn_ids:
            break

    from loopy.schedule.checker.utils import (
        sorted_union_of_names_in_isl_sets,
        create_symbolic_map_from_tuples,
        add_dims_to_isl_set,
    )

    def _pad_tuple_with_zeros(tup, length):
        return tup[:] + tuple([0]*(length-len(tup)))

    def _simplify_dims(tup0, tup1):
        new_tup0 = []
        new_tup1 = []
        # loop over dims
        for d0, d1 in zip(tup0, tup1):
            if isinstance(d0, int) and isinstance(d1, int):
                # Both vals are ints for this dim

                # If the ints match, this dim doesn't provide info about the
                # relative ordering of these two statements,
                # so skip (remove) this dim.

                # Otherwise, the ints inform us about the relative ordering of
                # two statements. While their values may be larger than 1 in
                # the lexicographic ordering describing a larger set of
                # statements, in a pairwise schedule, only ints 0 and 1 are
                # necessary to specify relative order. To keep the pairwise
                # schedules as simple and comprehensible as possible, use only
                # integers 0 and 1 to specify relative orderings in integer lex
                # dims.
                # (doesn't take much extra time since we are already going
                # through these to remove unnecessary map dims)

                if d0 == d1:
                    continue
                elif d0 > d1:
                    new_tup0.append(1)
                    new_tup1.append(0)
                else:  # d1 > d0
                    new_tup0.append(0)
                    new_tup1.append(1)
            else:
                # keep this dim
                new_tup0.append(d0)
                new_tup1.append(d1)
        return tuple(new_tup0), tuple(new_tup1)

    def _get_map_for_stmt_inst(insn_id, lex_points, int_sid, out_names_sched):

        # Get inames domain for statement instance (a BasicSet)
        dom = knl.get_inames_domain(
            knl.id_to_insn[insn_id].within_inames)

        # create space (an isl space in current implementation)
        # {('statement', <inames> used in statement domain>) ->
        #  (lexicographic ordering dims)}
        dom_inames_ordered = sorted_union_of_names_in_isl_sets([dom])

        in_names_sched = [STATEMENT_VAR_NAME] + dom_inames_ordered[:]
        sched_space = isl.Space.create_from_names(
            isl.DEFAULT_CONTEXT,
            in_=in_names_sched, out=out_names_sched, params=[])

        # Insert 'statement' dim into domain so that its space allows
        # for intersection with sched map later
        dom_to_intersect = [
            add_dims_to_isl_set(
                dom, isl.dim_type.set, [STATEMENT_VAR_NAME], 0), ]

        # Each map representing the schedule will map
        # statement instances -> lex time.
        # Right now, statement instance tuples consist of single int.
        # Add all inames from domains to each map domain tuple.
        tuple_pair = [(
            (int_sid, ) + tuple(dom_inames_ordered),
            lex_points
            )]

        # create map
        return create_symbolic_map_from_tuples(
            tuple_pairs_with_domains=zip(tuple_pair, dom_to_intersect),
            space=sched_space,
            )

    pairwise_schedules = []
    for insn_id_before, insn_id_after in insn_id_pairs:
        lex_tup_before = stmt_instances[insn_id_before]
        lex_tup_after = stmt_instances[insn_id_after]

        # simplify tuples to the extent possible ------------------------------------

        # At this point, pairwise sub-schedule may contain lex point tuples
        # missing dimensions; the values in these missing dims should
        # be zero, so add them.
        max_lex_dims = max(len(lex_tup_before), len(lex_tup_after))
        lex_tup_before = _pad_tuple_with_zeros(lex_tup_before, max_lex_dims)
        lex_tup_after = _pad_tuple_with_zeros(lex_tup_after, max_lex_dims)

        lex_tup_before, lex_tup_after = _simplify_dims(lex_tup_before, lex_tup_after)

        # Now generate maps from the blueprint --------------------------------------

        out_names_sched = [LEX_VAR_PREFIX+str(i) for i in range(len(lex_tup_before))]

        # Determine integer IDs that will represent each statement in mapping
        # (dependency map creation assumes sid_before=0 and sid_after=1, unless
        # before and after refer to same stmt, in which case sid_before=sid_after=0)
        int_sid_before = 0
        int_sid_after = 0 if insn_id_before == insn_id_after else 1

        map_before = _get_map_for_stmt_inst(
            insn_id_before, lex_tup_before, int_sid_before, out_names_sched)
        map_after = _get_map_for_stmt_inst(
            insn_id_after, lex_tup_after, int_sid_after, out_names_sched)

        pairwise_schedules.append((map_before, map_after))

    return pairwise_schedules

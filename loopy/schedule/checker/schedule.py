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


def generate_pairwise_schedule(
        knl,
        linearization_items,
        before_insn_id,
        after_insn_id,
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
        representing the a pairwise schedule as two mappings
        from statement instances to lexicographic time, one for
        each of the two statements.
    """

    # For each statement, create a :class:`ImmutableRecord` describing the set of
    # statement instances. Contains the insn_id and a list representing points
    # in the lexicographic ordering containing items of :class:`int` or
    # :class:`str` :mod:`loopy` inames.
    stmt_instance_set_before = None
    stmt_instance_set_after = None

    from loopy.schedule import (EnterLoop, LeaveLoop, Barrier, RunInstruction)
    from pytools import ImmutableRecord

    # go through linearization_items and generate pairwise sub-schedule

    # keep track of the next tuple of points in our lexicographic
    # ordering, initially this as a 1-d point with value 0
    next_insn_lex_tuple = [0]
    stmt_added_since_prev_block_at_tier = [False]
    max_lex_dim = 0
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

            # only process before/after insns, otherwise ignore
            stmt_added = False

            if lp_insn_id == before_insn_id:
                # add before sched item
                stmt_instance_set_before = ImmutableRecord(
                        insn_id=lp_insn_id,
                        lex_points=tuple(next_insn_lex_tuple[:]))
                stmt_added = True

            if lp_insn_id == after_insn_id:
                # add after sched item
                stmt_instance_set_after = ImmutableRecord(
                        insn_id=lp_insn_id,
                        lex_points=tuple(next_insn_lex_tuple[:]))
                stmt_added = True

            # Note: before/after may refer to same stmt, in which case
            # both of the above conditionals execute

            if stmt_added:

                # track the max number of lex dims used
                if len(next_insn_lex_tuple) > max_lex_dim:
                    max_lex_dim = len(next_insn_lex_tuple)

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
        # to save time, stop when we've created both statements
        if stmt_instance_set_before and stmt_instance_set_after:
            break

    # At this point, pairwise sub-schedule may contain lex point tuples
    # missing dimensions; the values in these missing dims should
    # be zero, so add them.

    def _pad_lex_tuple_with_zeros(stmt_inst, length):
        return ImmutableRecord(
            insn_id=stmt_inst.insn_id,
            lex_points=stmt_inst.lex_points[:] + tuple(
                [0]*(length-len(stmt_inst.lex_points)))
            )

    stmt_instance_set_before = _pad_lex_tuple_with_zeros(
        stmt_instance_set_before, max_lex_dim)
    stmt_instance_set_after = _pad_lex_tuple_with_zeros(
        stmt_instance_set_after, max_lex_dim)

    # Now generate maps from the blueprint ---------------------------------------

    from loopy.schedule.checker.utils import (
        list_var_names_in_isl_sets,
        create_symbolic_map_from_tuples,
        add_dims_to_isl_set,
    )

    params_sched = []
    out_names_sched = [LEX_VAR_PREFIX+str(i) for i in range(max_lex_dim)]

    def _get_map_for_stmt_inst(stmt_inst, int_sid):

        # Get inames domain for statement instance (a BasicSet)
        dom = knl.get_inames_domain(
            knl.id_to_insn[stmt_inst.insn_id].within_inames)

        # create space (an isl space in current implementation)
        # {('statement', <inames> used in statement domain>) ->
        #  (lexicographic ordering dims)}
        dom_inames_ordered = list_var_names_in_isl_sets([dom])

        in_names_sched = [STATEMENT_VAR_NAME] + dom_inames_ordered[:]
        sched_space = isl.Space.create_from_names(
            isl.DEFAULT_CONTEXT,
            in_=in_names_sched, out=out_names_sched, params=params_sched)

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
            stmt_inst.lex_points
            )]

        # create map
        return create_symbolic_map_from_tuples(
            tuple_pairs_with_domains=zip(tuple_pair, dom_to_intersect),
            space=sched_space,
            )

    # Determine integer IDs that will represent each statement in mapping
    # (dependency map creation assumes sid_before=0 and sid_after=1, unless
    # before and after refer to same stmt, in which case sid_before=sid_after=0)
    int_sid_before = 0
    int_sid_after = 0 if (
        stmt_instance_set_before.insn_id == stmt_instance_set_after.insn_id
        ) else 1

    map_before = _get_map_for_stmt_inst(stmt_instance_set_before, int_sid_before)
    map_after = _get_map_for_stmt_inst(stmt_instance_set_after, int_sid_after)

    return (map_before, map_after)

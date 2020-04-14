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


class LexScheduleStatement(object):
    """A representation of a :mod:`loopy` statement.

    .. attribute:: insn_id

       A :class:`str` specifying the instruction id.

    .. attribute:: int_id

       A :class:`int` uniquely identifying the instruction.

    .. attribute:: within_inames

       A :class:`list` of :class:`str` inames identifying the loops within
       which this statement will be executed.

    """

    def __init__(
            self,
            insn_id,  # loopy insn id
            int_id=None,  # sid int (statement id within LexSchedule)
            within_inames=None,  # [string, ]
            ):
        self.insn_id = insn_id  # string
        self.int_id = int_id
        self.within_inames = within_inames

    def __eq__(self, other):
        return (
            self.insn_id == other.insn_id
            and self.int_id == other.int_id
            and self.within_inames == other.within_inames
            )

    def __hash__(self):
        return hash(repr(self))

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        key_builder.rec(key_hash, self.insn_id)
        key_builder.rec(key_hash, self.int_id)
        key_builder.rec(key_hash, self.within_inames)

    def __str__(self):
        if self.int_id is not None:
            int_id = ":%d" % (self.int_id)
        else:
            int_id = ""
        if self.within_inames:
            within_inames = " {%s}" % (",".join(self.within_inames))
        else:
            within_inames = ""
        return "%s%s%s" % (
            self.insn_id, int_id, within_inames)


class LexScheduleStatementInstance(object):
    """A representation of a :mod:`loopy` statement instance.

    .. attribute:: stmt

       A :class:`LexScheduleStatement`.

    .. attribute:: lex_pt

       A list of :class:`int` or as :class:`str` :mod:`loopy` inames representing
       a point or set of points in a lexicographic ordering.

    """

    def __init__(
            self,
            stmt,  # a LexScheduleStatement
            lex_pt,  # [string/int, ]
            ):
        self.stmt = stmt
        self.lex_pt = lex_pt

    def __str__(self):
        return "{%s, %s}" % (self.stmt, self.lex_pt)


class LexSchedule(object):
    """A program ordering represented as a mapping from statement
       instances to points in a lexicographic ordering.

    .. attribute:: stmt_instance_before

       A :class:`LexScheduleStatementInstance` describing the dependee
       statement's order relative to the depender statment by mapping
       a statement to a point or set of points in a lexicographic
       ordering. Points in lexicographic ordering are represented as
       a list of :class:`int` or as :class:`str` :mod:`loopy` inames.

    .. attribute:: stmt_instance_after

       A :class:`LexScheduleStatementInstance` describing the depender
       statement's order relative to the dependee statment by mapping
       a statement to a point or set of points in a lexicographic
       ordering. Points in lexicographic ordering are represented as
       a list of :class:`int` or as :class:`str` :mod:`loopy` inames.

    .. attribute:: statement_var_name

       A :class:`str` specifying the name of the isl variable used
       to represent the unique :class:`int` statement id.

    .. attribute:: lex_var_prefix

       A :class:`str` specifying the prefix to be used for the variables
       representing the dimensions in the lexicographic ordering. E.g.,
       a prefix of "lex" might yield variables "lex0", "lex1", "lex2".

    """

    statement_var_name = "statement"
    lex_var_prefix = "l"

    def __init__(
            self,
            linearization_items_ordered,
            before_insn_id,
            after_insn_id,
            prohibited_var_names=[],
            loops_to_ignore=set(),
            ):
        """
        :arg linearization_items_ordered: A list of :class:`ScheduleItem` whose
            order will be described by this :class:`LexSchedule`.

        :arg before_insn_id: A :class:`str` instruction id specifying
            the dependee in this pair of instructions.

        :arg after_insn_id: A :class:`str` instruction id specifying
            the depender in this pair of instructions.

        :arg prohibited_var_names: A list of :class:`str` variable names
            that may not be used as the statement variable name (e.g.,
            because they are already being used as inames).

        """

        # LexScheduleStatements
        self.stmt_instance_before = None
        self.stmt_instance_after = None

        # make sure we don't have an iname name conflict
        # TODO use loopy's existing tool for ensuring unique var names
        assert not any(
            iname == self.statement_var_name for iname in prohibited_var_names)

        from loopy.schedule import (EnterLoop, LeaveLoop, Barrier, RunInstruction)

        # go through linearization_items_ordered and generate self.lex_schedule

        # keep track of the next point in our lexicographic ordering
        # initially this as a 1-d point with value 0
        next_insn_lex_pt = [0]
        stmt_since_last_block_at_tier = [False]
        next_sid = 0
        stmt_added_since_last_EnterLoop = False
        stmt_added_since_last_LeaveLoop = False
        #stmt_added_since_last_new_block = False  # blocks start at open/close loop
        for linearization_item in linearization_items_ordered:
            if isinstance(linearization_item, EnterLoop):
                iname = linearization_item.iname
                if iname in loops_to_ignore:
                    continue

                # We could always increment next_insn_lex_pt[-1] here since this new
                # section of code comes after the previous section (statements
                # since last opened/closed loop), but if we have not added any statements
                # within this block yet, we don't have to
                # (effectively ignoring that section of code).
                if stmt_since_last_block_at_tier[-1]:
                    next_insn_lex_pt[-1] = next_insn_lex_pt[-1]+1
                    stmt_since_last_block_at_tier[-1] = False

                # upon entering a loop, we enter a new (deeper) tier,
                # add one lex dimension for the loop variable,
                # add second lex dim to enumerate code blocks within new loop, and
                # append a dim to stmt_since_last_block_at_tier to represent new tier
                next_insn_lex_pt.append(iname)
                next_insn_lex_pt.append(0)
                stmt_since_last_block_at_tier.append(False)
            elif isinstance(linearization_item, LeaveLoop):
                if linearization_item.iname in loops_to_ignore:
                    continue
                # upon leaving a loop,
                # pop lex dimension for enumerating code blocks within this loop, and
                # pop lex dimension for the loop variable, and
                # increment lex dim val enumerating items in current code block
                next_insn_lex_pt.pop()
                next_insn_lex_pt.pop()

                # We could always increment next_insn_lex_pt[-1] here since this new
                # block of code comes after the previous block (all statements
                # since last opened/closed loop), but if we have not added any statements
                # within this block yet, we don't have to
                # (effectively ignoring that section of code).
                stmt_since_last_block_at_tier.pop()
                if stmt_since_last_block_at_tier[-1]:
                    next_insn_lex_pt[-1] = next_insn_lex_pt[-1]+1
                    stmt_since_last_block_at_tier[-1] = False
            elif isinstance(linearization_item, (RunInstruction, Barrier)):
                from loopy.schedule.checker.utils import (
                    _get_insn_id_from_linearization_item,
                )
                lp_insn_id = _get_insn_id_from_linearization_item(linearization_item)
                if lp_insn_id is None:
                    # TODO make sure it's okay to ignore barriers without id
                    # (because they'll never be part of a dependency?)
                    # matmul example has barrier that fails this assertion...
                    # assert linearization_item.originating_insn_id is not None
                    continue

                # only process before/after insns, otherwise ignore
                if lp_insn_id == before_insn_id and lp_insn_id == after_insn_id:
                    # add before sched item
                    self.stmt_instance_before = LexScheduleStatementInstance(
                            LexScheduleStatement(
                                insn_id=lp_insn_id,
                                int_id=next_sid,  # int representing insn
                                ),
                            next_insn_lex_pt[:])
                    # add after sched item
                    self.stmt_instance_after = LexScheduleStatementInstance(
                            LexScheduleStatement(
                                insn_id=lp_insn_id,
                                int_id=next_sid,  # int representing insn
                                ),
                            next_insn_lex_pt[:])

                    # increment lex dim val enumerating items in current code block
                    next_insn_lex_pt[-1] = next_insn_lex_pt[-1] + 1
                    next_sid += 1

                    # all current (nested) blocks now contain a statement
                    stmt_since_last_block_at_tier = [True]*len(stmt_since_last_block_at_tier)
                elif lp_insn_id == before_insn_id:
                    # add before sched item
                    self.stmt_instance_before = LexScheduleStatementInstance(
                            LexScheduleStatement(
                                insn_id=lp_insn_id,
                                int_id=next_sid,  # int representing insn
                                ),
                            next_insn_lex_pt[:])

                    # increment lex dim val enumerating items in current code block
                    next_insn_lex_pt[-1] = next_insn_lex_pt[-1] + 1
                    next_sid += 1

                    # all current (nested) blocks now contain a statement
                    stmt_since_last_block_at_tier = [True]*len(stmt_since_last_block_at_tier)
                elif lp_insn_id == after_insn_id:
                    # add after sched item
                    self.stmt_instance_after = LexScheduleStatementInstance(
                            LexScheduleStatement(
                                insn_id=lp_insn_id,
                                int_id=next_sid,  # int representing insn
                                ),
                            next_insn_lex_pt[:])

                    # increment lex dim val enumerating items in current code block
                    next_insn_lex_pt[-1] = next_insn_lex_pt[-1] + 1
                    next_sid += 1

                    # all current (nested) blocks now contain a statement
                    stmt_since_last_block_at_tier = [True]*len(stmt_since_last_block_at_tier)
            else:
                pass
            # to save time, stop when we've created both statements
            if self.stmt_instance_before and self.stmt_instance_after:
                break

        # at this point, lex_schedule may contain lex points missing dimensions,
        # the values in these missing dims should be zero, so add them
        self.pad_lex_pts_with_zeros()

    def loopy_insn_id_to_lex_sched_id(self):
        """Return a dictionary mapping insn_id to int_id, where ``insn_id`` and
            ``int_id`` refer to the ``insn_id`` and ``int_id`` attributes of
            :class:`LexScheduleStatement`.
        """
        return {
            self.stmt_instance_before.stmt.insn_id:
                self.stmt_instance_before.stmt.int_id,
            self.stmt_instance_after.stmt.insn_id:
                self.stmt_instance_after.stmt.int_id,
            }

    def max_lex_dims(self):
        return max([
            len(self.stmt_instance_before.lex_pt),
            len(self.stmt_instance_after.lex_pt)])

    def pad_lex_pts_with_zeros(self):
        """Find the maximum number of lexicographic dimensions represented
            in the lexicographic ordering, and if any
            :class:`LexScheduleStatement` maps to a point in lexicographic
            time with fewer dimensions, add a zero for each of the missing
            dimensions.
        """

        max_lex_dim = self.max_lex_dims()
        self.stmt_instance_before = LexScheduleStatementInstance(
            self.stmt_instance_before.stmt,
            self.stmt_instance_before.lex_pt[:] + [0]*(
                max_lex_dim-len(self.stmt_instance_before.lex_pt))
            )
        self.stmt_instance_after = LexScheduleStatementInstance(
            self.stmt_instance_after.stmt,
            self.stmt_instance_after.lex_pt[:] + [0]*(
                max_lex_dim-len(self.stmt_instance_after.lex_pt))
            )

    def create_isl_maps(
            self,
            dom_before,
            dom_after,
            dom_inames_ordered_before=None,
            dom_inames_ordered_after=None,
            ):
        """Create two isl maps representing lex schedule as two mappings
            from statement instances to lexicographic time, one for
            the dependee and one for the depender.

        :arg dom_before: A :class:`islpy.BasicSet` representing the
            domain for the dependee statement.

        :arg dom_after: A :class:`islpy.BasicSet` representing the
            domain for the dependee statement.

        :arg dom_inames_ordered_before: A list of :class:`str`
            representing the union of inames used in instances of the
            dependee statement. ``statement_var_name`` and
            ``dom_inames_ordered_before`` are the names of the dims of
            the space of the ISL map domain for the dependee.

        :arg dom_inames_ordered_after: A list of :class:`str`
            representing the union of inames used in instances of the
            depender statement. ``statement_var_name`` and
            ``dom_inames_ordered_after`` are the names of the dims of
            the space of the ISL map domain for the depender.

        :returns: A two-tuple containing two :class:`islpy.Map`s
            representing the schedule as two mappings
            from statement instances to lexicographic time, one for
            the dependee and one for the depender.

        """

        from loopy.schedule.checker.utils import (
            create_symbolic_isl_map_from_tuples,
            add_dims_to_isl_set
        )

        from loopy.schedule.checker.utils import (
            list_var_names_in_isl_sets,
        )
        if dom_inames_ordered_before is None:
            dom_inames_ordered_before = list_var_names_in_isl_sets(
                [dom_before])
        if dom_inames_ordered_after is None:
            dom_inames_ordered_after = list_var_names_in_isl_sets(
                [dom_after])

        # create an isl space
        # {('statement', <inames> used in >=1 statement domain>) ->
        #  (lexicographic ordering dims)}
        from loopy.schedule.checker.utils import (
            get_isl_space
        )
        params_sched = []
        out_names_sched = self.get_lex_var_names()

        in_names_sched_before = [
            self.statement_var_name] + dom_inames_ordered_before[:]
        sched_space_before = get_isl_space(
            params_sched, in_names_sched_before, out_names_sched)
        in_names_sched_after = [
            self.statement_var_name] + dom_inames_ordered_after[:]
        sched_space_after = get_isl_space(
            params_sched, in_names_sched_after, out_names_sched)

        # Insert 'statement' dim into domain so that its space allows for
        # intersection with sched map later
        doms_to_intersect_before = [
                add_dims_to_isl_set(
                    dom_before, isl.dim_type.set,
                    [self.statement_var_name], 0),
                ]
        doms_to_intersect_after = [
                add_dims_to_isl_set(
                    dom_after, isl.dim_type.set,
                    [self.statement_var_name], 0),
                ]

        # Each isl map representing the schedule maps
        # statement instances -> lex time

        # Right now, statement tuples consist of single int.
        # Add all inames from domains to map domain tuples.

        # create isl map
        return (
            create_symbolic_isl_map_from_tuples(
                zip(
                    [(
                        (self.stmt_instance_before.stmt.int_id,)
                        + tuple(dom_inames_ordered_before),
                        self.stmt_instance_before.lex_pt
                    )],
                    doms_to_intersect_before
                ),
                sched_space_before, self.statement_var_name),
            create_symbolic_isl_map_from_tuples(
                zip(
                    [(
                        (self.stmt_instance_after.stmt.int_id,)
                        + tuple(dom_inames_ordered_after),
                        self.stmt_instance_after.lex_pt)],
                    doms_to_intersect_after
                ),
                sched_space_after, self.statement_var_name)
            )

    def get_lex_var_names(self):
        return [self.lex_var_prefix+str(i)
                for i in range(self.max_lex_dims())]

    def __eq__(self, other):
        return (
            self.stmt_instance_before == other.stmt_instance_before
            and self.stmt_instance_after == other.stmt_instance_after)

    def __str__(self):
        sched_str = "Before: {\n"
        domain_elem = "[%s=%s,<inames>]" % (
            self.statement_var_name,
            self.stmt_instance_before.stmt.int_id)
        sched_str += "%s -> %s;\n" % (domain_elem, self.stmt_instance_before.lex_pt)
        sched_str += "}\n"

        sched_str += "After: {\n"
        domain_elem = "[%s=%s,<inames>]" % (
            self.statement_var_name,
            self.stmt_instance_after.stmt.int_id)
        sched_str += "%s -> %s;\n" % (domain_elem, self.stmt_instance_after.lex_pt)
        sched_str += "}"
        return sched_str

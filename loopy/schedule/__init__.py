from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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


import six
from pytools import ImmutableRecord
import sys
import islpy as isl
from loopy.diagnostic import warn_with_kernel, LoopyError  # noqa

from pytools import memoize_method, MinRecursionLimit, ProcessLogger

from pytools.persistent_dict import WriteOncePersistentDict
from loopy.tools import LoopyKeyBuilder
from loopy.version import DATA_MODEL_VERSION

import logging
logger = logging.getLogger(__name__)


# {{{ schedule items

class ScheduleItem(ImmutableRecord):
    __slots__ = []

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """
        for field_name in self.hash_fields:
            key_builder.rec(key_hash, getattr(self, field_name))


class BeginBlockItem(ScheduleItem):
    pass


class EndBlockItem(ScheduleItem):
    pass


class EnterLoop(BeginBlockItem):
    hash_fields = __slots__ = ["iname"]


class LeaveLoop(EndBlockItem):
    hash_fields = __slots__ = ["iname"]


class RunInstruction(ScheduleItem):
    hash_fields = __slots__ = ["insn_id"]


class CallKernel(BeginBlockItem):
    hash_fields = __slots__ = ["kernel_name", "extra_args", "extra_inames"]


class ReturnFromKernel(EndBlockItem):
    hash_fields = __slots__ = ["kernel_name"]


class Barrier(ScheduleItem):
    """
    .. attribute:: comment

        A plain-text comment explaining why the barrier was inserted.

    .. attribute:: synchronization_kind

        ``"local"`` or ``"global"``

    .. attribute:: mem_kind

        ``"local"`` or ``"global"``

    .. attribute:: originating_insn_id
    """

    hash_fields = ["comment", "synchronization_kind", "mem_kind"]
    __slots__ = hash_fields + ["originating_insn_id"]

# }}}


# {{{ schedule utilities

def gather_schedule_block(schedule, start_idx):
    assert isinstance(schedule[start_idx], BeginBlockItem)
    level = 0

    i = start_idx
    while i < len(schedule):
        if isinstance(schedule[i], BeginBlockItem):
            level += 1
        elif isinstance(schedule[i], EndBlockItem):
            level -= 1

            if level == 0:
                return schedule[start_idx:i+1], i+1

        i += 1

    assert False


def generate_sub_sched_items(schedule, start_idx):
    if not isinstance(schedule[start_idx], BeginBlockItem):
        yield start_idx, schedule[start_idx]

    level = 0
    i = start_idx
    while i < len(schedule):
        sched_item = schedule[i]
        if isinstance(sched_item, BeginBlockItem):
            level += 1

        elif isinstance(sched_item, EndBlockItem):
            level -= 1

        else:
            yield i, sched_item

        if level == 0:
            return

        i += 1

    assert False


def get_insn_ids_for_block_at(schedule, start_idx):
    return frozenset(
            sub_sched_item.insn_id
            for i, sub_sched_item in generate_sub_sched_items(
                schedule, start_idx)
            if isinstance(sub_sched_item, RunInstruction))


def find_active_inames_at(kernel, sched_index):
    active_inames = []

    from loopy.schedule import EnterLoop, LeaveLoop
    for sched_item in kernel.schedule[:sched_index]:
        if isinstance(sched_item, EnterLoop):
            active_inames.append(sched_item.iname)
        if isinstance(sched_item, LeaveLoop):
            active_inames.pop()

    return set(active_inames)


def has_barrier_within(kernel, sched_index):
    sched_item = kernel.schedule[sched_index]

    if isinstance(sched_item, BeginBlockItem):
        loop_contents, _ = gather_schedule_block(
                kernel.schedule, sched_index)
        from pytools import any
        return any(isinstance(subsched_item, Barrier)
                for subsched_item in loop_contents)
    elif isinstance(sched_item, Barrier):
        return True
    else:
        return False


def find_used_inames_within(kernel, sched_index):
    sched_item = kernel.schedule[sched_index]

    if isinstance(sched_item, BeginBlockItem):
        loop_contents, _ = gather_schedule_block(
                kernel.schedule, sched_index)
        run_insns = [subsched_item
                for subsched_item in loop_contents
                if isinstance(subsched_item, RunInstruction)]
    elif isinstance(sched_item, RunInstruction):
        run_insns = [sched_item]
    else:
        return set()

    result = set()
    for sched_item in run_insns:
        result.update(kernel.insn_inames(sched_item.insn_id))

    return result


def find_loop_nest_with_map(kernel):
    """Returns a dictionary mapping inames to other inames that are
    always nested with them.
    """
    result = {}

    from loopy.kernel.data import ConcurrentTag, IlpBaseTag

    all_nonpar_inames = set(
            iname for iname in kernel.all_inames()
            if not kernel.iname_tags_of_type(iname,
                    (ConcurrentTag, IlpBaseTag)))

    iname_to_insns = kernel.iname_to_insns()

    for iname in all_nonpar_inames:
        result[iname] = set(other_iname
            for insn in iname_to_insns[iname]
            for other_iname in kernel.insn_inames(insn) & all_nonpar_inames)

    return result


def find_loop_nest_around_map(kernel):
    """Returns a dictionary mapping inames to other inames that are
    always nested around them.
    """
    result = {}

    all_inames = kernel.all_inames()

    iname_to_insns = kernel.iname_to_insns()

    # examine pairs of all inames--O(n**2), I know.
    from loopy.kernel.data import IlpBaseTag
    for inner_iname in all_inames:
        result[inner_iname] = set()
        for outer_iname in all_inames:
            if inner_iname == outer_iname:
                continue

            if kernel.iname_tags_of_type(outer_iname, IlpBaseTag):
                # ILP tags are special because they are parallel tags
                # and therefore 'in principle' nest around everything.
                # But they're realized by the scheduler as a loop
                # at the innermost level, so we'll cut them some
                # slack here.
                continue

            if iname_to_insns[inner_iname] < iname_to_insns[outer_iname]:
                result[inner_iname].add(outer_iname)

    for dom_idx, dom in enumerate(kernel.domains):
        for outer_iname in dom.get_var_names(isl.dim_type.param):
            if outer_iname not in all_inames:
                continue

            for inner_iname in dom.get_var_names(isl.dim_type.set):
                result[inner_iname].add(outer_iname)

    return result


def find_loop_insn_dep_map(kernel, loop_nest_with_map, loop_nest_around_map):
    """Returns a dictionary mapping inames to other instruction ids that need to
    be scheduled before the iname should be eligible for scheduling.
    """

    result = {}

    from loopy.kernel.data import ConcurrentTag, IlpBaseTag
    for insn in kernel.instructions:
        for iname in kernel.insn_inames(insn):
            if kernel.iname_tags_of_type(iname, ConcurrentTag):
                continue

            iname_dep = result.setdefault(iname, set())

            for dep_insn_id in insn.depends_on:
                if dep_insn_id in iname_dep:
                    # already depending, nothing to check
                    continue

                dep_insn = kernel.id_to_insn[dep_insn_id]
                dep_insn_inames = kernel.insn_inames(dep_insn)

                if iname in dep_insn_inames:
                    # Nothing to be learned, dependency is in loop over iname
                    # already.
                    continue

                # To make sure dep_insn belongs outside of iname, we must prove
                # that all inames that dep_insn will be executed in nest
                # outside of the loop over *iname*. (i.e. nested around, or
                # before).

                may_add_to_loop_dep_map = True
                for dep_insn_iname in dep_insn_inames:
                    if dep_insn_iname in loop_nest_around_map[iname]:
                        # dep_insn_iname is guaranteed to nest outside of iname
                        # -> safe.
                        continue

                    if kernel.iname_tags_of_type(dep_insn_iname,
                                (ConcurrentTag, IlpBaseTag)):
                        # Parallel tags don't really nest, so we'll disregard
                        # them here.
                        continue

                    if dep_insn_iname not in loop_nest_with_map.get(iname, []):
                        # dep_insn_iname does not nest with iname, so its nest
                        # must occur outside.
                        continue

                    may_add_to_loop_dep_map = False
                    break

                if not may_add_to_loop_dep_map:
                    continue

                logger.debug("{knl}: loop dependency map: iname '{iname}' "
                        "depends on '{dep_insn}' via '{insn}'"
                        .format(
                            knl=kernel.name,
                            iname=iname,
                            dep_insn=dep_insn_id,
                            insn=insn.id))

                iname_dep.add(dep_insn_id)

    return result


def group_insn_counts(kernel):
    result = {}

    for insn in kernel.instructions:
        for grp in insn.groups:
            result[grp] = result.get(grp, 0) + 1

    return result


def gen_dependencies_except(kernel, insn_id, except_insn_ids):
    insn = kernel.id_to_insn[insn_id]
    for dep_id in insn.depends_on:

        if dep_id in except_insn_ids:
            continue

        yield dep_id

        for sub_dep_id in gen_dependencies_except(kernel, dep_id, except_insn_ids):
            yield sub_dep_id


def get_priority_tiers(wanted, priorities):
    # Get highest priority tier candidates: These are the first inames
    # of all the given priority constraints
    candidates = set()
    for prio in priorities:
        for p in prio:
            if p in wanted:
                candidates.add(p)
                break

    # Now shrink this set by removing those inames that are prohibited
    # by other constraints
    bad_candidates = []
    for c1 in candidates:
        for c2 in candidates:
            for prio in priorities:
                try:
                    if prio.index(c1) < prio.index(c2):
                        bad_candidates.append(c2)
                except ValueError:
                    # A ValueError in tuple.index just states that one of
                    # the candidates is not present in the priority constraint
                    pass
    candidates = candidates - set(bad_candidates)

    if candidates:
        # We found a valid priority tier
        yield candidates
    else:
        # If we did not, stop the generator
        return

    # Now reduce the input data for recursion
    priorities = frozenset([tuple(i for i in prio if i not in candidates)
                            for prio in priorities
                            ]) - frozenset([()])
    wanted = wanted - candidates

    # Yield recursively
    for tier in get_priority_tiers(wanted, priorities):
        yield tier


def sched_item_to_insn_id(sched_item):
    # Helper for use in generator expressions, i.e.
    # (... for insn_id in sched_item_to_insn_id(item) ...)
    if isinstance(sched_item, RunInstruction):
        yield sched_item.insn_id
    elif isinstance(sched_item, Barrier):
        if (hasattr(sched_item, "originating_insn_id")
                and sched_item.originating_insn_id is not None):
            yield sched_item.originating_insn_id

# }}}


# {{{ debug help

def format_insn_id(kernel, insn_id):
    Fore = kernel.options._fore  # noqa
    Style = kernel.options._style  # noqa
    return Fore.GREEN + insn_id + Style.RESET_ALL


def format_insn(kernel, insn_id):
    insn = kernel.id_to_insn[insn_id]
    Fore = kernel.options._fore  # noqa
    Style = kernel.options._style  # noqa
    from loopy.kernel.instruction import (
            MultiAssignmentBase, NoOpInstruction, BarrierInstruction)
    if isinstance(insn, MultiAssignmentBase):
        return "%s%s%s = %s%s%s  {id=%s}" % (
            Fore.CYAN, ", ".join(str(a) for a in insn.assignees), Style.RESET_ALL,
            Fore.MAGENTA, str(insn.expression), Style.RESET_ALL,
            format_insn_id(kernel, insn_id))
    elif isinstance(insn, BarrierInstruction):
        mem_kind = ''
        if insn.synchronization_kind == 'local':
            mem_kind = '{mem_kind=%s}' % insn.mem_kind

        return "[%s] %s... %sbarrier%s%s" % (
                format_insn_id(kernel, insn_id),
                Fore.MAGENTA, insn.synchronization_kind[0], mem_kind,
                Style.RESET_ALL)
    elif isinstance(insn, NoOpInstruction):
        return "[%s] %s... nop%s" % (
                format_insn_id(kernel, insn_id),
                Fore.MAGENTA, Style.RESET_ALL)
    else:
        return "[%s] %s%s%s" % (
                format_insn_id(kernel, insn_id),
                Fore.CYAN, str(insn), Style.RESET_ALL)


def dump_schedule(kernel, schedule):
    lines = []
    indent = ""

    from loopy.kernel.data import MultiAssignmentBase
    for sched_item in schedule:
        if isinstance(sched_item, EnterLoop):
            lines.append(indent + "for %s" % sched_item.iname)
            indent += "    "
        elif isinstance(sched_item, LeaveLoop):
            indent = indent[:-4]
            lines.append(indent + "end %s" % sched_item.iname)
        elif isinstance(sched_item, CallKernel):
            lines.append(indent +
                         "CALL KERNEL %s(extra_args=%s, extra_inames=%s)" % (
                             sched_item.kernel_name,
                             sched_item.extra_args,
                             sched_item.extra_inames))
            indent += "    "
        elif isinstance(sched_item, ReturnFromKernel):
            indent = indent[:-4]
            lines.append(indent + "RETURN FROM KERNEL %s" % sched_item.kernel_name)
        elif isinstance(sched_item, RunInstruction):
            insn = kernel.id_to_insn[sched_item.insn_id]
            if isinstance(insn, MultiAssignmentBase):
                insn_str = format_insn(kernel, sched_item.insn_id)
            else:
                insn_str = sched_item.insn_id
            lines.append(indent + insn_str)
        elif isinstance(sched_item, Barrier):
            lines.append(indent + "... %sbarrier" %
                         sched_item.synchronization_kind[0])
        else:
            assert False

    return "\n".join(
            "% 4d: %s" % (i, line)
            for i, line in enumerate(lines))


class ScheduleDebugger:
    def __init__(self, debug_length=None, interactive=True):
        self.longest_rejected_schedule = []
        self.success_counter = 0
        self.dead_end_counter = 0
        self.debug_length = debug_length
        self.interactive = interactive

        self.elapsed_store = 0
        self.start()
        self.wrote_status = 0

        self.update()

    def update(self):
        if (
                (self.success_counter + self.dead_end_counter) % 50 == 0
                and self.elapsed_time() > 10
                ):
            sys.stdout.write("\rscheduling... %d successes, "
                    "%d dead ends (longest %d)" % (
                        self.success_counter,
                        self.dead_end_counter,
                        len(self.longest_rejected_schedule)))
            sys.stdout.flush()
            self.wrote_status = 2

    def log_success(self, schedule):
        self.success_counter += 1
        self.update()

    def log_dead_end(self, schedule):
        if len(schedule) > len(self.longest_rejected_schedule):
            self.longest_rejected_schedule = schedule
        self.dead_end_counter += 1
        self.update()

    def done_scheduling(self):
        if self.wrote_status:
            sys.stdout.write("\rscheduler finished"+40*" "+"\n")
            sys.stdout.flush()

    def elapsed_time(self):
        from time import time
        return self.elapsed_store + time() - self.start_time

    def stop(self):
        if self.wrote_status == 2:
            sys.stdout.write("\r"+80*" "+"\n")
            self.wrote_status = 1

        from time import time
        self.elapsed_store += time()-self.start_time

    def start(self):
        from time import time
        self.start_time = time()


class ScheduleDebugInput(Exception):
    pass

# }}}


# {{{ scheduling algorithm

class SchedulerState(ImmutableRecord):
    """
    .. attribute:: kernel

    .. attribute:: loop_nest_around_map

    .. attribute:: loop_priority

        #FIXME: incorrect docs.
        See :func:`loop_nest_around_map`.

    .. attribute:: breakable_inames

    .. attribute:: ilp_inames

    .. attribute:: vec_inames

    .. attribute:: parallel_inames

        *Note:* ``ilp`` and ``vec`` are not 'parallel' for the purposes of the
        scheduler.  See :attr:`ilp_inames`, :attr:`vec_inames`.

    .. rubric:: Time-varying scheduler state

    .. attribute:: insn_ids_to_try

        #FIXME: docs?

    .. attribute:: active_inames

        A tuple of active inames.

    .. attribute:: entered_inames

        A :class:`frozenset` of all inames ever entered.

    .. attribute:: enclosing_subkernel_inames

        The inames of the last entered subkernel

    .. attribute:: schedule

    .. attribute:: scheduled_insn_ids

    .. attribute:: unscheduled_insn_ids

    .. attribute:: preschedule

        A sequence of schedule items that must be inserted into the
        schedule, maintaining the same relative ordering. Newly scheduled
        items may interleave this sequence.

    .. attribute:: prescheduled_insn_ids

        A :class:`frozenset` of any instruction that started prescheduled

    .. attribute:: prescheduled_inames

        A :class:`frozenset` of any iname that started prescheduled

    .. attribute:: may_schedule_global_barriers

        Whether global barrier scheduling is allowed

    .. attribute:: within_subkernel

        Whether the scheduler is inside a subkernel

    .. attribute:: group_insn_counts

        A mapping from instruction group names to the number of instructions
        contained in them.

    .. attribute:: active_group_counts

        A mapping from instruction group names to the number of instructions
        in them that are left to schedule. If a group name occurs in this
        mapping, that group is considered active.
    """

    @property
    def last_entered_loop(self):
        if self.active_inames:
            return self.active_inames[-1]
        else:
            return None

    @memoize_method
    def get_insn_ids_in_a_topologically_sorted_order(self):
        from pytools.graph import compute_topological_order

        rev_dep_map = {insn.id: set() for insn in self.kernel.instructions}
        for insn in self.kernel.instructions:
            for dep in insn.depends_on:
                rev_dep_map[dep].add(insn.id)

        return compute_topological_order(rev_dep_map)


def schedule_as_many_run_insns_as_possible(sched_state):
    """
    Returns an instance of :class:`loopy.schedule.SchedulerState`, by appending
    all available instructions in the current loop nesting to the schedule.
    """

    next_preschedule_item = (
        sched_state.preschedule[0]
        if len(sched_state.preschedule) > 0
        else None)

    if isinstance(next_preschedule_item, (CallKernel, ReturnFromKernel, Barrier)):
        return sched_state

    if not sched_state.within_subkernel:
        # cannot schedule RunInstructions when not in subkernel
        return sched_state

    # {{{ topological sort

    toposorted_insn_ids = tuple(insn_id for insn_id in
            sched_state.get_insn_ids_in_a_topologically_sorted_order() if
            insn_id in sched_state.unscheduled_insn_ids and (
                sched_state.kernel.id_to_insn[insn_id].within_inames >=
                frozenset(sched_state.active_inames)))

    # }}}

    # select the top instructions in toposorted_insns only which have active
    # inames corresponding to those of sched_state
    from loopy.kernel.instruction import MultiAssignmentBase

    updated_sched_state = sched_state.copy()

    num_insns_to_be_scheduled = 0

    for insn_id in toposorted_insn_ids:
        insn = sched_state.kernel.id_to_insn[insn_id]
        if isinstance(insn, MultiAssignmentBase):
            if insn.within_inames == frozenset(sched_state.active_inames):
                num_insns_to_be_scheduled += 1
                continue
        break

    newly_scheduled_insn_ids = toposorted_insn_ids[:num_insns_to_be_scheduled]
    num_presched_insns_newly_scheduled = len(set(newly_scheduled_insn_ids) &
            sched_state.prescheduled_insn_ids)

    assert all(isinstance(sched_item, RunInstruction) and sched_item.insn_id in
            newly_scheduled_insn_ids for sched_item in
            sched_state.preschedule[:num_presched_insns_newly_scheduled])
    sched_items = tuple(RunInstruction(insn_id=insn_id) for insn_id in
            newly_scheduled_insn_ids)

    updated_schedule = updated_sched_state.schedule + sched_items
    updated_scheduled_insn_ids = (updated_sched_state.scheduled_insn_ids
            | frozenset(newly_scheduled_insn_ids))
    updated_unscheduled_insn_ids = (
            updated_sched_state.unscheduled_insn_ids
            - frozenset(newly_scheduled_insn_ids))
    updated_sched_state = updated_sched_state.copy(
            insn_ids_to_try=None,
            schedule=updated_schedule,
            scheduled_insn_ids=updated_scheduled_insn_ids,
            unscheduled_insn_ids=updated_unscheduled_insn_ids,
            preschedule=sched_state.preschedule[num_presched_insns_newly_scheduled:]
            )

    return updated_sched_state


def generate_loop_schedules_internal(
        sched_state, debug=None):
    # allow_insn is set to False initially and after entering each loop
    # to give loops containing high-priority instructions a chance.
    kernel = sched_state.kernel
    Fore = kernel.options._fore  # noqa
    Style = kernel.options._style  # noqa

    sched_state = (
            schedule_as_many_run_insns_as_possible(sched_state))

    active_inames_set = frozenset(sched_state.active_inames)

    next_preschedule_item = (
        sched_state.preschedule[0]
        if len(sched_state.preschedule) > 0
        else None)

    # {{{ decide about debug mode

    debug_mode = False

    if debug is not None:
        if (debug.debug_length is not None
                and len(sched_state.schedule) >= debug.debug_length):
            debug_mode = True

    if debug_mode:
        if debug.wrote_status == 2:
            print()
        print(75*"=")
        print("KERNEL:")
        print(kernel.stringify(with_dependencies=True))
        print(75*"=")
        print("CURRENT SCHEDULE:")
        print(dump_schedule(sched_state.kernel, sched_state.schedule))
        if sched_state.preschedule:
            print(75*"=")
            print("PRESCHEDULED ITEMS AWAITING SCHEDULING:")
            print(dump_schedule(sched_state.kernel, sched_state.preschedule))
        print(75*"=")
        print("LOOP NEST MAP (inner: outer):")
        for iname, val in six.iteritems(sched_state.loop_nest_around_map):
            print("%s : %s" % (iname, ", ".join(val)))
        print(75*"=")

        if debug.debug_length == len(debug.longest_rejected_schedule):
            print("WHY IS THIS A DEAD-END SCHEDULE?")

    #if len(schedule) == 2:
        #from pudb import set_trace; set_trace()

    # }}}

    # {{{ see if we have reached the start/end of kernel in the preschedule

    if isinstance(next_preschedule_item, CallKernel):
        assert sched_state.within_subkernel is False
        for result in generate_loop_schedules_internal(
                sched_state.copy(
                    schedule=sched_state.schedule + (next_preschedule_item,),
                    preschedule=sched_state.preschedule[1:],
                    within_subkernel=True,
                    may_schedule_global_barriers=False,
                    enclosing_subkernel_inames=sched_state.active_inames),
                debug=debug):
            yield result

    if isinstance(next_preschedule_item, ReturnFromKernel):
        assert sched_state.within_subkernel is True
        # Make sure all subkernel inames have finished.
        if sched_state.active_inames == sched_state.enclosing_subkernel_inames:
            for result in generate_loop_schedules_internal(
                    sched_state.copy(
                        schedule=sched_state.schedule + (next_preschedule_item,),
                        preschedule=sched_state.preschedule[1:],
                        within_subkernel=False,
                        may_schedule_global_barriers=True),
                    debug=debug):
                yield result

    # }}}

    # {{{ see if there are pending barriers in the preschedule

    # Barriers that do not have an originating instruction are handled here.
    # (These are automatically inserted by insert_barriers().) Barriers with
    # originating instructions are handled as part of normal instruction
    # scheduling below.
    if (
            isinstance(next_preschedule_item, Barrier)
            and next_preschedule_item.originating_insn_id is None):
        for result in generate_loop_schedules_internal(
                    sched_state.copy(
                        schedule=sched_state.schedule + (next_preschedule_item,),
                        preschedule=sched_state.preschedule[1:]),
                    debug=debug):
            yield result

    # }}}

    # {{{ see if any insns are ready to be scheduled now

    # Also take note of insns that have a chance of being schedulable inside
    # the current loop nest, in this set:

    reachable_insn_ids = set()
    active_groups = frozenset(sched_state.active_group_counts)

    def insn_sort_key(insn_id):
        insn = kernel.id_to_insn[insn_id]

        # Sort by insn.id as a last criterion to achieve deterministic
        # schedule generation order.
        return (insn.priority, len(active_groups & insn.groups), insn.id)

    # Use previous instruction sorting result if it is available
    if sched_state.insn_ids_to_try is None:
        insn_ids_to_try = sorted(
                # Non-prescheduled instructions go first.
                sched_state.unscheduled_insn_ids - sched_state.prescheduled_insn_ids,
                key=insn_sort_key, reverse=True)
    else:
        insn_ids_to_try = sched_state.insn_ids_to_try

    insn_ids_to_try.extend(
        insn_id
        for item in sched_state.preschedule
        for insn_id in sched_item_to_insn_id(item))

    for insn_id in insn_ids_to_try:
        insn = kernel.id_to_insn[insn_id]

        is_ready = insn.depends_on <= sched_state.scheduled_insn_ids

        if not is_ready:
            if debug_mode:
                # These are not that interesting when understanding scheduler
                # failures.

                # print("instruction '%s' is missing insn depedencies '%s'" % (
                #         format_insn(kernel, insn.id), ",".join(
                #             insn.depends_on - sched_state.scheduled_insn_ids)))
                pass
            continue

        want = kernel.insn_inames(insn) - sched_state.parallel_inames
        have = active_inames_set - sched_state.parallel_inames

        if want != have:
            is_ready = False

            if debug_mode:
                if want-have:
                    print("instruction '%s' is missing inames '%s'"
                            % (format_insn(kernel, insn.id), ",".join(want-have)))
                if have-want:
                    print("instruction '%s' won't work under inames '%s'"
                            % (format_insn(kernel, insn.id), ",".join(have-want)))

        # {{{ check if scheduling this insn is compatible with preschedule

        if insn_id in sched_state.prescheduled_insn_ids:
            if isinstance(next_preschedule_item, RunInstruction):
                next_preschedule_insn_id = next_preschedule_item.insn_id
            elif isinstance(next_preschedule_item, Barrier):
                assert next_preschedule_item.originating_insn_id is not None
                next_preschedule_insn_id = next_preschedule_item.originating_insn_id
            else:
                next_preschedule_insn_id = None

            if next_preschedule_insn_id != insn_id:
                if debug_mode:
                    print("can't schedule '%s' because another preschedule "
                          "instruction precedes it" % format_insn(kernel, insn.id))
                is_ready = False

        # }}}

        # {{{ check if scheduler state allows insn scheduling

        from loopy.kernel.instruction import BarrierInstruction
        if isinstance(insn, BarrierInstruction) and \
                insn.synchronization_kind == "global":
            if not sched_state.may_schedule_global_barriers:
                if debug_mode:
                    print("can't schedule '%s' because global barriers are "
                          "not currently allowed" % format_insn(kernel, insn.id))
                is_ready = False
        else:
            if not sched_state.within_subkernel:
                if debug_mode:
                    print("can't schedule '%s' because not within subkernel"
                          % format_insn(kernel, insn.id))
                is_ready = False

        # }}}

        # {{{ determine group-based readiness

        if insn.conflicts_with_groups & active_groups:
            is_ready = False

            if debug_mode:
                print("instruction '%s' conflicts with active group(s) '%s'"
                        % (insn.id, ",".join(
                            active_groups & insn.conflicts_with_groups)))

        # }}}

        # {{{ determine reachability

        if (not is_ready and have <= want):
            reachable_insn_ids.add(insn_id)

        # }}}

        if is_ready and debug_mode:
            print("ready to schedule '%s'" % format_insn(kernel, insn.id))

        if is_ready and not debug_mode:
            iid_set = frozenset([insn.id])

            # {{{ update active group counts for added instruction

            if insn.groups:
                new_active_group_counts = sched_state.active_group_counts.copy()

                for grp in insn.groups:
                    if grp in new_active_group_counts:
                        new_active_group_counts[grp] -= 1
                        if new_active_group_counts[grp] == 0:
                            del new_active_group_counts[grp]

                    else:
                        new_active_group_counts[grp] = (
                                sched_state.group_insn_counts[grp] - 1)
            else:
                new_active_group_counts = sched_state.active_group_counts

            # }}}

            # {{{ update instruction_ids_to_try

            new_insn_ids_to_try = list(insn_ids_to_try)
            new_insn_ids_to_try.remove(insn.id)

            # invalidate instruction_ids_to_try when active group changes
            if set(new_active_group_counts.keys()) != set(
                    sched_state.active_group_counts.keys()):
                new_insn_ids_to_try = None

            # }}}

            new_sched_state = sched_state.copy(
                    scheduled_insn_ids=sched_state.scheduled_insn_ids | iid_set,
                    unscheduled_insn_ids=sched_state.unscheduled_insn_ids - iid_set,
                    insn_ids_to_try=new_insn_ids_to_try,
                    schedule=(
                        sched_state.schedule + (RunInstruction(insn_id=insn.id),)),
                    preschedule=(
                        sched_state.preschedule
                        if insn_id not in sched_state.prescheduled_insn_ids
                        else sched_state.preschedule[1:]),
                    active_group_counts=new_active_group_counts,
                    )

            # Don't be eager about entering/leaving loops--if progress has been
            # made, revert to top of scheduler and see if more progress can be
            # made.
            for sub_sched in generate_loop_schedules_internal(
                    new_sched_state,
                    debug=debug):
                yield sub_sched

            if not sched_state.group_insn_counts:
                # No groups: We won't need to backtrack on scheduling
                # instructions.
                return

    # }}}

    # {{{ see if we're ready to leave the innermost loop

    last_entered_loop = sched_state.last_entered_loop

    if last_entered_loop is not None:
        can_leave = True

        if (
                last_entered_loop in sched_state.prescheduled_inames
                and not (
                    isinstance(next_preschedule_item, LeaveLoop)
                    and next_preschedule_item.iname == last_entered_loop)):
            # A prescheduled loop can only be left if the preschedule agrees.
            if debug_mode:
                print("cannot leave '%s' because of preschedule constraints"
                      % last_entered_loop)
            can_leave = False
        elif last_entered_loop not in sched_state.breakable_inames:
            # If the iname is not breakable, then check that we've
            # scheduled all the instructions that require it.

            for insn_id in sched_state.unscheduled_insn_ids:
                insn = kernel.id_to_insn[insn_id]
                if last_entered_loop in kernel.insn_inames(insn):
                    if debug_mode:
                        print("cannot leave '%s' because '%s' still depends on it"
                                % (last_entered_loop, format_insn(kernel, insn.id)))

                        # check if there's a dependency of insn that needs to be
                        # outside of last_entered_loop.
                        for subdep_id in gen_dependencies_except(kernel, insn_id,
                                sched_state.scheduled_insn_ids):
                            want = (kernel.insn_inames(subdep_id)
                                    - sched_state.parallel_inames)
                            if (
                                    last_entered_loop not in want):
                                print(
                                    "%(warn)swarning:%(reset_all)s '%(iname)s', "
                                    "which the schedule is "
                                    "currently stuck inside of, seems mis-nested. "
                                    "'%(subdep)s' must occur " "before '%(dep)s', "
                                    "but '%(subdep)s must be outside "
                                    "'%(iname)s', whereas '%(dep)s' must be back "
                                    "in it.%(reset_all)s\n"
                                    "  %(subdep_i)s\n"
                                    "  %(dep_i)s"
                                    % {
                                        "warn": Fore.RED + Style.BRIGHT,
                                        "reset_all": Style.RESET_ALL,
                                        "iname": last_entered_loop,
                                        "subdep": format_insn_id(kernel, subdep_id),
                                        "dep": format_insn_id(kernel, insn_id),
                                        "subdep_i": format_insn(kernel, subdep_id),
                                        "dep_i": format_insn(kernel, insn_id),
                                        })

                    can_leave = False
                    break

        if can_leave:
            can_leave = False

            # We may only leave this loop if we've scheduled an instruction
            # since entering it.

            seen_an_insn = False
            ignore_count = 0
            for sched_item in sched_state.schedule[::-1]:
                if isinstance(sched_item, RunInstruction):
                    seen_an_insn = True
                elif isinstance(sched_item, LeaveLoop):
                    ignore_count += 1
                elif isinstance(sched_item, EnterLoop):
                    if ignore_count:
                        ignore_count -= 1
                    else:
                        assert sched_item.iname == last_entered_loop
                        if seen_an_insn:
                            can_leave = True
                        break

            if can_leave and not debug_mode:

                for sub_sched in generate_loop_schedules_internal(
                        sched_state.copy(
                            schedule=(
                                sched_state.schedule
                                + (LeaveLoop(iname=last_entered_loop),)),
                            active_inames=sched_state.active_inames[:-1],
                            preschedule=(
                                sched_state.preschedule
                                if last_entered_loop
                                not in sched_state.prescheduled_inames
                                else sched_state.preschedule[1:]),
                        ),
                        debug=debug):
                    yield sub_sched

                return

    # }}}

    # {{{ see if any loop can be entered now

    # Find inames that are being referenced by as yet unscheduled instructions.
    needed_inames = set()
    for insn_id in sched_state.unscheduled_insn_ids:
        needed_inames.update(kernel.insn_inames(insn_id))

    needed_inames = (needed_inames
            # There's no notion of 'entering' a parallel loop
            - sched_state.parallel_inames

            # Don't reenter a loop we're already in.
            - active_inames_set)

    if debug_mode:
        print(75*"-")
        print("inames still needed :", ",".join(needed_inames))
        print("active inames :", ",".join(sched_state.active_inames))
        print("inames entered so far :", ",".join(sched_state.entered_inames))
        print("reachable insns:", ",".join(reachable_insn_ids))
        print("active groups (with insn counts):", ",".join(
            "%s: %d" % (grp, c)
            for grp, c in six.iteritems(sched_state.active_group_counts)))
        print(75*"-")

    if needed_inames:
        iname_to_usefulness = {}

        for iname in needed_inames:

            # {{{ check if scheduling this iname now is allowed/plausible

            if (
                    iname in sched_state.prescheduled_inames
                    and not (
                        isinstance(next_preschedule_item, EnterLoop)
                        and next_preschedule_item.iname == iname)):
                if debug_mode:
                    print("scheduling %s prohibited by preschedule constraints"
                          % iname)
                continue

            currently_accessible_inames = (
                    active_inames_set | sched_state.parallel_inames)
            if (
                    not sched_state.loop_nest_around_map[iname]
                    <= currently_accessible_inames):
                if debug_mode:
                    print("scheduling %s prohibited by loop nest-around map" % iname)
                continue

            if (
                    not sched_state.loop_insn_dep_map.get(iname, set())
                    <= sched_state.scheduled_insn_ids):
                if debug_mode:
                    print(
                            "scheduling {iname} prohibited by loop dependency map "
                            "(needs '{needed_insns})'"
                            .format(
                                iname=iname,
                                needed_insns=", ".join(
                                    sched_state.loop_insn_dep_map.get(iname, set())
                                    -
                                    sched_state.scheduled_insn_ids)))

                continue

            iname_home_domain = kernel.domains[kernel.get_home_domain_index(iname)]
            from islpy import dim_type
            iname_home_domain_params = set(
                    iname_home_domain.get_var_names(dim_type.param))

            # The previous check should have ensured this is true, because
            # the loop_nest_around_map takes the domain dependency graph into
            # consideration.
            assert (iname_home_domain_params & kernel.all_inames()
                    <= currently_accessible_inames)

            # Check if any parameters are temporary variables, and if so, if their
            # writes have already been scheduled.

            data_dep_written = True
            for domain_par in (
                    iname_home_domain_params
                    &
                    set(kernel.temporary_variables)):
                writer_insn, = kernel.writer_map()[domain_par]
                if writer_insn not in sched_state.scheduled_insn_ids:
                    data_dep_written = False
                    if debug_mode:
                        print("iname '%s' not scheduled because domain "
                                "parameter '%s' is not yet available"
                                % (iname, domain_par))
                    break

            if not data_dep_written:
                continue

            # }}}

            # {{{ determine if that gets us closer to being able to schedule an insn

            usefulness = None  # highest insn priority enabled by iname

            hypothetically_active_loops = active_inames_set | set([iname])
            for insn_id in reachable_insn_ids:
                insn = kernel.id_to_insn[insn_id]

                want = kernel.insn_inames(insn)

                if hypothetically_active_loops <= want:
                    if usefulness is None:
                        usefulness = insn.priority
                    else:
                        usefulness = max(usefulness, insn.priority)

            if usefulness is None:
                if debug_mode:
                    print("iname '%s' deemed not useful" % iname)
                continue

            iname_to_usefulness[iname] = usefulness

            # }}}

        # {{{ tier building

        # Build priority tiers. If a schedule is found in the first tier, then
        # loops in the second are not even tried (and so on).
        loop_priority_set = set().union(*[set(prio)
                                          for prio in
                                          sched_state.kernel.loop_priority])
        useful_loops_set = set(six.iterkeys(iname_to_usefulness))
        useful_and_desired = useful_loops_set & loop_priority_set

        if useful_and_desired:
            wanted = (
                useful_and_desired
                - sched_state.ilp_inames
                - sched_state.vec_inames
                )
            priority_tiers = [t for t in
                              get_priority_tiers(wanted,
                                                 sched_state.kernel.loop_priority
                                                 )
                              ]

            # Update the loop priority set, because some constraints may have
            # have been contradictary.
            loop_priority_set = set().union(*[set(t) for t in priority_tiers])

            priority_tiers.append(
                    useful_loops_set
                    - loop_priority_set
                    - sched_state.ilp_inames
                    - sched_state.vec_inames
                    )
        else:
            priority_tiers = [
                    useful_loops_set
                    - sched_state.ilp_inames
                    - sched_state.vec_inames
                    ]

        # vectorization must be the absolute innermost loop
        priority_tiers.extend([
            [iname]
            for iname in sched_state.ilp_inames
            if iname in useful_loops_set
            ])

        priority_tiers.extend([
            [iname]
            for iname in sched_state.vec_inames
            if iname in useful_loops_set
            ])

        # }}}

        if debug_mode:
            print("useful inames: %s" % ",".join(useful_loops_set))
        else:
            for tier in priority_tiers:
                found_viable_schedule = False

                for iname in sorted(tier,
                        key=lambda iname: (
                            iname_to_usefulness.get(iname, 0),
                            # Sort by iname to achieve deterministic
                            # ordering of generated schedules.
                            iname),
                        reverse=True):

                    for sub_sched in generate_loop_schedules_internal(
                            sched_state.copy(
                                schedule=(
                                    sched_state.schedule
                                    + (EnterLoop(iname=iname),)),
                                active_inames=(
                                    sched_state.active_inames + (iname,)),
                                entered_inames=(
                                    sched_state.entered_inames
                                    | frozenset((iname,))),
                                preschedule=(
                                    sched_state.preschedule
                                    if iname not in sched_state.prescheduled_inames
                                    else sched_state.preschedule[1:]),
                                ),
                            debug=debug):
                        found_viable_schedule = True
                        yield sub_sched

                if found_viable_schedule:
                    return

    # }}}

    if debug_mode:
        print(75*"=")
        inp = six.moves.input("Hit Enter for next schedule, "
                "or enter a number to examine schedules of a "
                "different length:")
        if inp:
            raise ScheduleDebugInput(inp)

    if (
            not sched_state.active_inames
            and not sched_state.unscheduled_insn_ids
            and not sched_state.preschedule):
        # if done, yield result
        debug.log_success(sched_state.schedule)

        yield sched_state.schedule

    else:
        if debug is not None:
            debug.log_dead_end(sched_state.schedule)

# }}}


# {{{ convert barrier instructions to proper barriers

def convert_barrier_instructions_to_barriers(kernel, schedule):
    from loopy.kernel.instruction import BarrierInstruction

    result = []
    for sched_item in schedule:
        if isinstance(sched_item, RunInstruction):
            insn = kernel.id_to_insn[sched_item.insn_id]
            if isinstance(insn, BarrierInstruction):
                result.append(Barrier(
                    synchronization_kind=insn.synchronization_kind,
                    mem_kind=insn.mem_kind,
                    originating_insn_id=insn.id,
                    comment="Barrier inserted due to %s" % insn.id))
                continue

        result.append(sched_item)

    return result

# }}}


# {{{ barrier insertion/verification

class DependencyRecord(ImmutableRecord):
    """
    .. attribute:: source

        A :class:`loopy.InstructionBase` instance.

    .. attribute:: target

        A :class:`loopy.InstructionBase` instance.

    .. attribute:: dep_descr

        A string containing a phrase describing the dependency. The variables
        '{src}' and '{tgt}' will be replaced by their respective instruction IDs.

    .. attribute:: variable

        A string, the name of the variable that caused the dependency to arise.

    .. attribute:: var_kind

        "global" or "local"
    """

    def __init__(self, source, target, dep_descr, variable, var_kind):
        ImmutableRecord.__init__(self,
                source=source,
                target=target,
                dep_descr=dep_descr,
                variable=variable,
                var_kind=var_kind)


class DependencyTracker(object):
    """
    A utility to help track dependencies between originating from a set
    of sources (as defined by :meth:`add_source`. For each target,
    dependencies can then be obtained using :meth:`gen_dependencies_with_target_at`.

    .. automethod:: add_source
    .. automethod:: gen_dependencies_with_target_at
    """

    def __init__(self, kernel, var_kind, reverse):
        """
        :arg var_kind: "global" or "local", the kind of variable based on which
            barrier-needing dependencies should be found.
        :arg reverse:
            In straight-line code, this  only tracks 'b depends on
            a'-type 'forward' dependencies. But a loop of the type::

                for i in range(10):
                    A
                    B

            effectively glues multiple copies of 'A;B' one after the other::

                A
                B
                A
                B
                ...

            Now, if B depends on (i.e. is required to be textually before) A in a
            way requiring a barrier, then we will assume that the reverse
            dependency exists as well, i.e. a barrier between the tail end of
            execution of B and the next beginning of A is also needed.

            Setting *reverse* to *True* tracks these reverse (instead of forward)
            dependencies.
        """
        self.kernel = kernel
        self.reverse = reverse
        self.var_kind = var_kind

        from loopy.symbolic import AccessRangeOverlapChecker
        self.overlap_checker = AccessRangeOverlapChecker(kernel)

        if var_kind == "local":
            self.relevant_vars = kernel.local_var_names()
        elif var_kind == "global":
            self.relevant_vars = kernel.global_var_names()
        else:
            raise ValueError("unknown 'var_kind': %s" % var_kind)

        from collections import defaultdict
        self.base_writer_map = defaultdict(set)
        self.base_access_map = defaultdict(set)
        self.temp_to_base_storage = kernel.get_temporary_to_base_storage_map()

    def map_to_base_storage(self, var_names):
        result = set(var_names)

        for name in var_names:
            bs = self.temp_to_base_storage.get(name)
            if bs is not None:
                result.add(bs)

        return result

    def discard_all_sources(self):
        self.base_writer_map.clear()
        self.base_access_map.clear()

    # Anything with 'base' in the name in this class contains names normalized
    # to their 'base_storage'.

    def add_source(self, source):
        """
        Specify that an instruction used as the source (depended-upon
        part) of a dependency edge is of interest to this tracker.
        """
        # If source is an insn ID, look up the actual instruction.
        source = self.kernel.id_to_insn.get(source, source)

        for written in self.map_to_base_storage(
                set(source.assignee_var_names()) & self.relevant_vars):
            self.base_writer_map[written].add(source.id)

        for read in self.map_to_base_storage(
                source.dependency_names() & self.relevant_vars):
            self.base_access_map[read].add(source.id)

    def gen_dependencies_with_target_at(self, target):
        """
        Generate :class:`DependencyRecord` instances for dependencies edges
        whose target is the given instruction.

        :arg target: The ID of the instruction for which dependencies
            with conflicting var access should be found.
        """
        # If target is an insn ID, look up the actual instruction.
        target = self.kernel.id_to_insn.get(target, target)

        for (
                tgt_dir, src_dir, src_base_var_to_accessor_map
                ) in [
                ("any", "w", self.base_writer_map),
                ("w", "any", self.base_access_map),
                ]:

            for dep in self.get_conflicting_accesses(
                    target, tgt_dir, src_dir, src_base_var_to_accessor_map):
                yield dep

    def get_conflicting_accesses(self, target, tgt_dir, src_dir,
            src_base_var_to_accessor_map):

        def get_written_names(insn):
            return set(insn.assignee_var_names()) & self.relevant_vars

        def get_accessed_names(insn):
            return insn.dependency_names() & self.relevant_vars

        dir_to_getter = {"w": get_written_names, "any": get_accessed_names}

        def filter_var_set_for_base_storage(var_name_set, base_storage_name):
            return set(
                    name
                    for name in var_name_set
                    if (self.temp_to_base_storage.get(name, name)
                        == base_storage_name))

        tgt_accessed_vars = dir_to_getter[tgt_dir](target)
        tgt_accessed_vars_base = self.map_to_base_storage(tgt_accessed_vars)

        for race_var_base in sorted(tgt_accessed_vars_base):
            for source_id in sorted(
                    src_base_var_to_accessor_map[race_var_base]):

                # {{{ no barrier if nosync

                if (not self.reverse and source_id in
                        self.kernel.get_nosync_set(target.id, scope=self.var_kind)):
                    continue
                if (self.reverse and target.id in
                        self.kernel.get_nosync_set(source_id, scope=self.var_kind)):
                    continue

                # }}}

                dep_descr = self.describe_dependency(source_id, target)
                if dep_descr is None:
                    continue

                source = self.kernel.id_to_insn[source_id]
                src_race_vars = filter_var_set_for_base_storage(
                        dir_to_getter[src_dir](source), race_var_base)
                tgt_race_vars = filter_var_set_for_base_storage(
                        tgt_accessed_vars, race_var_base)

                race_var = race_var_base

                # Only one (non-base_storage) race variable name: Data is not
                # being passed between aliases, so we may look at indices.
                if src_race_vars == tgt_race_vars and len(src_race_vars) == 1:
                    race_var, = src_race_vars

                    if not (
                        self.overlap_checker.do_access_ranges_overlap_conservative(
                                target.id, tgt_dir, source_id, src_dir, race_var)):
                        continue

                yield DependencyRecord(
                        source=source,
                        target=target,
                        dep_descr=dep_descr,
                        variable=race_var,
                        var_kind=self.var_kind)

    def describe_dependency(self, source_id, target):
        dep_descr = None

        source = self.kernel.id_to_insn[source_id]

        if self.reverse:
            source, target = target, source

        target_deps = self.kernel.recursive_insn_dep_map()[target.id]
        if source.id in target_deps:
            if self.reverse:
                dep_descr = "{tgt} rev-depends on {src}"
            else:
                dep_descr = "{tgt} depends on {src}"

        grps = source.groups & target.conflicts_with_groups
        if not grps:
            grps = target.groups & source.conflicts_with_groups

        if grps:
            dep_descr = "{src} conflicts with {tgt} (via '%s')" % ", ".join(grps)

        return dep_descr


def barrier_kind_more_or_equally_global(kind1, kind2):
    return (kind1 == kind2) or (kind1 == "global" and kind2 == "local")


def insn_ids_reaching_end_without_intervening_barrier(schedule, kind):
    return _insn_ids_reaching_end(schedule, kind, reverse=False)


def insn_ids_reachable_from_start_without_intervening_barrier(schedule, kind):
    return _insn_ids_reaching_end(schedule, kind, reverse=True)


def _insn_ids_reaching_end(schedule, kind, reverse):
    if reverse:
        schedule = reversed(schedule)
        enter_scope_item_kind = LeaveLoop
        leave_scope_item_kind = EnterLoop
    else:
        enter_scope_item_kind = EnterLoop
        leave_scope_item_kind = LeaveLoop

    insn_ids_alive_at_scope = [set()]

    for sched_item in schedule:
        if isinstance(sched_item, enter_scope_item_kind):
            insn_ids_alive_at_scope.append(set())
        elif isinstance(sched_item, leave_scope_item_kind):
            innermost_scope = insn_ids_alive_at_scope.pop()
            # Instructions in deeper scopes are alive but could be killed by
            # barriers at a shallower level, e.g.:
            #
            # for i
            #     insn0
            # end
            # barrier()   <= kills insn0
            #
            # Hence we merge this scope into the parent scope.
            insn_ids_alive_at_scope[-1].update(innermost_scope)
        elif isinstance(sched_item, Barrier):
            # This barrier kills only the instruction ids that are alive at
            # the current scope (or deeper). Without further analysis, we
            # can't assume that instructions at shallower scope can be
            # killed by deeper barriers, since loops might be empty, e.g.:
            #
            # insn0          <= isn't killed by barrier (i loop could be empty)
            # for i
            #     insn1      <= is killed by barrier
            #     for j
            #         insn2  <= is killed by barrier
            #     end
            #     barrier()
            # end
            if barrier_kind_more_or_equally_global(
                    sched_item.synchronization_kind, kind):
                insn_ids_alive_at_scope[-1].clear()
        else:
            insn_ids_alive_at_scope[-1] |= set(
                    insn_id for insn_id in sched_item_to_insn_id(sched_item))

    assert len(insn_ids_alive_at_scope) == 1
    return insn_ids_alive_at_scope[-1]


def append_barrier_or_raise_error(schedule, dep, verify_only):
    if verify_only:
        from loopy.diagnostic import MissingBarrierError
        raise MissingBarrierError(
                "Dependency '%s' (for variable '%s') "
                "requires synchronization "
                "by a %s barrier (add a 'no_sync_with' "
                "instruction option to state that no "
                "synchronization is needed)"
                % (
                    dep.dep_descr.format(
                        tgt=dep.target.id, src=dep.source.id),
                    dep.variable,
                    dep.var_kind))
    else:
        comment = "for %s (%s)" % (
                dep.variable, dep.dep_descr.format(
                    tgt=dep.target.id, src=dep.source.id))
        schedule.append(Barrier(
            comment=comment,
            synchronization_kind=dep.var_kind,
            mem_kind=dep.var_kind,
            originating_insn_id=None))


def insert_barriers(kernel, schedule, synchronization_kind, verify_only, level=0):
    """
    :arg synchronization_kind: "local" or "global".
        The :attr:`Barrier.synchronization_kind` to be inserted. Generally, this
        function will be called once for each kind of barrier at the top level, where
        more global barriers should be inserted first.
    :arg verify_only: do not insert barriers, only complain if they are
        missing.
    :arg level: the current level of loop nesting, 0 for outermost.
    """

    # {{{ insert barriers at outermost scheduling level

    def insert_barriers_at_outer_level(schedule, reverse=False):
        dep_tracker = DependencyTracker(kernel, var_kind=synchronization_kind,
                                        reverse=reverse)

        if reverse:
            # Populate the dependency tracker with sources from the tail end of
            # the schedule block.
            for insn_id in (
                    insn_ids_reaching_end_without_intervening_barrier(
                        schedule, synchronization_kind)):
                dep_tracker.add_source(insn_id)

        result = []

        i = 0
        while i < len(schedule):
            sched_item = schedule[i]

            if isinstance(sched_item, EnterLoop):
                subloop, new_i = gather_schedule_block(schedule, i)

                loop_head = (
                    insn_ids_reachable_from_start_without_intervening_barrier(
                        subloop, synchronization_kind))

                loop_tail = (
                    insn_ids_reaching_end_without_intervening_barrier(
                        subloop, synchronization_kind))

                # Checks if a barrier is needed before the loop. This handles
                # dependencies with targets that can be reached without an
                # intervening barrier from the start of the loop:
                #
                # a[x] = ...      <= source
                # for i
                #     ... = a[y]  <= target
                #     barrier()
                #     ...
                from itertools import chain
                for dep in chain.from_iterable(
                        dep_tracker.gen_dependencies_with_target_at(insn)
                        for insn in loop_head):
                    append_barrier_or_raise_error(result, dep, verify_only)
                    # This barrier gets inserted outside the loop, hence it is
                    # executed unconditionally and so kills all sources before
                    # the loop.
                    dep_tracker.discard_all_sources()
                    break

                result.extend(subloop)

                # Handle dependencies with sources not killed inside the loop:
                #
                # for i
                #     ...
                #     barrier()
                #     b[i] = ...  <= source
                # end for
                # ... = f(b)      <= target
                for item in loop_tail:
                    dep_tracker.add_source(item)

                i = new_i

            elif isinstance(sched_item, Barrier):
                result.append(sched_item)
                if barrier_kind_more_or_equally_global(
                        sched_item.synchronization_kind, synchronization_kind):
                    dep_tracker.discard_all_sources()
                i += 1

            elif isinstance(sched_item, RunInstruction):
                for dep in dep_tracker.gen_dependencies_with_target_at(
                        sched_item.insn_id):
                    append_barrier_or_raise_error(result, dep, verify_only)
                    dep_tracker.discard_all_sources()
                    break
                result.append(sched_item)
                dep_tracker.add_source(sched_item.insn_id)
                i += 1

            elif isinstance(sched_item, (CallKernel, ReturnFromKernel)):
                result.append(sched_item)
                i += 1

            else:
                raise ValueError("unexpected schedule item type '%s'"
                        % type(sched_item).__name__)

        return result

    # }}}

    # {{{ recursively insert barriers in loops

    result = []
    i = 0
    while i < len(schedule):
        sched_item = schedule[i]

        if isinstance(sched_item, EnterLoop):
            subloop, new_i = gather_schedule_block(schedule, i)
            new_subloop = insert_barriers(
                    kernel, subloop[1:-1], synchronization_kind, verify_only,
                    level + 1)
            result.append(subloop[0])
            result.extend(new_subloop)
            result.append(subloop[-1])
            i = new_i

        elif isinstance(sched_item,
                (Barrier, RunInstruction, CallKernel, ReturnFromKernel)):
            result.append(sched_item)
            i += 1

        else:
            raise ValueError("unexpected schedule item type '%s'"
                    % type(sched_item).__name__)

    # }}}

    result = insert_barriers_at_outer_level(result)

    # When level = 0 there is no loop.
    if level != 0:
        result = insert_barriers_at_outer_level(result, reverse=True)

    return result

# }}}


class MinRecursionLimitForScheduling(MinRecursionLimit):
    def __init__(self, kernel):
        MinRecursionLimit.__init__(self,
                len(kernel.instructions) * 2 + len(kernel.all_inames()) * 4)


# {{{ main scheduling entrypoint

def generate_loop_schedules(kernel, debug_args={}):
    """
    .. warning::

        This function needs to be called inside (another layer) of a
        :class:`MinRecursionLimitForScheduling` context manager, and the
        context manager needs to end *after* the last reference to the
        generators has gone out of scope. Otherwise, the high-recursion-limit
        generator chain may not be successfully garbage-collected and cause an
        internal error in the Python runtime.
    """

    with MinRecursionLimitForScheduling(kernel):
        for sched in generate_loop_schedules_inner(kernel, debug_args=debug_args):
            yield sched


def generate_loop_schedules_inner(kernel, debug_args={}):
    from loopy.kernel import KernelState
    if kernel.state not in (KernelState.PREPROCESSED, KernelState.LINEARIZED):
        raise LoopyError("cannot schedule a kernel that has not been "
                "preprocessed")

    from loopy.check import pre_schedule_checks
    pre_schedule_checks(kernel)

    schedule_count = 0

    debug = ScheduleDebugger(**debug_args)

    preschedule = kernel.schedule if kernel.state == KernelState.LINEARIZED else ()

    prescheduled_inames = set(
            insn.iname
            for insn in preschedule
            if isinstance(insn, EnterLoop))

    prescheduled_insn_ids = set(
        insn_id
        for item in preschedule
        for insn_id in sched_item_to_insn_id(item))

    from loopy.kernel.data import (IlpBaseTag, ConcurrentTag, VectorizeTag,
                                   filter_iname_tags_by_type)
    ilp_inames = set(
            iname
            for iname, tags in six.iteritems(kernel.iname_to_tags)
            if filter_iname_tags_by_type(tags, IlpBaseTag))
    vec_inames = set(
            iname
            for iname, tags in six.iteritems(kernel.iname_to_tags)
            if filter_iname_tags_by_type(tags, VectorizeTag))
    parallel_inames = set(
            iname
            for iname, tags in six.iteritems(kernel.iname_to_tags)
            if filter_iname_tags_by_type(tags, ConcurrentTag))

    loop_nest_with_map = find_loop_nest_with_map(kernel)
    loop_nest_around_map = find_loop_nest_around_map(kernel)
    sched_state = SchedulerState(
            kernel=kernel,
            loop_nest_around_map=loop_nest_around_map,
            loop_insn_dep_map=find_loop_insn_dep_map(
                kernel,
                loop_nest_with_map=loop_nest_with_map,
                loop_nest_around_map=loop_nest_around_map),
            breakable_inames=ilp_inames,
            ilp_inames=ilp_inames,
            vec_inames=vec_inames,

            prescheduled_inames=prescheduled_inames,
            prescheduled_insn_ids=prescheduled_insn_ids,

            # time-varying part
            active_inames=(),
            entered_inames=frozenset(),
            enclosing_subkernel_inames=(),

            schedule=(),

            unscheduled_insn_ids=set(insn.id for insn in kernel.instructions),
            scheduled_insn_ids=frozenset(),
            within_subkernel=kernel.state != KernelState.LINEARIZED,
            may_schedule_global_barriers=True,

            preschedule=preschedule,
            insn_ids_to_try=None,

            # ilp and vec are not parallel for the purposes of the scheduler
            parallel_inames=parallel_inames - ilp_inames - vec_inames,

            group_insn_counts=group_insn_counts(kernel),
            active_group_counts={})

    schedule_gen_kwargs = {}

    def print_longest_dead_end():
        if debug.interactive:
            print("Loo.py will now show you the scheduler state at the point")
            print("where the longest (dead-end) schedule was generated, in the")
            print("the hope that some of this makes sense and helps you find")
            print("the issue.")
            print()
            print("To disable this interactive behavior, pass")
            print("  debug_args=dict(interactive=False)")
            print("to generate_loop_schedules().")
            print(75*"-")
            six.moves.input("Enter:")
            print()
            print()

            debug.debug_length = len(debug.longest_rejected_schedule)
            while True:
                try:
                    for _ in generate_loop_schedules_internal(
                            sched_state, debug=debug, **schedule_gen_kwargs):
                        pass

                except ScheduleDebugInput as e:
                    debug.debug_length = int(str(e))
                    continue

                break

    try:
        for gen_sched in generate_loop_schedules_internal(
                sched_state, debug=debug, **schedule_gen_kwargs):
            debug.stop()

            gen_sched = convert_barrier_instructions_to_barriers(
                    kernel, gen_sched)

            gsize, lsize = kernel.get_grid_size_upper_bounds()

            if (gsize or lsize):
                if not kernel.options.disable_global_barriers:
                    logger.debug("%s: barrier insertion: global" % kernel.name)
                    gen_sched = insert_barriers(kernel, gen_sched,
                            synchronization_kind="global", verify_only=True)

                logger.debug("%s: barrier insertion: local" % kernel.name)
                gen_sched = insert_barriers(kernel, gen_sched,
                    synchronization_kind="local", verify_only=False)
                logger.debug("%s: barrier insertion: done" % kernel.name)

            new_kernel = kernel.copy(
                    schedule=gen_sched,
                    state=KernelState.LINEARIZED)

            from loopy.schedule.device_mapping import \
                    map_schedule_onto_host_or_device
            if kernel.state != KernelState.LINEARIZED:
                # Device mapper only gets run once.
                new_kernel = map_schedule_onto_host_or_device(new_kernel)

            from loopy.schedule.tools import add_extra_args_to_schedule
            new_kernel = add_extra_args_to_schedule(new_kernel)
            yield new_kernel

            debug.start()

            schedule_count += 1

    except KeyboardInterrupt:
        print()
        print(75*"-")
        print("Interrupted during scheduling")
        print(75*"-")
        print_longest_dead_end()
        raise

    debug.done_scheduling()
    if not schedule_count:
        print(75*"-")
        print("ERROR: Sorry--loo.py did not find a schedule for your kernel.")
        print(75*"-")
        print_longest_dead_end()
        raise RuntimeError("no valid schedules found")

    logger.info("%s: schedule done" % kernel.name)

# }}}


schedule_cache = WriteOncePersistentDict(
        "loopy-schedule-cache-v4-"+DATA_MODEL_VERSION,
        key_builder=LoopyKeyBuilder())


def _get_one_scheduled_kernel_inner(kernel):
    # This helper function exists to ensure that the generator chain is fully
    # out of scope after the function returns. This allows it to be
    # garbage-collected in the exit handler of the
    # MinRecursionLimitForScheduling context manager in the surrounding
    # function, because it possilby cannot be safely collected with a lower
    # recursion limit without crashing the Python runtime.
    #
    # See https://gitlab.tiker.net/inducer/sumpy/issues/31 for context.

    return next(iter(generate_loop_schedules(kernel)))


def get_one_scheduled_kernel(kernel):
    warn_with_kernel(
        kernel, "get_one_scheduled_kernel_deprecated",
        "get_one_scheduled_kernel is deprecated. "
        "Use get_one_linearized_kernel instead.",
        DeprecationWarning)
    return get_one_linearized_kernel(kernel)


def get_one_linearized_kernel(kernel):
    from loopy import CACHING_ENABLED

    sched_cache_key = kernel
    from_cache = False

    if CACHING_ENABLED:
        try:
            result = schedule_cache[sched_cache_key]

            logger.debug("%s: schedule cache hit" % kernel.name)
            from_cache = True
        except KeyError:
            pass

    if not from_cache:
        with ProcessLogger(logger, "%s: schedule" % kernel.name):
            with MinRecursionLimitForScheduling(kernel):
                result = _get_one_scheduled_kernel_inner(kernel)

    if CACHING_ENABLED and not from_cache:
        schedule_cache.store_if_not_present(sched_cache_key, result)

    return result


# vim: foldmethod=marker

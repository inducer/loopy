from __future__ import division

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


from pytools import Record
import sys
import islpy as isl

import logging
logger = logging.getLogger(__name__)


# {{{ schedule items

class ScheduleItem(Record):
    __slots__ = []


class EnterLoop(ScheduleItem):
    __slots__ = ["iname"]


class LeaveLoop(ScheduleItem):
    __slots__ = ["iname"]


class RunInstruction(ScheduleItem):
    __slots__ = ["insn_id"]


class Barrier(ScheduleItem):
    __slots__ = ["comment"]

# }}}


# {{{ schedule utilities

def gather_schedule_subloop(schedule, start_idx):
    assert isinstance(schedule[start_idx], EnterLoop)
    level = 0

    i = start_idx
    while i < len(schedule):
        if isinstance(schedule[i], EnterLoop):
            level += 1
        if isinstance(schedule[i], LeaveLoop):
            level -= 1

            if level == 0:
                return schedule[start_idx:i+1], i+1

        i += 1

    assert False


def get_barrier_needing_dependency(kernel, target, source, unordered=False):
    from loopy.kernel.data import InstructionBase
    if not isinstance(source, InstructionBase):
        source = kernel.id_to_insn[source]
    if not isinstance(target, InstructionBase):
        target = kernel.id_to_insn[target]

    local_vars = kernel.local_var_names()

    tgt_write = set(target.assignee_var_names()) & local_vars
    tgt_read = target.read_dependency_names() & local_vars

    src_write = set(source.assignee_var_names()) & local_vars
    src_read = source.read_dependency_names() & local_vars

    waw = tgt_write & src_write
    raw = tgt_read & src_write
    war = tgt_write & src_read

    for var_name in raw | war:
        if not unordered:
            assert source.id in target.insn_deps
        return (target, source, var_name)

    if source is target:
        return None

    for var_name in waw:
        assert (source.id in target.insn_deps
                or source is target)
        return (target, source, var_name)

    return None


def get_barrier_dependent_in_schedule(kernel, source, schedule,
        unordered):
    """
    :arg source: an instruction id for the source of the dependency
    """

    for sched_item in schedule:
        if isinstance(sched_item, RunInstruction):
            temp_res = get_barrier_needing_dependency(
                    kernel, sched_item.insn_id, source, unordered=unordered)
            if temp_res:
                return temp_res
        elif isinstance(sched_item, Barrier):
            return


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

    if isinstance(sched_item, EnterLoop):
        loop_contents, _ = gather_schedule_subloop(
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

    if isinstance(sched_item, EnterLoop):
        loop_contents, _ = gather_schedule_subloop(
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


def loop_nest_map(kernel):
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
        for outer_iname in kernel.all_inames():
            tag = kernel.iname_to_tag.get(outer_iname)
            if isinstance(tag, IlpBaseTag):
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
            if outer_iname not in kernel.all_inames():
                continue

            for inner_iname in dom.get_var_names(isl.dim_type.set):
                result[inner_iname].add(outer_iname)

    return result

# }}}


# {{{ debug help

def dump_schedule(schedule):
    entries = []
    for sched_item in schedule:
        if isinstance(sched_item, EnterLoop):
            entries.append("<%s>" % sched_item.iname)
        elif isinstance(sched_item, LeaveLoop):
            entries.append("</%s>" % sched_item.iname)
        elif isinstance(sched_item, RunInstruction):
            entries.append(sched_item.insn_id)
        elif isinstance(sched_item, Barrier):
            entries.append("|")
        else:
            assert False

    return " ".join(entries)


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
# }}}


# {{{ scheduling algorithm

class SchedulerState(Record):
    pass


def generate_loop_schedules_internal(sched_state, loop_priority, schedule=[],
        allow_boost=False, allow_insn=False, debug=None):
    # allow_insn is set to False initially and after entering each loop
    # to give loops containing high-priority instructions a chance.

    kernel = sched_state.kernel
    all_insn_ids = set(insn.id for insn in kernel.instructions)

    scheduled_insn_ids = set(sched_item.insn_id for sched_item in schedule
            if isinstance(sched_item, RunInstruction))

    unscheduled_insn_ids = all_insn_ids - scheduled_insn_ids

    if allow_boost is None:
        rec_allow_boost = None
    else:
        rec_allow_boost = False

    # {{{ find active and entered loops

    active_inames = []
    entered_inames = set()

    for sched_item in schedule:
        if isinstance(sched_item, EnterLoop):
            active_inames.append(sched_item.iname)
            entered_inames.add(sched_item.iname)
        if isinstance(sched_item, LeaveLoop):
            active_inames.pop()

    if active_inames:
        last_entered_loop = active_inames[-1]
    else:
        last_entered_loop = None
    active_inames_set = set(active_inames)

    # }}}

    # {{{ decide about debug mode

    debug_mode = False

    if debug is not None:
        if (debug.debug_length is not None
                and len(schedule) >= debug.debug_length):
            debug_mode = True

    if debug_mode:
        if debug.wrote_status == 2:
            print
        print 75*"="
        print "KERNEL:"
        print kernel
        print 75*"="
        print "CURRENT SCHEDULE:"
        print "%s (length: %d)" % (dump_schedule(schedule), len(schedule))
        print("(LEGEND: entry: <iname>, exit: </iname>, instructions "
                "w/ no delimiters)")
        #print "boost allowed:", allow_boost
        print 75*"="
        print "LOOP NEST MAP:"
        for iname, val in sched_state.loop_nest_map.iteritems():
            print "%s : %s" % (iname, ", ".join(val))
        print 75*"="
        print "WHY IS THIS A DEAD-END SCHEDULE?"

    #if len(schedule) == 2:
        #from pudb import set_trace; set_trace()

    # }}}

    # {{{ see if any insns are ready to be scheduled now

    # Also take note of insns that have a chance of being schedulable inside
    # the current loop nest, in this set:

    reachable_insn_ids = set()

    for insn_id in sorted(unscheduled_insn_ids,
            key=lambda insn_id: kernel.id_to_insn[insn_id].priority,
            reverse=True):

        insn = kernel.id_to_insn[insn_id]

        is_ready = set(insn.insn_deps) <= scheduled_insn_ids

        if not is_ready:
            if debug_mode:
                print "instruction '%s' is missing insn depedencies '%s'" % (
                        insn.id, ",".join(set(insn.insn_deps) - scheduled_insn_ids))
            continue

        want = kernel.insn_inames(insn) - sched_state.parallel_inames
        have = active_inames_set - sched_state.parallel_inames

        # If insn is boostable, it may be placed inside a more deeply
        # nested loop without harm.

        if allow_boost:
            # Note that the inames in 'insn.boostable_into' necessarily won't
            # be contained in 'want'.
            have = have - insn.boostable_into

        if want != have:
            is_ready = False

            if debug_mode:
                if want-have:
                    print ("instruction '%s' is missing inames '%s'"
                            % (insn.id, ",".join(want-have)))
                if have-want:
                    print ("instruction '%s' won't work under inames '%s'"
                            % (insn.id, ",".join(have-want)))

        # {{{ determine reachability

        if (not is_ready and have <= want):
            reachable_insn_ids.add(insn_id)

        # }}}

        if is_ready and allow_insn:
            if debug_mode:
                print "scheduling '%s'" % insn.id
            scheduled_insn_ids.add(insn.id)
            schedule = schedule + [RunInstruction(insn_id=insn.id)]

            # Don't be eager about entering/leaving loops--if progress has been
            # made, revert to top of scheduler and see if more progress can be
            # made.

            for sub_sched in generate_loop_schedules_internal(
                    sched_state, loop_priority, schedule,
                    allow_boost=rec_allow_boost, debug=debug,
                    allow_insn=True):
                yield sub_sched

            return

    # }}}

    # {{{ see if we're ready to leave the innermost loop

    if last_entered_loop is not None:
        can_leave = True

        if last_entered_loop not in sched_state.breakable_inames:
            # If the iname is not breakable, then check that we've
            # scheduled all the instructions that require it.

            for insn_id in unscheduled_insn_ids:
                insn = kernel.id_to_insn[insn_id]
                if last_entered_loop in kernel.insn_inames(insn):
                    if debug_mode:
                        print("cannot leave '%s' because '%s' still depends on it"
                                % (last_entered_loop, insn.id))
                    can_leave = False
                    break

        if can_leave:
            can_leave = False

            # We may only leave this loop if we've scheduled an instruction
            # since entering it.

            seen_an_insn = False
            ignore_count = 0
            for sched_item in schedule[::-1]:
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

            if can_leave:
                schedule = schedule + [LeaveLoop(iname=last_entered_loop)]

                for sub_sched in generate_loop_schedules_internal(
                        sched_state, loop_priority, schedule,
                        allow_boost=rec_allow_boost, debug=debug,
                        allow_insn=allow_insn):
                    yield sub_sched

                return

    # }}}

    # {{{ see if any loop can be entered now

    # Find inames that are being referenced by as yet unscheduled instructions.
    needed_inames = set()
    for insn_id in unscheduled_insn_ids:
        needed_inames.update(kernel.insn_inames(insn_id))

    needed_inames = (needed_inames
            # There's no notion of 'entering' a parallel loop
            - sched_state.parallel_inames

            # Don't reenter a loop we're already in.
            - active_inames_set)

    if debug_mode:
        print 75*"-"
        print "inames still needed :", ",".join(needed_inames)
        print "active inames :", ",".join(active_inames)
        print "inames entered so far :", ",".join(entered_inames)
        print "reachable insns:", ",".join(reachable_insn_ids)
        print 75*"-"

    if needed_inames:
        iname_to_usefulness = {}

        for iname in needed_inames:

            # {{{ check if scheduling this iname now is allowed/plausible

            currently_accessible_inames = (
                    active_inames_set | sched_state.parallel_inames)
            if not sched_state.loop_nest_map[iname] <= currently_accessible_inames:
                if debug_mode:
                    print "scheduling %s prohibited by loop nest map" % iname
                continue

            iname_home_domain = kernel.domains[kernel.get_home_domain_index(iname)]
            from islpy import dim_type
            iname_home_domain_params = set(
                    iname_home_domain.get_var_names(dim_type.param))

            # The previous check should have ensured this is true, because
            # the loop_nest_map takes the domain dependency graph into
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
                if writer_insn not in scheduled_insn_ids:
                    data_dep_written = False
                    break

            if not data_dep_written:
                continue

            # }}}

            # {{{ determine if that gets us closer to being able to schedule an insn

            usefulness = None  # highest insn priority enabled by iname

            hypothetically_active_loops = active_inames_set | set([iname])
            for insn_id in reachable_insn_ids:
                insn = kernel.id_to_insn[insn_id]

                want = kernel.insn_inames(insn) | insn.boostable_into

                if hypothetically_active_loops <= want:
                    if usefulness is None:
                        usefulness = insn.priority
                    else:
                        usefulness = max(usefulness, insn.priority)

            if usefulness is None:
                if debug_mode:
                    print "iname '%s' deemed not useful" % iname
                continue

            iname_to_usefulness[iname] = usefulness

            # }}}

        # {{{ tier building

        # Build priority tiers. If a schedule is found in the first tier, then
        # loops in the second are not even tried (and so on).

        loop_priority_set = set(loop_priority)
        useful_loops_set = set(iname_to_usefulness.iterkeys())
        useful_and_desired = useful_loops_set & loop_priority_set

        if useful_and_desired:
            priority_tiers = [[iname]
                    for iname in loop_priority
                    if iname in useful_and_desired
                    and iname not in sched_state.lowest_priority_inames]

            priority_tiers.append(
                    useful_loops_set
                    - loop_priority_set
                    - sched_state.lowest_priority_inames)
        else:
            priority_tiers = [useful_loops_set - sched_state.lowest_priority_inames]

        priority_tiers.extend([
            [iname]
            for iname in sched_state.lowest_priority_inames
            if iname in useful_loops_set
            ])

        # }}}

        if debug_mode:
            print "useful inames: %s" % ",".join(useful_loops_set)

        for tier in priority_tiers:
            found_viable_schedule = False

            for iname in sorted(tier,
                    key=lambda iname: iname_to_usefulness.get(iname, 0),
                    reverse=True):
                new_schedule = schedule + [EnterLoop(iname=iname)]

                for sub_sched in generate_loop_schedules_internal(
                        sched_state, loop_priority, new_schedule,
                        allow_boost=rec_allow_boost,
                        debug=debug):
                    found_viable_schedule = True
                    yield sub_sched

            if found_viable_schedule:
                return

    # }}}

    if debug_mode:
        print 75*"="
        raw_input("Hit Enter for next schedule:")

    if not active_inames and not unscheduled_insn_ids:
        # if done, yield result
        debug.log_success(schedule)

        yield schedule

    else:
        if not allow_insn:
            # try again with boosting allowed
            for sub_sched in generate_loop_schedules_internal(
                    sched_state, loop_priority, schedule=schedule,
                    allow_boost=allow_boost, debug=debug,
                    allow_insn=True):
                yield sub_sched

        if not allow_boost and allow_boost is not None:
            # try again with boosting allowed
            for sub_sched in generate_loop_schedules_internal(
                    sched_state, loop_priority, schedule=schedule,
                    allow_boost=True, debug=debug,
                    allow_insn=allow_insn):
                yield sub_sched
        else:
            # dead end
            if debug is not None:
                debug.log_dead_end(schedule)

# }}}


# {{{ barrier insertion

def insert_barriers(kernel, schedule, level=0):
    result = []
    owed_barriers = set()

    loop_had_barrier = [False]

    # A 'pre-barrier' is a special case that is only necessary once per loop
    # iteration to protect the tops of local-mem variable assignments from
    # being entered before all reads in the previous loop iteration are
    # complete.  Once the loop has had a barrier, this is not a concern any
    # more, and any further write-after-read hazards will be covered by
    # dependencies for which the 'normal' mechanism below will generate
    # barriers.

    def issue_barrier(is_pre_barrier, dep):
        if result and isinstance(result[-1], Barrier):
            return

        if is_pre_barrier:
            if loop_had_barrier[0] or level == 0:
                return

        owed_barriers.clear()

        cmt = None
        if dep is not None:
            target, source, var = dep
            if is_pre_barrier:
                cmt = "pre-barrier: %s" % var
            else:
                cmt = "dependency: %s" % var

        loop_had_barrier[0] = True
        result.append(Barrier(comment=cmt))

    i = 0
    while i < len(schedule):
        sched_item = schedule[i]

        if isinstance(sched_item, EnterLoop):
            subloop, new_i = gather_schedule_subloop(schedule, i)

            subresult, sub_owed_barriers = insert_barriers(
                    kernel, subloop[1:-1], level+1)

            # {{{ issue dependency-based barriers for contents of nested loop

            # (i.e. if anything *in* the loop depends on something beforehand)

            for insn_id in owed_barriers:
                dep = get_barrier_dependent_in_schedule(kernel, insn_id, subresult,
                        unordered=False)
                if dep:
                    issue_barrier(is_pre_barrier=False, dep=dep)
                    break

            # }}}
            # {{{ issue pre-barriers for contents of nested loop

            if not loop_had_barrier[0]:
                for insn_id in sub_owed_barriers:
                    dep = get_barrier_dependent_in_schedule(
                            kernel, insn_id, schedule, unordered=True)
                    if dep:
                        issue_barrier(is_pre_barrier=True, dep=dep)

            # }}}

            result.append(subloop[0])
            result.extend(subresult)
            result.append(subloop[-1])

            owed_barriers.update(sub_owed_barriers)

            i = new_i

        elif isinstance(sched_item, RunInstruction):
            i += 1

            insn = kernel.id_to_insn[sched_item.insn_id]

            # {{{ issue dependency-based barriers for this instruction

            for dep_src_insn_id in set(insn.insn_deps) & owed_barriers:
                dep = get_barrier_needing_dependency(kernel, insn, dep_src_insn_id)
                if dep:
                    issue_barrier(is_pre_barrier=False, dep=dep)

            # }}}

            for assignee_name in insn.assignee_var_names():
                assignee_temp_var = kernel.temporary_variables.get(
                        assignee_name)
                if assignee_temp_var is not None and assignee_temp_var.is_local:
                    dep = get_barrier_dependent_in_schedule(
                            kernel, insn.id, schedule,
                            unordered=True)

                    if dep:
                        issue_barrier(is_pre_barrier=True, dep=dep)

                    owed_barriers.add(insn.id)
            result.append(sched_item)

        else:
            assert False

    return result, owed_barriers

# }}}


# {{{ main scheduling entrypoint

def generate_loop_schedules(kernel, debug_args={}):
    loop_priority = kernel.loop_priority

    from loopy.preprocess import preprocess_kernel
    kernel = preprocess_kernel(kernel)

    from loopy.check import run_automatic_checks
    run_automatic_checks(kernel)

    logger.info("schedule %s: start" % kernel.name)

    schedule_count = 0

    debug = ScheduleDebugger(**debug_args)

    from loopy.kernel.data import IlpBaseTag, ParallelTag
    ilp_inames = set(
            iname
            for iname in kernel.all_inames()
            if isinstance(kernel.iname_to_tag.get(iname), IlpBaseTag))
    parallel_inames = set(
            iname for iname in kernel.all_inames()
            if isinstance(kernel.iname_to_tag.get(iname), ParallelTag))

    sched_state = SchedulerState(
            kernel=kernel,
            loop_nest_map=loop_nest_map(kernel),
            breakable_inames=ilp_inames,
            lowest_priority_inames=ilp_inames,
            # ILP is not parallel for the purposes of the scheduler
            parallel_inames=parallel_inames - ilp_inames)

    generators = [
            generate_loop_schedules_internal(sched_state, loop_priority,
                debug=debug, allow_boost=None),
            generate_loop_schedules_internal(sched_state, loop_priority,
                debug=debug)]
    for gen in generators:
        for gen_sched in gen:
            gen_sched, owed_barriers = insert_barriers(kernel, gen_sched)
            if owed_barriers:
                from warnings import warn
                from loopy.diagnostic import LoopyAdvisory
                warn("Barrier insertion finished without inserting barriers for "
                        "local memory writes in these instructions: '%s'. "
                        "This often means that local memory was "
                        "written, but never read."
                        % ",".join(owed_barriers), LoopyAdvisory)

            debug.stop()
            yield kernel.copy(schedule=gen_sched)
            debug.start()

            schedule_count += 1

        # if no-boost mode yielded a viable schedule, stop now
        if schedule_count:
            break

    debug.done_scheduling()

    if not schedule_count:
        if debug.interactive:
            print 75*"-"
            print "ERROR: Sorry--loo.py did not find a schedule for your kernel."
            print 75*"-"
            print "Loo.py will now show you the scheduler state at the point"
            print "where the longest (dead-end) schedule was generated, in the"
            print "the hope that some of this makes sense and helps you find"
            print "the issue."
            print
            print "To disable this interactive behavior, pass"
            print "  debug_args=dict(interactive=False)"
            print "to generate_loop_schedules()."
            print 75*"-"
            raw_input("Enter:")
            print
            print

            debug.debug_length = len(debug.longest_rejected_schedule)
            for _ in generate_loop_schedules_internal(sched_state, loop_priority,
                    debug=debug):
                pass

        raise RuntimeError("no valid schedules found")

    logger.info("schedule %s: done" % kernel.name)

# }}}


def get_one_scheduled_kernel(kernel):
    kernel_count = 0

    for scheduled_kernel in generate_loop_schedules(kernel):
        kernel_count += 1

        if kernel_count == 1:
            # use the first schedule
            result = scheduled_kernel

        if kernel_count == 2:
            from warnings import warn
            warn("kernel scheduling was ambiguous--more than one "
                    "schedule found, ignoring", stacklevel=2)
            break

    return result


# vim: foldmethod=marker

from __future__ import division

from pytools import Record
import sys




# {{{ schedule items

class EnterLoop(Record):
    __slots__ = ["iname"]

class LeaveLoop(Record):
    __slots__ = ["iname"]

class RunInstruction(Record):
    __slots__ = ["insn_id"]

class Barrier(Record):
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
    from loopy.kernel import Instruction
    if not isinstance(source, Instruction):
        source = kernel.id_to_insn[source]
    if not isinstance(target, Instruction):
        target = kernel.id_to_insn[target]

    local_vars = kernel.local_var_names()

    tgt_write = set([target.get_assignee_var_name()]) & local_vars
    tgt_read = target.get_read_var_names() & local_vars

    src_write = set([source.get_assignee_var_name()]) & local_vars
    src_read = source.get_read_var_names() & local_vars

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






def get_barrier_dependent_in_schedule(kernel, source, schedule):
    """
    :arg source: an instruction id for the source of the dependency
    """
    unordered = False
    for sched_item in schedule:
        if isinstance(sched_item, RunInstruction) and sched_item.insn_id == source:
            unordered = True

    for sched_item in schedule:
        if isinstance(sched_item, RunInstruction):
            temp_res = get_barrier_needing_dependency(
                    kernel, sched_item.insn_id, source, unordered=unordered)
            if temp_res:
                return temp_res




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
        else:
            assert False

    return " ".join(entries), len(entries)

class SchedulerDebugger:
    def __init__(self, debug_length):
        self.longest_rejected_schedule = []
        self.success_counter = 0
        self.dead_end_counter = 0
        self.debug_length = debug_length

        self.update()

    def update(self):
        if (self.success_counter + self.dead_end_counter) % 50 == 0:
            sys.stdout.write("\rscheduling... %d successes, "
                    "%d dead ends (longest %d)" % (
                        self.success_counter,
                        self.dead_end_counter,
                        len(self.longest_rejected_schedule)))
            sys.stdout.flush()

    def log_success(self, schedule):
        self.success_counter += 1
        self.update()

    def log_dead_end(self, schedule):
        if len(schedule) > len(self.longest_rejected_schedule):
            self.longest_rejected_schedule = schedule
        self.dead_end_counter += 1
        self.update()

    def done_scheduling(self):
        sys.stdout.write("\rscheduler finished                                         \n")
        sys.stdout.flush()

# }}}

# {{{ scheduling algorithm

def generate_loop_schedules_internal(kernel, loop_priority, schedule=[], debug=None):
    all_insn_ids = set(insn.id for insn in kernel.instructions)

    scheduled_insn_ids = set(sched_item.insn_id for sched_item in schedule
            if isinstance(sched_item, RunInstruction))

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

    from loopy.kernel import ParallelTag
    parallel_inames = set(
            iname for iname in kernel.all_inames()
            if isinstance(kernel.iname_to_tag.get(iname), ParallelTag))

    # }}}

    # {{{ decide about debug mode

    debug_mode = False
    if debug is not None:
        if (debug.debug_length is not None
                and len(schedule) >= debug.debug_length):
            debug_mode = True

    if debug_mode:
        print kernel
        print "--------------------------------------------"
        print dump_schedule(schedule)

    #if len(schedule) == 3:
        #from pudb import set_trace; set_trace()

    if debug_mode:
        print "active:", ",".join(active_inames)
        print "entered:", ",".join(entered_inames)

    # }}}

    made_progress = False

    # {{{ see if any insn can be scheduled now

    unscheduled_insn_ids = list(all_insn_ids - scheduled_insn_ids)
    insns_with_satisfied_deps = set()

    for insn_id in unscheduled_insn_ids:
        insn = kernel.id_to_insn[insn_id]

        schedule_now = set(insn.insn_deps) <= scheduled_insn_ids
        if schedule_now:
            insns_with_satisfied_deps.add(insn_id)

        if not schedule_now:
            if debug_mode:
                print "instruction '%s' is missing insn depedencies '%s'" % (
                        insn.id, ",".join(set(insn.insn_deps) - scheduled_insn_ids))
            continue

        if insn.boostable == True:
            # If insn is boostable, it may be placed inside a more deeply
            # nested loop without harm.

            # But if it can be scheduled on the way *out* of the currently
            # active loops, now is not the right moment.

            schedulable_at_loop_levels = []

            for active_loop_count in xrange(len(active_inames), -1, -1):
                outer_active_inames = set(active_inames[:active_loop_count])
                if (
                        kernel.insn_inames(insn) - parallel_inames
                        <=
                        outer_active_inames - parallel_inames):

                    schedulable_at_loop_levels.append(active_loop_count)

            if schedulable_at_loop_levels != [len(active_inames)]:
                schedule_now = False
                if debug_mode:
                    if schedulable_at_loop_levels:
                        print ("instruction '%s' will be scheduled when more "
                                "loops have been exited" % insn.id)
                    else:
                        print ("instruction '%s' is missing inames '%s'"
                                % (insn.id, ",".join(
                                    (kernel.insn_inames(insn) - parallel_inames)
                                    -
                                    (outer_active_inames - parallel_inames))))

        elif insn.boostable == False:
            # If insn is not boostable, we must insist that it is placed inside
            # the exactly correct set of loops.

            schedule_now = schedule_now and (
                    kernel.insn_inames(insn) - parallel_inames
                    ==
                    active_inames_set - parallel_inames)

            if debug_mode:
                print ("instruction '%s' is not boostable and doesn't "
                        "match the active inames" % insn.id)

        else:
            raise RuntimeError("instruction '%s' has undetermined boostability"
                    % insn.id)

        if schedule_now:
            scheduled_insn_ids.add(insn.id)
            schedule = schedule + [RunInstruction(insn_id=insn.id)]
            made_progress = True

    unscheduled_insn_ids = list(all_insn_ids - scheduled_insn_ids)

    # }}}

    # {{{ see if we're ready to leave a loop

    if  last_entered_loop is not None:
        can_leave = True
        for insn_id in unscheduled_insn_ids:
            insn = kernel.id_to_insn[insn_id]
            if last_entered_loop in kernel.insn_inames(insn):
                can_leave = False
                break

        if can_leave:
            schedule = schedule + [LeaveLoop(iname=last_entered_loop)]
            made_progress = True

    # }}}

    # {{{ see if any loop can be entered now

    available_loops = (kernel.all_referenced_inames()
            # loops can only be entered once
            - entered_inames
            # there's no notion of 'entering' a parallel loop
            - parallel_inames
            )

    # Don't be eager about scheduling new loops--if progress has been made,
    # revert to top of scheduler and see if more progress can be made another
    # way. (hence 'and not made_progress')

    if available_loops and not made_progress:
        useful_loops = []

        for iname in available_loops:
            # {{{ determine if that gets us closer to being able to scheduling an insn

            useful = False

            hypothetical_active_loops = active_inames_set | set([iname])
            for insn_id in unscheduled_insn_ids:
                if insn_id not in insns_with_satisfied_deps:
                    continue

                insn = kernel.id_to_insn[insn_id]
                if insn.boostable:
                    # if insn is boostable, just increasing the number of used
                    # inames is enough--not necessarily all must be truly 'useful'.

                    if iname in kernel.insn_inames(insn):
                        useful = True
                        break
                else:
                    if hypothetical_active_loops <= kernel.insn_inames(insn):
                        useful = True
                        break

            if not useful:
                if debug_mode:
                    print "iname '%s' deemed not useful" % iname
                continue

            useful_loops.append(iname)

            # }}}

        # {{{ tier building

        # Build priority tiers. If a schedule is found in the first tier, then
        # loops in the second are not even tried (and so on).

        loop_priority_set = set(loop_priority)
        useful_and_desired = set(useful_loops) & loop_priority_set

        if useful_and_desired:
            priority_tiers = [[iname]
                    for iname in loop_priority
                    if iname in useful_and_desired]

            priority_tiers.append(
                    set(useful_loops) - loop_priority_set)
        else:
            priority_tiers = [useful_loops]

        # }}}

        if debug_mode:
            print "useful inames: %s" % ",".join(useful_loops)
            raw_input("Enter:")

        for tier in priority_tiers:
            found_viable_schedule = False

            for iname in tier:
                new_schedule = schedule + [EnterLoop(iname=iname)]
                for sub_sched in generate_loop_schedules_internal(
                        kernel, loop_priority, new_schedule,
                        debug=debug):
                    found_viable_schedule = True
                    yield sub_sched

            if found_viable_schedule:
                return

    # }}}


    if debug_mode:
        raw_input("Enter:")

    if not active_inames and not available_loops and not unscheduled_insn_ids:
        # if done, yield result
        yield schedule
    elif made_progress:
        # if not done, but made some progress--try from the top
            for sub_sched in generate_loop_schedules_internal(
                    kernel, loop_priority, schedule,
                    debug=debug):
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

            # {{{ issue dependency-based barriers for contents of nested loop

            # (i.e. if anything *in* the loop depends on something beforehand)

            for insn_id in owed_barriers:
                dep = get_barrier_dependent_in_schedule(kernel, insn_id, subloop)
                if dep:
                    issue_barrier(is_pre_barrier=False, dep=dep)
                    break

            # }}}

            subresult, sub_owed_barriers = insert_barriers(
                    kernel, subloop[1:-1], level+1)

            # {{{ issue pre-barriers for contents of nested loop

            if not loop_had_barrier[0]:
                for insn_id in sub_owed_barriers:
                    dep = get_barrier_dependent_in_schedule(
                            kernel, insn_id, schedule)
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

            assignee_temp_var = kernel.temporary_variables.get(
                    insn.get_assignee_var_name())
            if assignee_temp_var is not None and assignee_temp_var.is_local:
                dep = get_barrier_dependent_in_schedule(kernel, insn.id, schedule)

                if dep:
                    issue_barrier(is_pre_barrier=True, dep=dep)

                result.append(sched_item)
                owed_barriers.add(insn.id)
            else:
                result.append(sched_item)

        else:
            assert False

    return result, owed_barriers

# }}}

# {{{ main scheduling entrypoint

def generate_loop_schedules(kernel, loop_priority=[], debug=False):
    from loopy.preprocess import preprocess_kernel
    kernel = preprocess_kernel(kernel)

    from loopy.check import run_automatic_checks
    run_automatic_checks(kernel)

    schedule_count = 0

    if debug:
        if debug == True:
            debug = SchedulerDebugger(None)
        else:
            debug = SchedulerDebugger(debug)
    else:
        debug = None

    for gen_sched in generate_loop_schedules_internal(kernel, loop_priority,
            debug=debug):
        gen_sched, owed_barriers = insert_barriers(kernel, gen_sched)
        if owed_barriers:
            from warnings import warn
            from loopy import LoopyAdvisory
            warn("Barrier insertion finished without inserting barriers for "
                    "local memory writes in these instructions: '%s'. "
                    "This often means that local memory was "
                    "written, but never read." % ",".join(owed_barriers), LoopyAdvisory)

        yield kernel.copy(schedule=gen_sched)

        schedule_count += 1

    if not schedule_count:
        raise RuntimeError("no valid schedules found")

# }}}





# vim: foldmethod=marker

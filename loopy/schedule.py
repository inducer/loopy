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

class SchedulerDebugger:
    def __init__(self, debug_length):
        self.longest_rejected_schedule = []
        self.success_counter = 0
        self.dead_end_counter = 0
        self.debug_length = debug_length

        self.elapsed_store = 0
        self.start()
        self.wrote_status = False

        self.update()

    def update(self):
        if (self.success_counter + self.dead_end_counter) % 50 == 0:
            if self.debug_length or self.elapsed_time() > 1:
                sys.stdout.write("\rscheduling... %d successes, "
                        "%d dead ends (longest %d)" % (
                            self.success_counter,
                            self.dead_end_counter,
                            len(self.longest_rejected_schedule)))
                sys.stdout.flush()
                self.wrote_status = True

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
        from time import time
        self.elapsed_store += time()-self.start_time

    def start(self):
        from time import time
        self.start_time = time()
# }}}

# {{{ scheduling algorithm

def generate_loop_schedules_internal(kernel, loop_priority, schedule=[], allow_boost=False, debug=None):
    all_insn_ids = set(insn.id for insn in kernel.instructions)

    scheduled_insn_ids = set(sched_item.insn_id for sched_item in schedule
            if isinstance(sched_item, RunInstruction))

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

    #print dump_schedule(schedule), len(schedule)
    if debug_mode:
        print kernel
        print "--------------------------------------------"
        print "CURRENT SCHEDULE:"
        print dump_schedule(schedule), len(schedule)
        print "boost allowed:", allow_boost
        print "--------------------------------------------"

    #if len(schedule) == 2:
        #from pudb import set_trace; set_trace()

    # }}}

    made_progress = False

    # {{{ see if any insn can be scheduled now

    # Also take note of insns that have a chance of being schedulable inside
    # the current loop nest, in this set:

    reachable_insn_ids = set()

    for insn_id in all_insn_ids - scheduled_insn_ids:
        insn = kernel.id_to_insn[insn_id]

        schedule_now = set(insn.insn_deps) <= scheduled_insn_ids

        if not schedule_now:
            if debug_mode:
                print "instruction '%s' is missing insn depedencies '%s'" % (
                        insn.id, ",".join(set(insn.insn_deps) - scheduled_insn_ids))
            continue

        want = kernel.insn_inames(insn) - parallel_inames
        have = active_inames_set - parallel_inames

        # If insn is boostable, it may be placed inside a more deeply
        # nested loop without harm.

        if allow_boost:
            have = have - insn.boostable_into

        if want != have:
            schedule_now = False

            if debug_mode:
                if want-have:
                    print ("instruction '%s' is missing inames '%s'"
                            % (insn.id, ",".join(want-have)))
                if have-want:
                    print ("instruction '%s' won't work under inames '%s'"
                            % (insn.id, ",".join(have-want)))

        # {{{ determine reachability

        if (not schedule_now and have <= want):
            reachable_insn_ids.add(insn_id)
        else:
            if debug_mode:
                print ("    '%s' also not reachable because it won't work under '%s'"
                        % (insn.id, ",".join(have-want)))

        # }}}

        if schedule_now:
            if debug_mode:
                print "scheduling '%s'" % insn.id
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
                if debug_mode:
                    print("cannot leave '%s' because '%s' still depends on it"
                            % (last_entered_loop, insn.id))
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

    if debug_mode:
        print "--------------------------------------------"
        print "available :", ",".join(available_loops)
        print "active:", ",".join(active_inames)
        print "entered:", ",".join(entered_inames)
        print "--------------------------------------------"
        print "reachable insns:", ",".join(reachable_insn_ids)
        print "--------------------------------------------"

    # Don't be eager about scheduling new loops--if progress has been made,
    # revert to top of scheduler and see if more progress can be made another
    # way. (hence 'and not made_progress')

    if available_loops and not made_progress:
        useful_loops = []

        for iname in available_loops:
            if not kernel.loop_nest_map()[iname] <= active_inames_set | parallel_inames:
                continue

            # {{{ determine if that gets us closer to being able to schedule an insn

            useful = False

            hypothetically_active_loops = active_inames_set | set([iname])
            for insn_id in reachable_insn_ids:
                insn = kernel.id_to_insn[insn_id]

                want = kernel.insn_inames(insn) | insn.boostable_into

                if hypothetically_active_loops <= want:
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

        for tier in priority_tiers:
            found_viable_schedule = False

            for iname in tier:
                new_schedule = schedule + [EnterLoop(iname=iname)]

                for sub_sched in generate_loop_schedules_internal(
                        kernel, loop_priority, new_schedule,
                        allow_boost=rec_allow_boost,
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
        debug.log_success(schedule)

        yield schedule

    elif made_progress:
        # if not done, but made some progress--try from the top
        for sub_sched in generate_loop_schedules_internal(
                kernel, loop_priority, schedule,
                allow_boost=rec_allow_boost, debug=debug):
            yield sub_sched
    else:
        if not allow_boost and allow_boost is not None:
            # try again with boosting allowed
            for sub_sched in generate_loop_schedules_internal(
                    kernel, loop_priority, schedule=schedule,
                    allow_boost=True, debug=debug):
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

            assignee_temp_var = kernel.temporary_variables.get(
                    insn.get_assignee_var_name())
            if assignee_temp_var is not None and assignee_temp_var.is_local:
                dep = get_barrier_dependent_in_schedule(kernel, insn.id, schedule,
                        unordered=True)

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

def generate_loop_schedules(kernel, loop_priority=[], debug=None):
    from loopy.preprocess import preprocess_kernel
    kernel = preprocess_kernel(kernel)

    from loopy.check import run_automatic_checks
    run_automatic_checks(kernel)

    schedule_count = 0

    debug = SchedulerDebugger(debug)

    generators = [
            generate_loop_schedules_internal(kernel, loop_priority,
                debug=debug, allow_boost=None),
            generate_loop_schedules_internal(kernel, loop_priority,
                debug=debug)]
    for gen in generators:
        for gen_sched in gen:
            gen_sched, owed_barriers = insert_barriers(kernel, gen_sched)
            if owed_barriers:
                from warnings import warn
                from loopy import LoopyAdvisory
                warn("Barrier insertion finished without inserting barriers for "
                        "local memory writes in these instructions: '%s'. "
                        "This often means that local memory was "
                        "written, but never read." % ",".join(owed_barriers), LoopyAdvisory)

            debug.stop()
            yield kernel.copy(schedule=gen_sched)
            debug.start()

            schedule_count += 1

        # if no-boost mode yielded a viable schedule, stop now
        if schedule_count:
            break

    debug.done_scheduling()

    if not schedule_count:
        raise RuntimeError("no valid schedules found")

# }}}





# vim: foldmethod=marker

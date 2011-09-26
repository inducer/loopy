from __future__ import division

from pytools import Record




# {{{ schedule items

class EnterLoop(Record):
    __slots__ = ["iname"]

class LeaveLoop(Record):
    __slots__ = ["iname"]

class RunInstruction(Record):
    __slots__ = ["insn_id"]

class Barrier(Record):
    __slots__ = []

# }}}




def fix_grid_sizes(kernel):
    from warnings import warn
    warn("fix_grid_sizes is unimplemented")
    return kernel




def generate_loop_dep_graph(kernel):
    """
    :return: a dict mapping an iname to the ones that need to be entered
        before it.
    """
    # FIXME perhaps useful?
    result = {}

    print "------------------------------------------------------"
    for i, insn_a in enumerate(kernel.instructions):
        print i, insn_a
        print insn_a.all_inames()

    print "------------------------------------------------------"
    all_inames = kernel.all_inames()
    for i_a, insn_a in enumerate(kernel.instructions):
        for i_b, insn_b in enumerate(kernel.instructions):
            if i_a == i_b:
                continue

            a = insn_a.all_inames()
            b = insn_b.all_inames()
            intersection = a & b
            sym_difference = (a|b) - intersection

            print i_a, i_b, intersection, sym_difference
            if a <= b or b <= a:
                for sd in sym_difference:
                    result.setdefault(sd, set()).update(intersection)

    print "------------------------------------------------------"
    return result




def adjust_local_temp_var_storage(kernel):
    from warnings import warn
    warn("adjust_local_temp_var_storage is unimplemented")
    return kernel




def find_writers(kernel):
    """
    :return: a dict that maps variable names to ids of insns that
        write to that variable.
    """
    writer_insn_ids = {}

    admissible_write_vars = (
            set(arg.name for arg in kernel.args)
            | set(tv.name for tv in kernel.temporary_variables))

    from pymbolic.primitives import Variable, Subscript
    for insn in kernel.instructions:
        if isinstance(insn.assignee, Variable):
            var_name = insn.assignee.name
        elif isinstance(insn.assignee, Subscript):
            var = insn.assignee.aggregate
            assert isinstance(var, Variable)
            var_name = var.name
        else:
            raise RuntimeError("invalid lvalue '%s'" % insn.assignee)

        if var_name not in admissible_write_vars:
            raise RuntimeError("writing to '%s' is not allowed" % var_name)

        writer_insn_ids.setdefault(var_name, set()).add(insn.id)

    return writer_insn_ids




def add_automatic_dependencies(kernel):
    writer_map = find_writers(kernel)

    arg_names = set(arg.name for arg in kernel.args)

    var_names = arg_names | set(tv.name for tv in kernel.temporary_variables)

    from loopy.symbolic import DependencyMapper
    dep_map = DependencyMapper(composite_leaves=False)
    new_insns = []
    for insn in kernel.instructions:
        read_vars = (
                set(var.name for var in dep_map(insn.expression)) 
                & var_names)

        auto_deps = []
        for var in read_vars:
            var_writers = writer_map.get(var, set())

            if not var_writers and var not in var_names:
                from warnings import warn
                warn("'%s' is read, but never written." % var)

            if len(var_writers) > 1 and not var_writers & set(insn.insn_deps):
                from warnings import warn
                warn("'%s' is written from more than one place, "
                        "but instruction '%s' (which reads this variable) "
                        "does not specify a dependency on any of the writers."
                        % (var, insn.id))

            if len(var_writers) == 1:
                auto_deps.extend(var_writers)

        new_insns.append(
                insn.copy(
                    insn_deps=insn.insn_deps + auto_deps))

    return kernel.copy(instructions=new_insns)




def generate_loop_schedules_internal(kernel, schedule=[]):
    print schedule

    #if len(schedule) == 8:
        #from pudb import set_trace; set_trace()

    all_insn_ids = set(insn.id for insn in kernel.instructions)

    scheduled_insn_ids = set(sched_item.insn_id for sched_item in schedule
            if isinstance(sched_item, RunInstruction))

    # {{{ find active and entered loops

    active_loops = []
    entered_loops = set()

    for sched_item in schedule:
        if isinstance(sched_item, EnterLoop):
            active_loops.append(sched_item.iname)
            entered_loops.add(sched_item.iname)
        if isinstance(sched_item, LeaveLoop):
            active_loops.pop()

    if active_loops:
        last_entered_loop = active_loops[-1]
    else:
        last_entered_loop = None
    active_loops = set(active_loops)

    # }}}

    made_progress = False

    # {{{ see if any insn can be scheduled now

    unscheduled_insn_ids = list(all_insn_ids - scheduled_insn_ids)

    for insn_id in unscheduled_insn_ids:
        insn = kernel.id_to_insn[insn_id]
        if (active_loops == insn.all_inames()
                and set(insn.insn_deps) <= scheduled_insn_ids):
            scheduled_insn_ids.add(insn.id)
            schedule = schedule + [RunInstruction(insn_id=insn.id)]
            made_progress = True

    unscheduled_insn_ids = list(all_insn_ids - scheduled_insn_ids)

    # }}}

    # {{{ see if any loop can be scheduled now

    available_loops = kernel.all_inames() - entered_loops

    if available_loops:
        found_something_useful = False

        for iname in available_loops:
            # {{{ determine if that gets us closer to being able to scheduling an insn

            useful = False

            hypothetical_active_loops = active_loops | set([iname])
            for insn_id in unscheduled_insn_ids:
                insn = kernel.id_to_insn[insn_id]
                if hypothetical_active_loops <= insn.all_inames():
                    useful = True
                    break

            if not useful:
                continue

            found_something_useful = True

            # }}}

            new_schedule = schedule + [EnterLoop(iname=iname)]
            for sub_sched in generate_loop_schedules_internal(
                    kernel, new_schedule):
                yield sub_sched

        if found_something_useful:
            return

    # }}}

    # {{{ see if we're ready to leave a loop

    if  last_entered_loop is not None:
        can_leave = True
        for insn_id in unscheduled_insn_ids:
            insn = kernel.id_to_insn[insn_id]
            if last_entered_loop in insn.all_inames():
                can_leave = False
                break

        if can_leave:
            schedule = schedule + [LeaveLoop(iname=last_entered_loop)]
            made_progress = True

    # }}}

    if not active_loops and not available_loops and not unscheduled_insn_ids:
        # if done, yield result
        yield schedule
    else:
        # if not done, but made some progress--try from the top
        if made_progress:
            for sub_sched in generate_loop_schedules_internal(kernel, schedule):
                yield sub_sched








def generate_loop_schedules(kernel):
    from loopy import realize_reduction
    kernel = realize_reduction(kernel)

    kernel = adjust_local_temp_var_storage(kernel)

    # {{{ check that all CSEs have been realized

    from loopy.symbolic import CSECallbackMapper

    def map_cse(expr, rec):
        raise RuntimeError("all CSEs must be realized before scheduling")

    for insn in kernel.instructions:
        CSECallbackMapper(map_cse)(insn.expression)

    # }}}

    kernel = fix_grid_sizes(kernel)

    if 0:
        loop_dep_graph = generate_loop_dep_graph(kernel)
        for k, v in loop_dep_graph.iteritems():
            print "%s: %s" % (k, ",".join(v))
        1/0

    kernel = add_automatic_dependencies(kernel)

    print kernel

    #grid_size, group_size = find_known_grid_and_group_sizes(kernel)

    #kernel = assign_grid_and_group_indices(kernel)

    for gen_sched in generate_loop_schedules_internal(kernel):
        print gen_sched

        if False:
            gen_sched = insert_barriers(gen_sched)
            schedule = insert_parallel_dim_check_points(schedule=gen_sched)
            yield kernel.copy(schedule=gen_sched)


    1/0





def generate_loop_schedules_old(kernel, hints=[]):
    # OLD!
    from loopy.kernel import TAG_GROUP_IDX, TAG_WORK_ITEM_IDX, TAG_ILP, ParallelTag

    prev_schedule = kernel.schedule
    if prev_schedule is None:
        prev_schedule = [
                ScheduledLoop(iname=iname)
                for iname in (
                    kernel.ordered_inames_by_tag_type(TAG_GROUP_IDX)
                    + kernel.ordered_inames_by_tag_type(TAG_WORK_ITEM_IDX))]

    scheduled_inames = set(sch_item.iname
            for sch_item in prev_schedule
            if isinstance(sch_item, ScheduledLoop))

    # have a schedulable prefetch? load, schedule it
    had_usable_prefetch = False
    locally_parallel_inames = set(
            iname for iname in scheduled_inames
            if isinstance(kernel.iname_to_tag.get(iname), 
                (TAG_ILP, TAG_WORK_ITEM_IDX)))

    for pf in kernel.prefetch.itervalues():
        # already scheduled? never mind then.
        if pf in prev_schedule:
            continue

        # a free variable not known yet? then we're not ready
        if not pf.free_variables() <= scheduled_inames:
            continue

        # a prefetch variable already scheduled, but not borrowable?
        # (only work item index variables are borrowable)

        if set(pf.all_inames()) & (scheduled_inames - locally_parallel_inames):
            # dead end: we won't be able to schedule this prefetch
            # in this branch. at least one of its loop dimensions
            # was already scheduled, and that dimension is not
            # borrowable.

            #print "UNSCHEDULABLE", kernel.schedule
            return

        new_kernel = kernel.copy(schedule=prev_schedule+[pf])
        for knl in generate_loop_schedules(new_kernel):
            had_usable_prefetch = True
            yield knl

    if had_usable_prefetch:
        # because we've already recursed
        return

    # Build set of potentially schedulable variables
    # Don't re-schedule already scheduled variables
    schedulable = kernel.all_inames() - scheduled_inames

    # Schedule in the following order:
    # - serial output inames
    # - remaining parallel output inames (i.e. ILP)
    # - output write
    # - reduction
    # Don't schedule reduction variables until all output
    # variables are taken care of. Once they are, schedule
    # output writing.
    parallel_output_inames = set(oin for oin in kernel.output_inames()
            if isinstance(kernel.iname_to_tag.get(oin), ParallelTag))

    serial_output_inames = kernel.output_inames() - parallel_output_inames

    if schedulable & serial_output_inames:
        schedulable = schedulable & serial_output_inames

    if schedulable & parallel_output_inames:
        schedulable  = schedulable & parallel_output_inames

    if kernel.output_inames() <= scheduled_inames:
        if not any(isinstance(sch_item, WriteOutput)
                for sch_item in prev_schedule):
            kernel = kernel.copy(
                    schedule=prev_schedule + [WriteOutput()])
            prev_schedule = kernel.schedule

    # Don't schedule variables that are prefetch axes
    # for not-yet-scheduled prefetches.
    unsched_prefetch_axes = set(iname
            for pf in kernel.prefetch.itervalues()
            if pf not in prev_schedule
            for iname in pf.all_inames()
            if not isinstance(kernel.iname_to_tag.get(iname), ParallelTag))
    schedulable -= unsched_prefetch_axes

    while hints and hints[0] in scheduled_inames:
        hints = hints[1:]

    if hints and hints[0] in schedulable:
        schedulable = set([hints[0]])

    if schedulable:
        # have a schedulable variable? schedule a loop for it, recurse
        for iname in schedulable:
            new_kernel = kernel.copy(schedule=prev_schedule+[ScheduledLoop(iname=iname)])
            for knl in generate_loop_schedules(new_kernel, hints):
                yield knl
    else:
        # all loop dimensions and prefetches scheduled?
        # great! yield the finished product if it is complete

        from loopy import LoopyAdvisory

        if hints:
            from warnings import warn
            warn("leftover schedule hints: "+ (", ".join(hints)),
                    LoopyAdvisory)

        all_inames_scheduled = len(scheduled_inames) == len(kernel.all_inames())

        from loopy.prefetch import LocalMemoryPrefetch
        all_pf_scheduled =  len(set(sch_item for sch_item in prev_schedule
            if isinstance(sch_item, LocalMemoryPrefetch))) == len(kernel.prefetch)
        output_scheduled = len(set(sch_item for sch_item in prev_schedule
            if isinstance(sch_item, WriteOutput))) == 1

        if all_inames_scheduled and all_pf_scheduled and output_scheduled:
            yield kernel




# vim: foldmethod=marker

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




def check_double_use_of_hw_dimensions(kernel):
    from loopy.kernel import UniqueTag

    for insn in kernel.instructions:
        insn_tag_keys = set()
        for iname in insn.all_inames():
            tag = kernel.iname_to_tag.get(iname)
            if isinstance(tag, UniqueTag):
                key = tag.key
                if key in insn_tag_keys:
                    raise RuntimeError("instruction '%s' has two "
                            "inames tagged '%s'" % (insn.id, tag))

                insn_tag_keys.add(key)




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
            | set(kernel.temporary_variables.iterkeys()))

    for insn in kernel.instructions:
        var_name = insn.get_assignee_var_name()

        if var_name not in admissible_write_vars:
            raise RuntimeError("writing to '%s' is not allowed" % var_name)

        writer_insn_ids.setdefault(var_name, set()).add(insn.id)

    return writer_insn_ids




def add_automatic_dependencies(kernel):
    writer_map = find_writers(kernel)

    arg_names = set(arg.name for arg in kernel.args)

    var_names = arg_names | set(kernel.temporary_variables.iterkeys())

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
    active_inames = set(active_inames)

    from loopy.kernel import ParallelTag
    parallel_inames = set(
            iname for iname in kernel.all_inames()
            if isinstance(kernel.iname_to_tag.get(iname), ParallelTag))

    # }}}

    made_progress = False

    # {{{ see if any insn can be scheduled now

    unscheduled_insn_ids = list(all_insn_ids - scheduled_insn_ids)

    for insn_id in unscheduled_insn_ids:
        insn = kernel.id_to_insn[insn_id]
        if (active_inames - parallel_inames 
                == insn.all_inames() - parallel_inames
                and set(insn.insn_deps) <= scheduled_insn_ids):
            scheduled_insn_ids.add(insn.id)
            schedule = schedule + [RunInstruction(insn_id=insn.id)]
            made_progress = True

    unscheduled_insn_ids = list(all_insn_ids - scheduled_insn_ids)

    # }}}

    # {{{ see if any loop can be entered now

    available_loops = (kernel.all_inames()
            # loops can only be entered once
            - entered_inames
            # there's no notion of 'entering' a parallel loop
            - parallel_inames
            )

    if available_loops:
        found_something_useful = False

        for iname in available_loops:
            # {{{ determine if that gets us closer to being able to scheduling an insn

            useful = False

            hypothetical_active_loops = active_inames | set([iname])
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

    if not active_inames and not available_loops and not unscheduled_insn_ids:
        # if done, yield result
        yield schedule
    else:
        # if not done, but made some progress--try from the top
        if made_progress:
            for sub_sched in generate_loop_schedules_internal(kernel, schedule):
                yield sub_sched




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



def has_dependent_in_schedule(kernel, insn_id, schedule):
    from pytools import any
    return any(sched_item
            for sched_item in schedule
            if isinstance(sched_item, RunInstruction)
            and kernel.id_to_insn[sched_item.insn_id].insn_deps)




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

    def issue_barrier(is_pre_barrier):
        owed_barriers.clear()
        if result and isinstance(result[-1], Barrier):
            return

        if is_pre_barrier:
            if loop_had_barrier[0] or level == 0:
                return

        loop_had_barrier[0] = True
        result.append(Barrier())

    i = 0
    while i < len(schedule):
        sched_item = schedule[i]

        if isinstance(sched_item, EnterLoop):
            subloop, new_i = gather_schedule_subloop(schedule, i)

            # {{{ issue dependency-based barriers for contents of nested loop

            for insn_id in owed_barriers:
                if has_dependent_in_schedule(kernel, insn_id, subloop):
                    issue_barrier(is_pre_barrier=False)
                    break

            # }}}

            subresult, sub_owed_barriers = insert_barriers(
                    kernel, subloop[1:-1], level+1)

            # {{{ issue pre-barriers for contents of nested loop

            if not loop_had_barrier:
                for insn_id in sub_owed_barriers:
                    if has_dependent_in_schedule(
                            kernel, insn_id, schedule):
                        issue_barrier(is_pre_barrier=True)

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

            if insn.id in owed_barriers:
                issue_barrier(is_pre_barrier=False)

            # }}}

            assignee_temp_var = kernel.temporary_variables.get(
                    insn.get_assignee_var_name())
            if assignee_temp_var is not None and assignee_temp_var.is_local:
                if level == 0:
                    assert has_dependent_in_schedule(
                            kernel, insn.id, schedule)

                if has_dependent_in_schedule(kernel, insn.id, schedule):
                    issue_barrier(is_pre_barrier=True)

                result.append(sched_item)
                owed_barriers.add(insn.id)
            else:
                result.append(sched_item)

        else:
            assert False

    return result, owed_barriers




def generate_loop_schedules(kernel):
    from loopy import realize_reduction
    kernel = realize_reduction(kernel)

    check_double_use_of_hw_dimensions(kernel)

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
        gen_sched, owed_barriers = insert_barriers(kernel, gen_sched)
        assert not owed_barriers

        print gen_sched

        if False:
            schedule = insert_parallel_dim_check_points(schedule=gen_sched)
            yield kernel.copy(schedule=gen_sched)


    1/0





# vim: foldmethod=marker

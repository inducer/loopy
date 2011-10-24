from __future__ import division

from pytools import Record
import pyopencl as cl
import pyopencl.characterize as cl_char




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

# {{{ rewrite reduction to imperative form

def realize_reduction(kernel):
    new_insns = []
    new_temporary_variables = kernel.temporary_variables.copy()

    from loopy.kernel import IlpTag

    def map_reduction(expr, rec):
        sub_expr = rec(expr.expr)

        # {{{ see if this reduction is nested inside some ILP loops

        ilp_inames = [iname
                for iname in insn.all_inames()
                if isinstance(kernel.iname_to_tag[iname], IlpTag)]

        from loopy.isl_helpers import static_max_of_pw_aff

        ilp_iname_lengths = []
        for iname in ilp_inames:
            bounds = kernel.get_iname_bounds(iname)

            ilp_iname_lengths.append(
                static_max_of_pw_aff(bounds.size, constants_only=True))

        # }}}

        from pymbolic import var

        target_var_name = kernel.make_unique_var_name("acc",
                extra_used_vars=set(tv for tv in new_temporary_variables))
        target_var = var(target_var_name)

        if ilp_inames:
            target_var = target_var[
                    tuple(var(ilp_iname) for ilp_iname in ilp_inames)]

        from loopy.kernel import Instruction

        from loopy.kernel import TemporaryVariable
        new_temporary_variables[target_var_name] = TemporaryVariable(
                name=target_var_name,
                dtype=expr.operation.dtype,
                shape=tuple(ilp_iname_lengths),
                is_local=False)

        init_insn = Instruction(
                id=kernel.make_unique_instruction_id(
                    extra_used_ids=set(ni.id for ni in new_insns)),
                assignee=target_var,
                forced_iname_deps=list(insn.all_inames() - set(expr.inames)),
                expression=expr.operation.neutral_element)

        new_insns.append(init_insn)

        reduction_insn = Instruction(
                id=kernel.make_unique_instruction_id(
                    extra_used_ids=set(ni.id for ni in new_insns)),
                assignee=target_var,
                expression=expr.operation(target_var, sub_expr),
                insn_deps=[init_insn.id],
                forced_iname_deps=list(insn.all_inames()))

        new_insns.append(reduction_insn)

        new_insn_insn_deps.append(reduction_insn.id)

        return target_var

    from loopy.symbolic import ReductionCallbackMapper
    cb_mapper = ReductionCallbackMapper(map_reduction)

    for insn in kernel.instructions:
        new_insn_insn_deps = []

        new_expression = cb_mapper(insn.expression)

        new_insn = insn.copy(
                    expression=new_expression,
                    insn_deps=insn.insn_deps
                        + new_insn_insn_deps)

        new_insns.append(new_insn)

    return kernel.copy(
            instructions=new_insns,
            temporary_variables=new_temporary_variables)

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



def has_dependent_in_schedule(kernel, insn_id, schedule):
    from pytools import any
    return any(sched_item
            for sched_item in schedule
            if isinstance(sched_item, RunInstruction)
            and kernel.id_to_insn[sched_item.insn_id].insn_deps)




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
        result.update(kernel.id_to_insn[sched_item.insn_id].all_inames())

    return result

# }}}

# {{{ hw axis sanity checks

def check_for_unused_hw_axes(kernel):
    group_size, local_size = kernel.get_grid_sizes_as_exprs()

    group_axes = set(range(len(group_size)))
    local_axes = set(range(len(local_size)))

    from loopy.kernel import LocalIndexTag, AutoLocalIndexTagBase, GroupIndexTag
    for insn in kernel.instructions:
        group_axes_used = set()
        local_axes_used = set()

        for iname in insn.all_inames():
            tag = kernel.iname_to_tag.get(iname)

            if isinstance(tag, LocalIndexTag):
                local_axes_used.add(tag.axis)
            elif isinstance(tag, GroupIndexTag):
                group_axes_used.add(tag.axis)
            elif isinstance(tag, AutoLocalIndexTagBase):
                raise RuntimeError("auto local tag encountered")

        if group_axes != group_axes_used:
            raise RuntimeError("instruction '%s' does not use all group hw axes"
                    % insn.id)
        if local_axes != local_axes_used:
            raise RuntimeError("instruction '%s' does not use all local hw axes"
                    % insn.id)





def check_for_double_use_of_hw_axes(kernel):
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

# }}}

# {{{ temp storage adjust for bank conflict

def adjust_local_temp_var_storage(kernel):
    new_temp_vars = {}

    lmem_size = cl_char.usable_local_mem_size(kernel.device)
    for temp_var in kernel.temporary_variables.itervalues():
        other_loctemp_nbytes = [tv.nbytes for tv in kernel.temporary_variables.itervalues()
                if tv.is_local and tv.name != temp_var.name]

        storage_shape = temp_var.storage_shape
        if storage_shape is None:
            storage_shape = temp_var.shape

        # sizes of all dims except the last one, which we may change
        # below to avoid bank conflicts
        from pytools import product
        other_dim_sizes = (tv.dtype.itemsize
                * product(storage_shape[:-1]))

        if kernel.device.local_mem_type == cl.device_local_mem_type.GLOBAL:
            # FIXME: could try to avoid cache associativity disasters
            new_storage_shape = storage_shape

        elif kernel.device.local_mem_type == cl.device_local_mem_type.LOCAL:
            min_mult = cl_char.local_memory_bank_count(kernel.device)
            good_incr = None
            new_storage_shape = storage_shape
            min_why_not = None

            for increment in range(storage_shape[-1]//2):

                test_storage_shape = storage_shape[:]
                test_storage_shape[-1] = test_storage_shape[-1] + increment
                new_mult, why_not = cl_char.why_not_local_access_conflict_free(
                        kernel.device, temp_var.dtype.itemsize,
                        temp_var.shape, test_storage_shape)

                # will choose smallest increment 'automatically'
                if new_mult < min_mult:
                    new_lmem_use = (other_loctemp_nbytes
                            + temp_var.dtype.itemsize*product(test_storage_shape))
                    if new_lmem_use < lmem_size:
                        new_storage_shape = test_storage_shape
                        min_mult = new_mult
                        min_why_not = why_not
                        good_incr = increment

            if min_mult != 1:
                from warnings import warn
                from loopy import LoopyAdvisory
                warn("could not find a conflict-free mem layout "
                        "for local variable '%s' "
                        "(currently: %dx conflict, increment: %d, reason: %s)"
                        % (temp_var.name, min_mult, good_incr, min_why_not),
                        LoopyAdvisory)
        else:
            from warnings import warn
            warn("unknown type of local memory")

            new_storage_shape = storage_shape

        new_temp_vars[temp_var.name] = temp_var.copy(storage_shape=new_storage_shape)

    return kernel.copy(temporary_variables=new_temp_vars)

# }}}

# {{{ automatic dependencies, find idempotent instructions

def find_accessors(kernel, readers):
    """
    :return: a dict that maps variable names to ids of insns that
        write to that variable.
    """
    result = {}

    admissible_vars = (
            set(arg.name for arg in kernel.args)
            | set(kernel.temporary_variables.iterkeys()))

    for insn in kernel.instructions:
        if readers:
            from loopy.symbolic import DependencyMapper
            var_names = DependencyMapper()(insn.expression) & admissible_vars
        else:
            var_name = insn.get_assignee_var_name()

            if var_name not in admissible_vars:
                raise RuntimeError("writing to '%s' is not allowed" % var_name)
            var_names = [var_name]

        for var_name in var_names:
            result.setdefault(var_name, set()).add(insn.id)

    return result




def add_idempotence_and_automatic_dependencies(kernel):
    writer_map = find_accessors(kernel, readers=False)

    arg_names = set(arg.name for arg in kernel.args)

    var_names = arg_names | set(kernel.temporary_variables.iterkeys())

    from loopy.symbolic import DependencyMapper
    dm = DependencyMapper(composite_leaves=False)
    dep_map = {}

    for insn in kernel.instructions:
        dep_map[insn.id] = (
                set(var.name for var in dm(insn.expression))
                & var_names)

    new_insns = []
    for insn in kernel.instructions:
        auto_deps = []

        # {{{ add automatic dependencies
        all_my_var_writers = set()
        for var in dep_map[insn.id]:
            var_writers = writer_map.get(var, set())
            all_my_var_writers |= var_writers

            if not var_writers and var not in arg_names:
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

        # }}}

        # {{{ find dependency loops, flag idempotence

        while True:
            last_all_my_var_writers = all_my_var_writers

            for writer_insn_id in last_all_my_var_writers:
                for var in dep_map[writer_insn_id]:
                    all_my_var_writers = all_my_var_writers | writer_map.get(var, set())

            if last_all_my_var_writers == all_my_var_writers:
                break

        # }}}

        new_insns.append(
                insn.copy(
                    insn_deps=insn.insn_deps + auto_deps,
                    idempotent=insn.id not in all_my_var_writers))

    return kernel.copy(instructions=new_insns)

# }}}

# {{{ guess good iname for local axis 0

def guess_good_iname_for_axis_0(kernel, insn):
    from loopy.kernel import ImageArg, ScalarArg

    approximate_arg_values = dict(
            (arg.name, arg.approximately)
            for arg in kernel.args
            if isinstance(arg, ScalarArg))

    # {{{ find all array accesses in insn

    from loopy.symbolic import ArrayAccessFinder
    ary_acc_exprs = list(ArrayAccessFinder()(insn.expression))

    from pymbolic.primitives import Subscript

    if isinstance(insn.assignee, Subscript):
        ary_acc_exprs.append(insn.assignee)

    # }}}

    # {{{ filter array accesses to only the global ones

    global_ary_acc_exprs = []

    for aae in ary_acc_exprs:
        ary_name = aae.aggregate.name
        arg = kernel.arg_dict.get(ary_name)
        if arg is None:
            continue

        if isinstance(arg, ImageArg):
            continue

        global_ary_acc_exprs.append(aae)

    # }}}

    # {{{ figure out which iname should get mapped to local axis 0

    # maps inames to vote counts
    vote_count_for_l0 = {}

    from loopy.symbolic import CoefficientCollector

    from pytools import argmin2, argmax2

    for aae in global_ary_acc_exprs:
        index_expr = aae.index
        if not isinstance(index_expr, tuple):
            index_expr = (index_expr,)

        ary_name = aae.aggregate.name
        arg = kernel.arg_dict.get(ary_name)

        ary_strides = arg.strides
        if ary_strides is None and len(index_expr) == 1:
            ary_strides = (1,)

        iname_to_stride = {}
        for iexpr_i, stride in zip(index_expr, ary_strides):
            coeffs = CoefficientCollector()(iexpr_i)
            for var_name, coeff in coeffs.iteritems():
                if var_name != 1:
                    new_stride = coeff*stride
                    old_stride = iname_to_stride.get(var_name, None)
                    if old_stride is None or new_stride < old_stride:
                        iname_to_stride[var_name] = new_stride

        from pymbolic import evaluate
        least_stride_iname, least_stride = argmin2((
                (iname,
                    evaluate(iname_to_stride[iname], approximate_arg_values))
                for iname in iname_to_stride),
                return_value=True)

        if least_stride == 1:
            vote_strength = 1
        else:
            vote_strength = 0.5

        vote_count_for_l0[least_stride_iname] = (
                vote_count_for_l0.get(least_stride_iname, 0)
                + vote_strength)

    return argmax2(vote_count_for_l0.iteritems())

    # }}}

# }}}

# {{{ assign automatic axes

def assign_automatic_axes(kernel, only_axis_0=True):
    from loopy.kernel import (AutoLocalIndexTagBase, LocalIndexTag,
            UnrollTag)

    global_size, local_size = kernel.get_grid_sizes_as_exprs(
            ignore_auto=True)

    def assign_axis(iname, axis=None):
        desired_length = kernel.get_constant_iname_length(iname)

        if axis is None:
            # {{{ find a suitable axis

            # find already assigned local axes (to avoid them)
            shorter_possible_axes = []
            test_axis = 0
            while True:
                if test_axis >= len(local_size):
                    break
                if test_axis in assigned_local_axes:
                    test_axis += 1
                    continue

                if local_size[test_axis] < desired_length:
                    shorter_possible_axes.append(test_axis)
                    test_axis += 1
                    continue
                else:
                    axis = test_axis
                    break

            # longest first
            shorter_possible_axes.sort(key=lambda ax: local_size[ax])

            if axis is None and shorter_possible_axes:
                axis = shorter_possible_axes[0]

            # }}}

        if axis is None:
            new_tag = None
        else:
            new_tag = LocalIndexTag(axis)
            if desired_length > local_size[axis]:
                from loopy import split_dimension
                return assign_automatic_axes(
                        split_dimension(kernel, iname, inner_length=local_size[axis],
                            outer_tag=UnrollTag(), inner_tag=new_tag),
                        only_axis_0=only_axis_0)

        new_iname_to_tag = kernel.iname_to_tag.copy()
        new_iname_to_tag[iname] = new_tag
        return assign_automatic_axes(kernel.copy(iname_to_tag=new_iname_to_tag),
                only_axis_0=only_axis_0)

    for insn in kernel.instructions:
        auto_axis_inames = [
                iname
                for iname in insn.all_inames()
                if isinstance(kernel.iname_to_tag.get(iname),
                    AutoLocalIndexTagBase)]

        if not auto_axis_inames:
            continue

        assigned_local_axes = set()

        for iname in insn.all_inames():
            tag = kernel.iname_to_tag.get(iname)
            if isinstance(tag, LocalIndexTag):
                assigned_local_axes.add(tag.axis)

        axis0_iname = guess_good_iname_for_axis_0(kernel, insn)

        axis0_iname_tag = kernel.iname_to_tag.get(axis0_iname)
        ax0_tag = LocalIndexTag(0)
        if (isinstance(axis0_iname_tag, AutoLocalIndexTagBase)
                and 0 not in assigned_local_axes):
            return assign_axis(axis0_iname, 0)

        if only_axis_0:
            continue

        # assign longest auto axis inames first
        auto_axis_inames.sort(key=kernel.get_constant_iname_length, reverse=True)

        next_axis = 0
        if auto_axis_inames:
            return assign_axis(auto_axis_inames.pop())

    # We've seen all instructions and not punted to recursion/restart because
    # of a new axis assignment.

    if only_axis_0:
        # If we were only assigining axis 0, then assign all the remaining 
        # axes next.
        return assign_automatic_axes(kernel, only_axis_0=False)
    else:
        # If were already assigning all axes and got here, we're now done.
        # All automatic axes are assigned.
        return kernel

# }}}

# {{{ scheduling algorithm

def generate_loop_schedules_internal(kernel, loop_priority, schedule=[]):
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

        if insn.idempotent == True:
            # If insn is idempotent, it may be placed inside a more deeply
            # nested loop without harm.

            iname_deps_satisfied = (
                    insn.all_inames() - parallel_inames
                    <=
                    active_inames - parallel_inames)

        elif insn.idempotent == False:
            # If insn is not idempotent, we must insist that it is placed inside
            # the exactly correct set of loops.

            iname_deps_satisfied = (
                    insn.all_inames() - parallel_inames
                    ==
                    active_inames - parallel_inames)

        else:
            raise RuntimeError("instruction '%s' has undetermined idempotence"
                    % insn.id)

        if (iname_deps_satisfied
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
        useful_loops = []

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

            useful_loops.append(iname)

            # }}}

        useful_and_desired = set(useful_loops) & set(loop_priority)
        if useful_and_desired:
            # restrict to the first ('highest-priority') loop that's useful

            for iname in loop_priority:
                if iname in useful_and_desired:
                    useful_loops = [iname]
                    break

        for iname in useful_loops:
            new_schedule = schedule + [EnterLoop(iname=iname)]
            for sub_sched in generate_loop_schedules_internal(
                    kernel, loop_priority, new_schedule):
                yield sub_sched

        if useful_loops:
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
            for sub_sched in generate_loop_schedules_internal(
                    kernel, loop_priority, schedule):
                yield sub_sched

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

            # (i.e. if anything *in* the loop depends on something beforehand)

            for insn_id in owed_barriers:
                if has_dependent_in_schedule(kernel, insn_id, subloop):
                    issue_barrier(is_pre_barrier=False)
                    break

            # }}}

            subresult, sub_owed_barriers = insert_barriers(
                    kernel, subloop[1:-1], level+1)

            # {{{ issue pre-barriers for contents of nested loop

            if not loop_had_barrier[0]:
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

            if set(insn.insn_deps) & owed_barriers:
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

# }}}

# {{{ main scheduling entrypoint

def generate_loop_schedules(kernel, loop_priority=[]):
    kernel = realize_reduction(kernel)

    # {{{ check that all CSEs have been realized

    from loopy.symbolic import CSECallbackMapper

    def map_cse(expr, rec):
        raise RuntimeError("all CSEs must be realized before scheduling")

    for insn in kernel.instructions:
        CSECallbackMapper(map_cse)(insn.expression)

    # }}}

    kernel = assign_automatic_axes(kernel)
    kernel = add_idempotence_and_automatic_dependencies(kernel)
    kernel = adjust_local_temp_var_storage(kernel)

    check_for_double_use_of_hw_axes(kernel)
    check_for_unused_hw_axes(kernel)

    schedule_count = 0

    for gen_sched in generate_loop_schedules_internal(kernel, loop_priority):
        gen_sched, owed_barriers = insert_barriers(kernel, gen_sched)
        assert not owed_barriers

        yield kernel.copy(schedule=gen_sched)

        schedule_count += 1

    if not schedule_count:
        raise RuntimeError("no valid schedules found")

# }}}





# vim: foldmethod=marker

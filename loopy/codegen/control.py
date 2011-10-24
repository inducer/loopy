"""Loop nest build top-level control/hoisting."""
from __future__ import division

from loopy.codegen import CodeGenerationState, gen_code_block
import islpy as isl




def get_admissible_conditional_inames_for(kernel, sched_index):
    """This function disallows conditionals on local-idx tagged
    inames if there is a barrier nested somewhere within.
    """

    from loopy.kernel import LocalIndexTag, HardwareParallelTag

    from loopy.schedule import find_active_inames_at, has_barrier_within
    result = find_active_inames_at(kernel, sched_index)

    has_barrier = has_barrier_within(kernel, sched_index)

    for iname, tag in kernel.iname_to_tag.iteritems():
        if isinstance(tag, HardwareParallelTag):
            if not has_barrier or not isinstance(tag, LocalIndexTag):
                result.add(iname)

    return result




def generate_code_for_sched_index(kernel, sched_index, codegen_state):
    from loopy.schedule import (EnterLoop, RunInstruction, Barrier)

    sched_item = kernel.schedule[sched_index]

    if isinstance(sched_item, EnterLoop):
        tag = kernel.iname_to_tag.get(sched_item.iname)

        from loopy.codegen.loop import (
                generate_unroll_loop,
                generate_sequential_loop_dim_code)

        from loopy.kernel import UnrollTag, SequentialTag
        if isinstance(tag, UnrollTag):
            func = generate_unroll_loop
        elif tag is None or isinstance(tag, SequentialTag):
            func = generate_sequential_loop_dim_code
        else:
            raise RuntimeError("encountered (invalid) EnterLoop for '%s', tagged '%s'"
                    % (sched_item.iname, tag))

        return func(kernel, sched_index, codegen_state)

    elif isinstance(sched_item, Barrier):
        from loopy.codegen import GeneratedInstruction
        from cgen import Statement as S
        return GeneratedInstruction(
                ast=S("barrier(CLK_LOCAL_MEM_FENCE)"),
                implemented_domain=None)

    elif isinstance(sched_item, RunInstruction):
        insn = kernel.id_to_insn[sched_item.insn_id]

        from loopy.codegen.instruction import generate_instruction_code
        return generate_instruction_code(kernel, insn, codegen_state)

    else:
        raise RuntimeError("unexpected schedule item type: %s"
                % type(sched_item))




def remove_inames_for_shared_hw_axes(kernel, cond_inames):
    """
    See if cond_inames contains references to two (or more) inames that
    boil down to the same tag. If so, exclude them. (We shouldn't be writing
    conditionals for such inames because we would be implicitly restricting
    the other inames as well.)
    """

    tag_key_use_count = {}

    from loopy.kernel import HardwareParallelTag

    for iname in cond_inames:
        tag = kernel.iname_to_tag.get(iname)

        if isinstance(tag, HardwareParallelTag):
            tag_key_use_count[tag.key] = tag_key_use_count.get(tag.key, 0) + 1

    multi_use_keys = set(
            key for key, count in tag_key_use_count.iteritems()
            if count > 1)

    multi_use_inames = set()
    for iname in cond_inames:
        tag = kernel.iname_to_tag.get(iname)
        if isinstance(tag, HardwareParallelTag) and tag.key in multi_use_keys:
            multi_use_inames.add(iname)

    return cond_inames - multi_use_inames




def build_loop_nest(kernel, sched_index, codegen_state):
    # Most of the complexity of this function goes towards finding groups of
    # instructions that can be nested inside a shared conditional.

    assert isinstance(codegen_state, CodeGenerationState)

    from loopy.schedule import (EnterLoop, LeaveLoop, RunInstruction, Barrier,
            gather_schedule_subloop)

    # {{{ pass 1: pre-scan schedule for my schedule items' indices

    my_sched_indices = []

    while sched_index < len(kernel.schedule):
        sched_item = kernel.schedule[sched_index]

        if isinstance(sched_item, LeaveLoop):
            break

        my_sched_indices.append(sched_index)

        if isinstance(sched_item, EnterLoop):
            _, sched_index = gather_schedule_subloop(
                    kernel.schedule, sched_index)
        elif isinstance(sched_item, Barrier):
            sched_index += 1

        elif isinstance(sched_item, RunInstruction):
            sched_index += 1
        else:
            raise RuntimeError("unexpected schedule item type: %s"
                    % type(sched_item))

    # }}}

    # {{{ pass 2: find admissible conditional inames for each schedule item

    admissible_cond_inames = [
            get_admissible_conditional_inames_for(kernel, sched_index)
            for sched_index in my_sched_indices]

    # }}}

    # {{{ pass 3: greedily group schedule items that share admissible inames

    def build_insn_group(sched_indices_and_cond_inames, codegen_state,
            min_iname_count=1):
        # min_iname_count serves to prevent infinite recursion by imposing a
        # bigger and bigger minimum size on the group of shared inames found.

        if not sched_indices_and_cond_inames:
            return []

        sched_index, cond_inames = sched_indices_and_cond_inames[0]

        # {{{ grow schedule item group

        # Keep growing schedule item group as long as group fulfills minimum
        # size requirement.

        current_iname_set = cond_inames

        idx = 1
        while (len(current_iname_set) >= min_iname_count
                and idx < len(sched_indices_and_cond_inames)):
            other_sched_index, other_cond_inames = sched_indices_and_cond_inames[idx]
            new_iname_set = current_iname_set & other_cond_inames

            if len(new_iname_set) >= min_iname_count:
                idx += 1
                current_iname_set = new_iname_set
            else:
                break

        # }}}

        if len(current_iname_set) >= min_iname_count:
            # Success: found a big enough group of inames for a conditional.
            # See if there are bounds checks available for that set.

            # {{{ see which inames were actually used in group

            # And only generate conditionals for those.
            from loopy.schedule import find_used_inames_within
            used_inames = set()
            for subsched_index, _ in sched_indices_and_cond_inames[0:idx]:
                used_inames |= find_used_inames_within(kernel, subsched_index)

            # }}}

            from loopy.codegen.bounds import generate_bounds_checks
            bounds_checks = generate_bounds_checks(kernel.domain,
                    remove_inames_for_shared_hw_axes(kernel,
                        current_iname_set & used_inames),
                    codegen_state.implemented_domain)
        else:
            bounds_checks = []

        if bounds_checks:
            check_set = isl.BasicSet.universe(kernel.space)
            for cns in bounds_checks:
                check_set = check_set.add_constraint(cns)

            new_codegen_state = codegen_state.intersect(check_set)
        else:
            new_codegen_state = codegen_state

        if idx == 1:
            # group only contains starting schedule item
            result = [generate_code_for_sched_index(kernel, sched_index, new_codegen_state)]
        else:
            # recurse with a bigger minimum iname count
            result = build_insn_group(sched_indices_and_cond_inames[0:idx],
                    new_codegen_state, len(current_iname_set)+1)

        if bounds_checks:
            from loopy.codegen import wrap_in_if
            from loopy.codegen.bounds import constraint_to_code
            result = [wrap_in_if(
                    [constraint_to_code(codegen_state.c_code_mapper, cns) for cns in bounds_checks],
                    gen_code_block(result))]

        return result + build_insn_group(
                sched_indices_and_cond_inames[idx:], codegen_state)

    # }}}

    return gen_code_block(
            build_insn_group(zip(
                my_sched_indices, admissible_cond_inames), codegen_state))




# vim: foldmethod=marker

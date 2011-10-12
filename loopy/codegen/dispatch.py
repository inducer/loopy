"""Loop nest build top-level dispatch."""
from __future__ import division

from loopy.codegen import CodeGenerationState, gen_code_block




def build_loop_nest(kernel, sched_index, codegen_state):
    assert isinstance(codegen_state, CodeGenerationState)

    from loopy.schedule import (EnterLoop, LeaveLoop, RunInstruction, Barrier,
            gather_schedule_subloop)
    from cgen import Statement as S

    result = []

    while sched_index < len(kernel.schedule):
        sched_item = kernel.schedule[sched_index]

        if isinstance(sched_item, LeaveLoop):
            break

        elif isinstance(sched_item, EnterLoop):
            tag = kernel.iname_to_tag[sched_item.iname]

            from loopy.codegen.loop import (
                    generate_unroll_or_ilp_code,
                    generate_parallel_loop_dim_code,
                    generate_sequential_loop_dim_code)

            from loopy.kernel import (TAG_UNROLL, TAG_ILP,
                    ParallelTagWithAxis)
            if isinstance(tag, (TAG_UNROLL, TAG_ILP)):
                func = generate_unroll_or_ilp_code
            elif isinstance(tag, ParallelTagWithAxis):
                func = generate_parallel_loop_dim_code
            else:
                func = generate_sequential_loop_dim_code

            result.append(func(kernel, sched_index, codegen_state))

            _, sched_index = gather_schedule_subloop(
                    kernel.schedule, sched_index)

        elif isinstance(sched_item, Barrier):
            result.append(S("barrier(CLK_LOCAL_MEM_FENCE)"))

            sched_index += 1

        elif isinstance(sched_item, RunInstruction):
            insn = kernel.id_to_insn[sched_item.insn_id]

            from loopy.codegen.instruction import generate_instruction_code

            result.append(
                    generate_instruction_code(kernel, insn, codegen_state))

            sched_index += 1

        else:
            raise RuntimeError("unexpected schedule item type: %s"
                    % type(sched_item))


    return gen_code_block(result)




# vim: foldmethod=marker

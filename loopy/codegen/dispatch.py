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




def build_loop_nest_old(kernel, sched_index, codegen_state, no_conditional_check=False):
    assert isinstance(exec_domain, ExecutionDomain)

    ccm = exec_domain.c_code_mapper

    from cgen import (POD, Initializer, Assign, Statement as S,
            Line)

    from loopy.codegen.bounds import (
            generate_bounds_checks,
            generate_bounds_checks_code,
            get_valid_check_vars,
            constraint_to_code)

    if not no_conditional_check:
        # {{{ see if there are any applicable conditionals

        applicable_constraints = generate_bounds_checks(
                kernel.domain,
                get_valid_check_vars(kernel, sched_index, allow_ilp=False),
                exec_domain.implemented_domain)

        if applicable_constraints:
            import islpy as isl
            exec_domain_restriction = isl.Set.universe(kernel.space)
            for cns in applicable_constraints:
                exec_domain_restriction = (exec_domain_restriction
                        .add_constraint(cns))

            exec_domain = exec_domain.intersect(exec_domain_restriction)

            inner = build_loop_nest(kernel, sched_index, exec_domain,
                    no_conditional_check=True)

            from loopy.codegen import wrap_in_if
            return wrap_in_if([
                constraint_to_code(ccm, cns)
                for cns in applicable_constraints],
                inner)

        # }}}

    if sched_index >= len(kernel.schedule):
        # {{{ write innermost loop body

        from pymbolic.primitives import Subscript

        # FIXME revert to unroll if actual bounds checks are needed?

        valid_index_vars = get_valid_check_vars(kernel, sched_index, allow_ilp=True)
        bounds_check_lists = [
                generate_bounds_checks_code(subd.c_code_mapper, kernel.domain,
                    valid_index_vars, subd.implemented_domain)
                for subd in exec_domain.subdomains]

        result = []
        for lvalue, expr in kernel.instructions:
            for i, subd in enumerate(exec_domain.subdomains):
                assert isinstance(lvalue, Subscript)
                name = lvalue.aggregate.name

                from loopy.codegen import wrap_in_if
                result.append(wrap_in_if(
                            bounds_check_lists[i],
                            S("tmp_%s_%d += %s"
                                % (name, i, subd.c_code_mapper(expr)))))

        return gen_code_block(result)

        # }}}

    sched_item = kernel.schedule[sched_index]

    from loopy.schedule import ScheduledLoop, WriteOutput
    from loopy.prefetch import LocalMemoryPrefetch, RegisterPrefetch
    from loopy.codegen.bounds import wrap_in_bounds_checks

    if isinstance(sched_item, ScheduledLoop):
        from loopy.codegen.loop import (
                generate_unroll_or_ilp_code,
                generate_parallel_loop_dim_code,
                generate_sequential_loop_dim_code)
        from loopy.kernel import (TAG_UNROLL, TAG_ILP,
                ParallelTagWithAxis)

        tag = kernel.iname_to_tag.get(sched_item.iname)

        if isinstance(tag, (TAG_UNROLL, TAG_ILP)):
            func = generate_unroll_or_ilp_code
        elif isinstance(tag, ParallelTagWithAxis):
            func = generate_parallel_loop_dim_code
        else:
            func = generate_sequential_loop_dim_code

        return func(kernel, sched_index, exec_domain)

    elif isinstance(sched_item, WriteOutput):
        result = (
                [Initializer(POD(kernel.arg_dict[lvalue.aggregate.name].dtype,
                    "tmp_%s_%d" % (lvalue.aggregate.name, i)), 0)
                    for i in range(len(exec_domain.subdomains))
                    for lvalue, expr in kernel.instructions]
                +[Line()]
                +[build_loop_nest(kernel, sched_index+1, 
                    exec_domain)]
                +[Line()])

        for i, subd in enumerate(exec_domain.subdomains):
            for lvalue, expr in kernel.instructions:
                assignment = Assign(subd.c_code_mapper(lvalue), "tmp_%s_%d" % (
                    lvalue.aggregate.name, i))

                wrapped_assign = wrap_in_bounds_checks(
                        subd.c_code_mapper, kernel.domain,
                        get_valid_check_vars(kernel, sched_index, allow_ilp=True),
                        subd.implemented_domain, assignment)

                result.append(wrapped_assign)

        return gen_code_block(result)

    elif isinstance(sched_item, LocalMemoryPrefetch):
        from loopy.codegen.prefetch import generate_prefetch_code
        return generate_prefetch_code(kernel, sched_index, 
                exec_domain)

    elif isinstance(sched_item, RegisterPrefetch):
        raise NotImplementedError("reg prefetch") # FIXME

        agg_name = sched_item.subscript_expr.aggregate.name
        return gen_code_block([
            wrap_in_bounds_checks(ccm, kernel, sched_index, implemented_domain,
                Initializer(POD(kernel.arg_dict[agg_name].dtype,
                    sched_item.new_name),
                    "%s[%s]"
                    % (agg_name,
                        ccm(sched_item.subscript_expr.index)))),

            build_loop_nest(kernel, sched_index+1, exec_domain)
            ])

    else:
        raise ValueError("invalid schedule item encountered")





# vim: foldmethod=marker

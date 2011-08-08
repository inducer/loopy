"""Loop nest build top-level dispatch."""
from __future__ import division

from loopy.codegen import ExecutionDomain, gen_code_block




def build_loop_nest(cgs, kernel, sched_index, exec_domain, no_conditional_check=False):
    assert isinstance(exec_domain, ExecutionDomain)

    ccm = cgs.c_code_mapper

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

            inner = build_loop_nest(cgs, kernel, sched_index, exec_domain,
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
                generate_bounds_checks_code(ccm, kernel.domain,
                    valid_index_vars, impl_domain)
                for assignments, impl_domain in
                    exec_domain]

        result = []
        for lvalue, expr in kernel.instructions:
            for i, (assignments, impl_domain) in \
                    enumerate(exec_domain):

                my_block = []
                if assignments:
                    my_block.extend(assignments+[Line()])

                assert isinstance(lvalue, Subscript)
                name = lvalue.aggregate.name

                from loopy.codegen import wrap_in_if
                my_block.append(
                        wrap_in_if(
                            bounds_check_lists[i],
                            S("tmp_%s_%d += %s"
                                % (name, i, ccm(expr)))))
                result.append(gen_code_block(my_block))

        return gen_code_block(result)

        # }}}

    sched_item = kernel.schedule[sched_index]

    from loopy.schedule import ScheduledLoop, WriteOutput
    from loopy.prefetch import LocalMemoryPrefetch, RegisterPrefetch
    from loopy.codegen.bounds import wrap_in_bounds_checks

    if isinstance(sched_item, ScheduledLoop):
        from loopy.codegen.loop_dim import (
                generate_unroll_or_ilp_code,
                generate_parallel_loop_dim_code,
                generate_sequential_loop_dim_code)
        from loopy.kernel import (BaseUnrollTag, TAG_ILP,
                ParallelTagWithAxis)

        tag = kernel.iname_to_tag.get(sched_item.iname)

        if isinstance(tag, (BaseUnrollTag, TAG_ILP)):
            func = generate_unroll_or_ilp_code
        elif isinstance(tag, ParallelTagWithAxis):
            func = generate_parallel_loop_dim_code
        else:
            func = generate_sequential_loop_dim_code

        return func(cgs, kernel, sched_index, exec_domain)

    elif isinstance(sched_item, WriteOutput):
        result = (
                [Initializer(POD(kernel.arg_dict[lvalue.aggregate.name].dtype,
                    "tmp_%s_%d" % (lvalue.aggregate.name, i)), 0)
                    for i in range(len(exec_domain))
                    for lvalue, expr in kernel.instructions]
                +[Line()]
                +[build_loop_nest(cgs, kernel, sched_index+1, 
                    exec_domain)]
                +[Line()])

        for i, (idx_assignments, impl_domain) in \
                enumerate(exec_domain):
            for lvalue, expr in kernel.instructions:
                assignment = Assign(ccm(lvalue), "tmp_%s_%d" % (
                    lvalue.aggregate.name, i))

                wrapped_assign = wrap_in_bounds_checks(
                        ccm, kernel.domain,
                        get_valid_check_vars(kernel, sched_index, allow_ilp=True),
                        impl_domain, assignment)

                cb = []
                if idx_assignments:
                    cb.extend(idx_assignments+[Line()])
                cb.append(wrapped_assign)

                result.append(gen_code_block(cb))

        return gen_code_block(result)

    elif isinstance(sched_item, LocalMemoryPrefetch):
        from loopy.codegen.prefetch import generate_prefetch_code
        return generate_prefetch_code(cgs, kernel, sched_index, 
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

            build_loop_nest(cgs, kernel, sched_index+1, exec_domain)
            ])

    else:
        raise ValueError("invalid schedule item encountered")





# vim: foldmethod=marker

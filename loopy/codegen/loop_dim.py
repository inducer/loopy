from __future__ import division

import numpy as np
from loopy.codegen import ExecutionDomain, gen_code_block
from pytools import Record
import islpy as isl
from loopy.codegen.dispatch import build_loop_nest





# {{{ conditional-minimizing slab decomposition

def get_slab_decomposition(cgs, kernel, sched_index, exec_domain):
    from loopy.isl import (cast_constraint_to_space,
            block_shift_constraint, negate_constraint)

    ccm = cgs.c_code_mapper
    space = kernel.space
    iname = kernel.schedule[sched_index].iname
    tag = kernel.iname_to_tag.get(iname)

    # {{{ attempt slab partition to reduce conditional count

    lb_cns_orig, ub_cns_orig = kernel.get_projected_bounds_constraints(iname)
    lb_cns_orig = cast_constraint_to_space(lb_cns_orig, space)
    ub_cns_orig = cast_constraint_to_space(ub_cns_orig, space)

    # jostle the constant in {lb,ub}_cns to see if we can get
    # fewer conditionals in the bulk middle segment

    class TrialRecord(Record):
        pass

    if (cgs.try_slab_partition
            and "outer" in iname):
        trial_cgs = cgs.copy(try_slab_partition=False)
        trials = []

        for lower_incr, upper_incr in [ (0,0), (0,-1), ]:

            lb_cns = block_shift_constraint(lb_cns_orig, iname, -lower_incr)
            ub_cns = block_shift_constraint(ub_cns_orig, iname, -upper_incr)

            bulk_slab = (isl.Set.universe(kernel.space)
                    .add_constraint(lb_cns)
                    .add_constraint(ub_cns))
            bulk_exec_domain = exec_domain.intersect(bulk_slab)
            inner = build_loop_nest(trial_cgs, kernel, sched_index+1,
                    bulk_exec_domain)

            trials.append((TrialRecord(
                lower_incr=lower_incr,
                upper_incr=upper_incr,
                bulk_slab=bulk_slab),
                (inner.num_conditionals,
                    # when all num_conditionals are equal, choose the
                    # one with the smallest bounds changes
                    abs(upper_incr)+abs(lower_incr))))

        from pytools import argmin2
        chosen = argmin2(trials)
    else:
        bulk_slab = (isl.Set.universe(kernel.space)
                .add_constraint(lb_cns_orig)
                .add_constraint(ub_cns_orig))
        chosen = TrialRecord(
                    lower_incr=0,
                    upper_incr=0,
                    bulk_slab=bulk_slab)

    # }}}

    # {{{ build slabs

    slabs = []
    if chosen.lower_incr:
        slabs.append(("initial", isl.Set.universe(kernel.space)
                .add_constraint(lb_cns_orig)
                .add_constraint(ub_cns_orig)
                .add_constraint(
                    negate_constraint(
                        block_shift_constraint(
                            lb_cns_orig, iname, -chosen.lower_incr)))))

    slabs.append(("bulk", chosen.bulk_slab))

    if chosen.upper_incr:
        slabs.append(("final", isl.Set.universe(kernel.space)
                .add_constraint(ub_cns_orig)
                .add_constraint(lb_cns_orig)
                .add_constraint(
                    negate_constraint(
                        block_shift_constraint(
                            ub_cns_orig, iname, -chosen.upper_incr)))))

    # }}}

    return lb_cns_orig, ub_cns_orig, slabs

# }}}

# {{{ unrolled/ILP loops

def generate_unroll_or_ilp_code(cgs, kernel, sched_index, exec_domain):
    from loopy.isl import (
            cast_constraint_to_space, solve_constraint_for_bound,
            block_shift_constraint)

    from cgen import (POD, Assign, Line, Statement as S, Initializer, Const)

    ccm = cgs.c_code_mapper
    space = kernel.space
    iname = kernel.schedule[sched_index].iname
    tag = kernel.iname_to_tag.get(iname)

    lower_cns, upper_cns = kernel.get_projected_bounds_constraints(iname)
    lower_cns = cast_constraint_to_space(lower_cns, space)
    upper_cns = cast_constraint_to_space(upper_cns, space)

    lower_kind, lower_bound = solve_constraint_for_bound(lower_cns, iname)
    upper_kind, upper_bound = solve_constraint_for_bound(upper_cns, iname)

    assert lower_kind == ">="
    assert upper_kind == "<"

    from pymbolic import flatten
    from pymbolic.mapper.constant_folder import CommutativeConstantFoldingMapper
    cfm = CommutativeConstantFoldingMapper()
    length = int(cfm(flatten(upper_bound-lower_bound)))

    def generate_idx_eq_slabs():
        for i in xrange(length):
            yield (i, isl.Set.universe(kernel.space)
                    .add_constraint(
                            block_shift_constraint(
                                lower_cns, iname, -i, as_equality=True)))

    from loopy.kernel import BaseUnrollTag, TAG_ILP, TAG_UNROLL_STATIC, TAG_UNROLL_INCR
    if isinstance(tag, BaseUnrollTag):
        result = [POD(np.int32, iname), Line()]

        for i, slab in generate_idx_eq_slabs():
            new_exec_domain = exec_domain.intersect(slab)
            inner = build_loop_nest(cgs, kernel, sched_index+1,
                    new_exec_domain)

            if isinstance(tag, TAG_UNROLL_STATIC):
                result.extend([
                    Assign(iname, ccm(lower_bound+i)),
                    Line(), inner])
            elif isinstance(tag, TAG_UNROLL_INCR):
                result.append(S("++%s" % iname))

        return gen_code_block(result)

    elif isinstance(tag, TAG_ILP):
        new_aaid = []
        for assignments, implemented_domain in exec_domain:
            for i, single_slab in generate_idx_eq_slabs():
                assignments = assignments + [
                        Initializer(Const(POD(np.int32, iname)), ccm(lower_bound+i))]
                new_aaid.append((assignments, 
                    implemented_domain.intersect(single_slab)))

                assignments = []

        overall_slab = (isl.Set.universe(kernel.space)
                .add_constraint(lower_cns)
                .add_constraint(upper_cns))

        return build_loop_nest(cgs, kernel, sched_index+1,
                ExecutionDomain(
                    exec_domain.implemented_domain.intersect(overall_slab),
                    new_aaid))
    else:
        assert False, "not supposed to get here"

# }}}

# {{{ parallel loop

def generate_parallel_loop_dim_code(cgs, kernel, sched_index, exec_domain):
    from loopy.isl import make_slab


    ccm = cgs.c_code_mapper
    space = kernel.space
    iname = kernel.schedule[sched_index].iname
    tag = kernel.iname_to_tag.get(iname)

    lb_cns_orig, ub_cns_orig, slabs = get_slab_decomposition(
            cgs, kernel, sched_index, exec_domain)

    # For a parallel loop dimension, the global loop bounds are
    # automatically obeyed--simply because no work items are launched
    # outside the requested grid.
    #
    # For a forced length, this is implemented by an if below.

    if tag.forced_length is None:
        exec_domain = exec_domain.intersect(
                isl.Set.universe(kernel.space)
                .add_constraint(lb_cns_orig)
                .add_constraint(ub_cns_orig))
    else:
        impl_len = tag.forced_length
        start, _ = kernel.get_projected_bounds(iname)
        exec_domain = exec_domain.intersect(
                make_slab(kernel.space, iname, start, start+impl_len))

    result = []
    nums_of_conditionals = []

    from loopy.codegen import add_comment

    for slab_name, slab in slabs:
        cmt = "%s slab for '%s'" % (slab_name, iname)
        if len(slabs) == 1:
            cmt = None

        new_kernel = kernel.copy(
                domain=kernel.domain.intersect(slab))
        result.append(
                add_comment(cmt,
                    build_loop_nest(cgs, kernel, sched_index+1,
                        exec_domain)))

    from loopy.codegen import gen_code_block
    return gen_code_block(result, is_alternatives=True)

# }}}

# {{{ sequential loop

def generate_sequential_loop_dim_code(cgs, kernel, sched_index, exec_domain):

    ccm = cgs.c_code_mapper
    space = kernel.space
    iname = kernel.schedule[sched_index].iname
    tag = kernel.iname_to_tag.get(iname)

    lb_cns_orig, ub_cns_orig, slabs = get_slab_decomposition(
            cgs, kernel, sched_index, exec_domain)

    result = []
    nums_of_conditionals = []

    for slab_name, slab in slabs:
        cmt = "%s slab for '%s'" % (slab_name, iname)
        if len(slabs) == 1:
            cmt = None

        new_exec_domain = exec_domain.intersect(slab)
        inner = build_loop_nest(cgs, kernel, sched_index+1,
                new_exec_domain)

        from loopy.codegen.bounds import wrap_in_for_from_constraints

        # regular loop
        if cmt is not None:
            from cgen import Comment
            result.append(Comment(cmt))
        result.append(
                wrap_in_for_from_constraints(ccm, iname, slab, inner))

    return gen_code_block(result)

# }}}

# vim: foldmethod=marker

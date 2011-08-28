from __future__ import division

import numpy as np
from loopy.codegen import ExecutionDomain, gen_code_block
from pytools import Record
import islpy as isl
from islpy import dim_type
from loopy.codegen.dispatch import build_loop_nest





def get_simple_loop_bounds(kernel, sched_index, iname, implemented_domain):
    from loopy.isl import cast_constraint_to_space
    from loopy.codegen.bounds import get_bounds_constraints, get_defined_vars
    lower_constraints_orig, upper_constraints_orig, equality_constraints_orig = \
            get_bounds_constraints(kernel.domain, iname,
                    frozenset([iname])
                    | frozenset(get_defined_vars(kernel, sched_index+1, allow_ilp=False)),
                    allow_parameters=True)

    assert not equality_constraints_orig
    from loopy.codegen.bounds import pick_simple_constraint
    lb_cns_orig = pick_simple_constraint(lower_constraints_orig, iname)
    ub_cns_orig = pick_simple_constraint(upper_constraints_orig, iname)

    lb_cns_orig = cast_constraint_to_space(lb_cns_orig, kernel.space)
    ub_cns_orig = cast_constraint_to_space(ub_cns_orig, kernel.space)

    return lb_cns_orig, ub_cns_orig

# {{{ conditional-minimizing slab decomposition

def get_slab_decomposition(kernel, sched_index, exec_domain):
    from loopy.isl import block_shift_constraint, negate_constraint

    ccm = exec_domain.c_code_mapper
    space = kernel.space
    iname = kernel.schedule[sched_index].iname
    tag = kernel.iname_to_tag.get(iname)

    lb_cns_orig, ub_cns_orig = get_simple_loop_bounds(kernel, sched_index, iname,
            exec_domain.implemented_domain)

    lower_incr, upper_incr = kernel.iname_slab_increments.get(iname, (0, 0))

    # {{{ build slabs

    slabs = []
    if lower_incr:
        slabs.append(("initial", isl.Set.universe(kernel.space)
                .add_constraint(lb_cns_orig)
                .add_constraint(ub_cns_orig)
                .add_constraint(
                    negate_constraint(
                        block_shift_constraint(
                            lb_cns_orig, iname, -lower_incr)))))

    slabs.append(("bulk", 
        (isl.Set.universe(kernel.space)
            .add_constraint(
                block_shift_constraint(lb_cns_orig, iname, -lower_incr))
            .add_constraint(
                block_shift_constraint(ub_cns_orig, iname, -upper_incr)))))

    if upper_incr:
        slabs.append(("final", isl.Set.universe(kernel.space)
                .add_constraint(ub_cns_orig)
                .add_constraint(lb_cns_orig)
                .add_constraint(
                    negate_constraint(
                        block_shift_constraint(
                            ub_cns_orig, iname, -upper_incr)))))

    # }}}

    return lb_cns_orig, ub_cns_orig, slabs

# }}}

# {{{ unrolled/ILP loops

def generate_unroll_or_ilp_code(kernel, sched_index, exec_domain):
    from loopy.isl import block_shift_constraint
    from loopy.codegen.bounds import solve_constraint_for_bound

    from cgen import (POD, Assign, Line, Statement as S, Initializer, Const)

    ccm = exec_domain.c_code_mapper
    space = kernel.space
    iname = kernel.schedule[sched_index].iname
    tag = kernel.iname_to_tag.get(iname)

    lower_cns, upper_cns = get_simple_loop_bounds(kernel, sched_index, iname,
            exec_domain.implemented_domain)

    lower_kind, lower_bound = solve_constraint_for_bound(lower_cns, iname)
    upper_kind, upper_bound = solve_constraint_for_bound(upper_cns, iname)

    assert lower_kind == ">="
    assert upper_kind == "<"

    proj_domain = (kernel.domain
            .project_out_except([iname], [dim_type.set])
            .project_out_except([], [dim_type.param])
            .remove_divs())
    assert proj_domain.is_bounded()
    success, length = proj_domain.count()
    assert success == 0

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
            inner = build_loop_nest(kernel, sched_index+1, new_exec_domain)

            if isinstance(tag, TAG_UNROLL_STATIC):
                result.extend([
                    Assign(iname, ccm(lower_bound+i)),
                    Line(), inner])
            elif isinstance(tag, TAG_UNROLL_INCR):
                result.append(S("++%s" % iname))

        return gen_code_block(result)

    elif isinstance(tag, TAG_ILP):
        new_subdomains = []
        for subd in exec_domain.subdomains:
            for i, single_slab in generate_idx_eq_slabs():
                from loopy.codegen import ExecutionSubdomain
                new_subdomains.append(
                        ExecutionSubdomain(
                            subd.implemented_domain.intersect(single_slab),
                            subd.c_code_mapper.copy_and_assign(
                                iname, lower_bound+i)))

        overall_slab = (isl.Set.universe(kernel.space)
                .add_constraint(lower_cns)
                .add_constraint(upper_cns))

        return build_loop_nest(kernel, sched_index+1,
                ExecutionDomain(
                    exec_domain.implemented_domain.intersect(overall_slab),
                    exec_domain.c_code_mapper,
                    new_subdomains))
    else:
        assert False, "not supposed to get here"

# }}}

# {{{ parallel loop

def generate_parallel_loop_dim_code(kernel, sched_index, exec_domain):
    from loopy.isl import make_slab

    ccm = exec_domain.c_code_mapper
    space = kernel.space
    iname = kernel.schedule[sched_index].iname
    tag = kernel.iname_to_tag.get(iname)

    lb_cns_orig, ub_cns_orig, slabs = get_slab_decomposition(
            kernel, sched_index, exec_domain)

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
        start, _, _ = kernel.get_bounds(iname, (iname,), allow_parameters=True)
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
                    build_loop_nest(new_kernel, sched_index+1, exec_domain)))

    from loopy.codegen import gen_code_block
    return gen_code_block(result, is_alternatives=True)

# }}}

# {{{ sequential loop

def generate_sequential_loop_dim_code(kernel, sched_index, exec_domain):

    ccm = exec_domain.c_code_mapper
    space = kernel.space
    iname = kernel.schedule[sched_index].iname
    tag = kernel.iname_to_tag.get(iname)

    lb_cns_orig, ub_cns_orig, slabs = get_slab_decomposition(
            kernel, sched_index, exec_domain)

    result = []
    nums_of_conditionals = []

    for slab_name, slab in slabs:
        cmt = "%s slab for '%s'" % (slab_name, iname)
        if len(slabs) == 1:
            cmt = None

        new_exec_domain = exec_domain.intersect(slab)
        inner = build_loop_nest(kernel, sched_index+1,
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

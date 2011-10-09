from __future__ import division

import numpy as np
from loopy.codegen import CodeGenerationState, gen_code_block
from pytools import Record
import islpy as isl
from islpy import dim_type
from loopy.codegen.dispatch import build_loop_nest





def get_simple_loop_bounds(kernel, sched_index, iname, implemented_domain):
    from loopy.isl import cast_constraint_to_space
    from loopy.codegen.bounds import get_bounds_constraints, get_defined_inames
    lower_constraints_orig, upper_constraints_orig, equality_constraints_orig = \
            get_bounds_constraints(kernel.domain, iname,
                    frozenset([iname])
                    | frozenset(get_defined_inames(kernel, sched_index+1, allow_ilp=False)),
                    allow_parameters=True)

    assert not equality_constraints_orig
    from loopy.codegen.bounds import pick_simple_constraint
    lb_cns_orig = pick_simple_constraint(lower_constraints_orig, iname)
    ub_cns_orig = pick_simple_constraint(upper_constraints_orig, iname)

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

    iname_tp, iname_idx = kernel.iname_to_dim[iname]

    slabs = []
    if lower_incr:
        slabs.append(("initial", isl.Set.universe(kernel.space)
                .add_constraint(lb_cns_orig)
                .add_constraint(ub_cns_orig)
                .add_constraint(
                    negate_constraint(
                        block_shift_constraint(
                            lb_cns_orig, iname_tp, iname_idx, -lower_incr)))))

    slabs.append(("bulk",
        (isl.Set.universe(kernel.space)
            .add_constraint(
                block_shift_constraint(lb_cns_orig, iname_tp, iname_idx, -lower_incr))
            .add_constraint(
                block_shift_constraint(ub_cns_orig, iname_tp, iname_idx, -upper_incr)))))

    if upper_incr:
        slabs.append(("final", isl.Set.universe(kernel.space)
                .add_constraint(ub_cns_orig)
                .add_constraint(lb_cns_orig)
                .add_constraint(
                    negate_constraint(
                        block_shift_constraint(
                            ub_cns_orig, iname_tp, iname_idx, -upper_incr)))))

    # }}}

    return lb_cns_orig, ub_cns_orig, slabs

# }}}

# {{{ unrolled/ILP loops

def generate_unroll_or_ilp_code(kernel, sched_index, codegen_state):
    from loopy.isl import block_shift_constraint
    from loopy.codegen.bounds import solve_constraint_for_bound

    from cgen import (POD, Assign, Line, Statement as S, Initializer, Const)

    ccm = codegen_state.c_code_mapper
    space = kernel.space
    iname = kernel.schedule[sched_index].iname
    tag = kernel.iname_to_tag.get(iname)

    lower_cns, upper_cns = get_simple_loop_bounds(kernel, sched_index, iname,
            codegen_state.implemented_domain)

    lower_kind, lower_bound = solve_constraint_for_bound(lower_cns, iname)
    upper_kind, upper_bound = solve_constraint_for_bound(upper_cns, iname)

    assert lower_kind == ">="
    assert upper_kind == "<"

    bounds = kernel.get_iname_bounds(iname)
    from loopy.isl import static_max_of_pw_aff
    from loopy.symbolic import pw_aff_to_expr

    length = int(pw_aff_to_expr(static_max_of_pw_aff(bounds.length)))
    lower_bound_pw_aff_pieces = bounds.lower_bound_pw_aff.coalesce().get_pieces()

    if len(lower_bound_pw_aff_pieces) > 1:
        raise NotImplementedError("lower bound for ILP/unroll needed conditional")

    (_, lower_bound_aff), = lower_bound_pw_aff_pieces

    def generate_idx_eq_slabs():
        for i in xrange(length):
            yield (i, isl.Set.universe(kernel.space)
                    .add_constraint(
                            block_shift_constraint(
                                lower_cns, iname, -i, as_equality=True)))

    from loopy.kernel import TAG_ILP, TAG_UNROLL
    if isinstance(tag, TAG_UNROLL):
        result = [POD(np.int32, iname), Line()]

        for i in range(length):
            idx_aff = lower_bound_aff + i
            new_codegen_state = codegen_state.fix(iname, idx_aff)
            result.append(
                    build_loop_nest(kernel, sched_index+1, new_codegen_state))

        return gen_code_block(result)

    elif isinstance(tag, TAG_ILP):
        new_ilp_instances = []
        for ilpi in codegen_state.ilp_instances:
            for i in range(length):
                idx_aff = lower_bound_aff + i
                new_ilp_instances.append(ilpi.fix(iname, idx_aff))

        overall_slab = (isl.Set.universe(kernel.space)
                .add_constraint(lower_cns)
                .add_constraint(upper_cns))

        return build_loop_nest(kernel, sched_index+1,
                CodeGenerationState(
                    codegen_state.implemented_domain.intersect(overall_slab),
                    codegen_state.c_code_mapper,
                    new_ilp_instances))

    else:
        raise RuntimeError("unexpected tag")

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

from __future__ import division

import numpy as np
from loopy.codegen import CodeGenerationState, gen_code_block
from pytools import Record
import islpy as isl
from islpy import dim_type
from loopy.codegen.dispatch import build_loop_nest





def get_simple_loop_bounds(kernel, sched_index, iname, implemented_domain):
    from loopy.codegen.bounds import get_bounds_constraints, get_defined_inames
    lower_constraints_orig, upper_constraints_orig, equality_constraints_orig = \
            get_bounds_constraints(kernel.domain, iname,
                    frozenset([iname])
                    | frozenset(get_defined_inames(kernel, sched_index+1)),
                    allow_parameters=True)

    assert not equality_constraints_orig
    from loopy.codegen.bounds import pick_simple_constraint
    lb_cns_orig = pick_simple_constraint(lower_constraints_orig, iname)
    ub_cns_orig = pick_simple_constraint(upper_constraints_orig, iname)

    return lb_cns_orig, ub_cns_orig




# {{{ conditional-minimizing slab decomposition

def get_slab_decomposition(kernel, iname, sched_index, codegen_state):
    from loopy.isl_helpers import block_shift_constraint, negate_constraint

    ccm = codegen_state.c_code_mapper
    space = kernel.space
    tag = kernel.iname_to_tag.get(iname)

    lb_cns_orig, ub_cns_orig = get_simple_loop_bounds(kernel, sched_index, iname,
            codegen_state.implemented_domain)

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

def generate_unroll_loop(kernel, sched_index, codegen_state):
    from loopy.isl_helpers import block_shift_constraint

    from cgen import (POD, Line)

    ccm = codegen_state.c_code_mapper
    space = kernel.space
    iname = kernel.schedule[sched_index].iname
    tag = kernel.iname_to_tag.get(iname)

    lower_cns, upper_cns = get_simple_loop_bounds(kernel, sched_index, iname,
            codegen_state.implemented_domain)

    bounds = kernel.get_iname_bounds(iname)
    from loopy.isl_helpers import static_max_of_pw_aff
    from loopy.symbolic import pw_aff_to_expr

    length = int(pw_aff_to_expr(static_max_of_pw_aff(bounds.length)))
    lower_bound_pw_aff_pieces = bounds.lower_bound_pw_aff.coalesce().get_pieces()

    if len(lower_bound_pw_aff_pieces) > 1:
        raise NotImplementedError("lower bound for unroll needs conditional/"
                "has more than one piece")

    (_, lower_bound_aff), = lower_bound_pw_aff_pieces

    def generate_idx_eq_slabs():
        for i in xrange(length):
            yield (i, isl.Set.universe(kernel.space)
                    .add_constraint(
                            block_shift_constraint(
                                lower_cns, iname, -i, as_equality=True)))

    from loopy.kernel import UnrollTag
    if isinstance(tag, UnrollTag):
        result = [POD(np.int32, iname), Line()]

        for i in range(length):
            idx_aff = lower_bound_aff + i
            new_codegen_state = codegen_state.fix(iname, idx_aff)
            result.append(
                    build_loop_nest(kernel, sched_index+1, new_codegen_state))

        return gen_code_block(result)

    else:
        raise RuntimeError("unexpected tag")

# }}}

# {{{ parallel loop

def set_up_hw_parallel_loops(kernel, sched_index, codegen_state, hw_inames_left=None):
    from loopy.kernel import UniqueTag, HardwareParallelTag, LocalIndexTag, GroupIndexTag

    if hw_inames_left is None:
        hw_inames_left = [iname
                for iname in kernel.all_inames()
                if isinstance(kernel.iname_to_tag.get(iname), HardwareParallelTag)]

    from loopy.codegen.dispatch import build_loop_nest
    if not hw_inames_left:
        return build_loop_nest(kernel, sched_index, codegen_state)

    global_size, local_size = kernel.get_grid_sizes()

    iname = hw_inames_left.pop()
    tag = kernel.iname_to_tag.get(iname)

    assert isinstance(tag, UniqueTag)

    other_inames_with_same_tag = [
            other_iname for other_iname in kernel.all_inames()
            if isinstance(kernel.iname_to_tag.get(other_iname), UniqueTag)
            and kernel.iname_to_tag.get(other_iname).key == tag.key
            and other_iname != iname]

    # {{{ 'implement' hardware axis boundaries

    if isinstance(tag, LocalIndexTag):
        hw_axis_size = local_size[tag.axis]
    elif isinstance(tag, GroupIndexTag):
        hw_axis_size = global_size[tag.axis]
    else:
        raise RuntimeError("unknown hardware parallel tag")

    result = []

    bounds = kernel.get_iname_bounds(iname)

    from loopy.isl_helpers import make_slab
    slab = make_slab(kernel.space, iname,
            bounds.lower_bound_pw_aff, bounds.lower_bound_pw_aff+hw_axis_size)
    codegen_state = codegen_state.intersect(slab)

    # }}}

    lb_cns_orig, ub_cns_orig, slabs = get_slab_decomposition(
            kernel, iname, sched_index, codegen_state)

    if other_inames_with_same_tag and len(slabs) > 1:
        raise RuntimeError("cannot do slab decomposition on inames that share "
                "a tag with other inames")

    ccm = codegen_state.c_code_mapper

    result = []

    from loopy.codegen import add_comment

    for slab_name, slab in slabs:
        cmt = "%s slab for '%s'" % (slab_name, iname)
        if len(slabs) == 1:
            cmt = None

        new_kernel = kernel.copy(domain=kernel.domain.intersect(slab))
        inner = set_up_hw_parallel_loops(
                new_kernel, sched_index, codegen_state, hw_inames_left)
        result.append(add_comment(cmt, inner))

    from loopy.codegen import gen_code_block
    return gen_code_block(result, is_alternatives=True)

# }}}

# {{{ sequential loop

def generate_sequential_loop_dim_code(kernel, sched_index, codegen_state):
    ccm = codegen_state.c_code_mapper
    space = kernel.space
    iname = kernel.schedule[sched_index].iname
    tag = kernel.iname_to_tag.get(iname)

    lb_cns_orig, ub_cns_orig, slabs = get_slab_decomposition(
            kernel, iname, sched_index, codegen_state)

    result = []

    for slab_name, slab in slabs:
        cmt = "%s slab for '%s'" % (slab_name, iname)
        if len(slabs) == 1:
            cmt = None

        new_codegen_state = codegen_state.intersect(slab)
        inner = build_loop_nest(kernel, sched_index+1,
                new_codegen_state)

        from loopy.codegen.bounds import wrap_in_for_from_constraints

        if cmt is not None:
            from cgen import Comment
            result.append(Comment(cmt))
        result.append(
                wrap_in_for_from_constraints(ccm, iname, slab, inner))

    return gen_code_block(result)

# }}}

# vim: foldmethod=marker

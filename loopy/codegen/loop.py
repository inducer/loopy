from __future__ import division

from loopy.codegen import gen_code_block
import islpy as isl
from loopy.codegen.control import build_loop_nest





def get_simple_loop_bounds(kernel, sched_index, iname, implemented_domain):
    from loopy.codegen.bounds import get_bounds_constraints, get_defined_inames
    lower_constraints_orig, upper_constraints_orig, equality_constraints_orig = \
            get_bounds_constraints(kernel.domain, iname,
                    frozenset([iname])
                    | frozenset(get_defined_inames(kernel, sched_index+1)),
                    allow_parameters=True)

    lower_constraints_orig.extend(equality_constraints_orig)
    upper_constraints_orig.extend(equality_constraints_orig)
    #assert not equality_constraints_orig

    from loopy.codegen.bounds import pick_simple_constraint
    lb_cns_orig = pick_simple_constraint(lower_constraints_orig, iname)
    ub_cns_orig = pick_simple_constraint(upper_constraints_orig, iname)

    return lb_cns_orig, ub_cns_orig




# {{{ conditional-minimizing slab decomposition

def get_slab_decomposition(kernel, iname, sched_index, codegen_state):
    lb_cns_orig, ub_cns_orig = get_simple_loop_bounds(kernel, sched_index, iname,
            codegen_state.implemented_domain)

    lower_incr, upper_incr = kernel.iname_slab_increments.get(iname, (0, 0))

    iname_tp, iname_idx = kernel.iname_to_dim[iname]

    if lower_incr or upper_incr:
        bounds = kernel.get_iname_bounds(iname)

        lower_bound_pw_aff_pieces = bounds.lower_bound_pw_aff.coalesce().get_pieces()
        upper_bound_pw_aff_pieces = bounds.upper_bound_pw_aff.coalesce().get_pieces()

        if len(lower_bound_pw_aff_pieces) > 1:
            raise NotImplementedError("lower bound for slab decomp of '%s' needs "
                    "conditional/has more than one piece" % iname)
        if len(upper_bound_pw_aff_pieces) > 1:
            raise NotImplementedError("upper bound for slab decomp of '%s' needs "
                    "conditional/has more than one piece" % iname)

        (_, lower_bound_aff), = lower_bound_pw_aff_pieces
        (_, upper_bound_aff), = upper_bound_pw_aff_pieces

        lower_bulk_bound = lb_cns_orig
        upper_bulk_bound = lb_cns_orig

        from loopy.isl_helpers import iname_rel_aff

        if lower_incr:
            assert lower_incr > 0
            lower_slab = ("initial", isl.Set.universe(kernel.space)
                    .add_constraint(lb_cns_orig)
                    .add_constraint(ub_cns_orig)
                    .add_constraint(
                        isl.Constraint.inequality_from_aff(
                            iname_rel_aff(kernel.space,
                                iname, "<", lower_bound_aff+lower_incr))))
            lower_bulk_bound = (
                    isl.Constraint.inequality_from_aff(
                        iname_rel_aff(kernel.space,
                            iname, ">=", lower_bound_aff+lower_incr)))
        else:
            lower_slab = None

        if upper_incr:
            assert upper_incr > 0
            upper_slab = ("final", isl.Set.universe(kernel.space)
                    .add_constraint(lb_cns_orig)
                    .add_constraint(ub_cns_orig)
                    .add_constraint(
                        isl.Constraint.inequality_from_aff(
                            iname_rel_aff(kernel.space,
                                iname, ">=", upper_bound_aff-upper_incr))))
            upper_bulk_bound = (
                    isl.Constraint.inequality_from_aff(
                        iname_rel_aff(kernel.space,
                            iname, "<", upper_bound_aff-upper_incr)))
        else:
            lower_slab = None

        slabs = []

        if lower_slab:
            slabs.append(lower_slab)
        slabs.append((
            ("bulk",
                (isl.Set.universe(kernel.space)
                    .add_constraint(lower_bulk_bound)
                    .add_constraint(upper_bulk_bound)))))
        if upper_slab:
            slabs.append(upper_slab)

        return slabs

    else:
        return [("bulk",
            (isl.Set.universe(kernel.space)
            .add_constraint(lb_cns_orig)
            .add_constraint(ub_cns_orig)))]

# }}}

# {{{ unrolled loops

def generate_unroll_loop(kernel, sched_index, codegen_state):
    iname = kernel.schedule[sched_index].iname
    tag = kernel.iname_to_tag.get(iname)

    bounds = kernel.get_iname_bounds(iname)

    from loopy.isl_helpers import (
            static_max_of_pw_aff, static_value_of_pw_aff)
    from loopy.symbolic import pw_aff_to_expr

    length = int(pw_aff_to_expr(
        static_max_of_pw_aff(bounds.size, constants_only=True)))
    lower_bound_aff = static_value_of_pw_aff(
            bounds.lower_bound_pw_aff.coalesce(),
            constants_only=False)

    from loopy.kernel import UnrollTag
    if isinstance(tag, UnrollTag):
        result = []

        for i in range(length):
            idx_aff = lower_bound_aff + i
            new_codegen_state = codegen_state.fix(iname, idx_aff, kernel.space)
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

    if not hw_inames_left:
        return build_loop_nest(kernel, sched_index, codegen_state)

    global_size, local_size = kernel.get_grid_sizes()

    hw_inames_left = hw_inames_left[:]
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
    from loopy.isl_helpers import static_value_of_pw_aff
    lower_bound = static_value_of_pw_aff(bounds.lower_bound_pw_aff,
            constants_only=False)
    slab = make_slab(kernel.space, iname,
            lower_bound, lower_bound+hw_axis_size)
    codegen_state = codegen_state.intersect(slab)

    # }}}

    slabs = get_slab_decomposition(
            kernel, iname, sched_index, codegen_state)

    if other_inames_with_same_tag and len(slabs) > 1:
        raise RuntimeError("cannot do slab decomposition on inames that share "
                "a tag with other inames")

    result = []

    from loopy.codegen import add_comment

    for slab_name, slab in slabs:
        cmt = "%s slab for '%s'" % (slab_name, iname)
        if len(slabs) == 1:
            cmt = None

        new_kernel = kernel.copy(domain=kernel.domain & slab)
        inner = set_up_hw_parallel_loops(
                new_kernel, sched_index, codegen_state, hw_inames_left)
        result.append(add_comment(cmt, inner))

    from loopy.codegen import gen_code_block
    return gen_code_block(result)

# }}}

# {{{ sequential loop

def generate_sequential_loop_dim_code(kernel, sched_index, codegen_state):
    ccm = codegen_state.c_code_mapper
    iname = kernel.schedule[sched_index].iname

    slabs = get_slab_decomposition(
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

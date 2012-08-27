"""isl helpers"""

from __future__ import division

import islpy as isl
from islpy import dim_type





def block_shift_constraint(cns, type, pos, multiple, as_equality=None):
    if as_equality != cns.is_equality():
        if as_equality:
            factory = isl.Constraint.equality_from_aff
        else:
            factory = isl.Constraint.inequality_from_aff

        cns = factory(cns.get_aff())

    cns = cns.set_constant(cns.get_constant()
            + cns.get_coefficient(type, pos)*multiple)

    return cns





def negate_constraint(cns):
    assert not cns.is_equality()
    # FIXME hackety hack
    my_set = (isl.BasicSet.universe(cns.get_space())
            .add_constraint(cns))
    my_set = my_set.complement()

    results = []
    def examine_basic_set(s):
        s.foreach_constraint(results.append)
    my_set.foreach_basic_set(examine_basic_set)
    result, = results
    return result




def make_index_map(set, index_expr):
    from loopy.symbolic import eq_constraint_from_expr

    if not isinstance(index_expr, tuple):
        index_expr = (index_expr,)

    amap = isl.Map.from_domain(set).add_dims(dim_type.out, len(index_expr))
    out_names = ["_ary_idx_%d" % i for i in range(len(index_expr))]

    dim = amap.get_space()
    all_constraints = tuple(
            eq_constraint_from_expr(dim, iexpr_i)
            for iexpr_i in index_expr)

    for i, out_name in enumerate(out_names):
        amap = amap.set_dim_name(dim_type.out, i, out_name)

    for i, (out_name, constr) in enumerate(zip(out_names, all_constraints)):
        constr.set_coefficients_by_name({out_name: -1})
        amap = amap.add_constraint(constr)

    return amap





def pw_aff_to_aff(pw_aff):
    if isinstance(pw_aff, isl.Aff):
        return pw_aff

    assert isinstance(pw_aff, isl.PwAff)
    pieces = pw_aff.get_pieces()

    if len(pieces) == 0:
        raise RuntimeError("PwAff does not have any pieces")
    if len(pieces) > 1:
        _, first_aff = pieces[0]
        for _, other_aff in pieces[1:]:
            if not first_aff.plain_is_equal(other_aff):
                raise NotImplementedError("only single-valued piecewise affine "
                        "expressions are supported here--encountered "
                        "multi-valued expression '%s'" % pw_aff)

        return first_aff

    return pieces[0][1]




def dump_space(ls):
    return " ".join("%s: %d" % (dt, ls.dim(getattr(dim_type, dt)))
            for dt in dim_type.names)




def make_slab(space, iname, start, stop):
    zero = isl.Aff.zero_on_domain(space)

    if isinstance(start, (isl.Aff, isl.PwAff)):
        start, zero = isl.align_two(pw_aff_to_aff(start), zero)
    if isinstance(stop, (isl.Aff, isl.PwAff)):
        stop, zero = isl.align_two(pw_aff_to_aff(stop), zero)

    space = zero.get_domain_space()

    from pymbolic.primitives import Expression
    from loopy.symbolic import aff_from_expr
    if isinstance(start, Expression):
        start = aff_from_expr(space, start)
    if isinstance(stop, Expression):
        stop = aff_from_expr(space, stop)

    if isinstance(start, int): start = zero + start
    if isinstance(stop, int): stop = zero + stop

    if isinstance(iname, str):
        iname_dt, iname_idx = zero.get_space().get_var_dict()[iname]
    else:
        iname_dt, iname_idx = iname

    iname_aff = zero.add_coefficient(iname_dt, iname_idx, 1)

    result = (isl.BasicSet.universe(space)
            # start <= iname
            .add_constraint(isl.Constraint.inequality_from_aff(
                iname_aff - start))
            # iname < stop
            .add_constraint(isl.Constraint.inequality_from_aff(
                stop-1 - iname_aff)))

    return result




def iname_rel_aff(space, iname, rel, aff):
    """*aff*'s domain space is allowed to not match *space*."""

    dt, pos = space.get_var_dict()[iname]
    assert dt == isl.dim_type.set

    from islpy import align_spaces
    aff = align_spaces(aff, isl.Aff.zero_on_domain(space))

    if rel in ["==", "<="]:
        return aff.add_coefficient(isl.dim_type.in_, pos, -1)
    elif rel == ">=":
        return aff.neg().add_coefficient(isl.dim_type.in_, pos, 1)
    elif rel == "<":
        return (aff-1).add_coefficient(isl.dim_type.in_, pos, -1)
    elif rel == ">":
        return (aff+1).neg().add_coefficient(isl.dim_type.in_, pos, 1)
    else:
        raise ValueError("unknown value of 'rel': %s" % rel)




def static_extremum_of_pw_aff(pw_aff, constants_only, set_method, what, context):
    pieces = pw_aff.get_pieces()
    if len(pieces) == 1:
        return pieces[0][1]

    reference = pw_aff.get_aggregate_domain()

    if context is not None:
        context = isl.align_spaces(context, pw_aff.get_domain_space())
        reference = reference.intersect(context)

    for set, candidate_aff in pieces:
        for use_gist in [False, True]:
            if use_gist:
                if context is not None:
                    candidate_aff = pw_aff.gist(set & context)
                else:
                    candidate_aff = pw_aff.gist(set)

            if constants_only and not candidate_aff.is_cst():
                continue

            if reference <= set_method(pw_aff, candidate_aff):
                return candidate_aff

    raise ValueError("a static %s was not found for PwAff '%s'"
            % (what, pw_aff))




def static_min_of_pw_aff(pw_aff, constants_only, context=None):
    return static_extremum_of_pw_aff(pw_aff, constants_only, isl.PwAff.ge_set,
            "minimum", context)

def static_max_of_pw_aff(pw_aff, constants_only, context=None):
    return static_extremum_of_pw_aff(pw_aff, constants_only, isl.PwAff.le_set,
            "maximum", context)

def static_value_of_pw_aff(pw_aff, constants_only, context=None):
    return static_extremum_of_pw_aff(pw_aff, constants_only, isl.PwAff.eq_set,
            "value", context)




def duplicate_axes(isl_obj, duplicate_inames, new_inames):
    if isinstance(isl_obj, list):
        return [
                duplicate_axes(i, duplicate_inames, new_inames)
                for i in isl_obj]

    if not duplicate_inames:
        return isl_obj

    # {{{ add dims

    start_idx = isl_obj.dim(dim_type.set)
    more_dims = isl_obj.insert_dims(
            dim_type.set, start_idx,
            len(duplicate_inames))

    for i, iname in enumerate(new_inames):
        new_idx = start_idx+i
        more_dims = more_dims.set_dim_name(
                dim_type.set, new_idx, iname)

    # }}}

    iname_to_dim = more_dims.get_space().get_var_dict()

    moved_dims = isl_obj.copy()

    for old_iname, new_iname in zip(duplicate_inames, new_inames):
        old_dt, old_idx = iname_to_dim[old_iname]
        new_dt, new_idx = iname_to_dim[new_iname]

        moved_dims = moved_dims.set_dim_name(
                old_dt, old_idx, new_iname)
        moved_dims = (moved_dims
                .move_dims(
                    dim_type.param, 0,
                    old_dt, old_idx, 1)
                .move_dims(
                    new_dt, new_idx-1,
                    dim_type.param, 0, 1))

        moved_dims = moved_dims.insert_dims(old_dt, old_idx, 1)
        moved_dims = moved_dims.set_dim_name(
                old_dt, old_idx, old_iname)

    return moved_dims.intersect(more_dims)





def is_nonnegative(expr, over_set):
    space = over_set.get_space()
    from loopy.symbolic import aff_from_expr
    try:
        aff = aff_from_expr(space, -expr-1)
    except:
        return None
    expr_neg_set = isl.BasicSet.universe(space).add_constraint(
            isl.Constraint.inequality_from_aff(aff))

    return over_set.intersect(expr_neg_set).is_empty()






# vim: foldmethod=marker

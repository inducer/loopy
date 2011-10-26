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

    from islpy import align_spaces
    if isinstance(start, isl.PwAff):
        start = align_spaces(pw_aff_to_aff(start), zero)
    if isinstance(stop, isl.PwAff):
        stop = align_spaces(pw_aff_to_aff(stop), zero)

    if isinstance(start, int): start = zero + start
    if isinstance(stop, int): stop = zero + stop

    iname_dt, iname_idx = zero.get_space().get_var_dict()[iname]
    iname_aff = zero.add_coefficient(iname_dt, iname_idx, 1)

    result = (isl.Set.universe(space)
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




def static_extremum_of_pw_aff(pw_aff, constants_only, set_method, what):
    pieces = pw_aff.get_pieces()
    if len(pieces) == 1:
        return pieces[0][1]

    agg_domain = pw_aff.get_aggregate_domain()

    for set, candidate_aff in pieces:
        if constants_only and not candidate_aff.is_cst():
            continue

        if set_method(pw_aff, candidate_aff) == agg_domain:
            return candidate_aff

    raise ValueError("a static %s was not found for PwAff '%s'"
            % (what, pw_aff))




def static_min_of_pw_aff(pw_aff, constants_only):
    return static_extremum_of_pw_aff(pw_aff, constants_only, isl.PwAff.ge_set,
            "minimum")

def static_max_of_pw_aff(pw_aff, constants_only):
    return static_extremum_of_pw_aff(pw_aff, constants_only, isl.PwAff.le_set,
            "maximum")

def static_value_of_pw_aff(pw_aff, constants_only):
    return static_extremum_of_pw_aff(pw_aff, constants_only, isl.PwAff.eq_set,
            "value")




def duplicate_axes(basic_set, duplicate_inames, new_inames):
    old_to_new = dict((old_iname, new_iname)
        for old_iname, new_iname in zip(duplicate_inames, new_inames))

    start_idx = basic_set.dim(dim_type.set)
    result = basic_set.insert_dims(
            dim_type.set, start_idx,
            len(duplicate_inames))

    old_var_dict = basic_set.get_space().get_var_dict()
    dup_iname_dims = [old_var_dict[iname] for iname in duplicate_inames]

    for i, iname in enumerate(new_inames):
        new_idx = start_idx+i
        result = result.set_dim_name(
                dim_type.set, new_idx, iname)

    set_bs, = basic_set.get_basic_sets()

    for cns in set_bs.get_constraints():
        if any(cns.involves_dims(*dim+(1,)) for dim in dup_iname_dims):
            assert not cns.is_div_constraint()
            if cns.is_equality():
                new_cns = cns.equality_alloc(result.get_space())
            else:
                new_cns = cns.inequality_alloc(result.get_space())

            new_coeffs = {}
            for key, val in cns.get_coefficients_by_name().iteritems():
                if key in old_to_new:
                    new_coeffs[old_to_new[key]] = val
                else:
                    new_coeffs[key] = val

            new_cns = new_cns.set_coefficients_by_name(new_coeffs)
            result = result.add_constraint(new_cns)

    return result




# vim: foldmethod=marker

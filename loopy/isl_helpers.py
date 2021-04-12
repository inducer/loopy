"""isl helpers"""


__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


from loopy.diagnostic import StaticValueFindingError

import islpy
import islpy.oppool as isl
from islpy import dim_type


def pw_aff_to_aff(pw_aff, isl_op_pool):
    if isinstance(pw_aff, isl.Aff):
        return pw_aff

    assert isinstance(pw_aff, isl.PwAff)
    pieces = pw_aff.get_pieces(isl_op_pool)

    if len(pieces) == 0:
        raise RuntimeError("PwAff does not have any pieces")
    if len(pieces) > 1:
        _, first_aff = pieces[0]
        for _, other_aff in pieces[1:]:
            if not first_aff.plain_is_equal(isl_op_pool, other_aff):
                raise NotImplementedError("only single-valued piecewise affine "
                        "expressions are supported here--encountered "
                        "multi-valued expression '%s'" % pw_aff)

        return first_aff

    return pieces[0][1]


def dump_space(ls):
    return " ".join(
            "%s: %d" % (dim_type.find_value(dt), ls.dim(dt))
            for dt in range(1 + dim_type.all))


# {{{ make_slab

def make_slab(space, iname, start, stop, isl_op_pool):
    zero = isl.Aff.zero_on_domain(space)

    if isinstance(start, (isl.Aff, isl.PwAff)):
        start, zero = isl.align_two(isl_op_pool, pw_aff_to_aff(start,
                                                               isl_op_pool),
                                    zero)
    if isinstance(stop, (isl.Aff, isl.PwAff)):
        stop, zero = isl.align_two(isl_op_pool, pw_aff_to_aff(stop,
                                                              isl_op_pool),
                                   zero)

    space = zero.get_domain_space(isl_op_pool)

    from pymbolic.primitives import Expression
    from loopy.symbolic import aff_from_expr
    if isinstance(start, Expression):
        start = aff_from_expr(space, start, isl_op_pool)
    if isinstance(stop, Expression):
        stop = aff_from_expr(space, stop, isl_op_pool)

    if isinstance(start, int):
        start = zero + start
    if isinstance(stop, int):
        stop = zero + stop

    if isinstance(iname, str):
        iname_dt, iname_idx = zero.get_space(
            isl_op_pool).get_var_dict(isl_op_pool)[iname]
    else:
        iname_dt, iname_idx = iname

    iname_aff = zero.add_coefficient_val(isl_op_pool, iname_dt, iname_idx, 1)

    result = (isl.BasicSet.universe(space)
            # start <= iname
            .add_constraint(isl_op_pool, isl.Constraint.inequality_from_aff(
                iname_aff.sub(isl_op_pool, start)))
            # iname < stop
            .add_constraint(isl_op_pool, isl.Constraint.inequality_from_aff(
                (stop-1).sub(isl_op_pool, iname_aff))))

    return result


def make_loop_bounds_from_pwaffs(space, iname, lbound, ubound, isl_op_pool):
    dt, pos = space.get_var_dict(isl_op_pool)[iname]
    iname_pwaff = isl.PwAff.var_on_domain(space, dt, pos)

    iname_pwaff, lbound = isl.align_two(isl_op_pool, iname_pwaff, lbound)
    iname_pwaff, ubound = isl.align_two(isl_op_pool, iname_pwaff, ubound)
    assert iname_pwaff.get_space(isl_op_pool).is_equal(isl_op_pool,
                                                       lbound.get_space(isl_op_pool))
    assert iname_pwaff.get_space(isl_op_pool).is_equal(isl_op_pool,
                                                       ubound.get_space(isl_op_pool))

    return (iname_pwaff.ge_set(isl_op_pool,
                               lbound).intersect(isl_op_pool,
                                                 iname_pwaff.le_set(isl_op_pool,
                                                                    ubound)))

# }}}


def iname_rel_aff(space, iname, rel, aff, isl_op_pool):
    """*aff*'s domain space is allowed to not match *space*."""

    dt, pos = space.get_var_dict(isl_op_pool)[iname]
    assert dt in [dim_type.set, dim_type.param]
    if dt == dim_type.set:
        dt = dim_type.in_

    aff = isl.align_spaces(isl_op_pool, aff, isl.Aff.zero_on_domain(space))

    if rel in ["==", "<="]:
        return aff.add_coefficient_val(isl_op_pool, dt, pos, -1)
    elif rel == ">=":
        return aff.neg(isl_op_pool).add_coefficient_val(isl_op_pool, dt, pos, 1)
    elif rel == "<":
        return (aff-1).add_coefficient_val(isl_op_pool, pos, -1)
    elif rel == ">":
        return (aff+1).neg(isl_op_pool).add_coefficient_val(isl_op_pool, dt, pos, 1)
    else:
        raise ValueError("unknown value of 'rel': %s" % rel)


# {{{ simplify_pw_aff

def simplify_pw_aff(pw_aff, isl_op_pool, context=None):
    if context is not None:
        pw_aff = pw_aff.gist_params(isl_op_pool, context)

    old_pw_aff = pw_aff

    while True:
        restart = False
        did_something = False

        pieces = pw_aff.get_pieces(isl_op_pool)
        for i, (dom_i, aff_i) in enumerate(pieces):
            for j, (dom_j, aff_j) in enumerate(pieces):
                if i == j:
                    continue

                if aff_i.gist(isl_op_pool, dom_j).is_equal(isl_op_pool, aff_j):
                    # aff_i is sufficient to conver aff_j, eliminate aff_j
                    new_pieces = pieces[:]
                    if i < j:
                        new_pieces.pop(j)
                        new_pieces.pop(i)
                    else:
                        new_pieces.pop(i)
                        new_pieces.pop(j)

                    pw_aff = isl.PwAff.alloc(dom_i.union(isl_op_pool, dom_j), aff_i)
                    for dom, aff in new_pieces:
                        pw_aff = pw_aff.union_max(isl_op_pool,
                                                  isl.PwAff.alloc(dom,
                                                                  aff))

                    restart = True
                    did_something = True
                    break

            if restart:
                break

        if not did_something:
            break

    assert pw_aff.get_aggregate_domain(isl_op_pool).is_subset(
        isl_op_pool, pw_aff.eq_set(isl_op_pool, old_pw_aff))

    return pw_aff

# }}}


# {{{ static_*_of_pw_aff

def static_extremum_of_pw_aff(pw_aff, constants_only, set_method, what,
                              isl_op_pool, context):
    if context is not None:
        context = isl.align_spaces(isl_op_pool, context,
                                   pw_aff.get_domain_space(isl_op_pool),
                                   obj_bigger_ok=True).params(isl_op_pool)
        pw_aff = pw_aff.gist(isl_op_pool, context)

    pieces = pw_aff.get_pieces(isl_op_pool)
    if len(pieces) == 1:
        (_, result), = pieces
        if constants_only and not result.is_cst(isl_op_pool):
            raise StaticValueFindingError("a numeric %s was not found for PwAff '%s'"
                    % (what, pw_aff))
        return result

    from pytools import flatten  #, memoize

    # @memoize (FIXME?)
    def is_bounded(set):
        assert set.dim(isl_op_pool, dim_type.set) == 0
        return (set
                .move_dims(isl_op_pool, dim_type.set, 0,
                    dim_type.param, 0, set.dim(isl_op_pool, dim_type.param))
                .is_bounded(isl_op_pool))

    # put constant bounds with unbounded validity first
    order = [
            (True, False),  # constant, unbounded validity
            (False, False),  # nonconstant, unbounded validity
            (True, True),  # constant, bounded validity
            (False, True),  # nonconstant, bounded validity
            ]

    pieces = flatten([
            [(set, aff) for set, aff in pieces
                if aff.is_cst(isl_op_pool) == want_is_constant
                and is_bounded(set) == want_is_bounded]
            for want_is_constant, want_is_bounded in order])

    reference = pw_aff.get_aggregate_domain(isl_op_pool)
    if context is not None:
        reference = reference.intersect(isl_op_pool, context)

    # {{{ find bounds that are also global bounds

    for set, candidate_aff in pieces:
        # gist can be time-consuming, try without first
        for use_gist in [False, True]:
            if use_gist:
                candidate_aff = candidate_aff.gist(isl_op_pool, set)

            if constants_only and not candidate_aff.is_cst(isl_op_pool):
                continue

            if reference.is_subset(isl_op_pool, set_method(pw_aff, isl_op_pool,
                                                           candidate_aff)):
                return candidate_aff

    # }}}

    raise StaticValueFindingError("a static %s was not found for PwAff '%s'"
            % (what, pw_aff))


def static_min_of_pw_aff(pw_aff, constants_only, isl_op_pool, context=None):
    return static_extremum_of_pw_aff(pw_aff, constants_only, isl.PwAff.ge_set,
                                     "minimum", isl_op_pool, context)


def static_max_of_pw_aff(pw_aff, constants_only, isl_op_pool, context=None):
    return static_extremum_of_pw_aff(pw_aff, constants_only, isl.PwAff.le_set,
                                     "maximum", isl_op_pool, context)


def static_value_of_pw_aff(pw_aff, constants_only, isl_op_pool, context=None):
    return static_extremum_of_pw_aff(pw_aff, constants_only, isl.PwAff.eq_set,
                                     "value", isl_op_pool, context)

# }}}


# {{{ duplicate_axes

def duplicate_axes(isl_obj, duplicate_inames, new_inames, isl_op_pool):
    if isinstance(isl_obj, list):
        return [
                duplicate_axes(i, duplicate_inames, new_inames, isl_op_pool)
                for i in isl_obj]

    if not duplicate_inames:
        return isl_obj

    # {{{ add dims

    start_idx = isl_obj.dim(isl_op_pool, dim_type.set)
    more_dims = isl_obj.insert_dims(
            isl_op_pool, dim_type.set, start_idx, len(duplicate_inames))

    for i, iname in enumerate(new_inames):
        new_idx = start_idx+i
        more_dims = more_dims.set_dim_name(
                isl_op_pool, dim_type.set, new_idx, iname)

    # }}}

    iname_to_dim = more_dims.get_space(isl_op_pool).get_var_dict(isl_op_pool)

    moved_dims = isl_obj.copy()

    for old_iname, new_iname in zip(duplicate_inames, new_inames):
        old_dt, old_idx = iname_to_dim[old_iname]
        new_dt, new_idx = iname_to_dim[new_iname]

        moved_dims = moved_dims.set_dim_name(
                isl_op_pool, old_dt, old_idx, new_iname)
        moved_dims = (moved_dims
                .move_dims(
                    isl_op_pool,
                    dim_type.param, 0,
                    old_dt, old_idx, 1)
                .move_dims(
                    isl_op_pool,
                    new_dt, new_idx-1,
                    dim_type.param, 0, 1))

        moved_dims = moved_dims.insert_dims(isl_op_pool, old_dt, old_idx, 1)
        moved_dims = moved_dims.set_dim_name(
                isl_op_pool, old_dt, old_idx, old_iname)

    more_dims, moved_dims = isl.align_two(isl_op_pool, more_dims, moved_dims)

    return moved_dims.intersect(isl_op_pool, more_dims)

# }}}


def is_nonnegative(expr, over_set, isl_op_pool):
    space = over_set.get_space(isl_op_pool)
    from loopy.symbolic import aff_from_expr
    try:
        with isl.SuppressedWarnings(space.get_ctx(isl_op_pool)):
            aff = aff_from_expr(space, -expr-1, isl_op_pool)
    except Exception:
        return None
    expr_neg_set = isl.BasicSet.universe(space).add_constraint(
            isl_op_pool,
            isl.Constraint.inequality_from_aff(aff))

    return over_set.intersect(isl_op_pool,
                              expr_neg_set).is_empty(isl_op_pool)


# {{{ convexify

def convexify(domain, isl_op_pool):
    """Try a few ways to get *domain* to be a BasicSet, i.e.
    explicitly convex.
    """

    if isinstance(domain, isl.BasicSet):
        return domain

    dom_bsets = domain.get_basic_sets(isl_op_pool)
    if len(dom_bsets) == 1:
        domain, = dom_bsets
        return domain

    hull_domain = domain.simple_hull(isl_op_pool)
    if isl.Set.from_basic_set(hull_domain) <= domain:
        return hull_domain

    domain = domain.coalesce(isl_op_pool)

    dom_bsets = domain.get_basic_sets(isl_op_pool)
    if domain.n_basic_set(isl_op_pool) == 1:
        domain, = dom_bsets
        return domain

    hull_domain = domain.simple_hull(isl_op_pool)
    if isl.Set.from_basic_set(hull_domain).is_subset(isl_op_pool, domain):
        return hull_domain

    dom_bsets = domain.get_basic_sets(isl_op_pool)
    assert len(dom_bsets) > 1

    print("PIECES:")
    for dbs in dom_bsets:
        print("  %s" % (isl.Set.from_basic_set(dbs).gist(isl_op_pool, domain)))
    raise NotImplementedError("Could not find convex representation of set")

# }}}


# {{{ boxify

def boxify(isl_op_pool, domain, box_inames, context):
    var_dict = domain.get_var_dict(isl_op_pool, dim_type.set)
    box_iname_indices = [var_dict[iname][1] for iname in box_inames]
    n_nonbox_inames = min(box_iname_indices)

    assert box_iname_indices == list(range(
            n_nonbox_inames, domain.dim(isl_op_pool, dim_type.set)))

    n_old_parameters = domain.dim(isl_op_pool, dim_type.param)
    domain = domain.move_dims(
            isl_op_pool, dim_type.param, n_old_parameters, dim_type.set, 0,
            n_nonbox_inames)

    result = domain
    zero = isl.Aff.zero_on_domain(result.get_space(isl_op_pool))

    for i in range(len(box_iname_indices)):
        result = result.eliminate(isl_op_pool, dim_type.set, i, 1)

        iname_aff = zero.add_coefficient_val(isl_op_pool, dim_type.in_, i, 1)

        def add_in_dims(aff):
            return aff.add_dims(isl_op_pool, dim_type.in_, len(box_inames))

        iname_min = add_in_dims(domain.dim_min(isl_op_pool, i)).coalesce(isl_op_pool)
        iname_max = add_in_dims(domain.dim_max(isl_op_pool, i)).coalesce(isl_op_pool)

        iname_slab = (iname_min.le_set(isl_op_pool, iname_aff)
                .intersect(isl_op_pool, iname_max.ge_set(isl_op_pool, iname_aff)))

        for i, iname in enumerate(box_inames):
            iname_slab = iname_slab.set_dim_name(isl_op_pool, dim_type.set, i, iname)

        if context is not None:
            iname_slab, context = isl.align_two(isl_op_pool, iname_slab, context)
            iname_slab = iname_slab.gist(isl_op_pool, context)
        iname_slab = iname_slab.coalesce(isl_op_pool)

        result = result.intersect(isl_op_pool, iname_slab)

    result = result.move_dims(isl_op_pool, dim_type.set, 0, dim_type.param,
                              n_old_parameters, n_nonbox_inames)

    return convexify(result, isl_op_pool)

# }}}


def simplify_via_aff(expr, isl_op_pool):
    from loopy.symbolic import aff_from_expr, aff_to_expr, get_dependencies
    deps = get_dependencies(expr)
    return aff_to_expr(aff_from_expr(
        isl.Space.create_from_names(islpy.DEFAULT_CONTEXT, list(deps)),
        expr, isl_op_pool), isl_op_pool)


def project_out(set, inames):
    for iname in inames:
        var_dict = set.get_var_dict()
        dt, dim_idx = var_dict[iname]
        set = set.project_out(dt, dim_idx, 1)

    return set


def obj_involves_variable(obj, var_name):
    loc = obj.get_var_dict().get(var_name)
    if loc is not None:
        if not obj.get_coefficient_val(*loc).is_zero():
            return True

    for idiv in obj.dim(dim_type.div):
        if obj_involves_variable(obj.get_div(idiv), var_name):
            return True

    return False


# {{{ get_simple_strides

def get_simple_strides(bset, key_by="name"):
    """Return a dictionary from inames to strides in bset. Each stride is
    returned as a :class:`islpy.Val`. If no stride can be determined, the
    corresponding key will not be present in the returned dictionary.

    This only recognizes simple strides involving single variables.

    :arg key_by: "index" or "name"
    """
    result = {}

    comp_div_set_pieces = convexify(bset.compute_divs()).get_basic_sets()
    assert len(comp_div_set_pieces) == 1
    bset, = comp_div_set_pieces

    def _get_indices_and_coeffs(obj, dts):
        result = []
        for dt in dts:
            for dim_idx in range(obj.dim(dt)):
                coeff_val = obj.get_coefficient_val(dt, dim_idx)
                if not coeff_val.is_zero():
                    result.append((dt, dim_idx, coeff_val))

        return result

    for cns in bset.get_constraints():
        if not cns.is_equality():
            continue
        aff = cns.get_aff()

        # recognizes constraints of the form
        #  -i0 + 2*floor((i0)/2) == 0

        divs_with_coeffs = _get_indices_and_coeffs(aff, [dim_type.div])
        if len(divs_with_coeffs) != 1:
            continue

        (_, idiv, div_coeff), = divs_with_coeffs

        div = aff.get_div(idiv)

        # check for sub-divs
        if _get_indices_and_coeffs(div, [dim_type.div]):
            # found one -> not supported
            continue

        denom = div.get_denominator_val().to_python()

        # if the coefficient in front of the div is not the same as the denominator
        if not div_coeff.div(denom).is_one():
            # not supported
            continue

        inames_and_coeffs = _get_indices_and_coeffs(
                div, [dim_type.param, dim_type.in_])

        if len(inames_and_coeffs) != 1:
            continue

        (dt, dim_idx, coeff), = inames_and_coeffs

        if not (coeff * denom).is_one():
            # not supported
            continue

        inames_and_coeffs = _get_indices_and_coeffs(
                aff, [dim_type.param, dim_type.in_])

        if len(inames_and_coeffs) != 1:
            continue

        (outer_dt, outer_dim_idx, outer_coeff), = inames_and_coeffs
        if (not outer_coeff.neg().is_one()
                or (outer_dt, outer_dim_idx) != (dt, dim_idx)):
            # not supported
            continue

        if key_by == "name":
            key = bset.get_dim_name(dt, dim_idx)
        elif key_by == "index":
            key_dt = dt if dt != dim_type.in_ else dim_type.set

            key = (key_dt, dim_idx)
        else:
            raise ValueError("invalid value of 'key_by")

        result[key] = denom

    return result

# }}}


# {{{ find_max_of_pwaff_with_params

def find_max_of_pwaff_with_params(pw_aff, n_allowed_params, isl_op_pool):
    if n_allowed_params is None:
        return pw_aff

    extra_dim_idx = pw_aff.dim(isl_op_pool, dim_type.param)
    pw_aff = pw_aff.add_dims(isl_op_pool, dim_type.param, 1)

    zero = isl.Aff.zero_on_domain(pw_aff.domain(isl_op_pool).get_space(isl_op_pool))
    extra_dim = zero.set_coefficient_val(isl_op_pool, dim_type.param,
                                         extra_dim_idx,
                                         1)

    pw_aff_set = pw_aff.eq_set(isl_op_pool, extra_dim)

    pw_aff_set = pw_aff_set.move_dims(
            isl_op_pool,
            dim_type.set, 0, dim_type.param, n_allowed_params,
            pw_aff_set.dim(isl_op_pool, dim_type.param) - n_allowed_params)

    return pw_aff_set.dim_max(isl_op_pool,
                              pw_aff_set.dim(isl_op_pool, dim_type.set)-1)

# }}}


# {{{ subst_into_pw(qpolynomial|aff)

def set_dim_name(obj, dt, pos, name):
    assert isinstance(name, str)
    if isinstance(obj, isl.PwQPolynomial):
        return obj.set_dim_name(dt, pos, name)
    elif isinstance(obj, isl.PwAff):
        # work around missing isl_pw_aff_set_dim_name for now.
        # https://github.com/inducer/loopy/pull/233/files#r580594032
        return obj.set_dim_id(dt, pos, isl.Id.read_from_str(obj.get_ctx(), name))
    else:
        raise NotImplementedError(f"not implemented for {type(obj)}.")


def get_param_subst_domain(new_space, base_obj, subst_dict):
    """Modify the :mod:`islpy` object *base_obj* to incorporate parameters for
    the keys of *subst_dict*, and rename existing parameters to include a
    trailing prime.

    :arg new_space: A :class:`islpy.Space` for that contains the keys of
        *subst_dict*
    :arg subst_dict: A dictionary mapping parameters occurring in *base_obj*
        to their values in terms of variables in *new_space*
    :returns: a tuple ``(base_obj, subst_domain, subst_dict)``, where
        *base_obj* is the passed *base_obj* with the space extended to cover
        the new parameters in *new_space*, *subst_domain* is an
        :class:`islpy.BasicSet` incorporating the constraints from *subst_dict*
        and existing in the same space as *base_obj*, and *subst_dict*
        is a copy of the passed *subst_dict* modified to incorporate primed
        variable names in the keys.
    """

    # {{{ rename subst_dict keys and base_obj parameters to include trailing prime

    i_begin_subst_space = base_obj.dim(dim_type.param)

    new_subst_dict = {}
    for i in range(i_begin_subst_space):
        old_name = base_obj.space.get_dim_name(dim_type.param, i)
        new_name = old_name + "'"
        new_subst_dict[new_name] = subst_dict[old_name]
        base_obj = set_dim_name(base_obj, dim_type.param, i, new_name)

    subst_dict = new_subst_dict
    del new_subst_dict

    # }}}

    # {{{ add dimensions to base_obj

    base_obj = base_obj.add_dims(dim_type.param, new_space.dim(dim_type.param))
    for i in range(new_space.dim(dim_type.param)):
        base_obj = set_dim_name(base_obj, dim_type.param, i+i_begin_subst_space,
                new_space.get_dim_name(dim_type.param, i))

    # }}}

    # {{{ build subst_domain

    subst_domain = isl.BasicSet.universe(base_obj.space).params()

    from loopy.symbolic import guarded_aff_from_expr
    for i in range(i_begin_subst_space):
        name = base_obj.space.get_dim_name(dim_type.param, i)
        aff = guarded_aff_from_expr(subst_domain.space, subst_dict[name])
        aff = aff.set_coefficient_val(dim_type.param, i, -1)
        subst_domain = subst_domain.add_constraint(
                isl.Constraint.equality_from_aff(aff))

    # }}}

    return base_obj, subst_domain, subst_dict


def subst_into_pwqpolynomial(new_space, poly, subst_dict):
    """
    Returns an instance of :class:`islpy.PwQPolynomial` with substitutions from
    *subst_dict* substituted into *poly*.

    :arg poly: an instance of :class:`islpy.PwQPolynomial`
    :arg subst_dict: a mapping from parameters of *poly* to
        :class:`pymbolic.primitives.Expression` made up of terms comprising the
        parameters of *new_space*. The expression must be affine in the param
        dims of *new_space*.
    """
    if not poly.get_pieces():
        # pw poly is univserally zero
        result = isl.PwQPolynomial.zero(new_space.insert_dims(dim_type.out, 0, 1))
        assert result.dim(dim_type.out) == 1
        return result

    i_begin_subst_space = poly.dim(dim_type.param)

    poly, subst_domain, subst_dict = get_param_subst_domain(
            new_space, poly, subst_dict)

    from loopy.symbolic import qpolynomial_to_expr, qpolynomial_from_expr
    new_pieces = []
    for valid_set, qpoly in poly.get_pieces():
        valid_set = valid_set & subst_domain
        if valid_set.plain_is_empty():
            continue

        valid_set = valid_set.project_out(dim_type.param, 0, i_begin_subst_space)
        from pymbolic.mapper.substitutor import (
                SubstitutionMapper, make_subst_func)
        sub_mapper = SubstitutionMapper(make_subst_func(subst_dict))
        expr = sub_mapper(qpolynomial_to_expr(qpoly))
        qpoly = qpolynomial_from_expr(valid_set.space, expr)

        new_pieces.append((valid_set, qpoly))

    if not new_pieces:
        raise ValueError("no pieces of PwQPolynomial survived the substitution")

    valid_set, qpoly = new_pieces[0]
    result = isl.PwQPolynomial.alloc(valid_set, qpoly)
    for valid_set, qpoly in new_pieces[1:]:
        result = result.add_disjoint(
                isl.PwQPolynomial.alloc(valid_set, qpoly))

    assert result.dim(dim_type.out)
    return result


def subst_into_pwaff(new_space, pwaff, subst_dict):
    """
    Returns an instance of :class:`islpy.PwAff` with substitutions from
    *subst_dict* substituted into *pwaff*.

    :arg pwaff: an instance of :class:`islpy.PwAff`
    :arg subst_dict: a mapping from parameters of *pwaff* to
        :class:`pymbolic.primitives.Expression` made up of terms comprising the
        parameters of *new_space*. The expression must be affine in the param
        dims of *new_space*.
    """
    from pymbolic.mapper.substitutor import (
            SubstitutionMapper, make_subst_func)
    from loopy.symbolic import aff_from_expr, aff_to_expr
    from functools import reduce

    i_begin_subst_space = pwaff.dim(dim_type.param)
    pwaff, subst_domain, subst_dict = get_param_subst_domain(
            new_space, pwaff, subst_dict)
    subst_mapper = SubstitutionMapper(make_subst_func(subst_dict))
    pwaffs = []

    for valid_set, qpoly in pwaff.get_pieces():
        valid_set = valid_set & subst_domain
        if valid_set.plain_is_empty():
            continue

        valid_set = valid_set.project_out(dim_type.param, 0, i_begin_subst_space)
        aff = aff_from_expr(valid_set.space, subst_mapper(aff_to_expr(qpoly)))

        pwaffs.append(isl.PwAff.alloc(valid_set, aff))

    if not pwaffs:
        raise ValueError("no pieces of PwAff survived the substitution")

    return reduce(lambda pwaff1, pwaff2: pwaff1.union_add(pwaff2),
                  pwaffs).coalesce()

# }}}

# vim: foldmethod=marker

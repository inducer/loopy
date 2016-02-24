"""isl helpers"""

from __future__ import division, absolute_import

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


from six.moves import range, zip

from loopy.diagnostic import StaticValueFindingError

import islpy as isl
from islpy import dim_type


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

    if isinstance(start, int):
        start = zero + start
    if isinstance(stop, int):
        stop = zero + stop

    if isinstance(iname, str):
        iname_dt, iname_idx = zero.get_space().get_var_dict()[iname]
    else:
        iname_dt, iname_idx = iname

    iname_aff = zero.add_coefficient_val(iname_dt, iname_idx, 1)

    result = (isl.BasicSet.universe(space)
            # start <= iname
            .add_constraint(isl.Constraint.inequality_from_aff(
                iname_aff - start))
            # iname < stop
            .add_constraint(isl.Constraint.inequality_from_aff(
                stop-1 - iname_aff)))

    return result


def make_slab_from_bound_pwaffs(space, iname, lbound, ubound):
    dt, pos = space.get_var_dict()[iname]
    iname_pwaff = isl.PwAff.var_on_domain(space, dt, pos)

    iname_pwaff, lbound = isl.align_two(iname_pwaff, lbound)
    iname_pwaff, ubound = isl.align_two(iname_pwaff, ubound)
    assert iname_pwaff.space == lbound.space
    assert iname_pwaff.space == ubound.space

    return convexify(
            iname_pwaff.ge_set(lbound)
            &
            iname_pwaff.le_set(ubound))


def iname_rel_aff(space, iname, rel, aff):
    """*aff*'s domain space is allowed to not match *space*."""

    dt, pos = space.get_var_dict()[iname]
    assert dt in [isl.dim_type.set, isl.dim_type.param]
    if dt == isl.dim_type.set:
        dt = isl.dim_type.in_

    from islpy import align_spaces
    aff = align_spaces(aff, isl.Aff.zero_on_domain(space))

    if rel in ["==", "<="]:
        return aff.add_coefficient_val(dt, pos, -1)
    elif rel == ">=":
        return aff.neg().add_coefficient_val(dt, pos, 1)
    elif rel == "<":
        return (aff-1).add_coefficient_val(dt, pos, -1)
    elif rel == ">":
        return (aff+1).neg().add_coefficient_val(dt, pos, 1)
    else:
        raise ValueError("unknown value of 'rel': %s" % rel)


def static_extremum_of_pw_aff(pw_aff, constants_only, set_method, what, context):
    if context is not None:
        context = isl.align_spaces(context, pw_aff.get_domain_space(),
                obj_bigger_ok=True).params()
        pw_aff = pw_aff.gist(context)

    pieces = pw_aff.get_pieces()
    if len(pieces) == 1:
        (_, result), = pieces
        if constants_only and not result.is_cst():
            raise ValueError("a numeric %s was not found for PwAff '%s'"
                    % (what, pw_aff))
        return result

    from pytools import memoize, flatten

    @memoize
    def is_bounded(set):
        assert set.dim(dim_type.set) == 0
        return (set
                .move_dims(dim_type.set, 0,
                    dim_type.param, 0, set.dim(dim_type.param))
                .is_bounded())

    # put constant bounds with unbounded validity first
    order = [
            (True, False),  # constant, unbounded validity
            (False, False),  # nonconstant, unbounded validity
            (True, True),  # constant, bounded validity
            (False, True),  # nonconstant, bounded validity
            ]

    pieces = flatten([
            [(set, aff) for set, aff in pieces
                if aff.is_cst() == want_is_constant
                and is_bounded(set) == want_is_bounded]
            for want_is_constant, want_is_bounded in order])

    reference = pw_aff.get_aggregate_domain()
    if context is not None:
        reference = reference.intersect(context)

    # {{{ find bounds that are also global bounds

    for set, candidate_aff in pieces:
        # gist can be time-consuming, try without first
        for use_gist in [False, True]:
            if use_gist:
                candidate_aff = candidate_aff.gist(set)

            if constants_only and not candidate_aff.is_cst():
                continue

            if reference <= set_method(pw_aff, candidate_aff):
                return candidate_aff

    # }}}

    raise StaticValueFindingError("a static %s was not found for PwAff '%s'"
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


def convexify(domain):
    """Try a few ways to get *domain* to be a BasicSet, i.e.
    explicitly convex.
    """

    if isinstance(domain, isl.BasicSet):
        return domain

    dom_bsets = domain.get_basic_sets()
    if len(dom_bsets) == 1:
        domain, = dom_bsets
        return domain

    hull_domain = domain.simple_hull()
    if isl.Set.from_basic_set(hull_domain) <= domain:
        return hull_domain

    domain = domain.coalesce()

    dom_bsets = domain.get_basic_sets()
    if len(domain.get_basic_sets()) == 1:
        domain, = dom_bsets
        return domain

    hull_domain = domain.simple_hull()
    if isl.Set.from_basic_set(hull_domain) <= domain:
        return hull_domain

    dom_bsets = domain.get_basic_sets()
    assert len(dom_bsets) > 1

    print("PIECES:")
    for dbs in dom_bsets:
        print("  %s" % (isl.Set.from_basic_set(dbs).gist(domain)))
    raise NotImplementedError("Could not find convex representation of set")


def boxify(cache_manager, domain, box_inames, context):
    var_dict = domain.get_var_dict(dim_type.set)
    box_iname_indices = [var_dict[iname][1] for iname in box_inames]
    n_nonbox_inames = min(box_iname_indices)

    assert box_iname_indices == list(range(
            n_nonbox_inames, domain.dim(dim_type.set)))

    n_old_parameters = domain.dim(dim_type.param)
    domain = domain.move_dims(
            dim_type.param, n_old_parameters, dim_type.set, 0, n_nonbox_inames)

    result = domain
    zero = isl.Aff.zero_on_domain(result.space)

    for i in range(len(box_iname_indices)):
        result = result.eliminate(dim_type.set, i, 1)

        iname_aff = zero.add_coefficient_val(dim_type.in_, i, 1)

        def add_in_dims(aff):
            return aff.add_dims(dim_type.in_, len(box_inames))

        iname_min = add_in_dims(cache_manager.dim_min(domain, i)).coalesce()
        iname_max = add_in_dims(cache_manager.dim_max(domain, i)).coalesce()

        iname_slab = (iname_min.le_set(iname_aff)
                .intersect(iname_max.ge_set(iname_aff)))

        for i, iname in enumerate(box_inames):
            iname_slab = iname_slab.set_dim_name(dim_type.set, i, iname)

        if context is not None:
            iname_slab, context = isl.align_two(iname_slab, context)
            iname_slab = iname_slab.gist(context)
        iname_slab = iname_slab.coalesce()

        result = result & iname_slab

    result = result.move_dims(
            dim_type.set, 0, dim_type.param, n_old_parameters, n_nonbox_inames)

    return convexify(result)


def simplify_via_aff(expr):
    from loopy.symbolic import aff_from_expr, aff_to_expr, get_dependencies
    deps = get_dependencies(expr)
    return aff_to_expr(aff_from_expr(
        isl.Space.create_from_names(isl.DEFAULT_CONTEXT, list(deps)),
        expr))


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
        if obj_involves_variable(obj.get_div(idiv)):
            return True

    return False


# {{{ performance tweak for dim_{min,max}: project first

def _runs_in_integer_set(s, max_int=None):
    if not s:
        return

    if max_int is None:
        max_int = max(s)

    i = 0
    while i < max_int:
        if i in s:
            start = i

            i += 1
            while i < max_int and i in s:
                i += 1

            end = i

            yield (start, end-start)

        else:
            i += 1


class TooManyInteractingDims(Exception):
    pass


def _find_aff_dims(aff, dim_types_and_gen_dim_types):
    result = []

    for dt, gen_dt in dim_types_and_gen_dim_types:
        for i in range(aff.dim(dt)):
            if not aff.get_coefficient_val(dt, i).is_zero():
                result.append((gen_dt, i))

    result = set(result)

    for i in range(aff.dim(dim_type.div)):
        if not aff.get_coefficient_val(dim_type.div, i).is_zero():
            result.update(_find_aff_dims(
                aff.get_div(i),
                dim_types_and_gen_dim_types))

    return result


def _transitive_closure(graph_dict):
    pass


def _find_noninteracting_dims(obj, dt, idx, other_dt, stop_at=6):
    if isinstance(obj, isl.BasicSet):
        basics = [obj]
    elif isinstance(obj, isl.Set):
        basics = obj.get_basic_sets()
    else:
        raise TypeError("unsupported arg type '%s'" % type(obj))

    connections = []
    for bs in basics:
        for c in bs.get_constraints():
            conn = _find_aff_dims(
                    c.get_aff(),
                    [(dim_type.param, dim_type.param), (dim_type.in_, dim_type.set)])
            if len(conn) > 1:
                connections.append(conn)

    interacting = set([(dt, idx)])

    while True:
        changed_something = False

        # Compute the connected component near (dt, idx) by fixed point iteration

        for conn in connections:
            prev_len = len(interacting)

            overlap = interacting & conn
            if overlap:
                interacting.update(conn)

            if len(interacting) != prev_len:
                changed_something = True

            if len(interacting) >= stop_at:
                raise TooManyInteractingDims()

        if not changed_something:
            break

    return set(range(obj.dim(other_dt))) - set(
            idx for dt, idx in interacting
            if dt == other_dt)


def _eliminate_noninteracting(obj, dt, idx, other_dt):
    obj = obj.compute_divs()
    try:
        nonint = _find_noninteracting_dims(obj, dt, idx, other_dt)

    except TooManyInteractingDims:
        return obj

    for first, n in _runs_in_integer_set(nonint):
        obj = obj.eliminate(other_dt, first, n)

    return obj


def dim_min_with_elimination(obj, idx):
    obj_elim = _eliminate_noninteracting(obj, dim_type.out, idx, dim_type.param)
    return obj_elim.dim_min(idx)


def dim_max_with_elimination(obj, idx):
    obj_elim = _eliminate_noninteracting(obj, dim_type.out, idx, dim_type.param)
    return obj_elim.dim_max(idx)

# }}}


def get_simple_strides(bset, key_by="name"):
    """Return a dictionary from inames to strides in bset. Each stride is
    returned as a :class:`islpy.Val`. If no stride can be determined, the
    corresponding key will not be present in the returned dictionary.

    This only recognizes simple strides involving single variables.

    :arg key_by: "index" or "name"
    """
    result = {}

    lspace = bset.get_local_space()
    for idiv in range(lspace.dim(dim_type.div)):
        div = lspace.get_div(idiv)

        # check for sub-divs
        supported = True
        for dim_idx in range(div.dim(dim_type.div)):
            coeff_val = div.get_coefficient_val(dim_type.div, dim_idx)
            if not coeff_val.is_zero():
                # sub-divs not supported
                supported = False
                break

        if not supported:
            continue

        denom = div.get_denominator_val().to_python()

        inames_and_coeffs = []
        for dt in [dim_type.param, dim_type.in_]:
            for dim_idx in range(div.dim(dt)):
                coeff_val = div.get_coefficient_val(dt, dim_idx) * denom
                if not coeff_val.is_zero():
                    inames_and_coeffs.append((dt, dim_idx, coeff_val))

        if len(inames_and_coeffs) != 1:
            continue

        (dt, dim_idx, coeff), = inames_and_coeffs

        if coeff != 1:
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

# vim: foldmethod=marker

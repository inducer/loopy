from __future__ import annotations


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


from collections.abc import Collection
from typing import TYPE_CHECKING, TypeVar

import islpy as isl
import namedisl as nisl
from islpy import dim_type
from namedisl import DimType

from loopy.diagnostic import (
    ExpressionToAffineConversionError,
    LoopyError,
    StaticValueFindingError,
)
from loopy.typing import InameStr, not_none


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pymbolic import ArithmeticExpression, Expression


def find_max_of_pwaff_with_params(
            pw_aff: nisl.PwAff,
            allowed_params: Collection[str] | None,
            *, cache: nisl.Cache | None,
        ):
    """Get a parametric maximmum of a :class:`namedisl.PwAff`.

    :arg allowed_params: 'None' means all, making this a no-op.
    """
    if allowed_params is None:
        return pw_aff

    out_name = "_lpy_output"
    pw_aff = pw_aff.add_dims(DimType.in_, [out_name])
    pw_aff_set = pw_aff.eq_set(pw_aff.var_pw_affs[out_name])
    pw_aff_set = pw_aff_set.move_dims(allowed_params, DimType.param)
    return pw_aff_set.dim_max(out_name, cache=cache)


def base_index_and_length(
        cache: nisl.Cache,
        set_: nisl.Set | nisl.BasicSet,
        iname: InameStr,
        context: nisl.Set | nisl.BasicSet | None = None,
        allowed_params_in_length: Collection[str] | None = None,
    ) -> tuple[ArithmeticExpression, ArithmeticExpression]:
    # copied and mildly adapted from
    # https://github.com/inducer/loopy/blob/eab7e4d3a4341229084de2d44a84bd9c60e3c611/loopy/kernel/tools.py#L454C1-L524C1
    """
    :arg n_allowed_params_in_length: Simplifies the 'length'
        argument so that only the first that many params
        (in the domain of *set_*) occur.
    """
    if isinstance(set_, nisl.BasicSet):
        set_ = set_.as_set()
    if isinstance(context, nisl.BasicSet):
        context = context.as_set()

    lower_bound_pw_aff = set_.dim_min(iname, cache=cache)
    upper_bound_pw_aff = set_.dim_max(iname, cache=cache)

    from loopy.diagnostic import StaticValueFindingError
    from loopy.isl_helpers import (
        static_max_of_pw_aff,
        static_min_of_pw_aff,
        static_value_of_pw_aff,
    )
    from loopy.symbolic import pw_aff_to_expr

    # {{{ first: try to find static lower bound value

    try:
        base_index_aff = static_value_of_pw_aff(
                lower_bound_pw_aff, constants_only=False,
                context=context)
    except StaticValueFindingError:
        base_index_aff = None

    if base_index_aff is not None:
        base_index = pw_aff_to_expr(base_index_aff.as_pw_aff())

        length = find_max_of_pwaff_with_params(
                (upper_bound_pw_aff
                - base_index_aff.as_pw_aff() + 1),
                allowed_params_in_length,
                cache=cache)
        length = pw_aff_to_expr(static_max_of_pw_aff(
                length, constants_only=False,
                context=context).as_pw_aff())

        return base_index, length

    # }}}

    # {{{ if that didn't work, try finding a lower bound

    base_index_aff = static_min_of_pw_aff(
            lower_bound_pw_aff, constants_only=False,
            context=context)

    base_index = pw_aff_to_expr(base_index_aff.as_pw_aff())

    length = find_max_of_pwaff_with_params(
            (upper_bound_pw_aff - base_index_aff.as_pw_aff() + 1),
            allowed_params_in_length,
            cache=cache,
    )
    length = pw_aff_to_expr(static_max_of_pw_aff(
            length, constants_only=False,
            context=context).as_pw_aff())

    return base_index, length


# {{{ simplify_pw_aff

def simplify_pw_aff(pw_aff: nisl.PwAff, context: nisl.Set | None = None):
    if context is not None:
        pw_aff = pw_aff.gist_params(context)

    old_pw_aff = pw_aff

    while True:
        restart = False
        did_something = False

        pieces = pw_aff.get_pieces()
        for i, (dom_i, aff_i) in enumerate(pieces):
            for j, (dom_j, aff_j) in enumerate(pieces):
                if i == j:
                    continue

                if aff_i.gist(dom_j) == aff_j:
                    # aff_i is sufficient to cover aff_j, eliminate aff_j
                    new_pieces = pieces[:]
                    if i < j:
                        new_pieces.pop(j)
                        new_pieces.pop(i)
                    else:
                        new_pieces.pop(i)
                        new_pieces.pop(j)

                    pw_aff = nisl.PwAff.from_piece_and_aff(dom_i | dom_j, aff_i)
                    for dom, aff in new_pieces:
                        pw_aff = pw_aff.union_max(
                            nisl.PwAff.from_piece_and_aff(dom, aff))

                    restart = True
                    did_something = True
                    break

            if restart:
                break

        if not did_something:
            break

    assert pw_aff.get_aggregate_domain() <= pw_aff.where("==", old_pw_aff)

    return pw_aff

# }}}


# {{{ static_*_of_pw_aff

def static_extremum_of_pw_aff(
            pw_aff: nisl.PwAff,
            constants_only: bool,
            set_method: Callable[[nisl.PwAff, nisl.PwAff], nisl.Set],
            what: str,
            context: nisl.Set | None,
        ) -> nisl.Aff:
    if context is not None:
        pw_aff = pw_aff.gist(context)

    pieces = pw_aff.get_pieces()
    if len(pieces) == 1:
        (_, result), = pieces
        if constants_only and not result.is_constant():
            raise StaticValueFindingError(
                f"a numeric {what} was not found for PwAff '{pw_aff}'")
        return result

    from pytools import flatten, memoize

    @memoize
    def is_bounded(set: nisl.Set):
        assert set.space.dim(DimType.out) == 0
        return (set
        .move_dims(set.space.dim_names(DimType.param), DimType.out)
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
                if aff.is_constant() == want_is_constant
                and is_bounded(set) == want_is_bounded]
            for want_is_constant, want_is_bounded in order])

    reference = pw_aff.get_aggregate_domain()
    if context is not None:
        reference = reference & context

    # {{{ find bounds that are also global bounds

    for set, candidate_aff in pieces:
        # gist can be time-consuming, try without first
        for use_gist in [False, True]:
            if use_gist:
                candidate_aff = candidate_aff.gist(set)

            if constants_only and not candidate_aff.is_constant():
                continue

            if reference <= set_method(pw_aff, candidate_aff.as_pw_aff()):
                return candidate_aff

    # }}}

    raise StaticValueFindingError("a static %s was not found for PwAff '%s'"
            % (what, pw_aff))


def static_min_of_pw_aff(
            pw_aff: nisl.PwAff,
            constants_only: bool,
            context: nisl.Set | None = None,
        ) -> nisl.Aff:
    return static_extremum_of_pw_aff(pw_aff, constants_only, nisl.PwAff.ge_set,
    "minimum", context)


def static_max_of_pw_aff(
            pw_aff: nisl.PwAff,
            constants_only: bool,
            context: nisl.Set | None = None,
        ) -> nisl.Aff:
    return static_extremum_of_pw_aff(pw_aff, constants_only, nisl.PwAff.le_set,
    "maximum", context)


def static_value_of_pw_aff(
            pw_aff: nisl.PwAff,
            constants_only: bool,
            context: nisl.Set | None = None,
        ) -> nisl.Aff:
    return static_extremum_of_pw_aff(pw_aff, constants_only, nisl.PwAff.eq_set,
    "value", context)

# }}}


# {{{ duplicate_axes

SetT = TypeVar("SetT", nisl.BasicSet, nisl.Set)


def duplicate_axes(
            isl_obj: SetT,
            duplicate_inames: Sequence[str],
            new_inames: Sequence[str]
        ) -> SetT:
    """
    Duplicates dim names in *duplicate_inames* with corresponding names in
    *new_inames*.

    .. testsetup::

        >>> import islpy as isl
        >>> from loopy.isl_helpers import duplicate_axes

    .. doctest::

        >>> bset = nisl.BasicSet("{[i, j]: 0<=i<10 and 0<=j<30}")
        >>> duplicate_axes(bset, ("i",), ("i'",))
        BasicSet("{ [i, j, i'] : 0 <= i <= 9 and 0 <= j <= 29 and 0 <= i' <= 9 }")
    """
    if not isinstance(isl_obj, (nisl.Set, nisl.BasicSet)):
        return [
                duplicate_axes(i, duplicate_inames, new_inames)
                for i in isl_obj]

    if not duplicate_inames:
        return isl_obj

    return isl_obj.rename_dims(zip(duplicate_inames, new_inames, strict=True)) & isl_obj


def duplicate_axes_multi(
            isl_obj: Sequence[SetT],
            duplicate_inames: Sequence[str],
            new_inames: Sequence[str]
        ) -> Sequence[SetT]:
    return [
            duplicate_axes(i, duplicate_inames, new_inames)
            for i in isl_obj]

# }}}


def is_nonnegative(expr: ArithmeticExpression, over_set: nisl.Set) -> bool | None:
    from pymbolic.primitives import Product

    from loopy.symbolic import guarded_pwaff_from_expr

    if isinstance(expr, Product) and all(
            is_nonnegative(child, over_set) for child in expr.children):
        return True

    try:
        pwaff = guarded_pwaff_from_expr(over_set.var_pw_affs, expr)
    except ExpressionToAffineConversionError:
        return None

    expr_neg_set = pwaff.where("<", 0)

    return (over_set & expr_neg_set).is_empty()


# {{{ convexify

def convexify(domain: nisl.Set) -> nisl.BasicSet:
    """Try a few ways to get *domain* to be a BasicSet, i.e.
    explicitly convex.
    """

    if isinstance(domain, nisl.BasicSet):
        return domain

    dom_bsets = domain.get_basic_sets()
    if len(dom_bsets) == 1:
        bset, = dom_bsets
        return bset

    hull_domain = domain.simple_hull()
    if hull_domain.as_set() <= domain:
        return hull_domain

    domain = domain.coalesce()

    dom_bsets = domain.get_basic_sets()
    if len(dom_bsets) == 1:
        bset, = dom_bsets
        return bset

    hull_domain = domain.simple_hull()
    if hull_domain.as_set() <= domain:
        return hull_domain

    dom_bsets = domain.get_basic_sets()
    assert len(dom_bsets) > 1

    print("PIECES:")
    for dbs in dom_bsets:
        print("  %s" % (dbs.as_set().gist(domain)))
    raise NotImplementedError("Could not find convex representation of set")

# }}}


# {{{ boxify

def boxify(
    cache: nisl.Cache,
    domain: nisl.Set,
    box_inames: frozenset[str],
    context: nisl.Set
):
    nonbox_names = domain.space.set_names - box_inames
    domain = domain.move_dims(nonbox_names, DimType.param)

    result = domain.eliminate(box_inames)

    v = result.var_pw_affs
    for box_iname in box_inames:
        iname_slab = (
            v[box_iname].where(">=", domain.dim_min(box_iname, cache=cache))
                & v[box_iname].where("<=", domain.dim_max(box_iname, cache=cache))
        )

        if context is not None:
            iname_slab = iname_slab.gist(context)
            iname_slab = iname_slab.coalesce()

        result = result & iname_slab

    result = result.move_dims(nonbox_names, DimType.out)

    return convexify(result)

# }}}


def obj_involves_variable(obj, var_name):
    loc = obj.get_var_dict().get(var_name)
    if loc is not None and not obj.get_coefficient_val(*loc).is_zero():
        return True

    for idiv in obj.dim(DimType.div):
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

        divs_with_coeffs = _get_indices_and_coeffs(aff, [DimType.div])
        if len(divs_with_coeffs) != 1:
            continue

        (_, idiv, div_coeff), = divs_with_coeffs

        div = aff.get_div(idiv)

        # check for sub-divs
        if _get_indices_and_coeffs(div, [DimType.div]):
            # found one -> not supported
            continue

        denom = div.get_denominator_val().to_python()

        # if the coefficient in front of the div is not the same as the denominator
        if not div_coeff.div(denom).is_one():
            # not supported
            continue

        inames_and_coeffs = _get_indices_and_coeffs(
                div, [DimType.param, DimType.in_])

        if len(inames_and_coeffs) != 1:
            continue

        (dt, dim_idx, coeff), = inames_and_coeffs

        if not (coeff * denom).is_one():
            # not supported
            continue

        inames_and_coeffs = _get_indices_and_coeffs(
                aff, [DimType.param, DimType.in_])

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
            key_dt = dt if dt != DimType.in_ else DimType.set

            key = (key_dt, dim_idx)
        else:
            raise ValueError("invalid value of 'key_by")

        result[key] = denom

    return result

# }}}


# {{{ subst_into_pw(qpolynomial|aff)

PwAffOrPolynomialT = TypeVar("PwAffOrPolynomialT", nisl.PwAff, nisl.PwQPolynomial)


def get_param_subst_domain(
            new_space: nisl.Space,
            base_obj: PwAffOrPolynomialT,
            subst_dict: Mapping[str, Expression],
        ) -> tuple[
            PwAffOrPolynomialT, nisl.Set,
            Mapping[str, Expression], Collection[str]]:
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

    new_subst_dict: dict[str, Expression] = {}
    renaming: list[tuple[str, str]] = []
    primed_names: list[str] = []
    for old_name in base_obj.space.param_names:
        new_name = old_name + "'"
        new_subst_dict[new_name] = subst_dict[old_name]
        primed_names.append(new_name)
        renaming.append((old_name, new_name))

    base_obj = base_obj.rename_dims(renaming)

    subst_dict = new_subst_dict
    del new_subst_dict

    # }}}

    # {{{ add dimensions to base_obj

    base_obj = base_obj.add_dims(DimType.param, new_space.param_names)

    # }}}

    # {{{ build subst_domain

    subst_domain = nisl.Set.universe(base_obj.space.as_set_space())
    v = subst_domain.var_pw_affs

    from loopy.symbolic import guarded_pwaff_from_expr
    for name in primed_names:
        subst_domain = subst_domain & (
            v[name].where("==",
                guarded_pwaff_from_expr(v, subst_dict[name])))

    # }}}

    return base_obj, subst_domain, subst_dict, primed_names


def subst_into_pwqpolynomial(
             new_space: nisl.Space,
             poly: nisl.PwQPolynomial,
             subst_dict: Mapping[str, Expression]
         ):
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
        result = nisl.PwQPolynomial.zero(new_space.insert_dims(DimType.out, 0, 1))
        assert result.dim(DimType.out) == 1
        return result

    i_begin_subst_space = poly.dim(DimType.param)

    poly, subst_domain, subst_dict = get_param_subst_domain(
            new_space, poly, subst_dict)

    from loopy.symbolic import qpolynomial_from_expr, qpolynomial_to_expr
    new_pieces = []
    for valid_set, qpoly in poly.get_pieces():
        valid_set = valid_set & subst_domain
        if valid_set.plain_is_empty():
            continue

        valid_set = valid_set.project_out(DimType.param, 0, i_begin_subst_space)
        from pymbolic.mapper.substitutor import SubstitutionMapper, make_subst_func
        sub_mapper = SubstitutionMapper(make_subst_func(subst_dict))
        expr = sub_mapper(qpolynomial_to_expr(qpoly))
        qpoly = qpolynomial_from_expr(valid_set.space, expr)

        new_pieces.append((valid_set, qpoly))

    if not new_pieces:
        raise ValueError("no pieces of PwQPolynomial survived the substitution")

    valid_set, qpoly = new_pieces[0]
    result = nisl.PwQPolynomial.alloc(valid_set, qpoly)
    for valid_set, qpoly in new_pieces[1:]:
        result = result.add_disjoint(
                nisl.PwQPolynomial.alloc(valid_set, qpoly))

    assert result.dim(DimType.out)
    return result


def subst_into_pwaff(
             new_space: nisl.Space,
             pwaff: nisl.PwAff,
             subst_dict: Mapping[str, Expression]
         ):
    """
    Returns an instance of :class:`islpy.PwAff` with substitutions from
    *subst_dict* substituted into *pwaff*.

    :arg pwaff: an instance of :class:`islpy.PwAff`
    :arg subst_dict: a mapping from parameters of *pwaff* to
        :class:`pymbolic.primitives.Expression` made up of terms comprising the
        parameters of *new_space*. The expression must be affine in the param
        dims of *new_space*.
    """
    from functools import reduce

    from pymbolic.mapper.substitutor import SubstitutionMapper, make_subst_func

    from loopy.symbolic import aff_from_expr, aff_to_expr

    pwaff, subst_domain, subst_dict, primed_names = get_param_subst_domain(
            new_space, pwaff, subst_dict)
    subst_mapper = SubstitutionMapper(make_subst_func(subst_dict))
    pwaffs: list[nisl.PwAff] = []

    for valid_set, qpoly in pwaff.get_pieces():
        valid_set = valid_set & subst_domain
        if valid_set.plain_is_empty():
            continue

        valid_set = valid_set.project_out(primed_names)
        aff = aff_from_expr(valid_set.var_affs, subst_mapper(aff_to_expr(qpoly)))

        pwaffs.append(nisl.PwAff.from_piece_and_aff(valid_set, aff))

    if not pwaffs:
        raise ValueError("no pieces of PwAff survived the substitution")

    return reduce(lambda pwaff1, pwaff2: pwaff1.union_add(pwaff2),
                  pwaffs).coalesce()

# }}}


# {{{ add_eq_constraint_from_names

def add_eq_constraint_from_names(isl_obj, var1, var2):
    """Add constraint *var1* = *var2* to an ISL object.

    :arg isl_obj: An :class:`islpy.Set` or  :class:`islpy.Map` to which
        a new constraint will be added.

    :arg var1: A :class:`str` specifying the name of the first variable
        involved in constraint *var1* = *var2*.

    :arg var2: A :class:`str` specifying the name of the second variable
        involved in constraint *var1* = *var2*.

    :returns: An object of the same type as *isl_obj* with the constraint
        *var1* = *var2*.

    """
    return isl_obj.add_constraint(
               nisl.Constraint.eq_from_names(
                   isl_obj.space,
                   {1: 0, var1: 1, var2: -1}))

# }}}


# vim: foldmethod=marker

from __future__ import division

import islpy as isl
from islpy import dim_type




# {{{ expression -> constraint conversion

def _constraint_from_expr(space, expr, constraint_factory):
    from loopy.symbolic import CoefficientCollector
    return constraint_factory(space,
            CoefficientCollector()(expr))

def eq_constraint_from_expr(space, expr):
    return _constraint_from_expr(
            space, expr, isl.Constraint.eq_from_names)

def ineq_constraint_from_expr(space, expr):
    return _constraint_from_expr(
            space, expr, isl.Constraint.ineq_from_names)

# }}}


# {{{ isl helpers

def get_bounds_constraints(bset, iname, space=None, admissible_vars=None):
    if isinstance(bset, isl.Set):
        bset, = bset.get_basic_sets()

    constraints = bset.get_constraints()

    if not isinstance(admissible_vars, set):
        admissible_vars = set(admissible_vars)

    lower = []
    upper = []
    equality = []

    if space is None:
        space = bset.get_dim()

    var_dict = space.get_var_dict()
    iname_tp, iname_idx = var_dict[iname]

    for cns in constraints:
        iname_coeff = int(cns.get_coefficient(iname_tp, iname_idx))

        if admissible_vars is not None:
            if not (set(cns.get_coefficients_by_name().iterkeys())
                    <= admissible_vars):
                continue

        if iname_coeff == 0:
            continue

        if cns.is_equality():
            equality.append(cns)
        elif iname_coeff < 0:
            upper.append(cns)
        else: #  iname_coeff > 0
            lower.append(cns)

    return lower, upper, equality


def get_projected_bounds_constraints(set, iname):
    """Get an overapproximation of the loop bounds for the variable *iname*,
    as constraints.
    """

    # project out every variable except iname
    projected_domain = isl.project_out_except(set, [iname], [dim_type.set])

    basic_sets = projected_domain.get_basic_sets()

    # FIXME perhaps use some form of hull here if there's more than one
    # basic set?
    bset, = basic_sets

    # Python-style, half-open bounds
    upper_bounds = []
    lower_bounds = []
    bset = bset.remove_divs()

    bset_iname_dim_type, bset_iname_idx = bset.get_dim().get_var_dict()[iname]

    def examine_constraint(cns):
        assert not cns.is_equality()
        assert not cns.is_div_constraint()

        coeffs = cns.get_coefficients_by_name()

        iname_coeff = int(coeffs.get(iname, 0))
        if iname_coeff == 0:
            return
        elif iname_coeff < 0:
            upper_bounds.append(cns)
        else: # iname_coeff > 0:
            lower_bounds.append(cns)

    bset.foreach_constraint(examine_constraint)

    lb, = lower_bounds
    ub, = upper_bounds

    return lb, ub




def solve_constraint_for_bound(cns, iname):
    rhs, iname_coeff = constraint_to_expr(cns, except_name=iname)

    if iname_coeff == 0:
        raise ValueError("cannot solve constraint for '%s'--"
                "constraint does not contain variable"
                % iname)

    from pymbolic import expand
    from pytools import div_ceil
    from pymbolic import flatten
    from pymbolic.mapper.constant_folder import CommutativeConstantFoldingMapper
    cfm = CommutativeConstantFoldingMapper()

    if iname_coeff > 0 or cns.is_equality():
        if cns.is_equality():
            kind = "=="
        else:
            kind = ">="

        return kind, cfm(flatten(div_ceil(expand(-rhs), iname_coeff)))
    else: # iname_coeff < 0
        from pytools import div_ceil
        return "<", cfm(flatten(div_ceil(rhs+1, -iname_coeff)))




def get_projected_bounds(set, iname):
    """Get an overapproximation of the loop bounds for the variable *iname*,
    as actual bounds.
    """

    lb_cns, ub_cns = get_projected_bounds_constraints(set, iname)

    for cns in [lb_cns, ub_cns]:
        iname_tp, iname_idx = lb_cns.get_dim().get_var_dict()[iname]
        iname_coeff = cns.get_coefficient(iname_tp, iname_idx)

        if iname_coeff == 0:
            continue

        kind, bound = solve_constraint_for_bound(cns, iname)
        if kind == "<":
            ub = bound
        elif kind == ">=":
            lb = bound
        else:
            raise ValueError("unsupported constraint kind")

    return lb, ub

def cast_constraint_to_space(cns, new_space, as_equality=None):
    if as_equality is None:
        as_equality = cns.is_equality()

    if as_equality:
        factory = isl.Constraint.eq_from_names
    else:
        factory = isl.Constraint.ineq_from_names
    return factory(new_space, cns.get_coefficients_by_name())

def block_shift_constraint(cns, iname, multiple, as_equality=None):
    cns = copy_constraint(cns, as_equality=as_equality)
    cns = cns.set_constant(cns.get_constant()
            + cns.get_coefficients_by_name()[iname]*multiple)
    return cns

def negate_constraint(cns):
    assert not cns.is_equality()
    # FIXME hackety hack
    my_set = (isl.BasicSet.universe(cns.get_dim())
            .add_constraint(cns))
    my_set = my_set.complement()

    results = []
    def examine_basic_set(s):
        s.foreach_constraint(results.append)
    my_set.foreach_basic_set(examine_basic_set)
    result, = results
    return result

def copy_constraint(cns, as_equality=None):
    return cast_constraint_to_space(cns, cns.get_dim(),
            as_equality=as_equality)

def get_dim_bounds(set, inames):
    vars = set.get_dim().get_var_dict(dim_type.set).keys()
    return [get_projected_bounds(set, v) for v in inames]

def count_box_from_bounds(bounds):
    from pytools import product
    return product(stop-start for start, stop in bounds)

def make_index_map(set, index_expr):
    if not isinstance(index_expr, tuple):
        index_expr = (index_expr,)

    amap = isl.Map.from_domain(set).add_dims(dim_type.out, len(index_expr))
    out_names = ["_ary_idx_%d" % i for i in range(len(index_expr))]

    dim = amap.get_dim()
    all_constraints = tuple(
            eq_constraint_from_expr(dim, iexpr_i)
            for iexpr_i in index_expr)

    for i, out_name in enumerate(out_names):
        amap = amap.set_dim_name(dim_type.out, i, out_name)

    for i, (out_name, constr) in enumerate(zip(out_names, all_constraints)):
        constr.set_coefficients_by_name({out_name: -1})
        amap = amap.add_constraint(constr)

    return amap

def make_slab(space, iname, start, stop):
    from pymbolic import var
    var_iname = var(iname)
    return (isl.Set.universe(space)
            # start <= inner
            .add_constraint(ineq_constraint_from_expr(
                space, var_iname -start))
            # inner < stop
            .add_constraint(ineq_constraint_from_expr(
                space, stop-1 - var_iname)))

def constraint_to_expr(cns, except_name=None):
    excepted_coeff = 0
    result = 0
    from pymbolic import var
    for var_name, coeff in cns.get_coefficients_by_name().iteritems():
        if isinstance(var_name, str):
            if var_name == except_name:
                excepted_coeff = int(coeff)
            else:
                result += int(coeff)*var(var_name)
        else:
            result += int(coeff)

    if except_name is not None:
        return result, excepted_coeff
    else:
        return result

# }}}

# vim: foldmethod=marker

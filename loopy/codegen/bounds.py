from __future__ import division

import islpy as isl
from islpy import dim_type
import numpy as np




# {{{ find bounds from set

def get_bounds_constraints(set, iname, admissible_inames, allow_parameters):
    if admissible_inames is not None or not allow_parameters:
        if admissible_inames is None:
            proj_type = []
        else:
            assert iname in admissible_inames
            proj_type = [dim_type.set]

        if not allow_parameters:
            proj_type.append(dim_type.param)

        set = set.eliminate_except(admissible_inames, proj_type)

    basic_sets = set.get_basic_sets()
    if len(basic_sets) > 1:
        set = set.coalesce()
        basic_sets = set.get_basic_sets()
        if len(basic_sets) > 1:
            raise RuntimeError("got non-convex set in bounds generation")

    bset, = basic_sets

    # FIXME perhaps use some form of hull here if there's more than one
    # basic set?

    lower = []
    upper = []
    equality = []

    space = bset.get_space()

    var_dict = space.get_var_dict()
    iname_tp, iname_idx = var_dict[iname]

    for cns in bset.get_constraints():
        assert not cns.is_div_constraint()

        iname_coeff = int(cns.get_coefficient(iname_tp, iname_idx))

        if iname_coeff == 0:
            continue

        if cns.is_equality():
            equality.append(cns)
        elif iname_coeff < 0:
            upper.append(cns)
        else: #  iname_coeff > 0
            lower.append(cns)

    return lower, upper, equality

def solve_constraint_for_bound(cns, iname):
    from warnings import warn
    warn("solve_constraint_for_bound deprecated?")

    from loopy.symbolic import constraint_to_expr
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

def get_bounds(set, iname, admissible_inames, allow_parameters):
    """Get an overapproximation of the loop bounds for the variable *iname*,
    as actual bounds.
    """

    from warnings import warn
    warn("deprecated")

    lower, upper, equality = get_bounds_constraints(
            set, iname, admissible_inames, allow_parameters)

    def do_solve(cns_list, assert_kind):
        result = []
        for cns in cns_list:
            kind, bound = solve_constraint_for_bound(cns, iname)
            assert kind == assert_kind
            result.append(bound)

        return result

    lower_bounds = do_solve(lower, ">=")
    upper_bounds = do_solve(upper, "<")
    equalities = do_solve(equality, "==")

    def agg_if_more_than_one(descr, agg_func, l):
        if len(l) == 0:
            raise ValueError("no %s bound found for '%s'" % (descr, iname))
        elif len(l) == 1:
            return l[0]
        else:
            return agg_func(l)

    from pymbolic.primitives import Min, Max
    return (agg_if_more_than_one("lower", Max, lower_bounds),
            agg_if_more_than_one("upper", Min, upper_bounds),
            equalities)

# }}}

# {{{ bounds check generator

def constraint_to_code(ccm, cns):
    if cns.is_equality():
        comp_op = "=="
    else:
        comp_op = ">="

    from loopy.symbolic import constraint_to_expr
    return "%s %s 0" % (ccm(constraint_to_expr(cns)), comp_op)

def filter_necessary_constraints(implemented_domain, constraints):
    space = implemented_domain.get_space()
    return [cns
        for cns in constraints
        if not implemented_domain.is_subset(
            isl.Set.universe(space).add_constraint(cns))]

def generate_bounds_checks(domain, check_vars, implemented_domain):
    domain_bset, = (domain
            .eliminate_except(check_vars, [dim_type.set])
            .coalesce()
            .get_basic_sets())

    return filter_necessary_constraints(
            implemented_domain, domain_bset.get_constraints())

def generate_bounds_checks_code(ccm, domain, check_vars, implemented_domain):
    return [constraint_to_code(ccm, cns) for cns in 
            generate_bounds_checks(domain, check_vars, implemented_domain)]

def wrap_in_bounds_checks(ccm, domain, check_vars, implemented_domain, stmt):
    from loopy.codegen import wrap_in_if
    return wrap_in_if(
            generate_bounds_checks_code(ccm, domain, check_vars,
                implemented_domain),
            stmt)

def wrap_in_for_from_constraints(ccm, iname, constraint_bset, stmt):
    # FIXME add admissible vars
    if isinstance(constraint_bset, isl.Set):
        constraint_bset, = constraint_bset.get_basic_sets()

    constraints = constraint_bset.get_constraints()

    from pymbolic import expand
    from pymbolic.mapper.constant_folder import CommutativeConstantFoldingMapper

    cfm = CommutativeConstantFoldingMapper()

    from loopy.symbolic import constraint_to_expr

    start_exprs = []
    end_conds = []
    equality_exprs = []

    for cns in constraints:
        rhs, iname_coeff = constraint_to_expr(cns, except_name=iname)

        if iname_coeff == 0:
            continue

        if cns.is_equality():
            kind, bound = solve_constraint_for_bound(cns, iname)
            assert kind == "=="
            equality_exprs.append(bound)
        elif iname_coeff < 0:
            from pymbolic import var
            rhs += iname_coeff*var(iname)
            end_conds.append("%s >= 0" %
                    ccm(cfm(expand(rhs))))
        else: #  iname_coeff > 0
            kind, bound = solve_constraint_for_bound(cns, iname)
            assert kind == ">="
            start_exprs.append(bound)

    if equality_exprs:
        assert len(equality_exprs) == 1

        equality_expr, = equality_exprs

        from loopy.codegen import gen_code_block
        from cgen import Initializer, POD, Const, Line
        return gen_code_block([
            Initializer(Const(POD(np.int32, iname)),
                ccm(equality_expr)),
            Line(),
            stmt,
            ])
    else:
        if len(start_exprs) > 1:
            from pymbolic.primitives import Max
            start_expr = Max(start_exprs)
        elif len(start_exprs) == 1:
            start_expr, = start_exprs
        else:
            raise RuntimeError("no starting value found for 'for' loop in '%s'"
                    % iname)

        from cgen import For
        from loopy.codegen import wrap_in
        return wrap_in(For,
                "int %s = %s" % (iname, ccm(start_expr)),
                " && ".join(end_conds),
                "++%s" % iname,
                stmt)

# }}}

# {{{ on which variables may a conditional depend?

def get_defined_inames(kernel, sched_index, allow_tag_classes=()):
    """
    :param exclude_tags: a tuple of tag classes to exclude
    """
    from loopy.schedule import EnterLoop, LeaveLoop

    result = set()

    for i, sched_item in enumerate(kernel.schedule):
        if i >= sched_index:
            break
        if isinstance(sched_item, EnterLoop):
            result.add(sched_item.iname)
        elif isinstance(sched_item, LeaveLoop):
            result.remove(sched_item.iname)

    for iname in kernel.all_inames():
        tag = kernel.iname_to_tag.get(iname)

        if isinstance(tag, allow_tag_classes):
            result.add(iname)

    return result

# }}}

# {{{

def pick_simple_constraint(constraints, iname):
    if len(constraints) == 0:
        raise RuntimeError("no constraint for '%s'" % iname)
    elif len(constraints) == 1:
        return constraints[0]

    from pymbolic.mapper.flop_counter import FlopCounter
    count_flops = FlopCounter()

    from pytools import argmin2
    return argmin2(
            (cns, count_flops(solve_constraint_for_bound(cns, iname)[1]))
            for cns in constraints)




# vim: foldmethod=marker

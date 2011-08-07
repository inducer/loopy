from __future__ import division

import islpy as isl
from islpy import dim_type




# {{{ bounds check generator

def constraint_to_code(ccm, cns):
    if cns.is_equality():
        comp_op = "=="
    else:
        comp_op = ">="

    from loopy.isl import constraint_to_expr
    return "%s %s 0" % (ccm(constraint_to_expr(cns)), comp_op)

def filter_necessary_constraints(implemented_domain, constraints):
    space = implemented_domain.get_dim()
    return [cns
        for cns in constraints
        if not implemented_domain.is_subset(
            isl.Set.universe(space)
            .add_constraint(cns))]

def generate_bounds_checks(ccm, domain, check_vars, implemented_domain):
    domain_bset, = domain.get_basic_sets()

    projected_domain_bset = isl.project_out_except(
            domain_bset, check_vars, [dim_type.set])

    space = domain.get_dim()

    cast_constraints = []

    from loopy.isl import cast_constraint_to_space

    def examine_constraint(cns):
        assert not cns.is_div_constraint()
        cast_constraints.append(
                cast_constraint_to_space(cns, space))

    projected_domain_bset.foreach_constraint(examine_constraint)

    necessary_constraints = filter_necessary_constraints(
            implemented_domain, cast_constraints)

    return [constraint_to_code(ccm, cns) for cns in necessary_constraints]

def wrap_in_bounds_checks(ccm, domain, check_vars, implemented_domain, stmt):
    from loopy.codegen import wrap_in_if
    return wrap_in_if(
            generate_bounds_checks(ccm, domain, check_vars,
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

    from loopy.isl import constraint_to_expr, solve_constraint_for_bound
    from pytools import any

    if any(cns.is_equality() for cns in constraints):
        raise NotImplementedError("equality constraints for 'for' loops")
    else:
        start_exprs = []
        end_conds = []

        for cns in constraints:
            rhs, iname_coeff = constraint_to_expr(cns, except_name=iname)

            if iname_coeff == 0:
                continue
            elif iname_coeff < 0:
                from pymbolic import var
                rhs += iname_coeff*var(iname)
                end_conds.append("%s >= 0" %
                        ccm(cfm(expand(rhs))))
            else: #  iname_coeff > 0
                kind, bound = solve_constraint_for_bound(cns, iname)
                assert kind == ">="
                start_exprs.append(bound)

    while len(start_exprs) >= 2:
        start_exprs.append(
                "max(%s, %s)" % (
                    ccm(start_exprs.pop()),
                    ccm(start_exprs.pop())))

    start_expr, = start_exprs # there has to be at least one

    from cgen import For
    from loopy.codegen import wrap_in
    return wrap_in(For,
            "int %s = %s" % (iname, start_expr),
            " && ".join(end_conds),
            "++%s" % iname,
            stmt)

# }}}




# vim: foldmethod=marker

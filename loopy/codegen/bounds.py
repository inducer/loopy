from __future__ import division

import islpy as isl
from islpy import dim_type




# {{{ find bounds from set

def get_bounds_constraints(set, iname, admissible_inames, allow_parameters):
    """May overapproximate."""
    if admissible_inames is not None or not allow_parameters:
        if admissible_inames is None:
            elim_type = []
        else:
            assert iname in admissible_inames
            elim_type = [dim_type.set]

        if not allow_parameters:
            elim_type.append(dim_type.param)

        set = set.eliminate_except(admissible_inames, elim_type)
        set = set.remove_divs()

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


    from pytools import any
    if any(cns.is_div_constraint() for cns in bset.get_constraints()):
        bset = bset.remove_divs()

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

# }}}

# {{{ bounds check generator

def constraint_to_code(ccm, cns):
    if cns.is_equality():
        comp_op = "=="
    else:
        comp_op = ">="

    from loopy.symbolic import constraint_to_expr
    return "%s %s 0" % (ccm(constraint_to_expr(cns), 'i'), comp_op)

def generate_bounds_checks(domain, check_inames, implemented_domain):
    """Will not overapproximate."""

    domain = (domain
            .eliminate_except(check_inames, [dim_type.set])
            .compute_divs())

    if isinstance(domain, isl.Set):
        bsets = domain.get_basic_sets()
        if len(bsets) != 1:
            domain = domain.coalesce()
            bsets = domain.get_basic_sets()
            if len(bsets) != 1:
                raise RuntimeError("domain of inames '%s' projected onto '%s' "
                        "did not reduce to a single conjunction"
                        % (", ".join(domain.get_var_names(dim_type.set)),
                            check_inames))

        domain, = bsets
    else:
        domain = domain

    domain = domain.remove_redundancies()
    domain = isl.Set.from_basic_set(domain)
    domain = isl.align_spaces(domain, implemented_domain)

    result = domain.gist(implemented_domain)

    from loopy.isl_helpers import convexify
    return convexify(result).get_constraints()

def wrap_in_bounds_checks(ccm, domain, check_inames, implemented_domain, stmt):
    bounds_checks = generate_bounds_checks(
            domain, check_inames,
            implemented_domain)

    bounds_check_set = isl.Set.universe(domain.get_space()).add_constraints(bounds_checks)
    bounds_check_set, new_implemented_domain = isl.align_two(
            bounds_check_set, implemented_domain)
    new_implemented_domain = new_implemented_domain & bounds_check_set

    condition_codelets = [
            constraint_to_code(ccm, cns) for cns in
            generate_bounds_checks(domain, check_inames, implemented_domain)]

    if condition_codelets:
        from cgen import If
        stmt = If("\n&& ".join(condition_codelets), stmt)

    return stmt, new_implemented_domain

def wrap_in_for_from_constraints(ccm, iname, constraint_bset, stmt,
        index_dtype):
    # FIXME add admissible vars
    if isinstance(constraint_bset, isl.Set):
        constraint_bset, = constraint_bset.get_basic_sets()

    constraints = constraint_bset.get_constraints()

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
                    ccm(cfm(rhs), 'i'))
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
            Initializer(Const(POD(index_dtype, iname)),
                ccm(equality_expr, 'i')),
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
                "int %s = %s" % (iname, ccm(start_expr, 'i')),
                " && ".join(end_conds),
                "++%s" % iname,
                stmt)

# }}}

# {{{ on which variables may a conditional depend?

def get_defined_inames(kernel, sched_index):
    from loopy.schedule import EnterLoop, LeaveLoop
    from loopy.kernel import ParallelTag

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

        # these are always defined
        if isinstance(tag, ParallelTag):
            result.add(iname)

    return frozenset(result)

# }}}

# {{{ get_simple_loop_bounds

def get_simple_loop_bounds(kernel, sched_index, iname, implemented_domain, iname_domain):
    from loopy.codegen.bounds import get_bounds_constraints, get_defined_inames
    lower_constraints_orig, upper_constraints_orig, equality_constraints_orig = \
            get_bounds_constraints(iname_domain, iname,
                    frozenset([iname])
                    | get_defined_inames(kernel, sched_index+1),
                    allow_parameters=True)

    lower_constraints_orig.extend(equality_constraints_orig)
    upper_constraints_orig.extend(equality_constraints_orig)
    #assert not equality_constraints_orig

    from loopy.codegen.bounds import pick_simple_constraint
    lb_cns_orig = pick_simple_constraint(lower_constraints_orig, iname)
    ub_cns_orig = pick_simple_constraint(upper_constraints_orig, iname)

    return lb_cns_orig, ub_cns_orig

# }}}

# {{{ pick_simple_constraint

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

# }}}







# vim: foldmethod=marker

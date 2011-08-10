from __future__ import division

import islpy as isl
from islpy import dim_type





# {{{ isl helpers

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

def make_index_map(set, index_expr):
    from loopy.symbolic import eq_constraint_from_expr

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
    from loopy.symbolic import ineq_constraint_from_expr
    from pymbolic import var
    var_iname = var(iname)
    return (isl.Set.universe(space)
            # start <= inner
            .add_constraint(ineq_constraint_from_expr(
                space, var_iname -start))
            # inner < stop
            .add_constraint(ineq_constraint_from_expr(
                space, stop-1 - var_iname)))

# }}}

# vim: foldmethod=marker

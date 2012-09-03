from __future__ import division

import islpy as isl
from islpy import dim_type




def constraint_to_code(ccm, cns):
    if cns.is_equality():
        comp_op = "=="
    else:
        comp_op = ">="

    from loopy.symbolic import constraint_to_expr
    return "%s %s 0" % (ccm(constraint_to_expr(cns), 'i'), comp_op)

# {{{ bounds check generator

def get_bounds_checks(domain, check_inames, implemented_domain,
        overapproximate):
    if isinstance(domain, isl.BasicSet):
        domain = isl.Set.from_basic_set(domain)
    domain = domain.remove_redundancies()
    result = domain.eliminate_except(check_inames, [dim_type.set])

    if overapproximate:
        # This is ok, because we're really looking for the
        # projection, with no remaining constraints from
        # the eliminated variables.
        result = result.remove_divs()
    else:
        result = result.compute_divs()

    result = isl.align_spaces(result, implemented_domain)
    result = result.gist(implemented_domain)

    if overapproximate:
        result = result.remove_divs()
    else:
        result = result.compute_divs()

    from loopy.isl_helpers import convexify
    result = convexify(result).get_constraints()
    return result

# }}}

# {{{ on which inames may a conditional depend?

def get_usable_inames_for_conditional(kernel, sched_index):
    from loopy.schedule import EnterLoop, LeaveLoop
    from loopy.kernel import ParallelTag, LocalIndexTagBase

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

        # Parallel inames are always defined, BUT local indices may not be used
        # in conditionals that cross barriers.

        if (isinstance(tag, ParallelTag)
                and not isinstance(tag, LocalIndexTagBase)):
            result.add(iname)

    return frozenset(result)

# }}}





# vim: foldmethod=marker

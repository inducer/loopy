from __future__ import division

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


import islpy as isl
from islpy import dim_type


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

    result, implemented_domain = isl.align_two(result, implemented_domain)
    result = result.gist(implemented_domain)

    if overapproximate:
        result = result.remove_divs()
    else:
        result = result.compute_divs()

    from loopy.isl_helpers import convexify
    result = convexify(result)
    return result.get_constraints()

# }}}


# {{{ on which inames may a conditional depend?

def get_usable_inames_for_conditional(kernel, sched_index):
    from loopy.schedule import EnterLoop, LeaveLoop
    from loopy.kernel.data import ParallelTag, LocalIndexTagBase, IlpBaseTag

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

        # Parallel inames are always defined, BUT:
        #
        # - local indices may not be used in conditionals that cross barriers.
        #
        # - ILP indices are not available in loop bounds, they only get defined
        #   at the innermost level of nesting.

        if (
                isinstance(tag, ParallelTag)
                and not isinstance(tag, LocalIndexTagBase)
                and not isinstance(tag, IlpBaseTag)
                ):
            result.add(iname)

    return frozenset(result)

# }}}


# vim: foldmethod=marker

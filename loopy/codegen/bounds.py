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


# {{{ approximate, convex bounds check generator

def get_approximate_convex_bounds_checks(domain, check_inames, implemented_domain):
    if isinstance(domain, isl.BasicSet):
        domain = isl.Set.from_basic_set(domain)
    domain = domain.remove_redundancies()
    result = domain.eliminate_except(check_inames, [dim_type.set])

    # This is ok, because we're really looking for the
    # projection, with no remaining constraints from
    # the eliminated variables.
    result = result.remove_divs()

    result, implemented_domain = isl.align_two(result, implemented_domain)
    result = result.gist(implemented_domain)

    # (see above)
    result = result.remove_divs()

    from loopy.isl_helpers import convexify
    result = convexify(result)
    return result.get_constraints()

# }}}


# {{{ on which inames may a conditional depend?

def get_usable_inames_for_conditional(kernel, sched_index):
    from loopy.schedule import (
        find_active_inames_at, get_insn_ids_for_block_at, has_barrier_within)
    from loopy.kernel.data import ParallelTag, LocalIndexTagBase, IlpBaseTag

    result = find_active_inames_at(kernel, sched_index)
    crosses_barrier = has_barrier_within(kernel, sched_index)

    # Find our containing subkernel. Grab inames for all insns from there.
    within_subkernel = False

    for sched_item_index, sched_item in enumerate(kernel.schedule[:sched_index+1]):
        from loopy.schedule import CallKernel, ReturnFromKernel
        if isinstance(sched_item, CallKernel):
            within_subkernel = True
            subkernel_index = sched_item_index
        elif isinstance(sched_item, ReturnFromKernel):
            within_subkernel = False

    if not within_subkernel:
        # Outside all subkernels - use only inames available to host.
        return frozenset(result)

    insn_ids_for_subkernel = get_insn_ids_for_block_at(
        kernel.schedule, subkernel_index)

    inames_for_subkernel = (
        iname
        for insn in insn_ids_for_subkernel
        for iname in kernel.insn_inames(insn))

    for iname in inames_for_subkernel:
        tag = kernel.iname_to_tag.get(iname)

        # Parallel inames are defined within a subkernel, BUT:
        #
        # - local indices may not be used in conditionals that cross barriers.
        #
        # - ILP indices are not available in loop bounds, they only get defined
        #   at the innermost level of nesting.

        if (
                isinstance(tag, ParallelTag)
                and not (isinstance(tag, LocalIndexTagBase) and crosses_barrier)
                and not isinstance(tag, IlpBaseTag)
                ):
            result.add(iname)

    return frozenset(result)

# }}}


# vim: foldmethod=marker

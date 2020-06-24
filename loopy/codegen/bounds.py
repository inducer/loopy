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

def get_usable_inames_for_conditional(kernel, sched_indices):

    from loopy.schedule import (
        find_active_inames_at, get_insn_ids_for_block_at, has_barrier_within,
        get_subkernel_indices)
    from loopy.kernel.data import (ConcurrentTag, LocalIndexTagBase,
                                   VectorizeTag,
                                   IlpBaseTag)
    active_inames_list = find_active_inames_at(kernel, sched_indices)
    crosses_barrier_list = has_barrier_within(kernel, sched_indices)
    # Find our containing subkernel. Grab inames for all insns from there.
    subkernel_index_list = get_subkernel_indices(kernel, sched_indices)

    inames_for_subkernel = {}

    for subknl_idx in set(idx for idx in subkernel_index_list if idx is not None):
        insn_ids_for_subkernel = get_insn_ids_for_block_at(
                kernel.schedule, subknl_idx)

        all_inames_in_the_subknl = set([
            iname
            for insn in insn_ids_for_subkernel
            for iname in kernel.insn_inames(insn)])

        def is_eligible_in_conditional(iname):
            # Parallel inames are defined within a subkernel, BUT:
            #
            # - ILP indices and vector lane indices are not available in loop
            #   bounds, they only get defined at the innermost level of nesting.
            return (
                    kernel.iname_tags_of_type(iname, ConcurrentTag)
                    and not kernel.iname_tags_of_type(iname, VectorizeTag)
                    and not kernel.iname_tags_of_type(iname, IlpBaseTag))

        inames_for_subkernel[subknl_idx] = [iname for iname in
                all_inames_in_the_subknl if is_eligible_in_conditional(iname)]

    result = []

    for active_inames, crosses_barrier, subknl_idx in zip(active_inames_list,
            crosses_barrier_list, subkernel_index_list):
        if subknl_idx is not None:
            for iname in inames_for_subkernel[subknl_idx]:
                # local indices may not be used in conditionals that cross barriers
                if (not (kernel.iname_tags_of_type(iname, LocalIndexTagBase)
                            and crosses_barrier)):
                    active_inames.add(iname)

        result.append(frozenset(active_inames))

    return result

# }}}


# vim: foldmethod=marker

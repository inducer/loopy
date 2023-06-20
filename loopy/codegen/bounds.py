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


from typing import FrozenSet
import islpy as isl
from islpy import dim_type
from loopy.codegen.tools import CodegenOperationCacheManager

from loopy.kernel import LoopKernel


# {{{ approximate, convex bounds check generator

def get_approximate_convex_bounds_checks(domain, check_inames,
        implemented_domain, op_cache_manager):
    if isinstance(domain, isl.BasicSet):
        domain = isl.Set.from_basic_set(domain)
    domain = domain.remove_redundancies()
    result = op_cache_manager.eliminate_except(domain, check_inames,
            (dim_type.set,))

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

def get_usable_inames_for_conditional(
        kernel: LoopKernel, sched_index: int,
        op_cache_manager: CodegenOperationCacheManager) -> FrozenSet[str]:
    active_inames = op_cache_manager.active_inames[sched_index]
    crosses_barrier = op_cache_manager.has_barrier_within[sched_index]

    # Find our containing subkernel. Grab inames for all insns from there.
    subkernel_index = op_cache_manager.callkernel_index[sched_index]

    if subkernel_index is None:
        # Outside all subkernels - use only inames available to host.
        assert isinstance(active_inames, frozenset)
        return active_inames

    concurrent_inames_in_subkernel = (
            op_cache_manager.get_concurrent_inames_in_a_callkernel(
                subkernel_index))

    # not all parallel inames are usable:
    #  - local indices may not be used in conditionals that cross barriers.
    #  - ILP indices and vector lane indices are not available in loop
    #    bounds, they only get defined at the innermost level of nesting.
    from loopy.schedule import find_used_inames_within
    from loopy.kernel.data import VectorizeTag, LocalInameTagBase, IlpBaseTag
    usable_concurrent_inames_in_subkernel = frozenset(
            iname for iname in concurrent_inames_in_subkernel
            if (not (kernel.iname_tags_of_type(iname, LocalInameTagBase)
                         and crosses_barrier)
                and not kernel.iname_tags_of_type(iname, VectorizeTag)
                and not kernel.iname_tags_of_type(iname, IlpBaseTag))
            ) & find_used_inames_within(kernel, sched_index)

    return active_inames | usable_concurrent_inames_in_subkernel

# }}}


# vim: foldmethod=marker

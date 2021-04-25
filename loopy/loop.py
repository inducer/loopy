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
from loopy.translation_unit import for_each_kernel


def potential_loop_nest_map(kernel):
    """Returns a dictionary mapping inames to other inames that *could*
    be nested around them.

    * :seealso: :func:`loopy.schedule.loop_nest_map`
    * :seealso: :func:`loopy.schedule.find_loop_nest_around_map`
    """

    result = {}

    all_inames = kernel.all_inames()
    iname_to_insns = kernel.iname_to_insns()

    # examine pairs of all inames--O(n**2), I know.
    for inner_iname in all_inames:
        inner_result = set()
        for outer_iname in all_inames:
            if inner_iname == outer_iname:
                continue

            if iname_to_insns[inner_iname] <= iname_to_insns[outer_iname]:
                inner_result.add(outer_iname)

        if inner_result:
            result[inner_iname] = inner_result

    return result


@for_each_kernel
def merge_loop_domains(kernel):
    # FIXME: This should be moved to loopy.transforms.iname
    from loopy.kernel.tools import is_domain_dependent_on_inames

    while True:
        lnm = potential_loop_nest_map(kernel)
        parents_per_domain = kernel.parents_per_domain()
        all_parents_per_domain = kernel.all_parents_per_domain()

        iname_to_insns = kernel.iname_to_insns()

        new_domains = None

        for inner_iname, outer_inames in lnm.items():
            for outer_iname in outer_inames:
                # {{{ check if it's safe to merge

                inner_domain_idx = kernel.get_home_domain_index(inner_iname)
                outer_domain_idx = kernel.get_home_domain_index(outer_iname)

                if inner_domain_idx == outer_domain_idx:
                    break

                if (not iname_to_insns[inner_iname]
                        or not iname_to_insns[outer_iname]):
                    # Inames without instructions occur when used in
                    # a SubArrayRef. We don't want monster SubArrayRef domains,
                    # so refuse to merge those.
                    continue

                if iname_to_insns[inner_iname] != iname_to_insns[outer_iname]:
                    # The two inames are imperfectly nested. Domain fusion
                    # might be invalid when the inner loop is empty, leading to
                    # the outer loop also being empty.

                    # FIXME: Not fully correct, does not consider reductions
                    # https://gitlab.tiker.net/inducer/loopy/issues/172
                    continue

                if (
                        outer_domain_idx in all_parents_per_domain[inner_domain_idx]
                        and not
                        outer_domain_idx == parents_per_domain[inner_domain_idx]):
                    # Outer domain is not a direct parent of the inner
                    # domain. Unable to merge.
                    continue

                outer_dom = kernel.domains[outer_domain_idx]
                inner_dom = kernel.domains[inner_domain_idx]

                outer_inames = set(outer_dom.get_var_names(isl.dim_type.set))
                if is_domain_dependent_on_inames(kernel, inner_domain_idx,
                        outer_inames):
                    # Bounds of inner domain depend on outer domain.
                    # Unable to merge.
                    continue

                # }}}

                new_domains = kernel.domains[:]
                min_idx = min(inner_domain_idx, outer_domain_idx)
                max_idx = max(inner_domain_idx, outer_domain_idx)

                del new_domains[max_idx]
                del new_domains[min_idx]

                outer_dom, inner_dom = isl.align_two(outer_dom, inner_dom)

                new_domains.insert(min_idx, inner_dom & outer_dom)
                break

            if new_domains:
                break

        if not new_domains:
            # Nothing was accomplished in the last loop trip, time to quit.
            break

        kernel = kernel.copy(domains=new_domains)

    return kernel


# vim: fdm=marker

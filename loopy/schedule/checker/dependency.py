__copyright__ = "Copyright (C) 2019 James Stevens"

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


class DependencyType:
    """Strings specifying a particular type of dependency relationship.

    .. attribute:: SAME

       A :class:`str` specifying the following dependency relationship:

       If ``S = {i, j, ...}`` is a set of inames used in both statements
       ``insn0`` and ``insn1``, and ``{i', j', ...}`` represent the values
       of the inames in ``insn0``, and ``{i, j, ...}`` represent the
       values of the inames in ``insn1``, then the dependency
       ``insn0 happens before insn1 iff SAME({i, j})`` specifies that
       ``insn0 happens before insn1 iff {i' = i and j' = j and ...}``.
       Note that ``SAME({}) = True``.

    .. attribute:: PRIOR

       A :class:`str` specifying the following dependency relationship:

       If ``S = {i, j, k, ...}`` is a set of inames used in both statements
       ``insn0`` and ``insn1``, and ``{i', j', k', ...}`` represent the values
       of the inames in ``insn0``, and ``{i, j, k, ...}`` represent the
       values of the inames in ``insn1``, then the dependency
       ``insn0 happens before insn1 iff PRIOR({i, j, k})`` specifies one of
       two possibilities, depending on whether the loop nest ordering is
       known. If the loop nest ordering is unknown, then
       ``insn0 happens before insn1 iff {i' < i and j' < j and k' < k ...}``.
       If the loop nest ordering is known, the condition becomes
       ``{i', j', k', ...}`` is lexicographically less than ``{i, j, k, ...}``,
       i.e., ``i' < i or (i' = i and j' < j) or (i' = i and j' = j and k' < k) ...``.

    """

    SAME = "same"
    PRIOR = "prior"


def filter_deps_by_intersection_with_SAME(knl):
    # Determine which dep relations have a non-empty intersection with
    # the SAME relation
    # TODO document

    from loopy.schedule.checker.utils import (
        append_mark_to_strings,
        partition_inames_by_concurrency,
        create_elementwise_comparison_conjunction_set,
        convert_map_to_set,
        convert_set_back_to_map,
    )
    from loopy.schedule.checker.schedule import (
        BEFORE_MARK,
    )
    _, non_conc_inames = partition_inames_by_concurrency(knl)

    # NOTE: deps filtered will map depender->dependee
    deps_filtered = {}
    for stmt in knl.instructions:

        if hasattr(stmt, "dependencies") and stmt.dependencies:

            depender_id = stmt.id

            for dependee_id, dep_maps in stmt.dependencies.items():

                # Continue if we've been told to ignore this dependee
                # (non_linearizing_deps is only an attribute of stmt in one
                # (unmerged) branch, and may be eliminated)
                if (
                        hasattr(stmt, "non_linearizing_deps") and
                        stmt.non_linearizing_deps is not None and
                        dependee_id in stmt.non_linearizing_deps):
                    continue

                # Continue if we already have this pair
                if depender_id in deps_filtered.keys() and (
                        dependee_id in deps_filtered[depender_id]):
                    continue

                for dep_map in dep_maps:
                    # Create isl map representing "SAME" dep for these two insns

                    # Get shared nonconcurrent inames
                    depender_inames = knl.id_to_insn[depender_id].within_inames
                    dependee_inames = knl.id_to_insn[dependee_id].within_inames
                    shared_nc_inames = (
                        depender_inames & dependee_inames & non_conc_inames)

                    # Temporarily convert to set
                    dep_set_space, n_in_dims, n_out_dims = convert_map_to_set(
                        dep_map.space)

                    # Create SAME relation
                    same_set_affs = isl.affs_from_space(dep_set_space)
                    same_set = create_elementwise_comparison_conjunction_set(
                        shared_nc_inames,
                        append_mark_to_strings(shared_nc_inames, BEFORE_MARK),
                        same_set_affs)

                    # Convert back to map
                    same_map = convert_set_back_to_map(
                        same_set, n_in_dims, n_out_dims)

                    # Don't need to intersect same_map with iname bounds (I think..?)

                    # See whether the intersection of dep map and SAME is empty
                    intersect_dep_and_same = same_map & dep_map
                    intersect_not_empty = not bool(intersect_dep_and_same.is_empty())

                    if intersect_not_empty:
                        deps_filtered.setdefault(depender_id, set()).add(dependee_id)
                        break  # No need to check any more deps for this pair

    return deps_filtered

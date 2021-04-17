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
from loopy.schedule.checker.schedule import BEFORE_MARK
dt = isl.dim_type


class LegacyDependencyType:
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


def create_elementwise_comparison_conjunction_set(
        names0, names1, islvars, op="eq"):
    """Create a set constrained by the conjunction of conditions comparing
       `names0` to `names1`.

    :arg names0: A list of :class:`str` representing variable names.

    :arg names1: A list of :class:`str` representing variable names.

    :arg islvars: A dictionary from variable names to :class:`islpy.PwAff`
        instances that represent each of the variables
        (islvars may be produced by `islpy.make_zero_and_vars`). The key
        '0' is also include and represents a :class:`islpy.PwAff` zero constant.

    :arg op: A :class:`str` describing the operator to use when creating
        the set constraints. Options: `eq` for `=`, `lt` for `<`

    :returns: A set involving `islvars` cosntrained by the constraints
        `{names0[0] <op> names1[0] and names0[1] <op> names1[1] and ...}`.

    """

    # initialize set with constraint that is always true
    conj_set = islvars[0].eq_set(islvars[0])
    for n0, n1 in zip(names0, names1):
        if op == "eq":
            conj_set = conj_set & islvars[n0].eq_set(islvars[n1])
        elif op == "lt":
            conj_set = conj_set & islvars[n0].lt_set(islvars[n1])

    return conj_set


def _convert_constraint_set_to_map(constraint_set, mv_count, src_position=None):
    constraint_map = isl.Map.from_domain(constraint_set)
    if src_position:
        return constraint_map.move_dims(
            dt.out, 0, dt.in_, src_position, mv_count)
    else:
        return constraint_map.move_dims(
            dt.out, 0, dt.in_, mv_count, mv_count)


def create_legacy_dependency_constraint(
        knl,
        insn_id_before,
        insn_id_after,
        deps,
        nests_outside_map=None,
        before_mark=BEFORE_MARK,
        ):
    """Create a statement dependency constraint represented as a map from
        each statement instance to statement instances that must occur later,
        i.e., ``{[s'=0, i', j'] -> [s=1, i, j] : condition on {i', j', i, j}}``
        indicates that statement ``0`` comes before statment ``1`` when the
        specified condition on inames ``i',j',i,j`` is met. ``i'`` and ``j'``
        are the values of inames ``i`` and ``j`` in first statement instance.

    :arg knl: A :class:`loopy.kernel.LoopKernel` containing the
        depender and dependee instructions.

    :arg insn_id_before: A :class:`str` specifying the :mod:`loopy`
        instruction id for the dependee statement.

    :arg insn_id_after: A :class:`str` specifying the :mod:`loopy`
        instruction id for the depender statement.

    :arg deps: A :class:`dict` mapping instances of :class:`LegacyDependencyType`
        to the :mod:`loopy` kernel inames involved in that particular
        dependency relationship.

    :returns: An :class:`islpy.Map` mapping each statement instance to all
        statement instances that must occur later according to the constraints.

    """

    from loopy.schedule.checker.utils import (
        make_islvars_with_mark,
        append_mark_to_strings,
        insert_and_name_isl_dims,
        reorder_dims_by_name,
        append_mark_to_isl_map_var_names,
        sorted_union_of_names_in_isl_sets,
    )
    from loopy.schedule.checker.schedule import STATEMENT_VAR_NAME
    # This function uses the dependency given to create the following constraint:
    # Statement [s,i,j] comes before statement [s',i',j'] iff <constraint>

    before_inames = knl.id_to_insn[insn_id_before].within_inames
    after_inames = knl.id_to_insn[insn_id_after].within_inames
    dom_before = knl.get_inames_domain(
        before_inames).project_out_except(before_inames, [dt.set])
    dom_after = knl.get_inames_domain(
        after_inames).project_out_except(after_inames, [dt.set])

    # create some (ordered) isl vars to use, e.g., {s, i, j, s', i', j'}
    dom_inames_ordered_before = sorted_union_of_names_in_isl_sets([dom_before])
    dom_inames_ordered_after = sorted_union_of_names_in_isl_sets([dom_after])
    islvars = make_islvars_with_mark(
        var_names_needing_mark=[STATEMENT_VAR_NAME]+dom_inames_ordered_before,
        other_var_names=[STATEMENT_VAR_NAME]+dom_inames_ordered_after,
        mark=before_mark,
        )
    statement_var_name_prime = STATEMENT_VAR_NAME+before_mark

    # initialize constraints to False
    # this will disappear as soon as we add a constraint
    all_constraints_set = islvars[0].eq_set(islvars[0] + 1)

    # for each (dep_type, inames) pair, create 'happens before' constraint,
    # all_constraints_set will be the union of all these constraints
    ldt = LegacyDependencyType
    for dep_type, inames in deps.items():
        # need to put inames in a list so that order of inames and inames'
        # matches when calling create_elementwise_comparison_conj...
        if not isinstance(inames, list):
            inames_list = list(inames)
        else:
            inames_list = inames[:]
        inames_prime = append_mark_to_strings(inames_list, before_mark)  # [j', k']

        if dep_type == ldt.SAME:
            # TODO test/handle case where inames list is empty (stmt0->stmt1 if true)
            constraint_set = create_elementwise_comparison_conjunction_set(
                    inames_prime, inames_list, islvars, op="eq")
        elif dep_type == ldt.PRIOR:

            priority_known = False
            # if nesting info is provided:
            if nests_outside_map:
                # assumes all loop_priority tuples are consistent

                # with multiple priority tuples, determine whether the combined
                # info they contain can give us a single, full proiritization,
                # e.g., if prios={(a, b), (b, c), (c, d, e)}, then we know
                # a -> b -> c -> d -> e

                # before reasoning about loop orderings,
                # remove irrelevant inames from nesting requirements
                # TODO more efficient way to do this?
                relevant_nests_outside_map = {}
                for iname, inside_inames in nests_outside_map.items():
                    # ignore irrelevant iname keys
                    if iname in inames_list:
                        # only keep relevant subset of iname vals
                        relevant_nests_outside_map[iname] = set(
                            inames_list) & nests_outside_map[iname]

                # get all orderings that are explicitly allowed by priorities
                from loopy.schedule.checker.utils import (
                    get_orderings_of_length_n)
                orders = get_orderings_of_length_n(
                    relevant_nests_outside_map,
                    required_length=len(inames_list),
                    #return_first_found=True,
                    return_first_found=False,  # slower; allows priorities test below
                    )

                if orders:
                    # test for invalid priorities (includes cycles)
                    if len(orders) != 1:
                        raise ValueError(
                            "create_dependency_constriant encountered invalid "
                            "priorities %s"
                            % (knl.loop_priority))
                    priority_known = True
                    priority_tuple = orders.pop()

            # if only one loop, we know the priority
            if not priority_known and len(inames_list) == 1:
                priority_tuple = tuple(inames_list)
                priority_known = True

            if priority_known:
                # PRIOR requires statement before complete previous iterations
                # of loops before statement after completes current iteration
                # according to loop nest order
                inames_list_nest_ordered = [
                    iname for iname in priority_tuple
                    if iname in inames_list]

                from loopy.schedule.checker import (
                    lexicographic_order_map as lom)

                constraint_set = lom.get_lex_order_set(
                    inames_list_nest_ordered,
                    before_mark,
                    islvars,
                    )
            else:  # priority not known
                # PRIOR requires upper left quadrant happen before:
                constraint_set = create_elementwise_comparison_conjunction_set(
                        inames_prime, inames_list, islvars, op="lt")

        # get ints representing statements in pairwise schedule
        s_before_int = 0
        s_after_int = 0 if insn_id_before == insn_id_after else 1

        # set statement_var_name == statement #
        constraint_set = constraint_set & islvars[statement_var_name_prime].eq_set(
            islvars[0]+s_before_int)
        constraint_set = constraint_set & islvars[STATEMENT_VAR_NAME].eq_set(
            islvars[0]+s_after_int)

        # union this constraint_set with all_constraints_set
        all_constraints_set = all_constraints_set | constraint_set

    # convert constraint set to map
    all_constraints_map = _convert_constraint_set_to_map(
        all_constraints_set,
        mv_count=len(dom_inames_ordered_after)+1,  # +1 for statement var
        src_position=len(dom_inames_ordered_before)+1,  # +1 for statement var
        )

    # now apply domain sets to constraint variables
    statement_var_idx = 0  # index of statement_var dimension in map
    # (anything other than 0 risks being out of bounds)

    # add statement variable to doms to enable intersection
    range_to_intersect = insert_and_name_isl_dims(
        dom_after, dt.out,
        [STATEMENT_VAR_NAME], statement_var_idx)
    domain_constraint_set = append_mark_to_isl_map_var_names(
        dom_before, dt.set, mark=before_mark)
    domain_to_intersect = insert_and_name_isl_dims(
        domain_constraint_set, dt.out,
        [statement_var_name_prime], statement_var_idx)

    # reorder inames to enable intersection (inames should already match at
    # this point)
    assert set(
        append_mark_to_strings(
            [STATEMENT_VAR_NAME] + dom_inames_ordered_before, before_mark)
        ) == set(domain_to_intersect.get_var_names(dt.out))
    assert set(
        [STATEMENT_VAR_NAME] + dom_inames_ordered_after
        ) == set(range_to_intersect.get_var_names(dt.out))
    domain_to_intersect = reorder_dims_by_name(
        domain_to_intersect, dt.out,
        append_mark_to_strings(
            [STATEMENT_VAR_NAME] + dom_inames_ordered_before, before_mark))
    range_to_intersect = reorder_dims_by_name(
        range_to_intersect,
        dt.out,
        [STATEMENT_VAR_NAME] + dom_inames_ordered_after)

    # intersect doms
    map_with_loop_domain_constraints = all_constraints_map.intersect_domain(
        domain_to_intersect).intersect_range(range_to_intersect)

    return map_with_loop_domain_constraints


def get_dependency_sources_and_sinks(knl, linearization_item_ids):
    """Implicitly create a directed graph with the linearization items specified
    by ``linearization_item_ids`` as nodes, and with edges representing a
    'happens before' relationship specfied by each legacy dependency between
    two instructions. Return the sources and sinks within this graph.

    :arg linearization_item_ids: A :class:`list` of :class:`str` representing
        loopy instruction ids.

    :returns: Two instances of :class:`set` of :class:`str` instruction ids
        representing the sources and sinks in the dependency graph.

    """
    sources = set()
    dependees = set()  # all dependees (within linearization_item_ids)
    for item_id in linearization_item_ids:
        # find the deps within linearization_item_ids
        deps = knl.id_to_insn[item_id].depends_on & linearization_item_ids
        if deps:
            # add deps to dependees
            dependees.update(deps)
        else:  # has no deps (within linearization_item_ids), this is a source
            sources.add(item_id)

    # sinks don't point to anyone
    sinks = linearization_item_ids - dependees

    # Note that some instructions may be both a source and a sink

    return sources, sinks

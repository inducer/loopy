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


class StatementPairDependencySet(object):
    """A set of dependencies between two statements.

    .. attribute:: statement_before

       A :class:`loopy.schedule.checker.schedule.StatementRef` depended
        on by statement_after.

    .. attribute:: statement_after

       A :class:`loopy.schedule.checker.schedule.StatementRef` which
        cdepends on statement_before.

    .. attribute:: deps

       A :class:`dict` mapping instances of :class:`DependencyType` to
       the :mod:`loopy` kernel inames involved in that particular
       dependency relationship.

    .. attribute:: dom_before

       A :class:`islpy.BasicSet` representing the domain for the
       dependee statement.

    .. attribute:: dom_after

       A :class:`islpy.BasicSet` representing the domain for the
       depender statement.

    """

    def __init__(
            self,
            statement_before,
            statement_after,
            deps,  # {dep_type: iname_set}
            dom_before=None,
            dom_after=None,
            ):
        self.statement_before = statement_before
        self.statement_after = statement_after
        self.deps = deps
        self.dom_before = dom_before
        self.dom_after = dom_after

    def __eq__(self, other):
        return (
            self.statement_before == other.statement_before
            and self.statement_after == other.statement_after
            and self.deps == other.deps
            and self.dom_before == other.dom_before
            and self.dom_after == other.dom_after
            )

    def __lt__(self, other):
        return self.__hash__() < other.__hash__()

    def __hash__(self):
        return hash(repr(self))

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        key_builder.rec(key_hash, self.statement_before)
        key_builder.rec(key_hash, self.statement_after)
        key_builder.rec(key_hash, self.deps)
        key_builder.rec(key_hash, self.dom_before)
        key_builder.rec(key_hash, self.dom_after)

    def __str__(self):
        result = "%s --before->\n%s iff\n    " % (
            self.statement_before, self.statement_after)
        return result + " and\n    ".join(
            ["(%s : %s)" % (dep_type, inames)
            for dep_type, inames in self.deps.items()])


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
    dim_type = isl.dim_type
    constraint_map = isl.Map.from_domain(constraint_set)
    if src_position:
        return constraint_map.move_dims(
            dim_type.out, 0, dim_type.in_, src_position, mv_count)
    else:
        return constraint_map.move_dims(
            dim_type.out, 0, dim_type.in_, mv_count, mv_count)


def create_dependency_constraint(
        statement_dep_set,
        loop_priorities,
        ):
    """Create a statement dependency constraint represented as a map from
        each statement instance to statement instances that must occur later,
        i.e., ``{[s'=0, i', j'] -> [s=1, i, j] : condition on {i', j', i, j}}``
        indicates that statement ``0`` comes before statment ``1`` when the
        specified condition on inames ``i',j',i,j`` is met. ``i'`` and ``j'``
        are the values of inames ``i`` and ``j`` in first statement instance.

    :arg statement_dep_set: A :class:`StatementPairDependencySet` describing
        the dependency relationship between the two statements.

    :arg loop_priorities: A list of tuples from the ``loop_priority``
        attribute of :class:`loopy.LoopKernel` specifying the loop nest
        ordering rules.

    :returns: An :class:`islpy.Map` mapping each statement instance to all
        statement instances that must occur later according to the constraints.

    """

    from loopy.schedule.checker.utils import (
        make_islvars_with_marker,
        append_apostrophes,
        add_dims_to_isl_set,
        insert_missing_dims_and_reorder_by_name,
        append_marker_to_isl_map_var_names,
        list_var_names_in_isl_sets,
    )
    from loopy.schedule.checker.schedule import STATEMENT_VAR_NAME
    # This function uses the dependency given to create the following constraint:
    # Statement [s,i,j] comes before statement [s',i',j'] iff <constraint>

    dom_inames_ordered_before = list_var_names_in_isl_sets(
        [statement_dep_set.dom_before])
    dom_inames_ordered_after = list_var_names_in_isl_sets(
        [statement_dep_set.dom_after])

    # create some (ordered) isl vars to use, e.g., {s, i, j, s', i', j'}
    islvars = make_islvars_with_marker(
        var_names_needing_marker=[STATEMENT_VAR_NAME]+dom_inames_ordered_before,
        other_var_names=[STATEMENT_VAR_NAME]+dom_inames_ordered_after,
        marker="'",
        )
    statement_var_name_prime = STATEMENT_VAR_NAME+"'"

    # initialize constraints to False
    # this will disappear as soon as we add a constraint
    all_constraints_set = islvars[0].eq_set(islvars[0] + 1)

    # for each (dep_type, inames) pair, create 'happens before' constraint,
    # all_constraints_set will be the union of all these constraints
    dt = DependencyType
    for dep_type, inames in statement_dep_set.deps.items():
        # need to put inames in a list so that order of inames and inames'
        # matches when calling create_elementwise_comparison_conj...
        if not isinstance(inames, list):
            inames_list = list(inames)
        else:
            inames_list = inames[:]
        inames_prime = append_apostrophes(inames_list)  # e.g., [j', k']

        if dep_type == dt.SAME:
            constraint_set = create_elementwise_comparison_conjunction_set(
                    inames_prime, inames_list, islvars, op="eq")
        elif dep_type == dt.PRIOR:

            priority_known = False
            # if nesting info is provided:
            if loop_priorities:
                # assumes all loop_priority tuples are consistent

                # with multiple priority tuples, determine whether the combined
                # info they contain can give us a single, full proiritization,
                # e.g., if prios={(a, b), (b, c), (c, d, e)}, then we know
                # a -> b -> c -> d -> e

                # remove irrelevant inames from priority tuples (because we're
                # about to perform a costly operation on remaining tuples)
                relevant_priorities = set()
                for p_tuple in loop_priorities:
                    new_tuple = [iname for iname in p_tuple if iname in inames_list]
                    # empty tuples and single tuples don't help us define
                    # a nesting, so ignore them (if we're dealing with a single
                    # iname, priorities will be ignored later anyway)
                    if len(new_tuple) > 1:
                        relevant_priorities.add(tuple(new_tuple))

                # create a mapping from each iname to inames that must be
                # nested inside that iname
                nested_inside = {}
                for outside_iname in inames_list:
                    nested_inside_inames = set()
                    for p_tuple in relevant_priorities:
                        if outside_iname in p_tuple:
                            nested_inside_inames.update([
                                inside_iname for inside_iname in
                                p_tuple[p_tuple.index(outside_iname)+1:]])
                    nested_inside[outside_iname] = nested_inside_inames

                from loopy.schedule.checker.utils import (
                    get_orderings_of_length_n)
                # get all orderings that are explicitly allowed by priorities
                orders = get_orderings_of_length_n(
                    nested_inside,
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
                            % (loop_priorities))
                    priority_known = True
                    priority_tuple = orders.pop()

            # if only one loop, we know the priority
            if not priority_known and len(inames_list) == 1:
                priority_tuple = tuple(inames_list)
                priority_known = True

            if priority_known:
                # PRIOR requires statement_before complete previous iterations
                # of loops before statement_after completes current iteration
                # according to loop nest order
                inames_list_nest_ordered = [
                    iname for iname in priority_tuple
                    if iname in inames_list]
                inames_list_nest_ordered_prime = append_apostrophes(
                    inames_list_nest_ordered)
                if set(inames_list_nest_ordered) != set(inames_list):
                    # TODO could this happen?
                    assert False

                from loopy.schedule.checker import (
                    lexicographic_order_map as lom)
                # TODO handle case where inames list is empty
                constraint_set = lom.get_lex_order_constraint(
                    inames_list_nest_ordered_prime,
                    inames_list_nest_ordered,
                    islvars,
                    )
            else:  # priority not known
                # PRIOR requires upper left quadrant happen before:
                constraint_set = create_elementwise_comparison_conjunction_set(
                        inames_prime, inames_list, islvars, op="lt")

        # get ints representing statements in PairwiseSchedule
        s_before_int = 0
        s_after_int = 0 if (
            statement_dep_set.statement_before.insn_id ==
            statement_dep_set.statement_after.insn_id
            ) else 1

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
    range_to_intersect = add_dims_to_isl_set(
        statement_dep_set.dom_after, isl.dim_type.out,
        [STATEMENT_VAR_NAME], statement_var_idx)
    domain_constraint_set = append_marker_to_isl_map_var_names(
        statement_dep_set.dom_before, isl.dim_type.set, marker="'")
    domain_to_intersect = add_dims_to_isl_set(
        domain_constraint_set, isl.dim_type.out,
        [statement_var_name_prime], statement_var_idx)

    # insert inames missing from doms to enable intersection
    domain_to_intersect = insert_missing_dims_and_reorder_by_name(
        domain_to_intersect, isl.dim_type.out,
        append_apostrophes([STATEMENT_VAR_NAME] + dom_inames_ordered_before))
    range_to_intersect = insert_missing_dims_and_reorder_by_name(
        range_to_intersect,
        isl.dim_type.out,
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

    return sources, sinks

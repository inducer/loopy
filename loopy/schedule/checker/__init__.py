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


# {{{ get pairwise statement orderings

def get_pairwise_statement_orderings(
        knl,
        lin_items,
        stmt_id_pairs,
        ):
    r"""For each statement pair in a subset of all statement pairs found in a
    linearized kernel, determine the (relative) order in which the statement
    instances are executed. For each pair, represent this relative ordering as
    a ``statement instance ordering`` (SIO): a map from each instance of the
    first statement to all instances of the second statement that occur
    later.

    :arg knl: A preprocessed :class:`loopy.kernel.LoopKernel` containing the
        linearization items that will be used to create the SIOs.

    :arg lin_items: A list of :class:`loopy.schedule.ScheduleItem`
        (to be renamed to `loopy.schedule.LinearizationItem`) containing all
        linearization items for which SIOs will be created. To allow usage of
        this routine during linearization, a truncated (i.e. partial)
        linearization may be passed through this argument.

    :arg stmt_id_pairs: A list containing pairs of statement identifiers.

    :returns: A dictionary mapping each two-tuple of statement identifiers
        provided in `stmt_id_pairs` to a :class:`collections.namedtuple`
        containing the intra-thread SIO (`sio_intra_thread`), intra-group SIO
        (`sio_intra_group`), and global SIO (`sio_global`), each realized
        as an :class:`islpy.Map` from each instance of the first
        statement to all instances of the second statement that occur later,
        as well as the intra-thread pairwise schedule (`pwsched_intra_thread`),
        intra-group pairwise schedule (`pwsched_intra_group`), and the global
        pairwise schedule (`pwsched_global`), each containing a pair of
        mappings from statement instances to points in a lexicographic
        ordering, one for each statement. Note that a pairwise schedule
        alone cannot be used to reproduce the corresponding SIO without the
        corresponding (unique) lexicographic order map, which is not returned.

    .. doctest:

        >>> import loopy as lp
        >>> import numpy as np
        >>> # Make kernel -----------------------------------------------------------
        >>> knl = lp.make_kernel(
        ...     "{[j,k]: 0<=j<pj and 0<=k<pk}",
        ...     [
        ...         "a[j] = j  {id=stmt_a}",
        ...         "b[k] = k+a[0]  {id=stmt_b,dep=stmt_a}",
        ...     ])
        >>> knl = lp.add_and_infer_dtypes(knl, {"a": np.float32, "b": np.float32})
        >>> # Get a linearization
        >>> knl = lp.get_one_linearized_kernel(lp.preprocess_kernel(knl))
        >>> # Get pairwise order info -----------------------------------------------
        >>> from loopy.schedule.checker import get_pairwise_statement_orderings
        >>> sio_dict = get_pairwise_statement_orderings(
        ...     knl,
        ...     knl.linearization,
        ...     [("stmt_a", "stmt_b")],
        ...     )
        >>> # Print map
        >>> print(str(sio_dict[("stmt_a", "stmt_b")].sio_intra_thread
        ...     ).replace("{ ", "{\n").replace(" :", "\n:"))
        [pj, pk] -> {
        [_lp_linchk_stmt' = 0, j'] -> [_lp_linchk_stmt = 1, k]
        : pj > 0 and pk > 0 and 0 <= j' < pj and 0 <= k < pk }

    """

    # {{{ make sure kernel has been preprocessed

    from loopy.kernel import KernelState
    assert knl.state in [
            KernelState.PREPROCESSED,
            KernelState.LINEARIZED]

    # }}}

    # {{{ Find any EnterLoop inames that are tagged as concurrent
    # so that get_pairwise_statement_orderings_inner() knows to ignore them
    # (In the future, this should only include inames tagged with 'vec'.)
    from loopy.schedule.checker.utils import (
        partition_inames_by_concurrency,
        get_EnterLoop_inames,
    )
    conc_inames, _ = partition_inames_by_concurrency(knl)
    enterloop_inames = get_EnterLoop_inames(lin_items)
    conc_loop_inames = conc_inames & enterloop_inames

    # The only concurrent EnterLoop inames should be Vec and ILP
    from loopy.kernel.data import (VectorizeTag, IlpBaseTag)
    for conc_iname in conc_loop_inames:
        # Assert that there exists an ilp or vectorize tag (out of the
        # potentially multiple other tags on this concurrent iname).
        assert any(
            isinstance(tag, (VectorizeTag, IlpBaseTag))
            for tag in knl.iname_to_tags[conc_iname])

    # }}}

    # {{{ Create two mappings from {statement instance: lex point}

    from loopy.schedule.checker.schedule import (
        get_pairwise_statement_orderings_inner
    )
    return get_pairwise_statement_orderings_inner(
        knl,
        lin_items,
        stmt_id_pairs,
        loops_to_ignore=conc_loop_inames,
        )

    # }}}

# }}}


def create_dependencies_from_legacy_knl(knl):
    """Return a kernel with additional dependencies constructed
    based on legacy kernel dependencies according to the
    following rules:

    (1) If a legacy dependency exists between ``insn0`` and ``insn1``,
    create the dependnecy ``SAME(SNC)`` where ``SNC`` is the set of
    non-concurrent inames used by both ``insn0`` and ``insn1``, and
    ``SAME`` is the relationship specified by the ``SAME`` attribute of
    :class:`loopy.schedule.checker.dependency.LegacyDependencyType`.

    (2) For each unique set of non-concurrent inames used by any statement
    (legacy term: `instruction'), i.e., for each loop tier,

        (a), find the set of all statements using a superset of those inames,

        (b), create a directed graph with these statements as nodes and
        edges representing a 'happens before' relationship specfied by
        each legacy dependency pair within these statements,

        (c), find the sources and sinks within this graph, and

        (d), connect each sink to each source (sink happens before source)
        with a ``PRIOR(SNC)`` dependency, where ``PRIOR`` is the
        relationship specified by the ``PRIOR`` attribute of
        :class:`loopy.schedule.checker.dependency.LegacyDependencyType`.

    :arg knl: A preprocessed :class:`loopy.kernel.LoopKernel`.

    :returns: A :class:`loopy.kernel.LoopKernel` containing additional
        dependencies as described above.

    """

    from loopy.schedule.checker.dependency import (
        create_legacy_dependency_constraint,
        get_dependency_sources_and_sinks,
        LegacyDependencyType,
    )
    from loopy.schedule.checker.utils import (
        partition_inames_by_concurrency,
        get_all_nonconcurrent_insn_iname_subsets,
        get_linearization_item_ids_within_inames,
        get_loop_nesting_map,
    )

    # Preprocess if not already preprocessed
    # note: kernels must always be preprocessed before scheduling

    from loopy.kernel import KernelState
    assert knl.state in [
        KernelState.PREPROCESSED, KernelState.LINEARIZED]

    # (Temporarily) collect all deps for each stmt in a dict
    # {stmt_after1: {stmt_before1: [dep1, dep2, ...], stmt_before2: [dep1, ...], ...}
    #  stmt_after2: {stmt_before1: [dep1, dep2, ...], stmt_before2: [dep1, ...], ...}
    #  ...}
    stmts_to_deps = {}

    # 1. For each pair of insns involved in a legacy dependency,
    # introduce SAME dep for set of shared, non-concurrent inames

    conc_inames, non_conc_inames = partition_inames_by_concurrency(knl)

    for stmt_after in knl.instructions:

        # Get any existing non-legacy dependencies, then add to these
        deps_for_stmt_after = stmt_after.dependencies.copy()

        # Loop over legacy dependees for this statement
        for stmt_before_id in stmt_after.depends_on:

            # Find non-concurrent inames shared between statements
            shared_non_conc_inames = (
                knl.id_to_insn[stmt_before_id].within_inames &
                stmt_after.within_inames &
                non_conc_inames)

            # Create a new dependency of type LegacyDependencyType.SAME
            dependency = create_legacy_dependency_constraint(
                knl,
                stmt_before_id,
                stmt_after.id,
                {LegacyDependencyType.SAME: shared_non_conc_inames},
                )

            # Add this dependency to this statement's dependencies
            deps_for_stmt_after.setdefault(stmt_before_id, []).append(dependency)

        if deps_for_stmt_after:
            # Add this statement's dependencies to stmts_to_deps map
            # (don't add deps directly to statement in kernel yet
            # because more are created below)
            new_deps_for_stmt_after = stmts_to_deps.get(stmt_after.id, {})
            for before_id_new, deps_new in deps_for_stmt_after.items():
                new_deps_for_stmt_after.setdefault(
                    before_id_new, []).extend(deps_new)
            stmts_to_deps[stmt_after.id] = new_deps_for_stmt_after

    # 2. For each set of insns within a loop tier, create a directed graph
    # based on legacy dependencies and introduce PRIOR dep from sinks to sources

    # Get loop nesting structure for use in creation of PRIOR dep
    # (to minimize time complexity, compute this once for whole kernel here)
    nests_outside_map = get_loop_nesting_map(knl, non_conc_inames)

    # Get all unique nonconcurrent within_iname sets
    non_conc_iname_subsets = get_all_nonconcurrent_insn_iname_subsets(
        knl, exclude_empty=True, non_conc_inames=non_conc_inames)

    for iname_subset in non_conc_iname_subsets:
        # Find linearization items within this iname set
        linearization_item_ids = get_linearization_item_ids_within_inames(
            knl, iname_subset)

        # Find sources and sinks (these sets may intersect)
        sources, sinks = get_dependency_sources_and_sinks(
            knl, linearization_item_ids)

        # Create deps
        for source_id in sources:

            # Get any existing non-legacy dependencies, then add to these
            source = knl.id_to_insn[source_id]
            deps_for_this_source = source.dependencies.copy()

            for sink_id in sinks:

                # Find non-concurrent inames shared between statements
                shared_non_conc_inames = (
                    knl.id_to_insn[sink_id].within_inames &
                    source.within_inames &
                    non_conc_inames)

                # Create a new dependency of type LegacyDependencyType.PRIOR
                dependency = create_legacy_dependency_constraint(
                    knl,
                    sink_id,
                    source_id,
                    {LegacyDependencyType.PRIOR: shared_non_conc_inames},
                    nests_outside_map=nests_outside_map,
                    )

                # Add this dependency to this statement's dependencies
                deps_for_this_source.setdefault(sink_id, []).append(dependency)

            if deps_for_this_source:
                # Add this statement's dependencies to stmts_to_deps map
                # (don't add deps directly to statement in kernel yet
                # because more are created below)
                new_deps_for_this_source = stmts_to_deps.get(source_id, {})
                for before_id_new, deps_new in deps_for_this_source.items():
                    new_deps_for_this_source.setdefault(
                        before_id_new, []).extend(deps_new)
                stmts_to_deps[source_id] = new_deps_for_this_source

    # Replace statements with new statements containing deps
    new_stmts = []
    for stmt_after in knl.instructions:
        if stmt_after.id in stmts_to_deps:
            new_stmts.append(
                stmt_after.copy(dependencies=stmts_to_deps[stmt_after.id]))
        else:
            new_stmts.append(
                stmt_after.copy())

    return knl.copy(instructions=new_stmts)


# {{{ find_unsatisfied_dependencies()

def find_unsatisfied_dependencies(
        knl,
        lin_items,
        ):
    """For each statement (:class:`loopy.InstructionBase`) found in a
    preprocessed kernel, determine which dependencies, if any, have been
    violated by the linearization described by `lin_items`, and return these
    dependencies.

    :arg knl: A preprocessed (or linearized) :class:`loopy.kernel.LoopKernel`
        containing the statements (:class:`loopy.InstructionBase`) whose
        dependencies will be checked against the linearization items.

    :arg lin_items: A list of :class:`loopy.schedule.ScheduleItem`
        (to be renamed to `loopy.schedule.LinearizationItem`) containing all
        linearization items in `knl.linearization`. To allow usage of
        this routine during linearization, a truncated (i.e. partial)
        linearization may be passed through this argument.

    :returns: A list of unsatisfied dependencies, each described using a
        :class:`collections.namedtuple` containing the following:

        - `statement_pair`: The (before, after) pair of statement IDs involved
          in the dependency.
        - `dependency`: An class:`islpy.Map` from each instance of the first
          statement to all instances of the second statement that must occur
          later.
        - `statement_ordering`: A statement ordering information tuple
          resulting from `lp.get_pairwise_statement_orderings`, a
          :class:`collections.namedtuple` containing the intra-thread
          statement instance ordering (SIO) (`sio_intra_thread`),
          intra-group SIO (`sio_intra_group`), and global
          SIO (`sio_global`), each realized as an :class:`islpy.Map` from each
          instance of the first statement to all instances of the second
          statement that occur later, as well as the intra-thread pairwise
          schedule (`pwsched_intra_thread`), intra-group pairwise schedule
          (`pwsched_intra_group`), and the global pairwise schedule
          (`pwsched_global`), each containing a pair of mappings from statement
          instances to points in a lexicographic ordering, one for each
          statement. Note that a pairwise schedule alone cannot be used to
          reproduce the corresponding SIO without the corresponding (unique)
          lexicographic order map, which is not returned.

    """

    # {{{ make sure kernel has been preprocessed

    # Note: kernels must always be preprocessed before scheduling
    from loopy.kernel import KernelState
    assert knl.state in [
            KernelState.PREPROCESSED,
            KernelState.LINEARIZED]

    # }}}

    # {{{ Create map from dependent statement id pairs to dependencies

    # To minimize time complexity, all pairwise schedules will be created
    # in one pass, which first requires finding all pairs of statements involved
    # in deps. We will also need to collect the deps for each statement pair,
    # so do this at the same time.

    stmt_pairs_to_deps = {}

    # stmt_pairs_to_deps:
    # {(stmt_id_before1, stmt_id_after1): [dep1, dep2, ...],
    #  (stmt_id_before2, stmt_id_after2): [dep1, dep2, ...],
    #  ...}

    for stmt_after in knl.instructions:
        for before_id, dep_list in stmt_after.dependencies.items():
            # (don't compare dep maps to maps found; duplicate deps should be rare)
            stmt_pairs_to_deps.setdefault(
                (before_id, stmt_after.id), []).extend(dep_list)
    # }}}

    # {{{ Get statement instance orderings

    pworders = get_pairwise_statement_orderings(
        knl,
        lin_items,
        stmt_pairs_to_deps.keys(),
        )

    # }}}

    # {{{ For each depender-dependee pair of statements, check all deps vs. SIO

    # Collect info about unsatisfied deps
    unsatisfied_deps = []
    from collections import namedtuple
    UnsatisfiedDependencyInfo = namedtuple(
        "UnsatisfiedDependencyInfo",
        ["statement_pair", "dependency", "statement_ordering"])

    for stmt_id_pair, dependencies in stmt_pairs_to_deps.items():

        # Get the pairwise ordering info (includes SIOs)
        pworder = pworders[stmt_id_pair]

        # Check each dep for this statement pair
        for dependency in dependencies:

            # Align constraint map space to match SIO so we can
            # check to see whether the constraint map is a subset of the SIO
            from loopy.schedule.checker.utils import (
                ensure_dim_names_match_and_align,
            )
            aligned_dep_map = ensure_dim_names_match_and_align(
                dependency, pworder.sio_intra_thread)

            # Spaces must match
            assert aligned_dep_map.space == pworder.sio_intra_thread.space
            assert aligned_dep_map.space == pworder.sio_intra_group.space
            assert aligned_dep_map.space == pworder.sio_global.space
            assert (aligned_dep_map.get_var_dict() ==
                pworder.sio_intra_thread.get_var_dict())
            assert (aligned_dep_map.get_var_dict() ==
                pworder.sio_intra_group.get_var_dict())
            assert (aligned_dep_map.get_var_dict() ==
                pworder.sio_global.get_var_dict())

            # Check dependency
            if not aligned_dep_map.is_subset(
                    pworder.sio_intra_thread |
                    pworder.sio_intra_group |
                    pworder.sio_global
                    ):

                unsatisfied_deps.append(UnsatisfiedDependencyInfo(
                    stmt_id_pair, aligned_dep_map, pworder))

                # Could break here if we don't care about remaining deps

    # }}}

    return unsatisfied_deps

# }}}

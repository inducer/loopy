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


# {{{ create a pairwise schedules for statement pairs

def get_schedules_for_statement_pairs(
        knl,
        linearization_items,
        insn_id_pairs,
        ):
    r"""For each statement pair in a subset of all statement pairs found in a
    linearized kernel, determine the (relative) order in which the statement
    instances are executed. For each pair, describe this relative ordering with
    a pair of mappings from statement instances to points in a single
    lexicographic ordering (a ``pairwise schedule''). When determining the
    relative ordering, ignore concurrent inames.

    :arg knl: A preprocessed :class:`loopy.kernel.LoopKernel` containing the
        linearization items that will be used to create a schedule.

    :arg linearization_items: A list of :class:`loopy.schedule.ScheduleItem`
        (to be renamed to `loopy.schedule.LinearizationItem`) containing
        all linearization items for which pairwise schedules will be
        created. To allow usage of this routine during linearization, a
        truncated (i.e. partial) linearization may be passed through this
        argument.

    :arg insn_id_pairs: A list containing pairs of instruction
        identifiers.

    :returns: A dictionary mapping each two-tuple of instruction identifiers
        provided in `insn_id_pairs` to a corresponding two-tuple containing two
        :class:`islpy.Map`\ s representing a pairwise schedule as two
        mappings from statement instances to lexicographic time, one for
        each of the two statements.

    .. doctest:

        >>> import loopy as lp
        >>> import numpy as np
        >>> # Make kernel -----------------------------------------------------------
        >>> knl = lp.make_kernel(
        ...     "{[i,j,k]: 0<=i<pi and 0<=j<pj and 0<=k<pk}",
        ...     [
        ...         "a[i,j] = j  {id=insn_a}",
        ...         "b[i,k] = k+a[i,0]  {id=insn_b,dep=insn_a}",
        ...     ])
        >>> knl = lp.add_and_infer_dtypes(knl, {"a": np.float32, "b": np.float32})
        >>> knl = lp.prioritize_loops(knl, "i,j")
        >>> knl = lp.prioritize_loops(knl, "i,k")
        >>> # Get a linearization
        >>> knl = lp.get_one_linearized_kernel(lp.preprocess_kernel(knl))
        >>> # Get a pairwise schedule -----------------------------------------------
        >>> from loopy.schedule.checker import get_schedules_for_statement_pairs
        >>> # Get two maps ----------------------------------------------------------
        >>> schedules = get_schedules_for_statement_pairs(
        ...     knl,
        ...     knl.linearization,
        ...     [("insn_a", "insn_b")],
        ...     )
        >>> # Print maps
        >>> print("\n".join(
        ...     str(m).replace("{ ", "{\n").replace(" :", "\n:")
        ...     for m in schedules[("insn_a", "insn_b")]
        ...     ))
        [pi, pj, pk] -> {
        [_lp_linchk_stmt = 0, i, j, k] -> [_lp_linchk_l0 = i, _lp_linchk_l1 = 0]
        : 0 <= i < pi and 0 <= j < pj and 0 <= k < pk }
        [pi, pj, pk] -> {
        [_lp_linchk_stmt = 1, i, j, k] -> [_lp_linchk_l0 = i, _lp_linchk_l1 = 1]
        : 0 <= i < pi and 0 <= j < pj and 0 <= k < pk }

    """

    # {{{ make sure kernel has been preprocessed

    from loopy.kernel import KernelState
    assert knl.state in [
            KernelState.PREPROCESSED,
            KernelState.LINEARIZED]

    # }}}

    # {{{ Find any EnterLoop inames that are tagged as concurrent
    # so that generate_pairwise_schedule() knows to ignore them
    # (In the future, this shouldn't be necessary because there
    # won't be any inames with ConcurrentTags in EnterLoop linearization items.
    # Test which exercises this: test_linearization_checker_with_stroud_bernstein())
    from loopy.schedule.checker.utils import (
        partition_inames_by_concurrency,
        get_EnterLoop_inames,
    )
    conc_inames, _ = partition_inames_by_concurrency(knl)
    enterloop_inames = get_EnterLoop_inames(linearization_items)
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

    # include only instructions involved in this dependency
    from loopy.schedule.checker.schedule import generate_pairwise_schedules
    return generate_pairwise_schedules(
        knl,
        linearization_items,
        insn_id_pairs,
        loops_to_ignore=conc_loop_inames,
        )

    # }}}

# }}}


def create_dependencies_from_legacy_knl(knl):
    """Return a set of
    :class:`loopy.schedule.checker.dependency.TBD`
    instances created for a :class:`loopy.LoopKernel` containing legacy
    depencencies.

    Create the new dependencies according to the following rules:

    (1) If a dependency exists between ``insn0`` and ``insn1``, create the
    dependnecy ``SAME(SNC)`` where ``SNC`` is the set of non-concurrent inames
    used by both ``insn0`` and ``insn1``, and ``SAME`` is the relationship
    specified by the ``SAME`` attribute of
    :class:`loopy.schedule.checker.dependency.LegacyDependencyType`.

    (2) For each unique set of non-concurrent inames used by any instruction
    (i.e., for each loop tier),

        (a), find the set of all instructions using a superset of those inames,

        (b), create a directed graph with these instructions as nodes and
        edges representing a 'happens before' relationship specfied by
        each legacy dependency pair within these instructions,

        (c), find the sources and sinks within this graph, and

        (d), connect each sink to each source (sink happens before source)
        with a ``PRIOR(SNC)`` dependency, where ``PRIOR`` is the
        relationship specified by the ``PRIOR`` attribute of
        :class:`loopy.schedule.checker.dependency.LegacyDependencyType`.

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

    from loopy import preprocess_kernel
    preprocessed_knl = preprocess_kernel(knl)
    # TODO just assert that it's already been preprocessed

    # TODO instead of keeping these in a set, attach each one to depender insn

    # Create constraint maps from kernel dependencies
    stmts_to_deps = {}  # used to (temporarily) collect all deps for each stmt

    # For each pair of insns involved in a legacy dependency,
    # introduce SAME dep for set of shared, non-concurrent inames

    conc_inames, non_conc_inames = partition_inames_by_concurrency(
        preprocessed_knl)

    for insn_after in preprocessed_knl.instructions:
        deps_for_stmt_after = insn_after.dependencies  # TODO copy?
        for insn_before_id in insn_after.depends_on:
            insn_before = preprocessed_knl.id_to_insn[insn_before_id]
            insn_before_inames = insn_before.within_inames
            insn_after_inames = insn_after.within_inames
            shared_non_conc_inames = (
                insn_before_inames & insn_after_inames & non_conc_inames)

            # TODO what to do if there is already a dep from insn_before->insn_after?
            # (currently just add a new one)

            # create a map representing constraints from the dependency,
            # which maps statement instance to all stmt instances that must occur
            # later and is acquired from the non-preprocessed kernel
            dependency = create_legacy_dependency_constraint(
                preprocessed_knl,
                insn_before_id,
                insn_after.id,
                {LegacyDependencyType.SAME: shared_non_conc_inames},
                nests_outside_map=None,  # not used for SAME
                )

            # add this dep map to this statement's deps
            deps_for_stmt_after.setdefault(insn_before_id, []).append(dependency)

        # store this statement's deps in stmts_to_deps
        # (don't add deps to stmt in knl yet because more are created below)
        #stmts_to_deps.setdefault(insn_after.id, []).append(deps_for_stmt_after)
        # TODO be more efficient here...
        new_deps_for_stmt_after = stmts_to_deps.get(insn_after.id, {})
        for before_id_new, deps_new in deps_for_stmt_after.items():
            new_deps_for_stmt_after.setdefault(before_id_new, []).extend(deps_new)
        stmts_to_deps[insn_after.id] = new_deps_for_stmt_after

    # loop-carried deps ------------------------------------------

    nests_outside_map = get_loop_nesting_map(knl, non_conc_inames)

    # Go through insns and get all unique insn.depends_on iname sets
    non_conc_iname_subsets = get_all_nonconcurrent_insn_iname_subsets(
        preprocessed_knl, exclude_empty=True, non_conc_inames=non_conc_inames)

    # For each set of insns within a given iname set, find sources and sinks.
    # Then make PRIOR dep from all sinks to all sources at previous iterations
    for iname_subset in non_conc_iname_subsets:
        # find linearization items within this iname set
        # TODO: could combine this step with the creation of the iname subsets
        # to save time?
        linearization_item_ids = get_linearization_item_ids_within_inames(
            preprocessed_knl, iname_subset)

        # find sources and sinks (these sets may have non-empty intersection)
        sources, sinks = get_dependency_sources_and_sinks(
            preprocessed_knl, linearization_item_ids)

        # create prior deps

        # in future, consider inserting single no-op source and sink
        for source_id in sources:
            deps_for_this_source = {}
            for sink_id in sinks:
                sink_insn_inames = preprocessed_knl.id_to_insn[sink_id].within_inames
                source_insn_inames = preprocessed_knl.id_to_insn[
                    source_id].within_inames
                shared_non_conc_inames = (
                    sink_insn_inames & source_insn_inames & non_conc_inames)

                # create a map representing constraints from the dependency,
                # which maps statement instance to all stmt instances that must occur
                # later and is acquired from the non-preprocessed kernel
                dependency = create_legacy_dependency_constraint(
                    preprocessed_knl,
                    sink_id,
                    source_id,
                    {LegacyDependencyType.PRIOR: shared_non_conc_inames},
                    nests_outside_map=nests_outside_map,
                    )
                # TODO in PRIOR case, there's some stuff happening in
                # create_legacy_dep_constraint^ that may be redundant and could be
                # done once for whole kernel

                # TODO what if there is already a different dep from sink->source?
                # add this dep map to this statement's deps
                deps_for_this_source.setdefault(sink_id, []).append(dependency)

            # store this statement's deps in stmts_to_deps
            #stmts_to_deps.setdefault(source_id, []).append(deps_for_this_source)
            # TODO be more efficient here...
            new_deps_for_this_source = stmts_to_deps.get(source_id, {})
            for before_id_new, deps_new in deps_for_this_source.items():
                new_deps_for_this_source.setdefault(
                    before_id_new, []).extend(deps_new)
            stmts_to_deps[source_id] = new_deps_for_this_source

    # replace instructions with new instructions containing deps
    new_instructions = []
    for insn_after in preprocessed_knl.instructions:
        new_instructions.append(
            insn_after.copy(dependencies=stmts_to_deps[insn_after.id]))

    return preprocessed_knl.copy(instructions=new_instructions)


def check_linearization_validity(
        knl,
        linearization_items,
        ):
    # TODO document

    from loopy.schedule.checker.lexicographic_order_map import (
        get_statement_ordering_map,
    )
    from loopy.schedule.checker.utils import (
        prettier_map_string,
    )
    from loopy.schedule.checker.schedule import (
        get_lex_order_map_for_sched_space,
    )

    # {{{ make sure kernel has been preprocessed

    # note: kernels must always be preprocessed before scheduling
    from loopy.kernel import KernelState
    assert knl.state in [
            KernelState.PREPROCESSED,
            KernelState.LINEARIZED]

    # }}}

    # {{{ Create map from dependent instruction id pairs to dependencies

    # To minimize time complexity, all pairwise schedules will be created
    # in one pass, which first requires finding all pairs of statements involved
    # in deps.
    # So, since we have to find these pairs anyway, collect their deps at
    # the same time so we don't have to do it again later during lin checking.

    stmts_to_deps = {}
    for insn_after in knl.instructions:
        for before_id, dep_list in insn_after.dependencies.items():
            stmts_to_deps.setdefault(
                (before_id, insn_after.id), []).extend(dep_list)
    # }}}

    schedules = get_schedules_for_statement_pairs(
        knl,
        linearization_items,
        stmts_to_deps.keys(),
        )

    # For each dependency, create+test linearization containing pair of insns------
    linearization_is_valid = True
    for (insn_id_before, insn_id_after), dependencies in stmts_to_deps.items():

        # Get pairwise schedule for stmts involved in the dependency:
        # two isl maps from {statement instance: lex point},
        isl_sched_map_before, isl_sched_map_after = schedules[
            (insn_id_before, insn_id_after)]

        # get map representing lexicographic ordering
        sched_lex_order_map = get_lex_order_map_for_sched_space(isl_sched_map_before)

        # create statement instance ordering,
        # maps each statement instance to all statement instances occuring later
        sio = get_statement_ordering_map(
            isl_sched_map_before,
            isl_sched_map_after,
            sched_lex_order_map,
            )

        # check each dep for this statement pair
        for dependency in dependencies:

            # reorder variables/params in constraint map space to match SIO so we can
            # check to see whether the constraint map is a subset of the SIO
            # (spaces must be aligned so that the variables in the constraint map
            # correspond to the same variables in the SIO)
            from loopy.schedule.checker.utils import (
                ensure_dim_names_match_and_align,
            )

            aligned_dep_map = ensure_dim_names_match_and_align(
                dependency, sio)

            import islpy as isl
            assert aligned_dep_map.space == sio.space
            assert (
                aligned_dep_map.space.get_var_names(isl.dim_type.in_)
                == sio.space.get_var_names(isl.dim_type.in_))
            assert (
                aligned_dep_map.space.get_var_names(isl.dim_type.out)
                == sio.space.get_var_names(isl.dim_type.out))
            assert (
                aligned_dep_map.space.get_var_names(isl.dim_type.param)
                == sio.space.get_var_names(isl.dim_type.param))

            if not aligned_dep_map.is_subset(sio):

                linearization_is_valid = False

                print("================ constraint check failure =================")
                print("Constraint map not subset of SIO")
                print("Dependencies:")
                print(insn_id_before+"->"+insn_id_after)
                print(prettier_map_string(dependency))
                print("Statement instance ordering:")
                print(prettier_map_string(sio))
                print("dependency.gist(sio):")
                print(prettier_map_string(aligned_dep_map.gist(sio)))
                print("sio.gist(dependency)")
                print(prettier_map_string(sio.gist(aligned_dep_map)))
                print("Loop priority known:")
                print(knl.loop_priority)
                print("===========================================================")

    return linearization_is_valid

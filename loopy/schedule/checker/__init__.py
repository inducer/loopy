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
    instances are executed. For each pair, represent this relative ordering
    using three ``statement instance orderings`` (SIOs):

    - The intra-thread SIO: A :class:`islpy.Map` from each instance of the
      first statement to all instances of the second statement that occur
      later, such that both statement instances in each before-after pair are
      executed within the same work-item (thread).

    - The intra-group SIO: A :class:`islpy.Map` from each instance of the first
      statement to all instances of the second statement that occur later, such
      that both statement instances in each before-after pair are executed
      within the same work-group (though potentially by different work-items).

    - The global SIO: A :class:`islpy.Map` from each instance of the first
      statement to all instances of the second statement that occur later, even
      if the two statement instances in a given before-after pair are executed
      within different work-groups.

    :arg knl: A preprocessed :class:`loopy.kernel.LoopKernel` containing the
        linearization items that will be used to create the SIOs.

    :arg lin_items: A list of :class:`loopy.schedule.ScheduleItem`
        (to be renamed to `loopy.schedule.LinearizationItem`) containing all
        linearization items for which SIOs will be created. To allow usage of
        this routine during linearization, a truncated (i.e. partial)
        linearization may be passed through this argument.

    :arg stmt_id_pairs: A sequence containing pairs of statement identifiers.

    :returns: A dictionary mapping each two-tuple of statement identifiers
        provided in `stmt_id_pairs` to a :class:`StatementOrdering`, which
        contains the three SIOs described above.

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
        >>> # Preprocess
        >>> knl = lp.preprocess_kernel(knl)
        >>> # Get a linearization
        >>> knl = lp.get_one_linearized_kernel(
        ...     knl["loopy_kernel"], knl.callables_table)
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

    # {{{ Create the SIOs

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


# {{{ find_unsatisfied_dependencies()

def find_unsatisfied_dependencies(
        knl,
        lin_items=None,
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
        linearization may be passed through this argument. If not provided,
        `knl.linearization` will be used.

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

    # {{{ Handle lin_items=None and make sure kernel has been preprocessed

    from loopy.kernel import KernelState
    if lin_items is None:
        assert knl.state == KernelState.LINEARIZED
        lin_items = knl.linearization
    else:
        # Note: kernels must always be preprocessed before scheduling
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

    from loopy.kernel.instruction import BarrierInstruction
    # TODO (fix) for now, don't check deps on/by barriers
    for stmt_after in knl.instructions:
        if not isinstance(stmt_after, BarrierInstruction):
            for before_id, dep_list in stmt_after.dependencies.items():
                if not isinstance(knl.id_to_insn[before_id], BarrierInstruction):
                    # (don't compare dep maps to maps found;
                    # duplicate deps should be rare)
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

# vim: foldmethod=marker

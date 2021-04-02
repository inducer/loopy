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
        insn_id_pairs,
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

    :arg insn_id_pairs: A list containing pairs of instruction identifiers.

    :returns: A dictionary mapping each two-tuple of instruction identifiers
        provided in `insn_id_pairs` to a :class:`collections.namedtuple`
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
        ...         "a[j] = j  {id=insn_a}",
        ...         "b[k] = k+a[0]  {id=insn_b,dep=insn_a}",
        ...     ])
        >>> knl = lp.add_and_infer_dtypes(knl, {"a": np.float32, "b": np.float32})
        >>> # Get a linearization
        >>> knl = lp.get_one_linearized_kernel(lp.preprocess_kernel(knl))
        >>> # Get pairwise order info -----------------------------------------------
        >>> from loopy.schedule.checker import get_pairwise_statement_orderings
        >>> sio_dict = get_pairwise_statement_orderings(
        ...     knl,
        ...     knl.linearization,
        ...     [("insn_a", "insn_b")],
        ...     )
        >>> # Print map
        >>> print(str(sio_dict[("insn_a", "insn_b")].sio_intra_thread
        ...     ).replace("{ ", "{\n").replace(" :", "\n:"))
        [pj, pk] -> {
        [_lp_linchk_stmt' = 0, j', k'] -> [_lp_linchk_stmt = 1, j, k]
        : 0 <= j' < pj and 0 <= k' < pk and 0 <= j < pj and 0 <= k < pk }

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
        insn_id_pairs,
        loops_to_ignore=conc_loop_inames,
        )

    # }}}

# }}}


def check_dependency_satisfaction(
        knl,
        linearization_items,
        ):
    # TODO document

    from loopy.schedule.checker.utils import (
        prettier_map_string,
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

    pworders = get_pairwise_statement_orderings(
        knl,
        linearization_items,
        stmts_to_deps.keys(),
        )

    # For each dependency, create+test linearization containing pair of insns------
    linearization_is_valid = True
    for (insn_id_before, insn_id_after), dependencies in stmts_to_deps.items():

        # Get pairwise ordering info for stmts involved in the dependency
        pworder = pworders[(insn_id_before, insn_id_after)]

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
                dependency, pworder.sio_intra_thread)

            assert aligned_dep_map.space == pworder.sio_intra_thread.space
            assert aligned_dep_map.space == pworder.sio_intra_group.space
            assert aligned_dep_map.space == pworder.sio_global.space
            assert (aligned_dep_map.get_var_dict() ==
                pworder.sio_intra_thread.get_var_dict())
            assert (aligned_dep_map.get_var_dict() ==
                pworder.sio_intra_group.get_var_dict())
            assert (aligned_dep_map.get_var_dict() ==
                pworder.sio_global.get_var_dict())

            if not aligned_dep_map.is_subset(
                    pworder.sio_intra_thread |
                    pworder.sio_intra_group |
                    pworder.sio_global
                    ):

                linearization_is_valid = False

                print("================ constraint check failure =================")
                print("Constraint map not subset of SIO")
                print("Dependencies:")
                print(insn_id_before+"->"+insn_id_after)
                print(prettier_map_string(dependency))
                print("Statement instance ordering:")
                print(prettier_map_string(pworder.sio_intra_thread))
                print("dependency.gist(pworder.sio_intra_thread):")
                print(prettier_map_string(
                    aligned_dep_map.gist(pworder.sio_intra_thread)))
                print("pworder.sio_intra_thread.gist(dependency)")
                print(prettier_map_string(
                    pworder.sio_intra_thread.gist(aligned_dep_map)))
                print("Loop priority known:")
                print(knl.loop_priority)
                print("===========================================================")

    return linearization_is_valid

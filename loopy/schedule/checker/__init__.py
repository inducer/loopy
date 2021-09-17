"""
.. autofunction:: get_pairwise_statement_orderings

.. automodule:: loopy.schedule.checker.schedule
"""


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
        perform_closure_checks=False,
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
        provided in `stmt_id_pairs` to a
        :class:`~loopy.schedule.checker.schedule.StatementOrdering`, which
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

    # FIXME Consider just putting this ilp/vec logic inside
    # get_pairwise_statement_orderings_inner; passing these in as
    # 'loops_to_ignore' made more sense when we were just dealing with the
    # intra-thread case.
    from loopy.schedule.checker.utils import (
        partition_inames_by_concurrency,
        get_EnterLoop_inames,
    )
    conc_inames, _ = partition_inames_by_concurrency(knl)
    enterloop_inames = get_EnterLoop_inames(lin_items)
    ilp_and_vec_inames = conc_inames & enterloop_inames

    # The only concurrent EnterLoop inames should be Vec and ILP
    from loopy.kernel.data import (VectorizeTag, IlpBaseTag)
    for conc_iname in ilp_and_vec_inames:
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
        ilp_and_vec_inames=ilp_and_vec_inames,
        perform_closure_checks=perform_closure_checks,
        )

    # }}}

# }}}

# vim: foldmethod=marker

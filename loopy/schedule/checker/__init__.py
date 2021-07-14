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
        >>> # Preprocess
        >>> knl = lp.preprocess_kernel(knl)
        >>> # Get a linearization
        >>> knl = lp.get_one_linearized_kernel(
        ...     knl["loopy_kernel"], knl.callables_table)
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

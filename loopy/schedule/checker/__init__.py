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


# {{{ create PairwiseScheduleBuilder for statement pair

def get_schedule_for_statement_pair(
        knl,
        linearization_items,
        insn_id_before,
        insn_id_after,
        ):
    """Create a :class:`loopy.schedule.checker.schedule.PairwiseScheduleBuilder`
    representing the order of two statements as a mapping from
    :class:`loopy.schedule.checker.StatementInstanceSet`
    to lexicographic time.

    :arg knl: A :class:`loopy.kernel.LoopKernel` containing the
        linearization items that will be used to create a schedule.

    :arg linearization_items: A list of :class:`loopy.schedule.ScheduleItem`
        (to be renamed to `loopy.schedule.LinearizationItem`) containing
        the two linearization items for which a schedule will be
        created. This list may be a *partial* linearization for a
        kernel since this function may be used during the linearization
        process.

    :arg insn_id_before: An instruction identifier that is unique within
        a :class:`loopy.kernel.LoopKernel`.

    :arg insn_id_after: An instruction identifier that is unique within
        a :class:`loopy.kernel.LoopKernel`.

    :returns: A :class:`loopy.schedule.checker.schedule.PairwiseScheduleBuilder`
        representing the order of two statements as a mapping from
        :class:`loopy.schedule.checker.StatementInstanceSet`
        to lexicographic time.

    Example usage::

        # Make kernel --------------------------------------------------------
        knl = lp.make_kernel(
            "{[i,j,k]: 0<=i<pi and 0<=j<pj and 0<=k<pk}",
            [
                "a[i,j] = j  {id=insn_a}",
                "b[i,k] = k+a[i,0]  {id=insn_b,dep=insn_a}",
            ])
        knl = lp.add_and_infer_dtypes(knl, {"a": np.float32, "b": np.float32})
        knl = lp.prioritize_loops(knl, "i,j")
        knl = lp.prioritize_loops(knl, "i,k")

        # Get a linearization
        knl = lp.get_one_linearized_kernel(lp.preprocess_kernel(knl))

        # Get a pairwise schedule --------------------------------------------

        from loopy.schedule.checker import (
            get_schedule_for_statement_pair,
        )
        sched_builder_ab = get_schedule_for_statement_pair(
            knl,
            knl.linearization,
            "insn_a",
            "insn_b",
            )

        # Get two isl maps from the PairwiseScheduleBuilder ------------------

        from loopy.schedule.checker import (
            get_isl_maps_from_PairwiseScheduleBuilder,
        )
        sched_a, sched_b = get_isl_maps_from_PairwiseScheduleBuilder(
            sched_builder_ab, knl)

        print(sched_a)
        print(sched_b)

    Example Output::

        [pi, pj, pk] -> {
        [_lp_linchk_statement = 0, i, j, k] ->
        [_lp_linchk_l0 = 0, _lp_linchk_l1 = i, _lp_linchk_l2 = 0,
        _lp_linchk_l3 = j, _lp_linchk_l4 = 0] :
        0 <= i < pi and 0 <= j < pj and 0 <= k < pk }
        [pi, pj, pk] -> {
        [_lp_linchk_statement = 1, i, j, k] ->
        [_lp_linchk_l0 = 0, _lp_linchk_l1 = i, _lp_linchk_l2 = 1,
        _lp_linchk_l3 = k, _lp_linchk_l4 = 0] :
        0 <= i < pi and 0 <= j < pj and 0 <= k < pk }

    """

    # {{{ preprocess if not already preprocessed

    from loopy import preprocess_kernel
    preproc_knl = preprocess_kernel(knl)

    # }}}

    # {{{ find any EnterLoop inames that are tagged as concurrent

    # so that PairwiseScheduleBuilder knows to ignore them
    # (In the future, this shouldn't be necessary because there
    #  won't be any inames with ConcurrentTags in EnterLoop linearization items.
    #  Test which exercises this: test_linearization_checker_with_stroud_bernstein())
    from loopy.schedule.checker.utils import (
        get_concurrent_inames,
        get_EnterLoop_inames,
    )
    conc_inames, _ = get_concurrent_inames(preproc_knl)
    enterloop_inames = get_EnterLoop_inames(linearization_items, preproc_knl)
    conc_loop_inames = conc_inames & enterloop_inames
    if conc_loop_inames:
        from warnings import warn
        warn(
            "get_schedule_for_statement_pair encountered EnterLoop for inames %s "
            "with ConcurrentTag(s) in linearization for kernel %s. "
            "Ignoring these loops." % (conc_loop_inames, preproc_knl.name))

    # }}}

    # {{{ Create PairwiseScheduleBuilder: mapping of {statement instance: lex point}

    # include only instructions involved in this dependency
    from loopy.schedule.checker.schedule import PairwiseScheduleBuilder
    return PairwiseScheduleBuilder(
        linearization_items,
        insn_id_before,
        insn_id_after,
        loops_to_ignore=conc_loop_inames,
        )

    # }}}

# }}}


# {{{ get_isl_maps_from_PairwiseScheduleBuilder

def get_isl_maps_from_PairwiseScheduleBuilder(sched_builder, knl):
    r"""Create a pair of :class:`islpy.Map`\ s representing a
    sub-schedule as two mappings from statement instances to lexicographic
    time, one for the dependee statement and one for the depender.

    :arg sched_builder: A
        :class:`loopy.schedule.checker.schedule.PairwiseScheduleBuilder`
        representing the order of two statements as a mapping from
        :class:`loopy.schedule.checker.StatementInstanceSet`
        to lexicographic time.

    :arg knl: A :class:`loopy.kernel.LoopKernel` containing the
        linearization items that will be used to create a schedule.

    :returns: A two-tuple containing two :class:`islpy.Map`\ s
        representing the schedule as two mappings
        from statement instances to lexicographic time, one for
        the dependee and one for the depender.
    """

    # {{{ Get iname domains
    dom_before = knl.get_inames_domain(
        knl.id_to_insn[
            sched_builder.stmt_instance_before.stmt_ref.insn_id].within_inames)
    dom_after = knl.get_inames_domain(
        knl.id_to_insn[
            sched_builder.stmt_instance_after.stmt_ref.insn_id].within_inames)
    # }}}

    # {{{ Get isl maps
    return sched_builder.create_isl_maps(dom_before, dom_after)
    # }}}

# }}}

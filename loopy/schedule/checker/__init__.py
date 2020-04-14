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


def get_schedule_for_statement_pair(
        knl,
        linearization_items,
        insn_id_before,
        insn_id_after,
        prohibited_var_names=set(),
        ):
    """A set of dependencies between two statements.

    .. arg insn_id_before: An instruction identifier that is unique within
        a :class:`loopy.kernel.LoopKernel`.

    .. arg insn_id_after: An instruction identifier that is unique within
        a :class:`loopy.kernel.LoopKernel`.

    """

    # We don't retrieve linearization items from knl because knl may not be
    # (fully) linearized yet. This function may be called part way through the
    # linearization process and receive the current (unfinished) set of
    # linearization items

    # Preprocess if not already preprocessed
    from loopy import preprocess_kernel
    preproc_knl = preprocess_kernel(knl)

    if not prohibited_var_names:
        prohibited_var_names = preproc_knl.all_inames()

    # Get EnterLoop inames tagged as concurrent so LexSchedule can ignore
    # (In the future, this shouldn't be necessary because there
    #  won't be any inames with ConcurrentTags in EnterLoop linearization items.
    #  Test exercising this: test_linearization_checker_with_stroud_bernstein())
    from loopy.schedule.checker.utils import (
        get_concurrent_inames,
        _get_EnterLoop_inames,
    )
    conc_inames, _ = get_concurrent_inames(preproc_knl)
    enterloop_inames = _get_EnterLoop_inames(linearization_items, preproc_knl)
    conc_loop_inames = conc_inames & enterloop_inames
    if conc_loop_inames:
        from warnings import warn
        warn(
            "get_schedule_for_statement_pair encountered EnterLoop for inames %s "
            "with ConcurrentTag(s) in linearization for kernel %s. "
            "Ignoring these loops." % (conc_loop_inames, preproc_knl.name))

    # Create LexSchedule: mapping of {statement instance: lex point}
    # include only instructions involved in this dependency
    from loopy.schedule.checker.schedule import LexSchedule
    return LexSchedule(
        linearization_items,
        insn_id_before,
        insn_id_after,
        prohibited_var_names=prohibited_var_names,
        loops_to_ignore=conc_loop_inames,
        )


def get_isl_maps_for_LexSchedule(
        lex_sched,
        knl,
        insn_id_before,
        insn_id_after,
        ):
    # Get two isl maps representing the LexSchedule,
    # one for the 'before' linearization item and one for 'after';
    # this requires the iname domains

    insn_before_inames = knl.id_to_insn[insn_id_before].within_inames
    insn_after_inames = knl.id_to_insn[insn_id_after].within_inames
    dom_before = knl.get_inames_domain(insn_before_inames)
    dom_after = knl.get_inames_domain(insn_after_inames)

    isl_sched_map_before, isl_sched_map_after = \
        lex_sched.create_isl_maps(
            dom_before,
            dom_after,
        )

    return isl_sched_map_before, isl_sched_map_after

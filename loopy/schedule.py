from __future__ import division

from pytools import Record




# {{{ schedule items

class ScheduledLoop(Record):
    __slots__ = ["iname"]

class WriteOutput(Record):
    pass

# plus loopy.prefetch.{Register,LocalMemory}Prefetch

# }}}

def generate_loop_schedules(kernel, hints=[]):
    from loopy.kernel import TAG_GROUP_IDX, TAG_WORK_ITEM_IDX, TAG_ILP, ParallelTag

    prev_schedule = kernel.schedule
    if prev_schedule is None:
        prev_schedule = [
                ScheduledLoop(iname=iname)
                for iname in (
                    kernel.ordered_inames_by_tag_type(TAG_GROUP_IDX)
                    + kernel.ordered_inames_by_tag_type(TAG_WORK_ITEM_IDX))]

    scheduled_inames = set(sch_item.iname
            for sch_item in prev_schedule
            if isinstance(sch_item, ScheduledLoop))

    # have a schedulable prefetch? load, schedule it
    had_usable_prefetch = False
    locally_parallel_inames = set(
            iname for iname in scheduled_inames
            if isinstance(kernel.iname_to_tag.get(iname), 
                (TAG_ILP, TAG_WORK_ITEM_IDX)))

    for pf in kernel.prefetch.itervalues():
        # already scheduled? never mind then.
        if pf in prev_schedule:
            continue

        # a free variable not known yet? then we're not ready
        if not pf.free_variables() <= scheduled_inames:
            continue

        # a prefetch variable already scheduled, but not borrowable?
        # (only work item index variables are borrowable)

        if set(pf.all_inames()) & (scheduled_inames - locally_parallel_inames):
            # dead end: we won't be able to schedule this prefetch
            # in this branch. at least one of its loop dimensions
            # was already scheduled, and that dimension is not
            # borrowable.

            #print "UNSCHEDULABLE", kernel.schedule
            return

        new_kernel = kernel.copy(schedule=prev_schedule+[pf])
        for knl in generate_loop_schedules(new_kernel):
            had_usable_prefetch = True
            yield knl

    if had_usable_prefetch:
        # because we've already recursed
        return

    # Build set of potentially schedulable variables
    # Don't re-schedule already scheduled variables
    schedulable = kernel.all_inames() - scheduled_inames

    # Schedule in the following order:
    # - serial output inames
    # - remaining parallel output inames (i.e. ILP)
    # - output write
    # - reduction
    # Don't schedule reduction variables until all output
    # variables are taken care of. Once they are, schedule
    # output writing.
    parallel_output_inames = set(oin for oin in kernel.output_inames()
            if isinstance(kernel.iname_to_tag.get(oin), ParallelTag))

    serial_output_inames = kernel.output_inames() - parallel_output_inames

    if schedulable & serial_output_inames:
        schedulable = schedulable & serial_output_inames

    if schedulable & parallel_output_inames:
        schedulable  = schedulable & parallel_output_inames

    if kernel.output_inames() <= scheduled_inames:
        if not any(isinstance(sch_item, WriteOutput)
                for sch_item in prev_schedule):
            kernel = kernel.copy(
                    schedule=prev_schedule + [WriteOutput()])
            prev_schedule = kernel.schedule

    # Don't schedule variables that are prefetch axes
    # for not-yet-scheduled prefetches.
    unsched_prefetch_axes = set(iname
            for pf in kernel.prefetch.itervalues()
            if pf not in prev_schedule
            for iname in pf.all_inames()
            if not isinstance(kernel.iname_to_tag.get(iname), ParallelTag))
    schedulable -= unsched_prefetch_axes

    while hints and hints[0] in scheduled_inames:
        hints = hints[1:]

    if hints and hints[0] in schedulable:
        schedulable = set([hints[0]])

    if schedulable:
        # have a schedulable variable? schedule a loop for it, recurse
        for iname in schedulable:
            new_kernel = kernel.copy(schedule=prev_schedule+[ScheduledLoop(iname=iname)])
            for knl in generate_loop_schedules(new_kernel, hints):
                yield knl
    else:
        # all loop dimensions and prefetches scheduled?
        # great! yield the finished product if it is complete

        from loopy import LoopyAdvisory

        if hints:
            from warnings import warn
            warn("leftover schedule hints: "+ (", ".join(hints)),
                    LoopyAdvisory)

        all_inames_scheduled = len(scheduled_inames) == len(kernel.all_inames())

        from loopy.prefetch import LocalMemoryPrefetch
        all_pf_scheduled =  len(set(sch_item for sch_item in prev_schedule
            if isinstance(sch_item, LocalMemoryPrefetch))) == len(kernel.prefetch)
        output_scheduled = len(set(sch_item for sch_item in prev_schedule
            if isinstance(sch_item, WriteOutput))) == 1

        if all_inames_scheduled and all_pf_scheduled and output_scheduled:
            yield kernel




# vim: foldmethod=marker

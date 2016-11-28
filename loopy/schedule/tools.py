from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2016 Matt Wala"

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

from loopy.kernel.data import temp_var_scope
from loopy.schedule import (BeginBlockItem, CallKernel, EndBlockItem,
                            RunInstruction, Barrier)

from pytools import memoize_method


# {{{ block boundary finder

def get_block_boundaries(schedule):
    """
    Return a dictionary mapping indices of
    :class:`loopy.schedule.BlockBeginItem`s to
    :class:`loopy.schedule.BlockEndItem`s and vice versa.
    """
    block_bounds = {}
    active_blocks = []
    for idx, sched_item in enumerate(schedule):
        if isinstance(sched_item, BeginBlockItem):
            active_blocks.append(idx)
        elif isinstance(sched_item, EndBlockItem):
            start = active_blocks.pop()
            block_bounds[start] = idx
            block_bounds[idx] = start
    return block_bounds

# }}}


# {{{ instruction query utility

class InstructionQuery(object):

    def __init__(self, kernel):
        self.kernel = kernel
        block_bounds = get_block_boundaries(kernel.schedule)
        subkernel_slices = {}
        from six import iteritems
        for start, end in iteritems(block_bounds):
            sched_item = kernel.schedule[start]
            if isinstance(sched_item, CallKernel):
                subkernel_slices[sched_item.kernel_name] = slice(start, end + 1)
        self.subkernel_slices = subkernel_slices

    @memoize_method
    def subkernels(self):
        return frozenset(self.subkernel_slices.keys())

    @memoize_method
    def insns_reading_or_writing(self, var):
        return frozenset(insn.id for insn in self.kernel.instructions
            if var in insn.read_dependency_names()
                or var in insn.assignee_var_names())

    @memoize_method
    def insns_in_subkernel(self, subkernel):
        return frozenset(sched_item.insn_id for sched_item
            in self.kernel.schedule[self.subkernel_slices[subkernel]]
            if isinstance(sched_item, RunInstruction))

    @memoize_method
    def temporaries_read_in_subkernel(self, subkernel):
        return frozenset(
            var
            for insn in self.insns_in_subkernel(subkernel)
            for var in self.kernel.id_to_insn[insn].read_dependency_names()
            if var in self.kernel.temporary_variables)

    @memoize_method
    def temporaries_written_in_subkernel(self, subkernel):
        return frozenset(
            var
            for insn in self.insns_in_subkernel(subkernel)
            for var in self.kernel.id_to_insn[insn].assignee_var_names()
            if var in self.kernel.temporary_variables)

    @memoize_method
    def temporaries_read_or_written_in_subkernel(self, subkernel):
        return (
            self.temporaries_read_in_subkernel(subkernel) |
            self.temporaries_written_in_subkernel(subkernel))

    @memoize_method
    def inames_in_subkernel(self, subkernel):
        subkernel_start = self.subkernel_slices[subkernel].start
        return frozenset(self.kernel.schedule[subkernel_start].extra_inames)

    @memoize_method
    def pre_and_post_barriers(self, subkernel):
        subkernel_start = self.subkernel_slices[subkernel].start
        subkernel_end = self.subkernel_slices[subkernel].stop

        def is_global_barrier(item):
            return isinstance(item, Barrier) and item.kind == "global"

        try:
            pre_barrier = next(item for item in
                    self.kernel.schedule[subkernel_start::-1]
                    if is_global_barrier(item)).originating_insn_id
        except StopIteration:
            pre_barrier = None

        try:
            post_barrier = next(item for item in
                    self.kernel.schedule[subkernel_end:]
                    if is_global_barrier(item)).originating_insn_id
        except StopIteration:
            post_barrier = None

        return (pre_barrier, post_barrier)

    @memoize_method
    def hw_inames(self, insn_id):
        """
        Return the inames that insn runs in and that are tagged as hardware
        parallel.
        """
        from loopy.kernel.data import HardwareParallelTag
        return set(iname for iname in self.kernel.insn_inames(insn_id)
                   if isinstance(self.kernel.iname_to_tag.get(iname),
                                 HardwareParallelTag))

    @memoize_method
    def common_hw_inames(self, insn_ids):
        """
        Return the common set of hardware parallel tagged inames among
        the list of instructions.
        """
        # Get the list of hardware inames in which the temporary is defined.
        if len(insn_ids) == 0:
            return set()
        return set.intersection(*(self.hw_inames(id) for id in insn_ids))

# }}}


# {{{ add extra args to schedule

def add_extra_args_to_schedule(kernel):
    """
    Fill the `extra_args` fields in all the :class:`loopy.schedule.CallKernel`
    instructions in the schedule with global temporaries.
    """
    new_schedule = []

    insn_query = InstructionQuery(kernel)

    for sched_item in kernel.schedule:
        if isinstance(sched_item, CallKernel):
            subrange_temporaries = (insn_query
                .temporaries_read_or_written_in_subkernel(sched_item.kernel_name))
            more_args = set(tv
                for tv in subrange_temporaries
                if
                kernel.temporary_variables[tv].scope == temp_var_scope.GLOBAL
                and
                kernel.temporary_variables[tv].initializer is None
                and
                tv not in sched_item.extra_args)
            new_schedule.append(sched_item.copy(
                extra_args=sched_item.extra_args + sorted(more_args)))
        else:
            new_schedule.append(sched_item)

    return kernel.copy(schedule=new_schedule)

# }}}

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


# {{{ block boundary finder

def get_block_boundaries(schedule):
    """
    Return a dictionary mapping indices of
    :class:`loopy.schedule.BlockBeginItem`s to
    :class:`loopy.schedule.BlockEndItem`s and vice versa.
    """
    from loopy.schedule import (BeginBlockItem, EndBlockItem)
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


# {{{ subkernel tools

def temporaries_read_in_subkernel(kernel, subkernel):
    from loopy.kernel.tools import get_subkernel_to_insn_id_map
    insn_ids = get_subkernel_to_insn_id_map(kernel)[subkernel]
    return frozenset(tv
            for insn_id in insn_ids
            for tv in kernel.id_to_insn[insn_id].read_dependency_names()
            if tv in kernel.temporary_variables)


def temporaries_written_in_subkernel(kernel, subkernel):
    from loopy.kernel.tools import get_subkernel_to_insn_id_map
    insn_ids = get_subkernel_to_insn_id_map(kernel)[subkernel]
    return frozenset(tv
            for insn_id in insn_ids
            for tv in kernel.id_to_insn[insn_id].write_dependency_names()
            if tv in kernel.temporary_variables)

# }}}


# {{{ add extra args to schedule

def add_extra_args_to_schedule(kernel):
    """
    Fill the `extra_args` fields in all the :class:`loopy.schedule.CallKernel`
    instructions in the schedule with global temporaries.
    """
    new_schedule = []
    from loopy.schedule import CallKernel

    for sched_item in kernel.schedule:
        if isinstance(sched_item, CallKernel):
            subkernel = sched_item.kernel_name

            used_temporaries = (
                    temporaries_read_in_subkernel(kernel, subkernel)
                    | temporaries_written_in_subkernel(kernel, subkernel))

            more_args = set(tv
                    for tv in used_temporaries
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

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

from loopy.diagnostic import LoopyError
from loopy.kernel.data import temp_var_scope
from loopy.schedule import (Barrier, BeginBlockItem, CallKernel, EndBlockItem,
                            EnterLoop, LeaveLoop, ReturnFromKernel,
                            RunInstruction)


def map_schedule_onto_host_or_device(kernel):
    from loopy.kernel import kernel_state
    assert kernel.state == kernel_state.SCHEDULED

    from functools import partial
    device_prog_name_gen = partial(
                kernel.get_var_name_generator(),
                kernel.target.device_program_name_prefix
                + kernel.name
                + kernel.target.device_program_name_suffix)

    if not kernel.target.split_kernel_at_global_barriers():
        new_schedule = (
            [CallKernel(kernel_name=device_prog_name_gen(),
                        extra_args=[],
                        extra_inames=[])] +
            list(kernel.schedule) +
            [ReturnFromKernel(kernel_name=kernel.name)])
        kernel = kernel.copy(schedule=new_schedule)
    else:
        kernel = map_schedule_onto_host_or_device_impl(
                kernel, device_prog_name_gen)

    return add_extra_args_to_schedule(kernel)


# {{{ Schedule / instruction utilities

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


# {{{ Use / def utilities

def filter_out_subscripts(exprs):
    """
    Remove subscripts from expressions in `exprs`.
    """
    result = set()
    from pymbolic.primitives import Subscript
    for expr in exprs:
        if isinstance(expr, Subscript):
            expr = expr.aggregate
        result.add(expr)
    return result


def filter_items_by_varname(pred, kernel, items):
    """
    Keep only the values in `items` whose variable names satisfy `pred`.
    """
    from pymbolic.primitives import Subscript, Variable
    result = set()
    for item in items:
        base = item
        if isinstance(base, Subscript):
            base = base.aggregate
        if isinstance(base, Variable):
            base = base.name
        if pred(kernel, base):
            result.add(item)
    return result


from functools import partial

filter_temporaries = partial(filter_items_by_varname,
    lambda kernel, name: name in kernel.temporary_variables)


def get_use_set(insn, include_subscripts=True):
    """
    Return the use-set of the instruction, for liveness analysis.
    """
    result = insn.read_dependency_names()
    if not include_subscripts:
        result = filter_out_subscripts(result)
    return result


def get_def_set(insn, include_subscripts=True):
    """
    Return the def-set of the instruction, for liveness analysis.
    """
    result = insn.write_dependency_names()
    if not include_subscripts:
        result = filter_out_subscripts(result)
    return result


def get_temporaries_defined_and_used_in_subrange(
        kernel, schedule, start_idx, end_idx):
    defs = set()
    uses = set()

    for idx in range(start_idx, end_idx + 1):
        sched_item = schedule[idx]
        if isinstance(sched_item, RunInstruction):
            insn = kernel.id_to_insn[sched_item.insn_id]
            defs.update(
                filter_temporaries(
                    kernel, get_def_set(insn)))
            uses.update(
                filter_temporaries(
                    kernel, get_use_set(insn)))

    return defs, uses

# }}}


def add_extra_args_to_schedule(kernel):
    """
    Fill the `extra_args` fields in all the :class:`loopy.schedule.CallKernel`
    instructions in the schedule with global temporaries.
    """
    new_schedule = []

    block_bounds = get_block_boundaries(kernel.schedule)
    for idx, sched_item in enumerate(kernel.schedule):
        if isinstance(sched_item, CallKernel):
            defs, uses = get_temporaries_defined_and_used_in_subrange(
                   kernel, kernel.schedule, idx + 1, block_bounds[idx] - 1)
            # Filter out temporaries that are global.
            extra_args = (tv for tv in defs | uses if
                kernel.temporary_variables[tv].scope == temp_var_scope.GLOBAL
                and
                kernel.temporary_variables[tv].initializer is None)
            new_schedule.append(sched_item.copy(extra_args=sorted(extra_args)))
        else:
            new_schedule.append(sched_item)

    return kernel.copy(schedule=new_schedule)


def map_schedule_onto_host_or_device_impl(kernel, device_prog_name_gen):
    schedule = kernel.schedule
    loop_bounds = get_block_boundaries(schedule)

    # {{{ Inner mapper function

    dummy_call = CallKernel(kernel_name="", extra_args=[], extra_inames=[])
    dummy_return = ReturnFromKernel(kernel_name="")

    def inner_mapper(start_idx, end_idx, new_schedule):
        schedule_required_splitting = False

        i = start_idx
        current_chunk = []
        while i <= end_idx:
            sched_item = schedule[i]

            if isinstance(sched_item, RunInstruction):
                current_chunk.append(sched_item)
                i += 1

            elif isinstance(sched_item, EnterLoop):
                loop_end = loop_bounds[i]
                inner_schedule = []
                loop_required_splitting = inner_mapper(
                    i + 1, loop_end - 1, inner_schedule)

                start_item = schedule[i]
                end_item = schedule[loop_end]

                i = loop_end + 1

                if loop_required_splitting:
                    schedule_required_splitting = True
                    if current_chunk:
                        new_schedule.extend(
                            [dummy_call.copy()] +
                            current_chunk +
                            [dummy_return.copy()])
                    new_schedule.extend(
                        [start_item] +
                        inner_schedule +
                        [end_item])
                    current_chunk = []
                else:
                    current_chunk.extend(
                        [start_item] +
                        inner_schedule +
                        [end_item])

            elif isinstance(sched_item, Barrier):
                if sched_item.kind == "global":
                    # Wrap the current chunk into a kernel call.
                    schedule_required_splitting = True
                    if current_chunk:
                        new_schedule.extend(
                            [dummy_call.copy()] +
                            current_chunk +
                            [dummy_return.copy()])
                    current_chunk = []
                else:
                    current_chunk.append(sched_item)
                i += 1
            else:
                raise LoopyError("unexpected type of schedule item: %s"
                        % type(sched_item).__name__)

        if current_chunk and schedule_required_splitting:
            # Wrap remainder of schedule into a kernel call.
            new_schedule.extend(
                [dummy_call.copy()] +
                current_chunk +
                [dummy_return.copy()])
        else:
            new_schedule.extend(current_chunk)

        return schedule_required_splitting

    # }}}

    new_schedule = []
    split_kernel = inner_mapper(0, len(schedule) - 1, new_schedule)
    if not split_kernel:
        # Wrap everything into a kernel call.
        new_schedule = (
            [dummy_call.copy()] +
            new_schedule +
            [dummy_return.copy()])

    # Assign names, extra_inames to CallKernel / ReturnFromKernel instructions
    inames = []

    for idx, sched_item in enumerate(new_schedule):
        if isinstance(sched_item, CallKernel):
            last_kernel_name = device_prog_name_gen()
            new_schedule[idx] = sched_item.copy(
                kernel_name=last_kernel_name,
                extra_inames=list(inames))
        elif isinstance(sched_item, ReturnFromKernel):
            new_schedule[idx] = sched_item.copy(
                kernel_name=last_kernel_name)
        elif isinstance(sched_item, EnterLoop):
            inames.append(sched_item.iname)
        elif isinstance(sched_item, LeaveLoop):
            inames.pop()

    new_kernel = kernel.copy(schedule=new_schedule)

    return new_kernel

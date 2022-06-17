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
from loopy.schedule import (Barrier, CallKernel, EnterLoop,
                            ReturnFromKernel, RunInstruction)
from loopy.schedule.tools import get_block_boundaries


def map_schedule_onto_host_or_device(kernel):
    # FIXME: Should be idempotent.
    from loopy.kernel import KernelState
    assert kernel.state == KernelState.LINEARIZED

    from functools import partial
    device_prog_name_gen = partial(
                kernel.get_var_name_generator(),
                kernel.target.device_program_name_prefix
                + kernel.name
                + kernel.target.device_program_name_suffix)

    if not kernel.target.split_kernel_at_global_barriers():
        new_schedule = (
            [CallKernel(kernel_name=device_prog_name_gen())] +
            list(kernel.linearization) +
            [ReturnFromKernel(kernel_name=kernel.name)])
        kernel = kernel.copy(linearization=new_schedule)
    else:
        kernel = map_schedule_onto_host_or_device_impl(
                kernel, device_prog_name_gen)

    return kernel


def map_schedule_onto_host_or_device_impl(kernel, device_prog_name_gen):
    schedule = kernel.linearization
    loop_bounds = get_block_boundaries(schedule)

    # {{{ inner mapper function

    dummy_call = CallKernel(kernel_name="")
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
                if sched_item.synchronization_kind == "global":
                    # Wrap the current chunk into a kernel call.
                    schedule_required_splitting = True
                    if current_chunk:
                        new_schedule.extend(
                            [dummy_call.copy()] +
                            current_chunk +
                            [dummy_return.copy()])
                    new_schedule.append(sched_item)
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

    # Assign names to CallKernel / ReturnFromKernel instructions

    for idx, sched_item in enumerate(new_schedule):
        if isinstance(sched_item, CallKernel):
            last_kernel_name = device_prog_name_gen()
            new_schedule[idx] = sched_item.copy(kernel_name=last_kernel_name)
        elif isinstance(sched_item, ReturnFromKernel):
            new_schedule[idx] = sched_item.copy(kernel_name=last_kernel_name)

    new_kernel = kernel.copy(linearization=new_schedule)

    return new_kernel

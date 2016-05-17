from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2016 Matt Wala"

__license__ = """
(unclear)
"""

# TODO: Matt, please replace the license header
# TODO: Should move to loopy.schedule.device_mapping

from pytools import Record
from loopy.diagnostic import LoopyError


class HostForLoop(Record):
    # TOOD: Should have docstring indicating what attributes can occur
    pass


class HostConditional(Record):
    # TOOD: Should have docstring indicating what attributes can occur
    pass


class HostBlock(Record):
    # TOOD: Should have docstring indicating what attributes can occur
    pass


class HostInvokeKernel(Record):
    # TOOD: Should have docstring indicating what attributes can occur
    pass


def map_schedule_onto_host_or_device(kernel):
    from functools import partial
    kernel_name_gen = partial(
            kernel.get_var_name_generator(),
            kernel.target.device_program_name_prefix
            + kernel.name
            + kernel.target.device_program_name_suffix)

    from loopy.schedule import (
        RunInstruction, EnterLoop, LeaveLoop, Barrier,
        CallKernel, ReturnFromKernel)

    # TODO: Assert that the kernel has been scheduled, etc.
    schedule = kernel.schedule

    # Map from loop start to loop end
    loop_bounds = {}
    active_loops = []
    for idx, sched_item in enumerate(schedule):
        if isinstance(sched_item, EnterLoop):
            active_loops.append(idx)
        elif isinstance(sched_item, LeaveLoop):
            loop_bounds[active_loops.pop()] = idx
    del active_loops

    # {{{ Inner mapper function

    def inner_mapper(start_idx, end_idx, new_schedule):
        # XXX: Doesn't do dependency analysis yet....
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
                        new_kernel_name = kernel_name_gen()
                        new_schedule.extend(
                            # TODO: Infer kernel arguments
                            [CallKernel(kernel_name=new_kernel_name)] +
                            # TODO: Load state into here
                            current_chunk +
                            # TODO: Save state right here
                            [ReturnFromKernel(kernel_name=new_kernel_name)])
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
                        new_kernel_name = kernel_name_gen()
                        new_schedule.extend(
                            # TODO: Infer kernel arguments
                            [CallKernel(kernel_name=new_kernel_name)] +
                            # TODO: Load state into here
                            current_chunk +
                            # TODO: Save state right here
                            [ReturnFromKernel(kernel_name=new_kernel_name)])
                    current_chunk = []
                else:
                    current_chunk.append(sched_item)
                i += 1
            else:
                raise LoopyError("unexepcted type of schedule item: %s"
                        % type(sched_item).__name__)

        if current_chunk and schedule_required_splitting:
            # Wrap remainder of schedule into a kernel call.
            new_kernel_name = kernel_name_gen()
            new_schedule.extend(
                # TODO: Infer kernel arguments
                [CallKernel(kernel_name=new_kernel_name)] +
                # TODO: Load state into here
                current_chunk +
                # TODO: Save state right here
                [ReturnFromKernel(kernel_name=new_kernel_name)])
        else:
            new_schedule.extend(current_chunk)

        return schedule_required_splitting

    # }}}

    new_schedule = []
    split_kernel = inner_mapper(0, len(schedule) - 1, new_schedule)
    if not split_kernel:
        # Wrap everything into a kernel call.
        new_schedule = (
            [CallKernel(kernel_name=kernel_name_gen())] +
            new_schedule +
            [ReturnFromKernel(kernel_name=kernel.name)])
    new_kernel = kernel.copy(schedule=new_schedule)
    return new_kernel

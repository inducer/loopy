__copyright__ = "Copyright (C) 2020 Kaushik Kulkarni"

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

from pytools import memoize_method
from loopy.schedule import (EnterLoop, LeaveLoop, CallKernel, ReturnFromKernel,
                            Barrier, BeginBlockItem, gather_schedule_block)


__doc__ = """
.. currentmodule:: loopy.codegen.tools

.. autofunction:: make_codegen_cache_manager

.. autoclass:: CodegenOperationCacheManager
"""


def make_codegen_cache_manager(kernel):
    """
    Returns an instance of :class:`CodegenOperationCacheManager` for *kernel*.
    """
    # {{{ computing active inames for sched indices

    active_inames = []

    for i in range(len(kernel.schedule)):
        if i == 0:
            active_inames.append(frozenset())
            continue

        sched_item = kernel.schedule[i-1]
        if isinstance(sched_item, EnterLoop):
            active_inames.append(active_inames[i-1]
                    | frozenset([sched_item.iname]))
        elif isinstance(sched_item, LeaveLoop):
            assert sched_item.iname in active_inames[i-1]
            active_inames.append(active_inames[i-1] - frozenset([sched_item.iname]))
        else:
            active_inames.append(active_inames[i-1])
    # }}}

    # {{{ computing call kernel indices

    # callkernel_index[i]: schedule index of CallKernel containing the point
    # just before ith-schedule item. None if the point is not a part of any
    # callkernel.
    callkernel_index = []

    for i in range(len(kernel.schedule)):
        if i == 0:
            callkernel_index.append(None)
            continue

        sched_item = kernel.schedule[i-1]

        if isinstance(sched_item, CallKernel):
            callkernel_index.append(i-1)
        elif isinstance(sched_item, ReturnFromKernel):
            callkernel_index.append(None)
        else:
            callkernel_index.append(callkernel_index[i-1])

    # }}}

    # {{{ computing has_barrier_within

    # has_barrier_within[i]: 'True' if the i-th schedule item's is a begin block
    # containing a barrier of if the i-th schedule item is a barrier.

    has_barrier_within = []

    for sched_idx, sched_item in enumerate(kernel.schedule):
        if isinstance(sched_item, BeginBlockItem):
            # TODO: calls to "gather_schedule_block" can be amortized
            _, endblock_index = gather_schedule_block(kernel.schedule,
                    sched_idx)
            has_barrier_within.append(any(
                    isinstance(kernel.schedule[i], Barrier)
                    for i in range(sched_idx+1, endblock_index)))
        elif isinstance(sched_item, Barrier):
            has_barrier_within.append(True)
        else:
            has_barrier_within.append(False)
    # }}}

    return CodegenOperationCacheManager(kernel, active_inames, callkernel_index,
            has_barrier_within)


class CodegenOperationCacheManager:
    """
    Caches operations arising during the codegen pipeline.

    .. attribute:: kernel

        An instance of :class:`loopy.LoopKernel` the cache manager is tied to.

    .. attribute:: active_inames

        An instance of :class:`list`, with the i-th entry being a :class:`frozenset`
        of active inames at the point just before i-th schedule item.

    .. attribute:: callkernel_index

        An instance of :class:`list`, with the i-th entry being the index of
        :class:`loopy.schedule.CallKernel` containing the point just before i-th
        schedule item.

    .. attribute:: has_barrier_within

        An instance of :class:`list`. The list's i-th entry is *True* if the i-th
        schedule item is a :class:`loopy.schedule.BeginBlockItem` containing a
        barrier or if the i-th schedule item is a :class:`loopy.schedule.Barrier`.

    .. automethod:: with_kernel
    .. automethod:: get_parallel_inames_in_a_callkernel
    """
    def __init__(self, kernel, active_inames, callkernel_index, has_barrier_within):
        self.kernel = kernel
        self.active_inames = active_inames
        self.callkernel_index = callkernel_index
        self.has_barrier_within = has_barrier_within

    def with_kernel(self, kernel):
        """
        Returns a new instance of :class:`CodegenOperationCacheManager`
        corresponding to *kernel* if the cached variables in *self* would
        be invalid for *kernel*, else returns *self*.
        """
        if ((self.kernel.instructions != kernel.instructions)
                or (self.kernel.schedule != kernel.schedule)
                or (self.kernel.inames != kernel.inames)):
            # cached values are invalidated, must create a new one
            return make_codegen_cache_manager(kernel)

        return self

    @memoize_method
    def get_insn_ids_for_block_at(self, sched_index):
        """
        Cached variant of :func:`loopy.schedule.get_insn_ids_for_block_at`.
        """
        from loopy.schedule import get_insn_ids_for_block_at
        return get_insn_ids_for_block_at(self.kernel.schedule, sched_index)

    @memoize_method
    def get_parallel_inames_in_a_callkernel(self, callkernel_index):
        """
        Returns a :class:`frozenset` of parallel inames in a callkernel

        :arg callkernel_index: Index of the :class:`loopy.schedule.CallKernel` in the
            :attr:`CodegenOperationCacheManager.kernel`'s schedule, whose parallel
            inames are to be found.
        """

        from loopy.kernel.data import ConcurrentTag
        assert isinstance(self.kernel.schedule[callkernel_index], CallKernel)
        insn_ids_in_subkernel = self.get_insn_ids_for_block_at(callkernel_index)

        inames_in_subkernel = {iname
                               for insn in insn_ids_in_subkernel
                               for iname in self.kernel.insn_inames(insn)}

        return frozenset([iname
                          for iname in inames_in_subkernel
                          if self.kernel.iname_tags_of_type(iname, ConcurrentTag)])


# vim: fdm=marker

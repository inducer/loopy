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

.. autoclass:: CodegenOperationCacheManager
"""


class CodegenOperationCacheManager:
    """
    Caches operations arising during the codegen pipeline.

    .. automethod:: with_kernel
    .. automethod:: get_parallel_inames_in_a_callkernel
    """
    def __init__(self, kernel):
        self._kernel = kernel

    def with_kernel(self, kernel):
        """
        Returns a new instance of :class:`CodegenOperationCacheManager`
        corresponding to *kernel* if the cached variables in *self* would
        be invalid for *kernel*, else returns *self*.
        """
        if ((self._kernel.instructions != kernel.instructions)
                or (self._kernel.schedule != kernel.schedule)
                # TODO: could be more precise by only looking the inames' attributes
                # relevant to CodegenOperationCacheManager
                or (self._kernel.inames != kernel.inames)):
            # cached values are invalidated, must create a new one
            return CodegenOperationCacheManager(kernel)

        return self

    @property
    @memoize_method
    def active_inames(self):
        """
        Returns an instance of :class:`list`, with the i-th entry being a
        :class:`frozenset` of active inames at the point just before i-th schedule
        item.
        """
        active_inames = []

        for i in range(len(self._kernel.schedule)):
            if i == 0:
                active_inames.append(frozenset())
                continue

            sched_item = self._kernel.schedule[i-1]
            if isinstance(sched_item, EnterLoop):
                active_inames.append(active_inames[i-1]
                        | frozenset([sched_item.iname]))
            elif isinstance(sched_item, LeaveLoop):
                assert sched_item.iname in active_inames[i-1]
                active_inames.append(active_inames[i-1]
                                     - frozenset([sched_item.iname]))
            else:
                active_inames.append(active_inames[i-1])

        return active_inames

    @property
    @memoize_method
    def callkernel_index(self):
        """
        Returns an instance of :class:`list`, with the i-th entry being the index of
        :class:`loopy.schedule.CallKernel` containing the point just before i-th
        schedule item.
        """
        callkernel_index = []

        for i in range(len(self._kernel.schedule)):
            if i == 0:
                callkernel_index.append(None)
                continue

            sched_item = self._kernel.schedule[i-1]

            if isinstance(sched_item, CallKernel):
                callkernel_index.append(i-1)
            elif isinstance(sched_item, ReturnFromKernel):
                callkernel_index.append(None)
            else:
                callkernel_index.append(callkernel_index[i-1])

        return callkernel_index

    @property
    @memoize_method
    def has_barrier_within(self):
        """
        Returns an instance of :class:`list`. The list's i-th entry is *True* if the
        i-th schedule item is a :class:`loopy.schedule.BeginBlockItem` containing a
        barrier or if the i-th schedule item is a :class:`loopy.schedule.Barrier`.
        """
        has_barrier_within = []

        for sched_idx, sched_item in enumerate(self._kernel.schedule):
            if isinstance(sched_item, BeginBlockItem):
                # TODO: calls to "gather_schedule_block" can be amortized
                _, endblock_index = gather_schedule_block(self._kernel.schedule,
                        sched_idx)
                has_barrier_within.append(any(
                        isinstance(self._kernel.schedule[i], Barrier)
                        for i in range(sched_idx+1, endblock_index)))
            elif isinstance(sched_item, Barrier):
                has_barrier_within.append(True)
            else:
                has_barrier_within.append(False)

        return has_barrier_within

    @memoize_method
    def get_insn_ids_for_block_at(self, sched_index):
        """
        Cached variant of :func:`loopy.schedule.get_insn_ids_for_block_at`.
        """
        from loopy.schedule import get_insn_ids_for_block_at
        return get_insn_ids_for_block_at(self._kernel.schedule, sched_index)

    @memoize_method
    def get_parallel_inames_in_a_callkernel(self, callkernel_index):
        """
        Returns a :class:`frozenset` of parallel inames in a callkernel

        :arg callkernel_index: Index of the :class:`loopy.schedule.CallKernel` in the
            :attr:`CodegenOperationCacheManager.kernel`'s schedule, whose parallel
            inames are to be found.
        """

        from loopy.kernel.data import ConcurrentTag
        assert isinstance(self._kernel.schedule[callkernel_index], CallKernel)
        insn_ids_in_subkernel = self.get_insn_ids_for_block_at(callkernel_index)

        inames_in_subkernel = {iname
                               for insn in insn_ids_in_subkernel
                               for iname in self._kernel.insn_inames(insn)}

        return frozenset([iname
                          for iname in inames_in_subkernel
                          if self._kernel.iname_tags_of_type(iname, ConcurrentTag)])


# vim: fdm=marker

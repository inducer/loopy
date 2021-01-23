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


from pytools import ImmutableRecord, memoize_method
from loopy.schedule import (EnterLoop, LeaveLoop, CallKernel, ReturnFromKernel,
                            Barrier, BeginBlockItem, gather_schedule_block)


__doc__ = """
.. currentmodule:: loopy.codegen.tools

.. autoclass:: CodegenOperationCacheManager
"""


class CodegenOperationCacheManager(ImmutableRecord):
    """
    Caches operations arising during the codegen pipeline.
    """
    def __init__(self, kernel):
        super().__init__()
        super().__setattr__("kernel", kernel)
        self.find_activate_inames_at_cache = [frozenset()]

    def __setattr__(self, key, val):
        if key == "kernel":
            # overwriting self.kernel is not allowed, as cache will become
            # invalidated
            raise ValueError("Cannot update CodegenOperationCacheManager.kernel,"
                             " create a new instance instead.")

        super().__setattr__(key, val)

    def find_active_inames_at(self, sched_index):
        """
        Returns a :class:`frozenset` of active inames at the point just before
        *sched_index*.
        """
        if len(self.find_activate_inames_at_cache) > sched_index:
            return self.find_activate_inames_at_cache[sched_index]

        last_index = len(self.find_activate_inames_at_cache)
        for sched_index_var in range(last_index, sched_index+1):
            sched_item = self.kernel.schedule[sched_index_var-1]
            if isinstance(sched_item, EnterLoop):
                res = (self.find_active_inames_at_cache[sched_index_var-1]
                        | frozenset([sched_item.iname]))
            elif isinstance(sched_item, LeaveLoop):
                assert sched_item.iname in \
                    self.find_active_inames_at_cache[sched_index_var-1]
                res = (self.find_active_inames_at_cache[sched_index_var-1]
                        - frozenset([sched_item.iname]))
            else:
                res = self.find_active_inames_at_cache[sched_index_var-1]
            self.find_activate_inames_at_cache.append(res)
        return self.find_activate_inames_at_cache[sched_index]

    @memoize_method
    def has_barrier_within(self, sched_index):
        """
        Checks if the kernel's schedule block from *sched_index* contains a
        barrier.
        """
        sched_item = self.kernel.schedule[sched_index]

        if isinstance(sched_item, BeginBlockItem):
            # TODO: calls to "gather_schedule_block" can be amortized
            _, endblock_index = gather_schedule_block(self.kernel.schedule,
                    sched_index)
            return any(self.has_barrier_within(i)
                       for i in range(sched_index+1, endblock_index))
        elif isinstance(sched_item, Barrier):
            return True
        else:
            return False

    @memoize_method
    def get_callkernel_index(self, sched_index):
        """
        Returns index of :class:`loopy.schedule.CallKernel` containing the point
        just before *sched_index*. Return *None* if the point is
        not a part of any sub-kernel.
        """
        if sched_index == 0:
            return None

        sched_item = self.kernel.schedule[sched_index-1]

        if isinstance(sched_item, CallKernel):
            return sched_index-1
        elif isinstance(sched_item, ReturnFromKernel):
            return None
        else:
            return self.get_callkernel_index(sched_index-1)

    @memoize_method
    def get_insn_ids_for_block_at(self, sched_index):
        """
        Cached variant of :func:`loopy.schedule.get_insn_ids_for_block_at`.
        """
        from loopy.schedule import get_insn_ids_for_block_at
        return get_insn_ids_for_block_at(self.kernel.schedule, sched_index)

    @memoize_method
    def get_usable_inames_for_conditional_in_subkernel(self, subknl_index,
                                                       crosses_barrier):
        """
        Returns a :class:`frozenset` of parallel inames are defined within a
        subkernel, BUT:

        - local indices may not be used in conditionals that cross barriers.

        - ILP indices and vector lane indices are not available in loop
          bounds, they only get defined at the innermost level of nesting.
        """

        assert isinstance(self.kernel.schedule[subknl_index], CallKernel)
        insn_ids_in_subkernel = self.get_insn_ids_for_block_at(subknl_index)

        inames_for_subkernel = set(
            iname
            for insn in insn_ids_in_subkernel
            for iname in self.kernel.insn_inames(insn))

        def filterfunc(iname):
            from loopy.kernel.data import (ConcurrentTag, LocalIndexTagBase,
                                           VectorizeTag,
                                           IlpBaseTag)
            return (self.kernel.iname_tags_of_type(iname, ConcurrentTag)
                    and not self.kernel.iname_tags_of_type(iname, VectorizeTag)
                    and not (self.kernel.iname_tags_of_type(iname, LocalIndexTagBase)
                             and crosses_barrier)
                    and not self.kernel.iname_tags_of_type(iname, IlpBaseTag))

        return frozenset([iname
                          for iname in inames_for_subkernel
                          if filterfunc(iname)])

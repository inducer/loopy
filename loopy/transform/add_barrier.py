from __future__ import annotations


__copyright__ = "Copyright (C) 2017 Kaushik Kulkarni"

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


from typing import TYPE_CHECKING

from loopy.kernel import LoopKernel
from loopy.kernel.instruction import BarrierInstruction
from loopy.match import ToMatchConvertible, parse_match
from loopy.transform.instruction import add_dependency
from loopy.translation_unit import for_each_kernel


if TYPE_CHECKING:
    from pytools.tag import Tag


__doc__ = """
.. currentmodule:: loopy

.. autofunction:: add_barrier
"""


# {{{ add_barrier

@for_each_kernel
def add_barrier(
    kernel: LoopKernel,
    insn_before: ToMatchConvertible,
    insn_after: ToMatchConvertible,
    id_based_on: str | None = None,
    tags: frozenset[Tag] | None = None,
    synchronization_kind: str = "global",
    mem_kind: str | None = None,
    within_inames: frozenset[str] | None = None,
) -> LoopKernel:
    """
    Returns a transformed version of *kernel* with an additional
    :class:`loopy.BarrierInstruction` inserted.

    :arg insn_before: Match expression that specifies the instruction(s)
        that the barrier instruction depends on.
    :arg insn_after: Match expression that specifies the instruction(s)
        that depend on the barrier instruction.
    :arg id_based_on: Prefix for the barrier instructions' ID.
    :arg tags: The tag of the group to which the barrier must be added
    :arg synchronization_kind: Kind of barrier to be added. May be "global" or
        "local"
    :arg mem_kind: Type of memory to be synchronized. May be "global" or
        "local". Ignored for "global" barriers. If not supplied, defaults to
        *synchronization_kind*
    :arg within_inames: A :class:`frozenset` of inames identifying the loops
        within which the barrier will be executed.
    """

    assert isinstance(kernel, LoopKernel)

    if mem_kind is None:
        mem_kind = synchronization_kind

    if id_based_on is None:
        id = kernel.make_unique_instruction_id(
            based_on=synchronization_kind[0]+"_barrier")
    else:
        id = kernel.make_unique_instruction_id(based_on=id_based_on)

    match = parse_match(insn_before)
    depends_on = frozenset(
        [insn.id for insn in kernel.instructions if match(kernel, insn)])

    barrier_to_add = BarrierInstruction(depends_on=depends_on,
                                        depends_on_is_final=True,
                                        id=id,
                                        within_inames=within_inames,
                                        tags=tags,
                                        synchronization_kind=synchronization_kind,
                                        mem_kind=mem_kind)

    new_kernel = kernel.copy(instructions=[*kernel.instructions, barrier_to_add])
    new_kernel = add_dependency(
        new_kernel, insn_match=insn_after, depends_on="id:" + id
    )

    return new_kernel

# }}}

# vim: foldmethod=marker

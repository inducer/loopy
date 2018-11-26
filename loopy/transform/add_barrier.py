from __future__ import division, absolute_import

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


from loopy.kernel.instruction import BarrierInstruction
from loopy.match import parse_match
from loopy.transform.instruction import add_dependency

__doc__ = """
.. currentmodule:: loopy

.. autofunction:: add_barrier
"""


# {{{ add_barrier

def add_barrier(knl, insn_before="", insn_after="", id_based_on=None,
                tags=None, synchronization_kind="global", mem_kind=None):
    """Takes in a kernel that needs to be added a barrier and returns a kernel
    which has a barrier inserted into it. It takes input of 2 instructions and
    then adds a barrier in between those 2 instructions. The expressions can
    be any inputs that are understood by :func:`loopy.match.parse_match`.

    :arg insn_before: String expression that specifies the instruction(s)
        before the barrier which is to be added
    :arg insn_after: String expression that specifies the instruction(s) after
        the barrier which is to be added
    :arg id: String on which the id of the barrier would be based on.
    :arg tags: The tag of the group to which the barrier must be added
    :arg synchronization_kind: Kind of barrier to be added. May be "global" or
        "local"
    :arg kind: Type of memory to be synchronied. May be "global" or "local". Ignored
        for "global" bariers.  If not supplied, defaults to *synchronization_kind*
    """

    if mem_kind is None:
        mem_kind = synchronization_kind

    if id_based_on is None:
        id = knl.make_unique_instruction_id(
            based_on=synchronization_kind[0]+"_barrier")
    else:
        id = knl.make_unique_instruction_id(based_on=id_based_on)

    match = parse_match(insn_before)
    insn_before_list = [insn.id for insn in knl.instructions if match(knl,
                        insn)]

    barrier_to_add = BarrierInstruction(depends_on=frozenset(insn_before_list),
                                        depends_on_is_final=True,
                                        id=id,
                                        tags=tags,
                                        synchronization_kind=synchronization_kind,
                                        mem_kind=mem_kind)

    new_knl = knl.copy(instructions=knl.instructions + [barrier_to_add])
    new_knl = add_dependency(kernel=new_knl,
                             insn_match=insn_after,
                             depends_on="id:"+id)

    return new_knl

# }}}

# vim: foldmethod=marker

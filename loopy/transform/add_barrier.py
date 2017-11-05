from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2017 Andreas Kloeckner"

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
from loopy.diagnostic import LoopyError

__doc__ = """
.. currentmodule:: loopy

.. autofunction:: add_barrier
"""


# {{{ add_barrier

def add_barrier(knl, id, insn0_expr, insn1_expr, tags=None, kind='global'):
    """Takes in a kernel that needs to be added a barrier and returns a kernel
    which has a barrier inserted into it. It takes input of 2 instructions and
    then adds a barrier in between those 2 instructions. See **parse_match()**
    for the expected format of expression.

    :arg id: String which would be the id of the barrier
    :arg id_insn0: String expression that specifies the first instruction
    :arg id_insn1: String expression that specifies the second instruction
    :arg tags: The tag of the group to which the barrier must be added
    :arg kind: Specifies whether the barrier is a local or a global one
    """

    match0 = parse_match(insn0_expr)
    insn0_set = [insn.id for insn in knl.instructions if match0(knl, insn)]
    match1 = parse_match(insn1_expr)
    insn1_set = [insn for insn in knl.instructions if match1(knl, insn)]

    if len(insn0_set)!=1 or len(insn1_set)!=1:
        raise LoopyError("The instruction expressions should match only 1\
instruction.")

    barrier_to_add = BarrierInstruction(depends_on=frozenset(insn0_set),
                                        depends_on_is_final=True,
                                        id = id,
                                        tags=tags,
                                        kind=kind)
    new_knl = knl.copy(instructions=knl.instructions + [barrier_to_add])
    print("id:"+id)
    new_knl = add_dependency(kernel=new_knl,
                            insn_match=insn1_expr,
                            depends_on="id:"+id)

    return new_knl

# }}}

# vim: foldmethod=marker

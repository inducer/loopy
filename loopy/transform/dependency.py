"""
.. autoclass:: AccessMapFinder
.. autofunction:: narrow_dependencies
"""
__copyright__ = "Copyright (C) 2022 Addison Alvey-Blanco"

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

import islpy as isl

from loopy.kernel.instruction import HappensAfter
from loopy.kernel import LoopKernel, InstructionBase
from loopy.translation_unit import for_each_kernel
from loopy.symbolic import WalkMapper, get_access_map, \
                           UnableToDetermineAccessRangeError
from loopy.typing import Expression

import pymbolic.primitives as p
from typing import List, Dict, Sequence
from pyrsistent import pmap, PMap
from warnings import warn


class AccessMapFinder(WalkMapper):
    """Finds and stores relations representing the accesses to an array by
    statement instances. Access maps can be found using an instruction's ID and
    a variable's name. Essentially a specialized version of
    BatchedAccessMapMapper.
    """

    def __init__(self, knl: LoopKernel) -> None:
        self.kernel = knl
        self._access_maps: PMap[str, PMap[str, isl.Map]] = pmap({})
        from collections import defaultdict  # FIXME remove this
        self.bad_subscripts: Dict[str, List[Expression]] = defaultdict(list)

        super().__init__()

    def get_map(self, insn_id: str, variable_name: str) -> isl.Map:
        """Retrieve an access map indexed by an instruction ID and variable
        name.
        """
        try:
            return self._access_maps[insn_id][variable_name]
        except KeyError:
            return None

    def map_subscript(self, expr, insn_id):
        domain = self.kernel.get_inames_domain(
                self.kernel.id_to_insn[insn_id].within_inames
        )
        WalkMapper.map_subscript(self, expr, insn_id)

        assert isinstance(expr.aggregate, p.Variable)

        arg_name = expr.aggregate.name
        subscript = expr.index_tuple

        try:
            access_map = get_access_map(
                    domain, subscript, self.kernel.assumptions)
        except UnableToDetermineAccessRangeError:
            # may not have enough info to generate access map at current point
            self.bad_subscripts[arg_name].append(expr)
            return

        # analyze what we have in our access map dict before storing map
        insn_to_args = self._access_maps.get(insn_id)
        if insn_to_args is not None:
            existing_relation = insn_to_args.get(arg_name)

            if existing_relation is not None:
                access_map |= existing_relation

            self._access_maps = self._access_maps.set(
                    insn_id, self._access_maps[insn_id].set(
                        arg_name, access_map))

        else:
            self._access_maps = self._access_maps.set(
                    insn_id, pmap({arg_name: access_map}))

    def map_linear_subscript(self, expr, insn_id):
        raise NotImplementedError("linear subscripts cannot be used with "
                                  "precise dependency finding. Use "
                                  "multidimensional accesses to take advantage "
                                  "of this feature.")

    def map_reduction(self, expr, insn_id):
        return WalkMapper.map_reduction(self, expr, insn_id)

    def map_type_cast(self, expr, insn_id):
        return self.rec(expr.child, insn_id)

    def map_sub_array_ref(self, expr, insn_id):
        raise NotImplementedError("Not yet implemented")


@for_each_kernel
def narrow_dependencies(knl: LoopKernel) -> LoopKernel:
    """Attempt to relax the dependency requirements between instructions in
    a loopy program. Computes the precise, statement-instance level dependencies
    between statements by using the affine array accesses of each statement. The
    :attr:`loopy.Instruction.happens_after` of each instruction is updated
    accordingly.
    """

    # precompute access maps for each instruction
    amf = AccessMapFinder(knl)
    for insn in knl.instructions:
        amf(insn.assignee, insn.id)
        amf(insn.expression, insn.id)

    writer_map = knl.writer_map()
    reader_map = knl.reader_map()

    writers = {insn.id: insn.write_dependency_names() - insn.within_inames
               for insn in knl.instructions}
    readers = {insn.id: insn.read_dependency_names() - insn.within_inames
               for insn in knl.instructions}

    pu.db

    new_insns = []

    return knl.copy(instructions=new_insns)

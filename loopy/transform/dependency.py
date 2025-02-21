from __future__ import annotations


"""
.. autoclass:: AccessMapFinder
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

from pyrsistent import PMap, pmap

import islpy as isl
import pymbolic.primitives as p

from loopy.kernel import LoopKernel
from loopy.symbolic import (
    UnableToDetermineAccessRangeError,
    WalkMapper,
    get_access_map,
)
from loopy.typing import Expression


class AccessMapFinder(WalkMapper):
    def __init__(self, knl: LoopKernel) -> None:
        self.kernel = knl
        self._access_maps: PMap[str, PMap[str, isl.Map]] = pmap({})  # type: ignore
        from collections import defaultdict

        self.bad_subscripts: dict[str, list[Expression]] = defaultdict(list)

        super().__init__()

    def get_map(self, insn_id: str, variable_name: str) -> isl.Map | None:  # type: ignore
        """Retrieve an access map indexed by an instruction ID and variable
        name.
        """
        try:
            return self._access_maps[insn_id][variable_name]
        except KeyError:
            return None

    def get_accessed_variables(self, insn_id: str) -> set[str] | None:
        try:
            return set(self._access_maps[insn_id].keys())
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
            access_map = get_access_map(domain, subscript, self.kernel.assumptions)
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
                insn_id, self._access_maps[insn_id].set(arg_name, access_map)
            )

        else:
            self._access_maps = self._access_maps.set(
                insn_id, pmap({arg_name: access_map})
            )

    def map_linear_subscript(self, expr, insn_id):
        raise NotImplementedError(
            "linear subscripts cannot be used with "
            "precise dependency finding. Use "
            "multidimensional accesses to take advantage "
            "of this feature."
        )

    def map_reduction(self, expr, insn_id):
        return WalkMapper.map_reduction(self, expr, insn_id)

    def map_type_cast(self, expr, insn_id):
        return self.rec(expr.child, insn_id)

    def map_sub_array_ref(self, expr, insn_id):
        raise NotImplementedError("Not yet implemented")

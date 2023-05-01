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
from islpy import dim_type

from loopy.kernel.instruction import HappensAfter
from loopy.kernel import LoopKernel
from loopy.translation_unit import for_each_kernel
from loopy.symbolic import WalkMapper, get_access_map, \
                           UnableToDetermineAccessRangeError
from loopy.typing import Expression

import pymbolic.primitives as p
from typing import List, Dict
from pyrsistent import pmap, PMap
from warnings import warn
from functools import reduce


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

    def get_map(self, insn_id: str, variable_name: str) -> isl.Map | None:
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

    .. note:: Requires an existing :attr:`loopy.Instruction.happens_after` to be
    defined.
    """

    def make_inames_unique(relation: isl.Map) -> isl.Map:
        """Append a single quote to all inames in the output of a map to ensure
        input/output inames do not match
        """
        for idim in range(relation.dim(dim_type.out)):
            iname = relation.get_dim_name(dim_type.out, idim) + "'"
            relation = relation.set_dim_name(dim_type.out, iname)

        return relation

    # precompute access maps for each instruction
    amf = AccessMapFinder(knl)
    for insn in knl.instructions:
        amf(insn.assignee, insn.id)
        amf(insn.expression, insn.id)

    # determine transitive dependencies between statements
    transitive_deps = {}

    deps_dag = {insn.id: insn.depends_on for insn in knl.instructions}

    from pytools.graph import compute_topological_order
    t_sort = compute_topological_order(deps_dag)

    for insn_id in t_sort:
        transitive_deps[insn_id] = reduce(
                frozenset.union,
                (transitive_deps.get(dep, frozenset([dep]))
                      for dep in
                      knl.id_to_insn[insn_id].depends_on),
                frozenset())

    # compute and store precise dependencies
    new_insns = []
    for insn in knl.instructions:
        happens_after = insn.happens_after

        # get access maps
        for dependency, variable in happens_after:
            assert isinstance(insn.id, str) # stop complaints

            after_map = amf.get_map(insn.id, variable)
            before_map = amf.get_map(dependency, variable)

            # scalar case(s)
            if before_map is None or after_map is None:
                warn("unable to determine the access map for %s. Defaulting to "
                     "a conservative dependency relation between %s and %s" %
                     (dependency, insn.id, dependency))

                # clean up any deps that do not contain a variable name
                happens_after = {insn_id: dep
                                 for insn_id, dep in happens_after.items()
                                 if dep.variable_name is not None}

                continue

            dims = [before_map.dim(dim_type.out),
                    before_map.dim(dim_type.in_),
                    after_map.dim(dim_type.out),
                    after_map.dim(dim_type.in_)]

            for i in range(len(dims)):
                if i == 0 or i == 1:
                    scalar_insn = dependency
                else:
                    scalar_insn = insn.id

                if dims[i] == 0:
                    warn("found evidence of a scalar access in %s. Defaulting "
                         "to a conservative dependency relation between "
                         "%s and %s" % (scalar_insn, dependency, insn.id))

                    # clean up any deps that do not contain a variable name
                    happens_after = {insn_id: dep
                                     for insn_id, dep in happens_after.items()
                                     if dep.variable_name is not None}

                    continue

            # non-scalar accesses
            dep_map = after_map.apply_range(before_map.reverse())
            dep_map -= dep_map.identity(dep_map.get_space())

            if insn.happens_after[dependency] is not None:
                lex_map = insn.happens_after[dependency].instances_rel
            else:
                lex_map = dep_map.lex_lt_map(dep_map)

            assert lex_map is not None
            if lex_map.space != dep_map.space:
                lex_map = make_inames_unique(lex_map)
                dep_map = make_inames_unique(dep_map)

                lex_map, dep_map = isl.align_two(lex_map, dep_map)

            dep_map = dep_map & lex_map

            happens_after = dict(
                    list(happens_after.items())
                    + [HappensAfter(variable, dep_map)])

        # clean up any deps that do not contain a variable name
        happens_after = {insn_id: dep
                         for insn_id, dep in happens_after.items()
                         if dep.variable_name is not None}

        new_insns.append(insn.copy(happens_after=happens_after))

    return knl.copy(instructions=new_insns)

"""
.. autofunction:: add_lexicographic_happens_after
"""

__copyright__ = "Copyright (C) 2023 Addison Alvey-Blanco"

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

from loopy import LoopKernel
from loopy.kernel.instruction import HappensAfter
from loopy.translation_unit import for_each_kernel


@for_each_kernel
def add_lexicographic_happens_after(knl: LoopKernel) -> LoopKernel:
    """Construct a sequential dependency specification between each instruction
    and the instruction immediately before it. This dependency information
    contains a lexicographic map which acts as a description of the precise,
    statement-instance level dependencies between statements.
    """

    new_insns = []
    for iafter, insn_after in enumerate(knl.instructions):
        if iafter == 0:
            new_insns.append(insn_after)

        else:
            insn_before = knl.instructions[iafter-1]

            domain_before = knl.get_inames_domain(insn_before.within_inames)
            domain_after = knl.get_inames_domain(insn_after.within_inames)

            shared_inames = insn_before.within_inames & insn_after.within_inames

            happens_after = isl.Map.from_domain_and_range(
                    domain_before,
                    domain_after)

            for idim in range(happens_after.dim(dim_type.out)):
                happens_after = happens_after.set_dim_name(
                        dim_type.out,
                        idim,
                        happens_after.get_dim_name(dim_type.out, idim) + "'")

            shared_inames_order_before = [
                    domain_before.get_dim_name(dim_type.out, idim)
                    for idim in range(domain_before.dim(dim_type.out))
                    if domain_before.get_dim_name(dim_type.out, idim)
                    in shared_inames]

            shared_inames_order_after = [
                    domain_after.get_dim_name(dim_type.out, idim)
                    for idim in range(domain_after.dim(dim_type.out))
                    if domain_after.get_dim_name(dim_type.out, idim)
                    in shared_inames]

            assert shared_inames_order_after == shared_inames_order_before
            shared_inames_order = shared_inames_order_after

            affs_in = isl.affs_from_space(happens_after.domain().space)
            affs_out = isl.affs_from_space(happens_after.range().space)

            lex_map = isl.Map.empty(happens_after.space)
            for iinnermost, innermost_iname in enumerate(shared_inames):
                innermost_map = affs_in[innermost_iname].lt_map(
                        affs_out[innermost_iname + "'"])

                for outer_iname in shared_inames_order[:iinnermost]:
                    innermost_map = innermost_map & (
                            affs_in[outer_iname].eq_map(
                                affs_out[outer_iname + "'"]))

                lex_map = lex_map | innermost_map

            happens_after = happens_after & lex_map

            new_happens_after = {
                    insn_before.id: HappensAfter(None, happens_after)}

            insn_after = insn_after.copy(happens_after=new_happens_after)

            new_insns.append(insn_after)

    return knl.copy(instructions=new_insns)


# vim: foldmethod=marker

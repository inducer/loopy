"""
.. autofunction:: add_lexicographic_happens_after
.. autofunction:: find_data_dependencies
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
def find_data_dependencies(knl: LoopKernel) -> LoopKernel:
    """Compute coarse-grained dependencies between statements in a Loopy program.
    Finds data dependencies based on the Bernstein Condition, i.e., statement S2
    depends on statement S1 if the union of true, anti, and flow dependencies
    between the two statements is non-empty and the instructions actually
    execute.
    """

    reader_map = knl.reader_map()
    writer_map = knl.writer_map()

    readers = {insn.id: insn.read_dependency_names() - insn.within_inames
               for insn in knl.instructions}
    writers = {insn.id: insn.write_dependency_names() - insn.within_inames
               for insn in knl.instructions}

    ordered_insns = {insn.id: i for i, insn in enumerate(knl.instructions)}

    new_insns = []
    for insn in knl.instructions:
        happens_after = insn.happens_after

        # read after write
        for variable in readers[insn.id]:
            for writer in writer_map.get(variable, set()) - {insn.id}:
                if ordered_insns[writer] < ordered_insns[insn.id]:
                    happens_after = dict(
                            list(happens_after.items())
                            + [(writer, HappensAfter(variable, None))])

        for variable in writers[insn.id]:
            # write after read
            for reader in reader_map.get(variable, set()) - {insn.id}:
                if ordered_insns[reader] < ordered_insns[insn.id]:
                    happens_after = dict(
                            list(happens_after.items())
                            + [(reader, HappensAfter(variable, None))])

            # write after write
            for writer in writer_map.get(variable, set()) - {insn.id}:
                if ordered_insns[writer] < ordered_insns[insn.id]:
                    happens_after = dict(
                            list(happens_after.items())
                            + [(writer, HappensAfter(variable, None))])

        # remove dependencies that do not specify a variable name
        happens_after = {
                insn_id: dependency
                for insn_id, dependency in happens_after.items()
                if dependency.variable_name is not None}

        new_insns.append(insn.copy(happens_after=happens_after))

    return knl.copy(instructions=new_insns)


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

            insn_before = knl.instructions[iafter - 1]
            shared_inames = insn_after.within_inames & insn_before.within_inames

            domain_before = knl.get_inames_domain(insn_before.within_inames)
            domain_after = knl.get_inames_domain(insn_after.within_inames)
            happens_before = isl.Map.from_domain_and_range(
                    domain_before, domain_after
            )

            for idim in range(happens_before.dim(dim_type.out)):
                happens_before = happens_before.set_dim_name(
                        dim_type.out, idim,
                        happens_before.get_dim_name(dim_type.out, idim) + "'"
                )

            n_inames_before = happens_before.dim(dim_type.in_)
            happens_before_set = happens_before.move_dims(
                    dim_type.out, 0,
                    dim_type.in_, 0,
                    n_inames_before).range()

            shared_inames_order_before = [
                    domain_before.get_dim_name(dim_type.out, idim)
                    for idim in range(domain_before.dim(dim_type.out))
                    if domain_before.get_dim_name(dim_type.out, idim)
                    in shared_inames
            ]
            shared_inames_order_after = [
                    domain_after.get_dim_name(dim_type.out, idim)
                    for idim in range(domain_after.dim(dim_type.out))
                    if domain_after.get_dim_name(dim_type.out, idim)
                    in shared_inames
            ]
            assert shared_inames_order_after == shared_inames_order_before
            shared_inames_order = shared_inames_order_after

            affs = isl.affs_from_space(happens_before_set.space)

            lex_set = isl.Set.empty(happens_before_set.space)
            for iinnermost, innermost_iname in enumerate(shared_inames_order):

                innermost_set = affs[innermost_iname].lt_set(
                        affs[innermost_iname+"'"]
                )

                for outer_iname in shared_inames_order[:iinnermost]:
                    innermost_set = innermost_set & (
                            affs[outer_iname].eq_set(affs[outer_iname + "'"])
                    )

                lex_set = lex_set | innermost_set

            lex_map = isl.Map.from_range(lex_set).move_dims(
                    dim_type.in_, 0,
                    dim_type.out, 0,
                    n_inames_before)

            happens_before = happens_before & lex_map

            new_happens_after = {
                insn_before.id: HappensAfter(None, happens_before)
            }

            insn_after = insn_after.copy(happens_after=new_happens_after)

            new_insns.append(insn_after)

    return knl.copy(instructions=new_insns)


# vim: foldmethod=marker

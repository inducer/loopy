from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2018 Tianjiao Sun"

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

from loopy.diagnostic import LoopyError
from loopy.kernel.instruction import CallInstruction
from loopy.symbolic import SubArrayRef

__doc__ = """
.. currentmodule:: loopy

.. autofunction:: pack_and_unpack_args_for_call
"""


# {{{ main entrypoint

def pack_and_unpack_args_for_call(kernel, call_name, args=None):
    """
    """
    new_domains = []
    new_tmps = kernel.temporary_variables.copy()
    new_calls = {}

    for insn in kernel.instructions:
        if not isinstance(insn, CallInstruction):
            # pack and unpack call only be done for CallInstructions.
            continue

        in_knl_callable = kernel.scoped_functions[
                insn.expression.function.name]

        if in_knl_callable.name != call_name:
            # not the function we're looking for.
            continue
        in_knl_callable = in_knl_callable.with_packing_for_args()

        vng = kernel.get_var_name_generator()
        ing = kernel.get_instruction_id_generator()

        parameters = insn.expression.parameters
        if args is None:
            args = [par.subscript.aggregate.name for par in parameters if
            isinstance(par, SubArrayRef)] + [assignee.subscript.aggregate.name for
                    assignee in insn.assignees if isinstance(assignee, SubArrayRef)]

        # {{{ sanity checks for args

        for arg in args:
            found_sub_array_ref = False
            for par in parameters + insn.assignees:
                if isinstance(par, SubArrayRef) and (
                        par.subscript.aggregate.name == arg):
                    found_sub_array_ref = True
                    break
            if not found_sub_array_ref:
                raise LoopyError("No match found for packing arg '%s' of call '%s' "
                        "at insn '%s'." % (arg, call_name, insn.id))

        # }}}

        packing = []
        unpacking = []
        new_id_to_parameters = {}

        from loopy.kernel.data import IlpBaseTag, VectorizeTag
        import islpy as isl
        from pymbolic import var

        dim_type = isl.dim_type.set
        ilp_inames = set(iname for iname in insn.within_inames if isinstance(
            kernel.iname_to_tag.get(iname), (IlpBaseTag, VectorizeTag)))
        new_ilp_inames = set()
        ilp_inames_map = {}
        for iname in ilp_inames:
            new_iname_name = vng(iname + "_ilp")
            ilp_inames_map[var(iname)] = var(new_iname_name)
            new_ilp_inames.add(new_iname_name)
        for iname in ilp_inames:
            new_domain = kernel.get_inames_domain(iname).copy()
            for i in range(new_domain.n_dim()):
                old_iname = new_domain.get_dim_name(dim_type, i)
                if old_iname in ilp_inames:
                    new_domain = new_domain.set_dim_name(
                        dim_type, i, ilp_inames_map[var(old_iname)].name)
            new_domains.append(new_domain)

        from pymbolic.mapper.substitutor import make_subst_func
        from loopy.symbolic import SubstitutionMapper

        id_to_parameters = tuple(enumerate(parameters)) + tuple(
                (-i-1, assignee) for i, assignee in enumerate(insn.assignees))

        for id, p in id_to_parameters:
            if isinstance(p, SubArrayRef) and p.subscript.aggregate.name in args:
                new_swept_inames = ilp_inames_map.copy()
                for iname in p.swept_inames:
                    new_swept_inames[iname] = var(vng(iname.name + "_pack"))
                new_domain = kernel.get_inames_domain(iname.name).copy()
                for i in range(new_domain.n_dim()):
                    old_iname = new_domain.get_dim_name(dim_type, i)
                    new_domain = new_domain.set_dim_name(
                        dim_type, i, new_swept_inames[var(old_iname)].name)
                new_domains.append(new_domain)

                arg = p.subscript.aggregate.name
                pack_name = vng(arg + "_pack")

                from loopy.kernel.data import (TemporaryVariable,
                        temp_var_scope)

                pack_tmp = TemporaryVariable(
                    name=pack_name,
                    dtype=kernel.arg_dict[arg].dtype,
                    scope=temp_var_scope.PRIVATE,
                )

                new_tmps[pack_name] = pack_tmp

                from loopy import Assignment
                subst_mapper = SubstitutionMapper(make_subst_func(
                    new_swept_inames))

                # {{{ getting the lhs assignee

                arg_in_caller = kernel.arg_dict[arg]

                from loopy.isl_helpers import simplify_via_aff, make_slab

                flatten_index = simplify_via_aff(
                        sum(dim_tag.stride*idx for dim_tag, idx in
                        zip(arg_in_caller.dim_tags, p.subscript.index_tuple)))

                new_indices = []
                for dim_tag in in_knl_callable.arg_id_to_descr[id].dim_tags:
                    ind = flatten_index // dim_tag.stride
                    flatten_index -= (dim_tag.stride * ind)
                    new_indices.append(ind)

                new_indices = tuple(simplify_via_aff(i) for i in new_indices)

                lhs_assignee = subst_mapper(var(pack_name).index(new_indices))

                # }}}

                packing.append(Assignment(
                    assignee=lhs_assignee,
                    expression=subst_mapper.map_subscript(p.subscript),
                    within_inames=insn.within_inames - ilp_inames | set(
                        new_swept_inames[i].name for i in p.swept_inames) | (
                            new_ilp_inames),
                    depends_on=insn.depends_on,
                    id=ing(insn.id+"_pack")
                ))

                unpacking.append(Assignment(
                    expression=lhs_assignee,
                    assignee=subst_mapper.map_subscript(p.subscript),
                    within_inames=insn.within_inames - ilp_inames | set(
                        new_swept_inames[i].name for i in p.swept_inames) | (
                            new_ilp_inames),
                    depends_on=frozenset([insn.id]),
                    id=ing(insn.id+"_unpack")
                ))

                # {{{ getting the new swept inames

                updated_swept_inames = []

                for i, _ in enumerate(
                        in_knl_callable.arg_id_to_descr[id].shape):
                    updated_swept_inames.append(var(vng("i_packsweep_"+arg)))

                ctx = kernel.isl_context
                space = isl.Space.create_from_names(ctx,
                        set=[iname.name for iname in updated_swept_inames])
                iname_set = isl.BasicSet.universe(space)
                for iname, axis_length in zip(updated_swept_inames,
                        in_knl_callable.arg_id_to_descr[id].shape):
                    iname_set = iname_set & make_slab(space, iname.name, 0,
                            axis_length)
                new_domains = new_domains + [iname_set]

                # }}}

                new_id_to_parameters[id] = SubArrayRef(tuple(updated_swept_inames),
                    (var(pack_name).index(tuple(updated_swept_inames))))
            else:
                new_id_to_parameters[id] = p

        if packing:
            subst_mapper = SubstitutionMapper(make_subst_func(ilp_inames_map))
            new_insn = insn.with_transformed_expressions(subst_mapper)
            new_params = [new_id_to_parameters[i] for i, _ in
                    enumerate(parameters)]
            new_assignees = [new_id_to_parameters[-i-1] for i, _ in
                    enumerate(insn.assignees)]
            new_params = [subst_mapper(p) for p in new_params]
            new_assignees = tuple(subst_mapper(a) for a in new_assignees)
            packing.append(
                new_insn.copy(
                    depends_on=new_insn.depends_on | set(
                        pack.id for pack in packing),
                    within_inames=new_insn.within_inames - ilp_inames | (
                        new_ilp_inames),
                    expression=new_insn.expression.function(*new_params),
                    assignees=new_assignees
                )
            )
            new_calls[insn] = packing + unpacking

    if new_calls:
        new_instructions = []
        for insn in kernel.instructions:
            if insn in new_calls:
                new_instructions.extend(new_calls[insn])
            else:
                new_instructions.append(insn)
        kernel = kernel.copy(
            domains=kernel.domains + new_domains,
            instructions=new_instructions,
            temporary_variables=new_tmps
        )

    return kernel

# }}}


# vim: foldmethod=marker

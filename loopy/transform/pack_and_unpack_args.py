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
    Returns a a copy of *kernel* with instructions appended to copy the
    arguments in *args* to match the alignment expected by the *call_name* in
    the kernel. The arguments are copied back to *args* with the appropriate
    data layout.

    :arg call_name: An instance of :class:`str` denoting the function call in
        the *kernel*.
    :arg args: A list of the arguments as instances of :class:`str` which must
        be packed and unpacked. If set *None*, it is interpreted that all the
        array arguments would be packed anf unpacked.
    """
    new_domains = []
    new_tmps = kernel.temporary_variables.copy()
    old_insn_to_new_insns = {}

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

        assert isinstance(args, list)

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

        # dict to store the new assignees and parameters, the mapping pattern
        # from id to parameters is identical to InKernelCallable.arg_id_to_dtype
        id_to_parameters = tuple(enumerate(parameters)) + tuple(
                (-i-1, assignee) for i, assignee in enumerate(insn.assignees))
        new_id_to_parameters = {}

        for id, p in id_to_parameters:
            if isinstance(p, SubArrayRef) and p.subscript.aggregate.name in args:
                new_pack_inames = ilp_inames_map.copy()  # packing-specific inames
                new_unpack_inames = ilp_inames_map.copy()  # unpacking-specific iname

                for iname in p.swept_inames:
                    new_pack_inames[iname] = var(vng(iname.name + "_pack"))
                    new_unpack_inames[iname] = var(vng(iname.name + "_unpack"))

                # Updating the domains corresponding to the new inames.
                new_domain_pack = kernel.get_inames_domain(iname.name).copy()
                new_domain_unpack = kernel.get_inames_domain(iname.name).copy()
                for i in range(new_domain_pack.n_dim()):
                    old_iname = new_domain_pack.get_dim_name(dim_type, i)
                    if var(old_iname) in new_pack_inames:
                        new_domain_pack = new_domain_pack.set_dim_name(
                            dim_type, i, new_pack_inames[var(old_iname)].name)
                        new_domain_unpack = new_domain_unpack.set_dim_name(
                            dim_type, i, new_unpack_inames[var(old_iname)].name)
                new_domains.append(new_domain_pack)
                new_domains.append(new_domain_unpack)

                arg = p.subscript.aggregate.name
                pack_name = vng(arg + "_pack")

                from loopy.kernel.data import (TemporaryVariable,
                        temp_var_scope)

                pack_tmp = TemporaryVariable(
                    name=pack_name,
                    dtype=kernel.arg_dict[arg].dtype,
                    dim_tags=in_knl_callable.arg_id_to_descr[id].dim_tags,
                    shape=in_knl_callable.arg_id_to_descr[id].shape,
                    scope=temp_var_scope.PRIVATE,
                )

                new_tmps[pack_name] = pack_tmp

                from loopy import Assignment
                pack_subst_mapper = SubstitutionMapper(make_subst_func(
                    new_pack_inames))
                unpack_subst_mapper = SubstitutionMapper(make_subst_func(
                    new_unpack_inames))

                # {{{ getting the lhs for packing and rhs for unpacking

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

                pack_lhs_assignee = pack_subst_mapper(
                        var(pack_name).index(new_indices))
                unpack_rhs = unpack_subst_mapper(
                        var(pack_name).index(new_indices))

                # }}}

                packing.append(Assignment(
                    assignee=pack_lhs_assignee,
                    expression=pack_subst_mapper.map_subscript(p.subscript),
                    within_inames=insn.within_inames - ilp_inames | set(
                        new_pack_inames[i].name for i in p.swept_inames) | (
                            new_ilp_inames),
                    depends_on=insn.depends_on,
                    id=ing(insn.id+"_pack"),
                    depends_on_is_final=True
                ))

                unpacking.append(Assignment(
                    expression=unpack_rhs,
                    assignee=unpack_subst_mapper.map_subscript(p.subscript),
                    within_inames=insn.within_inames - ilp_inames | set(
                        new_unpack_inames[i].name for i in p.swept_inames) | (
                            new_ilp_inames),
                    id=ing(insn.id+"_unpack"),
                    depends_on=frozenset([insn.id]),
                    depends_on_is_final=True
                ))

                # {{{ creating the sweep inames for the new sub array refs

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
            new_params = tuple(subst_mapper(new_id_to_parameters[i]) for i, _ in
                    enumerate(parameters))
            new_assignees = tuple(subst_mapper(new_id_to_parameters[-i-1])
                    for i, _ in enumerate(insn.assignees))
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
            old_insn_to_new_insns[insn] = packing + unpacking

    if old_insn_to_new_insns:
        new_instructions = []
        for insn in kernel.instructions:
            if insn in old_insn_to_new_insns:
                # Replacing the current instruction with the group of
                # instructions including the packing and unpacking instructions
                new_instructions.extend(old_insn_to_new_insns[insn])
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

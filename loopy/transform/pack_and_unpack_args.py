__copyright__ = "Copyright (C) 2018 Tianjiao Sun, Kaushik Kulkarni"

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

from immutables import Map
from loopy.diagnostic import LoopyError
from loopy.kernel.instruction import CallInstruction
from loopy.translation_unit import TranslationUnit
from loopy.kernel import LoopKernel
from loopy.kernel.function_interface import CallableKernel, ScalarCallable
from loopy.symbolic import SubArrayRef

__doc__ = """
.. currentmodule:: loopy

.. autofunction:: pack_and_unpack_args_for_call
"""


def pack_and_unpack_args_for_call_for_single_kernel(kernel,
        callables_table, call_name, args_to_pack=None,
        args_to_unpack=None):
    """
    Returns a a copy of *kernel* with instructions appended to copy the
    arguments in *args* to match the alignment expected by the *call_name* in
    the kernel. The arguments are copied back to *args* with the appropriate
    data layout.

    :arg call_name: An instance of :class:`str` denoting the function call in
        the *kernel*.
    :arg args_to_unpack: A list of the arguments as instances of :class:`str` which
        must be packed. If set *None*, it is interpreted that all the array
        arguments would be packed.
    :arg args_to_unpack: A list of the arguments as instances of :class:`str`
        which must be unpacked. If set *None*, it is interpreted that
        all the array arguments should be unpacked.
    """
    assert isinstance(kernel, LoopKernel)
    new_domains = []
    new_tmps = kernel.temporary_variables.copy()
    old_insn_to_new_insns = {}

    for insn in kernel.instructions:
        if not isinstance(insn, CallInstruction):
            # pack and unpack call only be done for CallInstructions.
            continue
        if insn.expression.function.name not in callables_table:
            continue

        in_knl_callable = callables_table[
                insn.expression.function.name]

        if in_knl_callable.name != call_name:
            # not the function we're looking for.
            continue
        in_knl_callable = in_knl_callable.with_packing_for_args()

        vng = kernel.get_var_name_generator()
        ing = kernel.get_instruction_id_generator()

        parameters = insn.expression.parameters
        if args_to_pack is None:
            args_to_pack = [par.subscript.aggregate.name for par in
                    parameters+insn.assignees if isinstance(par, SubArrayRef)
                    and (par.swept_inames)]
        if args_to_unpack is None:
            args_to_unpack = [par.subscript.aggregate.name for par in
                    parameters+insn.assignees if isinstance(par, SubArrayRef)
                    and (par.swept_inames)]

        # {{{ sanity checks for args

        assert isinstance(args_to_pack, list)
        assert isinstance(args_to_unpack, list)

        for arg in args_to_pack:
            found_sub_array_ref = False

            for par in parameters + insn.assignees:
                # checking that the given args is a sub array ref
                if isinstance(par, SubArrayRef) and (
                        par.subscript.aggregate.name == arg):
                    found_sub_array_ref = True
                    break
            if not found_sub_array_ref:
                raise LoopyError("No match found for packing arg '%s' of call '%s' "
                        "at insn '%s'." % (arg, call_name, insn.id))
        for arg in args_to_unpack:
            if arg not in args_to_pack:
                raise LoopyError("Argument %s should be packed in order to be "
                        "unpacked." % arg)

        # }}}

        packing_insns = []
        unpacking_insns = []

        # {{{ handling ilp tags

        from loopy.kernel.data import IlpBaseTag, VectorizeTag
        import islpy as isl
        from pymbolic import var

        dim_type = isl.dim_type.set
        ilp_inames = {iname for iname in insn.within_inames
                         if all(isinstance(tag, (IlpBaseTag, VectorizeTag))
                                for tag in kernel.iname_to_tags.get(iname, []))}
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

        # }}}

        from pymbolic.mapper.substitutor import make_subst_func
        from loopy.symbolic import SubstitutionMapper

        # dict to store the new assignees and parameters, the mapping pattern
        # from arg_id to parameters is identical to InKernelCallable.arg_id_to_dtype
        id_to_parameters = tuple(enumerate(parameters)) + tuple(
                (-i-1, assignee) for i, assignee in enumerate(insn.assignees))
        new_id_to_parameters = {}

        for arg_id, p in id_to_parameters:
            if isinstance(p, SubArrayRef) and (p.subscript.aggregate.name in
                    args_to_pack):
                new_pack_inames = ilp_inames_map.copy()  # packing-specific inames
                new_unpack_inames = ilp_inames_map.copy()  # unpacking-specific iname

                new_pack_inames = {iname: var(vng(iname.name +
                    "_pack")) for iname in p.swept_inames}
                new_unpack_inames = {iname: var(vng(iname.name +
                    "_unpack")) for iname in p.swept_inames}

                # Updating the domains corresponding to the new inames.
                for iname in p.swept_inames:
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
                        AddressSpace)

                if arg in kernel.arg_dict:
                    arg_in_caller = kernel.arg_dict[arg]
                else:
                    arg_in_caller = kernel.temporary_variables[arg]

                pack_tmp = TemporaryVariable(
                    name=pack_name,
                    dtype=arg_in_caller.dtype,
                    dim_tags=in_knl_callable.arg_id_to_descr[arg_id].dim_tags,
                    shape=in_knl_callable.arg_id_to_descr[arg_id].shape,
                    address_space=AddressSpace.PRIVATE,
                )

                new_tmps[pack_name] = pack_tmp

                from loopy import Assignment
                pack_subst_mapper = SubstitutionMapper(make_subst_func(
                    new_pack_inames))
                unpack_subst_mapper = SubstitutionMapper(make_subst_func(
                    new_unpack_inames))

                # {{{ getting the lhs for packing and rhs for unpacking

                from loopy.symbolic import simplify_via_aff
                from loopy.isl_helpers import make_slab

                flatten_index = simplify_via_aff(
                        sum(dim_tag.stride*idx for dim_tag, idx in
                        zip(arg_in_caller.dim_tags, p.subscript.index_tuple)))

                new_indices = []
                for dim_tag in in_knl_callable.arg_id_to_descr[arg_id].dim_tags:
                    ind = flatten_index // dim_tag.stride
                    flatten_index -= (dim_tag.stride * ind)
                    new_indices.append(ind)

                new_indices = tuple(simplify_via_aff(i) for i in new_indices)

                pack_lhs_assignee = pack_subst_mapper(
                        var(pack_name).index(new_indices))
                unpack_rhs = unpack_subst_mapper(
                        var(pack_name).index(new_indices))

                # }}}

                packing_insns.append(Assignment(
                    assignee=pack_lhs_assignee,
                    expression=pack_subst_mapper.map_subscript(p.subscript),
                    within_inames=insn.within_inames - ilp_inames | {
                        new_pack_inames[i].name for i in p.swept_inames} | (
                            new_ilp_inames),
                    depends_on=insn.depends_on,
                    id=ing(insn.id+"_pack"),
                    depends_on_is_final=True
                ))

                if p.subscript.aggregate.name in args_to_unpack:
                    unpacking_insns.append(Assignment(
                        expression=unpack_rhs,
                        assignee=unpack_subst_mapper.map_subscript(p.subscript),
                        within_inames=insn.within_inames - ilp_inames | {
                            new_unpack_inames[i].name for i in p.swept_inames} | (
                                new_ilp_inames),
                        id=ing(insn.id+"_unpack"),
                        depends_on=frozenset([insn.id]),
                        depends_on_is_final=True
                    ))

                # {{{ creating the sweep inames for the new sub array refs

                updated_swept_inames = []

                for _ in in_knl_callable.arg_id_to_descr[arg_id].shape:
                    updated_swept_inames.append(var(vng("i_packsweep_"+arg)))

                ctx = kernel.isl_context
                space = isl.Space.create_from_names(ctx,
                        set=[iname.name for iname in updated_swept_inames])
                iname_set = isl.BasicSet.universe(space)
                for iname, axis_length in zip(updated_swept_inames,
                        in_knl_callable.arg_id_to_descr[arg_id].shape):
                    iname_set = iname_set & make_slab(space, iname.name, 0,
                            axis_length)
                new_domains = new_domains + [iname_set]

                # }}}

                new_id_to_parameters[arg_id] = SubArrayRef(
                        tuple(updated_swept_inames),
                        (var(pack_name).index(tuple(updated_swept_inames))))
            else:
                new_id_to_parameters[arg_id] = p

        if packing_insns:
            subst_mapper = SubstitutionMapper(make_subst_func(ilp_inames_map))
            new_call_insn = insn.with_transformed_expressions(subst_mapper)
            new_params = tuple(subst_mapper(new_id_to_parameters[i]) for i, _ in
                    enumerate(parameters))
            new_assignees = tuple(subst_mapper(new_id_to_parameters[-i-1])
                    for i, _ in enumerate(insn.assignees))
            new_call_insn = new_call_insn.copy(
                    depends_on=new_call_insn.depends_on | {
                        pack.id for pack in packing_insns},
                    within_inames=new_call_insn.within_inames - ilp_inames | (
                        new_ilp_inames),
                    expression=new_call_insn.expression.function(*new_params),
                    assignees=new_assignees)
            old_insn_to_new_insns[insn.id] = (packing_insns + [new_call_insn] +
                    unpacking_insns)

    if old_insn_to_new_insns:
        new_instructions = []
        for insn in kernel.instructions:
            if insn.id in old_insn_to_new_insns:
                # Replacing the current instruction with the group of
                # instructions including the packing and unpacking instructions
                new_instructions.extend(old_insn_to_new_insns[insn.id])
            else:
                # for the instructions that depend on the call instruction that
                # are to be packed and unpacked, we need to add the complete
                # instruction block as a dependency for them.
                new_depends_on = insn.depends_on
                if insn.depends_on & set(old_insn_to_new_insns):
                    # need to add the unpack instructions on dependencies.
                    for old_insn_id in insn.depends_on & set(old_insn_to_new_insns):
                        new_depends_on |= frozenset(i.id for i
                                in old_insn_to_new_insns[old_insn_id])
                new_instructions.append(insn.copy(depends_on=new_depends_on))
        kernel = kernel.copy(
            domains=kernel.domains + new_domains,
            instructions=new_instructions,
            temporary_variables=new_tmps
        )

    return kernel


def pack_and_unpack_args_for_call(program, *args, **kwargs):
    assert isinstance(program, TranslationUnit)

    new_callables = {}
    for func_id, in_knl_callable in program.callables_table.items():
        if isinstance(in_knl_callable, CallableKernel):
            new_subkernel = pack_and_unpack_args_for_call_for_single_kernel(
                    in_knl_callable.subkernel, program.callables_table,
                    *args, **kwargs)
            in_knl_callable = in_knl_callable.copy(
                    subkernel=new_subkernel)
        elif isinstance(in_knl_callable, ScalarCallable):
            pass
        else:
            raise NotImplementedError("Unknown type of callable %s." % (
                type(in_knl_callable).__name__))

        new_callables[func_id] = in_knl_callable

    return program.copy(callables_table=Map(new_callables))

# vim: foldmethod=marker

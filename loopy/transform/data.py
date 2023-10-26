__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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

from warnings import warn

from dataclasses import dataclass, replace

from typing import Optional, Tuple, Dict, cast

import numpy as np
from immutables import Map
from islpy import dim_type

from pytools import MovedFunctionDeprecationWrapper

from loopy.diagnostic import LoopyError
from loopy.kernel.data import AddressSpace, ImageArg, auto, TemporaryVariable

from loopy.types import LoopyType
from loopy.typing import ExpressionT
from loopy.translation_unit import TranslationUnit, for_each_kernel
from loopy.kernel import LoopKernel
from loopy.kernel.function_interface import CallableKernel, ScalarCallable


# {{{ convenience: add_prefetch

# {{{ process footprint_subscripts

def _add_kernel_axis(kernel, axis_name, start, stop, base_inames):
    from loopy.kernel.tools import DomainChanger
    domch = DomainChanger(kernel, base_inames)

    domain = domch.domain
    new_dim_idx = domain.dim(dim_type.set)
    domain = (domain
            .insert_dims(dim_type.set, new_dim_idx, 1)
            .set_dim_name(dim_type.set, new_dim_idx, axis_name))

    from loopy.symbolic import get_dependencies
    deps = get_dependencies(start) | get_dependencies(stop)
    assert deps <= kernel.all_params()

    param_names = domain.get_var_names(dim_type.param)
    for dep in deps:
        if dep not in param_names:
            new_dim_idx = domain.dim(dim_type.param)
            domain = (domain
                    .insert_dims(dim_type.param, new_dim_idx, 1)
                    .set_dim_name(dim_type.param, new_dim_idx, dep))

    from loopy.isl_helpers import make_slab
    slab = make_slab(domain.get_space(), axis_name, start, stop)

    domain = domain & slab

    return kernel.copy(domains=domch.get_domains_with(domain))


def _process_footprint_subscripts(kernel, rule_name, sweep_inames,
        footprint_subscripts, arg):
    """Track applied iname rewrites, deal with slice specifiers ':'."""

    name_gen = kernel.get_var_name_generator()

    from pymbolic.primitives import Variable

    if footprint_subscripts is None:
        return kernel, rule_name, sweep_inames, []

    if not isinstance(footprint_subscripts, (list, tuple)):
        footprint_subscripts = [footprint_subscripts]

    inames_to_be_removed = []

    new_footprint_subscripts = []
    for fsub in footprint_subscripts:
        if isinstance(fsub, str):
            from loopy.symbolic import parse
            fsub = parse(fsub)

        if not isinstance(fsub, tuple):
            fsub = (fsub,)

        if len(fsub) != arg.num_user_axes():
            raise ValueError("sweep index '%s' has the wrong number of dimensions"
                    % str(fsub))

        for subst_map in kernel.applied_iname_rewrites:
            from loopy.symbolic import SubstitutionMapper
            from pymbolic.mapper.substitutor import make_subst_func
            fsub = SubstitutionMapper(make_subst_func(subst_map))(fsub)

        from loopy.symbolic import get_dependencies
        fsub_dependencies = get_dependencies(fsub)

        new_fsub = []
        for axis_nr, fsub_axis in enumerate(fsub):
            from pymbolic.primitives import Slice
            if isinstance(fsub_axis, Slice):
                if fsub_axis.children != (None,):
                    raise NotImplementedError("add_prefetch only "
                            "supports full slices")

                axis_name = name_gen(
                        based_on="%s_fetch_axis_%d" % (arg.name, axis_nr))

                kernel = _add_kernel_axis(kernel, axis_name, 0, arg.shape[axis_nr],
                        frozenset(sweep_inames) | fsub_dependencies)
                sweep_inames = sweep_inames + [axis_name]

                inames_to_be_removed.append(axis_name)
                new_fsub.append(Variable(axis_name))

            else:
                new_fsub.append(fsub_axis)

        new_footprint_subscripts.append(tuple(new_fsub))
        del new_fsub

    footprint_subscripts = new_footprint_subscripts
    del new_footprint_subscripts

    subst_use = [Variable(rule_name)(*si) for si in footprint_subscripts]
    return kernel, subst_use, sweep_inames, inames_to_be_removed

# }}}


def add_prefetch_for_single_kernel(kernel, callables_table, var_name,
        sweep_inames=None, dim_arg_names=None,

        default_tag=None,

        rule_name=None,
        temporary_name=None,
        temporary_address_space=None, temporary_scope=None,
        footprint_subscripts=None,
        fetch_bounding_box=False,
        fetch_outer_inames=None,
        prefetch_insn_id=None,
        within=None):
    """See :func:`add_prefetch` for detailed, user-facing documentation."""

    assert isinstance(kernel, LoopKernel)
    if sweep_inames is None:
        sweep_inames = []

    # {{{ fish indexing out of var_name and into footprint_subscripts

    from loopy.symbolic import parse
    parsed_var_name = parse(var_name)

    from pymbolic.primitives import Variable, Subscript
    if isinstance(parsed_var_name, Variable):
        # nothing to see
        pass
    elif isinstance(parsed_var_name, Subscript):
        if footprint_subscripts is not None:
            raise TypeError("if footprint_subscripts is specified, then var_name "
                    "may not contain a subscript")

        assert isinstance(parsed_var_name.aggregate, Variable)
        footprint_subscripts = [parsed_var_name.index]
        parsed_var_name = parsed_var_name.aggregate
    else:
        raise ValueError("var_name must either be a variable name or a subscript")

    # }}}

    # {{{ fish out tags

    from loopy.symbolic import TaggedVariable
    if isinstance(parsed_var_name, TaggedVariable):
        var_name = parsed_var_name.name
        tags = parsed_var_name.tags
    else:
        var_name = parsed_var_name.name
        tags = ()

    # }}}

    c_name = var_name
    from loopy.kernel.instruction import LegacyStringInstructionTag
    tag_suffix = "_".join(tag.value for tag in tags
            if isinstance(tag, LegacyStringInstructionTag))
    if tag_suffix:
        c_name = c_name + "_" + tag_suffix

    var_name_gen = kernel.get_var_name_generator()

    if rule_name is None:
        rule_name = var_name_gen("%s_fetch_rule" % c_name)
    if temporary_name is None:
        temporary_name = var_name_gen("%s_fetch" % c_name)

    var_descr = kernel.get_var_descriptor(var_name)

    # {{{ make parameter names and unification template

    parameters = []
    for i in range(var_descr.num_user_axes()):
        based_on = "%s_dim_%d" % (c_name, i)
        if var_descr.dim_names is not None:
            based_on = "{}_dim_{}".format(c_name, var_descr.dim_names[i])
        if dim_arg_names is not None and i < len(dim_arg_names):
            based_on = dim_arg_names[i]

        par_name = var_name_gen(based_on=based_on)
        parameters.append(par_name)

    from pymbolic import var
    uni_template = parsed_var_name
    if len(parameters) > 1:
        uni_template = uni_template.index(
                tuple(var(par_name) for par_name in parameters))
    elif len(parameters) == 1:
        uni_template = uni_template.index(var(parameters[0]))

    # }}}

    from loopy.transform.subst import extract_subst
    kernel = extract_subst(kernel, rule_name, uni_template, parameters,
            within=within)

    if isinstance(sweep_inames, str):
        sweep_inames = [s.strip() for s in sweep_inames.split(",")]
    else:
        # copy, standardize to list
        sweep_inames = list(sweep_inames)

    kernel, subst_use, sweep_inames, inames_to_be_removed = \
            _process_footprint_subscripts(
                    kernel,  rule_name, sweep_inames,
                    footprint_subscripts, var_descr)

    # Our _not_provided is actually a different object from the one in the
    # precompute module, but precompute acutally uses that to adjust its
    # warning message.

    from loopy.transform.precompute import precompute_for_single_kernel
    new_kernel = precompute_for_single_kernel(kernel, callables_table,
            subst_use, sweep_inames, precompute_inames=dim_arg_names,
            default_tag=default_tag, dtype=var_descr.dtype,
            fetch_bounding_box=fetch_bounding_box,
            temporary_name=temporary_name,
            temporary_address_space=temporary_address_space,
            precompute_outer_inames=fetch_outer_inames,
            compute_insn_id=prefetch_insn_id,
            within=within)

    # {{{ remove inames that were temporarily added by slice sweeps

    new_domains = new_kernel.domains[:]

    for iname in inames_to_be_removed:
        home_domain_index = kernel.get_home_domain_index(iname)
        domain = new_domains[home_domain_index]

        dt, idx = domain.get_var_dict()[iname]
        assert dt == dim_type.set

        new_domains[home_domain_index] = domain.project_out(dt, idx, 1)

    new_kernel = new_kernel.copy(domains=new_domains)

    # }}}

    # If the rule survived past precompute() (i.e. some accesses fell outside
    # the footprint), get rid of it before moving on.
    if rule_name in new_kernel.substitutions:
        from loopy.transform.subst import expand_subst
        return expand_subst(new_kernel, "... > id:"+rule_name)
    else:
        return new_kernel


def add_prefetch(program, *args, **kwargs):
    """Prefetch all accesses to the variable *var_name*, with all accesses
    being swept through *sweep_inames*.

    :arg var_name: A string, the name of the variable being prefetched.
        This may be a 'tagged variable name' (such as ``field$mytag``
        to restrict the effect of the operation to only variable accesses
        with a matching tag.

        This may also be a subscripted version of the variable, in which
        case this access dictates the footprint that is prefetched,
        e.g. ``A[:,:]`` or ``field[i,j,:,:]``. In this case, accesses
        in the kernel are disregarded.

    :arg sweep_inames: A list of inames, or a comma-separated string of them.
        This routine 'sweeps' all accesses to *var_name* through all allowed
        values of the *sweep_inames* to generate a footprint. All values
        in this footprint are then stored in a temporary variable, and
        the original variable accesses replaced with accesses to this
        temporary.

    :arg dim_arg_names: List of names representing each fetch axis.
        These names show up as inames in the generated fetch code

    :arg default_tag: The :ref:`implementation tag <iname-tags>` to
        assign to the inames driving the prefetch code. Use *None* to
        leave them undefined (to assign them later by hand). The current
        default will make them local axes and automatically split them to
        fit the work group size, but this default will disappear in favor
        of simply leaving them untagged in 2019.x. For 2018.x, a warning
        will be issued if no *default_tag* is specified.

    :arg rule_name: base name of the generated temporary variable.
    :arg temporary_name: The name of the temporary to be used.
    :arg temporary_address_space: The :class:`AddressSpace` to use for the
        temporary.
    :arg footprint_subscripts: A list of tuples indicating the index (i.e.
        subscript) tuples used to generate the footprint.

        If only one such set of indices is desired, this may also be specified
        directly by putting an index expression into *var_name*. Substitutions
        such as those occurring in dimension splits are recorded and also
        applied to these indices.

    :arg fetch_bounding_box: To fit within :mod:`loopy`'s execution model,
        the 'footprint' of the fetch currently has to be a convex set.
        Sometimes this is not the case, e.g. for a high-order stencil::

              o
              o
            ooooo
              o
              o

        The footprint of the stencil when 'swept' over a base domain
        would look like this, and because of the 'missing corners',
        this set is not convex::

              oooooooooo
              oooooooooo
            oooooooooooooo
            oooooooooooooo
            oooooooooooooo
            oooooooooooooo
              oooooooooo
              oooooooooo

        Passing ``fetch_bounding_box=True`` gives :mod:`loopy` permission
        to instead fetch the 'bounding box' of the footprint, i.e.
        this set in the stencil example::

            OOooooooooooOO
            OOooooooooooOO
            oooooooooooooo
            oooooooooooooo
            oooooooooooooo
            oooooooooooooo
            OOooooooooooOO
            OOooooooooooOO

        Note the added corners marked with "``O``". The resulting footprint is
        guaranteed to be convex.


    :arg fetch_outer_inames: The inames within which the fetch
        instruction is nested. If *None*, make an educated guess.

    :arg fetch_insn_id: The ID of the instruction generated to perform the
        prefetch.

    :arg within: a stack match as understood by
        :func:`loopy.match.parse_stack_match` to select the instructions where
        *var_name* is to be prefetched.

    This function internally uses :func:`extract_subst` and :func:`precompute`.
    """
    assert isinstance(program, TranslationUnit)

    new_callables = {}
    for func_id, in_knl_callable in program.callables_table.items():
        if isinstance(in_knl_callable, CallableKernel):
            new_subkernel = add_prefetch_for_single_kernel(
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

# }}}


# {{{ change variable kinds

@for_each_kernel
def change_arg_to_image(kernel, name):
    new_args = []
    for arg in kernel.args:
        if arg.name == name:
            assert arg.offset == 0
            assert arg.shape is not None
            new_args.append(ImageArg(arg.name, dtype=arg.dtype, shape=arg.shape))
        else:
            new_args.append(arg)

    return kernel.copy(args=new_args)

# }}}


# {{{ tag array axes

@for_each_kernel
def tag_array_axes(kernel, ary_names, dim_tags):
    """
    :arg dim_tags: a tuple of
        :class:`loopy.kernel.array.ArrayDimImplementationTag` or a string that
        parses to one. See :func:`loopy.kernel.array.parse_array_dim_tags` for a
        description of the allowed string format.

        For example, *dim_tags* could be ``"N2,N0,N1"`` to determine
        that the second axis is the fastest-varying, the last is
        the next-fastest, and the first is the slowest.

    .. versionchanged:: 2016.2

        This function was called ``tag_data_axes`` before version 2016.2.
    """

    from loopy.kernel.tools import ArrayChanger

    if isinstance(ary_names, str):
        ary_names = [ary_name.strip() for ary_name in ary_names.split(",")]

    for ary_name in ary_names:
        achng = ArrayChanger(kernel, ary_name)
        ary = achng.get()

        from loopy.kernel.array import parse_array_dim_tags
        new_dim_tags = parse_array_dim_tags(dim_tags,
                n_axes=ary.num_user_axes(),
                use_increasing_target_axes=ary.max_target_axes > 1,
                dim_names=ary.dim_names)

        ary = ary.copy(dim_tags=tuple(new_dim_tags))

        kernel = achng.with_changed_array(ary)

    return kernel


tag_data_axes = (
        MovedFunctionDeprecationWrapper(tag_array_axes))

# }}}


# {{{ set_array_axis_names

@for_each_kernel
def set_array_axis_names(kernel, ary_names, dim_names):
    """
    .. versionchanged:: 2016.2

        This function was called ``set_array_dim_names`` before version 2016.2.
    """
    from loopy.kernel.tools import ArrayChanger
    if isinstance(ary_names, str):
        ary_names = ary_names.split(",")

    if isinstance(dim_names, str):
        dim_names = tuple(dim_names.split(","))

    for ary_name in ary_names:
        achng = ArrayChanger(kernel, ary_name)
        ary = achng.get()

        ary = ary.copy(dim_names=dim_names)

        kernel = achng.with_changed_array(ary)

    return kernel


set_array_dim_names = (MovedFunctionDeprecationWrapper(
    set_array_axis_names))

# }}}


# {{{ remove_unused_arguments

@for_each_kernel
def remove_unused_arguments(kernel):
    new_args = []

    import loopy as lp
    exp_kernel = lp.expand_subst(kernel)

    refd_vars = set(kernel.all_params())
    for insn in exp_kernel.instructions:
        refd_vars.update(insn.dependency_names())

    from loopy.kernel.array import ArrayBase, FixedStrideArrayDimTag
    from loopy.symbolic import get_dependencies
    from itertools import chain

    def tolerant_get_deps(expr):
        if expr is None or expr is lp.auto:
            return set()
        return get_dependencies(expr)

    for ary in chain(kernel.args, kernel.temporary_variables.values()):
        if isinstance(ary, ArrayBase):
            refd_vars.update(
                    tolerant_get_deps(ary.shape)
                    | tolerant_get_deps(ary.offset))

            for dim_tag in ary.dim_tags:
                if isinstance(dim_tag, FixedStrideArrayDimTag):
                    refd_vars.update(
                            tolerant_get_deps(dim_tag.stride))

    for arg in kernel.args:
        if arg.name in refd_vars:
            new_args.append(arg)

    return kernel.copy(args=new_args)

# }}}


# {{{ alias_temporaries

@for_each_kernel
def alias_temporaries(kernel, names, base_name_prefix=None,
        synchronize_for_exclusive_use=True):
    """Sets all temporaries given by *names* to be backed by a single piece of
    storage.

    :arg synchronize_for_exclusive_use: A :class:`bool`. If ``True``, this also
        introduces ordering structures ("groups") to prevent the usage to ensure
        that the live ranges (i.e. the regions of code where each of the
        temporaries is used) do not overlap. This will allow two (or more)
        temporaries to share the same storage space as long as their live
        ranges do not need to be concurrent.
    :arg base_name_prefix: an identifier to be used for the common storage
        area

    .. versionchanged:: 2016.3

        Added *synchronize_for_exclusive_use* flag.
        ``synchronize_for_exclusive_use=True`` was the previous default
        behavior.
    """
    gng = kernel.get_group_name_generator()
    group_names = [gng("tmpgrp_"+name) for name in names]

    if base_name_prefix is None:
        base_name_prefix = "temp_storage"

    from pytools import UniqueNameGenerator
    vng = UniqueNameGenerator(
            kernel.all_variable_names()
            | {tv.base_storage
                for tv in kernel.temporary_variables.values()
                if tv.base_storage is not None})
    base_name = vng(base_name_prefix)

    names_set = set(names)

    if synchronize_for_exclusive_use:
        new_insns = []
        for insn in kernel.instructions:
            temp_deps = insn.dependency_names() & names_set

            if not temp_deps:
                new_insns.append(insn)
                continue

            if len(temp_deps) > 1:
                raise LoopyError("Instruction {insn} refers to multiple of the "
                        "temporaries being aliased, namely '{temps}'. Cannot alias."
                        .format(
                            insn=insn.id,
                            temps=", ".join(temp_deps)))

            temp_name, = temp_deps
            temp_idx = names.index(temp_name)
            group_name = group_names[temp_idx]
            other_group_names = (
                    frozenset(group_names[:temp_idx])
                    | frozenset(group_names[temp_idx+1:]))

            new_insns.append(
                    insn.copy(
                        groups=insn.groups | frozenset([group_name]),
                        conflicts_with_groups=(
                            insn.conflicts_with_groups | other_group_names)))
    else:
        new_insns = kernel.instructions

    new_temporary_variables = {}
    for tv in kernel.temporary_variables.values():
        if tv.name in names_set:
            if tv.base_storage is not None:
                raise LoopyError("temporary variable '{tv}' already has "
                        "a defined storage array -- cannot alias"
                        .format(tv=tv.name))

            new_temporary_variables[tv.name] = \
                    tv.copy(
                            base_storage=base_name,
                            _base_storage_access_may_be_aliasing=False)
        else:
            new_temporary_variables[tv.name] = tv

    return kernel.copy(
            instructions=new_insns,
            temporary_variables=new_temporary_variables)

# }}}


# {{{ set argument order

@for_each_kernel
def set_argument_order(kernel, arg_names):
    """
    :arg arg_names: A list (or comma-separated string) or argument
        names. All arguments must be in this list.
    """
    #FIXME: @inducer -- shoulld this only affect the root kernel, or should it
    # take a within?

    if isinstance(arg_names, str):
        arg_names = arg_names.split(",")

    new_args = []
    old_arg_dict = kernel.arg_dict.copy()

    for arg_name in arg_names:
        try:
            arg = old_arg_dict.pop(arg_name)
        except KeyError:
            raise LoopyError("unknown argument '%s'"
                    % arg_name)

        new_args.append(arg)

    if old_arg_dict:
        raise LoopyError("incomplete argument list passed "
                "to set_argument_order. Left over: '%s'"
                % ", ".join(arg_name for arg_name in old_arg_dict))

    return kernel.copy(args=new_args)

# }}}


# {{{ rename argument

@for_each_kernel
def rename_argument(kernel, old_name, new_name, existing_ok=False):
    """
    .. versionadded:: 2016.2
    """

    var_name_gen = kernel.get_var_name_generator()

    if old_name not in kernel.arg_dict:
        raise LoopyError("old arg name '%s' does not exist" % old_name)

    does_exist = var_name_gen.is_name_conflicting(new_name)

    if does_exist and not existing_ok:
        raise LoopyError("argument name '%s' conflicts with an existing identifier"
                "--cannot rename" % new_name)

    # {{{ instructions

    from pymbolic import var
    subst_dict = {old_name: var(new_name)}

    from loopy.symbolic import (
            RuleAwareSubstitutionMapper,
            SubstitutionRuleMappingContext)
    from pymbolic.mapper.substitutor import make_subst_func
    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, var_name_gen)
    smap = RuleAwareSubstitutionMapper(rule_mapping_context,
                    make_subst_func(subst_dict),
                    within=lambda kernel, insn, stack: True)

    kernel = rule_mapping_context.finish_kernel(smap.map_kernel(kernel))

    # }}}

    # {{{ args

    new_args = []
    for arg in kernel.args:
        if arg.name == old_name:
            arg = arg.copy(name=new_name)

        new_args.append(arg)

    # }}}

    # {{{ domain/assumptions

    def rename_arg_in_basic_set(dom):
        dom_var_dict = dom.get_var_dict()
        if old_name in dom_var_dict:
            dt, pos = dom_var_dict[old_name]
            dom = dom.set_dim_name(dt, pos, new_name)

        return dom

    new_domains = []
    for dom in kernel.domains:
        dom = rename_arg_in_basic_set(dom)
        new_domains.append(dom)

    new_assumptions = rename_arg_in_basic_set(kernel.assumptions)

    # }}}

    return kernel.copy(domains=new_domains, args=new_args,
            assumptions=new_assumptions)

# }}}


# {{{ set temporary address space

@for_each_kernel
def set_temporary_address_space(kernel, temp_var_names, address_space):
    """
    :arg temp_var_names: a container with membership checking,
        or a comma-separated string of variables for which the
        address space is to be set.
    :arg address_space: One of the values from :class:`loopy.AddressSpace`, or one
        of the strings ``"private"``, ``"local"``, or ``"global"``.
    """

    if isinstance(temp_var_names, str):
        temp_var_names = [s.strip() for s in temp_var_names.split(",")]

    from loopy.kernel.data import AddressSpace
    if isinstance(address_space, str):
        try:
            address_space = getattr(AddressSpace, address_space.upper())
        except AttributeError:
            raise LoopyError("address_space '%s' unknown" % address_space)

    if not isinstance(address_space, int) or address_space not in [
            AddressSpace.PRIVATE,
            AddressSpace.LOCAL,
            AddressSpace.GLOBAL]:
        raise LoopyError("invalid address_space '%s'" % address_space)

    new_temp_vars = kernel.temporary_variables.copy()
    for tv_name in temp_var_names:
        try:
            tv = new_temp_vars[tv_name]
        except KeyError:
            raise LoopyError("temporary '%s' not found" % tv_name)

        new_temp_vars[tv_name] = tv.copy(address_space=address_space)

    return kernel.copy(temporary_variables=new_temp_vars)


def set_temporary_scope(kernel, temp_var_names, address_space):
    from warnings import warn
    warn("set_temporary_scope is deprecated and will stop working in "
            "July 2022. Use set_temporary_address_space instead.",
            DeprecationWarning, stacklevel=2)

    return set_temporary_address_space(kernel, temp_var_names, address_space)

# }}}


# {{{ reduction_arg_to_subst_rule

@for_each_kernel
def reduction_arg_to_subst_rule(
        kernel, inames, insn_match=None, subst_rule_name=None):
    if isinstance(inames, str):
        inames = [s.strip() for s in inames.split(",")]

    inames_set = frozenset(inames)

    substs = kernel.substitutions.copy()

    var_name_gen = kernel.get_var_name_generator()

    def map_reduction(expr, rec, nresults=1):
        if frozenset(expr.inames) != inames_set:
            return type(expr)(
                    operation=expr.operation,
                    inames=expr.inames,
                    expr=rec(expr.expr),
                    allow_simultaneous=expr.allow_simultaneous)

        if subst_rule_name is None:
            subst_rule_prefix = "red_%s_arg" % "_".join(inames)
            my_subst_rule_name = var_name_gen(subst_rule_prefix)
        else:
            my_subst_rule_name = subst_rule_name

        if my_subst_rule_name in substs:
            raise LoopyError("substitution rule '%s' already exists"
                    % my_subst_rule_name)

        from loopy.kernel.data import SubstitutionRule
        substs[my_subst_rule_name] = SubstitutionRule(
                name=my_subst_rule_name,
                arguments=tuple(inames),
                expression=expr.expr)

        from pymbolic import var
        iname_vars = [var(iname) for iname in inames]

        return type(expr)(
                operation=expr.operation,
                inames=expr.inames,
                expr=var(my_subst_rule_name)(*iname_vars),
                allow_simultaneous=expr.allow_simultaneous)

    from loopy.symbolic import ReductionCallbackMapper
    cb_mapper = ReductionCallbackMapper(map_reduction)

    from loopy.kernel.data import MultiAssignmentBase

    new_insns = []
    for insn in kernel.instructions:
        if not isinstance(insn, MultiAssignmentBase):
            new_insns.append(insn)
        else:
            new_insns.append(insn.copy(expression=cb_mapper(insn.expression)))

    return kernel.copy(
            instructions=new_insns,
            substitutions=substs)

# }}}


# {{{ add_padding_to_avoid_bank_conflicts

# experimental not exported/documented for now
@for_each_kernel
def add_padding_to_avoid_bank_conflicts(kernel, device):
    import pyopencl as cl
    import pyopencl.characterize as cl_char

    new_temp_vars = {}

    from loopy.kernel.data import AddressSpace

    lmem_size = cl_char.usable_local_mem_size(device)
    for temp_var in kernel.temporary_variables.values():
        if temp_var.address_space != AddressSpace.LOCAL:
            new_temp_vars[temp_var.name] = \
                    temp_var.copy(storage_shape=temp_var.shape)
            continue

        if not temp_var.shape:
            # scalar, no need to mess with storage shape
            new_temp_vars[temp_var.name] = temp_var
            continue

        other_loctemp_nbytes = [
                tv.nbytes
                for tv in kernel.temporary_variables.values()
                if tv.address_space == AddressSpace.LOCAL
                and tv.name != temp_var.name]

        storage_shape = temp_var.storage_shape

        if storage_shape is None:
            storage_shape = temp_var.shape

        storage_shape = list(storage_shape)

        # sizes of all dims except the last one, which we may change
        # below to avoid bank conflicts
        from pytools import product

        if device.local_mem_type == cl.device_local_mem_type.GLOBAL:
            # FIXME: could try to avoid cache associativity disasters
            new_storage_shape = storage_shape

        elif device.local_mem_type == cl.device_local_mem_type.LOCAL:
            min_mult = cl_char.local_memory_bank_count(device)
            good_incr = None
            new_storage_shape = storage_shape
            min_why_not = None

            for increment in range(storage_shape[-1]//2):

                test_storage_shape = storage_shape[:]
                test_storage_shape[-1] = test_storage_shape[-1] + increment
                new_mult, why_not = cl_char.why_not_local_access_conflict_free(
                        device, temp_var.dtype.itemsize,
                        temp_var.shape, test_storage_shape)

                # will choose smallest increment 'automatically'
                if new_mult < min_mult:
                    new_lmem_use = (sum(other_loctemp_nbytes)
                            + temp_var.dtype.itemsize*product(test_storage_shape))
                    if new_lmem_use < lmem_size:
                        new_storage_shape = test_storage_shape
                        min_mult = new_mult
                        min_why_not = why_not
                        good_incr = increment

            if min_mult != 1:
                from warnings import warn
                from loopy.diagnostic import LoopyAdvisory
                warn("could not find a conflict-free mem layout "
                        "for local variable '%s' "
                        "(currently: %dx conflict, increment: %s, reason: %s)"
                        % (temp_var.name, min_mult, good_incr, min_why_not),
                        LoopyAdvisory)
        else:
            from warnings import warn
            warn("unknown type of local memory")

            new_storage_shape = storage_shape

        new_temp_vars[temp_var.name] = temp_var.copy(
                storage_shape=tuple(new_storage_shape))

    return kernel.copy(temporary_variables=new_temp_vars)

# }}}


# {{{ allocate_temporaries_for_base_storage

@dataclass(frozen=True)
class _BaseStorageInfo:
    name: str
    next_offset: ExpressionT
    approx_nbytes: Optional[int] = None


def _sym_max(a: ExpressionT, b: ExpressionT) -> ExpressionT:
    from numbers import Number
    if isinstance(a, Number) and isinstance(b, Number):
        # https://github.com/python/mypy/issues/3186
        return max(a, b)  # type: ignore[call-overload]
    else:
        from pymbolic.primitives import Max
        return Max((a, b))


@for_each_kernel
def allocate_temporaries_for_base_storage(kernel: LoopKernel,
        only_address_space: Optional[int] = None,
        aliased=True,
        max_nbytes: Optional[int] = None,
        _implicitly_run=False,
        ) -> LoopKernel:
    from pytools import product

    new_tvs = dict(kernel.temporary_variables)
    made_changes = False

    vng = kernel.get_var_name_generator()

    name_aspace_dtype_to_bsi: Dict[
            Tuple[str, AddressSpace, LoopyType], _BaseStorageInfo] = {}

    for tv in sorted(
            kernel.temporary_variables.values(),
            key=lambda key_tv: key_tv.name):
        if tv.base_storage and tv.initializer:
            raise LoopyError(
                    f"Temporary '{tv.name}' has both base_storage "
                    "and an initializer. That's not allowed.")
        if tv.offset and not tv.base_storage:
            raise LoopyError(
                    f"Temporary '{tv.name}' has an offset and no base_storage. "
                    "That's not allowed.")

        if (tv.base_storage
                and tv.base_storage not in kernel.temporary_variables
                and (
                    only_address_space is None
                    or tv.address_space == only_address_space)):
            made_changes = True
            assert isinstance(tv.dtype, LoopyType)

            if tv.address_space is auto:
                raise LoopyError("When allocating base storage for temporary "
                        f"'{tv.name}', the address space of the temporary "
                        "was not yet determined (set to 'auto').")

            assert isinstance(tv.shape, tuple)
            ary_size = product(si for si in tv.shape)
            if isinstance(ary_size, (int, np.integer)):
                approx_array_nbytes = tv.dtype.numpy_dtype.itemsize * ary_size
            else:
                # FIXME: Could use approximate values of ValueArgs
                approx_array_nbytes = 0

            bs_key = (tv.base_storage,
                      cast(AddressSpace, tv.address_space), tv.dtype)
            bsi = name_aspace_dtype_to_bsi.get(bs_key)

            if bsi is None or (
                    # are we out of space?
                    not aliased
                    and max_nbytes is not None
                    and bsi.approx_nbytes is not None
                    and bsi.approx_nbytes + approx_array_nbytes > max_nbytes):
                bsi = name_aspace_dtype_to_bsi[bs_key] = _BaseStorageInfo(
                        name=vng(tv.base_storage),
                        next_offset=0,
                        approx_nbytes=None if aliased else 0)

                new_tvs[bsi.name] = TemporaryVariable(
                    name=bsi.name,
                    dtype=tv.dtype,
                    shape=(0,),
                    address_space=tv.address_space)

            new_tvs[tv.name] = tv.copy(
                base_storage=bsi.name,
                offset=bsi.next_offset,
                _base_storage_access_may_be_aliasing=(
                    aliased if tv._base_storage_access_may_be_aliasing is None
                    else tv._base_storage_access_may_be_aliasing))

            bs_tv = new_tvs[bsi.name]
            assert isinstance(bs_tv.shape, tuple)
            bs_size, = bs_tv.shape
            if aliased:
                new_bs_size = _sym_max(bs_size, ary_size)
            else:
                new_bs_size = bs_size + ary_size

                assert bsi.approx_nbytes is not None
                name_aspace_dtype_to_bsi[bs_key] = replace(bsi,
                    next_offset=bsi.next_offset + ary_size,
                    approx_nbytes=bsi.approx_nbytes + approx_array_nbytes)

            new_tvs[bsi.name] = new_tvs[bsi.name].copy(shape=(new_bs_size,))

    if made_changes:
        if _implicitly_run:
            warn("Base storage allocation was performed implicitly during "
                    "preprocessing. This is deprecated and will stop working "
                    "in 2023. Call loopy.allocate_temporaries_for_base_storage "
                    "explicitly to avoid this warning.", DeprecationWarning)

        return kernel.copy(temporary_variables=new_tvs)
    else:
        return kernel

# }}}

# vim: foldmethod=marker

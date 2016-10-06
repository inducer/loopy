from __future__ import division, absolute_import

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

import six  # noqa

from loopy.diagnostic import LoopyError
from islpy import dim_type

from loopy.kernel.data import ImageArg

from pytools import MovedFunctionDeprecationWrapper


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


def add_prefetch(kernel, var_name, sweep_inames=[], dim_arg_names=None,
        default_tag="l.auto", rule_name=None,
        temporary_name=None,
        temporary_scope=None, temporary_is_local=None,
        footprint_subscripts=None,
        fetch_bounding_box=False,
        fetch_outer_inames=None):
    """Prefetch all accesses to the variable *var_name*, with all accesses
    being swept through *sweep_inames*.

    :arg dim_arg_names: List of names representing each fetch axis.
    :arg rule_name: base name of the generated temporary variable.
    :arg footprint_subscripts: A list of tuples indicating the index (i.e.
        subscript) tuples used to generate the footprint.

        If only one such set of indices is desired, this may also be specified
        directly by putting an index expression into *var_name*. Substitutions
        such as those occurring in dimension splits are recorded and also
        applied to these indices.

    :arg fetch_outer_inames: The inames within which the fetch
        instruction is nested. If *None*, make an educated guess.

    This function combines :func:`extract_subst` and :func:`precompute`.
    """

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

    # {{{ fish out tag

    from loopy.symbolic import TaggedVariable
    if isinstance(parsed_var_name, TaggedVariable):
        var_name = parsed_var_name.name
        tag = parsed_var_name.tag
    else:
        var_name = parsed_var_name.name
        tag = None

    # }}}

    c_name = var_name
    if tag is not None:
        c_name = c_name + "_" + tag

    var_name_gen = kernel.get_var_name_generator()

    if rule_name is None:
        rule_name = var_name_gen("%s_fetch_rule" % c_name)
    if temporary_name is None:
        temporary_name = var_name_gen("%s_fetch" % c_name)

    arg = kernel.arg_dict[var_name]

    # {{{ make parameter names and unification template

    parameters = []
    for i in range(arg.num_user_axes()):
        based_on = "%s_dim_%d" % (c_name, i)
        if arg.dim_names is not None:
            based_on = "%s_dim_%s" % (c_name, arg.dim_names[i])
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
    kernel = extract_subst(kernel, rule_name, uni_template, parameters)

    if isinstance(sweep_inames, str):
        sweep_inames = [s.strip() for s in sweep_inames.split(",")]
    else:
        # copy, standardize to list
        sweep_inames = list(sweep_inames)

    kernel, subst_use, sweep_inames, inames_to_be_removed = \
            _process_footprint_subscripts(
                    kernel,  rule_name, sweep_inames,
                    footprint_subscripts, arg)

    from loopy.transform.precompute import precompute
    new_kernel = precompute(kernel, subst_use, sweep_inames,
            precompute_inames=dim_arg_names,
            default_tag=default_tag, dtype=arg.dtype,
            fetch_bounding_box=fetch_bounding_box,
            temporary_name=temporary_name,
            temporary_scope=temporary_scope, temporary_is_local=temporary_is_local,
            precompute_outer_inames=fetch_outer_inames)

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

# }}}


# {{{ change variable kinds

def change_arg_to_image(knl, name):
    new_args = []
    for arg in knl.args:
        if arg.name == name:
            assert arg.offset == 0
            assert arg.shape is not None
            new_args.append(ImageArg(arg.name, dtype=arg.dtype, shape=arg.shape))
        else:
            new_args.append(arg)

    return knl.copy(args=new_args)

# }}}


# {{{ tag array axes

def tag_array_axes(knl, ary_names, dim_tags):
    """
    .. versionchanged:: 2016.2

        This function was called :func:`tag_data_axes` before version 2016.2.
    """

    from loopy.kernel.tools import ArrayChanger

    if isinstance(ary_names, str):
        ary_names = [ary_name.strip() for ary_name in ary_names.split(",")]

    for ary_name in ary_names:
        achng = ArrayChanger(knl, ary_name)
        ary = achng.get()

        from loopy.kernel.array import parse_array_dim_tags
        new_dim_tags = parse_array_dim_tags(dim_tags,
                n_axes=ary.num_user_axes(),
                use_increasing_target_axes=ary.max_target_axes > 1,
                dim_names=ary.dim_names)

        ary = ary.copy(dim_tags=tuple(new_dim_tags))

        knl = achng.with_changed_array(ary)

    return knl


tag_data_axes = MovedFunctionDeprecationWrapper(tag_array_axes)

# }}}


# {{{ set_array_axis_names

def set_array_axis_names(kernel, ary_names, dim_names):
    """
    .. versionchanged:: 2016.2

        This function was called :func:`set_array_dim_names` before version 2016.2.
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

set_array_dim_names = MovedFunctionDeprecationWrapper(set_array_axis_names)

# }}}


# {{{ remove_unused_arguments

def remove_unused_arguments(knl):
    new_args = []

    import loopy as lp
    exp_knl = lp.expand_subst(knl)

    refd_vars = set(knl.all_params())
    for insn in exp_knl.instructions:
        refd_vars.update(insn.dependency_names())

    from loopy.kernel.array import ArrayBase, FixedStrideArrayDimTag
    from loopy.symbolic import get_dependencies
    from itertools import chain

    def tolerant_get_deps(expr):
        if expr is None or expr is lp.auto:
            return set()
        return get_dependencies(expr)

    for ary in chain(knl.args, six.itervalues(knl.temporary_variables)):
        if isinstance(ary, ArrayBase):
            refd_vars.update(
                    tolerant_get_deps(ary.shape)
                    | tolerant_get_deps(ary.offset))

            for dim_tag in ary.dim_tags:
                if isinstance(dim_tag, FixedStrideArrayDimTag):
                    refd_vars.update(
                            tolerant_get_deps(dim_tag.stride))

    for arg in knl.args:
        if arg.name in refd_vars:
            new_args.append(arg)

    return knl.copy(args=new_args)

# }}}


# {{{ alias_temporaries

def alias_temporaries(knl, names, base_name_prefix=None):
    """Sets all temporaries given by *names* to be backed by a single piece of
    storage. Also introduces ordering structures ("groups") to prevent the
    usage of each temporary to interfere with another.

    :arg base_name_prefix: an identifier to be used for the common storage
        area
    """
    gng = knl.get_group_name_generator()
    group_names = [gng("tmpgrp_"+name) for name in names]

    if base_name_prefix is None:
        base_name_prefix = "temp_storage"

    vng = knl.get_var_name_generator()
    base_name = vng(base_name_prefix)

    names_set = set(names)

    new_insns = []
    for insn in knl.instructions:
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

    new_temporary_variables = {}
    for tv in six.itervalues(knl.temporary_variables):
        if tv.name in names_set:
            if tv.base_storage is not None:
                raise LoopyError("temporary variable '{tv}' already has "
                        "a defined storage array -- cannot alias"
                        .format(tv=tv.name))

            new_temporary_variables[tv.name] = \
                    tv.copy(base_storage=base_name)
        else:
            new_temporary_variables[tv.name] = tv

    return knl.copy(
            instructions=new_insns,
            temporary_variables=new_temporary_variables)

# }}}


# {{{ set argument order

def set_argument_order(kernel, arg_names):
    """
    :arg arg_names: A list (or comma-separated string) or argument
        names. All arguments must be in this list.
    """

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
                    within=lambda knl, insn, stack: True)

    kernel = smap.map_kernel(kernel)

    new_args = []
    for arg in kernel.args:
        if arg.name == old_name:
            arg = arg.copy(name=new_name)

        new_args.append(arg)

    return kernel.copy(args=new_args)

# }}}


# {{{ set temporary scope

def set_temporary_scope(kernel, temp_var_names, scope):
    """
    :arg temp_var_names: a container with membership checking,
        or a comma-separated string of variables for which the
        scope is to be set.
    :arg scope: One of the values from :class:`temp_var_scope`, or one
        of the strings ``"private"``, ``"local"``, or ``"global"``.
    """

    if isinstance(temp_var_names, str):
        temp_var_names = [s.strip() for s in temp_var_names.split(",")]

    from loopy.kernel.data import temp_var_scope
    if isinstance(scope, str):
        try:
            scope = getattr(temp_var_scope, scope.upper())
        except AttributeError:
            raise LoopyError("scope '%s' unknown" % scope)

    if not isinstance(scope, int) or scope not in [
            temp_var_scope.PRIVATE,
            temp_var_scope.LOCAL,
            temp_var_scope.GLOBAL]:
        raise LoopyError("invalid scope '%s'" % scope)

    new_temp_vars = kernel.temporary_variables.copy()
    for tv_name in temp_var_names:
        try:
            tv = new_temp_vars[tv_name]
        except KeyError:
            raise LoopyError("temporary '%s' not found" % tv_name)

        new_temp_vars[tv_name] = tv.copy(scope=scope)

    return kernel.copy(temporary_variables=new_temp_vars)

# }}}


# {{{ reduction_arg_to_subst_rule

def reduction_arg_to_subst_rule(knl, inames, insn_match=None, subst_rule_name=None):
    if isinstance(inames, str):
        inames = [s.strip() for s in inames.split(",")]

    inames_set = frozenset(inames)

    substs = knl.substitutions.copy()

    var_name_gen = knl.get_var_name_generator()

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
    for insn in knl.instructions:
        if not isinstance(insn, MultiAssignmentBase):
            new_insns.append(insn)
        else:
            new_insns.append(insn.copy(expression=cb_mapper(insn.expression)))

    return knl.copy(
            instructions=new_insns,
            substitutions=substs)

# }}}


# vim: foldmethod=marker

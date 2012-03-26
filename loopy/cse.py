from __future__ import division

import islpy as isl
from islpy import dim_type
from loopy.symbolic import get_dependencies, SubstitutionMapper
from pymbolic.mapper.substitutor import make_subst_func
import numpy as np

from pytools import Record
from pymbolic import var




class InvocationDescriptor(Record):
    __slots__ = [
            "expr",
            "args",
            "expands_footprint",
            "is_in_footprint",

            # Record from which substitution rule this invocation of the rule
            # being precomputed originated. If all invocations end up being
            # in-footprint, then the replacement with the prefetch can be made
            # within the rule.
            "from_subst_rule"
            ]




def to_parameters_or_project_out(param_inames, set_inames, set):
    for iname in set.get_space().get_var_dict().keys():
        if iname in param_inames:
            dt, idx = set.get_space().get_var_dict()[iname]
            set = set.move_dims(
                    dim_type.param, set.dim(dim_type.param),
                    dt, idx, 1)
        elif iname in set_inames:
            pass
        else:
            dt, idx = set.get_space().get_var_dict()[iname]
            set = set.project_out(dt, idx, 1)

    return set




# {{{ construct storage->sweep map

def build_per_access_storage_to_sweep_map(invdesc, domain_dup_sweep,
        storage_axis_names, storage_axis_sources, prime_sweep_inames):

    map_space = domain_dup_sweep.get_space()
    stor_dim = len(storage_axis_names)
    rn = map_space.dim(dim_type.out)

    map_space = map_space.add_dims(dim_type.in_, stor_dim)
    for i, saxis in enumerate(storage_axis_names):
        # arg names are initially primed, to be replaced with unprimed
        # base-0 versions below

        map_space = map_space.set_dim_name(dim_type.in_, i, saxis+"'")

    # map_space: [stor_axes'] -> [domain](dup_sweep_index)[dup_sweep]

    set_space = map_space.move_dims(
            dim_type.out, rn,
            dim_type.in_, 0, stor_dim).range()

    # set_space: [domain](dup_sweep_index)[dup_sweep][stor_axes']

    stor2sweep = None

    from loopy.symbolic import aff_from_expr

    for saxis, saxis_source in zip(storage_axis_names, storage_axis_sources):
        if isinstance(saxis_source, int):
            # an argument
            cns = isl.Constraint.equality_from_aff(
                    aff_from_expr(set_space,
                        var(saxis+"'")
                        - prime_sweep_inames(invdesc.args[saxis_source])))
        else:
            # a 'bare' sweep iname
            cns = isl.Constraint.equality_from_aff(
                    aff_from_expr(set_space,
                        var(saxis+"'")
                        - prime_sweep_inames(var(saxis_source))))

        cns_map = isl.BasicMap.from_constraint(cns)
        if stor2sweep is None:
            stor2sweep = cns_map
        else:
            stor2sweep = stor2sweep.intersect(cns_map)

    stor2sweep = stor2sweep.move_dims(
            dim_type.in_, 0,
            dim_type.out, rn, stor_dim)

    # stor2sweep is back in map_space
    return stor2sweep

def build_global_storage_to_sweep_map(invocation_descriptors,
        dup_sweep_index, domain_dup_sweep,
        storage_axis_names, storage_axis_sources, prime_sweep_inames):
    """
    As a side effect, this fills out is_in_footprint in the
    invocation descriptors.
    """

    # The storage map goes from storage axes to domain_dup_sweep.
    # The first len(arg_names) storage dimensions are the rule's arguments.

    global_stor2sweep = None

    # build footprint
    for invdesc in invocation_descriptors:
        if invdesc.expands_footprint:
            stor2sweep = build_per_access_storage_to_sweep_map(invdesc, domain_dup_sweep,
                    storage_axis_names, storage_axis_sources, prime_sweep_inames)

            if global_stor2sweep is None:
                global_stor2sweep = stor2sweep
            else:
                global_stor2sweep = global_stor2sweep.union(stor2sweep)

            invdesc.is_in_footprint = True

    if isinstance(global_stor2sweep, isl.BasicMap):
        global_stor2sweep = isl.Map.from_basic_map(stor2sweep)
    global_stor2sweep = global_stor2sweep.intersect_range(domain_dup_sweep)

    # function to move non-sweep inames into parameter space
    def move_non_sweep_to_par(s2smap):
        sp = s2smap.get_space()
        return s2smap.move_dims(
                dim_type.param, sp.dim(dim_type.param),
                dim_type.out, 0, dup_sweep_index)

    global_s2s_par_dom = move_non_sweep_to_par(global_stor2sweep).domain()

    # check if non-footprint-building invocation descriptors fall into footprint
    for invdesc in invocation_descriptors:
        stor2sweep = build_per_access_storage_to_sweep_map(invdesc, domain_dup_sweep,
                    storage_axis_names, storage_axis_sources, prime_sweep_inames)

        if isinstance(stor2sweep, isl.BasicMap):
            stor2sweep = isl.Map.from_basic_map(stor2sweep)

        stor2sweep = move_non_sweep_to_par(
                stor2sweep.intersect_range(domain_dup_sweep))

        is_in_footprint = stor2sweep.domain().is_subset(
                global_s2s_par_dom)

        if not invdesc.expands_footprint:
            invdesc.is_in_footprint = is_in_footprint
        else:
            assert is_in_footprint

    return global_stor2sweep

# }}}

# {{{ compute storage bounds

def compute_bounds(kernel, subst_name, stor2sweep, sweep_inames,
        storage_axis_names):

    # move non-sweep inames into parameter space

    dup_sweep_index = kernel.space.dim(dim_type.out)
    # map_space: [stor_axes'] -> [domain](dup_sweep_index)[dup_sweep]

    sp = stor2sweep.get_space()
    bounds_footprint_map = stor2sweep.move_dims(
            dim_type.param, sp.dim(dim_type.param),
            dim_type.out, 0, dup_sweep_index)

    # compute bounds for each storage axis
    storage_domain = bounds_footprint_map.domain().coalesce()

    if not storage_domain.is_bounded():
        raise RuntimeError("In precomputation of substitution '%s': "
                "sweep did not result in a bounded storage domain"
                % subst_name)

    from loopy.kernel import find_var_base_indices_and_shape_from_inames
    return find_var_base_indices_and_shape_from_inames(
            storage_domain, [saxis+"'" for saxis in storage_axis_names],
            kernel.cache_manager, context=kernel.assumptions)

# }}}





def get_access_info(kernel, subst_name,
        storage_axis_names, storage_axis_sources,
        sweep_inames, invocation_descriptors):

    # {{{ duplicate sweep inames

    primed_sweep_inames = [psin+"'" for psin in sweep_inames]
    from loopy.isl_helpers import duplicate_axes
    dup_sweep_index = kernel.space.dim(dim_type.out)
    domain_dup_sweep = duplicate_axes(
            kernel.domain, sweep_inames,
            primed_sweep_inames)

    prime_sweep_inames = SubstitutionMapper(make_subst_func(
        dict((sin, var(psin)) for sin, psin in zip(sweep_inames, primed_sweep_inames))))

    # }}}

    stor2sweep = build_global_storage_to_sweep_map(
            invocation_descriptors, dup_sweep_index, domain_dup_sweep,
            storage_axis_names, storage_axis_sources, prime_sweep_inames)

    storage_base_indices, storage_shape = compute_bounds(
            kernel, subst_name, stor2sweep, sweep_inames,
            storage_axis_names)

    # compute augmented domain

    # {{{ filter out unit-length dimensions

    non1_storage_axis_names = []
    non1_storage_base_indices = []
    non1_storage_shape = []

    for saxis, bi, l in zip(storage_axis_names, storage_base_indices, storage_shape):
        if l != 1:
            non1_storage_axis_names.append(saxis)
            non1_storage_base_indices.append(bi)
            non1_storage_shape.append(l)

    # }}}

    # {{{ subtract off the base indices
    # add the new, base-0 indices as new in dimensions

    sp = stor2sweep.get_space()
    stor_idx = sp.dim(dim_type.out)

    n_stor = len(storage_axis_names)
    nn1_stor = len(non1_storage_axis_names)

    aug_domain = stor2sweep.move_dims(
            dim_type.out, stor_idx,
            dim_type.in_, 0,
            n_stor).range()

    # aug_domain space now:
    # [domain](dup_sweep_index)[dup_sweep](stor_idx)[stor_axes']

    aug_domain = aug_domain.insert_dims(dim_type.set, stor_idx, nn1_stor)
    for i, name in enumerate(non1_storage_axis_names):
        aug_domain = aug_domain.set_dim_name(dim_type.set, stor_idx+i, name)

    # aug_domain space now:
    # [domain](dup_sweep_index)[dup_sweep](stor_idx)[stor_axes'][n1_stor_axes]

    from loopy.symbolic import aff_from_expr
    for saxis, bi, s in zip(storage_axis_names, storage_base_indices, storage_shape):
        if s != 1:
            cns = isl.Constraint.equality_from_aff(
                    aff_from_expr(aug_domain.get_space(),
                        var(saxis) - (var(saxis+"'") - bi)))

            aug_domain = aug_domain.add_constraint(cns)

    # }}}

    # eliminate (primed) storage axes with non-zero base indices

    aug_domain = aug_domain.eliminate(dim_type.set, stor_idx+nn1_stor, n_stor)
    aug_domain = aug_domain.remove_dims(dim_type.set, stor_idx+nn1_stor, n_stor)

    # {{{ eliminate duplicated sweep_inames

    nsweep = len(sweep_inames)
    aug_domain = aug_domain.eliminate(dim_type.set, dup_sweep_index, nsweep)
    aug_domain = aug_domain.remove_dims(dim_type.set, dup_sweep_index, nsweep)

    # }}}

    return (non1_storage_axis_names, aug_domain,
            storage_base_indices, non1_storage_base_indices, non1_storage_shape)





def simplify_via_aff(expr):
    from loopy.symbolic import aff_from_expr, aff_to_expr
    deps = get_dependencies(expr)
    return aff_to_expr(aff_from_expr(
        isl.Space.create_from_names(isl.Context(), list(deps)),
        expr))




def precompute(kernel, subst_use, dtype, sweep_inames=[],
        storage_axes=None, new_storage_axis_names=None, storage_axis_to_tag={},
        default_tag="l.auto"):
    """Precompute the expression described in the substitution rule determined by
    *subst_use* and store it in a temporary array. A precomputation needs two
    things to operate, a list of *sweep_inames* (order irrelevant) and an
    ordered list of *storage_axes* (whose order will describe the axis ordering
    of the temporary array).

    :arg subst_use: Describes what to prefetch.

    The following objects may be given for *subst_use*:

    * The name of the substitution rule.

    * The tagged name ("name$tag") of the substitution rule.

    * A list of invocations of the substitution rule.
      This list of invocations, when swept across *sweep_inames*, then serves
      to define the footprint of the precomputation.

      Invocations may be tagged ("name$tag") to filter out a subset of the
      usage sites of the substitution rule. (Namely those usage sites that
      use the same tagged name.)

      Invocations may be given as a string or as a
      :class:`pymbolic.primitives.Expression` object.

      If only one invocation is to be given, then the only entry of the list
      may be given directly.

    If the list of invocations generating the footprint is not given,
    all (tag-matching, if desired) usage sites of the substitution rule
    are used to determine the footprint.

    The following cases can arise for each sweep axis:

    * The axis is an iname that occurs within arguments specified at
      usage sites of the substitution rule. This case is assumed covered
      by the storage axes provided for the argument.

    * The axis is an iname that occurs within the *value* of the rule, but not
      within its arguments. A new, dedicated storage axis is allocated for
      such an axis.

    :arg sweep_inames: A :class:`list` of inames and/or rule argument names to be swept.
    :arg storage_axes: A :class:`list` of inames and/or rule argument names/indices to be used as storage axes.

    If `storage_axes` is not specified, it defaults to the arrangement
    `<direct sweep axes><arguments>` with the direct sweep axes being the
    slower-varying indices.

    Trivial storage axes (i.e. axes of length 1 with respect to the sweep) are
    eliminated.
    """

    # {{{ check, standardize arguments

    for iname in sweep_inames:
        if iname not in kernel.all_inames():
            raise RuntimeError("sweep iname '%s' is not a known iname"
                    % iname)

    if isinstance(storage_axes, str):
        raise TypeError("storage_axes may not be a string--likely a leftover "
                "footprint_generators argument")

    if isinstance(subst_use, str):
        subst_use = [subst_use]

    footprint_generators = None

    subst_name = None
    subst_tag = None

    from pymbolic.primitives import Variable, Call
    from loopy.symbolic import parse, TaggedVariable

    for use in subst_use:
        if isinstance(use, str):
            use = parse(use)

        if isinstance(use, Call):
            if footprint_generators is None:
                footprint_generators = []

            footprint_generators.append(use)
            subst_name_as_expr = use.function
        else:
            subst_name_as_expr = use

        if isinstance(subst_name_as_expr, TaggedVariable):
            new_subst_name = subst_name_as_expr.name
            new_subst_tag = subst_name_as_expr.tag
        elif isinstance(subst_name_as_expr, Variable):
            new_subst_name = subst_name_as_expr.name
            new_subst_tag = None
        else:
            raise ValueError("unexpected type of subst_name")

        if (subst_name, subst_tag) == (None, None):
            subst_name, subst_tag = new_subst_name, new_subst_tag
        else:
            if (subst_name, subst_tag) != (new_subst_name, new_subst_tag):
                raise ValueError("not all uses in subst_use agree "
                        "on rule name and tag")

    # }}}

    # {{{ process invocations in footprint generators, start invocation_descriptors

    invocation_descriptors = []

    if footprint_generators:
        for fpg in footprint_generators:
            from pymbolic.primitives import Variable, Call
            if isinstance(fpg, Variable):
                args = ()
            elif isinstance(fpg, Call):
                args = fpg.parameters
            else:
                raise ValueError("footprint generator must "
                        "be substitution rule invocation")

            invocation_descriptors.append(
                    InvocationDescriptor(expr=fpg, args=args,
                        expands_footprint=True,
                        from_subst_rule=None))

    # }}}

    c_subst_name = subst_name.replace(".", "_")

    from loopy.kernel import parse_tag
    default_tag = parse_tag(default_tag)

    subst = kernel.substitutions[subst_name]
    arg_names = subst.arguments

    # {{{ gather up invocations in kernel code, finish invocation_descriptors

    current_subst_rule_stack = []

    # We need to work on the fully expanded form of an expression.
    # To that end, instantiate a substitutor.
    from loopy.symbolic import ParametrizedSubstitutor
    rules_except_mine = kernel.substitutions.copy()
    del rules_except_mine[subst_name]
    subst_expander = ParametrizedSubstitutor(rules_except_mine,
            one_level=True)

    def gather_substs(expr, name, tag, args, rec):
        if subst_name != name:
            if name in subst_expander.rules:
                # We can't deal with invocations that involve other substitution's
                # arguments. Therefore, fully expand each encountered substitution
                # rule and look at the invocations of subst_name occurring in its
                # body.

                expanded_expr = subst_expander(expr)
                current_subst_rule_stack.append(name)
                result = rec(expanded_expr)
                current_subst_rule_stack.pop()
                return result

            else:
                return None

        if subst_tag is not None and subst_tag != tag:
            # use fall-back identity mapper
            return None

        if len(args) != len(subst.arguments):
            raise RuntimeError("Rule '%s' invoked with %d arguments (needs %d)"
                    % (subst_name, len(args), len(subst.arguments), ))

        arg_deps = get_dependencies(args)
        if not arg_deps <= kernel.all_inames():
            raise RuntimeError("CSE arguments in '%s' do not consist "
                    "exclusively of inames" % expr)

        if current_subst_rule_stack:
            current_subst_rule = current_subst_rule_stack[-1]
        else:
            current_subst_rule = None

        invocation_descriptors.append(
                InvocationDescriptor(expr=expr, args=args,
                    expands_footprint=footprint_generators is None,
                    from_subst_rule=current_subst_rule))
        return expr

    from loopy.symbolic import SubstitutionCallbackMapper
    scm = SubstitutionCallbackMapper(names_filter=None, func=gather_substs)

    for insn in kernel.instructions:
        scm(insn.expression)

    if not invocation_descriptors:
        raise RuntimeError("no invocations of '%s' found" % subst_name)

    # }}}

    sweep_inames = list(sweep_inames)

    # {{{ see if we need extra storage dimensions

    # find inames used in argument dependencies

    usage_arg_deps = set()
    for invdesc in invocation_descriptors:
        if not invdesc.expands_footprint:
            continue

        for arg in invdesc.args:
            usage_arg_deps.update(get_dependencies(arg))

    extra_storage_axes = list(set(sweep_inames) - usage_arg_deps)

    if storage_axes is None:
        storage_axes = (
                extra_storage_axes
                + list(xrange(len(arg_names))))

    # }}}

    newly_created_var_names = set()

    # {{{ process storage_axes argument

    # (and substitute in subst_expressions if any variable name changes are necessary)

    expr_subst_dict = {}

    storage_axis_names = []
    storage_axis_sources = [] # number for arg#, or iname
    storage_axis_name_to_tag = {}

    for i, saxis in enumerate(storage_axes):
        tag_lookup_saxis = saxis

        if saxis in subst.arguments:
            saxis = subst.arguments.index(saxis)

        storage_axis_sources.append(saxis)

        if isinstance(saxis, int):
            # argument index
            name = old_name = subst.arguments[saxis]
        else:
            old_name = saxis
            name = "%s_%s" % (c_subst_name, old_name)

        if new_storage_axis_names is not None and i < len(new_storage_axis_names):
            name = new_storage_axis_names[i]
            tag_lookup_saxis = name
            if name in (kernel.all_variable_names() | newly_created_var_names):
                raise RuntimeError("new storage axis name '%s' already exists" % name)

        if name in (kernel.all_variable_names()
                | newly_created_var_names):
            name = kernel.make_unique_var_name(
                    based_on=name, extra_used_vars=newly_created_var_names)

        storage_axis_names.append(name)
        storage_axis_name_to_tag[name] = storage_axis_to_tag.get(
                tag_lookup_saxis, default_tag)

        newly_created_var_names.add(name)
        expr_subst_dict[old_name] = var(name)

    del storage_axis_to_tag
    del storage_axes
    del new_storage_axis_names

    compute_expr = (
            SubstitutionMapper(make_subst_func(expr_subst_dict))
            (subst.expression))

    del expr_subst_dict

    # }}}

    (non1_storage_axis_names, new_domain,
            storage_base_indices, non1_storage_base_indices, non1_storage_shape)= \
                    get_access_info(kernel, subst_name,
                            storage_axis_names, storage_axis_sources,
                            sweep_inames, invocation_descriptors)

    # {{{ try a few ways to get new_domain to be convex

    if len(new_domain.get_basic_sets()) > 1:
        hull_new_domain = new_domain.simple_hull()
        if hull_new_domain <= new_domain:
            new_domain = hull_new_domain

    new_domain = new_domain.coalesce()

    if len(new_domain.get_basic_sets()) > 1:
        hull_new_domain = new_domain.simple_hull()
        if hull_new_domain <= new_domain:
            new_domain = hull_new_domain

    if isinstance(new_domain, isl.Set):
        dom_bsets = new_domain.get_basic_sets()
        if len(dom_bsets) > 1:
            raise NotImplementedError("Substitution '%s' yielded a non-convex footprint"
                    % subst_name)

        new_domain, = dom_bsets

    # }}}

    # {{{ set up compute insn

    target_var_name = kernel.make_unique_var_name(based_on=c_subst_name,
            extra_used_vars=newly_created_var_names)

    assignee = var(target_var_name)

    if non1_storage_axis_names:
        assignee = assignee[tuple(var(iname) for iname in non1_storage_axis_names)]

    def zero_length_1_arg(arg_name):
        if arg_name in non1_storage_axis_names:
            return var(arg_name)
        else:
            return 0

    compute_expr = (SubstitutionMapper(
        make_subst_func(dict(
            (arg_name, zero_length_1_arg(arg_name)+bi)
            for arg_name, bi in zip(storage_axis_names, storage_base_indices)
            )))
        (compute_expr))

    from loopy.kernel import Instruction
    compute_insn = Instruction(
            id=kernel.make_unique_instruction_id(based_on=c_subst_name),
            assignee=assignee,
            expression=compute_expr)

    # }}}

    # {{{ substitute rule into expressions in kernel (if within footprint)

    left_unused_subst_rule_invocations = [False]

    def do_substs(expr, name, tag, args, rec):
        if tag != subst_tag:
            left_unused_subst_rule_invocations[0] = True
            return expr

        # {{{ check if current use is in-footprint

        if current_subst_rule is None:
            # The current subsitution was *not* found inside another
            # substitution rule. Try and dig up the corresponding invocation
            # descriptor.

            found = False
            for invdesc in invocation_descriptors:
                if expr == invdesc.expr:
                    found = True
                    break

            if footprint_generators is None:
                # We only have a right to find the expression if the
                # invocation descriptors if they were generated by a scan
                # of the code in the first place. If the user gave us
                # the footprint generators, that isn't true.

                assert found, expr

            if not found or not invdesc.is_in_footprint:
                left_unused_subst_rule_invocations[0] = True
                return expr

        else:
            # The current subsitution *was* found inside another substitution
            # rule. We can't dig up the corresponding invocation descriptor,
            # because it was the result of expanding that outer substitution
            # rule. But we do know what the current outer substitution rule is,
            # and we can check if all uses within that rule were uniformly
            # in-footprint. If so, we'll go ahead, otherwise we'll bomb out.

            current_rule_invdescs_in_footprint = [
                    invdesc.is_in_footprint
                    for invdesc in invocation_descriptors
                    if invdesc.from_subst_rule == current_subst_rule]

            from pytools import all
            all_in = all(current_rule_invdescs_in_footprint)
            all_out = all(not b for b in current_rule_invdescs_in_footprint)

            assert not (all_in and all_out)

            if not (all_in or all_out):
                raise RuntimeError("substitution '%s' (being precomputed) is used "
                        "from within substitution '%s', but not all uses of "
                        "'%s' within '%s' "
                        "are uniformly within-footprint or outside of the footprint, "
                        "making a unique replacement of '%s' impossible. Please expand "
                        "'%s' and try again."
                        % (subst_name, current_subst_rule,
                            subst_name, current_subst_rule,
                            subst_name, current_subst_rule))

            if all_out:
                left_unused_subst_rule_invocations[0] = True
                return expr

            assert all_in

        # }}}

        if len(args) != len(subst.arguments):
            raise ValueError("invocation of '%s' with too few arguments"
                    % name)

        stor_subscript = []
        for sax_name, sax_source, sax_base_idx in zip(
                storage_axis_names, storage_axis_sources, storage_base_indices):
            if sax_name not in non1_storage_axis_names:
                continue

            if isinstance(sax_source, int):
                # an argument
                ax_index = args[sax_source]
            else:
                # an iname
                ax_index = var(sax_source)

            ax_index = simplify_via_aff(ax_index - sax_base_idx)
            stor_subscript.append(ax_index)

        new_outer_expr = var(target_var_name)
        if stor_subscript:
            new_outer_expr = new_outer_expr[tuple(stor_subscript)]

        return new_outer_expr
        # can't possibly be nested, don't recurse

    new_insns = [compute_insn]

    current_subst_rule = None
    sub_map = SubstitutionCallbackMapper([subst_name], do_substs)
    for insn in kernel.instructions:
        new_insn = insn.copy(expression=sub_map(insn.expression))
        new_insns.append(new_insn)

    # also catch uses of our rule in other substitution rules
    new_substs = {}
    for s in kernel.substitutions.itervalues():
        current_subst_rule = s.name
        new_substs[s.name] = s.copy(
                expression=sub_map(s.expression))

    # If the subst above caught all uses of the subst rule, get rid of it.
    if not left_unused_subst_rule_invocations[0]:
        del new_substs[subst_name]

    # }}}

    # {{{ fill out new_iname_to_tag

    new_iname_to_tag = kernel.iname_to_tag.copy()
    for arg_name in non1_storage_axis_names:
        new_iname_to_tag[arg_name] = storage_axis_name_to_tag[arg_name]

    # }}}

    # {{{ set up temp variable

    from loopy.kernel import TemporaryVariable

    new_temporary_variables = kernel.temporary_variables.copy()
    temp_var = TemporaryVariable(
            name=target_var_name,
            dtype=np.dtype(dtype),
            base_indices=(0,)*len(non1_storage_shape),
            shape=non1_storage_shape,
            is_local=None)

    new_temporary_variables[target_var_name] = temp_var

    # }}}

    return kernel.copy(
            domain=new_domain,
            instructions=new_insns,
            substitutions=new_substs,
            temporary_variables=new_temporary_variables,
            iname_to_tag=new_iname_to_tag)




# vim: foldmethod=marker

from __future__ import division

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


import islpy as isl
from islpy import dim_type
from loopy.symbolic import (get_dependencies, SubstitutionMapper,
        ExpandingIdentityMapper)
from pymbolic.mapper.substitutor import make_subst_func
import numpy as np

from pytools import Record
from pymbolic import var


class InvocationDescriptor(Record):
    __slots__ = [
            "args",
            "expands_footprint",
            "is_in_footprint",

            # Remember where the invocation happened, in terms of the expansion
            # call stack.
            "expansion_stack",
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

def build_per_access_storage_to_domain_map(invdesc, domain,
        storage_axis_names, storage_axis_sources,
        prime_sweep_inames):

    map_space = domain.get_space()
    stor_dim = len(storage_axis_names)
    rn = map_space.dim(dim_type.out)

    map_space = map_space.add_dims(dim_type.in_, stor_dim)
    for i, saxis in enumerate(storage_axis_names):
        # arg names are initially primed, to be replaced with unprimed
        # base-0 versions below

        map_space = map_space.set_dim_name(dim_type.in_, i, saxis+"'")

    # map_space: [stor_axes'] -> [domain](dup_sweep_index)[dup_sweep](rn)

    set_space = map_space.move_dims(
            dim_type.out, rn,
            dim_type.in_, 0, stor_dim).range()

    # set_space: [domain](dup_sweep_index)[dup_sweep](rn)[stor_axes']

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


def move_to_par_from_out(s2smap, except_inames):
    while True:
        var_dict = s2smap.get_var_dict(dim_type.out)
        todo_inames = set(var_dict) - except_inames
        if todo_inames:
            iname = todo_inames.pop()

            _, dim_idx = var_dict[iname]
            s2smap = s2smap.move_dims(
                    dim_type.param, s2smap.dim(dim_type.param),
                    dim_type.out, dim_idx, 1)
        else:
            return s2smap


def build_global_storage_to_sweep_map(kernel, invocation_descriptors,
        domain_dup_sweep, dup_sweep_index,
        storage_axis_names, storage_axis_sources,
        sweep_inames, primed_sweep_inames, prime_sweep_inames):
    """
    As a side effect, this fills out is_in_footprint in the
    invocation descriptors.
    """

    # The storage map goes from storage axes to the domain.
    # The first len(arg_names) storage dimensions are the rule's arguments.

    global_stor2sweep = None

    # build footprint
    for invdesc in invocation_descriptors:
        if invdesc.expands_footprint:
            stor2sweep = build_per_access_storage_to_domain_map(
                    invdesc, domain_dup_sweep,
                    storage_axis_names, storage_axis_sources,
                    prime_sweep_inames)

            if global_stor2sweep is None:
                global_stor2sweep = stor2sweep
            else:
                global_stor2sweep = global_stor2sweep.union(stor2sweep)

            invdesc.is_in_footprint = True

    if isinstance(global_stor2sweep, isl.BasicMap):
        global_stor2sweep = isl.Map.from_basic_map(stor2sweep)
    global_stor2sweep = global_stor2sweep.intersect_range(domain_dup_sweep)

    # space for global_stor2sweep:
    # [stor_axes'] -> [domain](dup_sweep_index)[dup_sweep](rn)

    # {{{ check if non-footprint-building invocation descriptors fall into footprint

    # Make all inames except the sweep parameters. (The footprint may depend on
    # those.) (I.e. only leave sweep inames as out parameters.)
    global_s2s_par_dom = move_to_par_from_out(
            global_stor2sweep, except_inames=frozenset(primed_sweep_inames)).domain()

    for invdesc in invocation_descriptors:
        if not invdesc.expands_footprint:
            arg_inames = (
                    set(global_s2s_par_dom.get_var_names(dim_type.param))
                    & kernel.all_inames())

            for arg in invdesc.args:
                arg_inames.update(get_dependencies(arg))
            arg_inames = frozenset(arg_inames)

            from loopy.kernel import CannotBranchDomainTree
            try:
                usage_domain = kernel.get_inames_domain(arg_inames)
            except CannotBranchDomainTree:
                # and that's the end of that.
                invdesc.is_in_footprint = False
                continue

            for i in xrange(usage_domain.dim(dim_type.set)):
                iname = usage_domain.get_dim_name(dim_type.set, i)
                if iname in sweep_inames:
                    usage_domain = usage_domain.set_dim_name(
                            dim_type.set, i, iname+"'")

            stor2sweep = build_per_access_storage_to_domain_map(invdesc,
                    usage_domain, storage_axis_names, storage_axis_sources,
                    prime_sweep_inames)

            if isinstance(stor2sweep, isl.BasicMap):
                stor2sweep = isl.Map.from_basic_map(stor2sweep)

            stor2sweep = stor2sweep.intersect_range(usage_domain)

            stor2sweep = move_to_par_from_out(stor2sweep,
                    except_inames=frozenset(primed_sweep_inames))

            s2s_domain = stor2sweep.domain()
            s2s_domain, aligned_g_s2s_parm_dom = isl.align_two(
                    s2s_domain, global_s2s_par_dom)

            arg_restrictions = (
                    aligned_g_s2s_parm_dom
                    .eliminate(dim_type.set, 0,
                        aligned_g_s2s_parm_dom.dim(dim_type.set))
                    .remove_divs())

            is_in_footprint = (arg_restrictions & s2s_domain).is_subset(
                    aligned_g_s2s_parm_dom)

            invdesc.is_in_footprint = is_in_footprint

    # }}}

    return global_stor2sweep

# }}}


# {{{ compute storage bounds

def find_var_base_indices_and_shape_from_inames(
        domain, inames, cache_manager, context=None):
    base_indices_and_sizes = [
            cache_manager.base_index_and_length(domain, iname, context)
            for iname in inames]
    return zip(*base_indices_and_sizes)


def compute_bounds(kernel, domain, subst_name, stor2sweep,
        primed_sweep_inames, storage_axis_names):

    bounds_footprint_map = move_to_par_from_out(
            stor2sweep, except_inames=frozenset(primed_sweep_inames))

    # compute bounds for each storage axis
    storage_domain = bounds_footprint_map.domain().coalesce()

    if not storage_domain.is_bounded():
        raise RuntimeError("In precomputation of substitution '%s': "
                "sweep did not result in a bounded storage domain"
                % subst_name)

    return find_var_base_indices_and_shape_from_inames(
            storage_domain, [saxis+"'" for saxis in storage_axis_names],
            kernel.cache_manager, context=kernel.assumptions)

# }}}


def get_access_info(kernel, domain, subst_name,
        storage_axis_names, storage_axis_sources,
        sweep_inames, invocation_descriptors):

    # {{{ duplicate sweep inames

    # The duplication is necessary, otherwise the storage fetch
    # inames remain weirdly tied to the original sweep inames.

    primed_sweep_inames = [psin+"'" for psin in sweep_inames]
    from loopy.isl_helpers import duplicate_axes
    dup_sweep_index = domain.space.dim(dim_type.out)
    domain_dup_sweep = duplicate_axes(
            domain, sweep_inames,
            primed_sweep_inames)

    prime_sweep_inames = SubstitutionMapper(make_subst_func(
        dict((sin, var(psin))
            for sin, psin in zip(sweep_inames, primed_sweep_inames))))

    # }}}

    stor2sweep = build_global_storage_to_sweep_map(
            kernel, invocation_descriptors,
            domain_dup_sweep, dup_sweep_index,
            storage_axis_names, storage_axis_sources,
            sweep_inames, primed_sweep_inames, prime_sweep_inames)

    storage_base_indices, storage_shape = compute_bounds(
            kernel, domain, subst_name, stor2sweep, primed_sweep_inames,
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
    aug_domain = aug_domain.project_out(dim_type.set, stor_idx+nn1_stor, n_stor)

    # eliminate duplicated sweep_inames
    nsweep = len(sweep_inames)
    aug_domain = aug_domain.project_out(dim_type.set, dup_sweep_index, nsweep)

    return (non1_storage_axis_names, aug_domain,
            storage_base_indices, non1_storage_base_indices, non1_storage_shape)


def simplify_via_aff(expr):
    from loopy.symbolic import aff_from_expr, aff_to_expr
    deps = get_dependencies(expr)
    return aff_to_expr(aff_from_expr(
        isl.Space.create_from_names(isl.Context(), list(deps)),
        expr))


class InvocationGatherer(ExpandingIdentityMapper):
    def __init__(self, kernel, subst_name, subst_tag, within):
        ExpandingIdentityMapper.__init__(self,
                kernel.substitutions, kernel.get_var_name_generator())

        from loopy.symbolic import SubstitutionRuleExpander
        self.subst_expander = SubstitutionRuleExpander(
                kernel.substitutions)

        self.kernel = kernel
        self.subst_name = subst_name
        self.subst_tag = subst_tag
        self.within = within

        self.invocation_descriptors = []

    def map_substitution(self, name, tag, arguments, expn_state):
        process_me = name == self.subst_name

        if self.subst_tag is not None and self.subst_tag != tag:
            process_me = False

        process_me = process_me and self.within(expn_state.stack)

        if not process_me:
            return ExpandingIdentityMapper.map_substitution(
                    self, name, tag, arguments, expn_state)

        rule = self.old_subst_rules[name]
        arg_context = self.make_new_arg_context(
                    name, rule.arguments, arguments, expn_state.arg_context)

        arg_deps = set()
        for arg_val in arg_context.itervalues():
            arg_deps = (arg_deps
                    | get_dependencies(self.subst_expander(arg_val, insn_id=None)))

        if not arg_deps <= self.kernel.all_inames():
            from warnings import warn
            warn("Precompute arguments in '%s(%s)' do not consist exclusively "
                    "of inames and constants--specifically, these are "
                    "not inames: %s. Ignoring." % (
                        name,
                        ", ".join(str(arg) for arg in arguments),
                        ", ".join(arg_deps - self.kernel.all_inames()),
                        ))

            return ExpandingIdentityMapper.map_substitution(
                    self, name, tag, arguments, expn_state)

        self.invocation_descriptors.append(
                InvocationDescriptor(
                    args=[arg_context[arg_name] for arg_name in rule.arguments],
                    expansion_stack=expn_state.stack))

        return 0  # exact value irrelevant


class InvocationReplacer(ExpandingIdentityMapper):
    def __init__(self, kernel, subst_name, subst_tag, within,
            invocation_descriptors,
            storage_axis_names, storage_axis_sources,
            storage_base_indices, non1_storage_axis_names,
            target_var_name):
        ExpandingIdentityMapper.__init__(self,
                kernel.substitutions, kernel.get_var_name_generator())

        from loopy.symbolic import SubstitutionRuleExpander
        self.subst_expander = SubstitutionRuleExpander(
                kernel.substitutions, kernel.get_var_name_generator())

        self.kernel = kernel
        self.subst_name = subst_name
        self.subst_tag = subst_tag
        self.within = within

        self.invocation_descriptors = invocation_descriptors

        self.storage_axis_names = storage_axis_names
        self.storage_axis_sources = storage_axis_sources
        self.storage_base_indices = storage_base_indices
        self.non1_storage_axis_names = non1_storage_axis_names

        self.target_var_name = target_var_name

    def map_substitution(self, name, tag, arguments, expn_state):
        process_me = name == self.subst_name

        if self.subst_tag is not None and self.subst_tag != tag:
            process_me = False

        process_me = process_me and self.within(expn_state.stack)

        # {{{ find matching invocation descriptor

        rule = self.old_subst_rules[name]
        arg_context = self.make_new_arg_context(
                    name, rule.arguments, arguments, expn_state.arg_context)
        args = [arg_context[arg_name] for arg_name in rule.arguments]

        if not process_me:
            return ExpandingIdentityMapper.map_substitution(
                    self, name, tag, arguments, expn_state)

        matching_invdesc = None
        for invdesc in self.invocation_descriptors:
            if invdesc.args == args and expn_state.stack:
                # Could be more than one, that's fine.
                matching_invdesc = invdesc
                break

        assert matching_invdesc is not None

        invdesc = matching_invdesc
        del matching_invdesc

        # }}}

        if not invdesc.is_in_footprint:
            return ExpandingIdentityMapper.map_substitution(
                    self, name, tag, arguments, expn_state)

        assert len(arguments) == len(rule.arguments)

        stor_subscript = []
        for sax_name, sax_source, sax_base_idx in zip(
                self.storage_axis_names,
                self.storage_axis_sources,
                self.storage_base_indices):
            if sax_name not in self.non1_storage_axis_names:
                continue

            if isinstance(sax_source, int):
                # an argument
                ax_index = arguments[sax_source]
            else:
                # an iname
                ax_index = var(sax_source)

            ax_index = simplify_via_aff(ax_index - sax_base_idx)
            stor_subscript.append(ax_index)

        new_outer_expr = var(self.target_var_name)
        if stor_subscript:
            new_outer_expr = new_outer_expr[tuple(stor_subscript)]

        # Can't possibly be nested, but recurse anyway to
        # make sure substitution rules referenced below here
        # do not get thrown away.
        self.rec(rule.expression, expn_state.copy(arg_context={}))

        return new_outer_expr


def precompute(kernel, subst_use, sweep_inames=[], within=None,
        storage_axes=None, new_storage_axis_names=None, storage_axis_to_tag={},
        default_tag="l.auto", dtype=None, fetch_bounding_box=False):
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

    :arg sweep_inames: A :class:`list` of inames and/or rule argument
        names to be swept.
    :arg storage_axes: A :class:`list` of inames and/or rule argument
        names/indices to be used as storage axes.
    :arg within: a stack match as understood by
        :func:`loopy.context_matching.parse_stack_match`.

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

    from loopy.context_matching import parse_stack_match
    within = parse_stack_match(within)

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
                    InvocationDescriptor(args=args,
                        expands_footprint=True,
                        expansion_stack=None))

    # }}}

    c_subst_name = subst_name.replace(".", "_")

    from loopy.kernel.data import parse_tag
    default_tag = parse_tag(default_tag)

    subst = kernel.substitutions[subst_name]
    arg_names = subst.arguments

    # {{{ gather up invocations in kernel code, finish invocation_descriptors

    invg = InvocationGatherer(kernel, subst_name, subst_tag, within)

    for insn in kernel.instructions:
        invg(insn.expression, insn.id)

    for invdesc in invg.invocation_descriptors:
        invocation_descriptors.append(
                invdesc.copy(expands_footprint=footprint_generators is None))

    if not invocation_descriptors:
        raise RuntimeError("no invocations of '%s' found" % subst_name)

    # }}}

    sweep_inames = list(sweep_inames)
    sweep_inames_set = frozenset(sweep_inames)

    # {{{ find inames used in arguments

    expanding_usage_arg_deps = set()

    for invdesc in invocation_descriptors:
        if invdesc.expands_footprint:
            for arg in invdesc.args:
                expanding_usage_arg_deps.update(
                        get_dependencies(arg) & kernel.all_inames())

    # }}}

    var_name_gen = kernel.get_var_name_generator()

    # {{{ use given / find new storage_axes

    # extra axes made necessary because they don't occur in the arguments
    extra_storage_axes = sweep_inames_set - expanding_usage_arg_deps

    from loopy.symbolic import SubstitutionRuleExpander
    submap = SubstitutionRuleExpander(kernel.substitutions)

    value_inames = get_dependencies(
            submap(subst.expression, insn_id=None)) & kernel.all_inames()
    if value_inames - expanding_usage_arg_deps < extra_storage_axes:
        raise RuntimeError("unreferenced sweep inames specified: "
                + ", ".join(extra_storage_axes
                    - value_inames - expanding_usage_arg_deps))

    new_iname_to_tag = {}

    if storage_axes is None:
        storage_axes = (
                list(extra_storage_axes)
                + list(xrange(len(arg_names))))

    expr_subst_dict = {}

    storage_axis_names = []
    storage_axis_sources = []  # number for arg#, or iname

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
            if var_name_gen.is_name_conflicting(name):
                raise RuntimeError("new storage axis name '%s' "
                        "conflicts with existing name" % name)

        name = var_name_gen(name)

        storage_axis_names.append(name)
        new_iname_to_tag[name] = storage_axis_to_tag.get(
                tag_lookup_saxis, default_tag)

        expr_subst_dict[old_name] = var(name)

    del storage_axis_to_tag
    del storage_axes
    del new_storage_axis_names

    compute_expr = (
            SubstitutionMapper(make_subst_func(expr_subst_dict))
            (subst.expression))

    del expr_subst_dict

    # }}}

    expanding_inames = sweep_inames_set | frozenset(expanding_usage_arg_deps)
    assert expanding_inames <= kernel.all_inames()

    # {{{ find domain to be changed

    from loopy.kernel.tools import DomainChanger
    domch = DomainChanger(kernel, expanding_inames)

    if domch.leaf_domain_index is not None:
        # If the sweep inames are at home in parent domains, then we'll add
        # fetches with loops over copies of these parent inames that will end
        # up being scheduled *within* loops over these parents.

        for iname in sweep_inames_set:
            if kernel.get_home_domain_index(iname) != domch.leaf_domain_index:
                raise RuntimeError("sweep iname '%s' is not 'at home' in the "
                        "sweep's leaf domain" % iname)

    # }}}

    (non1_storage_axis_names, new_domain,
            storage_base_indices, non1_storage_base_indices, non1_storage_shape) = \
                    get_access_info(kernel, domch.domain, subst_name,
                            storage_axis_names, storage_axis_sources,
                            sweep_inames, invocation_descriptors)

    from loopy.isl_helpers import convexify, boxify
    if fetch_bounding_box:
        new_domain = boxify(kernel.cache_manager, new_domain, storage_axis_names,
                kernel.assumptions)
    else:
        new_domain = convexify(new_domain)

    for saxis in storage_axis_names:
        if saxis not in non1_storage_axis_names:
            del new_iname_to_tag[saxis]

    # {{{ set up compute insn

    target_var_name = var_name_gen(based_on=c_subst_name)

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

    from loopy.kernel.data import Instruction
    compute_insn = Instruction(
            id=kernel.make_unique_instruction_id(based_on=c_subst_name),
            assignee=assignee,
            expression=compute_expr)

    # }}}

    # {{{ substitute rule into expressions in kernel (if within footprint)

    invr = InvocationReplacer(kernel, subst_name, subst_tag, within,
            invocation_descriptors,
            storage_axis_names, storage_axis_sources,
            storage_base_indices, non1_storage_axis_names,
            target_var_name)

    kernel = invr.map_kernel(kernel)

    # }}}

    # {{{ set up temp variable

    import loopy as lp
    if dtype is None:
        dtype = lp.auto
    else:
        dtype = np.dtype(dtype)

    from loopy.kernel.data import TemporaryVariable

    new_temporary_variables = kernel.temporary_variables.copy()
    temp_var = TemporaryVariable(
            name=target_var_name,
            dtype=dtype,
            base_indices=(0,)*len(non1_storage_shape),
            shape=tuple(non1_storage_shape),
            is_local=None)

    new_temporary_variables[target_var_name] = temp_var

    # }}}

    kernel = kernel.copy(
            domains=domch.get_domains_with(new_domain),
            instructions=[compute_insn] + kernel.instructions,
            temporary_variables=new_temporary_variables)

    from loopy import tag_inames
    return tag_inames(kernel, new_iname_to_tag)




# vim: foldmethod=marker

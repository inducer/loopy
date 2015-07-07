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


import six
from six.moves import range, zip

import islpy as isl
from islpy import dim_type

from loopy.symbolic import (RuleAwareIdentityMapper, RuleAwareSubstitutionMapper,
        SubstitutionRuleMappingContext,
        TaggedVariable, Reduction, LinearSubscript, )
from loopy.diagnostic import LoopyError, LoopyWarning


# {{{ imported user interface

from loopy.library.function import (
        default_function_mangler, single_arg_function_mangler)

from loopy.kernel.data import (
        auto,
        ValueArg, GlobalArg, ConstantArg, ImageArg,
        ExpressionInstruction, CInstruction,
        TemporaryVariable)

from loopy.kernel import LoopKernel
from loopy.kernel.tools import (
        get_dot_dependency_graph,
        show_dependency_graph,
        add_dtypes,
        add_and_infer_dtypes)
from loopy.kernel.creation import make_kernel, UniqueName
from loopy.library.reduction import register_reduction_parser
from loopy.subst import extract_subst, expand_subst, temporary_to_subst
from loopy.precompute import precompute
from loopy.buffer import buffer_array
from loopy.fusion import fuse_kernels
from loopy.padding import (split_arg_axis, find_padding_multiple,
        add_padding)
from loopy.preprocess import (preprocess_kernel, realize_reduction,
        infer_unknown_types)
from loopy.schedule import generate_loop_schedules, get_one_scheduled_kernel
from loopy.codegen import generate_code, generate_body
from loopy.compiled import CompiledKernel
from loopy.options import Options
from loopy.auto_test import auto_test_vs_ref
from loopy.frontend.fortran import (c_preprocess, parse_transformed_fortran,
        parse_fortran)

__all__ = [
        "TaggedVariable", "Reduction", "LinearSubscript",

        "auto",

        "LoopKernel",

        "ValueArg", "ScalarArg", "GlobalArg", "ArrayArg", "ConstantArg", "ImageArg",
        "TemporaryVariable",

        "ExpressionInstruction", "CInstruction",

        "default_function_mangler", "single_arg_function_mangler",

        "make_kernel", "UniqueName",

        "register_reduction_parser",

        "extract_subst", "expand_subst", "temporary_to_subst",
        "precompute", "buffer_array",
        "fuse_kernels",
        "split_arg_axis", "find_padding_multiple", "add_padding",

        "get_dot_dependency_graph",
        "show_dependency_graph",
        "add_dtypes",
        "infer_argument_dtypes", "add_and_infer_dtypes",

        "preprocess_kernel", "realize_reduction", "infer_unknown_types",
        "generate_loop_schedules", "get_one_scheduled_kernel",
        "generate_code", "generate_body",

        "CompiledKernel",

        "auto_test_vs_ref",

        "Options",

        "make_kernel",
        "c_preprocess", "parse_transformed_fortran", "parse_fortran",

        "LoopyError", "LoopyWarning",

        # {{{ from this file

        "split_iname", "join_inames", "tag_inames", "duplicate_inames",
        "rename_iname", "link_inames", "remove_unused_inames",
        "set_loop_priority", "add_prefetch"
        "find_instructions", "map_instructions",
        "set_instruction_priority", "add_dependency",
        "change_arg_to_image", "tag_data_axes",
        "split_reduction_inward", "split_reduction_outward",
        "fix_parameters",
        "register_preamble_generators",
        "register_symbol_manglers",
        "register_function_manglers",

        # }}}
        ]


# }}}


# {{{ split inames

class _InameSplitter(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, within,
            split_iname, outer_iname, inner_iname, replacement_index):
        super(_InameSplitter, self).__init__(rule_mapping_context)

        self.within = within

        self.split_iname = split_iname
        self.outer_iname = outer_iname
        self.inner_iname = inner_iname

        self.replacement_index = replacement_index

    def map_reduction(self, expr, expn_state):
        if (self.split_iname in expr.inames
                and self.split_iname not in expn_state.arg_context
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)):
            new_inames = list(expr.inames)
            new_inames.remove(self.split_iname)
            new_inames.extend([self.outer_iname, self.inner_iname])

            from loopy.symbolic import Reduction
            return Reduction(expr.operation, tuple(new_inames),
                        self.rec(expr.expr, expn_state))
        else:
            return super(_InameSplitter, self).map_reduction(expr, expn_state)

    def map_variable(self, expr, expn_state):
        if (expr.name == self.split_iname
                and self.split_iname not in expn_state.arg_context
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)):
            return self.replacement_index
        else:
            return super(_InameSplitter, self).map_variable(expr, expn_state)


def split_iname(kernel, split_iname, inner_length,
        outer_iname=None, inner_iname=None,
        outer_tag=None, inner_tag=None,
        slabs=(0, 0), do_tagged_check=True,
        within=None):
    """
    :arg within: a stack match as understood by
        :func:`loopy.context_matching.parse_stack_match`.
    """

    existing_tag = kernel.iname_to_tag.get(split_iname)
    from loopy.kernel.data import ForceSequentialTag
    if do_tagged_check and (
            existing_tag is not None
            and not isinstance(existing_tag, ForceSequentialTag)):
        raise LoopyError("cannot split already tagged iname '%s'" % split_iname)

    if split_iname not in kernel.all_inames():
        raise ValueError("cannot split loop for unknown variable '%s'" % split_iname)

    applied_iname_rewrites = kernel.applied_iname_rewrites[:]

    vng = kernel.get_var_name_generator()

    if outer_iname is None:
        outer_iname = vng(split_iname+"_outer")
    if inner_iname is None:
        inner_iname = vng(split_iname+"_inner")

    def process_set(s):
        var_dict = s.get_var_dict()

        if split_iname not in var_dict:
            return s

        orig_dim_type, _ = var_dict[split_iname]

        outer_var_nr = s.dim(orig_dim_type)
        inner_var_nr = s.dim(orig_dim_type)+1

        s = s.add_dims(orig_dim_type, 2)
        s = s.set_dim_name(orig_dim_type, outer_var_nr, outer_iname)
        s = s.set_dim_name(orig_dim_type, inner_var_nr, inner_iname)

        from loopy.isl_helpers import make_slab

        space = s.get_space()
        inner_constraint_set = (
                make_slab(space, inner_iname, 0, inner_length)
                # name = inner + length*outer
                .add_constraint(isl.Constraint.eq_from_names(
                    space, {
                        split_iname: 1,
                        inner_iname: -1,
                        outer_iname: -inner_length})))

        name_dim_type, name_idx = space.get_var_dict()[split_iname]
        s = s.intersect(inner_constraint_set)

        if within is None:
            s = s.project_out(name_dim_type, name_idx, 1)

        return s

    new_domains = [process_set(dom) for dom in kernel.domains]

    from pymbolic import var
    inner = var(inner_iname)
    outer = var(outer_iname)
    new_loop_index = inner + outer*inner_length

    subst_map = {var(split_iname): new_loop_index}
    applied_iname_rewrites.append(subst_map)

    # {{{ update forced_iname deps

    new_insns = []
    for insn in kernel.instructions:
        if split_iname in insn.forced_iname_deps:
            new_forced_iname_deps = (
                    (insn.forced_iname_deps.copy()
                    - frozenset([split_iname]))
                    | frozenset([outer_iname, inner_iname]))
        else:
            new_forced_iname_deps = insn.forced_iname_deps

        insn = insn.copy(
                forced_iname_deps=new_forced_iname_deps)

        new_insns.append(insn)

    # }}}

    iname_slab_increments = kernel.iname_slab_increments.copy()
    iname_slab_increments[outer_iname] = slabs

    new_loop_priority = []
    for prio_iname in kernel.loop_priority:
        if prio_iname == split_iname:
            new_loop_priority.append(outer_iname)
            new_loop_priority.append(inner_iname)
        else:
            new_loop_priority.append(prio_iname)

    kernel = kernel.copy(
            domains=new_domains,
            iname_slab_increments=iname_slab_increments,
            instructions=new_insns,
            applied_iname_rewrites=applied_iname_rewrites,
            loop_priority=new_loop_priority)

    from loopy.context_matching import parse_stack_match
    within = parse_stack_match(within)

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    ins = _InameSplitter(rule_mapping_context, within,
            split_iname, outer_iname, inner_iname, new_loop_index)

    kernel = ins.map_kernel(kernel)
    kernel = rule_mapping_context.finish_kernel(kernel)

    if existing_tag is not None:
        kernel = tag_inames(kernel,
                {outer_iname: existing_tag, inner_iname: existing_tag})

    return tag_inames(kernel, {outer_iname: outer_tag, inner_iname: inner_tag})

# }}}


# {{{ join inames

class _InameJoiner(RuleAwareSubstitutionMapper):
    def __init__(self, rule_mapping_context, within, subst_func,
            joined_inames, new_iname):
        super(_InameJoiner, self).__init__(rule_mapping_context,
                subst_func, within)

        self.joined_inames = set(joined_inames)
        self.new_iname = new_iname

    def map_reduction(self, expr, expn_state):
        expr_inames = set(expr.inames)
        overlap = (self.join_inames & expr_inames
                - set(expn_state.arg_context))
        if overlap and self.within(
                expn_state.kernel,
                expn_state.instruction,
                expn_state.stack):
            if overlap != expr_inames:
                raise LoopyError(
                        "Cannot join inames '%s' if there is a reduction "
                        "that does not use all of the inames being joined. "
                        "(Found one with just '%s'.)"
                        % (
                            ", ".join(self.joined_inames),
                            ", ".join(expr_inames)))

            new_inames = expr_inames - self.joined_inames
            new_inames.add(self.new_iname)

            from loopy.symbolic import Reduction
            return Reduction(expr.operation, tuple(new_inames),
                        self.rec(expr.expr, expn_state))
        else:
            return super(_InameJoiner, self).map_reduction(expr, expn_state)


def join_inames(kernel, inames, new_iname=None, tag=None, within=None):
    """
    :arg inames: fastest varying last
    :arg within: a stack match as understood by
        :func:`loopy.context_matching.parse_stack_match`.
    """

    # now fastest varying first
    inames = inames[::-1]

    if new_iname is None:
        new_iname = kernel.get_var_name_generator()("_and_".join(inames))

    from loopy.kernel.tools import DomainChanger
    domch = DomainChanger(kernel, frozenset(inames))
    for iname in inames:
        if kernel.get_home_domain_index(iname) != domch.leaf_domain_index:
            raise LoopyError("iname '%s' is not 'at home' in the "
                    "join's leaf domain" % iname)

    new_domain = domch.domain
    new_dim_idx = new_domain.dim(dim_type.set)
    new_domain = new_domain.add_dims(dim_type.set, 1)
    new_domain = new_domain.set_dim_name(dim_type.set, new_dim_idx, new_iname)

    joint_aff = zero = isl.Aff.zero_on_domain(new_domain.space)
    subst_dict = {}
    base_divisor = 1

    from pymbolic import var

    for i, iname in enumerate(inames):
        iname_dt, iname_idx = zero.get_space().get_var_dict()[iname]
        iname_aff = zero.add_coefficient_val(iname_dt, iname_idx, 1)

        joint_aff = joint_aff + base_divisor*iname_aff

        bounds = kernel.get_iname_bounds(iname, constants_only=True)

        from loopy.isl_helpers import (
                static_max_of_pw_aff, static_value_of_pw_aff)
        from loopy.symbolic import pw_aff_to_expr

        length = int(pw_aff_to_expr(
            static_max_of_pw_aff(bounds.size, constants_only=True)))

        try:
            lower_bound_aff = static_value_of_pw_aff(
                    bounds.lower_bound_pw_aff.coalesce(),
                    constants_only=False)
        except Exception as e:
            raise type(e)("while finding lower bound of '%s': " % iname)

        my_val = var(new_iname) // base_divisor
        if i+1 < len(inames):
            my_val %= length
        my_val += pw_aff_to_expr(lower_bound_aff)
        subst_dict[iname] = my_val

        base_divisor *= length

    from loopy.isl_helpers import iname_rel_aff
    new_domain = new_domain.add_constraint(
            isl.Constraint.equality_from_aff(
                iname_rel_aff(new_domain.get_space(), new_iname, "==", joint_aff)))

    for i, iname in enumerate(inames):
        iname_to_dim = new_domain.get_space().get_var_dict()
        iname_dt, iname_idx = iname_to_dim[iname]

        if within is None:
            new_domain = new_domain.project_out(iname_dt, iname_idx, 1)

    def subst_forced_iname_deps(fid):
        result = set()
        for iname in fid:
            if iname in inames:
                result.add(new_iname)
            else:
                result.add(iname)

        return frozenset(result)

    new_insns = [
            insn.copy(
                forced_iname_deps=subst_forced_iname_deps(insn.forced_iname_deps))
            for insn in kernel.instructions]

    kernel = (kernel
            .copy(
                instructions=new_insns,
                domains=domch.get_domains_with(new_domain),
                applied_iname_rewrites=kernel.applied_iname_rewrites + [subst_dict]
                ))

    from loopy.context_matching import parse_stack_match
    within = parse_stack_match(within)

    from pymbolic.mapper.substitutor import make_subst_func
    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    ijoin = _InameJoiner(rule_mapping_context, within,
            make_subst_func(subst_dict),
            inames, new_iname)

    kernel = rule_mapping_context.finish_kernel(
            ijoin.map_kernel(kernel))

    if tag is not None:
        kernel = tag_inames(kernel, {new_iname: tag})

    return kernel

# }}}


# {{{ tag inames

def tag_inames(kernel, iname_to_tag, force=False):
    from loopy.kernel.data import parse_tag

    iname_to_tag = dict((iname, parse_tag(tag))
            for iname, tag in six.iteritems(iname_to_tag))

    from loopy.kernel.data import (ParallelTag, AutoLocalIndexTagBase,
            ForceSequentialTag)

    new_iname_to_tag = kernel.iname_to_tag.copy()
    for iname, new_tag in six.iteritems(iname_to_tag):
        if iname not in kernel.all_inames():
            raise LoopyError("iname '%s' does not exist" % iname)

        old_tag = kernel.iname_to_tag.get(iname)

        retag_ok = False

        if isinstance(old_tag, (AutoLocalIndexTagBase, ForceSequentialTag)):
            retag_ok = True

        if not retag_ok and old_tag is not None and new_tag is None:
            raise ValueError("cannot untag iname '%s'" % iname)

        if iname not in kernel.all_inames():
            raise ValueError("cannot tag '%s'--not known" % iname)

        if isinstance(new_tag, ParallelTag) \
                and isinstance(old_tag, ForceSequentialTag):
            raise ValueError("cannot tag '%s' as parallel--"
                    "iname requires sequential execution" % iname)

        if isinstance(new_tag, ForceSequentialTag) \
                and isinstance(old_tag, ParallelTag):
            raise ValueError("'%s' is already tagged as parallel, "
                    "but is now prohibited from being parallel "
                    "(likely because of participation in a precompute or "
                    "a reduction)" % iname)

        if (not retag_ok) and (not force) \
                and old_tag is not None and (old_tag != new_tag):
            raise LoopyError("'%s' is already tagged '%s'--cannot retag"
                    % (iname, old_tag))

        new_iname_to_tag[iname] = new_tag

    return kernel.copy(iname_to_tag=new_iname_to_tag)

# }}}


# {{{ duplicate inames

class _InameDuplicator(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context,
            old_to_new, within):
        super(_InameDuplicator, self).__init__(rule_mapping_context)

        self.old_to_new = old_to_new
        self.old_inames_set = set(six.iterkeys(old_to_new))
        self.within = within

    def map_reduction(self, expr, expn_state):
        if (set(expr.inames) & self.old_inames_set
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)):
            new_inames = tuple(
                    self.old_to_new.get(iname, iname)
                    if iname not in expn_state.arg_context
                    else iname
                    for iname in expr.inames)

            from loopy.symbolic import Reduction
            return Reduction(expr.operation, new_inames,
                        self.rec(expr.expr, expn_state))
        else:
            return super(_InameDuplicator, self).map_reduction(expr, expn_state)

    def map_variable(self, expr, expn_state):
        new_name = self.old_to_new.get(expr.name)

        if (new_name is None
                or expr.name in expn_state.arg_context
                or not self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)):
            return super(_InameDuplicator, self).map_variable(expr, expn_state)
        else:
            from pymbolic import var
            return var(new_name)

    def map_instruction(self, kernel, insn):
        if not self.within(kernel, insn, ()):
            return insn

        new_fid = frozenset(
                self.old_to_new.get(iname, iname)
                for iname in insn.forced_iname_deps)
        return insn.copy(forced_iname_deps=new_fid)


def duplicate_inames(knl, inames, within, new_inames=None, suffix=None,
        tags={}):
    """
    :arg within: a stack match as understood by
        :func:`loopy.context_matching.parse_stack_match`.
    """

    # {{{ normalize arguments, find unique new_inames

    if isinstance(inames, str):
        inames = [iname.strip() for iname in inames.split(",")]

    if isinstance(new_inames, str):
        new_inames = [iname.strip() for iname in new_inames.split(",")]

    from loopy.context_matching import parse_stack_match
    within = parse_stack_match(within)

    if new_inames is None:
        new_inames = [None] * len(inames)

    if len(new_inames) != len(inames):
        raise ValueError("new_inames must have the same number of entries as inames")

    name_gen = knl.get_var_name_generator()

    for i, iname in enumerate(inames):
        new_iname = new_inames[i]

        if new_iname is None:
            new_iname = iname

            if suffix is not None:
                new_iname += suffix

            new_iname = name_gen(new_iname)

        else:
            if name_gen.is_name_conflicting(new_iname):
                raise ValueError("new iname '%s' conflicts with existing names"
                        % new_iname)

            name_gen.add_name(new_iname)

        new_inames[i] = new_iname

    # }}}

    # {{{ duplicate the inames

    for old_iname, new_iname in zip(inames, new_inames):
        from loopy.kernel.tools import DomainChanger
        domch = DomainChanger(knl, frozenset([old_iname]))

        from loopy.isl_helpers import duplicate_axes
        knl = knl.copy(
                domains=domch.get_domains_with(
                    duplicate_axes(domch.domain, [old_iname], [new_iname])))

    # }}}

    # {{{ change the inames in the code

    rule_mapping_context = SubstitutionRuleMappingContext(
            knl.substitutions, name_gen)
    indup = _InameDuplicator(rule_mapping_context,
            old_to_new=dict(list(zip(inames, new_inames))),
            within=within)

    knl = rule_mapping_context.finish_kernel(
            indup.map_kernel(knl))

    # }}}

    # {{{ realize tags

    for old_iname, new_iname in zip(inames, new_inames):
        new_tag = tags.get(old_iname)
        if new_tag is not None:
            knl = tag_inames(knl, {new_iname: new_tag})

    # }}}

    return knl

# }}}


def rename_iname(knl, old_iname, new_iname, within):
    """
    :arg within: a stack match as understood by
        :func:`loopy.context_matching.parse_stack_match`.
    """

    var_name_gen = knl.get_var_name_generator()

    if var_name_gen.is_name_conflicting(new_iname):
        raise ValueError("iname '%s' conflicts with an existing identifier"
                "--cannot rename" % new_iname)

    knl = duplicate_inames(knl, [old_iname], within=within, new_inames=[new_iname])
    knl = remove_unused_inames(knl, [old_iname])

    return knl


# {{{ link inames

def link_inames(knl, inames, new_iname, within=None, tag=None):
    # {{{ normalize arguments

    if isinstance(inames, str):
        inames = inames.split(",")

    var_name_gen = knl.get_var_name_generator()
    new_iname = var_name_gen(new_iname)

    # }}}

    # {{{ ensure that each iname is used at most once in each instruction

    inames_set = set(inames)

    if 0:
        # FIXME!
        for insn in knl.instructions:
            insn_inames = knl.insn_inames(insn.id) | insn.reduction_inames()

            if len(insn_inames & inames_set) > 1:
                raise LoopyError("To-be-linked inames '%s' are used in "
                        "instruction '%s'. No more than one such iname can "
                        "be used in one instruction."
                        % (", ".join(insn_inames & inames_set), insn.id))

    # }}}

    from loopy.kernel.tools import DomainChanger
    domch = DomainChanger(knl, tuple(inames))

    # {{{ ensure that projections are identical

    unrelated_dom_inames = list(
            set(domch.domain.get_var_names(dim_type.set))
            - inames_set)

    domain = domch.domain

    # move all inames to be linked to end to prevent shuffly confusion
    for iname in inames:
        dt, index = domain.get_var_dict()[iname]
        assert dt == dim_type.set

        # move to tail of param dim_type
        domain = domain.move_dims(
                    dim_type.param, domain.dim(dim_type.param),
                    dt, index, 1)
        # move to tail of set dim_type
        domain = domain.move_dims(
                    dim_type.set, domain.dim(dim_type.set),
                    dim_type.param, domain.dim(dim_type.param)-1, 1)

    projections = [
            domch.domain.project_out_except(
                unrelated_dom_inames + [iname], [dim_type.set])
            for iname in inames]

    all_equal = True
    first_proj = projections[0]
    for proj in projections[1:]:
        all_equal = all_equal and (proj <= first_proj and first_proj <= proj)

    if not all_equal:
        raise LoopyError("Inames cannot be linked because their domain "
                "constraints are not the same.")

    del domain  # messed up for testing, do not use

    # }}}

    # change the domain
    from loopy.isl_helpers import duplicate_axes
    knl = knl.copy(
            domains=domch.get_domains_with(
                duplicate_axes(domch.domain, [inames[0]], [new_iname])))

    # {{{ change the code

    from pymbolic import var
    subst_dict = dict((iname, var(new_iname)) for iname in inames)

    from loopy.context_matching import parse_stack_match
    within = parse_stack_match(within)

    from pymbolic.mapper.substitutor import make_subst_func
    rule_mapping_context = SubstitutionRuleMappingContext(
            knl.substitutions, var_name_gen)
    ijoin = RuleAwareSubstitutionMapper(rule_mapping_context,
                    make_subst_func(subst_dict), within)

    knl = rule_mapping_context.finish_kernel(
            ijoin.map_kernel(knl))

    # }}}

    knl = remove_unused_inames(knl, inames)

    if tag is not None:
        knl = tag_inames(knl, {new_iname: tag})

    return knl

# }}}


# {{{ remove unused inames

def remove_unused_inames(knl, inames=None):
    """Delete those among *inames* that are unused, i.e. project them
    out of the domain. If these inames pose implicit restrictions on
    other inames, these restrictions will persist as existentially
    quantified variables.

    :arg inames: may be an iterable of inames or a string of comma-separated inames.
    """

    # {{{ normalize arguments

    if inames is None:
        inames = knl.all_inames()
    elif isinstance(inames, str):
        inames = inames.split(",")

    # }}}

    # {{{ check which inames are unused

    inames = set(inames)
    used_inames = set()
    for insn in knl.instructions:
        used_inames.update(knl.insn_inames(insn.id))

    unused_inames = inames - used_inames

    # }}}

    # {{{ remove them

    from loopy.kernel.tools import DomainChanger

    for iname in unused_inames:
        domch = DomainChanger(knl, (iname,))

        dom = domch.domain
        dt, idx = dom.get_var_dict()[iname]
        dom = dom.project_out(dt, idx, 1)

        knl = knl.copy(domains=domch.get_domains_with(dom))

    # }}}

    return knl

# }}}


# {{{ set loop priority

def set_loop_priority(kernel, loop_priority):
    """Indicates the textual order in which loops should be entered in the
    kernel code. Note that this priority has an advisory role only. If the
    kernel logically requires a different nesting, priority is ignored.
    Priority is only considered if loop nesting is ambiguous.

    :arg: an iterable of inames, or, for brevity, a comma-seaprated string of
        inames
    """

    if isinstance(loop_priority, str):
        loop_priority = [s.strip() for s in loop_priority.split(",")]

    return kernel.copy(loop_priority=loop_priority)

# }}}


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
        default_tag="l.auto", rule_name=None, footprint_subscripts=None,
        fetch_bounding_box=False):
    """Prefetch all accesses to the variable *var_name*, with all accesses
    being swept through *sweep_inames*.

    :ivar dim_arg_names: List of names representing each fetch axis.
    :ivar rule_name: base name of the generated temporary variable.
    :ivar footprint_subscripts: A list of tuples indicating the index (i.e.
        subscript) tuples used to generate the footprint.

        If only one such set of indices is desired, this may also be specified
        directly by putting an index expression into *var_name*. Substitutions
        such as those occurring in dimension splits are recorded and also
        applied to these indices.

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
        rule_name = var_name_gen("%s_fetch" % c_name)

    arg = kernel.arg_dict[var_name]

    # {{{ make parameter names and unification template

    parameters = []
    for i in range(arg.num_user_axes()):
        based_on = "%s_dim_%d" % (c_name, i)
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

    kernel = extract_subst(kernel, rule_name, uni_template, parameters)

    if isinstance(sweep_inames, str):
        sweep_inames = [s.strip() for s in sweep_inames.split(",")]

    kernel, subst_use, sweep_inames, inames_to_be_removed = \
            _process_footprint_subscripts(
                    kernel,  rule_name, sweep_inames,
                    footprint_subscripts, arg)
    new_kernel = precompute(kernel, subst_use, sweep_inames,
            precompute_inames=dim_arg_names,
            default_tag=default_tag, dtype=arg.dtype,
            fetch_bounding_box=fetch_bounding_box)

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
        return expand_subst(new_kernel, "id:"+rule_name)
    else:
        return new_kernel

# }}}


# {{{ instruction processing

def find_instructions(kernel, insn_match):
    from loopy.context_matching import parse_match
    match = parse_match(insn_match)
    return [insn for insn in kernel.instructions if match(kernel, insn)]


def map_instructions(kernel, insn_match, f):
    from loopy.context_matching import parse_match
    match = parse_match(insn_match)

    new_insns = []

    for insn in kernel.instructions:
        if match(kernel, insn):
            new_insns.append(f(insn))
        else:
            new_insns.append(insn)

    return kernel.copy(instructions=new_insns)


def set_instruction_priority(kernel, insn_match, priority):
    """Set the priority of instructions matching *insn_match* to *priority*.

    *insn_match* may be any instruction id match understood by
    :func:`loopy.context_matching.parse_match`.
    """

    def set_prio(insn):
        return insn.copy(priority=priority)

    return map_instructions(kernel, insn_match, set_prio)


def add_dependency(kernel, insn_match, dependency):
    """Add the instruction dependency *dependency* to the instructions matched
    by *insn_match*.

    *insn_match* may be any instruction id match understood by
    :func:`loopy.context_matching.parse_match`.
    """

    if dependency not in kernel.id_to_insn:
        raise LoopyError("cannot add dependency on non-existent instruction ID '%s'"
                % dependency)

    def add_dep(insn):
        new_deps = insn.insn_deps
        added_deps = frozenset([dependency])
        if new_deps is None:
            new_deps = added_deps
        else:
            new_deps = new_deps | added_deps

        return insn.copy(insn_deps=new_deps)

    return map_instructions(kernel, insn_match, add_dep)


def remove_instructions(kernel, insn_ids):
    """Return a new kernel with instructions in *insn_ids* removed.

    Dependencies across (one, for now) deleted isntructions are propagated.
    Behavior is undefined for now for chains of dependencies within the
    set of deleted instructions.
    """

    if not insn_ids:
        return kernel

    assert isinstance(insn_ids, set)
    id_to_insn = kernel.id_to_insn

    new_insns = []
    for insn in kernel.instructions:
        if insn.id in insn_ids:
            continue

        # transitively propagate dependencies
        # (only one level for now)
        if insn.insn_deps is None:
            insn_deps = frozenset()
        else:
            insn_deps = insn.insn_deps

        new_deps = insn_deps - insn_ids

        for dep_id in insn_deps & insn_ids:
            new_deps = new_deps | id_to_insn[dep_id].insn_deps

        new_insns.append(insn.copy(insn_deps=frozenset(new_deps)))

    return kernel.copy(
            instructions=new_insns)

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


# {{{ tag data axes

def tag_data_axes(knl, ary_names, dim_tags):
    for ary_name in ary_names.split(","):
        ary_name = ary_name.strip()
        if ary_name in knl.temporary_variables:
            ary = knl.temporary_variables[ary_name]
        elif ary_name in knl.arg_dict:
            ary = knl.arg_dict[ary_name]
        else:
            raise NameError("array '%s' was not found" % ary_name)

        from loopy.kernel.array import parse_array_dim_tags
        new_dim_tags = parse_array_dim_tags(dim_tags,
                use_increasing_target_axes=ary.max_target_axes > 1)

        ary = ary.copy(dim_tags=tuple(new_dim_tags))

        if ary_name in knl.temporary_variables:
            new_tv = knl.temporary_variables.copy()
            new_tv[ary_name] = ary
            knl = knl.copy(temporary_variables=new_tv)

        elif ary_name in knl.arg_dict:
            new_args = []
            for arg in knl.args:
                if arg.name == ary_name:
                    new_args.append(ary)
                else:
                    new_args.append(arg)

            knl = knl.copy(args=new_args)

        else:
            raise NameError("array '%s' was not found" % ary_name)

    return knl

# }}}


# {{{ split_reduction

class _ReductionSplitter(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, within, inames, direction):
        super(_ReductionSplitter, self).__init__(
                rule_mapping_context)

        self.within = within
        self.inames = inames
        self.direction = direction

    def map_reduction(self, expr, expn_state):
        if set(expr.inames) & set(expn_state.arg_context):
            # FIXME
            raise NotImplementedError()

        if (self.inames <= set(expr.inames)
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)):
            leftover_inames = set(expr.inames) - self.inames

            from loopy.symbolic import Reduction
            if self.direction == "in":
                return Reduction(expr.operation, tuple(leftover_inames),
                        Reduction(expr.operation, tuple(self.inames),
                            self.rec(expr.expr, expn_state)))
            elif self.direction == "out":
                return Reduction(expr.operation, tuple(self.inames),
                        Reduction(expr.operation, tuple(leftover_inames),
                            self.rec(expr.expr, expn_state)))
            else:
                assert False
        else:
            return super(_ReductionSplitter, self).map_reduction(expr, expn_state)


def _split_reduction(kernel, inames, direction, within=None):
    if direction not in ["in", "out"]:
        raise ValueError("invalid value for 'direction': %s" % direction)

    if isinstance(inames, str):
        inames = inames.split(",")
    inames = set(inames)

    from loopy.context_matching import parse_stack_match
    within = parse_stack_match(within)

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    rsplit = _ReductionSplitter(rule_mapping_context,
            within, inames, direction)
    return rule_mapping_context.finish_kernel(
            rsplit.map_kernel(kernel))


def split_reduction_inward(kernel, inames, within=None):
    """Takes a reduction of the form::

        sum([i,j,k], ...)

    and splits it into two nested reductions::

        sum([j,k], sum([i], ...))

    In this case, *inames* would have been ``"i"`` indicating that
    the iname ``i`` should be made the iname governing the inner reduction.

    :arg inames: A list of inames, or a comma-separated string that can
        be parsed into those
    """

    return _split_reduction(kernel, inames, "in", within)


def split_reduction_outward(kernel, inames, within=None):
    """Takes a reduction of the form::

        sum([i,j,k], ...)

    and splits it into two nested reductions::

        sum([i], sum([j,k], ...))

    In this case, *inames* would have been ``"i"`` indicating that
    the iname ``i`` should be made the iname governing the outer reduction.

    :arg inames: A list of inames, or a comma-separated string that can
        be parsed into those
    """

    return _split_reduction(kernel, inames, "out", within)

# }}}


# {{{ fix_parameter

def _fix_parameter(kernel, name, value):
    def process_set(s):
        var_dict = s.get_var_dict()

        try:
            dt, idx = var_dict[name]
        except KeyError:
            return s

        value_aff = isl.Aff.zero_on_domain(s.space) + value

        from loopy.isl_helpers import iname_rel_aff
        name_equal_value_aff = iname_rel_aff(s.space, name, "==", value_aff)

        s = (s
                .add_constraint(
                    isl.Constraint.equality_from_aff(name_equal_value_aff))
                .project_out(dt, idx, 1))

        return s

    new_domains = [process_set(dom) for dom in kernel.domains]

    from pymbolic.mapper.substitutor import make_subst_func
    subst_func = make_subst_func({name: value})

    from loopy.symbolic import SubstitutionMapper, PartialEvaluationMapper
    subst_map = SubstitutionMapper(subst_func)
    ev_map = PartialEvaluationMapper()

    def map_expr(expr):
        return ev_map(subst_map(expr))

    from loopy.kernel.array import ArrayBase
    new_args = []
    for arg in kernel.args:
        if arg.name == name:
            # remove from argument list
            continue

        if not isinstance(arg, ArrayBase):
            new_args.append(arg)
        else:
            new_args.append(arg.map_exprs(map_expr))

    new_temp_vars = {}
    for tv in six.itervalues(kernel.temporary_variables):
        new_temp_vars[tv.name] = tv.map_exprs(map_expr)

    from loopy.context_matching import parse_stack_match
    within = parse_stack_match(None)

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    esubst_map = RuleAwareSubstitutionMapper(
            rule_mapping_context, subst_func, within=within)
    return (
            rule_mapping_context.finish_kernel(
                esubst_map.map_kernel(kernel))
            .copy(
                domains=new_domains,
                args=new_args,
                temporary_variables=new_temp_vars,
                assumptions=process_set(kernel.assumptions),
                ))


def fix_parameters(kernel, **value_dict):
    for name, value in six.iteritems(value_dict):
        kernel = _fix_parameter(kernel, name, value)

    return kernel

# }}}


# {{{ assume

def assume(kernel, assumptions):
    if isinstance(assumptions, str):
        assumptions_set_str = "[%s] -> { : %s}" \
                % (",".join(s for s in kernel.outer_params()),
                    assumptions)
        assumptions = isl.BasicSet.read_from_str(kernel.domains[0].get_ctx(),
                assumptions_set_str)

    if not isinstance(assumptions, isl.BasicSet):
        raise TypeError("'assumptions' must be a BasicSet or a string")

    old_assumptions, new_assumptions = isl.align_two(kernel.assumptions, assumptions)

    return kernel.copy(
            assumptions=old_assumptions.params() & new_assumptions.params())

# }}}


# {{{ set_options

def set_options(kernel, *args, **kwargs):
    """Return a new kernel with the options given as keyword arguments, or from
    a string representation passed in as the first (and only) positional
    argument.

    See also :class:`Options`.
    """

    if args and kwargs:
        raise TypeError("cannot pass both positional and keyword arguments")

    new_opt = kernel.options.copy()

    if kwargs:
        for key, val in six.iteritems(kwargs):
            if not hasattr(new_opt, key):
                raise ValueError("unknown option '%s'" % key)

            setattr(new_opt, key, val)
    else:
        if len(args) != 1:
            raise TypeError("exactly one positional argument is required if "
                    "no keyword args are given")
        arg, = args

        from loopy.options import make_options
        new_opt.update(make_options(arg))

    return kernel.copy(options=new_opt)

# }}}


# {{{ library registration

def register_preamble_generators(kernel, preamble_generators):
    new_pgens = kernel.preamble_generators[:]
    for pgen in preamble_generators:
        if pgen not in new_pgens:
            new_pgens.insert(0, pgen)

    return kernel.copy(preamble_generators=new_pgens)


def register_symbol_manglers(kernel, manglers):
    new_manglers = kernel.symbol_manglers[:]
    for m in manglers:
        if m not in new_manglers:
            new_manglers.insert(0, m)

    return kernel.copy(symbol_manglers=new_manglers)


def register_function_manglers(kernel, manglers):
    new_manglers = kernel.function_manglers[:]
    for m in manglers:
        if m not in new_manglers:
            new_manglers.insert(0, m)

    return kernel.copy(function_manglers=new_manglers)

# }}}


# {{{ cache control

import os
CACHING_ENABLED = (
    "LOOPY_NO_CACHE" not in os.environ
    and
    "CG_NO_CACHE" not in os.environ)


def set_caching_enabled(flag):
    """Set whether :mod:`loopy` is allowed to use disk caching for its various
    code generation stages.
    """
    global CACHING_ENABLED
    CACHING_ENABLED = flag


class CacheMode(object):
    """A context manager for setting whether :mod:`loopy` is allowed to use
    disk caches.
    """

    def __init__(self, new_flag):
        self.new_flag = new_flag

    def __enter__(self):
        global CACHING_ENABLED
        self.previous_mode = CACHING_ENABLED
        CACHING_ENABLED = self.new_flag

    def __exit__(self, exc_type, exc_val, exc_tb):
        global CACHING_ENABLED
        CACHING_ENABLED = self.previous_mode
        del self.previous_mode

# }}}


# {{{ data layout change

def make_copy_kernel(new_dim_tags, old_dim_tags=None):
    """Returns a :class:`LoopKernel` that changes the data layout
    of a variable (called "input") to the new layout specified by
    *new_dim_tags* from the one specified by *old_dim_tags*.
    *old_dim_tags* defaults to an all-C layout of the same rank
    as the one given by *new_dim_tags*.
    """

    from loopy.kernel.array import (parse_array_dim_tags,
            SeparateArrayArrayDimTag, VectorArrayDimTag)
    new_dim_tags = parse_array_dim_tags(new_dim_tags)

    rank = len(new_dim_tags)
    if old_dim_tags is None:
        old_dim_tags = parse_array_dim_tags(",".join(rank * ["c"]))
    elif isinstance(old_dim_tags, str):
        old_dim_tags = parse_array_dim_tags(old_dim_tags)

    indices = ["i%d" % i for i in range(rank)]
    shape = ["n%d" % i for i in range(rank)]
    commad_indices = ", ".join(indices)
    bounds = " and ".join(
            "0<=%s<%s" % (ind, shape_i)
            for ind, shape_i in zip(indices, shape))

    set_str = "{[%s]: %s}" % (
                commad_indices,
                bounds
                )
    result = make_kernel(set_str,
            "output[%s] = input[%s]"
            % (commad_indices, commad_indices))

    result = tag_data_axes(result, "input", old_dim_tags)
    result = tag_data_axes(result, "output", new_dim_tags)

    unrolled_tags = (SeparateArrayArrayDimTag, VectorArrayDimTag)
    for i in range(rank):
        if (isinstance(new_dim_tags[i], unrolled_tags)
                or isinstance(old_dim_tags[i], unrolled_tags)):
            result = tag_inames(result, {indices[i]: "unr"})

    return result

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


# {{{ affine map inames

def affine_map_inames(kernel, old_inames, new_inames, equations):
    """Return a new *kernel* where the affine transform
    specified by *equations* has been applied to the inames.

    :arg old_inames: A list of inames to be replaced by affine transforms
        of their values.
        May also be a string of comma-separated inames.

    :arg new_inames: A list of new inames that are not yet used in *kernel*,
        but have their values established in terms of *old_inames* by
        *equations*.
        May also be a string of comma-separated inames.
    :arg equations: A list of equations estabilishing a relationship
        between *old_inames* and *new_inames*. Each equation may be
        a tuple ``(lhs, rhs)`` of expressions or a string, with left and
        right hand side of the equation separated by ``=``.
    """

    # {{{ check and parse arguments

    if isinstance(new_inames, str):
        new_inames = new_inames.split(",")
        new_inames = [iname.strip() for iname in new_inames]
    if isinstance(old_inames, str):
        old_inames = old_inames.split(",")
        old_inames = [iname.strip() for iname in old_inames]
    if isinstance(equations, str):
        equations = [equations]

    import re
    eqn_re = re.compile(r"^([^=]+)=([^=]+)$")

    def parse_equation(eqn):
        if isinstance(eqn, str):
            eqn_match = eqn_re.match(eqn)
            if not eqn_match:
                raise ValueError("invalid equation: %s" % eqn)

            from loopy.symbolic import parse
            lhs = parse(eqn_match.group(1))
            rhs = parse(eqn_match.group(2))
            return (lhs, rhs)
        elif isinstance(eqn, tuple):
            if len(eqn) != 2:
                raise ValueError("unexpected length of equation tuple, "
                        "got %d, should be 2" % len(eqn))
            return eqn
        else:
            raise ValueError("unexpected type of equation"
                    "got %d, should be string or tuple"
                    % type(eqn).__name__)

    equations = [parse_equation(eqn) for eqn in equations]

    all_vars = kernel.all_variable_names()
    for iname in new_inames:
        if iname in all_vars:
            raise LoopyError("new iname '%s' is already used in kernel"
                    % iname)

    for iname in old_inames:
        if iname not in kernel.all_inames():
            raise LoopyError("old iname '%s' not known" % iname)

    # }}}

    # {{{ substitute iname use

    from pymbolic.algorithm import solve_affine_equations_for
    old_inames_to_expr = solve_affine_equations_for(old_inames, equations)

    subst_dict = dict(
            (v.name, expr)
            for v, expr in old_inames_to_expr.items())

    var_name_gen = kernel.get_var_name_generator()

    from pymbolic.mapper.substitutor import make_subst_func
    from loopy.context_matching import parse_stack_match

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, var_name_gen)
    old_to_new = RuleAwareSubstitutionMapper(rule_mapping_context,
            make_subst_func(subst_dict), within=parse_stack_match(None))

    kernel = (
            rule_mapping_context.finish_kernel(
                old_to_new.map_kernel(kernel))
            .copy(
                applied_iname_rewrites=kernel.applied_iname_rewrites + [subst_dict]
                ))

    # }}}

    # {{{ change domains

    new_inames_set = set(new_inames)
    old_inames_set = set(old_inames)

    new_domains = []
    for idom, dom in enumerate(kernel.domains):
        dom_var_dict = dom.get_var_dict()
        old_iname_overlap = [
                iname
                for iname in old_inames
                if iname in dom_var_dict]

        if not old_iname_overlap:
            new_domains.append(dom)
            continue

        from loopy.symbolic import get_dependencies
        dom_new_inames = set()
        dom_old_inames = set()

        # mapping for new inames to dim_types
        new_iname_dim_types = {}

        dom_equations = []
        for iname in old_iname_overlap:
            for ieqn, (lhs, rhs) in enumerate(equations):
                eqn_deps = get_dependencies(lhs) | get_dependencies(rhs)
                if iname in eqn_deps:
                    dom_new_inames.update(eqn_deps & new_inames_set)
                    dom_old_inames.update(eqn_deps & old_inames_set)

                if dom_old_inames:
                    dom_equations.append((lhs, rhs))

                this_eqn_old_iname_dim_types = set(
                        dom_var_dict[old_iname][0]
                        for old_iname in eqn_deps & old_inames_set)

                if this_eqn_old_iname_dim_types:
                    if len(this_eqn_old_iname_dim_types) > 1:
                        raise ValueError("inames '%s' (from equation %d (0-based)) "
                                "in domain %d (0-based) are not "
                                "of a uniform dim_type"
                                % (", ".join(eqn_deps & old_inames_set), ieqn, idom))

                    this_eqn_new_iname_dim_type, = this_eqn_old_iname_dim_types

                    for new_iname in eqn_deps & new_inames_set:
                        if new_iname in new_iname_dim_types:
                            if (this_eqn_new_iname_dim_type
                                    != new_iname_dim_types[new_iname]):
                                raise ValueError("dim_type disagreement for "
                                        "iname '%s' (from equation %d (0-based)) "
                                        "in domain %d (0-based)"
                                        % (new_iname, ieqn, idom))
                        else:
                            new_iname_dim_types[new_iname] = \
                                    this_eqn_new_iname_dim_type

        if not dom_old_inames <= set(dom_var_dict):
            raise ValueError("domain %d (0-based) does not know about "
                    "all old inames (specifically '%s') needed to define new inames"
                    % (idom, ", ".join(dom_old_inames - set(dom_var_dict))))

        # add inames to domain with correct dim_types
        dom_new_inames = list(dom_new_inames)
        for iname in dom_new_inames:
            dt = new_iname_dim_types[iname]
            iname_idx = dom.dim(dt)
            dom = dom.add_dims(dt, 1)
            dom = dom.set_dim_name(dt, iname_idx, iname)

        # add equations
        from loopy.symbolic import aff_from_expr
        for lhs, rhs in dom_equations:
            dom = dom.add_constraint(
                    isl.Constraint.equality_from_aff(
                        aff_from_expr(dom.space, rhs - lhs)))

        # project out old inames
        for iname in dom_old_inames:
            dt, idx = dom.get_var_dict()[iname]
            dom = dom.project_out(dt, idx, 1)

        new_domains.append(dom)

    # }}}

    return kernel.copy(domains=new_domains)

# }}}


# {{{ fold constants

def fold_constants(kernel):
    from loopy.symbolic import ConstantFoldingMapper
    cfm = ConstantFoldingMapper()

    new_insns = [
            insn.with_transformed_expressions(cfm)
            for insn in kernel.instructions]

    new_substs = dict(
            (sub.name,
                sub.copy(expression=cfm(sub.expression)))
            for sub in six.itervalues(kernel.substitutions))

    return kernel.copy(
            instructions=new_insns,
            substitutions=new_substs)

# }}}


# {{{ tag_instructions

def tag_instructions(kernel, new_tag, within=None):
    from loopy.context_matching import parse_match
    within = parse_match(within)

    new_insns = []
    for insn in kernel.instructions:
        if within(kernel, insn):
            new_insns.append(
                    insn.copy(tags=insn.tags + (new_tag,)))
        else:
            new_insns.append(insn)

    return kernel.copy(instructions=new_insns)

# }}}

# vim: foldmethod=marker

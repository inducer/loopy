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


def register_mpz_with_pymbolic():
    from pymbolic.primitives import register_constant_class
    import gmpy
    mpz_type = type(gmpy.mpz(1))
    register_constant_class(mpz_type)

register_mpz_with_pymbolic()

import islpy as isl
from islpy import dim_type

from loopy.symbolic import ExpandingIdentityMapper, ExpandingSubstitutionMapper
from loopy.diagnostic import LoopyError


# {{{ imported user interface

from loopy.library.function import (
        default_function_mangler, single_arg_function_mangler,
        opencl_function_mangler)

from loopy.library.preamble import default_preamble_generator

from loopy.library.symbol import opencl_symbol_mangler

from loopy.kernel.data import (
        auto,
        ValueArg, GlobalArg, ConstantArg, ImageArg,
        ExpressionInstruction, CInstruction,
        TemporaryVariable)

from loopy.kernel import LoopKernel
from loopy.kernel.tools import (
        add_argument_dtypes,
        add_and_infer_argument_dtypes)
from loopy.kernel.creation import make_kernel
from loopy.library.reduction import register_reduction_parser
from loopy.subst import extract_subst, expand_subst
from loopy.precompute import precompute
from loopy.padding import (split_arg_axis, find_padding_multiple,
        add_padding)
from loopy.preprocess import (preprocess_kernel, realize_reduction,
        infer_unknown_types)
from loopy.schedule import generate_loop_schedules
from loopy.codegen import generate_code
from loopy.compiled import CompiledKernel
from loopy.flags import LoopyFlags
from loopy.auto_test import auto_test_vs_ref

__all__ = [
        "auto",

        "LoopKernel",

        "ValueArg", "ScalarArg", "GlobalArg", "ArrayArg", "ConstantArg", "ImageArg",
        "TemporaryVariable",

        "ExpressionInstruction", "CInstruction",

        "default_function_mangler", "single_arg_function_mangler",
        "opencl_function_mangler", "opencl_symbol_mangler",
        "default_preamble_generator",
        "make_kernel",
        "register_reduction_parser",

        "extract_subst", "expand_subst",
        "precompute",
        "split_arg_axis", "find_padding_multiple", "add_padding",

        "get_dot_dependency_graph", "add_argument_dtypes",
        "infer_argument_dtypes", "add_and_infer_argument_dtypes",

        "preprocess_kernel", "realize_reduction", "infer_unknown_types",
        "generate_loop_schedules",
        "generate_code",

        "CompiledKernel",

        "auto_test_vs_ref",

        "LoopyFlags",

        "make_kernel",

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

class _InameSplitter(ExpandingIdentityMapper):
    def __init__(self, kernel, within,
            split_iname, outer_iname, inner_iname, replacement_index):
        ExpandingIdentityMapper.__init__(self,
                kernel.substitutions, kernel.get_var_name_generator())

        self.within = within

        self.split_iname = split_iname
        self.outer_iname = outer_iname
        self.inner_iname = inner_iname

        self.replacement_index = replacement_index

    def map_reduction(self, expr, expn_state):
        if self.split_iname in expr.inames and self.within(expn_state.stack):
            new_inames = list(expr.inames)
            new_inames.remove(self.split_iname)
            new_inames.extend([self.outer_iname, self.inner_iname])

            from loopy.symbolic import Reduction
            return Reduction(expr.operation, tuple(new_inames),
                        self.rec(expr.expr, expn_state))
        else:
            return ExpandingIdentityMapper.map_reduction(self, expr, expn_state)

    def map_variable(self, expr, expn_state):
        if expr.name == self.split_iname and self.within(expn_state.stack):
            return self.replacement_index
        else:
            return ExpandingIdentityMapper.map_variable(self, expr, expn_state)


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

    kernel = (kernel
            .copy(domains=new_domains,
                iname_slab_increments=iname_slab_increments,
                instructions=new_insns,
                applied_iname_rewrites=applied_iname_rewrites))

    from loopy.context_matching import parse_stack_match
    within = parse_stack_match(within)

    ins = _InameSplitter(kernel, within,
            split_iname, outer_iname, inner_iname, new_loop_index)

    kernel = ins.map_kernel(kernel)

    if existing_tag is not None:
        kernel = tag_inames(kernel,
                {outer_iname: existing_tag, inner_iname: existing_tag})

    return tag_inames(kernel, {outer_iname: outer_tag, inner_iname: inner_tag})

# }}}


# {{{ join inames

class _InameJoiner(ExpandingSubstitutionMapper):
    def __init__(self, kernel, within, subst_func, joined_inames, new_iname):
        ExpandingSubstitutionMapper.__init__(self,
                kernel.substitutions, kernel.get_var_name_generator(),
                subst_func, within)

        self.joined_inames = set(joined_inames)
        self.new_iname = new_iname

    def map_reduction(self, expr, expn_state):
        expr_inames = set(expr.inames)
        overlap = self.join_inames & expr_inames
        if overlap and self.within(expn_state.stack):
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
            return ExpandingIdentityMapper.map_reduction(self, expr, expn_state)


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
        iname_aff = zero.add_coefficient(iname_dt, iname_idx, 1)

        joint_aff = joint_aff + base_divisor*iname_aff

        bounds = kernel.get_iname_bounds(iname)

        from loopy.isl_helpers import (
                static_max_of_pw_aff, static_value_of_pw_aff)
        from loopy.symbolic import pw_aff_to_expr

        length = int(pw_aff_to_expr(
            static_max_of_pw_aff(bounds.size, constants_only=True)))

        try:
            lower_bound_aff = static_value_of_pw_aff(
                    bounds.lower_bound_pw_aff.coalesce(),
                    constants_only=False)
        except Exception, e:
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
    ijoin = _InameJoiner(kernel, within,
            make_subst_func(subst_dict),
            inames, new_iname)

    kernel = ijoin.map_kernel(kernel)

    if tag is not None:
        kernel = tag_inames(kernel, {new_iname: tag})

    return kernel

# }}}


# {{{ tag inames

def tag_inames(kernel, iname_to_tag, force=False):
    from loopy.kernel.data import parse_tag

    iname_to_tag = dict((iname, parse_tag(tag))
            for iname, tag in iname_to_tag.iteritems())

    from loopy.kernel.data import (ParallelTag, AutoLocalIndexTagBase,
            ForceSequentialTag)

    new_iname_to_tag = kernel.iname_to_tag.copy()
    for iname, new_tag in iname_to_tag.iteritems():
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

class _InameDuplicator(ExpandingIdentityMapper):
    def __init__(self, rules, make_unique_var_name,
            old_to_new, within):
        ExpandingIdentityMapper.__init__(self,
                rules, make_unique_var_name)

        self.old_to_new = old_to_new
        self.old_inames_set = set(old_to_new.iterkeys())
        self.within = within

    def map_reduction(self, expr, expn_state):
        if set(expr.inames) & self.old_inames_set and self.within(expn_state.stack):
            new_inames = tuple(
                    self.old_to_new.get(iname, iname)
                    for iname in expr.inames)

            from loopy.symbolic import Reduction
            return Reduction(expr.operation, new_inames,
                        self.rec(expr.expr, expn_state))
        else:
            return ExpandingIdentityMapper.map_reduction(self, expr, expn_state)

    def map_variable(self, expr, expn_state):
        new_name = self.old_to_new.get(expr.name)

        if new_name is None or not self.within(expn_state.stack):
            return ExpandingIdentityMapper.map_variable(self, expr, expn_state)
        else:
            from pymbolic import var
            return var(new_name)


def duplicate_inames(knl, inames, within, new_inames=None, suffix=None,
        tags={}):
    """
    :arg within: a stack match as understood by
        :func:`loopy.context_matching.parse_stack_match`.
    """

    # {{{ normalize arguments, find unique new_inames

    if isinstance(inames, str):
        inames = inames.split(",")
    if isinstance(new_inames, str):
        new_inames = new_inames.split(",")

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
            name_gen.add_name(new_iname)
            raise ValueError("new iname '%s' conflicts with existing names"
                    % new_iname)

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

    indup = _InameDuplicator(knl.substitutions, name_gen,
            old_to_new=dict(zip(inames, new_inames)),
            within=within)

    knl = indup.map_kernel(knl)

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

    knl = duplicate_inames([old_iname], within, [new_iname])
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
        print proj.gist(first_proj)
        print first_proj.gist(proj)
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
    ijoin = ExpandingSubstitutionMapper(knl.substitutions, var_name_gen,
                    make_subst_func(subst_dict), within)

    knl = ijoin.map_kernel(knl)

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
            raise ValueError("sweep index '%s' has the wrong number of dimensions")

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
        uni_template = uni_template[tuple(var(par_name) for par_name in parameters)]
    elif len(parameters) == 1:
        uni_template = uni_template[var(parameters[0])]

    # }}}

    kernel = extract_subst(kernel, rule_name, uni_template, parameters)

    kernel, subst_use, sweep_inames, inames_to_be_removed = \
            _process_footprint_subscripts(
                    kernel,  rule_name, sweep_inames,
                    footprint_subscripts, arg)
    new_kernel = precompute(kernel, subst_use, sweep_inames,
            new_storage_axis_names=dim_arg_names,
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
        return expand_subst(new_kernel, rule_name)
    else:
        return new_kernel

# }}}


# {{{ instruction processing

def find_instructions(kernel, insn_match):
    from loopy.context_matching import parse_id_match
    match = parse_id_match(insn_match)
    return [insn for insn in kernel.instructions if match(insn.id, None)]


def map_instructions(kernel, insn_match, f):
    from loopy.context_matching import parse_id_match
    match = parse_id_match(insn_match)

    new_insns = []

    for insn in kernel.instructions:
        if match(insn.id, None):
            new_insns.append(f(insn))
        else:
            new_insns.append(insn)

    return kernel.copy(instructions=new_insns)


def set_instruction_priority(kernel, insn_match, priority):
    """Set the priority of instructions matching *insn_match* to *priority*.

    *insn_match* may be any instruction id match understood by
    :func:`loopy.context_matching.parse_id_match`.
    """

    def set_prio(insn):
        return insn.copy(priority=priority)

    return map_instructions(kernel, insn_match, set_prio)


def add_dependency(kernel, insn_match, dependency):
    """Add the instruction dependency *dependency* to the instructions matched
    by *insn_match*.

    *insn_match* may be any instruction id match understood by
    :func:`loopy.context_matching.parse_id_match`.
    """

    def add_dep(insn):
        return insn.copy(insn_deps=insn.insn_deps + [dependency])

    return map_instructions(kernel, insn_match, add_dep)

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

class _ReductionSplitter(ExpandingIdentityMapper):
    def __init__(self, kernel, within, inames, direction):
        ExpandingIdentityMapper.__init__(self,
                kernel.substitutions, kernel.get_var_name_generator())

        self.within = within
        self.inames = inames
        self.direction = direction

    def map_reduction(self, expr, expn_state):
        if self.inames <= set(expr.inames) and self.within(expn_state.stack):
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
            return ExpandingIdentityMapper.map_reduction(self, expr, expn_state)


def _split_reduction(kernel, inames, direction, within=None):
    if direction not in ["in", "out"]:
        raise ValueError("invalid value for 'direction': %s" % direction)

    if isinstance(inames, str):
        inames = inames.split(",")
    inames = set(inames)

    from loopy.context_matching import parse_stack_match
    within = parse_stack_match(within)

    rsplit = _ReductionSplitter(kernel, within, inames, direction)
    return rsplit.map_kernel(kernel)


def split_reduction_inward(kernel, inames, within=None):
    # FIXME document me
    return _split_reduction(kernel, inames, "in", within)


def split_reduction_outward(kernel, inames, within=None):
    # FIXME document me
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

    from loopy.symbolic import SubstitutionMapper
    subst_map = SubstitutionMapper(subst_func)

    from loopy.kernel.array import ArrayBase
    new_args = []
    for arg in kernel.args:
        if arg.name == name:
            # remove from argument list
            continue

        if not isinstance(arg, ArrayBase):
            new_args.append(arg)
        else:
            new_args.append(arg.map_exprs(subst_map))

    new_temp_vars = {}
    for tv in kernel.temporary_variables.itervalues():
        new_temp_vars[tv.name] = tv.map_exprs(subst_map)

    from loopy.context_matching import parse_stack_match
    within = parse_stack_match(None)

    from loopy.symbolic import ExpandingSubstitutionMapper
    esubst_map = ExpandingSubstitutionMapper(
            kernel.substitutions, kernel.get_var_name_generator(),
            subst_func, within=within)
    return (esubst_map.map_kernel(kernel)
            .copy(
                domains=new_domains,
                args=new_args,
                assumptions=process_set(kernel.assumptions),
                ))


def fix_parameters(kernel, **value_dict):
    for name, value in value_dict.iteritems():
        kernel = _fix_parameter(kernel, name, value)

    return kernel

# }}}


# {{{ library registration

def register_preamble_generators(kernel, preamble_generators):
    new_pgens = kernel.preamble_generators[:]
    for pgen in preamble_generators:
        if pgen not in new_pgens:
            new_pgens.append(pgen)

    return kernel.copy(preamble_generators=new_pgens)


def register_symbol_manglers(kernel, manglers):
    new_manglers = kernel.symbol_manglers[:]
    for m in manglers:
        if m not in manglers:
            new_manglers.append(m)

    return kernel.copy(symbol_manglers=new_manglers)


def register_function_manglers(kernel, manglers):
    new_manglers = kernel.function_manglers[:]
    for m in manglers:
        if m not in manglers:
            new_manglers.append(m)

    return kernel.copy(function_manglers=new_manglers)

# }}}


# vim: foldmethod=marker

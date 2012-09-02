from __future__ import division

def register_mpz_with_pymbolic():
    from pymbolic.primitives import register_constant_class
    import gmpy
    mpz_type = type(gmpy.mpz(1))
    register_constant_class(mpz_type)

register_mpz_with_pymbolic()

import islpy as isl
from islpy import dim_type




class LoopyAdvisory(UserWarning):
    pass

# {{{ imported user interface

from loopy.kernel import ValueArg, ScalarArg, GlobalArg, ArrayArg, ConstantArg, ImageArg

from loopy.kernel import (AutoFitLocalIndexTag, get_dot_dependency_graph,
        LoopKernel, Instruction,
        default_function_mangler, single_arg_function_mangler, opencl_function_mangler,
        default_preamble_generator)
from loopy.creation import make_kernel
from loopy.reduction import register_reduction_parser
from loopy.subst import extract_subst, expand_subst
from loopy.cse import precompute
from loopy.preprocess import preprocess_kernel, realize_reduction
from loopy.schedule import generate_loop_schedules
from loopy.codegen import generate_code
from loopy.compiled import CompiledKernel, auto_test_vs_ref
from loopy.check import check_kernels

__all__ = ["ValueArg", "ScalarArg", "GlobalArg", "ArrayArg", "ConstantArg", "ImageArg",
        "LoopKernel",
        "Instruction",
        "default_function_mangler", "single_arg_function_mangler",
        "opencl_function_mangler", "opencl_symbol_mangler",
        "default_preamble_generator",
        "make_kernel",
        "register_reduction_parser",
        "get_dot_dependency_graph",
        "preprocess_kernel", "realize_reduction",
        "generate_loop_schedules",
        "generate_code",
        "CompiledKernel", "auto_test_vs_ref", "check_kernels",
        "make_kernel", "split_dimension", "join_dimensions",
        "tag_dimensions",
        "extract_subst", "expand_subst",
        "precompute", "add_prefetch"
        ]

class infer_type:
    pass

# }}}

# {{{ dimension split

def split_dimension(kernel, split_iname, inner_length,
        outer_iname=None, inner_iname=None,
        outer_tag=None, inner_tag=None,
        slabs=(0, 0), do_tagged_check=True):

    existing_tag = kernel.iname_to_tag.get(split_iname)
    from loopy.kernel import ForceSequentialTag
    if do_tagged_check and (
            existing_tag is not None
            and not isinstance(existing_tag, ForceSequentialTag)):
        raise RuntimeError("cannot split already tagged iname '%s'" % split_iname)

    if split_iname not in kernel.all_inames():
        raise ValueError("cannot split loop for unknown variable '%s'" % split_iname)

    applied_iname_rewrites = kernel.applied_iname_rewrites[:]

    if outer_iname is None:
        outer_iname = split_iname+"_outer"
    if inner_iname is None:
        inner_iname = split_iname+"_inner"

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
                    space, {split_iname:1, inner_iname: -1, outer_iname:-inner_length})))

        name_dim_type, name_idx = space.get_var_dict()[split_iname]
        return (s
                .intersect(inner_constraint_set)
                .eliminate(name_dim_type, name_idx, 1)
                .remove_dims(name_dim_type, name_idx, 1))

    new_domains = [process_set(dom) for dom in kernel.domains]

    from pymbolic import var
    inner = var(inner_iname)
    outer = var(outer_iname)
    new_loop_index = inner + outer*inner_length

    # {{{ actually modify instructions

    from loopy.symbolic import ReductionLoopSplitter

    rls = ReductionLoopSplitter(split_iname, outer_iname, inner_iname)
    new_insns = []
    for insn in kernel.instructions:
        subst_map = {var(split_iname): new_loop_index}
        applied_iname_rewrites.append(subst_map)

        from loopy.symbolic import SubstitutionMapper
        subst_mapper = SubstitutionMapper(subst_map.get)

        new_expr = subst_mapper(rls(insn.expression))

        if split_iname in insn.forced_iname_deps:
            new_forced_iname_deps = (
                    (insn.forced_iname_deps.copy()
                    - frozenset([split_iname]))
                    | frozenset([outer_iname, inner_iname]))
        else:
            new_forced_iname_deps = insn.forced_iname_deps

        insn = insn.copy(
                assignee=subst_mapper(insn.assignee),
                expression=new_expr,
                forced_iname_deps=new_forced_iname_deps)

        new_insns.append(insn)

    # }}}

    iname_slab_increments = kernel.iname_slab_increments.copy()
    iname_slab_increments[outer_iname] = slabs
    result = (kernel
            .map_expressions(subst_mapper, exclude_instructions=True)
            .copy(domains=new_domains,
                iname_slab_increments=iname_slab_increments,
                instructions=new_insns,
                applied_iname_rewrites=applied_iname_rewrites,
                ))

    if existing_tag is not None:
        result = tag_dimensions(result,
                {outer_iname: existing_tag, inner_iname: existing_tag})

    return tag_dimensions(result, {outer_iname: outer_tag, inner_iname: inner_tag})

# }}}

# {{{ dimension join

def join_dimensions(kernel, inames, new_iname=None, tag=AutoFitLocalIndexTag()):
    """
    :arg inames: fastest varying last
    """

    # now fastest varying first
    inames = inames[::-1]

    if new_iname is None:
        new_iname = kernel.make_unique_var_name("_and_".join(inames))

    new_domain = kernel.domain
    new_dim_idx = new_domain.dim(dim_type.set)
    new_domain = new_domain.add_dims(dim_type.set, 1)
    new_domain = new_domain.set_dim_name(dim_type.set, new_dim_idx, new_iname)

    joint_aff = zero = isl.Aff.zero_on_domain(kernel.space)
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
        lower_bound_aff = static_value_of_pw_aff(
                bounds.lower_bound_pw_aff.coalesce(),
                constants_only=False)

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
        new_domain = new_domain.eliminate(iname_dt, iname_idx, 1)
        new_domain = new_domain.remove_dims(iname_dt, iname_idx, 1)

    from loopy.symbolic import SubstitutionMapper
    from pymbolic.mapper.substitutor import make_subst_func
    subst_map = SubstitutionMapper(make_subst_func(subst_dict))

    def subst_forced_iname_deps(fid):
        result = set()
        for iname in fid:
            if iname in inames:
                result.add(new_iname)
            else:
                result.add(iname)

        return result

    new_insns = [
            insn.copy(
                assignee=subst_map(insn.assignee),
                expression=subst_map(insn.expression),
                forced_iname_deps=subst_forced_iname_deps(insn.forced_iname_deps))
            for insn in kernel.instructions]

    result = (kernel
            .map_expressions(subst_map, exclude_instructions=True)
            .copy(
                instructions=new_insns, domain=new_domain,
                applied_iname_rewrites=kernel.applied_iname_rewrites + [subst_map]
                ))

    return tag_dimensions(result, {new_iname: tag})

# }}}

# {{{ dimension tag

def tag_dimensions(kernel, iname_to_tag, force=False):
    from loopy.kernel import parse_tag

    iname_to_tag = dict((iname, parse_tag(tag))
            for iname, tag in iname_to_tag.iteritems())

    from loopy.kernel import (ParallelTag, AutoLocalIndexTagBase,
            ForceSequentialTag)

    new_iname_to_tag = kernel.iname_to_tag.copy()
    for iname, new_tag in iname_to_tag.iteritems():
        if iname not in kernel.all_inames():
            raise RuntimeError("iname '%s' does not exist" % iname)

        old_tag = kernel.iname_to_tag.get(iname)

        retag_ok = False

        if isinstance(old_tag, (AutoLocalIndexTagBase, ForceSequentialTag)):
            retag_ok = True

        if not retag_ok and old_tag is not None and new_tag is None:
            raise ValueError("cannot untag iname '%s'" % iname)

        if iname not in kernel.all_inames():
            raise ValueError("cannot tag '%s'--not known" % iname)

        if isinstance(new_tag, ParallelTag) and isinstance(old_tag, ForceSequentialTag):
            raise ValueError("cannot tag '%s' as parallel--"
                    "iname requires sequential execution" % iname)

        if isinstance(new_tag, ForceSequentialTag) and isinstance(old_tag, ParallelTag):
            raise ValueError("'%s' is already tagged as parallel, "
                    "but is now prohibited from being parallel "
                    "(likely because of participation in a precompute or "
                    "a reduction)" % iname)

        if (not retag_ok) and (not force) and old_tag is not None and (old_tag != new_tag):
            raise RuntimeError("'%s' is already tagged '%s'--cannot retag"
                    % (iname, old_tag))

        new_iname_to_tag[iname] = new_tag

    return kernel.copy(iname_to_tag=new_iname_to_tag)

# }}}

# {{{ convenience: add_prefetch

# {{{ process footprint_subscripts

def _add_kernel_axis(kernel, axis_name, start, stop, base_inames):
    from loopy.kernel import DomainChanger
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
        footprint_subscripts, arg, newly_created_vars):
    """Track applied iname rewrites, deal with slice specifiers ':'."""

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

        if len(fsub) != arg.dimensions:
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

                axis_name = kernel.make_unique_var_name(
                        based_on="%s_fetch_axis_%d" % (arg.name, axis_nr),
                        extra_used_vars=newly_created_vars)

                newly_created_vars.add(axis_name)
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
        default_tag="l.auto", rule_name=None, footprint_subscripts=None):
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

    if rule_name is None:
        rule_name = kernel.make_unique_var_name("%s_fetch" % c_name)

    newly_created_vars = set([rule_name])

    arg = kernel.arg_dict[var_name]

    # {{{ make parameter names and unification template

    parameters = []
    for i in range(arg.dimensions):
        based_on = "%s_dim_%d" % (c_name, i)
        if dim_arg_names is not None and i < len(dim_arg_names):
            based_on = dim_arg_names[i]

        par_name = kernel.make_unique_var_name(based_on=based_on,
                extra_used_vars=newly_created_vars)
        newly_created_vars.add(par_name)
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
                    footprint_subscripts, arg, newly_created_vars)

    new_kernel = precompute(kernel, subst_use, arg.dtype, sweep_inames,
            new_storage_axis_names=dim_arg_names,
            default_tag=default_tag)

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

class _IdMatch(object):
    def __init__(self, value):
        self.value = value

class _ExactIdMatch(_IdMatch):
    def __call__(self, insn):
        return insn.id == self.value

class _ReIdMatch:
    def __call__(self, insn):
        return self.value.match(insn.id) is not None

def _parse_insn_match(insn_match):
    import re
    colon_idx = insn_match.find(":")
    if colon_idx == -1:
        return _ExactIdMatch(insn_match)

    match_tp = insn_match[:colon_idx]
    match_val = insn_match[colon_idx+1:]

    if match_tp == "glob":
        from fnmatch import translate
        return _ReIdMatch(re.compile(translate(match_val)))
    elif match_tp == "re":
        return _ReIdMatch(re.compile(match_val))
    else:
        raise ValueError("match type '%s' not understood" % match_tp)




def find_instructions(kernel, insn_match):
    match = _parse_insn_match(insn_match)
    return [insn for insn in kernel.instructions if match(insn)]

def map_instructions(kernel, insn_match, f):
    match = _parse_insn_match(insn_match)

    new_insns = []

    for insn in kernel.instructions:
        if match(insn):
            new_insns.append(f(insn))
        else:
            new_insns.append(insn)

    return kernel.copy(instructions=new_insns)

def set_instruction_priority(kernel, insn_match, priority):
    """Set the priority of instructions matching *insn_match* to *priority*.

    *insn_match* may be an instruction id, a regular expression prefixed by `re:`,
    or a file-name-style glob prefixed by `glob:`.
    """

    def set_prio(insn): return insn.copy(priority=priority)
    return map_instructions(kernel, insn_match, set_prio)

def add_dependency(kernel, insn_match, dependency):
    """Add the instruction dependency *dependency* to the instructions matched
    by *insn_match*.

    *insn_match* may be an instruction id, a regular expression prefixed by `re:`,
    or a file-name-style glob prefixed by `glob:`.
    """

    def add_dep(insn): return insn.copy(insn_deps=insn.insn_deps + [dependency])
    return map_instructions(kernel, insn_match, add_dep)

# }}}




# vim: foldmethod=marker

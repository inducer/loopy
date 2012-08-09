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

from loopy.kernel import ScalarArg, GlobalArg, ArrayArg, ConstantArg, ImageArg

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
from loopy.compiled import CompiledKernel, drive_timing_run, auto_test_vs_ref
from loopy.check import check_kernels

__all__ = ["ScalarArg", "GlobalArg", "ArrayArg", "ConstantArg", "ImageArg",
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
        "CompiledKernel", "drive_timing_run", "auto_test_vs_ref", "check_kernels",
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

    if do_tagged_check and kernel.iname_to_tag.get(split_iname) is not None:
        raise RuntimeError("cannot split already tagged iname '%s'" % split_iname)

    if split_iname not in kernel.all_inames():
        raise ValueError("cannot split loop for unknown variable '%s'" % split_iname)

    applied_iname_rewrites = kernel.applied_iname_rewrites[:]

    if outer_iname is None:
        outer_iname = split_iname+"_outer"
    if inner_iname is None:
        inner_iname = split_iname+"_inner"

    def process_set(s):
        outer_var_nr = s.dim(dim_type.set)
        inner_var_nr = s.dim(dim_type.set)+1

        s = s.add_dims(dim_type.set, 2)
        s = s.set_dim_name(dim_type.set, outer_var_nr, outer_iname)
        s = s.set_dim_name(dim_type.set, inner_var_nr, inner_iname)

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
            new_forced_iname_deps = insn.forced_iname_deps.copy()
            new_forced_iname_deps.remove(split_iname)
            new_forced_iname_deps.update([outer_iname, inner_iname])
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

    from loopy.kernel import (ParallelTag, AutoLocalIndexTagBase)

    new_iname_to_tag = kernel.iname_to_tag.copy()
    for iname, new_tag in iname_to_tag.iteritems():
        old_tag = kernel.iname_to_tag.get(iname)

        retag_ok = False

        if isinstance(old_tag, AutoLocalIndexTagBase):
            retag_ok = True

        if not retag_ok and old_tag is not None and new_tag is None:
            raise ValueError("cannot untag iname '%s'" % iname)

        if iname not in kernel.all_inames():
            raise ValueError("cannot tag '%s'--not known" % iname)

        if isinstance(new_tag, ParallelTag) and iname in kernel.sequential_inames:
            raise ValueError("cannot tag '%s' as parallel--"
                    "iname requires sequential execution" % iname)

        if (not retag_ok) and (not force) and old_tag is not None and (old_tag != new_tag):
            raise RuntimeError("'%s' is already tagged '%s'--cannot retag"
                    % (iname, old_tag))

        new_iname_to_tag[iname] = new_tag

    return kernel.copy(iname_to_tag=new_iname_to_tag)

# }}}

# {{{ convenience: add_prefetch

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

    kernel = extract_subst(kernel, rule_name, uni_template, parameters)

    if footprint_subscripts is not None:
        if not isinstance(footprint_subscripts, (list, tuple)):
            footprint_subscripts = [footprint_subscripts]

        def standardize_footprint_indices(si):
            if isinstance(si, str):
                from loopy.symbolic import parse
                si = parse(si)

            if not isinstance(si, tuple):
                si = (si,)

            if len(si) != arg.dimensions:
                raise ValueError("sweep index '%s' has the wrong number of dimensions")

            for subst_map in kernel.applied_iname_rewrites:
                from loopy.symbolic import SubstitutionMapper
                from pymbolic.mapper.substitutor import make_subst_func
                si = SubstitutionMapper(make_subst_func(subst_map))(si)

            return si

        footprint_subscripts = [standardize_footprint_indices(si) for si in footprint_subscripts]

        from pymbolic.primitives import Variable
        subst_use = [
                Variable(rule_name)(*si) for si in footprint_subscripts]
    else:
        subst_use = rule_name

    new_kernel = precompute(kernel, subst_use, arg.dtype, sweep_inames,
            new_storage_axis_names=dim_arg_names,
            default_tag=default_tag)

    # If the rule survived past precompute() (i.e. some accesses fell outside
    # the footprint), get rid of it before moving on.
    if rule_name in new_kernel.substitutions:
        return expand_subst(new_kernel, rule_name)
    else:
        return new_kernel


# }}}




# vim: foldmethod=marker

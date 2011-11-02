from __future__ import division

def register_mpz_with_pymbolic():
    from pymbolic.primitives import register_constant_class
    import gmpy
    mpz_type = type(gmpy.mpz(1))
    register_constant_class(mpz_type)

register_mpz_with_pymbolic()

import islpy as isl
from islpy import dim_type
import numpy as np




class LoopyAdvisory(UserWarning):
    pass

# {{{ imported user interface

from loopy.kernel import ScalarArg, ArrayArg, ImageArg

from loopy.kernel import AutoFitLocalIndexTag
from loopy.cse import realize_cse
from loopy.preprocess import preprocess_kernel
from loopy.schedule import generate_loop_schedules
from loopy.codegen import generate_code
from loopy.compiled import CompiledKernel, drive_timing_run, auto_test_vs_seq
from loopy.check import check_kernels

__all__ = ["ScalarArg", "ArrayArg", "ImageArg",
        "preprocess_kernel", "generate_loop_schedules",
        "generate_code",
        "CompiledKernel", "drive_timing_run", "check_kernels",
        "make_kernel", "split_dimension", "join_dimensions",
        "tag_dimensions", "realize_cse", "add_prefetch"
        ]

# }}}

# {{{ kernel creation

def make_kernel(*args, **kwargs):
    """Second pass of kernel creation. Think about requests for iname duplication
    and temporary variable declaration received as part of string instructions.
    """

    from loopy.kernel import LoopKernel
    knl = LoopKernel(*args, **kwargs)

    knl = tag_dimensions(
            knl.copy(iname_to_tag_requests=None),
            knl.iname_to_tag_requests)

    new_insns = []
    new_domain = knl.domain
    new_temp_vars = knl.temporary_variables.copy()
    new_iname_to_tag = knl.iname_to_tag.copy()

    newly_created_vars = set()

    from loopy.symbolic import CSESubstitutor
    cse_sub = CSESubstitutor(knl.cses)

    for insn in knl.instructions:
        insn = insn.copy(expression=cse_sub(insn.expression))

        # {{{ sanity checking

        if not set(insn.forced_iname_deps) <= knl.all_inames():
            raise ValueError("In instruction '%s': "
                    "cannot force dependency on inames '%s'--"
                    "they don't exist" % (
                        insn.id,
                        ",".join(
                            set(insn.forced_iname_deps)-knl.all_inames())))

        # }}}

        # {{{ iname duplication

        if insn.duplicate_inames_and_tags:
            insn_dup_iname_to_tag = dict(insn.duplicate_inames_and_tags)

            if not set(insn_dup_iname_to_tag.keys()) <= knl.all_inames():
                raise ValueError("In instruction '%s': "
                        "cannot duplicate inames '%s'--"
                        "they don't exist" % (
                            insn.id,
                            ",".join(
                                set(insn_dup_iname_to_tag.keys())-knl.all_inames())))

            # {{{ duplicate non-reduction inames

            reduction_inames = insn.reduction_inames()

            inames_to_duplicate = [iname
                    for iname, tag in insn.duplicate_inames_and_tags
                    if iname not in reduction_inames]

            new_inames = [
                    knl.make_unique_var_name(
                        based_on=iname+"_"+insn.id,
                        extra_used_vars=
                        newly_created_vars)
                    for iname in inames_to_duplicate]

            for old_iname, new_iname in zip(inames_to_duplicate, new_inames):
                new_tag = insn_dup_iname_to_tag[old_iname]
                if new_tag is None:
                    new_tag = AutoFitLocalIndexTag()
                new_iname_to_tag[new_iname] = new_tag

            newly_created_vars.update(new_inames)

            from loopy.isl_helpers import duplicate_axes
            new_domain = duplicate_axes(new_domain, inames_to_duplicate, new_inames)

            from loopy.symbolic import SubstitutionMapper
            from pymbolic.mapper.substitutor import make_subst_func
            from pymbolic import var
            old_to_new = dict(
                    (old_iname, var(new_iname))
                    for old_iname, new_iname in zip(inames_to_duplicate, new_inames))
            subst_map = SubstitutionMapper(make_subst_func(old_to_new))
            new_expression = subst_map(insn.expression)

            # }}}

            if len(inames_to_duplicate) < len(insn.duplicate_inames_and_tags):
                raise RuntimeError("cannot use [|...] syntax to rename reduction "
                        "inames")

            insn = insn.copy(
                    assignee=subst_map(insn.assignee),
                    expression=new_expression,
                    forced_iname_deps=set(
                        old_to_new.get(iname, iname) for iname in insn.forced_iname_deps),
                    duplicate_inames_and_tags=[])

        # }}}

        # {{{ temporary variable creation

        from loopy.kernel import (
                find_var_base_indices_and_shape_from_inames,
                TemporaryVariable)

        if insn.temp_var_type is not None:
            assignee_name = insn.get_assignee_var_name()

            assignee_indices = []
            from pymbolic.primitives import Variable
            for index_expr in insn.get_assignee_indices():
                if (not isinstance(index_expr, Variable)
                        or not index_expr.name in insn.all_inames()):
                    raise RuntimeError(
                            "only plain inames are allowed in "
                            "the lvalue index when declaring the "
                            "variable '%s' in an instruction"
                            % assignee_name)

                assignee_indices.append(index_expr.name)

            base_indices, shape = \
                    find_var_base_indices_and_shape_from_inames(
                            new_domain, assignee_indices)

            new_temp_vars[assignee_name] = TemporaryVariable(
                    name=assignee_name,
                    dtype=np.dtype(insn.temp_var_type),
                    is_local=None,
                    base_indices=base_indices,
                    shape=shape)

            newly_created_vars.add(assignee_name)

            insn = insn.copy(temp_var_type=None)

        # }}}

        new_insns.append(insn)

    return knl.copy(
            instructions=new_insns,
            domain=new_domain,
            temporary_variables=new_temp_vars,
            iname_to_tag=new_iname_to_tag,
            iname_to_tag_requests=[],
            cses={})

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

    if outer_iname is None:
        outer_iname = split_iname+"_outer"
    if inner_iname is None:
        inner_iname = split_iname+"_inner"

    outer_var_nr = kernel.space.dim(dim_type.set)
    inner_var_nr = kernel.space.dim(dim_type.set)+1

    def process_set(s):
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

    new_domain = process_set(kernel.domain)

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
            .copy(domain=new_domain,
                iname_slab_increments=iname_slab_increments,
                instructions=new_insns,
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

    result = kernel.copy(
            instructions=new_insns,
            domain=new_domain)

    return tag_dimensions(result, {new_iname: tag})

# }}}

# {{{ dimension tag

def tag_dimensions(kernel, iname_to_tag, force=False):
    from loopy.kernel import parse_tag

    iname_to_tag = dict((iname, parse_tag(tag))
            for iname, tag in iname_to_tag.iteritems())

    from loopy.kernel import ParallelTag

    new_iname_to_tag = kernel.iname_to_tag.copy()
    for iname, new_tag in iname_to_tag.iteritems():
        old_tag = kernel.iname_to_tag.get(iname)

        if old_tag is not None and new_tag is None:
            raise ValueError("cannot untag iname '%s'" % iname)

        if new_tag is None:
            continue

        if iname not in kernel.all_inames():
            raise ValueError("cannot tag '%s'--not known" % iname)

        if isinstance(new_tag, ParallelTag) and iname in kernel.sequential_inames:
            raise ValueError("cannot tag '%s' as parallel--"
                    "iname requires sequential execution" % iname)

        if (not force) and old_tag is not None and (old_tag != new_tag):
            raise RuntimeError("'%s' is already tagged '%s'--cannot retag"
                    % (iname, old_tag))

        new_iname_to_tag[iname] = new_tag

    return kernel.copy(iname_to_tag=new_iname_to_tag)

# }}}

# {{{ convenience: add_prefetch

def add_prefetch(kernel, var_name, fetch_dims=[], new_inames=None, default_tag="l.auto"):
    used_cse_tags = set()
    def map_cse(expr, rec):
        used_cse_tags.add(expr.tag)
        rec(expr.child)

    new_cse_tags = set()

    def get_unique_cse_tag():
        from loopy.tools import generate_unique_possibilities
        for cse_tag in generate_unique_possibilities(prefix="fetch_"+var_name):
            if cse_tag not in used_cse_tags:
                used_cse_tags.add(cse_tag)
                new_cse_tags.add(cse_tag)
                return cse_tag

    from loopy.symbolic import VariableFetchCSEMapper
    vf_cse_mapper = VariableFetchCSEMapper(var_name, get_unique_cse_tag)
    kernel = kernel.copy(instructions=[
            insn.copy(expression=vf_cse_mapper(insn.expression))
            for insn in kernel.instructions])

    if var_name in kernel.arg_dict:
        dtype = kernel.arg_dict[var_name].dtype
    else:
        dtype = kernel.temporary_variables[var_name].dtype

    for cse_tag in new_cse_tags:
        kernel = realize_cse(kernel, cse_tag, dtype, fetch_dims,
                new_inames=new_inames, default_tag=default_tag)

    return kernel

# }}}





# vim: foldmethod=marker

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
from loopy.preprocess import preprocess_kernel
from loopy.schedule import generate_loop_schedules
from loopy.compiled import CompiledKernel, drive_timing_run
from loopy.check import check_kernels

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

    # {{{ reduction iname duplication helper function

    def duplicate_reduction_inames(reduction_expr, rec):
        duplicate_inames = [iname
                for iname, tag in insn.duplicate_inames_and_tags]

        child = rec(reduction_expr.expr)
        new_red_inames = []
        did_something = False

        for iname in reduction_expr.inames:
            if iname in duplicate_inames:
                new_iname = knl.make_unique_var_name(iname, newly_created_vars)

                old_insn_inames.append(iname)
                new_insn_inames.append(new_iname)
                newly_created_vars.add(new_iname)
                new_red_inames.append(new_iname)
                did_something = True
            else:
                new_red_inames.append(iname)

        if did_something:
            from loopy.symbolic import SubstitutionMapper
            from pymbolic.mapper.substitutor import make_subst_func
            from pymbolic import var
            subst_dict = dict(
                    (old_iname, var(new_iname))
                    for old_iname, new_iname in zip(
                        reduction_expr.inames, new_red_inames))
            subst_map = SubstitutionMapper(make_subst_func(subst_dict))

            child = subst_map(child)

            for old_iname, new_iname in zip(reduction_expr.inames, new_red_inames):
                new_iname_to_tag[new_iname] = insn_dup_iname_to_tag[old_iname]

        from loopy.symbolic import Reduction
        return Reduction(
                operation=reduction_expr.operation,
                inames=tuple(new_red_inames),
                expr=child)

    # }}}

    for insn in knl.instructions:
        # {{{ iname duplication

        if insn.duplicate_inames_and_tags:

            insn_dup_iname_to_tag = dict(insn.duplicate_inames_and_tags)

            # {{{ duplicate non-reduction inames

            reduction_inames = insn.reduction_inames()

            duplicate_inames = [iname
                    for iname, tag in insn.duplicate_inames_and_tags
                    if iname not in reduction_inames]

            new_inames = [
                    knl.make_unique_var_name(
                        iname,
                        extra_used_vars=
                        newly_created_vars)
                    for iname in duplicate_inames]

            for old_iname, new_iname in zip(duplicate_inames, new_inames):
                new_tag = insn_dup_iname_to_tag[old_iname]
                if new_tag is None:
                    new_tag = AutoFitLocalIndexTag()
                new_iname_to_tag[new_iname] = new_tag

            newly_created_vars.update(new_inames)

            from loopy.isl_helpers import duplicate_axes
            new_domain = duplicate_axes(new_domain, duplicate_inames, new_inames)

            from loopy.symbolic import SubstitutionMapper
            from pymbolic.mapper.substitutor import make_subst_func
            from pymbolic import var
            old_to_new = dict(
                    (old_iname, var(new_iname))
                    for old_iname, new_iname in zip(duplicate_inames, new_inames))
            subst_map = SubstitutionMapper(make_subst_func(old_to_new))
            new_expression = subst_map(insn.expression)

            # }}}

            # {{{ duplicate reduction inames

            if len(duplicate_inames) < len(insn.duplicate_inames_and_tags):
                # there must've been requests to duplicate reduction inames
                old_insn_inames = []
                new_insn_inames = []

                from loopy.symbolic import ReductionCallbackMapper
                new_expression = (
                        ReductionCallbackMapper(duplicate_reduction_inames)
                        (new_expression))

                from loopy.isl_helpers import duplicate_axes
                for old, new in zip(old_insn_inames, new_insn_inames):
                    new_domain = duplicate_axes(new_domain, [old], [new])

            # }}}

            insn = insn.copy(
                    assignee=subst_map(insn.assignee),
                    expression=new_expression,
                    forced_iname_deps=set(
                        old_to_new.get(iname, iname) for iname in insn.forced_iname_deps),
                    )

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

            from loopy.kernel import LocalIndexTagBase
            from pytools import any
            is_local = any(
                    isinstance(new_iname_to_tag.get(iname), LocalIndexTagBase)
                    for iname in assignee_indices)

            base_indices, shape = \
                    find_var_base_indices_and_shape_from_inames(
                            new_domain, assignee_indices)

            new_temp_vars[assignee_name] = TemporaryVariable(
                    name=assignee_name,
                    dtype=np.dtype(insn.temp_var_type),
                    is_local=is_local,
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
            iname_to_tag=new_iname_to_tag)

# }}}

# {{{ user-facing kernel manipulation functionality

def split_dimension(kernel, iname, inner_length,
        outer_iname=None, inner_iname=None,
        outer_tag=None, inner_tag=None,
        slabs=(0, 0)):

    if iname not in kernel.all_inames():
        raise ValueError("cannot split loop for unknown variable '%s'" % iname)

    if outer_iname is None:
        outer_iname = iname+"_outer"
    if inner_iname is None:
        inner_iname = iname+"_inner"

    outer_var_nr = kernel.space.dim(dim_type.set)
    inner_var_nr = kernel.space.dim(dim_type.set)+1

    def process_set(s):
        s = s.add_dims(dim_type.set, 2)
        s.set_dim_name(dim_type.set, outer_var_nr, outer_iname)
        s.set_dim_name(dim_type.set, inner_var_nr, inner_iname)

        from loopy.isl_helpers import make_slab

        space = s.get_space()
        inner_constraint_set = (
                make_slab(space, inner_iname, 0, inner_length)
                # name = inner + length*outer
                .add_constraint(isl.Constraint.eq_from_names(
                    space, {iname:1, inner_iname: -1, outer_iname:-inner_length})))

        name_dim_type, name_idx = space.get_var_dict()[iname]
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

    rls = ReductionLoopSplitter(iname, outer_iname, inner_iname)
    new_insns = []
    for insn in kernel.instructions:
        subst_map = {var(iname): new_loop_index}

        from loopy.symbolic import SubstitutionMapper
        subst_mapper = SubstitutionMapper(subst_map.get)

        new_expr = subst_mapper(rls(insn.expression))

        if iname in insn.forced_iname_deps:
            new_forced_iname_deps = insn.forced_iname_deps.copy()
            new_forced_iname_deps.remove(iname)
            new_forced_iname_deps.update([outer_iname, inner_iname])
        else:
            new_forced_iname_deps = insn.forced_iname_deps

        insn = insn.copy(
                assignee=subst_mapper(insn.assignee),
                expression=new_expr,
                forced_iname_deps=new_forced_iname_deps
                )

        new_insns.append(insn)

    # }}}

    iname_slab_increments = kernel.iname_slab_increments.copy()
    iname_slab_increments[outer_iname] = slabs
    result = (kernel
            .copy(domain=new_domain,
                iname_slab_increments=iname_slab_increments,
                instructions=new_insns))

    return tag_dimensions(result, {outer_iname: outer_tag, inner_iname: inner_tag})




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




def tag_dimensions(kernel, iname_to_tag):
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

        if old_tag is not None and (old_tag != new_tag):
            raise RuntimeError("'%s' is already tagged '%s'--cannot retag"
                    % (iname, old_tag))

        new_iname_to_tag[iname] = new_tag

    return kernel.copy(iname_to_tag=new_iname_to_tag)





def realize_cse(kernel, cse_tag, dtype, duplicate_inames=[], parallel_inames=None,
        dup_iname_to_tag={}, new_inames=None, default_tag_class=AutoFitLocalIndexTag):
    """
    :arg duplicate_inames: which inames are supposed to be separate loops
        in the CSE. Also determines index order of temporary array.
    :arg parallel_inames: only a convenient interface for dup_iname_to_tag
    """

    dtype = np.dtype(dtype)

    from pytools import any

    # {{{ process parallel_inames and dup_iname_to_tag arguments

    if parallel_inames is None:
        # default to all-parallel
        parallel_inames = duplicate_inames

    dup_iname_to_tag = dup_iname_to_tag.copy()
    for piname in parallel_inames:
        dup_iname_to_tag[piname] = default_tag_class()

    for diname in duplicate_inames:
        dup_iname_to_tag.setdefault(diname, None)

    if not set(dup_iname_to_tag.iterkeys()) <= set(duplicate_inames):
        raise RuntimeError("paralleization/tag info for non-duplicated inames "
                "may not be passed")

    # here, all information is consolidated into dup_iname_to_tag

    # }}}

    # {{{ process new_inames argument, think of new inames for inames to be duplicated

    if new_inames is None:
        new_inames = [None] * len(duplicate_inames)

    if len(new_inames) != len(duplicate_inames):
        raise ValueError("If given, the new_inames argument must have the "
                "same length as duplicate_inames")

    temp_new_inames = []
    for old_iname, new_iname in zip(duplicate_inames, new_inames):
        if new_iname is None:
            new_iname = kernel.make_unique_var_name(old_iname)
        temp_new_inames.append(new_iname)

    new_inames = temp_new_inames

    old_to_new_iname = dict(zip(duplicate_inames, new_inames))

    # }}}

    target_var_name = kernel.make_unique_var_name(cse_tag)

    from loopy.kernel import (LocalIndexTagBase, GroupIndexTag, IlpTag)
    target_var_is_local = any(
            isinstance(tag, LocalIndexTagBase)
            for tag in dup_iname_to_tag.itervalues())

    cse_lookup_table = {}

    cse_result_insns = []

    def map_cse(expr, rec):
        if expr.prefix != cse_tag:
            return

        # FIXME stencils and variable shuffle detection would happen here

        try:
            cse_replacement, dep_id = cse_lookup_table[expr]
        except KeyError:
            pass
        else:
            return cse_replacement

        if cse_result_insns:
            raise RuntimeError("CSE tag '%s' is not unique" % cse_tag)

        # {{{ decide what to do with each iname

        forced_iname_deps = set()

        from loopy.symbolic import IndexVariableFinder
        dependencies = IndexVariableFinder(
                include_reduction_inames=False)(expr.child)

        parent_inames = insn.all_inames() | insn.reduction_inames()
        assert dependencies <= parent_inames

        for iname in parent_inames:
            if iname in duplicate_inames:
                tag = dup_iname_to_tag[iname]
            else:
                tag = kernel.iname_to_tag.get(iname)

            if isinstance(tag, LocalIndexTagBase):
                kind = "l"
            elif isinstance(tag, GroupIndexTag):
                kind = "g"
            elif isinstance(tag, IlpTag):
                kind = "i"
            else:
                kind = "o"

            if iname not in duplicate_inames and iname in dependencies:
                if (
                        (target_var_is_local and kind in "li")
                        or
                        (not target_var_is_local and kind in "i")):
                    raise RuntimeError(
                            "When realizing CSE with tag '%s', encountered iname "
                            "'%s' which is depended upon by the CSE and tagged "
                            "'%s', but not duplicated. The CSE would "
                            "inherit this iname, which would lead to a write race. "
                            "A likely solution of this problem is to also duplicate this "
                            "iname."
                            % (expr.prefix, iname, tag))

            if iname in duplicate_inames and kind == "g":
                raise RuntimeError("duplicating the iname '%s' into "
                        "group index axes is not helpful, as they cannot "
                        "collaborate in computing a local variable"
                        %iname)

            if iname in dependencies:
                if not target_var_is_local and iname in duplicate_inames and kind == "l":
                    raise RuntimeError("invalid: hardware-parallelized "
                            "fetch into private variable")

                # otherwise: all happy
                continue

            # the iname is *not* a dependency of the fetch expression
            if iname in duplicate_inames:
                raise RuntimeError("duplicating an iname ('%s') "
                        "that the CSE ('%s') does not depend on "
                        "does not make sense" % (iname, expr.child))

            # Which iname dependencies are carried over from CSE host
            # to the CSE compute instruction?

            if not target_var_is_local:
                # If we're writing to a private variable, then each
                # hardware-parallel iname must execute its own copy of
                # the CSE compute instruction. After all, each work item
                # has its own set of private variables.

                force_dependency = kind in "gl"
            else:
                # If we're writing to a local variable, then all other local
                # dimensions see our updates, and thus they do *not* need to
                # execute their own copy of this instruction.

                force_dependency = kind == "g"

            if force_dependency:
                forced_iname_deps.add(iname)

        # }}}

        # {{{ concoct new inner and outer expressions

        from pymbolic import var
        assignee = var(target_var_name)
        new_outer_expr = assignee

        if duplicate_inames:
            assignee = assignee[tuple(
                var(iname) for iname in new_inames
                )]
            new_outer_expr = new_outer_expr[tuple(
                var(iname) for iname in duplicate_inames
                )]

        from loopy.symbolic import SubstitutionMapper
        from pymbolic.mapper.substitutor import make_subst_func
        subst_map = SubstitutionMapper(make_subst_func(
            dict(
                (old_iname, var(new_iname))
                for old_iname, new_iname in zip(duplicate_inames, new_inames))))
        new_inner_expr = subst_map(rec(expr.child))

        # }}}

        from loopy.kernel import Instruction
        new_insn = Instruction(
                id=kernel.make_unique_instruction_id(based_on=cse_tag),
                assignee=assignee,
                expression=new_inner_expr,
                forced_iname_deps=forced_iname_deps)

        cse_result_insns.append(new_insn)

        return new_outer_expr

    from loopy.symbolic import CSECallbackMapper
    cse_cb_mapper = CSECallbackMapper(map_cse)

    new_insns = []
    for insn in kernel.instructions:
        was_empty = not bool(cse_result_insns)
        new_expr = cse_cb_mapper(insn.expression)

        if was_empty and cse_result_insns:
            new_insns.append(insn.copy(expression=new_expr))
        else:
            new_insns.append(insn)

    new_insns.extend(cse_result_insns)

    # {{{ build new domain, duplicating each constraint on duplicated inames

    from loopy.isl_helpers import duplicate_axes
    new_domain = duplicate_axes(kernel.domain, duplicate_inames, new_inames)

    # }}}

    # {{{ set up data for temp variable


    from loopy.kernel import (TemporaryVariable,
            find_var_base_indices_and_shape_from_inames)

    target_var_base_indices, target_var_shape = \
            find_var_base_indices_and_shape_from_inames(
                    new_domain, new_inames)

    new_temporary_variables = kernel.temporary_variables.copy()
    new_temporary_variables[target_var_name] = TemporaryVariable(
            name=target_var_name,
            dtype=dtype,
            base_indices=target_var_base_indices,
            shape=target_var_shape,
            is_local=target_var_is_local)

    # }}}

    new_iname_to_tag = kernel.iname_to_tag.copy()
    for old_iname, new_iname in zip(duplicate_inames, new_inames):
        new_iname_to_tag[new_iname] = dup_iname_to_tag[old_iname]

    return kernel.copy(
            domain=new_domain,
            instructions=new_insns,
            temporary_variables=new_temporary_variables,
            iname_to_tag=new_iname_to_tag)





# {{{ convenience

def add_prefetch(kernel, var_name, fetch_dims=[], new_inames=None):
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
                new_inames=new_inames)

    return kernel

# }}}





# vim: foldmethod=marker

from __future__ import division
import isl

def register_mpz_with_pymbolic():
    from pymbolic.primitives import register_constant_class
    import gmpy
    mpz_type = type(gmpy.mpz(1))
    register_constant_class(mpz_type)

register_mpz_with_pymbolic()

import islpy as isl
from islpy import dim_type
import numpy as np




# TODO list moved to MEMO in root




class LoopyAdvisory(UserWarning):
    pass

# {{{ imported user interface

from loopy.kernel import ScalarArg, ArrayArg, ImageArg

from loopy.kernel import LoopKernel
from loopy.schedule import generate_loop_schedules
from loopy.prefetch import insert_register_prefetches
from loopy.compiled import CompiledKernel, drive_timing_run

# }}}

# {{{ user-facing kernel manipulation functionality

def split_dimension(kernel, iname, inner_length, padded_length=None,
        outer_iname=None, inner_iname=None,
        outer_tag=None, inner_tag=None,
        outer_slab_increments=(0, -1), no_slabs=None):

    if iname not in kernel.all_inames():
        raise ValueError("cannot split loop for unknown variable '%s'" % iname)

    if no_slabs:
        outer_slab_increments = (0, 0)

    if padded_length is not None:
        inner_tag = inner_tag.copy(forced_length=padded_length)

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

        from loopy.isl import make_slab

        space = s.get_space()
        inner_constraint_set = (
                make_slab(space, inner_iname, 0, inner_length)
                # name = inner + length*outer
                .add_constraint(isl.Constraint.eq_from_names(
                    space, {iname:1, inner_iname: -1, outer_iname:-inner_length})))

        name_dim_type, name_idx = space.get_var_dict()[iname]
        return (s
                .intersect(inner_constraint_set)
                .project_out(name_dim_type, name_idx, 1))

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
            new_forced_iname_deps = insn.forced_iname_deps[:]
            new_forced_iname_deps.remove(iname)
            new_forced_iname_deps.extend([outer_iname, inner_iname])
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
    iname_slab_increments[outer_iname] = outer_slab_increments
    result = (kernel
            .copy(domain=new_domain,
                iname_slab_increments=iname_slab_increments,
                iname_to_dim=None,
                instructions=new_insns))

    return tag_dimensions(result, {outer_iname: outer_tag, inner_iname: inner_tag})




def tag_dimensions(kernel, iname_to_tag):
    from loopy.kernel import parse_tag

    iname_to_tag = dict((iname, parse_tag(tag))
            for iname, tag in iname_to_tag.iteritems())

    new_iname_to_tag = kernel.iname_to_tag.copy()
    for iname, new_tag in iname_to_tag.iteritems():
        if new_tag is None:
            continue

        if iname not in kernel.all_inames():
            raise ValueError("cannot tag '%s'--not known" % iname)

        old_tag = kernel.iname_to_tag.get(iname)
        if old_tag is not None and (old_tag != new_tag):
            raise RuntimeError("'%s' is already tagged '%s'--cannot retag"
                    % (iname, old_tag))

        new_iname_to_tag[iname] = new_tag

    return kernel.copy(iname_to_tag=new_iname_to_tag)





def realize_cse(kernel, cse_tag, dtype, duplicate_inames=[], parallel_inames=None,
        dup_iname_to_tag={}, new_inames=None):
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
    from loopy.kernel import TAG_AUTO_LOCAL_IDX
    for piname in parallel_inames:
        dup_iname_to_tag[piname] = TAG_AUTO_LOCAL_IDX()

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

    temp_new_inames = []
    for old_iname, new_iname in zip(duplicate_inames, new_inames):
        if new_iname is None:
            new_iname = kernel.make_unique_var_name(old_iname)
        temp_new_inames.append(new_iname)

    new_inames = temp_new_inames

    old_to_new_iname = dict(zip(duplicate_inames, new_inames))

    # }}}

    target_var_name = kernel.make_unique_var_name(cse_tag)

    from loopy.kernel import (TAG_LOCAL_IDX, TAG_AUTO_LOCAL_IDX,
            TAG_GROUP_IDX)
    target_var_is_local = any(
            isinstance(tag, (TAG_LOCAL_IDX, TAG_AUTO_LOCAL_IDX))
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

        parent_inames = insn.all_inames()
        forced_iname_deps = []

        from loopy.symbolic import IndexVariableFinder
        dependencies = IndexVariableFinder()(expr.child)

        assert dependencies <= parent_inames

        for iname in parent_inames:
            if iname in duplicate_inames:
                tag = dup_iname_to_tag[iname]
            else:
                tag = kernel.iname_to_tag[iname]

            if isinstance(tag, (TAG_LOCAL_IDX, TAG_AUTO_LOCAL_IDX)):
                kind = "l"
            elif isinstance(tag, TAG_GROUP_IDX):
                kind = "g"
            else:
                kind = "o"

            if iname in duplicate_inames and kind == "g":
                raise RuntimeError("duplicating inames into "
                        "group index axes is not helpful, as they cannot "
                        "collaborate in computing a local variable")

            if iname in dependencies:
                if not target_var_is_local and iname in duplicate_inames and kind == "l":
                    raise RuntimeError("invalid: parallelized "
                            "fetch into private variable")

                # otherwise: all happy
                continue

            # the iname is *not* a dependency of the fetch expression
            if iname in duplicate_inames:
                raise RuntimeError("duplicating an iname "
                        "that the CSE does not depend on "
                        "does not make sense")

            force_dependency = True
            if kind == "l" and target_var_is_local:
                force_dependency = False

            if force_dependency:
                forced_iname_deps.append(iname)

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

        from pymbolic import substitute
        new_inner_expr = substitute(rec(expr.child), dict(
            (old_iname, var(new_iname))
            for old_iname, new_iname in zip(duplicate_inames, new_inames)))

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

    dup_iname_dims = [kernel.iname_to_dim[iname]
            for iname in duplicate_inames]
    old_to_new = dict((old_iname, new_iname)
        for old_iname, new_iname in zip(duplicate_inames, new_inames))

    def realize_duplication(set):
        start_idx = set.dim(dim_type.set)
        result = set.insert_dims(
                dim_type.set, start_idx,
                len(duplicate_inames))

        new_iname_to_dim = kernel.iname_to_dim.copy()
        for i, iname in enumerate(new_inames):
            new_idx = start_idx+i
            result = result.set_dim_name(
                    dim_type.set, new_idx, iname)
            new_iname_to_dim[iname] = (dim_type.set, new_idx)


        set_bs, = set.get_basic_sets()

        for cns in set_bs.get_constraints():
            if any(cns.involves_dims(*dim+(1,)) for dim in dup_iname_dims):
                assert not cns.is_div_constraint()
                if cns.is_equality():
                    new_cns = cns.equality_alloc(result.get_space())
                else:
                    new_cns = cns.inequality_alloc(result.get_space())

                new_coeffs = {}
                for key, val in cns.get_coefficients_by_name().iteritems():
                    if key in old_to_new:
                        new_coeffs[old_to_new[key]] = val
                    else:
                        new_coeffs[key] = val

                new_cns = new_cns.set_coefficients_by_name(new_coeffs)
                result = result.add_constraint(new_cns)

        return result, new_iname_to_dim

    new_domain, new_iname_to_dim = realize_duplication(kernel.domain)

    # }}}

    # {{{ set up data for temp variable

    target_var_base_indices = []
    target_var_shape = []

    for iname in new_inames:
        lower_bound_pw_aff = new_domain.dim_min(new_iname_to_dim[iname][1])
        upper_bound_pw_aff = new_domain.dim_max(new_iname_to_dim[iname][1])

        from loopy.isl import static_max_of_pw_aff
        from loopy.symbolic import pw_aff_to_expr

        target_var_shape.append(static_max_of_pw_aff(
                upper_bound_pw_aff - lower_bound_pw_aff + 1, constants_only=True))
        target_var_base_indices.append(pw_aff_to_expr(lower_bound_pw_aff))

    from loopy.kernel import TemporaryVariable
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
            iname_to_dim=new_iname_to_dim,
            iname_to_tag=new_iname_to_tag)




def get_problems(kernel, parameters):
    """
    :return: *(max_severity, list of (severity, msg))*, where *severity* ranges from 1-5.
        '5' means 'will certainly not run'.
    """
    msgs = []

    def msg(severity, s):
        msgs.append((severity, s))

    glens, llens = kernel.get_grid_sizes_as_exprs()

    from pymbolic import evaluate
    glens = evaluate(glens, parameters)
    llens = evaluate(llens, parameters)

    if (max(len(glens), len(llens))
            > kernel.device.max_work_item_dimensions):
        msg(5, "too many work item dimensions")

    for i in range(len(llens)):
        if llens[i] > kernel.device.max_work_item_sizes[i]:
            msg(5, "group axis %d too big" % i)

    from pytools import product
    if product(llens) > kernel.device.max_work_group_size:
        msg(5, "work group too big")

    import pyopencl as cl
    from pyopencl.characterize import usable_local_mem_size
    if kernel.local_mem_use() > usable_local_mem_size(kernel.device):
        if kernel.device.local_mem_type == cl.device_local_mem_type.LOCAL:
            msg(5, "using too much local memory")
        else:
            msg(4, "using more local memory than available--"
                    "possibly OK due to cache nature")

    const_arg_count = sum(
            1 for arg in kernel.args
            if isinstance(arg, ArrayArg) and arg.constant_mem)

    if const_arg_count > kernel.device.max_constant_args:
        msg(5, "too many constant arguments")

    max_severity = 0
    for sev, msg in msgs:
        max_severity = max(sev, max_severity)
    return max_severity, msgs




def check_kernels(kernel_gen, parameters, kill_level_min=3,
        warn_level_min=1):
    for kernel in kernel_gen:
        max_severity, msgs = get_problems(kernel, parameters)

        for severity, msg in msgs:
            if severity >= warn_level_min:
                from warnings import warn
                from loopy import LoopyAdvisory
                warn(msg, LoopyAdvisory)

        if max_severity < kill_level_min:
            yield kernel

# }}}

# {{{ high-level modifiers

def get_input_access_descriptors(kernel):
    """Return a dictionary mapping input vectors to
    a list of input access descriptor. An input access
    descriptor is a tuple (input_vec, index_expr).
    """
    1/0 # broken

    from loopy.symbolic import VariableIndexExpressionCollector

    from pytools import flatten
    result = {}
    for ivec in kernel.input_vectors():
        result[ivec] = set(
                (ivec, iexpr)
                for iexpr in flatten(
                    VariableIndexExpressionCollector(ivec)(expression)
                    for lvalue, expression in kernel.instructions
                    ))

    return result

def add_prefetch(kernel, input_access_descr, fetch_dims, loc_fetch_axes={}):
    """
    :arg input_access_descr: see :func:`get_input_access_descriptors`.
        May also be the name of the variable if there is only one
        reference to that variable.
    :arg fetch_dims: loop dimensions indexing the input variable on which
        the prefetch is to be carried out.
    """
    1/0 # broken

    if isinstance(input_access_descr, str):
        var_name = input_access_descr
        var_iads = get_input_access_descriptors(kernel)[var_name]

        if len(var_iads) != 1:
            raise ValueError("input access descriptor for variable %s is "
                    "not unique" % var_name)

        input_access_descr, = var_iads

    def parse_fetch_dim(iname):
        if isinstance(iname, str):
            iname = (iname,)

        return tuple(kernel.tag_or_iname_to_iname(s) for s in iname)

    fetch_dims = [parse_fetch_dim(fd) for fd in fetch_dims]
    ivec, iexpr = input_access_descr

    new_prefetch = getattr(kernel, "prefetch", {}).copy()
    if input_access_descr in new_prefetch:
        raise ValueError("a prefetch descriptor for the input access %s[%s] "
                "already exists" % (ivec, iexpr))

    from loopy.prefetch import LocalMemoryPrefetch
    new_prefetch[input_access_descr] = LocalMemoryPrefetch(
            kernel=kernel,
            input_vector=ivec,
            index_expr=iexpr,
            fetch_dims=fetch_dims,
            loc_fetch_axes=loc_fetch_axes)

    return kernel.copy(prefetch=new_prefetch)

# }}}





# vim: foldmethod=marker

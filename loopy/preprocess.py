from __future__ import division

import pyopencl as cl
import pyopencl.characterize as cl_char




# {{{ local temporary finding

def mark_local_temporaries(kernel):
    new_temp_vars = {}
    from loopy.kernel import LocalIndexTagBase

    writers = kernel.find_writers()

    from loopy.symbolic import get_dependencies

    for temp_var in kernel.temporary_variables.itervalues():
        my_writers = writers[temp_var.name]

        has_local_parallel_write = False
        for insn_id in my_writers:
            insn = kernel.id_to_insn[insn_id]
            has_local_parallel_write = has_local_parallel_write or any(
                    isinstance(kernel.iname_to_tag.get(iname), LocalIndexTagBase)
                    for iname in get_dependencies(insn.get_assignee_indices())
                    & kernel.all_inames())

        new_temp_vars[temp_var.name] = temp_var.copy(
                is_local=has_local_parallel_write)

    return kernel.copy(temporary_variables=new_temp_vars)

# }}}

# {{{ reduction iname duplication

def duplicate_reduction_inames(kernel):

    # {{{ helper function

    newly_created_vars = set()

    def duplicate_reduction_inames(reduction_expr, rec):
        child = rec(reduction_expr.expr)
        new_red_inames = []
        did_something = False

        for iname in reduction_expr.inames:
            if iname.startswith("@"):
                new_iname = kernel.make_unique_var_name(iname[1:]+"_"+insn.id,
                        newly_created_vars)

                old_insn_inames.append(iname.lstrip("@"))
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
                        reduction_expr.untagged_inames, new_red_inames))
            subst_map = SubstitutionMapper(make_subst_func(subst_dict))

            child = subst_map(child)

        from loopy.symbolic import Reduction
        return Reduction(
                operation=reduction_expr.operation,
                inames=tuple(new_red_inames),
                expr=child)

    # }}}

    new_domain = kernel.domain
    new_insns = []

    for insn in kernel.instructions:
        old_insn_inames = []
        new_insn_inames = []

        from loopy.symbolic import ReductionCallbackMapper
        new_insns.append(insn.copy(
            expression=ReductionCallbackMapper(duplicate_reduction_inames)
            (insn.expression)))

        from loopy.isl_helpers import duplicate_axes
        for old, new in zip(old_insn_inames, new_insn_inames):
            new_domain = duplicate_axes(new_domain, [old], [new])

    return kernel.copy(
            instructions=new_insns,
            domain=new_domain)

# }}}

# {{{ rewrite reduction to imperative form

def realize_reduction(kernel):
    new_insns = []
    new_temporary_variables = kernel.temporary_variables.copy()

    from loopy.kernel import IlpTag

    def map_reduction(expr, rec):
        sub_expr = rec(expr.expr)

        # {{{ see if this reduction is nested inside some ILP loops

        ilp_inames = [iname
                for iname in kernel.insn_inames(insn)
                if isinstance(kernel.iname_to_tag.get(iname), IlpTag)]

        from loopy.isl_helpers import static_max_of_pw_aff

        ilp_iname_lengths = []
        for iname in ilp_inames:
            bounds = kernel.get_iname_bounds(iname)

            from loopy.symbolic import pw_aff_to_expr
            ilp_iname_lengths.append(
                    int(pw_aff_to_expr(
                        static_max_of_pw_aff(bounds.size, constants_only=True))))

        # }}}

        from pymbolic import var

        target_var_name = kernel.make_unique_var_name("acc",
                extra_used_vars=set(tv for tv in new_temporary_variables))
        target_var = var(target_var_name)

        if ilp_inames:
            target_var = target_var[
                    tuple(var(ilp_iname) for ilp_iname in ilp_inames)]

        from loopy.kernel import Instruction

        from loopy.kernel import TemporaryVariable
        new_temporary_variables[target_var_name] = TemporaryVariable(
                name=target_var_name,
                dtype=expr.operation.dtype,
                shape=tuple(ilp_iname_lengths),
                is_local=False)

        init_insn = Instruction(
                id=kernel.make_unique_instruction_id(
                    based_on="%s_%s_init" % (insn.id, "_".join(expr.inames)),
                    extra_used_ids=set(ni.id for ni in new_insns)),
                assignee=target_var,
                forced_iname_deps=kernel.insn_inames(insn) - set(expr.inames),
                expression=expr.operation.neutral_element)

        new_insns.append(init_insn)

        reduction_insn = Instruction(
                id=kernel.make_unique_instruction_id(
                    based_on="%s_%s_update" % (insn.id, "_".join(expr.inames)),
                    extra_used_ids=set(ni.id for ni in new_insns)),
                assignee=target_var,
                expression=expr.operation(target_var, sub_expr),
                insn_deps=set([init_insn.id]) | insn.insn_deps,
                forced_iname_deps=kernel.insn_inames(insn) | set(expr.inames))

        new_insns.append(reduction_insn)

        new_insn_insn_deps.add(reduction_insn.id)

        return target_var

    from loopy.symbolic import ReductionCallbackMapper
    cb_mapper = ReductionCallbackMapper(map_reduction)

    for insn in kernel.instructions:
        new_insn_insn_deps = set()

        new_expression = cb_mapper(insn.expression)

        new_insn = insn.copy(
                    expression=new_expression,
                    insn_deps=insn.insn_deps
                        | new_insn_insn_deps,
                    forced_iname_deps=kernel.insn_inames(insn))

        new_insns.append(new_insn)

    return kernel.copy(
            instructions=new_insns,
            temporary_variables=new_temporary_variables)

# }}}

# {{{ automatic dependencies, find boostability of instructions

def add_boostability_and_automatic_dependencies(kernel):
    writer_map = kernel.find_writers()

    arg_names = set(arg.name for arg in kernel.args)

    var_names = arg_names | set(kernel.temporary_variables.iterkeys())

    from loopy.symbolic import DependencyMapper
    dm = DependencyMapper(composite_leaves=False)
    dep_map = {}

    for insn in kernel.instructions:
        dep_map[insn.id] = (
                set(var.name for var in dm(insn.expression))
                & var_names)

    non_boostable_vars = set()

    new_insns = []
    for insn in kernel.instructions:
        auto_deps = set()

        # {{{ add automatic dependencies

        all_my_var_writers = set()
        for var in dep_map[insn.id]:
            var_writers = writer_map.get(var, set())
            all_my_var_writers |= var_writers

            if not var_writers and var not in arg_names:
                from warnings import warn
                warn("'%s' is read, but never written." % var)

            if len(var_writers) > 1 and not var_writers & set(insn.insn_deps):
                from warnings import warn
                warn("'%s' is written from more than one place, "
                        "but instruction '%s' (which reads this variable) "
                        "does not specify a dependency on any of the writers."
                        % (var, insn.id))

            if len(var_writers) == 1:
                auto_deps.update(var_writers)

        # }}}

        # {{{ find dependency loops, flag boostability

        while True:
            last_all_my_var_writers = all_my_var_writers

            for writer_insn_id in last_all_my_var_writers:
                for var in dep_map[writer_insn_id]:
                    all_my_var_writers = all_my_var_writers | writer_map.get(var, set())

            if last_all_my_var_writers == all_my_var_writers:
                break

        # }}}

        boostable = insn.id not in all_my_var_writers

        if not boostable:
            non_boostable_vars.add(insn.get_assignee_var_name())

        new_insns.append(
                insn.copy(
                    insn_deps=insn.insn_deps | auto_deps,
                    boostable=boostable))

    # {{{ remove boostability from isns that access non-boostable vars

    new2_insns = []
    for insn in new_insns:
        accessed_vars = (
                set([insn.get_assignee_var_name()])
                | insn.get_read_var_names())

        boostable = insn.boostable and not bool(non_boostable_vars & accessed_vars)
        new2_insns.append(insn.copy(boostable=boostable))

    # }}}

    return kernel.copy(instructions=new2_insns)

# }}}

# {{{ guess good iname for local axis 0

def get_axis_0_ranking(kernel, insn):
    from loopy.kernel import ImageArg, ScalarArg

    approximate_arg_values = dict(
            (arg.name, arg.approximately)
            for arg in kernel.args
            if isinstance(arg, ScalarArg))

    # {{{ find all array accesses in insn

    from loopy.symbolic import ArrayAccessFinder
    ary_acc_exprs = list(ArrayAccessFinder()(insn.expression))

    from pymbolic.primitives import Subscript

    if isinstance(insn.assignee, Subscript):
        ary_acc_exprs.append(insn.assignee)

    # }}}

    # {{{ filter array accesses to only the global ones

    global_ary_acc_exprs = []

    for aae in ary_acc_exprs:
        ary_name = aae.aggregate.name
        arg = kernel.arg_dict.get(ary_name)
        if arg is None:
            continue

        if isinstance(arg, ImageArg):
            continue

        global_ary_acc_exprs.append(aae)

    # }}}

    # {{{ figure out axis 0 candidates

    from loopy.kernel import AutoLocalIndexTagBase
    axis0_candidates = set(
            iname
            for iname in kernel.insn_inames(insn)
            if isinstance(kernel.iname_to_tag.get(iname),
                AutoLocalIndexTagBase))

    # }}}

    # {{{ figure out which iname should get mapped to local axis 0

    # maps inames to vote counts
    vote_count_for_l0 = {}

    from loopy.symbolic import CoefficientCollector

    from pytools import argmin2

    saw_relevant_access = False

    for aae in global_ary_acc_exprs:
        index_expr = aae.index
        if not isinstance(index_expr, tuple):
            index_expr = (index_expr,)

        ary_name = aae.aggregate.name
        arg = kernel.arg_dict.get(ary_name)

        ary_strides = arg.strides
        if ary_strides is None and len(index_expr) == 1:
            ary_strides = (1,)

        # {{{ construct iname_to_stride

        iname_to_stride = {}
        for iexpr_i, stride in zip(index_expr, ary_strides):
            coeffs = CoefficientCollector()(iexpr_i)
            for var_name, coeff in coeffs.iteritems():
                if var_name != 1:
                    new_stride = coeff*stride
                    old_stride = iname_to_stride.get(var_name, None)
                    if old_stride is None or new_stride < old_stride:
                        iname_to_stride[var_name] = new_stride

        # }}}

        if set(iname_to_stride.keys()) & axis0_candidates:
            saw_relevant_access = True

        if iname_to_stride:
            from pymbolic import evaluate
            least_stride_iname, least_stride = argmin2((
                    (iname,
                        evaluate(iname_to_stride[iname], approximate_arg_values))
                    for iname in iname_to_stride),
                    return_value=True)

            if least_stride == 1:
                vote_strength = 1
            else:
                vote_strength = 0.5

            vote_count_for_l0[least_stride_iname] = (
                    vote_count_for_l0.get(least_stride_iname, 0)
                    + vote_strength)

    if saw_relevant_access:
        return sorted((iname for iname in kernel.insn_inames(insn)),
                key=lambda iname: vote_count_for_l0.get(iname, 0),
                reverse=True)
    else:
        return None

    # }}}

# }}}

# {{{ assign automatic axes

def assign_automatic_axes(kernel, phase="axis0", local_size=None):
    from loopy.kernel import (AutoLocalIndexTagBase, LocalIndexTag,
            UnrollTag)

    # Realize that at this point in time, axis lengths are already
    # fixed. So we compute them once and pass them to our recursive
    # copies.

    if local_size is None:
        _, local_size = kernel.get_grid_sizes_as_exprs(
                ignore_auto=True)

    # {{{ axis assignment helper function

    def assign_axis(iname, axis=None):
        """Assign iname to local axis *axis* and start over by calling
        the surrounding function assign_automatic_axes.

        If *axis* is None, find a suitable axis automatically.
        """
        desired_length = kernel.get_constant_iname_length(iname)

        if axis is None:
            # {{{ find a suitable axis

            shorter_possible_axes = []
            test_axis = 0
            while True:
                if test_axis >= len(local_size):
                    break
                if test_axis in assigned_local_axes:
                    test_axis += 1
                    continue

                if local_size[test_axis] < desired_length:
                    shorter_possible_axes.append(test_axis)
                    test_axis += 1
                    continue
                else:
                    axis = test_axis
                    break

            # The loop above will find an unassigned local axis
            # that has enough 'room' for the iname. In the same traversal,
            # it also finds theoretically assignable axes that are shorter,
            # in the variable shorter_possible_axes.

            if axis is None and shorter_possible_axes:
                # sort as longest first
                shorter_possible_axes.sort(key=lambda ax: local_size[ax])
                axis = shorter_possible_axes[0]

            # }}}

        if axis is None:
            new_tag = None
        else:
            new_tag = LocalIndexTag(axis)
            if desired_length > local_size[axis]:
                from loopy import split_dimension
                return assign_automatic_axes(
                        split_dimension(kernel, iname, inner_length=local_size[axis],
                            outer_tag=UnrollTag(), inner_tag=new_tag,
                            do_tagged_check=False),
                        phase=phase, local_size=local_size)

        new_iname_to_tag = kernel.iname_to_tag.copy()
        new_iname_to_tag[iname] = new_tag
        return assign_automatic_axes(kernel.copy(iname_to_tag=new_iname_to_tag),
                phase=phase, local_size=local_size)

    # }}}

    # {{{ main assignment loop

    # assignment proceeds in two phases:

    # - "axis0": Only axis 0 is assigned on instructions that carry out
    #   global array access based on l.auto axes
    #
    # - "rest": All other l.auto axes are assigned haphazardly.

    for insn in kernel.instructions:
        auto_axis_inames = [
                iname
                for iname in kernel.insn_inames(insn)
                if isinstance(kernel.iname_to_tag.get(iname),
                    AutoLocalIndexTagBase)]

        if not auto_axis_inames:
            continue

        assigned_local_axes = set()

        for iname in kernel.insn_inames(insn):
            tag = kernel.iname_to_tag.get(iname)
            if isinstance(tag, LocalIndexTag):
                assigned_local_axes.add(tag.axis)

        if 0 < len(local_size) and 0 not in assigned_local_axes:
            axis0_ranking = get_axis_0_ranking(kernel, insn)
            if axis0_ranking is not None:
                for axis0_iname in axis0_ranking:
                    axis0_iname_tag = kernel.iname_to_tag.get(axis0_iname)
                    if isinstance(axis0_iname_tag, AutoLocalIndexTagBase):
                        return assign_axis(axis0_iname, 0)

        if phase == "axis0":
            continue

        # assign longest auto axis inames first
        auto_axis_inames.sort(key=kernel.get_constant_iname_length, reverse=True)

        if auto_axis_inames:
            return assign_axis(auto_axis_inames.pop())

    # }}}

    # We've seen all instructions and not punted to recursion/restart because
    # of a new axis assignment.

    if phase == "axis0":
        # If we were only assigining axis 0, then assign all the remaining
        # axes next.
        return assign_automatic_axes(kernel, phase="rest",
                local_size=local_size)
    else:
        # If were already assigning all axes and got here, we're now done.
        # All automatic axes are assigned.
        return kernel

# }}}

# {{{ temp storage adjust for bank conflict

def adjust_local_temp_var_storage(kernel):
    new_temp_vars = {}

    lmem_size = cl_char.usable_local_mem_size(kernel.device)
    for temp_var in kernel.temporary_variables.itervalues():
        if not temp_var.is_local:
            new_temp_vars[temp_var.name] = temp_var.copy(storage_shape=temp_var.shape)
            continue

        other_loctemp_nbytes = [tv.nbytes for tv in kernel.temporary_variables.itervalues()
                if tv.is_local and tv.name != temp_var.name]

        storage_shape = temp_var.storage_shape

        if storage_shape is None:
            storage_shape = temp_var.shape

        # sizes of all dims except the last one, which we may change
        # below to avoid bank conflicts
        from pytools import product
        other_dim_sizes = (tv.dtype.itemsize
                * product(storage_shape[:-1]))

        if kernel.device.local_mem_type == cl.device_local_mem_type.GLOBAL:
            # FIXME: could try to avoid cache associativity disasters
            new_storage_shape = storage_shape

        elif kernel.device.local_mem_type == cl.device_local_mem_type.LOCAL:
            min_mult = cl_char.local_memory_bank_count(kernel.device)
            good_incr = None
            new_storage_shape = storage_shape
            min_why_not = None

            for increment in range(storage_shape[-1]//2):

                test_storage_shape = storage_shape[:]
                test_storage_shape[-1] = test_storage_shape[-1] + increment
                new_mult, why_not = cl_char.why_not_local_access_conflict_free(
                        kernel.device, temp_var.dtype.itemsize,
                        temp_var.shape, test_storage_shape)

                # will choose smallest increment 'automatically'
                if new_mult < min_mult:
                    new_lmem_use = (sum(other_loctemp_nbytes)
                            + temp_var.dtype.itemsize*product(test_storage_shape))
                    if new_lmem_use < lmem_size:
                        new_storage_shape = test_storage_shape
                        min_mult = new_mult
                        min_why_not = why_not
                        good_incr = increment

            if min_mult != 1:
                from warnings import warn
                from loopy import LoopyAdvisory
                warn("could not find a conflict-free mem layout "
                        "for local variable '%s' "
                        "(currently: %dx conflict, increment: %d, reason: %s)"
                        % (temp_var.name, min_mult, good_incr, min_why_not),
                        LoopyAdvisory)
        else:
            from warnings import warn
            warn("unknown type of local memory")

            new_storage_shape = storage_shape

        new_temp_vars[temp_var.name] = temp_var.copy(storage_shape=new_storage_shape)

    return kernel.copy(temporary_variables=new_temp_vars)

# }}}




def preprocess_kernel(kernel):
    kernel = mark_local_temporaries(kernel)
    kernel = duplicate_reduction_inames(kernel)
    kernel = realize_reduction(kernel)

    # {{{ check that all CSEs have been realized

    from loopy.symbolic import CSECallbackMapper

    def map_cse(expr, rec):
        raise RuntimeError("all CSEs must be realized before scheduling")

    for insn in kernel.instructions:
        CSECallbackMapper(map_cse)(insn.expression)

    # }}}

    kernel = assign_automatic_axes(kernel)
    kernel = add_boostability_and_automatic_dependencies(kernel)
    kernel = adjust_local_temp_var_storage(kernel)

    return kernel




# vim: foldmethod=marker

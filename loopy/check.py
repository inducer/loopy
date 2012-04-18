from __future__ import division




# {{{ sanity checks run during scheduling

def check_for_unused_hw_axes_in_insns(kernel):
    group_size, local_size = kernel.get_grid_sizes_as_exprs()

    group_axes = set(ax for ax, length in enumerate(group_size))
    local_axes = set(ax for ax, length in enumerate(local_size))

    # alternative: just disregard length-1 dimensions?

    from loopy.kernel import LocalIndexTag, AutoLocalIndexTagBase, GroupIndexTag
    for insn in kernel.instructions:
        if insn.boostable:
            continue

        group_axes_used = set()
        local_axes_used = set()

        for iname in kernel.insn_inames(insn):
            tag = kernel.iname_to_tag.get(iname)

            if isinstance(tag, LocalIndexTag):
                local_axes_used.add(tag.axis)
            elif isinstance(tag, GroupIndexTag):
                group_axes_used.add(tag.axis)
            elif isinstance(tag, AutoLocalIndexTagBase):
                raise RuntimeError("auto local tag encountered")

        if group_axes != group_axes_used:
            raise RuntimeError("instruction '%s' does not use all group hw axes "
                    "(available: %s used:%s)"
                    % (insn.id,
                        ",".join(str(i) for i in group_axes),
                        ",".join(str(i) for i in group_axes_used)))
        if local_axes != local_axes_used:
            raise RuntimeError("instruction '%s' does not use all local hw axes"
                    "(available: %s used:%s)"
                    % (insn.id,
                        ",".join(str(i) for i in local_axes),
                        ",".join(str(i) for i in local_axes_used)))





def check_for_double_use_of_hw_axes(kernel):
    from loopy.kernel import UniqueTag

    for insn in kernel.instructions:
        insn_tag_keys = set()
        for iname in kernel.insn_inames(insn):
            tag = kernel.iname_to_tag.get(iname)
            if isinstance(tag, UniqueTag):
                key = tag.key
                if key in insn_tag_keys:
                    raise RuntimeError("instruction '%s' has multiple "
                            "inames tagged '%s'" % (insn.id, tag))

                insn_tag_keys.add(key)




def check_for_inactive_iname_access(kernel):
    from loopy.symbolic import DependencyMapper
    depmap = DependencyMapper()

    for insn in kernel.instructions:
        expression_indices = depmap(insn.expression)
        expression_inames = expression_indices & kernel.all_inames()

        if not expression_inames <= kernel.insn_inames(insn):
            raise RuntimeError(
                    "instructiosn '%s' references "
                    "inames that the instruction does not depend on"
                    % insn.id)




def check_for_write_races(kernel):
    from loopy.symbolic import DependencyMapper
    from loopy.kernel import ParallelTag, GroupIndexTag, IlpBaseTag
    depmap = DependencyMapper()

    for insn in kernel.instructions:
        assignee_name = insn.get_assignee_var_name()
        assignee_indices = depmap(insn.get_assignee_indices())

        def strip_var(expr):
            from pymbolic.primitives import Variable
            assert isinstance(expr, Variable)
            return expr.name

        assignee_indices = set(strip_var(index) for index in assignee_indices)

        assignee_inames = assignee_indices & kernel.all_inames()
        if not assignee_inames <= kernel.insn_inames(insn):
            raise RuntimeError(
                    "assignee of instructiosn '%s' references "
                    "iname that the instruction does not depend on"
                    % insn.id)

        inames_without_write_dep = None

        if assignee_name in kernel.arg_dict:
            # Any parallel tags that are not depended upon by the assignee
            # will cause write races.

            parallel_insn_inames = set(
                    iname
                    for iname in kernel.insn_inames(insn)
                    if isinstance(kernel.iname_to_tag.get(iname), ParallelTag))

            inames_without_write_dep = parallel_insn_inames - (
                    assignee_inames & parallel_insn_inames)

        elif assignee_name in kernel.temporary_variables:
            temp_var = kernel.temporary_variables[assignee_name]
            if temp_var.is_local == True:
                local_parallel_insn_inames = set(
                        iname
                        for iname in kernel.insn_inames(insn)
                        if isinstance(kernel.iname_to_tag.get(iname), ParallelTag)
                        and not isinstance(kernel.iname_to_tag.get(iname), GroupIndexTag))

                inames_without_write_dep = local_parallel_insn_inames - (
                        assignee_inames & local_parallel_insn_inames)

            elif temp_var.is_local == False:
                #from loopy.kernel import IlpBaseTag
                #ilp_inames = set(
                        #iname
                        #for iname in kernel.insn_inames(insn)
                        #if isinstance(kernel.iname_to_tag.get(iname), IlpBaseTag))

                #inames_without_write_dep = ilp_inames - assignee_inames

                inames_without_write_dep = set()

            else:
                raise RuntimeError("temp var '%s' hasn't decided on "
                        "whether it is local" % temp_var.name)

        else:
            raise RuntimeError("invalid assignee name in instruction '%s'"
                    % insn.id)

        assert inames_without_write_dep is not None

        if inames_without_write_dep:
            raise RuntimeError(
                    "instruction '%s' contains a write race: "
                    "instruction will be run across parallel iname(s) '%s', which "
                    "is/are not referenced in the lhs index"
                    % (insn.id, ",".join(inames_without_write_dep)))

def check_for_orphaned_user_hardware_axes(kernel):
    from loopy.kernel import LocalIndexTag
    for axis in kernel.local_sizes:
        found = False
        for tag in kernel.iname_to_tag.itervalues():
            if isinstance(tag, LocalIndexTag) and tag.axis == axis:
                found = True
                break

        if not found:
            raise RuntimeError("user-requested local hardware axis %d "
                    "has no iname mapped to it" % axis)

# }}}

def run_automatic_checks(kernel):
    check_for_orphaned_user_hardware_axes(kernel)
    check_for_double_use_of_hw_axes(kernel)
    check_for_unused_hw_axes_in_insns(kernel)
    check_for_inactive_iname_access(kernel)
    check_for_write_races(kernel)


# {{{ sanity-check for implemented domains of each instruction

def check_implemented_domains(kernel, implemented_domains):
    from islpy import dim_type

    parameter_inames = set(
            kernel.domain.get_dim_name(dim_type.param, i)
            for i in range(kernel.domain.dim(dim_type.param)))

    from islpy import align_spaces
    assumptions = align_spaces(kernel.assumptions, kernel.domain)

    for insn_id, idomains in implemented_domains.iteritems():
        insn = kernel.id_to_insn[insn_id]

        assert idomains

        insn_impl_domain = idomains[0]
        for idomain in idomains[1:]:
            insn_impl_domain = insn_impl_domain | idomain
        insn_impl_domain = (
                (insn_impl_domain & assumptions)
                .project_out_except(kernel.insn_inames(insn), [dim_type.set]))

        desired_domain = ((kernel.domain & assumptions)
            .project_out_except(kernel.insn_inames(insn), [dim_type.set]))

        if insn_impl_domain != desired_domain:
            i_minus_d = insn_impl_domain - desired_domain
            d_minus_i = desired_domain - insn_impl_domain

            lines = []
            for kind, diff_set in [
                    ("implemented, but not desired", i_minus_d),
                    ("desired, but not implemented", d_minus_i)]:
                diff_set = diff_set.coalesce()
                pt = diff_set.sample_point()
                if pt.is_void():
                    continue

                #pt_set = isl.Set.from_point(pt)
                #lines.append("point implemented: %s" % (pt_set <= insn_impl_domain))
                #lines.append("point desired: %s" % (pt_set <= desired_domain))

                iname_to_dim = pt.get_space().get_var_dict()
                point_axes = []
                for iname in kernel.insn_inames(insn) | parameter_inames:
                    tp, dim = iname_to_dim[iname]
                    point_axes.append("%s=%d" % (iname, pt.get_coordinate(tp, dim)))

                lines.append(
                        "sample point %s: %s" % (kind, ", ".join(point_axes)))

            raise RuntimeError("sanity check failed--implemented and desired "
                    "domain for instruction '%s' do not match\n\n"
                    "implemented: %s\n\n"
                    "desired:%s\n\n%s"
                    % (insn_id, insn_impl_domain, desired_domain, "\n".join(lines)))

    # placate the assert at the call site
    return True

# }}}

# {{{ user-invoked checks

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
    from pymbolic.mapper.evaluator import UnknownVariableError
    try:
        glens = evaluate(glens, parameters)
        llens = evaluate(llens, parameters)
    except UnknownVariableError, name:
        raise RuntimeError("When checking your kernel for problems, "
                "a value for parameter '%s' was not available. Pass "
                "it in the 'parameters' kwarg to check_kernels()."
                % name)

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

    from loopy.kernel import ConstantArg
    const_arg_count = sum(
            1 for arg in kernel.args
            if isinstance(arg, ConstantArg))

    if const_arg_count > kernel.device.max_constant_args:
        msg(5, "too many constant arguments")

    max_severity = 0
    for sev, msg in msgs:
        max_severity = max(sev, max_severity)
    return max_severity, msgs




def check_kernels(kernel_gen, parameters={}, kill_level_min=5,
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

# vim: foldmethod=marker

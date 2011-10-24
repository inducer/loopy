from __future__ import division




# {{{ sanity checks run during scheduling

def check_for_unused_hw_axes(kernel):
    group_size, local_size = kernel.get_grid_sizes_as_exprs()

    group_axes = set(range(len(group_size)))
    local_axes = set(range(len(local_size)))

    from loopy.kernel import LocalIndexTag, AutoLocalIndexTagBase, GroupIndexTag
    for insn in kernel.instructions:
        group_axes_used = set()
        local_axes_used = set()

        for iname in insn.all_inames():
            tag = kernel.iname_to_tag.get(iname)

            if isinstance(tag, LocalIndexTag):
                local_axes_used.add(tag.axis)
            elif isinstance(tag, GroupIndexTag):
                group_axes_used.add(tag.axis)
            elif isinstance(tag, AutoLocalIndexTagBase):
                raise RuntimeError("auto local tag encountered")

        if group_axes != group_axes_used:
            raise RuntimeError("instruction '%s' does not use all group hw axes"
                    % insn.id)
        if local_axes != local_axes_used:
            raise RuntimeError("instruction '%s' does not use all local hw axes"
                    % insn.id)





def check_for_double_use_of_hw_axes(kernel):
    from loopy.kernel import UniqueTag

    for insn in kernel.instructions:
        insn_tag_keys = set()
        for iname in insn.all_inames():
            tag = kernel.iname_to_tag.get(iname)
            if isinstance(tag, UniqueTag):
                key = tag.key
                if key in insn_tag_keys:
                    raise RuntimeError("instruction '%s' has two "
                            "inames tagged '%s'" % (insn.id, tag))

                insn_tag_keys.add(key)




def check_for_inactive_iname_access(kernel):
    from loopy.symbolic import DependencyMapper
    depmap = DependencyMapper()

    for insn in kernel.instructions:
        expression_indices = depmap(insn.expression)
        expression_inames = expression_indices & kernel.all_inames()

        if not expression_inames <= insn.all_inames():
            raise RuntimeError(
                    "instructiosn '%s' references "
                    "inames that the instruction does not depend on"
                    % insn.id)




def check_for_write_races(kernel):
    from pymbolic.primitives import Subscript, Variable
    from loopy.symbolic import DependencyMapper
    from loopy.kernel import ParallelTag, GroupIndexTag, IlpTag
    depmap = DependencyMapper()

    for insn in kernel.instructions:
        if isinstance(insn.assignee, Subscript):
            assert isinstance(insn.assignee, Subscript)

            var = insn.assignee.aggregate
            assert isinstance(var, Variable)
            assignee_name = var.name
            assignee_indices = depmap(insn.assignee.index)
        elif isinstance(insn.assignee, Variable):
            assignee_name = insn.assignee.name
            assignee_indices = set()
        else:
            raise RuntimeError("assignee for instruction '%s' not understood"
                    % insn.id)

        def strip_var(expr):
            assert isinstance(expr, Variable)
            return expr.name

        assignee_indices = set(strip_var(index) for index in assignee_indices)

        assignee_inames = assignee_indices & kernel.all_inames()
        if not assignee_inames <= insn.all_inames():
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
                    for iname in insn.all_inames()
                    if isinstance(kernel.iname_to_tag.get(iname), ParallelTag))

            inames_without_write_dep = parallel_insn_inames - (
                    assignee_inames & parallel_insn_inames)

        elif assignee_name in kernel.temporary_variables:
            temp_var = kernel.temporary_variables[assignee_name]
            if temp_var.is_local:
                local_parallel_insn_inames = set(
                        iname
                        for iname in insn.all_inames()
                        if isinstance(kernel.iname_to_tag.get(iname), ParallelTag)
                        and not isinstance(kernel.iname_to_tag.get(iname), GroupIndexTag))

                inames_without_write_dep = local_parallel_insn_inames - (
                        assignee_inames & local_parallel_insn_inames)

            else:
                ilp_inames = set(
                        iname
                        for iname in insn.all_inames()
                        if isinstance(kernel.iname_to_tag.get(iname), IlpTag))

                inames_without_write_dep = ilp_inames - (
                        assignee_inames & ilp_inames)


        else:
            raise RuntimeError("invalid assignee name in instruction '%s'"
                    % insn.id)

        assert inames_without_write_dep is not None

        if inames_without_write_dep:
            raise RuntimeError(
                    "instruction '%s' contains a write race: "
                    "instruction will be run across parallel iname(s) '%s', which "
                    "is/are not referenced in the assignee index"
                    % (insn.id, ",".join(inames_without_write_dep)))

# }}}




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
                .project_out_except(insn.all_inames(), [dim_type.set]))

        desired_domain = ((kernel.domain & assumptions)
            .project_out_except(insn.all_inames(), [dim_type.set]))

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

                point_axes = []
                for iname in insn.all_inames() | parameter_inames:
                    tp, dim = kernel.iname_to_dim[iname]
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

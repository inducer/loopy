from __future__ import absolute_import, division, print_function

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
from six.moves import range

from islpy import dim_type
import islpy as isl
from loopy.symbolic import WalkMapper
from loopy.diagnostic import LoopyError, WriteRaceConditionWarning, warn_with_kernel

import logging
logger = logging.getLogger(__name__)


# {{{ sanity checks run pre-scheduling

def check_insn_attributes(kernel):
    all_insn_ids = set(insn.id for insn in kernel.instructions)

    for insn in kernel.instructions:
        if not insn.forced_iname_deps <= kernel.all_inames():
            raise LoopyError("insn '%s' has unknown forced iname "
                    "dependencies: %s"
                    % (insn.id, ", ".join(
                        insn.forced_iname_deps - kernel.all_inames())))

        if insn.depends_on is not None and not insn.depends_on <= all_insn_ids:
            raise LoopyError("insn '%s' has unknown instruction "
                    "dependencies: %s"
                    % (insn.id, ", ".join(
                        insn.depends_on - all_insn_ids)))


def check_loop_priority_inames_known(kernel):
    for iname in kernel.loop_priority:
        if iname not in kernel.all_inames():
            raise LoopyError("unknown iname '%s' in loop priorities" % iname)


def check_for_double_use_of_hw_axes(kernel):
    from loopy.kernel.data import UniqueTag

    for insn in kernel.instructions:
        insn_tag_keys = set()
        for iname in kernel.insn_inames(insn):
            tag = kernel.iname_to_tag.get(iname)
            if isinstance(tag, UniqueTag):
                key = tag.key
                if key in insn_tag_keys:
                    raise LoopyError("instruction '%s' has multiple "
                            "inames tagged '%s'" % (insn.id, tag))

                insn_tag_keys.add(key)


def check_for_inactive_iname_access(kernel):
    for insn in kernel.instructions:
        expression_inames = insn.read_dependency_names() & kernel.all_inames()

        if not expression_inames <= kernel.insn_inames(insn):
            raise LoopyError(
                    "instruction '%s' references "
                    "inames '%s' that the instruction does not depend on"
                    % (insn.id,
                        ", ".join(expression_inames - kernel.insn_inames(insn))))


def _is_racing_iname_tag(tv, tag):
    from loopy.kernel.data import (temp_var_scope,
            LocalIndexTagBase, GroupIndexTag, ParallelTag, auto)

    if tv.scope == temp_var_scope.PRIVATE:
        return (
                isinstance(tag, ParallelTag)
                and not isinstance(tag, (LocalIndexTagBase, GroupIndexTag)))

    elif tv.scope == temp_var_scope.LOCAL:
        return (
                isinstance(tag, ParallelTag)
                and not isinstance(tag, GroupIndexTag))

    elif tv.scope == temp_var_scope.GLOBAL:
        return isinstance(tag, ParallelTag)

    elif tv.scope == auto:
        raise LoopyError("scope of temp var '%s' has not yet been"
                "determined" % tv.name)

    else:
        raise ValueError("unexpected value of temp_var.scope for "
                "temporary variable '%s'" % tv.name)


def check_for_write_races(kernel):
    from loopy.kernel.data import ParallelTag

    iname_to_tag = kernel.iname_to_tag.get
    for insn in kernel.instructions:
        for assignee_name, assignee_indices in zip(
                insn.assignee_var_names(),
                insn.assignee_subscript_deps()):
            assignee_inames = assignee_indices & kernel.all_inames()
            if not assignee_inames <= kernel.insn_inames(insn):
                raise LoopyError(
                        "assignee of instructiosn '%s' references "
                        "iname that the instruction does not depend on"
                        % insn.id)

            if assignee_name in kernel.arg_dict:
                # Any parallel tags that are not depended upon by the assignee
                # will cause write races.

                raceable_parallel_insn_inames = set(
                        iname
                        for iname in kernel.insn_inames(insn)
                        if isinstance(iname_to_tag(iname), ParallelTag))

            elif assignee_name in kernel.temporary_variables:
                temp_var = kernel.temporary_variables[assignee_name]
                raceable_parallel_insn_inames = set(
                            iname
                            for iname in kernel.insn_inames(insn)
                            if _is_racing_iname_tag(temp_var, iname_to_tag(iname)))

            else:
                raise LoopyError("invalid assignee name in instruction '%s'"
                        % insn.id)

            race_inames = \
                    raceable_parallel_insn_inames - assignee_inames

            if race_inames:
                warn_with_kernel(kernel, "write_race(%s)" % insn.id,
                        "instruction '%s' contains a write race: "
                        "instruction will be run across parallel iname(s) "
                        "'%s', which is/are not referenced in the lhs index"
                        % (insn.id, ",".join(race_inames)),
                        WriteRaceConditionWarning)


def check_for_orphaned_user_hardware_axes(kernel):
    from loopy.kernel.data import LocalIndexTag
    for axis in kernel.local_sizes:
        found = False
        for tag in six.itervalues(kernel.iname_to_tag):
            if isinstance(tag, LocalIndexTag) and tag.axis == axis:
                found = True
                break

        if not found:
            raise LoopyError("user-requested local hardware axis %d "
                    "has no iname mapped to it" % axis)


def check_for_data_dependent_parallel_bounds(kernel):
    from loopy.kernel.data import ParallelTag

    for i, dom in enumerate(kernel.domains):
        dom_inames = set(dom.get_var_names(dim_type.set))
        par_inames = set(iname
                for iname in dom_inames
                if isinstance(kernel.iname_to_tag.get(iname), ParallelTag))

        if not par_inames:
            continue

        parameters = set(dom.get_var_names(dim_type.param))
        for par in parameters:
            if par in kernel.temporary_variables:
                raise LoopyError("Domain number %d has a data-dependent "
                        "parameter '%s' and contains parallel "
                        "inames '%s'. This is not allowed (for now)."
                        % (i, par, ", ".join(par_inames)))


class _AccessCheckMapper(WalkMapper):
    def __init__(self, kernel, domain, insn_id):
        self.kernel = kernel
        self.domain = domain
        self.insn_id = insn_id

    def map_subscript(self, expr):
        WalkMapper.map_subscript(self, expr)

        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        shape = None
        var_name = expr.aggregate.name
        if var_name in self.kernel.arg_dict:
            arg = self.kernel.arg_dict[var_name]
            shape = arg.shape
        elif var_name in self.kernel.temporary_variables:
            tv = self.kernel.temporary_variables[var_name]
            shape = tv.shape

        if shape is not None:
            subscript = expr.index

            if not isinstance(subscript, tuple):
                subscript = (subscript,)

            from loopy.symbolic import get_dependencies, get_access_range

            available_vars = set(self.domain.get_var_dict())
            shape_deps = set()
            for shape_axis in shape:
                if shape_axis is not None:
                    shape_deps.update(get_dependencies(shape_axis))

            if not (get_dependencies(subscript) <= available_vars
                    and shape_deps <= available_vars):
                return

            if len(subscript) != len(shape):
                raise LoopyError("subscript to '%s' in '%s' has the wrong "
                        "number of indices (got: %d, expected: %d)" % (
                            expr.aggregate.name, expr,
                            len(subscript), len(shape)))

            try:
                access_range = get_access_range(self.domain, subscript,
                        self.kernel.assumptions)
            except isl.Error:
                # Likely: index was non-linear, nothing we can do.
                return
            except TypeError:
                # Likely: index was non-linear, nothing we can do.
                return

            shape_domain = isl.BasicSet.universe(access_range.get_space())
            for idim in range(len(subscript)):
                shape_axis = shape[idim]

                if shape_axis is not None:
                    from loopy.isl_helpers import make_slab
                    slab = make_slab(
                            shape_domain.get_space(), (dim_type.in_, idim),
                            0, shape_axis)

                    shape_domain = shape_domain.intersect(slab)

            if not access_range.is_subset(shape_domain):
                raise LoopyError("'%s' in instruction '%s' "
                        "accesses out-of-bounds array element"
                        % (expr, self.insn_id))


def check_bounds(kernel):
    temp_var_names = set(kernel.temporary_variables)
    for insn in kernel.instructions:
        domain = kernel.get_inames_domain(kernel.insn_inames(insn))

        # data-dependent bounds? can't do much
        if set(domain.get_var_names(dim_type.param)) & temp_var_names:
            continue

        acm = _AccessCheckMapper(kernel, domain, insn.id)
        insn.with_transformed_expressions(acm)


def check_write_destinations(kernel):
    for insn in kernel.instructions:
        for wvar in insn.assignee_var_names():
            if wvar in kernel.all_inames():
                raise LoopyError("iname '%s' may not be written" % wvar)

            insn_domain = kernel.get_inames_domain(kernel.insn_inames(insn))
            insn_params = set(insn_domain.get_var_names(dim_type.param))

            if wvar in kernel.all_params():
                if wvar not in kernel.temporary_variables:
                    raise LoopyError("domain parameter '%s' may not be written"
                            "--it is not a temporary variable" % wvar)

                if wvar in insn_params:
                    raise LoopyError("domain parameter '%s' may not be written "
                            "inside a domain dependent on it" % wvar)

            if not (wvar in kernel.temporary_variables
                    or wvar in kernel.arg_dict) and wvar not in kernel.all_params():
                raise LoopyError

# }}}


def pre_schedule_checks(kernel):
    try:
        logger.info("pre-schedule check %s: start" % kernel.name)

        check_for_orphaned_user_hardware_axes(kernel)
        check_for_double_use_of_hw_axes(kernel)
        check_insn_attributes(kernel)
        check_loop_priority_inames_known(kernel)
        check_for_inactive_iname_access(kernel)
        check_for_write_races(kernel)
        check_for_data_dependent_parallel_bounds(kernel)
        check_bounds(kernel)
        check_write_destinations(kernel)

        logger.info("pre-schedule check %s: done" % kernel.name)
    except KeyboardInterrupt:
        raise
    except:
        print(75*"=")
        print("failing kernel during pre-schedule check:")
        print(75*"=")
        print(kernel)
        print(75*"=")
        raise


# {{{ post-schedule / pre-code-generation checks

# {{{ check for unused hw axes

def _check_for_unused_hw_axes_in_kernel_chunk(kernel, sched_index=None):
    from loopy.schedule import (CallKernel, RunInstruction,
            Barrier, EnterLoop, LeaveLoop, ReturnFromKernel,
            get_insn_ids_for_block_at, gather_schedule_block)

    if sched_index is None:
        group_axes = set()
        local_axes = set()

        i = 0
        loop_end_i = past_end_i = len(kernel.schedule)
    else:
        assert isinstance(kernel.schedule[sched_index], CallKernel)
        _, past_end_i = gather_schedule_block(kernel.schedule, sched_index)
        group_size, local_size = kernel.get_grid_sizes_for_insn_ids_as_exprs(
                get_insn_ids_for_block_at(kernel.schedule, sched_index))

        group_axes = set(ax for ax, length in enumerate(group_size))
        local_axes = set(ax for ax, length in enumerate(local_size))

        i = sched_index + 1
        assert isinstance(kernel.schedule[past_end_i - 1], ReturnFromKernel)
        loop_end_i = past_end_i - 1

    # alternative: just disregard length-1 dimensions?

    from loopy.kernel.data import LocalIndexTag, AutoLocalIndexTagBase, GroupIndexTag

    while i < loop_end_i:
        sched_item = kernel.schedule[i]
        if isinstance(sched_item, CallKernel):
            i = _check_for_unused_hw_axes_in_kernel_chunk(kernel, i)

        elif isinstance(sched_item, RunInstruction):
            insn = kernel.id_to_insn[sched_item.insn_id]
            i += 1

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
                    raise LoopyError("auto local tag encountered")

            if group_axes != group_axes_used:
                raise LoopyError("instruction '%s' does not use all group hw axes "
                        "(available: %s used:%s)"
                        % (insn.id,
                            ",".join(str(i) for i in group_axes),
                            ",".join(str(i) for i in group_axes_used)))
            if local_axes != local_axes_used:
                raise LoopyError("instruction '%s' does not use all local hw axes "
                        "(available: %s used:%s)"
                        % (insn.id,
                            ",".join(str(i) for i in local_axes),
                            ",".join(str(i) for i in local_axes_used)))

        elif isinstance(sched_item, (Barrier, EnterLoop, LeaveLoop)):
            i += 1
            continue

        else:
            raise TypeError(
                    "schedule item not understood: %s" % type(sched_item).__name__)

    return past_end_i


def check_for_unused_hw_axes_in_insns(kernel):
    if kernel.schedule:
        _check_for_unused_hw_axes_in_kernel_chunk(kernel)

# }}}


# {{{ check that atomic ops are used exactly on atomic arrays

def check_that_atomic_ops_are_used_exactly_on_atomic_arrays(kernel):
    from loopy.kernel.data import ArrayBase, Assignment
    from loopy.types import AtomicType
    atomicity_candidates = (
            set(v.name for v in six.itervalues(kernel.temporary_variables)
                if isinstance(v.dtype, AtomicType))
            |
            set(v.name for v in kernel.args
                if isinstance(v, ArrayBase)
                and isinstance(v.dtype, AtomicType)))

    for insn in kernel.instructions:
        if not isinstance(insn, Assignment):
            continue

        atomic_accesses = set(a.var_name for a in insn.atomicity)
        if not atomic_accesses <= atomicity_candidates:
            raise LoopyError("atomic access in instruction '%s' to "
                    "non-atomic variable(s) '%s'"
                    % (insn.id,
                        ",".join(atomic_accesses - atomicity_candidates)))

        accessed_atomic_vars = insn.dependency_names() & atomicity_candidates
        if not accessed_atomic_vars <= atomic_accesses:
            raise LoopyError("atomic variable(s) '%s' in instruction '%s' "
                    "used in non-atomic access"
                    % (
                        ",".join(accessed_atomic_vars - atomic_accesses),
                        insn.id))

# }}}


# {{{ check that shapes and strides are arguments

def check_that_shapes_and_strides_are_arguments(kernel):
    from loopy.kernel.data import ValueArg
    from loopy.kernel.array import ArrayBase, FixedStrideArrayDimTag
    from loopy.symbolic import get_dependencies
    import loopy as lp

    integer_arg_names = set(
            arg.name
            for arg in kernel.args
            if isinstance(arg, ValueArg)
            and arg.dtype.is_integral())

    for arg in kernel.args:
        if isinstance(arg, ArrayBase):
            if isinstance(arg.shape, tuple):
                shape_deps = set()
                for shape_axis in arg.shape:
                    if shape_axis is not None:
                        shape_deps.update(get_dependencies(shape_axis))

                if not shape_deps <= integer_arg_names:
                    raise LoopyError("'%s' has a shape that depends on "
                            "non-argument(s): %s" % (
                                arg.name, ", ".join(shape_deps-integer_arg_names)))

            if arg.dim_tags is None:
                continue

            for dim_tag in arg.dim_tags:
                if isinstance(dim_tag, FixedStrideArrayDimTag):
                    if dim_tag.stride is lp.auto:
                        continue

                    deps = get_dependencies(dim_tag.stride)
                    if not deps <= integer_arg_names:
                        raise LoopyError("'%s' has a stride that depends on "
                                "non-argument(s): %s" % (
                                    arg.name, ", ".join(deps-integer_arg_names)))

# }}}


def pre_codegen_checks(kernel):
    try:
        logger.info("pre-codegen check %s: start" % kernel.name)

        check_for_unused_hw_axes_in_insns(kernel)
        check_that_atomic_ops_are_used_exactly_on_atomic_arrays(kernel)
        kernel.target.pre_codegen_check(kernel)
        check_that_shapes_and_strides_are_arguments(kernel)

        logger.info("pre-codegen check %s: done" % kernel.name)
    except:
        print(75*"=")
        print("failing kernel during pre-schedule check:")
        print(75*"=")
        print(kernel)
        print(75*"=")
        raise

# }}}


# {{{ sanity-check for implemented domains of each instruction

def check_implemented_domains(kernel, implemented_domains, code=None):
    from islpy import dim_type

    from islpy import align_two

    last_idomains = None
    last_insn_inames = None

    for insn_id, idomains in six.iteritems(implemented_domains):
        insn = kernel.id_to_insn[insn_id]

        assert idomains

        insn_inames = kernel.insn_inames(insn)

        # {{{ if we've checked the same thing before, no need to check it again

        if last_idomains is not None and last_insn_inames is not None:
            if idomains == last_idomains and insn_inames == last_insn_inames:
                continue

        last_idomains = idomains
        last_insn_inames = insn_inames

        # }}}

        insn_impl_domain = idomains[0]
        for idomain in idomains[1:]:
            insn_impl_domain = insn_impl_domain | idomain
        assumption_non_param = isl.BasicSet.from_params(kernel.assumptions)
        assumptions, insn_impl_domain = align_two(
                assumption_non_param, insn_impl_domain)
        insn_impl_domain = (
                (insn_impl_domain & assumptions)
                .project_out_except(insn_inames, [dim_type.set]))

        insn_domain = kernel.get_inames_domain(insn_inames)
        insn_parameters = frozenset(insn_domain.get_var_names(dim_type.param))
        assumptions, insn_domain = align_two(assumption_non_param, insn_domain)
        desired_domain = ((insn_domain & assumptions)
            .project_out_except(insn_inames, [dim_type.set])
            .project_out_except(insn_parameters, [dim_type.param]))

        insn_impl_domain = (insn_impl_domain
                .project_out_except(insn_parameters, [dim_type.param]))
        insn_impl_domain, desired_domain = align_two(
                insn_impl_domain, desired_domain)

        if insn_impl_domain != desired_domain:
            i_minus_d = insn_impl_domain - desired_domain
            d_minus_i = desired_domain - insn_impl_domain

            parameter_inames = set(
                    insn_domain.get_dim_name(dim_type.param, i)
                    for i in range(insn_domain.dim(dim_type.param)))

            lines = []
            for kind, diff_set, gist_domain in [
                    ("implemented, but not desired", i_minus_d,
                        desired_domain.gist(insn_impl_domain)),
                    ("desired, but not implemented", d_minus_i,
                        insn_impl_domain.gist(desired_domain))]:

                if diff_set.is_empty():
                    continue

                diff_set = diff_set.coalesce()
                pt = diff_set.sample_point()
                assert not pt.is_void()

                #pt_set = isl.Set.from_point(pt)
                #lines.append("point implemented: %s" % (pt_set <= insn_impl_domain))
                #lines.append("point desired: %s" % (pt_set <= desired_domain))

                iname_to_dim = pt.get_space().get_var_dict()
                point_axes = []
                for iname in kernel.insn_inames(insn) | parameter_inames:
                    tp, dim = iname_to_dim[iname]
                    point_axes.append("%s=%d" % (
                        iname, pt.get_coordinate_val(tp, dim).to_python()))

                lines.append(
                        "sample point in %s: %s" % (kind, ", ".join(point_axes)))
                lines.append(
                        "gist of %s: %s" % (kind, gist_domain))

            if code is not None:
                print(79*"-")
                print("CODE:")
                print(79*"-")
                from loopy.compiled import get_highlighted_cl_code
                print(get_highlighted_cl_code(code))
                print(79*"-")

            raise LoopyError("sanity check failed--implemented and desired "
                    "domain for instruction '%s' do not match\n\n"
                    "implemented: %s\n\n"
                    "desired:%s\n\n%s"
                    % (insn_id, insn_impl_domain, desired_domain, "\n".join(lines)))

    # placate the assert at the call site
    return True

# }}}


# vim: foldmethod=marker

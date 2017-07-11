from __future__ import division, absolute_import

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
from loopy.diagnostic import (
        LoopyError, WriteRaceConditionWarning, warn_with_kernel,
        LoopyAdvisory)

import islpy as isl

from pytools.persistent_dict import PersistentDict

from loopy.tools import LoopyKeyBuilder
from loopy.version import DATA_MODEL_VERSION
from loopy.kernel.data import make_assignment
# for the benefit of loopy.statistics, for now
from loopy.type_inference import infer_unknown_types

import logging
logger = logging.getLogger(__name__)


# {{{ prepare for caching

def prepare_for_caching(kernel):
    import loopy as lp
    new_args = []

    tgt = kernel.target

    for arg in kernel.args:
        dtype = arg.dtype
        if dtype is not None and dtype is not lp.auto and dtype.target is not tgt:
            arg = arg.copy(dtype=dtype.with_target(kernel.target))

        new_args.append(arg)

    new_temporary_variables = {}
    for name, temp in six.iteritems(kernel.temporary_variables):
        dtype = temp.dtype
        if dtype is not None and dtype is not lp.auto and dtype.target is not tgt:
            temp = temp.copy(dtype=dtype.with_target(tgt))

        new_temporary_variables[name] = temp

    kernel = kernel.copy(
            args=new_args,
            temporary_variables=new_temporary_variables)

    return kernel

# }}}


# {{{ check for writes to predicates

def check_for_writes_to_predicates(kernel):
    from loopy.symbolic import get_dependencies
    for insn in kernel.instructions:
        pred_vars = (
                frozenset.union(
                    *(get_dependencies(pred) for pred in insn.predicates))
                if insn.predicates else frozenset())
        written_pred_vars = frozenset(insn.assignee_var_names()) & pred_vars
        if written_pred_vars:
            raise LoopyError("In instruction '%s': may not write to "
                    "variable(s) '%s' involved in the instruction's predicates"
                    % (insn.id, ", ".join(written_pred_vars)))

# }}}


# {{{ check reduction iname uniqueness

def check_reduction_iname_uniqueness(kernel):
    iname_to_reduction_count = {}
    iname_to_nonsimultaneous_reduction_count = {}

    def map_reduction(expr, rec):
        rec(expr.expr)

        for iname in expr.inames:
            iname_to_reduction_count[iname] = (
                    iname_to_reduction_count.get(iname, 0) + 1)
            if not expr.allow_simultaneous:
                iname_to_nonsimultaneous_reduction_count[iname] = (
                        iname_to_nonsimultaneous_reduction_count.get(iname, 0) + 1)

        return expr

    from loopy.symbolic import ReductionCallbackMapper
    cb_mapper = ReductionCallbackMapper(map_reduction)

    for insn in kernel.instructions:
        insn.with_transformed_expressions(cb_mapper)

    for iname, count in six.iteritems(iname_to_reduction_count):
        nonsimul_count = iname_to_nonsimultaneous_reduction_count.get(iname, 0)

        if nonsimul_count and count > 1:
            raise LoopyError("iname '%s' used in more than one reduction. "
                    "(%d of them, to be precise.) "
                    "Since this usage can easily cause loop scheduling "
                    "problems, this is prohibited by default. "
                    "Use loopy.make_reduction_inames_unique() to fix this. "
                    "If you are sure that this is OK, write the reduction "
                    "as 'simul_reduce(...)' instead of 'reduce(...)'"
                    % (iname, count))

# }}}


# {{{ decide temporary scope

def _get_compute_inames_tagged(kernel, insn, tag_base):
    return set(iname
            for iname in kernel.insn_inames(insn.id)
            if isinstance(kernel.iname_to_tag.get(iname), tag_base))


def _get_assignee_inames_tagged(kernel, insn, tag_base, tv_names):
    return set(iname
            for aname, adeps in zip(
                insn.assignee_var_names(),
                insn.assignee_subscript_deps())
            for iname in adeps & kernel.all_inames()
            if aname in tv_names
            if isinstance(kernel.iname_to_tag.get(iname), tag_base))


def find_temporary_scope(kernel):
    logger.debug("%s: find temporary scope" % kernel.name)

    new_temp_vars = {}
    from loopy.kernel.data import (LocalIndexTagBase, GroupIndexTag,
            temp_var_scope)
    import loopy as lp

    writers = kernel.writer_map()

    base_storage_to_aliases = {}

    kernel_var_names = kernel.all_variable_names(include_temp_storage=False)

    for temp_var in six.itervalues(kernel.temporary_variables):
        if temp_var.base_storage is not None:
            # no nesting allowed
            if temp_var.base_storage in kernel_var_names:
                raise LoopyError("base_storage for temporary '%s' is '%s', "
                        "which is an existing variable name"
                        % (temp_var.name, temp_var.base_storage))

            base_storage_to_aliases.setdefault(
                    temp_var.base_storage, []).append(temp_var.name)

    for temp_var in six.itervalues(kernel.temporary_variables):
        # Only fill out for variables that do not yet know if they're
        # local. (I.e. those generated by implicit temporary generation.)

        if temp_var.scope is not lp.auto:
            new_temp_vars[temp_var.name] = temp_var
            continue

        tv_names = (frozenset([temp_var.name])
                | frozenset(base_storage_to_aliases.get(temp_var.base_storage, [])))
        my_writers = writers.get(temp_var.name, frozenset())
        if temp_var.base_storage is not None:
            for alias in base_storage_to_aliases.get(temp_var.base_storage, []):
                my_writers = my_writers | writers.get(alias, frozenset())

        desired_scope_per_insn = []
        for insn_id in my_writers:
            insn = kernel.id_to_insn[insn_id]

            # A write race will emerge if:
            #
            # - the variable is local
            #   and
            # - the instruction is run across more inames (locally) parallel
            #   than are reflected in the assignee indices.

            locparallel_compute_inames = _get_compute_inames_tagged(
                    kernel, insn, LocalIndexTagBase)

            locparallel_assignee_inames = _get_assignee_inames_tagged(
                    kernel, insn, LocalIndexTagBase, tv_names)

            grpparallel_compute_inames = _get_compute_inames_tagged(
                    kernel, insn, GroupIndexTag)

            grpparallel_assignee_inames = _get_assignee_inames_tagged(
                    kernel, insn, GroupIndexTag, temp_var.name)

            assert locparallel_assignee_inames <= locparallel_compute_inames
            assert grpparallel_assignee_inames <= grpparallel_compute_inames

            desired_scope = temp_var_scope.PRIVATE
            for iname_descr, scope_descr, apin, cpin, scope in [
                    ("local", "local", locparallel_assignee_inames,
                        locparallel_compute_inames, temp_var_scope.LOCAL),
                    ("group", "global", grpparallel_assignee_inames,
                        grpparallel_compute_inames, temp_var_scope.GLOBAL),
                    ]:

                if (apin != cpin and bool(apin)):
                    warn_with_kernel(
                            kernel,
                            "write_race_%s(%s)" % (scope_descr, insn_id),
                            "instruction '%s' looks invalid: "
                            "it assigns to indices based on %s IDs, but "
                            "its temporary '%s' cannot be made %s because "
                            "a write race across the iname(s) '%s' would emerge. "
                            "(Do you need to add an extra iname to your prefetch?)"
                            % (insn_id, iname_descr, temp_var.name, scope_descr,
                                ", ".join(cpin - apin)),
                            WriteRaceConditionWarning)

                if (apin == cpin

                        # doesn't want to be in this scope if there aren't any
                        # parallel inames of that kind:
                        and bool(cpin)):
                    desired_scope = max(desired_scope, scope)

            desired_scope_per_insn.append(desired_scope)

        if not desired_scope_per_insn:
            if temp_var.initializer is None:
                warn_with_kernel(kernel, "temp_to_write(%s)" % temp_var.name,
                        "temporary variable '%s' never written, eliminating"
                        % temp_var.name, LoopyAdvisory)
            else:
                raise LoopyError("temporary variable '%s': never written, "
                        "cannot automatically determine scope"
                        % temp_var.name)

            continue

        overall_scope = max(desired_scope_per_insn)

        from pytools import all
        if not all(iscope == overall_scope for iscope in desired_scope_per_insn):
            raise LoopyError("not all instructions agree on the "
                    "the desired scope (private/local/global) of  the "
                    "temporary '%s'" % temp_var.name)

        new_temp_vars[temp_var.name] = temp_var.copy(scope=overall_scope)

    return kernel.copy(temporary_variables=new_temp_vars)

# }}}


# {{{ rewrite reduction to imperative form


# {{{ utils (not stateful)

from collections import namedtuple


_InameClassification = namedtuple("_InameClassifiction",
                                  "sequential, local_parallel, nonlocal_parallel")


def _classify_reduction_inames(kernel, inames):
    sequential = []
    local_par = []
    nonlocal_par = []

    from loopy.kernel.data import (
            LocalIndexTagBase, UnrolledIlpTag, UnrollTag, VectorizeTag,
            ParallelTag)

    for iname in inames:
        iname_tag = kernel.iname_to_tag.get(iname)

        if isinstance(iname_tag, (UnrollTag, UnrolledIlpTag)):
            # These are nominally parallel, but we can live with
            # them as sequential.
            sequential.append(iname)

        elif isinstance(iname_tag, LocalIndexTagBase):
            local_par.append(iname)

        elif isinstance(iname_tag, (ParallelTag, VectorizeTag)):
            nonlocal_par.append(iname)

        else:
            sequential.append(iname)

    return _InameClassification(
            tuple(sequential), tuple(local_par), tuple(nonlocal_par))


def _add_params_to_domain(domain, param_names):
    dim_type = isl.dim_type
    nparams_orig = domain.dim(dim_type.param)
    domain = domain.add_dims(dim_type.param, len(param_names))

    for param_idx, param_name in enumerate(param_names):
        domain = domain.set_dim_name(
                dim_type.param, param_idx + nparams_orig, param_name)

    return domain


def _move_set_to_param_dims_except(domain, except_dims):
    dim_type = isl.dim_type

    iname_idx = 0
    for iname in domain.get_var_names(dim_type.set):
        if iname not in except_dims:
            domain = domain.move_dims(
                    dim_type.param, 0,
                    dim_type.set, iname_idx, 1)
            iname_idx -= 1
        iname_idx += 1

    return domain


def _domain_depends_on_given_set_dims(domain, set_dim_names):
    set_dim_names = frozenset(set_dim_names)

    return any(
            set_dim_names & set(constr.get_coefficients_by_name())
            for constr in domain.get_constraints())


def _check_reduction_is_triangular(kernel, expr, scan_param):
    """Check whether the reduction within `expr` with scan parameters described by
    the structure `scan_param` is triangular. This attempts to verify that the
    domain for the scan and sweep inames is as follows:

    [params] -> {
        [other inames..., scan_iname, sweep_iname]:
            (sweep_min_value
                <= sweep_iname
                <= sweep_max_value)
            and
            (scan_min_value
                <= scan_iname
                <= stride * (sweep_iname - sweep_min_value) + scan_min_value)
            and
            (irrelevant constraints)
    }
    """

    orig_domain = kernel.get_inames_domain(
            frozenset((scan_param.sweep_iname, scan_param.scan_iname)))

    sweep_iname = scan_param.sweep_iname
    scan_iname = scan_param.scan_iname
    affs = isl.affs_from_space(orig_domain.space)

    sweep_lower_bound = isl.align_spaces(
            scan_param.sweep_lower_bound,
            affs[0],
            across_dim_types=True)

    sweep_upper_bound = isl.align_spaces(
            scan_param.sweep_upper_bound,
            affs[0],
            across_dim_types=True)

    scan_lower_bound = isl.align_spaces(
            scan_param.scan_lower_bound,
            affs[0],
            across_dim_types=True)

    from itertools import product

    for (sweep_lb_domain, sweep_lb_aff), \
        (sweep_ub_domain, sweep_ub_aff), \
        (scan_lb_domain, scan_lb_aff) in \
            product(sweep_lower_bound.get_pieces(),
                    sweep_upper_bound.get_pieces(),
                    scan_lower_bound.get_pieces()):

        # Assumptions inherited from the domains of the pwaffs
        assumptions = sweep_lb_domain & sweep_ub_domain & scan_lb_domain

        # Sweep iname constraints
        hyp_domain = affs[sweep_iname].ge_set(sweep_lb_aff)
        hyp_domain &= affs[sweep_iname].le_set(sweep_ub_aff)

        # Scan iname constraints
        hyp_domain &= affs[scan_iname].ge_set(scan_lb_aff)
        hyp_domain &= affs[scan_iname].le_set(
                scan_param.stride * (affs[sweep_iname] - sweep_lb_aff)
                + scan_lb_aff)

        hyp_domain, = (hyp_domain & assumptions).get_basic_sets()
        test_domain, = (orig_domain & assumptions).get_basic_sets()

        hyp_gist_against_test = hyp_domain.gist(test_domain)
        if _domain_depends_on_given_set_dims(hyp_gist_against_test,
                (sweep_iname, scan_iname)):
            return False, (
                    "gist of hypothesis against test domain "
                    "has sweep or scan dependent constraints: '%s'"
                    % hyp_gist_against_test)

        test_gist_against_hyp = test_domain.gist(hyp_domain)
        if _domain_depends_on_given_set_dims(test_gist_against_hyp,
                (sweep_iname, scan_iname)):
            return False, (
                   "gist of test against hypothesis domain "
                   "has sweep or scan dependent constraint: '%s'"
                   % test_gist_against_hyp)

    return True, "ok"


_ScanCandidateParameters = namedtuple(
        "_ScanCandidateParameters",
        "sweep_iname, scan_iname, sweep_lower_bound, "
        "sweep_upper_bound, scan_lower_bound, stride")


def _try_infer_scan_candidate_from_expr(
        kernel, expr, within_inames, sweep_iname=None):
    """Analyze `expr` and determine if it can be implemented as a scan.
    """
    from loopy.symbolic import Reduction
    assert isinstance(expr, Reduction)

    if len(expr.inames) != 1:
        raise ValueError(
                "Multiple inames in reduction: '%s'" % (", ".join(expr.inames),))

    scan_iname, = expr.inames

    from loopy.kernel.tools import DomainChanger
    dchg = DomainChanger(kernel, (scan_iname,))
    domain = dchg.get_original_domain()

    if sweep_iname is None:
        try:
            sweep_iname = _try_infer_sweep_iname(
                    domain, scan_iname, kernel.all_inames())
        except ValueError as v:
            raise ValueError(
                    "Couldn't determine a sweep iname for the scan "
                    "expression '%s': %s" % (expr, v))

    try:
        sweep_lower_bound, sweep_upper_bound, scan_lower_bound = (
                _try_infer_scan_and_sweep_bounds(
                    kernel, scan_iname, sweep_iname, within_inames))
    except ValueError as v:
        raise ValueError(
                "Couldn't determine bounds for the scan with expression '%s' "
                "(sweep iname: '%s', scan iname: '%s'): %s"
                % (expr, sweep_iname, scan_iname, v))

    try:
        stride = _try_infer_scan_stride(
                kernel, scan_iname, sweep_iname, sweep_lower_bound)
    except ValueError as v:
        raise ValueError(
                "Couldn't determine a scan stride for the scan with expression '%s' "
                "(sweep iname: '%s', scan iname: '%s'): %s"
                % (expr, sweep_iname, scan_iname, v))

    return _ScanCandidateParameters(sweep_iname, scan_iname, sweep_lower_bound,
            sweep_upper_bound, scan_lower_bound, stride)


def _try_infer_sweep_iname(domain, scan_iname, candidate_inames):
    """The sweep iname is the outer iname which guides the scan.

    E.g. for a domain of {[i,j]: 0<=i<n and 0<=j<=i}, i is the sweep iname.
    """
    constrs = domain.get_constraints()
    sweep_iname_candidate = None

    for constr in constrs:
        candidate_vars = set([
                var for var in constr.get_var_dict()
                if var in candidate_inames])

        # Irrelevant constraint - skip
        if scan_iname not in candidate_vars:
            continue

        # No additional inames - skip
        if len(candidate_vars) == 1:
            continue

        candidate_vars.remove(scan_iname)

        # Depends on more than one iname - error
        if len(candidate_vars) > 1:
            raise ValueError(
                    "More than one sweep iname candidate for scan iname '%s' found "
                    "(via constraint '%s')" % (scan_iname, constr))

        next_candidate = candidate_vars.pop()

        if sweep_iname_candidate is None:
            sweep_iname_candidate = next_candidate
            defining_constraint = constr
        else:
            # Check next_candidate consistency
            if sweep_iname_candidate != next_candidate:
                raise ValueError(
                        "More than one sweep iname candidate for scan iname '%s' "
                        "found (via constraints '%s', '%s')" %
                        (scan_iname, defining_constraint, constr))

    if sweep_iname_candidate is None:
        raise ValueError(
                "Couldn't find any sweep iname candidates for "
                "scan iname '%s'" % scan_iname)

    return sweep_iname_candidate


def _try_infer_scan_and_sweep_bounds(kernel, scan_iname, sweep_iname, within_inames):
    domain = kernel.get_inames_domain(frozenset((sweep_iname, scan_iname)))
    domain = _move_set_to_param_dims_except(domain, (sweep_iname, scan_iname))

    var_dict = domain.get_var_dict()
    sweep_idx = var_dict[sweep_iname][1]
    scan_idx = var_dict[scan_iname][1]

    domain = domain.project_out_except(
            within_inames | kernel.non_iname_variable_names(), (isl.dim_type.param,))

    try:
        with isl.SuppressedWarnings(domain.get_ctx()):
            sweep_lower_bound = domain.dim_min(sweep_idx)
            sweep_upper_bound = domain.dim_max(sweep_idx)
            scan_lower_bound = domain.dim_min(scan_idx)
    except isl.Error as e:
        raise ValueError("isl error: %s" % e)

    return (sweep_lower_bound, sweep_upper_bound, scan_lower_bound)


def _try_infer_scan_stride(kernel, scan_iname, sweep_iname, sweep_lower_bound):
    """The stride is the number of steps the scan iname takes per iteration
    of the sweep iname. This is allowed to be an integer constant.

    E.g. for a domain of {[i,j]: 0<=i<n and 0<=j<=6*i}, the stride is 6.
    """
    dim_type = isl.dim_type

    domain = kernel.get_inames_domain(frozenset([sweep_iname, scan_iname]))
    domain_with_sweep_param = _move_set_to_param_dims_except(domain, (scan_iname,))

    domain_with_sweep_param = domain_with_sweep_param.project_out_except(
            (sweep_iname, scan_iname), (dim_type.set, dim_type.param))

    scan_iname_idx = domain_with_sweep_param.find_dim_by_name(
            dim_type.set, scan_iname)

    # Should be equal to k * sweep_iname, where k is the stride.

    try:
        with isl.SuppressedWarnings(domain_with_sweep_param.get_ctx()):
            scan_iname_range = (
                    domain_with_sweep_param.dim_max(scan_iname_idx)
                    - domain_with_sweep_param.dim_min(scan_iname_idx)
                    ).gist(domain_with_sweep_param.params())
    except isl.Error as e:
        raise ValueError("isl error: '%s'" % e)

    scan_iname_pieces = scan_iname_range.get_pieces()

    if len(scan_iname_pieces) > 1:
        raise ValueError("range in multiple pieces: %s" % scan_iname_range)
    elif len(scan_iname_pieces) == 0:
        raise ValueError("empty range found for iname '%s'" % scan_iname)

    scan_iname_constr, scan_iname_aff = scan_iname_pieces[0]

    if not scan_iname_constr.plain_is_universe():
        raise ValueError("found constraints: %s" % scan_iname_constr)

    if scan_iname_aff.dim(dim_type.div):
        raise ValueError("aff has div: %s" % scan_iname_aff)

    coeffs = scan_iname_aff.get_coefficients_by_name(dim_type.param)

    if len(coeffs) == 0:
        try:
            scan_iname_aff.get_constant_val()
        except:
            raise ValueError("range for aff isn't constant: '%s'" % scan_iname_aff)

        # If this point is reached we're assuming the domain is of the form
        # {[i,j]: i=0 and j=0}, so the stride is technically 1 - any value
        # this function returns will be verified later by
        # _check_reduction_is_triangular().
        return 1

    if sweep_iname not in coeffs:
        raise ValueError("didn't find sweep iname in coeffs: %s" % sweep_iname)

    stride = coeffs[sweep_iname]

    if not stride.is_int():
        raise ValueError("stride not an integer: %s" % stride)

    if not stride.is_pos():
        raise ValueError("stride not positive: %s" % stride)

    return stride.to_python()


def _get_domain_with_iname_as_param(domain, iname):
    dim_type = isl.dim_type

    if domain.find_dim_by_name(dim_type.param, iname) >= 0:
        return domain

    iname_idx = domain.find_dim_by_name(dim_type.set, iname)

    assert iname_idx >= 0, (iname, domain)

    return domain.move_dims(
        dim_type.param, domain.dim(dim_type.param),
        dim_type.set, iname_idx, 1)


def _create_domain_for_sweep_tracking(orig_domain,
        tracking_iname, sweep_iname, sweep_min_value, scan_min_value, stride):
    dim_type = isl.dim_type

    subd = isl.BasicSet.universe(orig_domain.params().space)

    # Add tracking_iname and sweep iname.

    subd = _add_params_to_domain(subd, (sweep_iname, tracking_iname))

    # Here we realize the domain:
    #
    # [..., i] -> {
    #  [j]: 0 <= j - l
    #       and
    #       j - l <= k * (i - m)
    #       and
    #       k * (i - m - 1) < j - l }
    # where
    #   * i is the sweep iname
    #   * j is the tracking iname
    #   * k is the stride for the scan
    #   * l is the lower bound for the scan
    #   * m is the lower bound for the sweep iname
    #
    affs = isl.affs_from_space(subd.space)

    subd &= (affs[tracking_iname] - scan_min_value).ge_set(affs[0])
    subd &= (affs[tracking_iname] - scan_min_value)\
            .le_set(stride * (affs[sweep_iname] - sweep_min_value))
    subd &= (affs[tracking_iname] - scan_min_value)\
            .gt_set(stride * (affs[sweep_iname] - sweep_min_value - 1))

    # Move tracking_iname into a set dim (NOT sweep iname).
    subd = subd.move_dims(
            dim_type.set, 0,
            dim_type.param, subd.dim(dim_type.param) - 1, 1)

    # Simplify (maybe).
    orig_domain_with_sweep_param = (
            _get_domain_with_iname_as_param(orig_domain, sweep_iname))
    subd = subd.gist_params(orig_domain_with_sweep_param.params())

    subd, = subd.get_basic_sets()

    return subd


def _hackily_ensure_multi_assignment_return_values_are_scoped_private(kernel):
    """
    Multi assignment function calls are currently lowered into OpenCL so that
    the function call::

       a, b = segmented_sum(x, y, z, w)

    becomes::

       a = segmented_sum_mangled(x, y, z, w, &b).

    For OpenCL, the scope of "b" is significant, and the preamble generation
    currently assumes the scope is always private. This function forces that to
    be the case by introducing temporary assignments into the kernel.
    """

    insn_id_gen = kernel.get_instruction_id_generator()
    var_name_gen = kernel.get_var_name_generator()

    new_or_updated_instructions = {}
    new_temporaries = {}

    dep_map = dict(
            (insn.id, insn.depends_on) for insn in kernel.instructions)

    inverse_dep_map = dict((insn.id, set()) for insn in kernel.instructions)

    import six
    for insn_id, deps in six.iteritems(dep_map):
        for dep in deps:
            inverse_dep_map[dep].add(insn_id)

    del dep_map

    # {{{ utils

    def _add_to_no_sync_with(insn_id, new_no_sync_with_params):
        insn = kernel.id_to_insn.get(insn_id)
        insn = new_or_updated_instructions.get(insn_id, insn)
        new_or_updated_instructions[insn_id] = (
                insn.copy(
                    no_sync_with=(
                        insn.no_sync_with | frozenset(new_no_sync_with_params))))

    def _add_to_depends_on(insn_id, new_depends_on_params):
        insn = kernel.id_to_insn.get(insn_id)
        insn = new_or_updated_instructions.get(insn_id, insn)
        new_or_updated_instructions[insn_id] = (
                insn.copy(
                    depends_on=insn.depends_on | frozenset(new_depends_on_params)))

    # }}}

    from loopy.kernel.instruction import CallInstruction
    for insn in kernel.instructions:
        if not isinstance(insn, CallInstruction):
            continue

        if len(insn.assignees) <= 1:
            continue

        assignees = insn.assignees
        assignee_var_names = insn.assignee_var_names()

        new_assignees = [assignees[0]]
        newly_added_assignments_ids = set()
        needs_replacement = False

        last_added_insn_id = insn.id

        from loopy.kernel.data import temp_var_scope, TemporaryVariable

        FIRST_POINTER_ASSIGNEE_IDX = 1  # noqa

        for assignee_nr, assignee_var_name, assignee in zip(
                range(FIRST_POINTER_ASSIGNEE_IDX, len(assignees)),
                assignee_var_names[FIRST_POINTER_ASSIGNEE_IDX:],
                assignees[FIRST_POINTER_ASSIGNEE_IDX:]):

            if (
                    assignee_var_name in kernel.temporary_variables
                    and
                    (kernel.temporary_variables[assignee_var_name].scope
                         == temp_var_scope.PRIVATE)):
                new_assignees.append(assignee)
                continue

            needs_replacement = True

            # {{{ generate a new assignent instruction

            new_assignee_name = var_name_gen(
                    "{insn_id}_retval_{assignee_nr}"
                    .format(insn_id=insn.id, assignee_nr=assignee_nr))

            new_assignment_id = insn_id_gen(
                    "{insn_id}_assign_retval_{assignee_nr}"
                    .format(insn_id=insn.id, assignee_nr=assignee_nr))

            newly_added_assignments_ids.add(new_assignment_id)

            import loopy as lp
            new_temporaries[new_assignee_name] = (
                    TemporaryVariable(
                        name=new_assignee_name,
                        dtype=lp.auto,
                        scope=temp_var_scope.PRIVATE))

            from pymbolic import var
            new_assignee = var(new_assignee_name)
            new_assignees.append(new_assignee)

            new_or_updated_instructions[new_assignment_id] = (
                    make_assignment(
                        assignees=(assignee,),
                        expression=new_assignee,
                        id=new_assignment_id,
                        depends_on=frozenset([last_added_insn_id]),
                        depends_on_is_final=True,
                        no_sync_with=(
                            insn.no_sync_with | frozenset([(insn.id, "any")])),
                        predicates=insn.predicates,
                        within_inames=insn.within_inames))

            last_added_insn_id = new_assignment_id

            # }}}

        if not needs_replacement:
            continue

        # {{{ update originating instruction

        orig_insn = new_or_updated_instructions.get(insn.id, insn)

        new_or_updated_instructions[insn.id] = (
                orig_insn.copy(assignees=tuple(new_assignees)))

        _add_to_no_sync_with(insn.id,
                [(id, "any") for id in newly_added_assignments_ids])

        # }}}

        # {{{ squash spurious memory dependencies amongst new assignments

        for new_insn_id in newly_added_assignments_ids:
            _add_to_no_sync_with(new_insn_id,
                    [(id, "any")
                     for id in newly_added_assignments_ids
                     if id != new_insn_id])

        # }}}

        # {{{ update instructions that depend on the originating instruction

        for inverse_dep in inverse_dep_map[insn.id]:
            _add_to_depends_on(inverse_dep, newly_added_assignments_ids)

            for insn_id, scope in (
                    new_or_updated_instructions[inverse_dep].no_sync_with):
                if insn_id == insn.id:
                    _add_to_no_sync_with(
                            inverse_dep,
                            [(id, scope) for id in newly_added_assignments_ids])

        # }}}

    new_temporary_variables = kernel.temporary_variables.copy()
    new_temporary_variables.update(new_temporaries)

    new_instructions = (
            list(new_or_updated_instructions.values())
            + list(insn
                for insn in kernel.instructions
                if insn.id not in new_or_updated_instructions))

    return kernel.copy(temporary_variables=new_temporary_variables,
                       instructions=new_instructions)


def _insert_subdomain_into_domain_tree(kernel, domains, subdomain):
    # Intersect with inames, because we could have captured some kernel params
    # in here too...
    dependent_inames = (
            frozenset(subdomain.get_var_names(isl.dim_type.param))
            & kernel.all_inames())
    idx, = kernel.get_leaf_domain_indices(dependent_inames)
    domains.insert(idx + 1, subdomain)

# }}}


def realize_reduction(kernel, insn_id_filter=None, unknown_types_ok=True,
                      automagic_scans_ok=False, force_scan=False,
                      force_outer_iname_for_scan=None):
    """Rewrites reductions into their imperative form. With *insn_id_filter*
    specified, operate only on the instruction with an instruction id matching
    *insn_id_filter*.

    If *insn_id_filter* is given, only the outermost level of reductions will be
    expanded, inner reductions will be left alone (because they end up in a new
    instruction with a different ID, which doesn't match the filter).

    If *insn_id_filter* is not given, all reductions in all instructions will
    be realized.

    If *automagic_scans_ok*, this function will attempt to rewrite triangular
    reductions as scans automatically.

    If *force_scan* is *True*, this function will attempt to rewrite *all*
    candidate reductions as scans and raise an error if this is not possible
    (this is most useful combined with *insn_id_filter*).

    If *force_outer_iname_for_scan* is not *None*, this function will attempt
    to realize candidate reductions as scans using the specified iname as the
    outer (sweep) iname.
    """

    logger.debug("%s: realize reduction" % kernel.name)

    new_insns = []
    new_iname_tags = {}

    insn_id_gen = kernel.get_instruction_id_generator()

    var_name_gen = kernel.get_var_name_generator()
    new_temporary_variables = kernel.temporary_variables.copy()
    inames_added_for_scan = set()
    inames_to_remove = set()

    # {{{ helpers

    def _strip_if_scalar(reference, val):
        if len(reference) == 1:
            return val[0]
        else:
            return val

    def preprocess_scan_arguments(
                insn, expr, nresults, scan_iname, track_iname,
                newly_generated_insn_id_set):
        """Does iname substitution within scan arguments and returns a set of values
        suitable to be passed to the binary op. Returns a tuple."""

        if nresults > 1:
            inner_expr = expr

            # In the case of a multi-argument scan, we need a name for each of
            # the arguments in order to pass them to the binary op - so we expand
            # items that are not "plain" tuples here.
            if not isinstance(inner_expr, tuple):
                get_args_insn_id = insn_id_gen(
                        "%s_%s_get" % (insn.id, "_".join(expr.inames)))

                inner_expr = expand_inner_reduction(
                        id=get_args_insn_id,
                        expr=inner_expr,
                        nresults=nresults,
                        depends_on=insn.depends_on,
                        within_inames=insn.within_inames | expr.inames,
                        within_inames_is_final=insn.within_inames_is_final)

                newly_generated_insn_id_set.add(get_args_insn_id)

            updated_inner_exprs = tuple(
                    replace_var_within_expr(sub_expr, scan_iname, track_iname)
                    for sub_expr in inner_expr)
        else:
            updated_inner_exprs = (
                    replace_var_within_expr(expr, scan_iname, track_iname),)

        return updated_inner_exprs

    def expand_inner_reduction(id, expr, nresults, depends_on, within_inames,
            within_inames_is_final):
        # FIXME: use make_temporaries
        from pymbolic.primitives import Call
        from loopy.symbolic import Reduction
        assert isinstance(expr, (Call, Reduction))

        temp_var_names = [
                var_name_gen(id + "_arg" + str(i))
                for i in range(nresults)]

        for name in temp_var_names:
            from loopy.kernel.data import TemporaryVariable, temp_var_scope
            new_temporary_variables[name] = TemporaryVariable(
                    name=name,
                    shape=(),
                    dtype=lp.auto,
                    scope=temp_var_scope.PRIVATE)

        from pymbolic import var
        temp_vars = tuple(var(n) for n in temp_var_names)

        call_insn = make_assignment(
                id=id,
                assignees=temp_vars,
                expression=expr,
                depends_on=depends_on,
                within_inames=within_inames,
                within_inames_is_final=within_inames_is_final)

        generated_insns.append(call_insn)

        return temp_vars

    # }}}

    # {{{ sequential

    def map_reduction_seq(expr, rec, nresults, arg_dtypes,
            reduction_dtypes):
        outer_insn_inames = temp_kernel.insn_inames(insn)

        from loopy.kernel.data import temp_var_scope
        acc_var_names = make_temporaries(
                name_based_on="acc_"+"_".join(expr.inames),
                nvars=nresults,
                shape=(),
                dtypes=reduction_dtypes,
                scope=temp_var_scope.PRIVATE)

        init_insn_depends_on = frozenset()

        global_barrier = lp.find_most_recent_global_barrier(temp_kernel, insn.id)

        if global_barrier is not None:
            init_insn_depends_on |= frozenset([global_barrier])

        from pymbolic import var
        acc_vars = tuple(var(n) for n in acc_var_names)

        init_id = insn_id_gen(
                "%s_%s_init" % (insn.id, "_".join(expr.inames)))

        init_insn = make_assignment(
                id=init_id,
                assignees=acc_vars,
                within_inames=outer_insn_inames - frozenset(expr.inames),
                within_inames_is_final=insn.within_inames_is_final,
                depends_on=init_insn_depends_on,
                expression=expr.operation.neutral_element(*arg_dtypes))

        generated_insns.append(init_insn)

        update_id = insn_id_gen(
                based_on="%s_%s_update" % (insn.id, "_".join(expr.inames)))

        update_insn_iname_deps = temp_kernel.insn_inames(insn) | set(expr.inames)
        if insn.within_inames_is_final:
            update_insn_iname_deps = insn.within_inames | set(expr.inames)

        reduction_insn_depends_on = set([init_id])

        # In the case of a multi-argument reduction, we need a name for each of
        # the arguments in order to pass them to the binary op - so we expand
        # items that are not "plain" tuples here.
        if nresults > 1 and not isinstance(expr.expr, tuple):
            get_args_insn_id = insn_id_gen(
                    "%s_%s_get" % (insn.id, "_".join(expr.inames)))

            reduction_expr = expand_inner_reduction(
                    id=get_args_insn_id,
                    expr=expr.expr,
                    nresults=nresults,
                    depends_on=insn.depends_on,
                    within_inames=update_insn_iname_deps,
                    within_inames_is_final=insn.within_inames_is_final)

            reduction_insn_depends_on.add(get_args_insn_id)
        else:
            reduction_expr = expr.expr

        reduction_insn = make_assignment(
                id=update_id,
                assignees=acc_vars,
                expression=expr.operation(
                    arg_dtypes,
                    _strip_if_scalar(acc_vars, acc_vars),
                    reduction_expr),
                depends_on=frozenset(reduction_insn_depends_on) | insn.depends_on,
                within_inames=update_insn_iname_deps,
                within_inames_is_final=insn.within_inames_is_final)

        generated_insns.append(reduction_insn)

        new_insn_add_depends_on.add(reduction_insn.id)

        if nresults == 1:
            assert len(acc_vars) == 1
            return acc_vars[0]
        else:
            return acc_vars

    # }}}

    # {{{ local-parallel

    def _get_int_iname_size(iname):
        from loopy.isl_helpers import static_max_of_pw_aff
        from loopy.symbolic import pw_aff_to_expr
        size = pw_aff_to_expr(
                static_max_of_pw_aff(
                    kernel.get_iname_bounds(iname).size,
                    constants_only=True))
        assert isinstance(size, six.integer_types)
        return size

    def _make_slab_set(iname, size):
        v = isl.make_zero_and_vars([iname])
        bs, = (
                v[0].le_set(v[iname])
                &
                v[iname].lt_set(v[0] + size)).get_basic_sets()
        return bs

    def _make_slab_set_from_range(iname, lbound, ubound):
        v = isl.make_zero_and_vars([iname])
        bs, = (
                v[iname].ge_set(v[0] + lbound)
                &
                v[iname].lt_set(v[0] + ubound)).get_basic_sets()
        return bs

    def map_reduction_local(expr, rec, nresults, arg_dtypes,
            reduction_dtypes):
        red_iname, = expr.inames

        size = _get_int_iname_size(red_iname)

        outer_insn_inames = temp_kernel.insn_inames(insn)

        from loopy.kernel.data import LocalIndexTagBase
        outer_local_inames = tuple(
                oiname
                for oiname in outer_insn_inames
                if isinstance(
                    kernel.iname_to_tag.get(oiname),
                    LocalIndexTagBase))

        from pymbolic import var
        outer_local_iname_vars = tuple(
                var(oiname) for oiname in outer_local_inames)

        outer_local_iname_sizes = tuple(
                _get_int_iname_size(oiname)
                for oiname in outer_local_inames)

        from loopy.kernel.data import temp_var_scope

        neutral_var_names = make_temporaries(
                name_based_on="neutral_"+red_iname,
                nvars=nresults,
                shape=(),
                dtypes=reduction_dtypes,
                scope=temp_var_scope.PRIVATE)

        acc_var_names = make_temporaries(
                name_based_on="acc_"+red_iname,
                nvars=nresults,
                shape=outer_local_iname_sizes + (size,),
                dtypes=reduction_dtypes,
                scope=temp_var_scope.LOCAL)

        acc_vars = tuple(var(n) for n in acc_var_names)

        # {{{ add separate iname to carry out the reduction

        # Doing this sheds any odd conditionals that may be active
        # on our red_iname.

        base_exec_iname = var_name_gen("red_"+red_iname)
        domains.append(_make_slab_set(base_exec_iname, size))
        new_iname_tags[base_exec_iname] = kernel.iname_to_tag[red_iname]

        # }}}

        base_iname_deps = outer_insn_inames - frozenset(expr.inames)

        neutral = expr.operation.neutral_element(*arg_dtypes)
        init_id = insn_id_gen("%s_%s_init" % (insn.id, red_iname))
        init_insn = make_assignment(
                id=init_id,
                assignees=tuple(
                    acc_var[outer_local_iname_vars + (var(base_exec_iname),)]
                    for acc_var in acc_vars),
                expression=neutral,
                within_inames=base_iname_deps | frozenset([base_exec_iname]),
                within_inames_is_final=insn.within_inames_is_final,
                depends_on=frozenset())
        generated_insns.append(init_insn)

        init_neutral_id = insn_id_gen("%s_%s_init_neutral" % (insn.id, red_iname))
        init_neutral_insn = make_assignment(
                id=init_neutral_id,
                assignees=tuple(var(nvn) for nvn in neutral_var_names),
                expression=neutral,
                within_inames=base_iname_deps | frozenset([base_exec_iname]),
                within_inames_is_final=insn.within_inames_is_final,
                depends_on=frozenset())
        generated_insns.append(init_neutral_insn)

        transfer_depends_on = set([init_neutral_id, init_id])

        # In the case of a multi-argument reduction, we need a name for each of
        # the arguments in order to pass them to the binary op - so we expand
        # items that are not "plain" tuples here.
        if nresults > 1 and not isinstance(expr.expr, tuple):
            get_args_insn_id = insn_id_gen(
                    "%s_%s_get" % (insn.id, red_iname))

            reduction_expr = expand_inner_reduction(
                    id=get_args_insn_id,
                    expr=expr.expr,
                    nresults=nresults,
                    depends_on=insn.depends_on,
                    within_inames=(
                        (outer_insn_inames - frozenset(expr.inames))
                        | frozenset([red_iname])),
                    within_inames_is_final=insn.within_inames_is_final)

            transfer_depends_on.add(get_args_insn_id)
        else:
            reduction_expr = expr.expr

        transfer_id = insn_id_gen("%s_%s_transfer" % (insn.id, red_iname))
        transfer_insn = make_assignment(
                id=transfer_id,
                assignees=tuple(
                    acc_var[outer_local_iname_vars + (var(red_iname),)]
                    for acc_var in acc_vars),
                expression=expr.operation(
                    arg_dtypes,
                    _strip_if_scalar(
                        neutral_var_names,
                        tuple(var(nvn) for nvn in neutral_var_names)),
                    reduction_expr),
                within_inames=(
                    (outer_insn_inames - frozenset(expr.inames))
                    | frozenset([red_iname])),
                within_inames_is_final=insn.within_inames_is_final,
                depends_on=frozenset([init_id, init_neutral_id]) | insn.depends_on,
                no_sync_with=frozenset([(init_id, "any")]))
        generated_insns.append(transfer_insn)

        cur_size = 1
        while cur_size < size:
            cur_size *= 2

        prev_id = transfer_id
        bound = size

        istage = 0
        while cur_size > 1:

            new_size = cur_size // 2
            assert new_size * 2 == cur_size

            stage_exec_iname = var_name_gen("red_%s_s%d" % (red_iname, istage))
            domains.append(_make_slab_set(stage_exec_iname, bound-new_size))
            new_iname_tags[stage_exec_iname] = kernel.iname_to_tag[red_iname]

            stage_id = insn_id_gen("red_%s_stage_%d" % (red_iname, istage))
            stage_insn = make_assignment(
                    id=stage_id,
                    assignees=tuple(
                        acc_var[outer_local_iname_vars + (var(stage_exec_iname),)]
                        for acc_var in acc_vars),
                    expression=expr.operation(
                        arg_dtypes,
                        _strip_if_scalar(acc_vars, tuple(
                            acc_var[
                                outer_local_iname_vars + (var(stage_exec_iname),)]
                            for acc_var in acc_vars)),
                        _strip_if_scalar(acc_vars, tuple(
                            acc_var[
                                outer_local_iname_vars + (
                                    var(stage_exec_iname) + new_size,)]
                            for acc_var in acc_vars))),
                    within_inames=(
                        base_iname_deps | frozenset([stage_exec_iname])),
                    within_inames_is_final=insn.within_inames_is_final,
                    depends_on=frozenset([prev_id]),
                    )

            generated_insns.append(stage_insn)
            prev_id = stage_id

            cur_size = new_size
            bound = cur_size
            istage += 1

        new_insn_add_depends_on.add(prev_id)
        new_insn_add_no_sync_with.add((prev_id, "any"))
        new_insn_add_within_inames.add(base_exec_iname or stage_exec_iname)

        if nresults == 1:
            assert len(acc_vars) == 1
            return acc_vars[0][outer_local_iname_vars + (0,)]
        else:
            return [acc_var[outer_local_iname_vars + (0,)] for acc_var in acc_vars]
    # }}}

    # {{{ utils (stateful)

    from pytools import memoize

    @memoize
    def get_or_add_sweep_tracking_iname_and_domain(
            scan_iname, sweep_iname, sweep_min_value, scan_min_value, stride,
            tracking_iname):
        domain = temp_kernel.get_inames_domain(frozenset((scan_iname, sweep_iname)))

        inames_added_for_scan.add(tracking_iname)

        new_domain = _create_domain_for_sweep_tracking(domain,
                tracking_iname, sweep_iname, sweep_min_value, scan_min_value, stride)

        _insert_subdomain_into_domain_tree(temp_kernel, domains, new_domain)

        return tracking_iname

    def replace_var_within_expr(expr, from_var, to_var):
        from pymbolic.mapper.substitutor import make_subst_func

        from loopy.symbolic import (
            SubstitutionRuleMappingContext, RuleAwareSubstitutionMapper)

        rule_mapping_context = SubstitutionRuleMappingContext(
            temp_kernel.substitutions, var_name_gen)

        from pymbolic import var
        mapper = RuleAwareSubstitutionMapper(
            rule_mapping_context,
            make_subst_func({from_var: var(to_var)}),
            within=lambda *args: True)

        return mapper(expr, temp_kernel, None)

    def make_temporaries(name_based_on, nvars, shape, dtypes, scope):
        var_names = [
                var_name_gen(name_based_on.format(index=i))
                for i in range(nvars)]

        from loopy.kernel.data import TemporaryVariable

        for name, dtype in zip(var_names, dtypes):
            new_temporary_variables[name] = TemporaryVariable(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    scope=scope)

        return var_names

    # }}}

    # {{{ sequential scan

    def map_scan_seq(expr, rec, nresults, arg_dtypes,
            reduction_dtypes, sweep_iname, scan_iname, sweep_min_value,
            scan_min_value, stride):
        outer_insn_inames = temp_kernel.insn_inames(insn)
        inames_to_remove.add(scan_iname)

        track_iname = var_name_gen(
                "{sweep_iname}__seq_scan"
                .format(scan_iname=scan_iname, sweep_iname=sweep_iname))

        get_or_add_sweep_tracking_iname_and_domain(
                scan_iname, sweep_iname, sweep_min_value, scan_min_value,
                stride, track_iname)

        from loopy.kernel.data import temp_var_scope
        acc_var_names = make_temporaries(
                name_based_on="acc_" + scan_iname,
                nvars=nresults,
                shape=(),
                dtypes=reduction_dtypes,
                scope=temp_var_scope.PRIVATE)

        from pymbolic import var
        acc_vars = tuple(var(n) for n in acc_var_names)

        init_id = insn_id_gen(
                "%s_%s_init" % (insn.id, "_".join(expr.inames)))

        init_insn_depends_on = frozenset()

        global_barrier = lp.find_most_recent_global_barrier(temp_kernel, insn.id)

        if global_barrier is not None:
            init_insn_depends_on |= frozenset([global_barrier])

        init_insn = make_assignment(
                id=init_id,
                assignees=acc_vars,
                within_inames=outer_insn_inames - frozenset(
                    (sweep_iname,) + expr.inames),
                within_inames_is_final=insn.within_inames_is_final,
                depends_on=init_insn_depends_on,
                expression=expr.operation.neutral_element(*arg_dtypes))

        generated_insns.append(init_insn)

        update_insn_depends_on = set([init_insn.id]) | insn.depends_on

        updated_inner_exprs = (
                preprocess_scan_arguments(insn, expr.expr, nresults,
                    scan_iname, track_iname, update_insn_depends_on))

        update_id = insn_id_gen(
                based_on="%s_%s_update" % (insn.id, "_".join(expr.inames)))

        update_insn_iname_deps = temp_kernel.insn_inames(insn) | set([track_iname])
        if insn.within_inames_is_final:
            update_insn_iname_deps = insn.within_inames | set([track_iname])

        scan_insn = make_assignment(
                id=update_id,
                assignees=acc_vars,
                expression=expr.operation(
                    arg_dtypes,
                    _strip_if_scalar(acc_vars, acc_vars),
                    _strip_if_scalar(acc_vars, updated_inner_exprs)),
                depends_on=frozenset(update_insn_depends_on),
                within_inames=update_insn_iname_deps,
                no_sync_with=insn.no_sync_with,
                within_inames_is_final=insn.within_inames_is_final)

        generated_insns.append(scan_insn)

        new_insn_add_depends_on.add(scan_insn.id)

        if nresults == 1:
            assert len(acc_vars) == 1
            return acc_vars[0]
        else:
            return acc_vars

    # }}}

    # {{{ local-parallel scan

    def map_scan_local(expr, rec, nresults, arg_dtypes,
            reduction_dtypes, sweep_iname, scan_iname,
            sweep_min_value, scan_min_value, stride):

        scan_size = _get_int_iname_size(sweep_iname)

        assert scan_size > 0

        if scan_size == 1:
            return map_reduction_seq(
                    expr, rec, nresults, arg_dtypes, reduction_dtypes)

        outer_insn_inames = temp_kernel.insn_inames(insn)

        from loopy.kernel.data import LocalIndexTagBase
        outer_local_inames = tuple(
                oiname
                for oiname in outer_insn_inames
                if isinstance(
                    kernel.iname_to_tag.get(oiname),
                    LocalIndexTagBase)
                and oiname != sweep_iname)

        from pymbolic import var
        outer_local_iname_vars = tuple(
                var(oiname) for oiname in outer_local_inames)

        outer_local_iname_sizes = tuple(
                _get_int_iname_size(oiname)
                for oiname in outer_local_inames)

        track_iname = var_name_gen(
                "{sweep_iname}__pre_scan"
                .format(scan_iname=scan_iname, sweep_iname=sweep_iname))

        get_or_add_sweep_tracking_iname_and_domain(
                scan_iname, sweep_iname, sweep_min_value, scan_min_value, stride,
                track_iname)

        # {{{ add separate iname to carry out the scan

        # Doing this sheds any odd conditionals that may be active
        # on our scan_iname.

        base_exec_iname = var_name_gen(sweep_iname + "__scan")
        domains.append(_make_slab_set(base_exec_iname, scan_size))
        new_iname_tags[base_exec_iname] = kernel.iname_to_tag[sweep_iname]

        # }}}

        from loopy.kernel.data import temp_var_scope

        read_var_names = make_temporaries(
                name_based_on="read_"+scan_iname+"_arg_{index}",
                nvars=nresults,
                shape=(),
                dtypes=reduction_dtypes,
                scope=temp_var_scope.PRIVATE)

        acc_var_names = make_temporaries(
                name_based_on="acc_"+scan_iname,
                nvars=nresults,
                shape=outer_local_iname_sizes + (scan_size,),
                dtypes=reduction_dtypes,
                scope=temp_var_scope.LOCAL)

        acc_vars = tuple(var(n) for n in acc_var_names)
        read_vars = tuple(var(n) for n in read_var_names)

        base_iname_deps = (outer_insn_inames
                - frozenset(expr.inames) - frozenset([sweep_iname]))

        neutral = expr.operation.neutral_element(*arg_dtypes)

        init_insn_depends_on = insn.depends_on

        global_barrier = lp.find_most_recent_global_barrier(temp_kernel, insn.id)

        if global_barrier is not None:
            init_insn_depends_on |= frozenset([global_barrier])

        init_id = insn_id_gen("%s_%s_init" % (insn.id, scan_iname))
        init_insn = make_assignment(
                id=init_id,
                assignees=tuple(
                    acc_var[outer_local_iname_vars + (var(base_exec_iname),)]
                    for acc_var in acc_vars),
                expression=neutral,
                within_inames=base_iname_deps | frozenset([base_exec_iname]),
                within_inames_is_final=insn.within_inames_is_final,
                depends_on=init_insn_depends_on)
        generated_insns.append(init_insn)

        transfer_insn_depends_on = set([init_insn.id]) | insn.depends_on

        updated_inner_exprs = (
                preprocess_scan_arguments(insn, expr.expr, nresults,
                    scan_iname, track_iname, transfer_insn_depends_on))

        from loopy.symbolic import Reduction

        from loopy.symbolic import pw_aff_to_expr
        sweep_min_value_expr = pw_aff_to_expr(sweep_min_value)

        transfer_id = insn_id_gen("%s_%s_transfer" % (insn.id, scan_iname))
        transfer_insn = make_assignment(
                id=transfer_id,
                assignees=tuple(
                    acc_var[outer_local_iname_vars
                            + (var(sweep_iname) - sweep_min_value_expr,)]
                    for acc_var in acc_vars),
                expression=Reduction(
                    operation=expr.operation,
                    inames=(track_iname,),
                    expr=_strip_if_scalar(acc_vars, updated_inner_exprs),
                    allow_simultaneous=False,
                    ),
                within_inames=outer_insn_inames - frozenset(expr.inames),
                within_inames_is_final=insn.within_inames_is_final,
                depends_on=frozenset(transfer_insn_depends_on),
                no_sync_with=frozenset([(init_id, "any")]) | insn.no_sync_with)

        generated_insns.append(transfer_insn)

        prev_id = transfer_id

        istage = 0
        cur_size = 1

        while cur_size < scan_size:
            stage_exec_iname = var_name_gen("%s__scan_s%d" % (sweep_iname, istage))
            domains.append(
                    _make_slab_set_from_range(stage_exec_iname, cur_size, scan_size))
            new_iname_tags[stage_exec_iname] = kernel.iname_to_tag[sweep_iname]

            for read_var, acc_var in zip(read_vars, acc_vars):
                read_stage_id = insn_id_gen(
                        "scan_%s_read_stage_%d" % (scan_iname, istage))

                read_stage_insn = make_assignment(
                        id=read_stage_id,
                        assignees=(read_var,),
                        expression=(
                                acc_var[
                                    outer_local_iname_vars
                                    + (var(stage_exec_iname) - cur_size,)]),
                        within_inames=(
                            base_iname_deps | frozenset([stage_exec_iname])),
                        within_inames_is_final=insn.within_inames_is_final,
                        depends_on=frozenset([prev_id]))

                if cur_size == 1:
                    # Performance hack: don't add a barrier here with transfer_insn.
                    # NOTE: This won't work if the way that local inames
                    # are lowered changes.
                    read_stage_insn = read_stage_insn.copy(
                            no_sync_with=(
                                read_stage_insn.no_sync_with
                                | frozenset([(transfer_id, "any")])))

                generated_insns.append(read_stage_insn)
                prev_id = read_stage_id

            write_stage_id = insn_id_gen(
                    "scan_%s_write_stage_%d" % (scan_iname, istage))
            write_stage_insn = make_assignment(
                    id=write_stage_id,
                    assignees=tuple(
                        acc_var[outer_local_iname_vars + (var(stage_exec_iname),)]
                        for acc_var in acc_vars),
                    expression=expr.operation(
                        arg_dtypes,
                        _strip_if_scalar(acc_vars, read_vars),
                        _strip_if_scalar(acc_vars, tuple(
                            acc_var[
                                outer_local_iname_vars + (var(stage_exec_iname),)]
                            for acc_var in acc_vars))
                        ),
                    within_inames=(
                        base_iname_deps | frozenset([stage_exec_iname])),
                    within_inames_is_final=insn.within_inames_is_final,
                    depends_on=frozenset([prev_id]),
                    )

            generated_insns.append(write_stage_insn)
            prev_id = write_stage_id

            cur_size *= 2
            istage += 1

        new_insn_add_depends_on.add(prev_id)
        new_insn_add_within_inames.add(sweep_iname)

        output_idx = var(sweep_iname) - sweep_min_value_expr

        if nresults == 1:
            assert len(acc_vars) == 1
            return acc_vars[0][outer_local_iname_vars + (output_idx,)]
        else:
            return [acc_var[outer_local_iname_vars + (output_idx,)]
                    for acc_var in acc_vars]

    # }}}

    # {{{ seq/par dispatch

    def map_reduction(expr, rec, nresults=1):
        # Only expand one level of reduction at a time, going from outermost to
        # innermost. Otherwise we get the (iname + insn) dependencies wrong.

        from loopy.type_inference import (
                infer_arg_and_reduction_dtypes_for_reduction_expression)
        arg_dtypes, reduction_dtypes = (
                infer_arg_and_reduction_dtypes_for_reduction_expression(
                        temp_kernel, expr, unknown_types_ok))

        outer_insn_inames = temp_kernel.insn_inames(insn)
        bad_inames = frozenset(expr.inames) & outer_insn_inames
        if bad_inames:
            raise LoopyError("reduction used within loop(s) that it was "
                    "supposed to reduce over: " + ", ".join(bad_inames))

        iname_classes = _classify_reduction_inames(temp_kernel, expr.inames)

        n_sequential = len(iname_classes.sequential)
        n_local_par = len(iname_classes.local_parallel)
        n_nonlocal_par = len(iname_classes.nonlocal_parallel)

        really_force_scan = force_scan and (
                len(expr.inames) != 1 or expr.inames[0] not in inames_added_for_scan)

        def _error_if_force_scan_on(cls, msg):
            if really_force_scan:
                raise cls(msg)

        may_be_implemented_as_scan = False
        if force_scan or automagic_scans_ok:
            from loopy.diagnostic import ReductionIsNotTriangularError

            try:
                # Try to determine scan candidate information (sweep iname, scan
                # iname, etc).
                scan_param = _try_infer_scan_candidate_from_expr(
                        temp_kernel, expr, outer_insn_inames,
                        sweep_iname=force_outer_iname_for_scan)

            except ValueError as v:
                error = str(v)

            else:
                # Ensures the reduction is triangular (somewhat expensive).
                may_be_implemented_as_scan, error = (
                        _check_reduction_is_triangular(
                            temp_kernel, expr, scan_param))

            if not may_be_implemented_as_scan:
                _error_if_force_scan_on(ReductionIsNotTriangularError, error)

        # {{{ sanity checks

        if n_local_par and n_sequential:
            raise LoopyError("Reduction over '%s' contains both parallel and "
                    "sequential inames. It must be split "
                    "(using split_reduction_{in,out}ward) "
                    "before code generation."
                    % ", ".join(expr.inames))

        if n_local_par > 1:
            raise LoopyError("Reduction over '%s' contains more than"
                    "one parallel iname. It must be split "
                    "(using split_reduction_{in,out}ward) "
                    "before code generation."
                    % ", ".join(expr.inames))

        if n_nonlocal_par:
            bad_inames = iname_classes.nonlocal_parallel
            raise LoopyError("the only form of parallelism supported "
                    "by reductions is 'local'--found iname(s) '%s' "
                    "respectively tagged '%s'"
                    % (", ".join(bad_inames),
                       ", ".join(kernel.iname_to_tag[iname]
                                 for iname in bad_inames)))

        if n_local_par == 0 and n_sequential == 0:
            from loopy.diagnostic import warn_with_kernel
            warn_with_kernel(kernel, "empty_reduction",
                    "Empty reduction found (no inames to reduce over). "
                    "Eliminating.")

            # We're not supposed to reduce/sum at all. (Note how this is distinct
            # from an empty reduction--there is an element here, just no inames
            # to reduce over. It's rather similar to an array with () shape in
            # numpy.)

            return expr.expr

        # }}}

        if may_be_implemented_as_scan:
            assert force_scan or automagic_scans_ok

            # We require the "scan" iname to be tagged sequential.
            if n_sequential:
                sweep_iname = scan_param.sweep_iname
                sweep_class = _classify_reduction_inames(kernel, (sweep_iname,))

                sequential = sweep_iname in sweep_class.sequential
                parallel = sweep_iname in sweep_class.local_parallel
                bad_parallel = sweep_iname in sweep_class.nonlocal_parallel

                if sweep_iname not in outer_insn_inames:
                    _error_if_force_scan_on(LoopyError,
                            "Sweep iname '%s' was detected, but is not an iname "
                            "for the instruction." % sweep_iname)
                elif bad_parallel:
                    _error_if_force_scan_on(LoopyError,
                            "Sweep iname '%s' has an unsupported parallel tag '%s' "
                            "- the only parallelism allowed is 'local'." %
                            (sweep_iname, temp_kernel.iname_to_tag[sweep_iname]))
                elif parallel:
                    return map_scan_local(
                            expr, rec, nresults, arg_dtypes, reduction_dtypes,
                            sweep_iname, scan_param.scan_iname,
                            scan_param.sweep_lower_bound,
                            scan_param.scan_lower_bound,
                            scan_param.stride)
                elif sequential:
                    return map_scan_seq(
                            expr, rec, nresults, arg_dtypes, reduction_dtypes,
                            sweep_iname, scan_param.scan_iname,
                            scan_param.sweep_lower_bound,
                            scan_param.scan_lower_bound,
                            scan_param.stride)

                # fallthrough to reduction implementation

            else:
                assert n_local_par > 0
                scan_iname, = expr.inames
                _error_if_force_scan_on(LoopyError,
                        "Scan iname '%s' is parallel tagged: this is not allowed "
                        "(only the sweep iname should be tagged if parallelism "
                        "is desired)." % scan_iname)

                # fallthrough to reduction implementation

        if n_sequential:
            assert n_local_par == 0
            return map_reduction_seq(
                    expr, rec, nresults, arg_dtypes, reduction_dtypes)
        else:
            assert n_local_par > 0
            return map_reduction_local(
                    expr, rec, nresults, arg_dtypes, reduction_dtypes)

    # }}}

    from loopy.symbolic import ReductionCallbackMapper
    cb_mapper = ReductionCallbackMapper(map_reduction)

    insn_queue = kernel.instructions[:]
    insn_id_replacements = {}
    domains = kernel.domains[:]

    temp_kernel = kernel

    import loopy as lp
    while insn_queue:
        new_insn_add_depends_on = set()
        new_insn_add_no_sync_with = set()
        new_insn_add_within_inames = set()

        generated_insns = []

        insn = insn_queue.pop(0)

        if insn_id_filter is not None and insn.id != insn_id_filter \
                or not isinstance(insn, lp.MultiAssignmentBase):
            new_insns.append(insn)
            continue

        nresults = len(insn.assignees)

        # Run reduction expansion.
        from loopy.symbolic import Reduction
        if isinstance(insn.expression, Reduction) and nresults > 1:
            new_expressions = cb_mapper(insn.expression, nresults=nresults)
        else:
            new_expressions = (cb_mapper(insn.expression),)

        if generated_insns:
            # An expansion happened, so insert the generated stuff plus
            # ourselves back into the queue.

            kwargs = insn.get_copy_kwargs(
                    depends_on=insn.depends_on
                    | frozenset(new_insn_add_depends_on),
                    no_sync_with=insn.no_sync_with
                    | frozenset(new_insn_add_no_sync_with),
                    within_inames=(
                        temp_kernel.insn_inames(insn)
                        | new_insn_add_within_inames))

            kwargs.pop("id")
            kwargs.pop("expression")
            kwargs.pop("assignee", None)
            kwargs.pop("assignees", None)
            kwargs.pop("temp_var_type", None)
            kwargs.pop("temp_var_types", None)

            if isinstance(insn.expression, Reduction) and nresults > 1:
                replacement_insns = [
                        lp.Assignment(
                            id=insn_id_gen(insn.id),
                            assignee=assignee,
                            expression=new_expr,
                            **kwargs)
                        for assignee, new_expr in zip(
                            insn.assignees, new_expressions)]

            else:
                new_expr, = new_expressions
                replacement_insns = [
                        make_assignment(
                            id=insn_id_gen(insn.id),
                            assignees=insn.assignees,
                            expression=new_expr,
                            **kwargs)
                        ]

            insn_id_replacements[insn.id] = [
                    rinsn.id for rinsn in replacement_insns]

            insn_queue = generated_insns + replacement_insns + insn_queue

            # The reduction expander needs an up-to-date kernel
            # object to find dependencies. Keep temp_kernel up-to-date.

            temp_kernel = kernel.copy(
                    instructions=new_insns + insn_queue,
                    temporary_variables=new_temporary_variables,
                    domains=domains)
            temp_kernel = lp.replace_instruction_ids(
                    temp_kernel, insn_id_replacements)

        else:
            # nothing happened, we're done with insn
            assert not new_insn_add_depends_on

            new_insns.append(insn)

    kernel = kernel.copy(
            instructions=new_insns,
            temporary_variables=new_temporary_variables,
            domains=domains)

    kernel = lp.replace_instruction_ids(kernel, insn_id_replacements)

    kernel = lp.tag_inames(kernel, new_iname_tags)

    # TODO: remove unused inames...

    kernel = (
            _hackily_ensure_multi_assignment_return_values_are_scoped_private(
                kernel))

    return kernel

# }}}


# {{{ find idempotence ("boostability") of instructions

def find_idempotence(kernel):
    logger.debug("%s: idempotence" % kernel.name)

    writer_map = kernel.writer_map()

    arg_names = set(arg.name for arg in kernel.args)

    var_names = arg_names | set(six.iterkeys(kernel.temporary_variables))

    reads_map = dict(
            (insn.id, insn.read_dependency_names() & var_names)
            for insn in kernel.instructions)

    from collections import defaultdict
    dep_graph = defaultdict(lambda: set())

    for insn in kernel.instructions:
        dep_graph[insn.id] = set(writer_id
                for var in reads_map[insn.id]
                for writer_id in writer_map.get(var, set()))

    # Find SCCs of dep_graph. These are used for checking if the instruction is
    # in a dependency cycle.
    from loopy.tools import compute_sccs

    sccs = dict((item, scc)
            for scc in compute_sccs(dep_graph)
            for item in scc)

    non_idempotently_updated_vars = set()

    new_insns = []
    for insn in kernel.instructions:
        boostable = len(sccs[insn.id]) == 1 and insn.id not in dep_graph[insn.id]

        if not boostable:
            non_idempotently_updated_vars.update(
                    insn.assignee_var_names())

        new_insns.append(insn.copy(boostable=boostable))

    # {{{ remove boostability from isns that access non-idempotently updated vars

    new2_insns = []
    for insn in new_insns:
        if insn.boostable and bool(
                non_idempotently_updated_vars & insn.dependency_names()):
            new2_insns.append(insn.copy(boostable=False))
        else:
            new2_insns.append(insn)

    # }}}

    return kernel.copy(instructions=new2_insns)

# }}}


# {{{ limit boostability

def limit_boostability(kernel):
    """Finds out which other inames an instruction's inames occur with
    and then limits boostability to just those inames.
    """

    logger.debug("%s: limit boostability" % kernel.name)

    iname_occurs_with = {}
    for insn in kernel.instructions:
        insn_inames = kernel.insn_inames(insn)
        for iname in insn_inames:
            iname_occurs_with.setdefault(iname, set()).update(insn_inames)

    iname_use_counts = {}
    for insn in kernel.instructions:
        for iname in kernel.insn_inames(insn):
            iname_use_counts[iname] = iname_use_counts.get(iname, 0) + 1

    single_use_inames = set(iname for iname, uc in six.iteritems(iname_use_counts)
            if uc == 1)

    new_insns = []
    for insn in kernel.instructions:
        if insn.boostable is None:
            raise LoopyError("insn '%s' has undetermined boostability" % insn.id)
        elif insn.boostable:
            boostable_into = set()
            for iname in kernel.insn_inames(insn):
                boostable_into.update(iname_occurs_with[iname])

            boostable_into -= kernel.insn_inames(insn) | single_use_inames

            # Even if boostable_into is empty, leave boostable flag on--it is used
            # for boosting into unused hw axes.

            insn = insn.copy(boostable_into=boostable_into)
        else:
            insn = insn.copy(boostable_into=set())

        new_insns.append(insn)

    return kernel.copy(instructions=new_insns)

# }}}


preprocess_cache = PersistentDict("loopy-preprocess-cache-v2-"+DATA_MODEL_VERSION,
        key_builder=LoopyKeyBuilder())


def preprocess_kernel(kernel, device=None):
    if device is not None:
        from warnings import warn
        warn("passing 'device' to preprocess_kernel() is deprecated",
                DeprecationWarning, stacklevel=2)

    from loopy.kernel import kernel_state
    if kernel.state >= kernel_state.PREPROCESSED:
        return kernel

    # {{{ cache retrieval

    from loopy import CACHING_ENABLED
    if CACHING_ENABLED:
        input_kernel = kernel

        try:
            result = preprocess_cache[kernel]
            logger.debug("%s: preprocess cache hit" % kernel.name)
            return result
        except KeyError:
            pass

    # }}}

    logger.info("%s: preprocess start" % kernel.name)

    from loopy.check import check_identifiers_in_subst_rules
    check_identifiers_in_subst_rules(kernel)

    # {{{ check that there are no l.auto-tagged inames

    from loopy.kernel.data import AutoLocalIndexTagBase
    for iname, tag in six.iteritems(kernel.iname_to_tag):
        if (isinstance(tag, AutoLocalIndexTagBase)
                 and iname in kernel.all_inames()):
            raise LoopyError("kernel with automatically-assigned "
                    "local axes passed to preprocessing")

    # }}}

    from loopy.transform.subst import expand_subst
    kernel = expand_subst(kernel)

    # Ordering restriction:
    # Type inference and reduction iname uniqueness don't handle substitutions.
    # Get them out of the way.

    kernel = infer_unknown_types(kernel, expect_completion=False)

    check_for_writes_to_predicates(kernel)
    check_reduction_iname_uniqueness(kernel)

    from loopy.kernel.creation import apply_single_writer_depencency_heuristic
    kernel = apply_single_writer_depencency_heuristic(kernel)

    # Ordering restrictions:
    #
    # - realize_reduction must happen after type inference because it needs
    #   to be able to determine the types of the reduced expressions.
    #
    # - realize_reduction must happen after default dependencies are added
    #   because it manipulates the depends_on field, which could prevent
    #   defaults from being applied.

    kernel = realize_reduction(kernel, unknown_types_ok=False)

    # Ordering restriction:
    # add_axes_to_temporaries_for_ilp because reduction accumulators
    # need to be duplicated by this.

    from loopy.transform.ilp import add_axes_to_temporaries_for_ilp_and_vec
    kernel = add_axes_to_temporaries_for_ilp_and_vec(kernel)

    kernel = find_temporary_scope(kernel)

    # boostability should be removed in 2017.x.
    kernel = find_idempotence(kernel)
    kernel = limit_boostability(kernel)

    kernel = kernel.target.preprocess(kernel)

    logger.info("%s: preprocess done" % kernel.name)

    kernel = kernel.copy(
            state=kernel_state.PREPROCESSED)

    # {{{ prepare for caching

    # PicklableDtype instances for example need to know the target they're working
    # towards in order to pickle and unpickle them. This is the first pass that
    # uses caching, so we need to be ready to pickle. This means propagating
    # this target information.

    if CACHING_ENABLED:
        input_kernel = prepare_for_caching(input_kernel)

    kernel = prepare_for_caching(kernel)

    # }}}

    if CACHING_ENABLED:
        preprocess_cache[input_kernel] = kernel

    return kernel

# vim: foldmethod=marker

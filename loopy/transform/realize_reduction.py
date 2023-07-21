__copyright__ = """
Copyright (C) 2012 Andreas Kloeckner
Copyright (C) 2022 University of Illinois Board of Trustees
"""

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


from dataclasses import dataclass, replace
from typing import (Tuple, Dict, Callable, List, Optional, Set, Sequence,
        FrozenSet)

import logging
logger = logging.getLogger(__name__)

from pytools import memoize_on_first_arg
from pytools.tag import Tag
import islpy as isl
from pymbolic.primitives import Expression

from immutables import Map

from loopy.kernel.data import make_assignment
from loopy.symbolic import ReductionCallbackMapper
from loopy.translation_unit import TranslationUnit
from loopy.kernel.function_interface import CallableKernel
from loopy.kernel.data import TemporaryVariable, AddressSpace
from loopy.kernel.instruction import (
        InstructionBase, MultiAssignmentBase, Assignment)
from loopy.kernel import LoopKernel
from loopy.diagnostic import (
        LoopyError, warn_with_kernel, ReductionIsNotTriangularError)
from loopy.transform.instruction import replace_instruction_ids_in_insn


# {{{ reduction realization context

@dataclass
class _ChangeFlag:
    changes_made: bool


@dataclass(frozen=True)
class _ReductionRealizationContext:
    # {{{ read-only

    mapper: "RealizeReductionCallbackMapper"

    force_scan: bool
    automagic_scans_ok: bool
    unknown_types_ok: bool

    # FIXME: This feels like a broken-by-design concept.
    force_outer_iname_for_scan: Optional[str]

    # We use the original kernel for a number of lookups whose value
    # we do not change and which might be already cached on it.
    orig_kernel: LoopKernel
    kernel: LoopKernel

    id_prefix: str

    # }}}

    # {{{ internally mutable, same across entire recursion

    insn_id_gen: Callable[[str], str]
    var_name_gen: Callable[[str], str]

    additional_temporary_variables: Dict[str, TemporaryVariable]
    additional_insns: List[InstructionBase]
    domains: List[isl.BasicSet]
    additional_iname_tags: Dict[str, Sequence[Tag]]
    # list only to facilitate mutation
    boxed_callables_table: List[Map]

    # FIXME: This is a broken-by-design concept. Local-parallel scans emit a
    # reduction internally. This serves to avoid force_scan acting on that
    # reduction.
    inames_added_for_scan: Set[str]

    # }}}

    # {{{ surrounding instruction, read-only (different at each recursive level)

    # These are attributes from 'surrounding' instruction, for generated
    # instructions to potentially inherit.
    surrounding_within_inames: FrozenSet[str]
    surrounding_depends_on: FrozenSet[str]
    surrounding_no_sync_with: FrozenSet[Tuple[str, str]]
    surrounding_predicates: FrozenSet[Expression]

    # }}}

    # {{{ surrounding instruction, internally mutable
    # (different at each recursive level)

    # These are requested additions to attributes of the surrounding instruction.

    # FIXME add_within_inames seems broken by design.
    surrounding_insn_add_within_inames: Set[str]

    surrounding_insn_add_depends_on: Set[str]
    surrounding_insn_add_no_sync_with: Set[Tuple[str, str]]

    # }}}

    # {{{ change tracking (same across entire recursion)

    _change_flag: _ChangeFlag

    @property
    def were_changes_made(self):
        return self._change_flag.changes_made

    def changes_made(self):
        self._change_flag.changes_made = True

    # }}}

    def new_subinstruction(self, *, within_inames, depends_on,
            no_sync_with=None, predicates=None):
        if no_sync_with is None:
            no_sync_with = self.surrounding_no_sync_with
        if predicates is None:
            predicates = self.surrounding_predicates

        return replace(self,
                surrounding_within_inames=within_inames,
                surrounding_depends_on=depends_on,
                surrounding_no_sync_with=no_sync_with,
                surrounding_predicates=predicates,

                surrounding_insn_add_within_inames=set(),
                surrounding_insn_add_depends_on=set(),
                surrounding_insn_add_no_sync_with=set())

    def get_insn_kwargs(self):
        return {
                "within_inames": (
                    self.surrounding_within_inames
                    | frozenset(self.surrounding_insn_add_within_inames)),
                "within_inames_is_final": True,
                "depends_on": (
                    self.surrounding_depends_on
                    | frozenset(self.surrounding_insn_add_depends_on)),
                "no_sync_with": (
                    self.surrounding_no_sync_with
                    | frozenset(self.surrounding_insn_add_no_sync_with)),
                "predicates": self.surrounding_predicates}

# }}}


# {{{ iname/domain wrangling

@dataclass(frozen=True)
class _InameClassification:
    sequential: Tuple[str, ...]
    local_parallel: Tuple[str, ...]
    nonlocal_parallel: Tuple[str, ...]


def _classify_reduction_inames(red_realize_ctx, inames):
    sequential = []
    local_par = []
    nonlocal_par = []

    from loopy.kernel.data import (
            LocalInameTagBase, UnrolledIlpTag, UnrollTag,
            ConcurrentTag, filter_iname_tags_by_type)

    for iname in inames:
        try:
            iname_tags = red_realize_ctx.additional_iname_tags[iname]
        except KeyError:
            iname_tags = red_realize_ctx.kernel.iname_tags(iname)

        if filter_iname_tags_by_type(iname_tags, (UnrollTag, UnrolledIlpTag)):
            # These are nominally parallel, but we can live with
            # them as sequential.
            sequential.append(iname)

        elif filter_iname_tags_by_type(iname_tags, LocalInameTagBase):
            local_par.append(iname)

        elif filter_iname_tags_by_type(iname_tags, ConcurrentTag):
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


def _insert_subdomain_into_domain_tree(kernel, domains, subdomain):
    # Intersect with inames, because we could have captured some kernel params
    # in here too...
    dependent_inames = (
            frozenset(subdomain.get_var_names(isl.dim_type.param))
            & kernel.all_inames())
    idx, = kernel.get_leaf_domain_indices(dependent_inames)
    domains.insert(idx + 1, subdomain)

# }}}


# {{{ scan inference

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
            affs[0])

    sweep_upper_bound = isl.align_spaces(
            scan_param.sweep_upper_bound,
            affs[0])

    scan_lower_bound = isl.align_spaces(
            scan_param.scan_lower_bound,
            affs[0])

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


@dataclass(frozen=True)
class _ScanCandidateParameters:
    sweep_iname: str
    scan_iname: str
    sweep_lower_bound: isl.PwAff
    sweep_upper_bound: isl.PwAff
    scan_lower_bound: isl.PwAff
    stride: int


def _try_infer_scan_candidate_from_expr(
        kernel, expr, within_inames, sweep_iname=None):
    """Analyze `expr` and determine if it can be implemented as a scan.
    """
    from loopy.symbolic import Reduction
    assert isinstance(expr, Reduction)

    if len(expr.inames) != 1:
        raise ValueError(
                "Multiple inames in reduction: '{}'".format(", ".join(expr.inames)))

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

    return _ScanCandidateParameters(
            sweep_iname=sweep_iname,
            scan_iname=scan_iname,
            sweep_lower_bound=sweep_lower_bound,
            sweep_upper_bound=sweep_upper_bound,
            scan_lower_bound=scan_lower_bound,
            stride=stride)


def _try_infer_sweep_iname(domain, scan_iname, candidate_inames):
    """The sweep iname is the outer iname which guides the scan.

    E.g. for a domain of {[i,j]: 0<=i<n and 0<=j<=i}, i is the sweep iname.
    """
    constrs = domain.get_constraints()
    sweep_iname_candidate = None

    for constr in constrs:
        candidate_vars = {
                var for var in constr.get_var_dict()
                if var in candidate_inames}

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
        except Exception:
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

# }}}


# {{{ domain creation for scans

def _get_domain_with_iname_as_param(domain, iname):
    dim_type = isl.dim_type

    if domain.find_dim_by_name(dim_type.param, iname) >= 0:
        return domain

    iname_idx = domain.find_dim_by_name(dim_type.set, iname)

    assert iname_idx >= 0, (iname, domain)

    return domain.move_dims(
        dim_type.param, domain.dim(dim_type.param),
        dim_type.set, iname_idx, 1)


def _create_domain_for_sweep_tracking(orig_domain, tracking_iname, scan_param):
    sp = scan_param

    dim_type = isl.dim_type

    subd = isl.BasicSet.universe(orig_domain.params().space)

    # Add tracking_iname and sweep iname.

    subd = _add_params_to_domain(subd, (sp.sweep_iname, tracking_iname))

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

    subd &= (affs[tracking_iname] - sp.scan_lower_bound).ge_set(affs[0])
    subd &= (affs[tracking_iname] - sp.scan_lower_bound)\
            .le_set(sp.stride * (affs[sp.sweep_iname] - sp.sweep_lower_bound))
    subd &= (affs[tracking_iname] - sp.scan_lower_bound)\
            .gt_set(sp.stride * (affs[sp.sweep_iname] - sp.sweep_lower_bound - 1))

    # Move tracking_iname into a set dim (NOT sweep iname).
    subd = subd.move_dims(
            dim_type.set, 0,
            dim_type.param, subd.dim(dim_type.param) - 1, 1)

    # Simplify (maybe).
    orig_domain_with_sweep_param = (
            _get_domain_with_iname_as_param(orig_domain, sp.sweep_iname))
    subd = subd.gist_params(orig_domain_with_sweep_param.params())

    subd, = subd.get_basic_sets()

    return subd

# }}}


# {{{ _hackily_ensure_multi_assignment_return_values_are_scoped_private

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

    dep_map = {
            insn.id: insn.depends_on for insn in kernel.instructions}

    inverse_dep_map = {insn.id: set() for insn in kernel.instructions}

    for insn_id, deps in dep_map.items():
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

    from loopy.kernel.instruction import CallInstruction, is_array_call
    for insn in kernel.instructions:
        if not isinstance(insn, CallInstruction):
            continue

        if len(insn.assignees) <= 1:
            continue

        if is_array_call(insn.assignees, insn.expression):
            continue

        assignees = insn.assignees
        assignee_var_names = insn.assignee_var_names()

        new_assignees = [assignees[0]]
        newly_added_assignments_ids = set()
        needs_replacement = False

        last_added_insn_id = insn.id

        FIRST_POINTER_ASSIGNEE_IDX = 1  # noqa

        for assignee_nr, assignee_var_name, assignee in zip(
                range(FIRST_POINTER_ASSIGNEE_IDX, len(assignees)),
                assignee_var_names[FIRST_POINTER_ASSIGNEE_IDX:],
                assignees[FIRST_POINTER_ASSIGNEE_IDX:]):

            if (
                    assignee_var_name in kernel.temporary_variables
                    and
                    (kernel.temporary_variables[assignee_var_name].address_space
                         == AddressSpace.PRIVATE)):
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

            new_temporaries[new_assignee_name] = (
                    TemporaryVariable(
                        name=new_assignee_name,
                        dtype=None,
                        shape=(),
                        address_space=AddressSpace.PRIVATE))

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

    if not new_temporaries and not new_or_updated_instructions:
        return kernel

    new_temporary_variables = kernel.temporary_variables.copy()
    new_temporary_variables.update(new_temporaries)

    new_instructions = (
            list(new_or_updated_instructions.values())
            + [insn
                for insn in kernel.instructions
                if insn.id not in new_or_updated_instructions])

    return kernel.copy(temporary_variables=new_temporary_variables,
                       instructions=new_instructions)

# }}}


# {{{ RealizeReductionCallbackMapper

class RealizeReductionCallbackMapper(ReductionCallbackMapper):
    def __init__(self, callback):
        super().__init__(callback)

    def map_reduction(self, expr, **kwargs):
        return self.callback(expr, **kwargs)

    def map_if(self, expr, *, red_realize_ctx, nresults):
        common_kwargs = {"nresults": nresults}

        # {{{ generate code for condition

        rrc_cond = replace(red_realize_ctx,
                surrounding_insn_add_depends_on=set(),
                surrounding_insn_add_no_sync_with=set(),
                surrounding_insn_add_within_inames=set())
        import pymbolic.primitives as prim
        rec_cond = self.rec(
                expr.condition,
                red_realize_ctx=rrc_cond,
                **common_kwargs)
        assert not rrc_cond.surrounding_insn_add_no_sync_with
        assert not rrc_cond.surrounding_insn_add_within_inames

        cond_dep_on = rrc_cond.surrounding_insn_add_depends_on
        red_realize_ctx.surrounding_insn_add_depends_on.update(cond_dep_on)

        # }}}

        return prim.If(rec_cond,
                       self.rec(expr.then,
                           red_realize_ctx=replace(
                               red_realize_ctx,
                               surrounding_depends_on=(
                                   red_realize_ctx.surrounding_depends_on
                                   | cond_dep_on),
                               surrounding_predicates=(
                                   red_realize_ctx.surrounding_predicates
                                   | frozenset([rec_cond])
                                   )),
                           **common_kwargs),
                       self.rec(expr.else_,
                           red_realize_ctx=replace(
                               red_realize_ctx,
                               surrounding_depends_on=(
                                   red_realize_ctx.surrounding_depends_on
                                   | cond_dep_on),
                               surrounding_predicates=(
                                   red_realize_ctx.surrounding_predicates
                                   | frozenset([prim.LogicalNot(rec_cond)])
                                   )),
                           **common_kwargs))

# }}}


# {{{ helpers

def _strip_if_scalar(reference, val):
    if len(reference) == 1:
        return val[0]
    else:
        return val


def _preprocess_scan_arguments(
        red_realize_ctx,
        expr, nresults, scan_iname, track_iname,
        newly_generated_insn_id_set):
    """Does iname substitution within scan arguments and returns a set of values
    suitable to be passed to the binary op. Returns a tuple."""

    if nresults > 1:
        inner_expr = expr

        # In the case of a multi-argument scan, we need a name for each of
        # the arguments in order to pass them to the binary op - so we expand
        # items that are not "plain" tuples here.
        if not isinstance(inner_expr, tuple):
            get_args_insn_id = red_realize_ctx.insn_id_gen(
                    f"{red_realize_ctx.id_prefix}_{'_'.join(expr.inames)}_get")

            inner_expr = expand_inner_reduction(
                    red_realize_ctx=red_realize_ctx,
                    id=get_args_insn_id,
                    expr=inner_expr,
                    nresults=nresults,
                    depends_on=red_realize_ctx.surrounding_depends_on,
                    within_inames=red_realize_ctx.surrounding_within_inames,
                    predicates=red_realize_ctx.surrounding_predicates,
                    )

            newly_generated_insn_id_set = (
                    newly_generated_insn_id_set | frozenset({get_args_insn_id}))

        updated_inner_exprs = tuple(
                replace_var_within_expr(
                    red_realize_ctx.kernel, red_realize_ctx.var_name_gen,
                    sub_expr, scan_iname, track_iname)
                for sub_expr in inner_expr)
    else:
        updated_inner_exprs = (
                replace_var_within_expr(
                    red_realize_ctx.kernel, red_realize_ctx.var_name_gen,
                    expr, scan_iname, track_iname),)

    return updated_inner_exprs, newly_generated_insn_id_set

# }}}


def expand_inner_reduction(
        red_realize_ctx, id, expr, nresults, depends_on, within_inames, predicates):
    # FIXME: use _make_temporaries
    from pymbolic.primitives import Call
    from loopy.symbolic import Reduction
    assert isinstance(expr, (Call, Reduction))

    temp_var_names = [
            red_realize_ctx.var_name_gen(id + "_arg" + str(i))
            for i in range(nresults)]

    for name in temp_var_names:
        red_realize_ctx.additional_temporary_variables[name] = TemporaryVariable(
                name=name,
                shape=(),
                dtype=None,
                address_space=AddressSpace.PRIVATE)

    from pymbolic import var
    temp_vars = tuple(var(n) for n in temp_var_names)

    call_insn = make_assignment(
            id=id,
            assignees=temp_vars,
            expression=expr,
            depends_on=depends_on,
            within_inames=within_inames,
            within_inames_is_final=True,
            predicates=predicates)

    red_realize_ctx.additional_insns.append(call_insn)

    return temp_vars


# {{{ reduction type: sequential

def map_reduction_seq(red_realize_ctx, expr, nresults, arg_dtypes, reduction_dtypes):
    orig_kernel = red_realize_ctx.orig_kernel

    acc_var_names = _make_temporaries(
            red_realize_ctx=red_realize_ctx,
            name_based_on="acc_"+"_".join(expr.inames),
            nvars=nresults,
            shape=(),
            dtypes=reduction_dtypes,
            address_space=AddressSpace.PRIVATE)

    from pymbolic import var
    acc_vars = tuple(var(n) for n in acc_var_names)

    init_id = red_realize_ctx.insn_id_gen(
            f"{red_realize_ctx.id_prefix}_{'_'.join(expr.inames)}_init")

    expression, red_realize_ctx.boxed_callables_table[0] = \
            expr.operation.neutral_element(
                    *arg_dtypes,
                    callables_table=red_realize_ctx.boxed_callables_table[0],
                    target=red_realize_ctx.orig_kernel.target)

    init_insn = make_assignment(
            id=init_id,
            assignees=acc_vars,
            within_inames=red_realize_ctx.surrounding_within_inames,
            within_inames_is_final=True,
            depends_on=frozenset(),
            expression=expression,

            # Do not inherit predicates: Those might read variables
            # that may not yet be set, and we don't have a great way
            # of figuring out what the dependencies of the accumulator
            # initializer should be.

            # This way, we may initialize a few too many accumulators,
            # but that's better than being incorrect.
            # https://github.com/inducer/loopy/issues/231
            )

    red_realize_ctx.additional_insns.append(init_insn)

    update_id = red_realize_ctx.insn_id_gen(
            based_on=f"{red_realize_ctx.id_prefix}_{'_'.join(expr.inames)}_update")

    update_red_realize_ctx = red_realize_ctx.new_subinstruction(
            within_inames=(
                red_realize_ctx.surrounding_within_inames
                | frozenset(expr.inames)),
            depends_on=(
                frozenset({init_id})
                | red_realize_ctx.surrounding_depends_on))

    reduction_expr = red_realize_ctx.mapper(
            expr.expr, red_realize_ctx=update_red_realize_ctx,
            nresults=1)

    # In the case of a multi-argument reduction, we need a name for each of
    # the arguments in order to pass them to the binary op - so we expand
    # items that are not "plain" tuples here.
    if nresults > 1 and not isinstance(reduction_expr, tuple):
        get_args_insn_id = red_realize_ctx.insn_id_gen(
                f"{red_realize_ctx.id_prefix}_{'_'.join(expr.inames)}_get")

        reduction_expr = expand_inner_reduction(
                red_realize_ctx=red_realize_ctx,
                id=get_args_insn_id,
                expr=reduction_expr,
                nresults=nresults,
                depends_on=red_realize_ctx.surrounding_depends_on,
                within_inames=update_red_realize_ctx.surrounding_within_inames,
                predicates=red_realize_ctx.surrounding_predicates,
                )

        update_red_realize_ctx.surrounding_insn_add_depends_on.add(get_args_insn_id)

    expression, red_realize_ctx.boxed_callables_table[0] = expr.operation(
            arg_dtypes,
            _strip_if_scalar(acc_vars, acc_vars),
            reduction_expr,
            red_realize_ctx.boxed_callables_table[0],
            orig_kernel.target)

    reduction_insn = make_assignment(
            id=update_id,
            assignees=acc_vars,
            expression=expression,
            **update_red_realize_ctx.get_insn_kwargs())

    red_realize_ctx.additional_insns.append(reduction_insn)
    red_realize_ctx.surrounding_insn_add_depends_on.add(reduction_insn.id)

    if nresults == 1:
        assert len(acc_vars) == 1
        return acc_vars[0]
    else:
        return acc_vars

# }}}


# {{{ reduction type: local-parallel

def _get_int_iname_size(kernel, iname):
    from loopy.isl_helpers import static_max_of_pw_aff
    from loopy.symbolic import pw_aff_to_expr
    size = pw_aff_to_expr(
            static_max_of_pw_aff(
                kernel.get_iname_bounds(iname).size,
                constants_only=True))
    assert isinstance(size, int)
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


def map_reduction_local(red_realize_ctx, expr, nresults, arg_dtypes,
        reduction_dtypes):
    orig_kernel = red_realize_ctx.orig_kernel

    red_iname, = expr.inames

    size = _get_int_iname_size(orig_kernel, red_iname)

    from loopy.kernel.data import LocalInameTagBase
    surrounding_local_inames = tuple(
            oiname for oiname in red_realize_ctx.surrounding_within_inames
            if orig_kernel.iname_tags_of_type(oiname, LocalInameTagBase))

    from pymbolic import var
    outer_local_iname_vars = tuple(
            var(oiname) for oiname in surrounding_local_inames)

    outer_local_iname_sizes = tuple(
            _get_int_iname_size(orig_kernel, oiname)
            for oiname in surrounding_local_inames)

    neutral_var_names = _make_temporaries(
            red_realize_ctx=red_realize_ctx,
            name_based_on="neutral_"+red_iname,
            nvars=nresults,
            shape=(),
            dtypes=reduction_dtypes,
            address_space=AddressSpace.PRIVATE)

    acc_var_names = _make_temporaries(
            red_realize_ctx=red_realize_ctx,
            name_based_on="acc_"+red_iname,
            nvars=nresults,
            shape=outer_local_iname_sizes + (size,),
            dtypes=reduction_dtypes,
            address_space=AddressSpace.LOCAL)

    acc_vars = tuple(var(n) for n in acc_var_names)

    # {{{ add separate iname to carry out the reduction

    # Doing this sheds any odd conditionals that may be active
    # on our red_iname.

    base_exec_iname = red_realize_ctx.var_name_gen("red_"+red_iname)
    red_realize_ctx.domains.append(_make_slab_set(base_exec_iname, size))
    red_realize_ctx.additional_iname_tags[base_exec_iname] \
            = orig_kernel.iname_tags(red_iname)

    # }}}

    neutral, red_realize_ctx.boxed_callables_table[0] = \
            expr.operation.neutral_element(*arg_dtypes,
                    callables_table=red_realize_ctx.boxed_callables_table[0],
                    target=orig_kernel.target)
    init_id = red_realize_ctx.insn_id_gen(
            f"{red_realize_ctx.id_prefix}_{red_iname}_init")
    init_insn = make_assignment(
            id=init_id,
            assignees=tuple(
                acc_var[outer_local_iname_vars + (var(base_exec_iname),)]
                for acc_var in acc_vars),
            expression=neutral,
            within_inames=(
                red_realize_ctx.surrounding_within_inames
                | frozenset([base_exec_iname])),
            within_inames_is_final=True,
            depends_on=frozenset(),
            # Do not inherit predicates: Those might read variables
            # that may not yet be set, and we don't have a great way
            # of figuring out what the dependencies of the accumulator
            # initializer should be.

            # This way, we may initialize a few too many accumulators,
            # but that's better than being incorrect.
            # https://github.com/inducer/loopy/issues/231
            )
    red_realize_ctx.additional_insns.append(init_insn)

    init_neutral_id = red_realize_ctx.insn_id_gen(
            f"{red_realize_ctx.id_prefix}_{red_iname}_init_neutral")
    init_neutral_insn = make_assignment(
            id=init_neutral_id,
            assignees=tuple(var(nvn) for nvn in neutral_var_names),
            expression=neutral,
            within_inames=(
                red_realize_ctx.surrounding_within_inames
                | frozenset([base_exec_iname])),
            within_inames_is_final=True,
            depends_on=frozenset(),
            predicates=red_realize_ctx.surrounding_predicates,
            )
    red_realize_ctx.additional_insns.append(init_neutral_insn)

    transfer_depends_on = {init_neutral_id, init_id}

    transfer_red_realize_ctx = red_realize_ctx.new_subinstruction(
            within_inames=(
                    red_realize_ctx.surrounding_within_inames
                    | frozenset([red_iname])),
            depends_on=(
                red_realize_ctx.surrounding_depends_on
                | frozenset([init_id, init_neutral_id])),
            no_sync_with=(
                red_realize_ctx.surrounding_no_sync_with
                | frozenset([(init_id, "any")])))

    reduction_expr = red_realize_ctx.mapper(
            expr.expr, red_realize_ctx=transfer_red_realize_ctx,
            nresults=1)

    # In the case of a multi-argument reduction, we need a name for each of
    # the arguments in order to pass them to the binary op - so we expand
    # items that are not "plain" tuples here.
    if nresults > 1 and not isinstance(reduction_expr, tuple):
        get_args_insn_id = red_realize_ctx.insn_id_gen(
                f"{red_realize_ctx.id_prefix}_{red_iname}_get")

        reduction_expr = expand_inner_reduction(
                red_realize_ctx=red_realize_ctx,
                id=get_args_insn_id,
                expr=reduction_expr,
                nresults=nresults,
                depends_on=red_realize_ctx.surrounding_depends_on,
                within_inames=transfer_red_realize_ctx.surrounding_within_inames,
                predicates=red_realize_ctx.surrounding_predicates,
                )

        transfer_depends_on.add(get_args_insn_id)

    transfer_id = red_realize_ctx.insn_id_gen(
            f"{red_realize_ctx.id_prefix}_{red_iname}_transfer")
    expression, red_realize_ctx.boxed_callables_table[0] = expr.operation(
            arg_dtypes,
            _strip_if_scalar(
                neutral_var_names,
                tuple(var(nvn) for nvn in neutral_var_names)),
            reduction_expr,
            red_realize_ctx.boxed_callables_table[0],
            orig_kernel.target)
    transfer_insn = make_assignment(
            id=transfer_id,
            assignees=tuple(
                acc_var[outer_local_iname_vars + (var(red_iname),)]
                for acc_var in acc_vars),
            expression=expression,
            **transfer_red_realize_ctx.get_insn_kwargs())
    red_realize_ctx.additional_insns.append(transfer_insn)

    cur_size = 1
    while cur_size < size:
        cur_size *= 2

    prev_id = transfer_id
    bound = size

    stage_exec_iname = None

    istage = 0
    while cur_size > 1:

        new_size = cur_size // 2
        assert new_size * 2 == cur_size

        stage_exec_iname = red_realize_ctx.var_name_gen(
                "red_%s_s%d" % (red_iname, istage))
        red_realize_ctx.domains.append(
                _make_slab_set(stage_exec_iname, bound-new_size))
        red_realize_ctx.additional_iname_tags[stage_exec_iname] \
                = orig_kernel.iname_tags(red_iname)

        stage_id = red_realize_ctx.insn_id_gen(
                "red_%s_stage_%d" % (red_iname, istage))

        expression, red_realize_ctx.boxed_callables_table[0] = expr.operation(
                arg_dtypes,
                _strip_if_scalar(acc_vars, tuple(
                    acc_var[
                        outer_local_iname_vars + (var(stage_exec_iname),)]
                    for acc_var in acc_vars)),
                _strip_if_scalar(acc_vars, tuple(
                    acc_var[
                        outer_local_iname_vars + (
                            var(stage_exec_iname) + new_size,)]
                    for acc_var in acc_vars)),
                red_realize_ctx.boxed_callables_table[0],
                orig_kernel.target)

        stage_insn = make_assignment(
                id=stage_id,
                assignees=tuple(
                    acc_var[outer_local_iname_vars + (var(stage_exec_iname),)]
                    for acc_var in acc_vars),
                expression=expression,
                within_inames=(
                    red_realize_ctx.surrounding_within_inames
                    | frozenset([stage_exec_iname])),
                within_inames_is_final=True,
                depends_on=frozenset([prev_id]),
                predicates=red_realize_ctx.surrounding_predicates,
                )

        red_realize_ctx.additional_insns.append(stage_insn)
        prev_id = stage_id

        cur_size = new_size
        bound = cur_size
        istage += 1

    red_realize_ctx.surrounding_insn_add_depends_on.add(prev_id)
    red_realize_ctx.surrounding_insn_add_no_sync_with.add((prev_id, "any"))
    red_realize_ctx.surrounding_insn_add_within_inames.add(
            stage_exec_iname or base_exec_iname)

    if nresults == 1:
        assert len(acc_vars) == 1
        return acc_vars[0][outer_local_iname_vars + (0,)]
    else:
        return [acc_var[outer_local_iname_vars + (0,)] for acc_var in
                acc_vars]
# }}}


# {{{ utils (stateful)

@memoize_on_first_arg
def _get_or_add_sweep_tracking_iname_and_domain(
        red_realize_ctx,
        scan_param,
        tracking_iname):
    kernel = red_realize_ctx.kernel

    domain = kernel.get_inames_domain(
            frozenset((scan_param.scan_iname, scan_param.sweep_iname)))

    red_realize_ctx.inames_added_for_scan.add(tracking_iname)

    new_domain = _create_domain_for_sweep_tracking(
            domain, tracking_iname, scan_param)

    _insert_subdomain_into_domain_tree(kernel, red_realize_ctx.domains, new_domain)

    return tracking_iname


def replace_var_within_expr(kernel, var_name_gen, expr, from_var, to_var):
    from pymbolic.mapper.substitutor import make_subst_func

    from loopy.symbolic import (
        SubstitutionRuleMappingContext, RuleAwareSubstitutionMapper)

    # FIXME: This is broken. SubstitutionRuleMappingContext produces a new
    # kernel (via finish_kernel) with new subst rules. These get dropped on the
    # floor here.
    rule_mapping_context = SubstitutionRuleMappingContext(
        kernel.substitutions, var_name_gen)

    from pymbolic import var
    mapper = RuleAwareSubstitutionMapper(
        rule_mapping_context,
        make_subst_func({from_var: var(to_var)}),
        within=lambda *args: True)

    return mapper(expr, kernel, None)


def _make_temporaries(
        red_realize_ctx, name_based_on, nvars, shape, dtypes, address_space):
    var_names = [
            red_realize_ctx.var_name_gen(name_based_on.format(index=i))
            for i in range(nvars)]

    from loopy.kernel.data import TemporaryVariable

    for name, dtype in zip(var_names, dtypes):
        red_realize_ctx.additional_temporary_variables[name] = TemporaryVariable(
                name=name,
                shape=shape,
                dtype=dtype,
                address_space=address_space)

    return var_names

# }}}


# {{{ reduction type: sequential scan

def map_scan_seq(red_realize_ctx, expr, nresults, arg_dtypes,
        reduction_dtypes, scan_param):

    track_iname = red_realize_ctx.var_name_gen(
            "{sweep_iname}__seq_scan"
            .format(sweep_iname=scan_param.sweep_iname))

    _get_or_add_sweep_tracking_iname_and_domain(
            red_realize_ctx, scan_param, track_iname)
    red_realize_ctx.additional_iname_tags[track_iname] = frozenset()

    from loopy.kernel.data import AddressSpace
    acc_var_names = _make_temporaries(
            red_realize_ctx=red_realize_ctx,
            name_based_on="acc_" + scan_param.scan_iname,
            nvars=nresults,
            shape=(),
            dtypes=reduction_dtypes,
            address_space=AddressSpace.PRIVATE)

    from pymbolic import var
    acc_vars = tuple(var(n) for n in acc_var_names)

    init_id = red_realize_ctx.insn_id_gen(
            f"{red_realize_ctx.id_prefix}_{'_'.join(expr.inames)}_init")

    init_insn_depends_on = frozenset()

    expression, red_realize_ctx.boxed_callables_table[0] = \
            expr.operation.neutral_element(*arg_dtypes,
                    callables_table=red_realize_ctx.boxed_callables_table[0],
                    target=red_realize_ctx.orig_kernel.target)

    init_insn = make_assignment(
            id=init_id,
            assignees=acc_vars,
            within_inames=(
                red_realize_ctx.surrounding_within_inames
                - frozenset((scan_param.sweep_iname,) + expr.inames)),
            within_inames_is_final=True,
            depends_on=init_insn_depends_on,
            expression=expression,
            # Do not inherit predicates: Those might read variables
            # that may not yet be set, and we don't have a great way
            # of figuring out what the dependencies of the accumulator
            # initializer should be.

            # This way, we may initialize a few too many accumulators,
            # but that's better than being incorrect.
            # https://github.com/inducer/loopy/issues/231
            )

    red_realize_ctx.additional_insns.append(init_insn)

    scan_insn_depends_on = {init_insn.id} | red_realize_ctx.surrounding_depends_on

    scan_red_realize_ctx = red_realize_ctx.new_subinstruction(
            within_inames=(
                red_realize_ctx.surrounding_within_inames
                | frozenset({scan_param.scan_iname})),
            depends_on=red_realize_ctx.surrounding_depends_on)

    reduction_expr = red_realize_ctx.mapper(
            expr.expr, red_realize_ctx=scan_red_realize_ctx,
            nresults=1)

    updated_inner_exprs, scan_insn_depends_on = _preprocess_scan_arguments(
            scan_red_realize_ctx,
            reduction_expr, nresults,
            scan_param.scan_iname, track_iname, scan_insn_depends_on)

    scan_id = red_realize_ctx.insn_id_gen(
            based_on=f"{red_realize_ctx.id_prefix}_{'_'.join(expr.inames)}_scan")

    expression, red_realize_ctx.boxed_callables_table[0] = expr.operation(
            arg_dtypes,
            _strip_if_scalar(acc_vars, acc_vars),
            _strip_if_scalar(acc_vars, updated_inner_exprs),
            red_realize_ctx.boxed_callables_table[0],
            red_realize_ctx.orig_kernel.target)

    scan_insn = make_assignment(
            id=scan_id,
            assignees=acc_vars,
            expression=expression,
            within_inames=(
                red_realize_ctx.surrounding_within_inames
                | frozenset(
                    scan_red_realize_ctx.surrounding_insn_add_within_inames)
                | {track_iname}),
            depends_on=(
                frozenset(scan_insn_depends_on)
                | frozenset(scan_red_realize_ctx.surrounding_insn_add_depends_on)
                ),
            no_sync_with=(
                red_realize_ctx.surrounding_no_sync_with
                | frozenset(scan_red_realize_ctx.surrounding_insn_add_no_sync_with)
                ),
            within_inames_is_final=True,
            predicates=red_realize_ctx.surrounding_predicates,
            )

    red_realize_ctx.additional_insns.append(scan_insn)
    red_realize_ctx.surrounding_insn_add_depends_on.add(scan_insn.id)

    if nresults == 1:
        assert len(acc_vars) == 1
        return acc_vars[0]
    else:
        return acc_vars

# }}}


# {{{ reduction type: local-parallel scan

def map_scan_local(red_realize_ctx, expr, nresults, arg_dtypes,
        reduction_dtypes, scan_param):

    orig_kernel = red_realize_ctx.orig_kernel

    scan_size = _get_int_iname_size(orig_kernel, scan_param.sweep_iname)

    assert scan_size > 0

    if scan_size == 1:
        return map_reduction_seq(red_realize_ctx,
                expr, nresults, arg_dtypes, reduction_dtypes)

    from loopy.kernel.data import LocalInameTagBase
    outer_local_inames = tuple(
            oiname for oiname in red_realize_ctx.surrounding_within_inames
            if orig_kernel.iname_tags_of_type(oiname, LocalInameTagBase)
            and oiname != scan_param.sweep_iname)

    from pymbolic import var
    outer_local_iname_vars = tuple(
            var(oiname) for oiname in outer_local_inames)

    outer_local_iname_sizes = tuple(
            _get_int_iname_size(orig_kernel, oiname)
            for oiname in outer_local_inames)

    track_iname = red_realize_ctx.var_name_gen(
            "{sweep_iname}__pre_scan"
            .format(sweep_iname=scan_param.sweep_iname))

    _get_or_add_sweep_tracking_iname_and_domain(
            red_realize_ctx,
            scan_param,
            track_iname)
    red_realize_ctx.additional_iname_tags[track_iname] = frozenset()

    # {{{ add separate iname to carry out the scan

    # Doing this sheds any odd conditionals that may be active
    # on our scan_iname.

    base_exec_iname = red_realize_ctx.var_name_gen(scan_param.sweep_iname + "__scan")
    red_realize_ctx.domains.append(_make_slab_set(base_exec_iname, scan_size))
    red_realize_ctx.additional_iname_tags[base_exec_iname] \
            = orig_kernel.iname_tags(scan_param.sweep_iname)

    # }}}

    read_var_names = _make_temporaries(
            red_realize_ctx=red_realize_ctx,
            name_based_on="read_"+scan_param.scan_iname+"_arg_{index}",
            nvars=nresults,
            shape=(),
            dtypes=reduction_dtypes,
            address_space=AddressSpace.PRIVATE)

    acc_var_names = _make_temporaries(
            red_realize_ctx=red_realize_ctx,
            name_based_on="acc_"+scan_param.scan_iname,
            nvars=nresults,
            shape=outer_local_iname_sizes + (scan_size,),
            dtypes=reduction_dtypes,
            address_space=AddressSpace.LOCAL)

    acc_vars = tuple(var(n) for n in acc_var_names)
    read_vars = tuple(var(n) for n in read_var_names)

    base_iname_deps = (
            red_realize_ctx.surrounding_within_inames
            - frozenset([scan_param.sweep_iname]))

    neutral, red_realize_ctx.boxed_callables_table[0] = \
            expr.operation.neutral_element(*arg_dtypes,
                    callables_table=red_realize_ctx.boxed_callables_table[0],
                    target=orig_kernel.target)

    init_id = red_realize_ctx.insn_id_gen(
            f"{red_realize_ctx.id_prefix}_{scan_param.scan_iname}_init")
    init_insn = make_assignment(
            id=init_id,
            assignees=tuple(
                acc_var[outer_local_iname_vars + (var(base_exec_iname),)]
                for acc_var in acc_vars),
            expression=neutral,
            within_inames=base_iname_deps | frozenset([base_exec_iname]),
            within_inames_is_final=True,
            depends_on=frozenset(),
            # Do not inherit predicates: Those might read variables
            # that may not yet be set, and we don't have a great way
            # of figuring out what the dependencies of the accumulator
            # initializer should be.

            # This way, we may initialize a few too many accumulators,
            # but that's better than being incorrect.
            # https://github.com/inducer/loopy/issues/231
            )
    red_realize_ctx.additional_insns.append(init_insn)

    transfer_insn_depends_on = (
            frozenset({init_insn.id})
            | red_realize_ctx.surrounding_depends_on)

    transfer_red_realize_ctx = red_realize_ctx.new_subinstruction(
            within_inames=(
                red_realize_ctx.surrounding_within_inames
                | frozenset({scan_param.scan_iname})),
            depends_on=red_realize_ctx.surrounding_depends_on)

    reduction_expr = red_realize_ctx.mapper(
            expr.expr, red_realize_ctx=transfer_red_realize_ctx,
            nresults=1)

    updated_inner_exprs, transfer_insn_depends_on = _preprocess_scan_arguments(
            red_realize_ctx,
            reduction_expr, nresults,
            scan_param.scan_iname, track_iname, transfer_insn_depends_on)

    from loopy.symbolic import Reduction
    pre_scan_reduction = Reduction(
                operation=expr.operation,
                inames=(track_iname,),
                expr=_strip_if_scalar(acc_vars, updated_inner_exprs),
                allow_simultaneous=False,
                )

    pre_scan_result = red_realize_ctx.mapper(
            pre_scan_reduction, red_realize_ctx=transfer_red_realize_ctx,
            nresults=len(acc_vars))

    from loopy.symbolic import pw_aff_to_expr
    sweep_lower_bound_expr = pw_aff_to_expr(scan_param.sweep_lower_bound)

    if nresults == 1:
        assert not isinstance(pre_scan_result, tuple)
        pre_scan_result = (pre_scan_result,)

    transfer_ids = frozenset()
    for acc_var, pre_scan_result_i in zip(acc_vars, pre_scan_result):
        transfer_id = red_realize_ctx.insn_id_gen(
                f"{red_realize_ctx.id_prefix}_{scan_param.scan_iname}_transfer")
        transfer_insn = make_assignment(
                id=transfer_id,
                assignees=(acc_var[outer_local_iname_vars
                    + (var(scan_param.sweep_iname) - sweep_lower_bound_expr,)],),
                expression=pre_scan_result_i,
                within_inames=(
                    red_realize_ctx.surrounding_within_inames
                    | transfer_red_realize_ctx.surrounding_insn_add_within_inames
                    | frozenset({scan_param.sweep_iname})),
                within_inames_is_final=True,
                depends_on=(
                    transfer_insn_depends_on
                    | transfer_red_realize_ctx.surrounding_insn_add_depends_on),
                no_sync_with=(
                    frozenset([(init_id, "any")])
                    | transfer_red_realize_ctx.surrounding_insn_add_no_sync_with),
                predicates=red_realize_ctx.surrounding_predicates,
                )

        red_realize_ctx.additional_insns.append(transfer_insn)
        transfer_ids = transfer_ids | frozenset({transfer_id})

    del transfer_id

    prev_ids = transfer_ids

    istage = 0
    cur_size = 1

    while cur_size < scan_size:
        stage_exec_iname = red_realize_ctx.var_name_gen(
                f"{scan_param.sweep_iname}__scan_s{istage}")
        red_realize_ctx.domains.append(
                _make_slab_set_from_range(stage_exec_iname, cur_size, scan_size))
        red_realize_ctx.additional_iname_tags[stage_exec_iname] \
                = orig_kernel.iname_tags(scan_param.sweep_iname)

        for read_var, acc_var in zip(read_vars, acc_vars):
            read_stage_id = red_realize_ctx.insn_id_gen(
                    f"scan_{scan_param.scan_iname}_read_stage_{istage}")

            read_stage_insn = make_assignment(
                    id=read_stage_id,
                    assignees=(read_var,),
                    expression=(
                            acc_var[
                                outer_local_iname_vars
                                + (var(stage_exec_iname) - cur_size,)]),
                    within_inames=(
                        base_iname_deps | frozenset([stage_exec_iname])),
                    within_inames_is_final=True,
                    depends_on=prev_ids,
                    predicates=red_realize_ctx.surrounding_predicates,
                    )

            if cur_size == 1:
                # Performance hack: don't add a barrier here with transfer_insn.
                # NOTE: This won't work if the way that local inames
                # are lowered changes.
                read_stage_insn = read_stage_insn.copy(
                        no_sync_with=(
                            read_stage_insn.no_sync_with
                            | frozenset([(tid, "any") for tid in transfer_ids])))

            red_realize_ctx.additional_insns.append(read_stage_insn)
            prev_ids = frozenset({read_stage_id})

        write_stage_id = red_realize_ctx.insn_id_gen(
                f"scan_{scan_param.scan_iname}_write_stage_{istage}")

        expression, red_realize_ctx.boxed_callables_table[0] = expr.operation(
            arg_dtypes,
            _strip_if_scalar(acc_vars, read_vars),
            _strip_if_scalar(acc_vars, tuple(
                acc_var[
                    outer_local_iname_vars + (var(stage_exec_iname),)]
                for acc_var in acc_vars)),
            red_realize_ctx.boxed_callables_table[0],
            orig_kernel.target)

        write_stage_insn = make_assignment(
                id=write_stage_id,
                assignees=tuple(
                    acc_var[outer_local_iname_vars + (var(stage_exec_iname),)]
                    for acc_var in acc_vars),
                expression=expression,
                within_inames=(
                    base_iname_deps | frozenset([stage_exec_iname])),
                within_inames_is_final=True,
                depends_on=prev_ids,
                predicates=red_realize_ctx.surrounding_predicates,
                )

        red_realize_ctx.additional_insns.append(write_stage_insn)
        prev_ids = frozenset({write_stage_id})

        cur_size *= 2
        istage += 1

    red_realize_ctx.surrounding_insn_add_depends_on.update(prev_ids)
    red_realize_ctx.surrounding_insn_add_within_inames.add(scan_param.sweep_iname)

    output_idx = var(scan_param.sweep_iname) - sweep_lower_bound_expr

    if nresults == 1:
        assert len(acc_vars) == 1
        return acc_vars[0][outer_local_iname_vars + (output_idx,)]
    else:
        return [acc_var[outer_local_iname_vars + (output_idx,)]
                for acc_var in acc_vars]

# }}}


# {{{ top-level dispatch among reduction types

def map_reduction(expr, *, red_realize_ctx, nresults):
    kernel_with_updated_domains = red_realize_ctx.kernel.copy(
            domains=red_realize_ctx.domains)

    from loopy.type_inference import (
            infer_arg_and_reduction_dtypes_for_reduction_expression)
    arg_dtypes, reduction_dtypes = (
            infer_arg_and_reduction_dtypes_for_reduction_expression(
                kernel_with_updated_domains, expr,
                red_realize_ctx.boxed_callables_table[0],
                red_realize_ctx.unknown_types_ok))

    bad_inames = frozenset(expr.inames) & red_realize_ctx.surrounding_within_inames
    if bad_inames:
        raise LoopyError("reduction used within loop(s) that it was "
                "supposed to reduce over: " + ", ".join(bad_inames))

    iname_classes = _classify_reduction_inames(red_realize_ctx, expr.inames)

    n_sequential = len(iname_classes.sequential)
    n_local_par = len(iname_classes.local_parallel)
    n_nonlocal_par = len(iname_classes.nonlocal_parallel)

    really_force_scan = red_realize_ctx.force_scan and (
            len(expr.inames) != 1
            or expr.inames[0] not in red_realize_ctx.inames_added_for_scan)

    def _error_if_force_scan_on(cls, msg):
        if really_force_scan:
            raise cls(msg)

    may_be_implemented_as_scan = False
    if red_realize_ctx.force_scan or red_realize_ctx.automagic_scans_ok:
        try:
            # Try to determine scan candidate information (sweep iname, scan
            # iname, etc).
            scan_param = _try_infer_scan_candidate_from_expr(
                    kernel_with_updated_domains, expr,
                    red_realize_ctx.surrounding_within_inames,
                    sweep_iname=red_realize_ctx.force_outer_iname_for_scan)

        except ValueError as v:
            error = str(v)

        else:
            # Ensures the reduction is triangular (somewhat expensive).
            may_be_implemented_as_scan, error = _check_reduction_is_triangular(
                        kernel_with_updated_domains, expr, scan_param)

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
        raise LoopyError("Reduction over '%s' contains more than "
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
                   ", ".join(str(red_realize_ctx.orig_kernel.iname_tags(iname))
                             for iname in bad_inames)))

    # }}}

    red_realize_ctx.changes_made()

    if n_local_par == 0 and n_sequential == 0:
        warn_with_kernel(red_realize_ctx.kernel, "empty_reduction",
                "Empty reduction found (no inames to reduce over). "
                "Eliminating.")

        # We're not supposed to reduce/sum at all. (Note how this is distinct
        # from an empty reduction--there is an element here, just no inames
        # to reduce over. It's rather similar to an array with () shape in
        # numpy.)

        return expr.expr

    if may_be_implemented_as_scan:
        assert red_realize_ctx.force_scan or red_realize_ctx.automagic_scans_ok

        # We require the "scan" iname to be tagged sequential.
        if n_sequential:
            sweep_iname = scan_param.sweep_iname
            sweep_class = _classify_reduction_inames(red_realize_ctx, (sweep_iname,))

            sequential = sweep_iname in sweep_class.sequential
            parallel = sweep_iname in sweep_class.local_parallel
            bad_parallel = sweep_iname in sweep_class.nonlocal_parallel

            if sweep_iname not in red_realize_ctx.surrounding_within_inames:
                _error_if_force_scan_on(LoopyError,
                        "Sweep iname '%s' was detected, but is not an iname "
                        "for the instruction." % sweep_iname)
            elif bad_parallel:
                _error_if_force_scan_on(LoopyError,
                        "Sweep iname '%s' has an unsupported parallel tag '%s' "
                        "- the only parallelism allowed is 'local'." %
                        (sweep_iname,
                         ", ".join(tag.key
                        for tag in red_realize_ctx.kernel.iname_tags(sweep_iname))))
            elif parallel:
                return map_scan_local(red_realize_ctx, expr, nresults,
                        arg_dtypes, reduction_dtypes, scan_param)
            elif sequential:
                return map_scan_seq(red_realize_ctx, expr, nresults,
                        arg_dtypes, reduction_dtypes, scan_param)

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
                red_realize_ctx,
                expr, nresults, arg_dtypes, reduction_dtypes)
    else:
        assert n_local_par > 0
        return map_reduction_local(
                red_realize_ctx,
                expr, nresults, arg_dtypes,
                reduction_dtypes)

# }}}


# {{{ realize_reduction_for_single_kernel

# @remove_any_newly_unused_inames
def realize_reduction_for_single_kernel(kernel, callables_table,
        insn_id_filter=None, unknown_types_ok=True, automagic_scans_ok=False,
        force_scan=False, force_outer_iname_for_scan=None):
    logger.debug("%s: realize reduction" % kernel.name)

    orig_kernel = kernel

    finished_insns = []

    insn_id_gen = kernel.get_instruction_id_generator()
    var_name_gen = kernel.get_var_name_generator()

    cb_mapper = RealizeReductionCallbackMapper(map_reduction)

    insn_queue = kernel.instructions[:]
    domains = kernel.domains[:]

    inames_added_for_scan = set()

    kernel_changed = False

    if insn_id_filter is None:
        insn_id_filter = set(kernel.id_to_insn)
    elif isinstance(insn_id_filter, str):
        insn_id_filter = {insn_id_filter}
    else:
        from collections.abc import Collection
        if not isinstance(insn_id_filter, Collection):
            raise LoopyError("'insn_id_filter' can be either None, a string or a"
                             f" collection of strings. Got {type(insn_id_filter)}.")
        else:
            insn_id_filter = set(insn_id_filter)

    while insn_queue:
        insn = insn_queue.pop(0)

        red_realize_ctx = _ReductionRealizationContext(
                mapper=cb_mapper,

                force_scan=force_scan,
                automagic_scans_ok=automagic_scans_ok,
                unknown_types_ok=unknown_types_ok,
                force_outer_iname_for_scan=force_outer_iname_for_scan,

                orig_kernel=orig_kernel,
                kernel=kernel,

                id_prefix=insn.id,

                insn_id_gen=insn_id_gen,
                var_name_gen=var_name_gen,

                additional_temporary_variables={},
                additional_insns=[],
                domains=domains,
                additional_iname_tags={},
                boxed_callables_table=[callables_table],

                inames_added_for_scan=inames_added_for_scan,

                surrounding_within_inames=insn.within_inames,
                surrounding_depends_on=insn.depends_on,
                surrounding_no_sync_with=insn.no_sync_with,
                surrounding_predicates=insn.predicates,

                surrounding_insn_add_within_inames=set(),
                surrounding_insn_add_depends_on=set(),
                surrounding_insn_add_no_sync_with=set(),

                _change_flag=_ChangeFlag(changes_made=False)
                )

        if insn.id not in insn_id_filter \
                or not isinstance(insn, MultiAssignmentBase):
            finished_insns.append(insn)
            continue

        nresults = len(insn.assignees)

        # Run reduction expansion.
        from loopy.symbolic import Reduction
        if isinstance(insn.expression, Reduction) and nresults > 1:
            new_expressions = cb_mapper(insn.expression,
                    red_realize_ctx=red_realize_ctx,
                    nresults=nresults)
        else:
            new_expressions = cb_mapper(insn.expression,
                    red_realize_ctx=red_realize_ctx,
                    nresults=1),

        if red_realize_ctx.were_changes_made:
            # An expansion happened, so insert the generated stuff plus
            # ourselves back into the queue.

            # {{{ apply changes

            kernel_changed = True

            callables_table = red_realize_ctx.boxed_callables_table[0]

            result_assignment_dep_on = (
                    insn.depends_on
                    | frozenset(red_realize_ctx.surrounding_insn_add_depends_on))
            kwargs = insn.get_copy_kwargs(
                    no_sync_with=insn.no_sync_with
                    | frozenset(red_realize_ctx.surrounding_insn_add_no_sync_with),
                    within_inames=(
                        insn.within_inames
                        | red_realize_ctx.surrounding_insn_add_within_inames))

            kwargs.pop("id")
            kwargs.pop("depends_on")
            kwargs.pop("expression")
            kwargs.pop("assignee", None)
            kwargs.pop("assignees", None)
            kwargs.pop("temp_var_type", None)
            kwargs.pop("temp_var_types", None)

            insn_id_replacements = {}

            if isinstance(insn.expression, Reduction) and nresults > 1:
                result_assignment_ids = [
                        insn_id_gen(insn.id) for i in range(nresults)]
                replacement_insns = [
                        Assignment(
                            id=result_assignment_ids[i],
                            depends_on=(
                                result_assignment_dep_on
                                | (frozenset([result_assignment_ids[i-1]])
                                    if i else frozenset())),
                            assignee=assignee,
                            expression=new_expr,
                            **kwargs)
                        for i, (assignee, new_expr) in enumerate(zip(
                            insn.assignees, new_expressions))]

                insn_id_replacements[insn.id] = [
                    rinsn.id for rinsn in replacement_insns]
            else:
                new_expr, = new_expressions
                # since we are replacing the instruction with
                # only one instruction, there's no need to replace id
                replacement_insns = [
                        make_assignment(
                            id=insn.id,
                            depends_on=result_assignment_dep_on,
                            assignees=insn.assignees,
                            expression=new_expr,
                            **kwargs)
                        ]

            additional_insns = red_realize_ctx.additional_insns

            # {{{ make additional insns depend on most recent global barrier

            # FIXME This is weird and hokey and ad-hoc and probably broken.
            # I *think* the idea is to keep a reduction/scan implementation
            # from crossing a global barrier, because that would be costly.

            # check first that the original kernel had global barriers
            # if not, we don't need to check. Since the function
            # kernel_has_global_barriers is cached, we don't do
            # extra work compared to not checking.

            from loopy.kernel.tools import (
                    kernel_has_global_barriers, find_most_recent_global_barrier)

            if kernel_has_global_barriers(orig_kernel):
                global_barrier = find_most_recent_global_barrier(kernel, insn.id)

                if global_barrier is not None:
                    gb_dep = frozenset([global_barrier])
                    additional_insns = [addl_insn.copy(
                        depends_on=addl_insn.depends_on | gb_dep)
                        for addl_insn in additional_insns]

            # }}}

            # The reduction expander needs an up-to-date kernel
            # object to find dependencies. Keep kernel up-to-date.
            new_temporary_variables = kernel.temporary_variables.copy()
            new_temporary_variables.update(
                    red_realize_ctx.additional_temporary_variables)

            finished_insns = [
                    replace_instruction_ids_in_insn(insn, insn_id_replacements)
                    for insn in finished_insns]
            insn_queue = [
                    replace_instruction_ids_in_insn(insn, insn_id_replacements)
                    for insn in insn_queue]

            finished_insns.extend(additional_insns)
            finished_insns.extend(replacement_insns)

            kernel = kernel.copy(
                    instructions=finished_insns + insn_queue,
                    temporary_variables=new_temporary_variables,
                    domains=domains)
            from loopy.transform.iname import tag_inames
            kernel = tag_inames(kernel, red_realize_ctx.additional_iname_tags)

            del insn_id_replacements

            # }}}

        else:
            # nothing happened, we're done with insn
            assert not red_realize_ctx.surrounding_insn_add_depends_on

            finished_insns.append(insn)

    if not kernel_changed:
        return orig_kernel, callables_table

    kernel = _hackily_ensure_multi_assignment_return_values_are_scoped_private(
                kernel)

    return kernel, callables_table

# }}}


def realize_reduction(t_unit, *args, **kwargs):
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

    :arg insn_id_filter: Can be one of:
        - An instance of :class:`str` specifying the ID of the instruction
          whose reductions are to be realized.
        - A collection of :class:`str` specifying the IDs of the instructions
          whose reductions are to be realized.
    """

    assert isinstance(t_unit, TranslationUnit)

    callables_table = dict(t_unit.callables_table)
    kernels_to_scan = [in_knl_callable.subkernel
            for in_knl_callable in t_unit.callables_table.values()
            if isinstance(in_knl_callable, CallableKernel)]

    for knl in kernels_to_scan:
        new_knl, callables_table = realize_reduction_for_single_kernel(
                knl, callables_table, *args, **kwargs)
        in_knl_callable = callables_table[knl.name].copy(
                subkernel=new_knl)
        callables_table[knl.name] = in_knl_callable

    return t_unit.copy(callables_table=Map(callables_table))

# vim: foldmethod=marker

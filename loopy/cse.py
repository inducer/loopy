from __future__ import division

import islpy as isl
from islpy import dim_type
from loopy.kernel import AutoFitLocalIndexTag
import numpy as np

from pytools import Record
from pymbolic import var




def check_cse_iname_deps(iname, duplicate_inames, tag, dependencies,
        target_var_is_local, cse):
    from loopy.kernel import (LocalIndexTagBase, GroupIndexTag, IlpTag)

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
                    % (cse.prefix, iname, tag))

    if iname in duplicate_inames and kind == "g":
        raise RuntimeError("duplicating the iname '%s' into "
                "group index axes is not helpful, as they cannot "
                "collaborate in computing a local variable"
                %iname)

    if iname in dependencies:
        if not target_var_is_local and iname in duplicate_inames and kind == "l":
            raise RuntimeError("invalid: hardware-parallelized "
                    "fetch into private variable")

        return

    # the iname is *not* a dependency of the fetch expression
    if iname in duplicate_inames:
        raise RuntimeError("duplicating an iname ('%s') "
                "that the CSE ('%s') does not depend on "
                "does not make sense" % (iname, cse.child))




class CSEDescriptor(Record):
    __slots__ = ["insn", "cse", "independent_inames",
            "lead_index_exprs"]




def to_parameters_or_project_out(param_inames, set_inames, set):
    for iname in set.get_space().get_var_dict().keys():
        if iname in param_inames:
            dt, idx = set.get_space().get_var_dict()[iname]
            set = set.move_dims(
                    dim_type.param, set.dim(dim_type.param),
                    dt, idx, 1)
        elif iname in set_inames:
            pass
        else:
            dt, idx = set.get_space().get_var_dict()[iname]
            set = set.project_out(dt, idx, 1)

    return set




def solve_affine_equations_for_lhs(targets, equations, parameters):
    # Not a very good solver: The desired variable must already
    # occur with a coefficient of 1 on the lhs, and with no other
    # targets on that lhs.

    from loopy.symbolic import CoefficientCollector
    coeff_coll = CoefficientCollector()

    target_values = {}

    for lhs, rhs in equations:
        lhs_coeffs = coeff_coll(lhs)
        rhs_coeffs = coeff_coll(rhs)

        def shift_to_rhs(key):
            rhs_coeffs[key] = rhs_coeffs.get(key, 0) - lhs_coeffs[key]
            del lhs_coeffs[key]

        for key in list(lhs_coeffs.iterkeys()):
            if key in targets:
                continue
            elif key in parameters or key == 1:
                shift_to_rhs(key)
            else:
                raise RuntimeError("unexpected key")

        if len(lhs_coeffs) > 1:
            raise RuntimeError("comically unable to solve '%s = %s' "
                    "for one of the target variables '%s'"
                    % (lhs, rhs, ",".join(targets)))

        (tgt_name, coeff), = lhs_coeffs.iteritems()
        if coeff != 1:
            raise RuntimeError("comically unable to solve '%s = %s' "
                    "for one of the target variables '%s'"
                    % (lhs, rhs, ",".join(targets)))

        solution = 0
        for key, coeff in rhs_coeffs.iteritems():
            if key == 1:
                solution += coeff
            else:
                solution += coeff*var(key)

        assert tgt_name not in target_values
        target_values[tgt_name] = solution

    return [target_values[tname] for tname in targets]




def process_cses(kernel, lead_csed, cse_descriptors):
    from pymbolic.mapper.unifier import BidirectionalUnifier

    # {{{ parameter set/dependency finding

    from loopy.symbolic import DependencyMapper
    internal_dep_mapper = DependencyMapper(composite_leaves=False)

    def get_deps(expr):
        return set(dep.name for dep in internal_dep_mapper(expr))

    # Everything that is not one of the duplicate/independent inames
    # is turned into a parameter.

    lead_csed.independent_inames = set(lead_csed.independent_inames)
    lead_deps = get_deps(lead_csed.cse.child) & kernel.all_inames()
    params = lead_deps - set(lead_csed.independent_inames)

    # }}}

    lead_domain = to_parameters_or_project_out(params,
            lead_csed.independent_inames, kernel.domain)
    lead_space = lead_domain.get_space()

    footprint = lead_domain

    uni_recs = []
    for csed in cse_descriptors:
        # {{{ find dependencies

        cse_deps = get_deps(csed.cse.child) & kernel.all_inames()
        csed.independent_inames = cse_deps - params

        # }}}

        # {{{ find unifier

        unif = BidirectionalUnifier(
                lhs_mapping_candidates=lead_csed.independent_inames,
                rhs_mapping_candidates=csed.independent_inames)
        unifiers = unif(lead_csed.cse.child, csed.cse.child)
        if not unifiers:
            raise RuntimeError("Unable to unify  "
            "CSEs '%s' and '%s'" % (lead_csed.cse.child, csed.cse.child))

        # }}}

        found_good_unifier = False

        for unifier in unifiers:
            # {{{ construct, check mapping

            rhs_domain = to_parameters_or_project_out(
                    params, csed.independent_inames, kernel.domain)
            rhs_space = rhs_domain.get_space()

            map_space = lead_space
            ln = lead_space.dim(dim_type.set)
            map_space = map_space.move_dims(dim_type.in_, 0, dim_type.set, 0, ln)
            rn = rhs_space.dim(dim_type.set)
            map_space = map_space.add_dims(dim_type.out, rn)
            for i, iname in enumerate(csed.independent_inames):
                map_space = map_space.set_dim_name(dim_type.out, i, iname+"'")

            set_space = map_space.move_dims(
                    dim_type.out, rn,
                    dim_type.in_, 0, ln).range()

            var_map = None

            from loopy.symbolic import aff_from_expr, PrimeAdder
            add_primes = PrimeAdder(csed.independent_inames)
            for lhs, rhs in unifier.equations:
                cns = isl.Constraint.equality_from_aff(
                        aff_from_expr(set_space, lhs - add_primes(rhs)))

                cns_map = isl.BasicMap.from_constraint(cns)
                if var_map is None:
                    var_map = cns_map
                else:
                    var_map = var_map.intersect(cns_map)

            var_map = var_map.move_dims(
                    dim_type.in_, 0,
                    dim_type.out, rn, ln)

            restr_rhs_map = (
                    isl.Map.from_basic_map(var_map)
                    .intersect_range(rhs_domain))

            # Sanity check: If the range of the map does not recover the
            # domain of the expression, the unifier must have been no
            # good.
            if restr_rhs_map.range() != rhs_domain:
                continue

            # Sanity check: Injectivity here means that unique lead indices
            # can be found for each 

            if not var_map.is_injective():
                raise RuntimeError("In CSEs '%s' and '%s': "
                        "cannot find lead indices uniquely"
                        % (lead_csed.cse.child, csed.cse.child))

            lead_index_set = restr_rhs_map.domain()

            footprint = footprint.union(lead_index_set)

            # FIXME: This restriction will be lifted in the future, and the
            # footprint will instead be used as the lead domain.

            if not lead_index_set.is_subset(lead_domain):
                raise RuntimeError("Index range of CSE '%s' does not cover a "
                        "subset of lead CSE '%s'"
                        % (csed.cse.child, lead_csed.cse.child))

            found_good_unifier = True

            # }}}

        if not found_good_unifier:
            raise RuntimeError("No valid unifier for '%s' and '%s'"
                    % (csed.cse.child, lead_csed.cse.child))

        uni_recs.append(unifier)

        # {{{ solve for lead indices

        csed.lead_index_exprs = solve_affine_equations_for_lhs(
                lead_csed.independent_inames,
                unifier.equations, params)

        # }}}

    return footprint.coalesce()





def make_compute_insn(kernel, lead_csed, target_var_name, target_var_is_local,
        independent_inames, new_inames, ind_iname_to_tag):
    insn = lead_csed.insn

    # {{{ decide whether to force a dep

    forced_iname_deps = set()

    from loopy.symbolic import IndexVariableFinder
    dependencies = IndexVariableFinder(
            include_reduction_inames=False)(lead_csed.cse.child)

    parent_inames = insn.all_inames() | insn.reduction_inames()
    assert dependencies <= parent_inames

    for iname in parent_inames:
        if iname in independent_inames:
            tag = ind_iname_to_tag[iname]
        else:
            tag = kernel.iname_to_tag.get(iname)

        check_cse_iname_deps(
                iname, independent_inames, tag, dependencies,
                target_var_is_local, lead_csed.cse)

    # }}}

    assignee = var(target_var_name)

    if new_inames:
        assignee = assignee[tuple(
            var(iname) for iname in new_inames
            )]

    from loopy.symbolic import SubstitutionMapper
    from pymbolic.mapper.substitutor import make_subst_func
    subst_map = SubstitutionMapper(make_subst_func(
        dict(
            (old_iname, var(new_iname))
            for old_iname, new_iname in zip(independent_inames, new_inames))))
    new_inner_expr = subst_map(lead_csed.cse.child)

    insn_prefix = lead_csed.cse.prefix
    if insn_prefix is None:
        insn_prefix = "cse"
    from loopy.kernel import Instruction
    return Instruction(
            id=kernel.make_unique_instruction_id(based_on=insn_prefix+"_compute"),
            assignee=assignee,
            expression=new_inner_expr,
            forced_iname_deps=forced_iname_deps)




def realize_cse(kernel, cse_tag, dtype, independent_inames=[],
        ind_iname_to_tag={}, new_inames=None, default_tag="l.auto",
        follow_tag=None):
    """
    :arg independent_inames: which inames are supposed to be separate loops
        in the CSE. Also determines index order of temporary array.
    """

    if not set(independent_inames) <= kernel.all_inames():
        raise ValueError("In CSE realization for '%s': "
                "cannot make inames '%s' independent--"
                "they don't already exist" % (
                    cse_tag,
                    ",".join(
                        set(independent_inames)-kernel.all_inames())))

    # {{{ process parallel_inames and ind_iname_to_tag arguments

    ind_iname_to_tag = ind_iname_to_tag.copy()

    from loopy.kernel import parse_tag
    default_tag = parse_tag(default_tag)
    for iname in independent_inames:
        ind_iname_to_tag.setdefault(iname, default_tag)

    if not set(ind_iname_to_tag.iterkeys()) <= set(independent_inames):
        raise RuntimeError("tags for non-new inames may not be passed")

    # here, all information is consolidated into ind_iname_to_tag

    # }}}

    # {{{ process new_inames argument, think of new inames for inames to be duplicated

    if new_inames is None:
        new_inames = [None] * len(independent_inames)

    if len(new_inames) != len(independent_inames):
        raise ValueError("If given, the new_inames argument must have the "
                "same length as independent_inames")

    temp_new_inames = []
    for old_iname, new_iname in zip(independent_inames, new_inames):
        if new_iname is None:
            if cse_tag is not None:
                based_on = old_iname+"_"+cse_tag
            else:
                based_on = old_iname

            new_iname = kernel.make_unique_var_name(based_on, set(temp_new_inames))
            assert new_iname != old_iname

        temp_new_inames.append(new_iname)

    new_inames = temp_new_inames

    # }}}

    from loopy.isl_helpers import duplicate_axes
    new_domain = duplicate_axes(kernel.domain, independent_inames, new_inames)

    # {{{ gather cse descriptors

    eligible_tags = [cse_tag]
    if follow_tag is not None:
        eligible_tags.append(follow_tag)

    cse_descriptors = []

    def gather_cses(cse, rec):
        if cse.prefix not in eligible_tags:
            rec(cse.child)
            return

        cse_descriptors.append(
                CSEDescriptor(insn=insn, cse=cse))
        # can't nest, don't recurse

    from loopy.symbolic import CSECallbackMapper
    cse_cb_mapper = CSECallbackMapper(gather_cses)

    for insn in kernel.instructions:
        cse_cb_mapper(insn.expression)

    # }}}

    # {{{ find/pick the lead cse

    if not cse_descriptors:
        raise RuntimeError("no CSEs tagged '%s' found" % cse_tag)

    lead_cse_indices = [i for i, csed in enumerate(cse_descriptors) 
            if csed.cse.prefix == cse_tag]
    if follow_tag is not None:
        if len(lead_cse_indices) != 1:
            raise RuntimeError("%d lead CSEs (should be exactly 1) found for tag '%s'"
                    % (len(lead_cse_indices), cse_tag))

        lead_idx, = lead_cse_indices
    else:
        # pick a lead CSE at random
        lead_idx = 0

    lead_csed = cse_descriptors.pop(lead_idx)
    lead_csed.independent_inames = independent_inames

    # }}}

    # FIXME: Do something with the footprint
    footprint = process_cses(kernel, lead_csed, cse_descriptors)

    # {{{ set up temp variable

    var_base = cse_tag
    if var_base is None:
        var_base = "cse"
    target_var_name = kernel.make_unique_var_name(var_base)

    from loopy.kernel import LocalIndexTagBase
    target_var_is_local = any(
            isinstance(tag, LocalIndexTagBase)
            for tag in ind_iname_to_tag.itervalues())

    from loopy.kernel import (TemporaryVariable,
            find_var_base_indices_and_shape_from_inames)

    target_var_base_indices, target_var_shape = \
            find_var_base_indices_and_shape_from_inames(
                    new_domain, new_inames)

    new_temporary_variables = kernel.temporary_variables.copy()
    new_temporary_variables[target_var_name] = TemporaryVariable(
            name=target_var_name,
            dtype=np.dtype(dtype),
            base_indices=target_var_base_indices,
            shape=target_var_shape,
            is_local=target_var_is_local)

    # }}}

    compute_insn = make_compute_insn(
            kernel, lead_csed, target_var_name, target_var_is_local,
            independent_inames, new_inames, ind_iname_to_tag)

    # {{{ substitute variable references into instructions

    def subst_cses(cse, rec):
        if cse is lead_csed.cse:
            csed = lead_csed

            lead_indices = [var(iname) for iname in independent_inames]
        else:
            found = False
            for csed in cse_descriptors:
                if cse is csed.cse:
                    found = True
                    break

            if not found:
                from pymbolic.primitives import CommonSubexpression
                return CommonSubexpression(
                        rec(cse.child), cse.prefix)

            lead_indices = csed.lead_index_exprs

        new_outer_expr = var(target_var_name)
        if lead_indices:
            new_outer_expr = new_outer_expr[tuple(lead_indices)]

        return new_outer_expr
        # can't nest, don't recurse

    cse_cb_mapper = CSECallbackMapper(subst_cses)

    new_insns = [compute_insn]

    for insn in kernel.instructions:
        new_expr = cse_cb_mapper(insn.expression)
        new_insns.append(insn.copy(
            expression=new_expr,
            forced_iname_deps=insn.all_inames()))

    # }}}

    new_iname_to_tag = kernel.iname_to_tag.copy()
    for old_iname, new_iname in zip(independent_inames, new_inames):
        new_iname_to_tag[new_iname] = ind_iname_to_tag[old_iname]

    return kernel.copy(
            domain=new_domain,
            instructions=new_insns,
            temporary_variables=new_temporary_variables,
            iname_to_tag=new_iname_to_tag)





# vim: foldmethod=marker

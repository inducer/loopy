from __future__ import division

import islpy as isl
from islpy import dim_type
from loopy.symbolic import get_dependencies, SubstitutionMapper
from pymbolic.mapper.substitutor import make_subst_func
import numpy as np

from pytools import Record
from pymbolic import var




def check_cse_iname_deps(iname, duplicate_inames, tag, dependencies, cse_tag, uni_template):
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
        if kind == "i":
            raise RuntimeError(
                    "When realizing CSE with tag '%s', encountered iname "
                    "'%s' which is depended upon by the CSE and tagged "
                    "'%s', but not duplicated. The CSE would "
                    "inherit this iname, which would lead to a write race. "
                    "A likely solution of this problem is to also duplicate this "
                    "iname."
                    % (cse_tag, iname, tag))

    if iname in duplicate_inames and kind == "g":
        raise RuntimeError("duplicating the iname '%s' into "
                "group index axes is not helpful, as they cannot "
                "collaborate in computing a local/private variable"
                %iname)

    if iname in dependencies:
        return

    # the iname is *not* a dependency of the fetch expression
    if iname in duplicate_inames:
        raise RuntimeError("duplicating an iname ('%s') "
                "that the CSE ('%s') does not depend on "
                "does not make sense" % (iname, uni_template))




class CSEDescriptor(Record):
    __slots__ = ["insn", "cse", "independent_inames",
            "unif_var_dict"]




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




def process_cses(kernel, uni_template,
        independent_inames, matching_vars, cse_descriptors):
    if not independent_inames:
        for csed in cse_descriptors:
            csed.lead_index_exprs = []
        return None

    from loopy.symbolic import UnidirectionalUnifier

    ind_inames_set = set(independent_inames)

    uni_iname_list = independent_inames + matching_vars
    footprint = None

    uni_recs = []
    matching_var_values = {}

    for csed in cse_descriptors:
        # {{{ find unifier

        unif = UnidirectionalUnifier(
                lhs_mapping_candidates=ind_inames_set | set(matching_vars))
        unifiers = unif(uni_template, csed.cse.child)
        if not unifiers:
            raise RuntimeError("Unable to unify  "
            "CSEs '%s' and '%s' (with lhs candidates '%s')" % (
                uni_template, csed.cse.child,
                ",".join(unif.lhs_mapping_candidates),
                ))

        # }}}

        found_good_unifier = False

        for unifier in unifiers:
            # {{{ construct, check mapping

            map_space = kernel.space
            ln = len(uni_iname_list)
            rn = kernel.space.dim(dim_type.out)

            map_space = map_space.add_dims(dim_type.in_, ln)
            for i, iname in enumerate(uni_iname_list):
                map_space = map_space.set_dim_name(dim_type.in_, i, iname)

            set_space = map_space.move_dims(
                    dim_type.out, rn,
                    dim_type.in_, 0, ln).range()

            var_map = None

            from loopy.symbolic import aff_from_expr
            for lhs, rhs in unifier.equations:
                cns = isl.Constraint.equality_from_aff(
                        aff_from_expr(set_space, lhs - rhs))

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
                    .intersect_range(kernel.domain))

            # Sanity check: If the range of the map does not recover the
            # domain of the expression, the unifier must have been no
            # good.
            if restr_rhs_map.range() != kernel.domain:
                continue

            # Sanity check: Injectivity here means that unique lead indices
            # can be found for each

            if not var_map.is_injective():
                raise RuntimeError("In CSEs '%s' and '%s': "
                        "cannot find lead indices uniquely"
                        % (uni_template, csed.cse.child))

            footprint_contrib = restr_rhs_map.domain()
            if footprint is None:
                footprint = footprint_contrib
            else:
                footprint = footprint.union(footprint_contrib)

            found_good_unifier = True

            # }}}

        if not found_good_unifier:
            raise RuntimeError("No valid unifier for '%s' and '%s'"
                    % (csed.cse.child, uni_template))

        uni_recs.append(unifier)

        # {{{ check that matching_vars have a unique_value

        csed.unif_var_dict = dict((lhs.name, rhs)
                for lhs, rhs in unifier.equations)
        for mv_name in matching_vars:
            if mv_name in matching_var_values:
                if matching_var_values[mv_name] != csed.unif_var_dict[mv_name]:
                    raise RuntimeError("two different expressions encountered "
                            "for matching variable: '%s' and '%s'" % (
                                matching_var_values[mv_name], csed.unif_var_dict[mv_name]))
            else:
                matching_var_values[mv_name] = csed.unif_var_dict[mv_name]

        # }}}

    return footprint, matching_var_values,





def make_compute_insn(kernel, cse_tag, uni_template,
        target_var_name, target_var_base_indices,
        independent_inames, ind_iname_to_tag, insn):

    # {{{ decide whether to force a dep

    from loopy.symbolic import IndexVariableFinder
    dependencies = IndexVariableFinder(
            include_reduction_inames=False)(uni_template)

    parent_inames = kernel.insn_inames(insn) | insn.reduction_inames()
    #print dependencies, parent_inames
    #assert dependencies <= parent_inames

    for iname in parent_inames:
        if iname in independent_inames:
            tag = ind_iname_to_tag[iname]
        else:
            tag = kernel.iname_to_tag.get(iname)

        check_cse_iname_deps(
                iname, independent_inames, tag, dependencies, cse_tag, uni_template)

    # }}}

    assignee = var(target_var_name)

    if independent_inames:
        assignee = assignee[tuple(
            var(iname)-bi
            for iname, bi in zip(independent_inames, target_var_base_indices)
            )]

    insn_prefix = cse_tag
    if insn_prefix is None:
        insn_prefix = "cse"
    from loopy.kernel import Instruction
    return Instruction(
            id=kernel.make_unique_instruction_id(based_on=insn_prefix+"_compute"),
            assignee=assignee,
            expression=uni_template)




def realize_cse(kernel, cse_tag, dtype, independent_inames=[],
        uni_template=None, ind_iname_to_tag={}, new_inames=None, default_tag="l.auto"):
    """
    :arg independent_inames: which inames are supposed to be separate loops
        in the CSE. Also determines index order of temporary array.
        The variables in independent_inames refer to the unification
        template.
    :arg uni_template: An expression against which all targeted subexpressions
        must unify

        If None, a unification template will be chosen from among the targeted
        CSEs. That CSE is chosen to depend on all the variables in
        *independent_inames*.  It is an error if no such expression can be
        found.

        May contain '*' wildcards that will have to match exactly across all
        unifications.

    Process:

    - Find all targeted CSEs.

    - Find *uni_template* as described above.

    - Turn all wildcards in *uni_template* into matching-relevant (but not
      independent, in the sense of *independent_inames*) variables.

    - Unify the CSEs with the unification template, detecting mappings
      of template variables to variables used in the CSE.

    - Find the (union) footprint of the CSEs in terms of the
      *independent_inames*.

    - Augment the kernel domain by that footprint and generate the fetch
      instruction.

    - Replace the CSEs according to the mapping detected in unification.
    """

    newly_created_var_names = set()

    # {{{ replace any wildcards in uni_template with new variables

    if isinstance(uni_template, str):
        from pymbolic import parse
        uni_template = parse(uni_template)

    def get_unique_var_name():
        if cse_tag is None:
            based_on = "cse_wc"
        else:
            based_on = cse_tag+"_wc"

        result = kernel.make_unique_var_name(
                based_on=based_on, extra_used_vars=newly_created_var_names)
        newly_created_var_names.add(result)
        return result

    if uni_template is not None:
        from loopy.symbolic import WildcardToUniqueVariableMapper
        wc_map = WildcardToUniqueVariableMapper(get_unique_var_name)
        uni_template = wc_map(uni_template)

    # }}}

    # {{{ process ind_iname_to_tag argument

    ind_iname_to_tag = ind_iname_to_tag.copy()

    from loopy.kernel import parse_tag
    default_tag = parse_tag(default_tag)
    for iname in independent_inames:
        ind_iname_to_tag.setdefault(iname, default_tag)

    if not set(ind_iname_to_tag.iterkeys()) <= set(independent_inames):
        raise RuntimeError("tags for non-new inames may not be passed")

    # here, all information is consolidated into ind_iname_to_tag

    # }}}

    # {{{ gather cse descriptors

    cse_descriptors = []

    def gather_cses(cse, rec):
        if cse.prefix != cse_tag:
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

    # {{{ find/pick a unification template

    if not cse_descriptors:
        raise RuntimeError("no CSEs tagged '%s' found" % cse_tag)

    if uni_template is None:
        for csed in cse_descriptors:
            if set(independent_inames) <= get_dependencies(csed.cse.child):
                # pick the first cse that has the required inames as the unification template
                uni_template = csed.cse.child
                break

        if uni_template is None:
            raise RuntimeError("could not find a suitable unification template that depends on "
                    "inames '%s'" % ",".join(independent_inames))

    # }}}

    # {{{ make sure that independent inames and kernel inames do not overlap

    # (and substitute in uni_template if any variable name changes are necessary)

    if set(independent_inames) & kernel.all_inames():
        old_to_new = {}

        new_independent_inames = []
        new_ind_iname_to_tag = {}
        for i, iname in enumerate(independent_inames):
            if iname in kernel.all_inames():
                based_on = iname
                if new_inames is not None and i < len(new_inames):
                    based_on = new_inames[i]

                new_iname = kernel.make_unique_var_name(
                        based_on=iname, extra_used_vars=newly_created_var_names)
                old_to_new[iname] = var(new_iname)
                newly_created_var_names.add(new_iname)
                new_independent_inames.append(new_iname)
                new_ind_iname_to_tag[new_iname] = ind_iname_to_tag[iname]
            else:
                new_independent_inames.append(iname)
                new_ind_iname_to_tag[iname] = ind_iname_to_tag[iname]

        independent_inames = new_independent_inames
        ind_iname_to_tag = new_ind_iname_to_tag
        uni_template = (
                SubstitutionMapper(make_subst_func(old_to_new))
                (uni_template))

    # }}}

    # {{{ deal with iname deps of uni_template that are not independent_inames

    # (We call these 'matching_vars', because they have to match exactly in
    # every CSE. As above, they might need to be renamed to make them unique
    # within the kernel.)

    matching_vars = []
    old_to_new = {}

    for iname in (get_dependencies(uni_template)
            - set(independent_inames)
            - kernel.non_iname_variable_names()):
        if iname in kernel.all_inames():
            # need to rename to be unique
            new_iname = kernel.make_unique_var_name(
                    based_on=iname, extra_used_vars=newly_created_var_names)
            old_to_new[iname] = var(new_iname)
            newly_created_var_names.add(new_iname)
            matching_vars.append(new_iname)
        else:
            matching_vars.append(iname)

    if old_to_new:
        uni_template = (
                SubstitutionMapper(make_subst_func(old_to_new))
                (uni_template))

    # }}}

    # {{{ align and intersect the footprint and the domain

    # (If there are independent inames, this adds extra dimensions to the domain.)

    footprint, matching_var_values = process_cses(kernel, uni_template,
            independent_inames, matching_vars,
            cse_descriptors)

    if isinstance(footprint, isl.Set):
        footprint = footprint.coalesce()
        footprint_bsets = footprint.get_basic_sets()
        if len(footprint_bsets) > 1:
            raise NotImplementedError("CSE '%s' yielded a non-convex footprint"
                    % cse_tag)

        footprint, = footprint_bsets

    ndim = kernel.space.dim(dim_type.set)
    footprint = footprint.insert_dims(dim_type.set, 0, ndim)
    for i in range(ndim):
        footprint = footprint.set_dim_name(dim_type.set, i,
                kernel.space.get_dim_name(dim_type.set, i))

    from islpy import align_spaces
    new_domain = align_spaces(kernel.domain, footprint).intersect(footprint)

    # set matching vars equal to their unified value, eliminate them
    from loopy.symbolic import aff_from_expr

    assert set(matching_var_values) == set(matching_vars)

    for var_name, value in matching_var_values.iteritems():
        cns = isl.Constraint.equality_from_aff(
                aff_from_expr(new_domain.get_space(), var(var_name) - value))
        new_domain = new_domain.add_constraint(cns)

    new_domain = (new_domain
            .eliminate(dim_type.set,
                new_domain.dim(dim_type.set)-len(matching_vars), len(matching_vars))
            .remove_dims(dim_type.set,
                new_domain.dim(dim_type.set)-len(matching_vars), len(matching_vars)))
    new_domain = new_domain.remove_redundancies()

    # }}}

    # }}}

    # {{{ set up temp variable

    var_base = cse_tag
    if var_base is None:
        var_base = "cse"
    target_var_name = kernel.make_unique_var_name(var_base)

    from loopy.kernel import (TemporaryVariable,
            find_var_base_indices_and_shape_from_inames)

    target_var_base_indices, target_var_shape = \
            find_var_base_indices_and_shape_from_inames(
                    new_domain, independent_inames)

    new_temporary_variables = kernel.temporary_variables.copy()
    new_temporary_variables[target_var_name] = TemporaryVariable(
            name=target_var_name,
            dtype=np.dtype(dtype),
            base_indices=target_var_base_indices,
            shape=target_var_shape,
            is_local=None)

    # }}}

    mv_subst = SubstitutionMapper(make_subst_func(
        dict((mv, matching_var_values[mv]) for mv in matching_vars)))

    compute_insn = make_compute_insn(
            kernel, cse_tag, mv_subst(uni_template),
            target_var_name, target_var_base_indices,
            independent_inames, ind_iname_to_tag,
            # pick one insn at random for dep check
            cse_descriptors[0].insn)

    # {{{ substitute variable references into instructions

    def subst_cses(cse, rec):
        found = False
        for csed in cse_descriptors:
            if cse is csed.cse:
                found = True
                break

        if not found:
            from pymbolic.primitives import CommonSubexpression
            return CommonSubexpression(
                    rec(cse.child), cse.prefix)

        indices = [csed.unif_var_dict[iname]-bi
                for iname, bi in zip(independent_inames, target_var_base_indices)]

        new_outer_expr = var(target_var_name)
        if indices:
            new_outer_expr = new_outer_expr[tuple(indices)]

        return new_outer_expr
        # can't nest, don't recurse

    cse_cb_mapper = CSECallbackMapper(subst_cses)

    new_insns = [compute_insn]

    for insn in kernel.instructions:
        new_expr = cse_cb_mapper(insn.expression)
        new_insns.append(insn.copy(expression=new_expr))

    # }}}

    new_iname_to_tag = kernel.iname_to_tag.copy()
    new_iname_to_tag.update(ind_iname_to_tag)

    return kernel.copy(
            domain=new_domain,
            instructions=new_insns,
            temporary_variables=new_temporary_variables,
            iname_to_tag=new_iname_to_tag)





# vim: foldmethod=marker

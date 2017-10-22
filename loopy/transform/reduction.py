from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2017 Matt Wala"

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


import loopy as lp
import islpy as isl

from loopy.kernel.data import iname_tag_to_temp_var_scope

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. currentmodule:: loopy

.. autofunction:: make_two_level_reduction
.. autofunction:: make_two_level_scan
"""


# {{{ two-level reduction

def make_two_level_reduction(
        kernel, insn_id, inner_length,
        nonlocal_storage_scope=None,
        nonlocal_tag=None,
        outer_tag=None,
        inner_tag=None):
    """
    Two level reduction, mediated through a "nonlocal" array.

    This turns a reduction of the form::

         [...] result = reduce(i, f(i))

    into::

         i -> inner + inner_length * outer

         [..., nl] nonlocal[nl] = reduce(inner, f(nl, inner))
         [...]     result       = reduce(outer, nonlocal[outer])
    """

    # {{{ sanity checks

    reduction = kernel.id_to_insn[insn_id].expression
    reduction_iname, = reduction.inames

    # }}}

    # {{{ get stable names for everything

    var_name_gen = kernel.get_var_name_generator()

    format_kwargs = {"insn": insn_id, "iname": reduction_iname}

    nonlocal_storage_name = var_name_gen(
            "{insn}_nonlocal".format(**format_kwargs))

    inner_iname = var_name_gen(
            "{iname}_inner".format(**format_kwargs))
    outer_iname = var_name_gen(
            "{iname}_outer".format(**format_kwargs))
    nonlocal_iname = var_name_gen(
            "{iname}_nonlocal".format(**format_kwargs))

    inner_subst = var_name_gen(
            "{insn}_inner_subst".format(**format_kwargs))

    # }}}

    # First we split this iname. This results in (roughly)
    #
    # [...] result = reduce([outer, inner], f(outer, inner))
    #
    # FIXME: within

    kernel = lp.split_iname(kernel, reduction_iname, inner_length,
            outer_iname=outer_iname, inner_iname=inner_iname)

    # Next, we split the reduction inward and then extract a substitution
    # rule for the reduction. This results in
    #
    # subst(outer) := reduce(inner, f(outer, inner))
    # [...] result = reduce([outer], subst(outer))
    #
    # FIXME: within, insn_match...

    kernel = lp.split_reduction_inward(kernel, inner_iname)
    from loopy.transform.data import reduction_arg_to_subst_rule
    kernel = reduction_arg_to_subst_rule(kernel, outer_iname,
                                         subst_rule_name=inner_subst)

    # Next, we precompute the inner iname into its own storage.

    # [...,nl] nonlocal[nl] = reduce(inner, f(nl, inner))
    # [...] result = reduce([outer], nonlocal[outer])

    kernel = lp.precompute(kernel, inner_subst,
                           sweep_inames=[outer_iname],
                           precompute_inames=[nonlocal_iname],
                           temporary_name=nonlocal_storage_name,
                           temporary_scope=nonlocal_storage_scope)

    return kernel

# }}}


# {{{ helpers for two-level scan

def _update_instructions(kernel, id_to_new_insn, copy=True):
    # FIXME: Even if this improves efficiency, this probably should not be
    # doing in-place updates, to avoid obscure caching bugs

    if not isinstance(id_to_new_insn, dict):
        id_to_new_insn = dict((insn.id, insn) for insn in id_to_new_insn)

    new_instructions = (
        list(insn for insn in kernel.instructions
             if insn.id not in id_to_new_insn)
        + list(id_to_new_insn.values()))

    if copy:
        kernel = kernel.copy()

    kernel.instructions = new_instructions
    return kernel


def _make_slab_set(iname, size):
    # FIXME: There is a very similar identically named function in
    # preprocess. Refactor.

    if not isinstance(size, (isl.PwAff, isl.Aff)):
        from loopy.symbolic import pwaff_from_expr
        size = pwaff_from_expr(
                isl.Space.params_alloc(isl.DEFAULT_CONTEXT, 0), size)

    base_space = size.get_domain_space()

    space = (base_space
            .add_dims(isl.dim_type.set, 1)
            .set_dim_name(isl.dim_type.set, base_space.dim(isl.dim_type.set), iname))

    v = isl.affs_from_space(space)

    size = isl.align_spaces(size, v[0])

    bs, = (
            v[0].le_set(v[iname])
            &
            v[iname].lt_set(v[0] + size)).get_basic_sets()

    return bs


def _add_subdomain_to_kernel(kernel, subdomain):
    domains = list(kernel.domains)
    # Filter out value parameters.
    dep_inames = (
            frozenset(subdomain.get_var_names(isl.dim_type.param))
            & kernel.all_inames())

    indices = kernel.get_leaf_domain_indices(dep_inames)

    if len(indices) == 0:
        domains.append(subdomain)
    elif len(indices) == 1:
        idx, = indices
        domains.insert(idx + 1, subdomain)
    else:
        print(indices)
        raise ValueError("more than 1 leaf index")

    return kernel.copy(domains=domains)


def _add_scan_subdomain(
        kernel, scan_iname, sweep_iname):
    """
    Add the following domain to the kernel::

        [sweep_iname] -> {[scan_iname] : 0 <= scan_iname <= sweep_iname }
    """
    sp = (
            isl.Space.set_alloc(isl.DEFAULT_CONTEXT, 1, 1)
            .set_dim_name(isl.dim_type.param, 0, sweep_iname)
            .set_dim_name(isl.dim_type.set, 0, scan_iname))

    affs = isl.affs_from_space(sp)

    subd, = (
            affs[scan_iname].le_set(affs[sweep_iname])
            &
            affs[scan_iname].ge_set(affs[0])).get_basic_sets()

    return _add_subdomain_to_kernel(kernel, subd)


def _expand_subst_within_expression(kernel, expr):
    from loopy.symbolic import (
            RuleAwareSubstitutionRuleExpander, SubstitutionRuleMappingContext)

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    submap = RuleAwareSubstitutionRuleExpander(
            rule_mapping_context,
            kernel.substitutions,
            within=lambda *args: True
            )
    return submap(expr, kernel, insn=None)


def _add_global_barrier(kernel, source, sink, barrier_id):
    from loopy.kernel.instruction import BarrierInstruction

    sources = (source,) if isinstance(source, str) else source
    sinks = (sink,) if isinstance(sink, str) else sink

    within_inames = kernel.id_to_insn[sources[0]].within_inames
    from itertools import chain
    for iname in chain(sources[1:], sinks):
        within_inames &= kernel.id_to_insn[iname].within_inames

    barrier_insn = BarrierInstruction(
            id=barrier_id,
            depends_on=frozenset(sources),
            within_inames=within_inames,
            kind="global")

    sink_insns = (kernel.id_to_insn[sink] for sink in sinks)
    updated_sinks = (
            sink.copy(depends_on=sink.depends_on | frozenset([barrier_id]))
            for sink in sink_insns)

    kernel = _update_instructions(
            kernel, chain([barrier_insn], updated_sinks), copy=True)

    return kernel


def _get_scan_level(sweep_iname):
    SWEEP_RE = r".*__l(\d+)(?:_outer)?"  # noqa

    import re
    match_result = re.match(SWEEP_RE, sweep_iname)

    if match_result is None:
        return 0

    return int(match_result.group(1))


def _get_base_iname(iname):
    BASE_INAME_RE = r"(.*)__l\d+(?:_outer)?"  # noqa

    import re
    match_result = re.match(BASE_INAME_RE, iname)

    if match_result is None:
        return iname

    return match_result.group(1)

# }}}


# {{{ two-level scan

def make_two_level_scan(
        kernel, insn_id,
        scan_iname,
        sweep_iname,
        inner_length,
        local_storage_name=None,
        local_storage_scope=None,
        local_storage_axes=None,
        nonlocal_storage_name=None,
        nonlocal_scan_storage_name=None,
        nonlocal_storage_scope=None,
        nonlocal_tag=None,
        slow_local_tag=None,
        fast_local_tag=None,
        fast_sweep_iname=None,
        slow_sweep_iname=None,
        local_scan_uses_fast_axis=True):
    """Two level scan, mediated through a "local" and "nonlocal" array.

    This turns a scan of the form::

         [...,i] result = reduce(j, f(j))

    into::

         [...,l',l''] <scan into local>
         [...,nlinit] nonlocal[0] = 0
         [...,nlinit] nonlocal[nlinit+1] = local[nlinit,-1]
         [...,nl]     <scan into nonlocal>
         [...,i',i''] result = nonlocal[i'] + local[i',i'']

    *sweep_iname* will be split into *fast_sweep_iname* and *slow_sweep_iname*.
    The names of *fast_sweep_iname* and *slow_sweep_iname* are supplied so that
    they can be passed to *local_storage_axes* if needed.

    :arg nonlocal_storage_name: The nonlocal storage that is an input to the
        nonlocal scan.
    :arg nonlocal_scan_storage_name: The nonlocal storage that is an output of
        the nonlocal scan.
    :arg local_storage_axes: A tuple of inames. For each iname, a corresponding
        axis will be added to the temporary array that does the local part of
        the scan (the "local" array). May be *None*, in which case it is
        automatically inferred from the tags of the inames.
    """

    # TODO: Test that this works even when doing split scans in a loop

    # {{{ sanity checks/input processing

    # FIXME: More sanity checks...

    insn = kernel.id_to_insn[insn_id]
    scan = insn.expression
    assert scan.inames[0] == scan_iname
    assert len(scan.inames) == 1
    del insn

    # }}}

    # {{{ get stable names for everything

    var_name_gen = kernel.get_var_name_generator()
    insn_id_gen = kernel.get_instruction_id_generator()

    level = _get_scan_level(sweep_iname)
    base_scan_iname = _get_base_iname(scan_iname)
    base_sweep_iname = _get_base_iname(sweep_iname)
    base_insn_id = _get_base_iname(insn_id)

    format_kwargs = {
            "insn": base_insn_id,
            "iname": base_scan_iname,
            "sweep": base_sweep_iname,
            "level": level,
            "next_level": level + 1}

    if fast_sweep_iname is None:
        fast_sweep_iname = var_name_gen(
                "{sweep}__l{level}".format(**format_kwargs))
    else:
        var_name_gen.add_name(fast_sweep_iname)

    if slow_sweep_iname is None:
        slow_sweep_iname = var_name_gen(
                "{sweep}__l{level}_outer".format(**format_kwargs))
    else:
        var_name_gen.add_iname(slow_sweep_iname)

    """
    nonlocal_init_head_outer_iname = var_name_gen(
            "{sweep}__l{level}_nlhead_outer".format(**format_kwargs))

    nonlocal_init_head_inner_iname = var_name_gen(
            "{sweep}__l{level}_nlhead_inner".format(**format_kwargs))
    """

    nonlocal_init_tail_outer_iname = var_name_gen(
            "{sweep}__l{level}_nltail_outer".format(**format_kwargs))

    # FIXME: This iname is not really needed. We should see about getting
    # rid of it. That would also make the write race warning business below
    # unnecessary.
    nonlocal_init_tail_inner_iname = var_name_gen(
            "{sweep}__l{level}_nltail_inner".format(**format_kwargs))

    nonlocal_iname = var_name_gen(
            "{sweep}__l{level}_nonloc".format(**format_kwargs))

    fast_local_iname = var_name_gen(
            "{sweep}__l{next_level}".format(**format_kwargs))

    fast_scan_iname = var_name_gen(
            "{iname}__l{next_level}".format(**format_kwargs))

    slow_local_iname = var_name_gen(
            "{sweep}__l{next_level}_outer".format(**format_kwargs))

    slow_scan_iname = var_name_gen(
            "{iname}__l{level}".format(**format_kwargs))

    subst_name = var_name_gen(
            "{insn}_inner_subst".format(**format_kwargs))

    local_subst_name = var_name_gen(
            "{insn}_local_subst".format(**format_kwargs))

    if local_storage_name is None:
        local_storage_name = var_name_gen(
            "{insn}__l{next_level}".format(**format_kwargs))
    else:
        var_name_gen.add_name(local_storage_name)

    if nonlocal_storage_name is None:
        nonlocal_storage_name = var_name_gen(
            "{insn}__l{level}_outer".format(**format_kwargs))
    else:
        var_name_gen.add_name(nonlocal_storage_name)

    if nonlocal_scan_storage_name is None:
        nonlocal_scan_storage_name = var_name_gen(
            "{insn}__l{level}_outer_scan".format(**format_kwargs))
    else:
        var_name_gen.add_name(nonlocal_scan_storage_name)

    local_scan_insn_id = insn_id_gen(
            "{insn}__l{next_level}".format(**format_kwargs))

    nonlocal_scan_insn_id = insn_id_gen(
            "{insn}__l{level}".format(**format_kwargs))

    format_kwargs.update({"nonlocal": nonlocal_storage_name})

    nonlocal_init_head_insn_id = insn_id_gen(
            "{nonlocal}_init_head".format(**format_kwargs))

    nonlocal_init_tail_insn_id = insn_id_gen(
            "{nonlocal}_init_tail".format(**format_kwargs))

    # }}}

    # {{{ parameter processing

    # It seems that local_storage_axes should be determined automatically.
    auto_local_storage_axes = [
            iname
            for iname, tag in [
                (slow_sweep_iname, slow_local_tag),
                (fast_sweep_iname, fast_local_tag)]

            # ">" is "more global"
            # In a way, global inames are automatically part of an access to a
            # more local array.
            if iname_tag_to_temp_var_scope(tag) <= local_storage_scope]

    if local_storage_axes is None:
        local_storage_axes = auto_local_storage_axes
    else:
        if list(local_storage_axes) != auto_local_storage_axes:
            raise ValueError("expected local_storage_axes (%s) did not match "
                    "provided local_storage_axes (%s)"
                    % (auto_local_storage_axes, local_storage_axes))

    # }}}

    # {{{ utils

    def pick_out_relevant_axes(full_indices, strip_scalar=False):
        assert len(full_indices) == 2
        iname_to_index = dict(
                zip((slow_sweep_iname, fast_sweep_iname), full_indices))

        result = []
        for iname in local_storage_axes:
            result.append(iname_to_index[iname])

        assert len(result) > 0

        return (tuple(result)
                if not (strip_scalar and len(result) == 1)
                else result[0])

    # }}}

    # {{{ prepare for two level scan

    # Turn the scan into a substitution rule, replace the original scan with a
    # nop and delete the scan iname.
    #
    # (The presence of the scan iname seems to be making precompute very confused.)

    from loopy.transform.data import reduction_arg_to_subst_rule
    kernel = reduction_arg_to_subst_rule(
            kernel, scan_iname, subst_rule_name=subst_name)

    kernel = _update_instructions(
            kernel,
            {insn_id: kernel.id_to_insn[insn_id].copy(expression=0)})

    """
    from loopy.kernel.instruction import NoOpInstruction
            {insn_id: NoOpInstruction(
                id=insn_id,
                depends_on=insn.depends_on,
                groups=insn.groups,
                conflicts_with_groups=insn.groups,
                no_sync_with=insn.no_sync_with,
                within_inames_is_final=insn.within_inames_is_final,
                within_inames=insn.within_inames,
                priority=insn.priority,
                boostable=insn.boostable,
                boostable_into=insn.boostable_into,
                predicates=insn.predicates,
                tags=insn.tags)},
            copy=False)
    """

    kernel = lp.remove_unused_inames(kernel, inames=(scan_iname,))

    # Make sure we got rid of everything
    assert scan_iname not in kernel.all_inames()

    # }}}

    # {{{ implement local scan

    from pymbolic import var

    # FIXME: This can probably be done using split_reduction_inward()
    # and will end up looking like less of a mess that way.

    if local_scan_uses_fast_axis:
        subst_expr = var(slow_sweep_iname) * inner_length + var(fast_scan_iname)
    else:
        subst_expr = var(slow_scan_iname) * inner_length + var(fast_sweep_iname)

    local_scan_expr = _expand_subst_within_expression(kernel,
            var(subst_name)(subst_expr))

    kernel = lp.split_iname(kernel, sweep_iname, inner_length,
            inner_iname=fast_sweep_iname, outer_iname=slow_sweep_iname,
            inner_tag=fast_local_tag, outer_tag=slow_local_tag)

    from loopy.kernel.data import SubstitutionRule
    from loopy.symbolic import Reduction

    local_scan_iname = (
            fast_scan_iname
            if local_scan_uses_fast_axis
            else slow_scan_iname)

    local_subst_arguments = (slow_sweep_iname, fast_sweep_iname)

    local_subst = SubstitutionRule(
            name=local_subst_name,
            arguments=local_subst_arguments,
            expression=Reduction(
                scan.operation, (local_scan_iname,), local_scan_expr))

    substitutions = kernel.substitutions.copy()
    substitutions[local_subst_name] = local_subst

    kernel = kernel.copy(substitutions=substitutions)

    all_precompute_inames = (slow_local_iname, fast_local_iname)

    precompute_inames = pick_out_relevant_axes(all_precompute_inames)
    sweep_inames = pick_out_relevant_axes((slow_sweep_iname, fast_sweep_iname))

    storage_axis_to_tag = {
            slow_sweep_iname: slow_local_tag,
            fast_sweep_iname: fast_local_tag,
            slow_local_iname: slow_local_tag,
            fast_local_iname: fast_local_tag}

    precompute_outer_inames = (
            frozenset(all_precompute_inames) - frozenset(precompute_inames))
    within_inames = (
            kernel.id_to_insn[insn_id].within_inames
            - frozenset([slow_sweep_iname, fast_sweep_iname]))

    from pymbolic import var

    local_precompute_xform_info = lp.precompute(kernel,
            [var(local_subst_name)(
                var(slow_sweep_iname), var(fast_sweep_iname))],
            sweep_inames=sweep_inames,
            precompute_inames=precompute_inames,
            storage_axes=local_storage_axes,
            storage_axis_to_tag=storage_axis_to_tag,
            precompute_outer_inames=precompute_outer_inames | within_inames,
            temporary_name=local_storage_name,
            temporary_scope=local_storage_scope,
            compute_insn_id=local_scan_insn_id,
            return_info_structure=True)

    kernel = local_precompute_xform_info.kernel
    local_scan_dep_id = local_precompute_xform_info.compute_dep_id

    # FIXME: Should make it so that compute_insn just gets created with these
    # deps in place.
    compute_insn_with_deps = kernel.id_to_insn[
            local_precompute_xform_info.compute_insn_id]
    compute_insn_with_deps = compute_insn_with_deps.copy(
            depends_on=compute_insn_with_deps.depends_on
            | kernel.id_to_insn[insn_id].depends_on)

    kernel = _update_instructions(kernel, (compute_insn_with_deps,))

    local_sweep_iname = (
            fast_local_iname
            if local_scan_uses_fast_axis
            else slow_local_iname)

    kernel = _add_scan_subdomain(kernel, local_scan_iname, local_sweep_iname)

    # }}}

    nonlocal_sweep_iname = (
            slow_sweep_iname
            if local_scan_uses_fast_axis
            else fast_sweep_iname)

    from loopy.kernel.data import ConcurrentTag
    if not isinstance(kernel.iname_to_tag[nonlocal_sweep_iname], ConcurrentTag):
        # FIXME
        raise NotImplementedError("outer iname must currently be concurrent because "
                "it occurs in the local scan and the final addition and one of "
                "those would need to be copied/renamed if it is non-concurrent. "
                "This split is currently unimplemented.")

    # {{{ implement local to nonlocal information transfer

    from loopy.isl_helpers import static_max_of_pw_aff
    from loopy.symbolic import pw_aff_to_expr

    # FIXME: Not sure if this is the right thing to do.
    local_storage_local_axis_len = (
            kernel.temporary_variables[local_storage_name].shape[-1]
            if local_scan_uses_fast_axis
            else kernel.temporary_variables[local_storage_name].shape[0])

    nonlocal_storage_len_pw_aff = static_max_of_pw_aff(
            kernel.get_iname_bounds(nonlocal_sweep_iname).size,
            constants_only=False)

    # FIXME: this shouldn't have to have an extra element.

    # FIXME: (Related) This information transfer should perhaps be done with a
    # ternary, but the bounds checker is currently too dumb to recognize that
    # that's OK.

    nonlocal_storage_len = pw_aff_to_expr(1 + nonlocal_storage_len_pw_aff)

    nonlocal_tail_inner_subd = _make_slab_set(nonlocal_init_tail_inner_iname, 1)
    kernel = _add_subdomain_to_kernel(kernel, nonlocal_tail_inner_subd)
    nonlocal_tail_outer_subd = _make_slab_set(
            nonlocal_init_tail_outer_iname, nonlocal_storage_len_pw_aff)
    kernel = _add_subdomain_to_kernel(kernel, nonlocal_tail_outer_subd)

    """
    nonlocal_head_inner_subd = _make_slab_set(nonlocal_init_head_inner_iname, 1)
    kernel = _add_subdomain_to_kernel(kernel, nonlocal_head_inner_subd)
    nonlocal_head_outer_subd = _make_slab_set(nonlocal_init_head_outer_iname, 1)
    kernel = _add_subdomain_to_kernel(kernel, nonlocal_head_outer_subd)
    """

    # FIXME: This was commented out so that the nonlocal init part is
    # sequential, as a workaround for ISPC. This should just get its own
    # parameter controlling the tag of the local-to-nonlocal transfer.
    """
    kernel = lp.tag_inames(kernel, {
            #nonlocal_init_head_outer_iname: slow_local_tag,
            #nonlocal_init_head_inner_iname: fast_local_tag,
            nonlocal_init_tail_outer_iname: fast_local_tag,
            nonlocal_init_tail_inner_iname: slow_local_tag})
    """

    for nls_name in [nonlocal_storage_name, nonlocal_scan_storage_name]:
        if nls_name not in kernel.temporary_variables:

            from loopy.kernel.data import TemporaryVariable
            new_temporary_variables = kernel.temporary_variables.copy()

            new_temporary_variables[nls_name] = (
                    TemporaryVariable(
                        nls_name,
                        shape=(nonlocal_storage_len,),
                        scope=nonlocal_storage_scope,
                        base_indices=lp.auto,
                        dtype=lp.auto))

            kernel = kernel.copy(temporary_variables=new_temporary_variables)

    from loopy.kernel.instruction import make_assignment

    nonlocal_init_head = make_assignment(
            id=nonlocal_init_head_insn_id,
            assignees=(var(nonlocal_storage_name)[0],),

            # FIXME: should be neutral element...
            expression=0,

            within_inames=(
                within_inames | frozenset([nonlocal_init_tail_outer_iname,
                                           nonlocal_init_tail_inner_iname])),
            no_sync_with=frozenset([(nonlocal_init_tail_insn_id, "any")]),
            predicates=(var(nonlocal_init_tail_inner_iname).eq(0),
                        var(nonlocal_init_tail_outer_iname).eq(0)),
            depends_on=frozenset([local_scan_dep_id]))

    if local_scan_uses_fast_axis:
        nonlocal_init_tail_index = (
                var(nonlocal_init_tail_outer_iname),
                var(nonlocal_init_tail_inner_iname)
                + local_storage_local_axis_len - 1)
    else:
        nonlocal_init_tail_index = (
                local_storage_local_axis_len - 1,
                var(nonlocal_init_tail_outer_iname))

    nonlocal_init_tail = make_assignment(
            id=nonlocal_init_tail_insn_id,
            assignees=(
                var(nonlocal_storage_name)[
                    var(nonlocal_init_tail_outer_iname) + 1],),
            expression=var(local_storage_name)[
                pick_out_relevant_axes(nonlocal_init_tail_index, strip_scalar=True)],
            no_sync_with=frozenset([(nonlocal_init_head_insn_id, "any")]),
            within_inames=(
                within_inames | frozenset([nonlocal_init_tail_outer_iname,
                                           nonlocal_init_tail_inner_iname])),
            depends_on=frozenset([local_scan_dep_id]))

    kernel = _update_instructions(
            kernel, (nonlocal_init_head, nonlocal_init_tail), copy=False)

    # The write race warnings are spurious - the inner iname is length
    # 1, so there's really no write race at all here.
    kernel = kernel.copy(
            silenced_warnings=kernel.silenced_warnings
            + ["write_race(%s)" % nonlocal_init_tail_insn_id]
            + ["write_race(%s)" % nonlocal_init_head_insn_id])

    # }}}

    insn = kernel.id_to_insn[insn_id]

    # {{{ implement nonlocal scan

    subd = _make_slab_set(nonlocal_iname, nonlocal_storage_len_pw_aff)
    kernel = _add_subdomain_to_kernel(kernel, subd)

    if nonlocal_tag is not None:
        kernel = lp.tag_inames(kernel, {nonlocal_iname: nonlocal_tag})

    nonlocal_scan_iname = (
            slow_scan_iname
            if local_scan_uses_fast_axis
            else fast_scan_iname)

    kernel = _add_scan_subdomain(kernel, nonlocal_scan_iname, nonlocal_iname)

    nonlocal_scan = make_assignment(
            id=nonlocal_scan_insn_id,
            assignees=(var(nonlocal_scan_storage_name)[var(nonlocal_iname)],),
            expression=Reduction(
                scan.operation,
                (nonlocal_scan_iname,),
                var(nonlocal_storage_name)[var(nonlocal_scan_iname)]),
            within_inames=within_inames | frozenset([nonlocal_iname]),
            depends_on=(
                frozenset([nonlocal_init_tail_insn_id, nonlocal_init_head_insn_id])))

    kernel = _update_instructions(kernel, (nonlocal_scan,), copy=False)

    if nonlocal_storage_scope == lp.temp_var_scope.GLOBAL:
        barrier_id = insn_id_gen(
                "{insn}_nonlocal_init_barrier".format(**format_kwargs))
        kernel = _add_global_barrier(kernel,
                source=(nonlocal_init_tail_insn_id, nonlocal_init_head_insn_id),
                sink=nonlocal_scan_insn_id,
                barrier_id=barrier_id)

    # }}}

    # {{{ replace scan with local + nonlocal

    updated_depends_on = (insn.depends_on
            | frozenset([nonlocal_scan_insn_id, local_scan_dep_id]))

    if nonlocal_storage_scope == lp.temp_var_scope.GLOBAL:
        barrier_id = insn_id_gen(
                "{insn}_nonlocal_scan_barrier".format(**format_kwargs))
        kernel = (_add_global_barrier(kernel,
                source=nonlocal_scan_insn_id, sink=insn_id, barrier_id=barrier_id))
        updated_depends_on |= frozenset([barrier_id])

    nonlocal_part = var(nonlocal_scan_storage_name)[var(nonlocal_sweep_iname)]

    local_part = var(local_storage_name)[
            pick_out_relevant_axes(
                (var(slow_sweep_iname), var(fast_sweep_iname)), strip_scalar=True)]

    updated_insn = insn.copy(
            no_sync_with=(
                insn.no_sync_with
                | frozenset([(nonlocal_scan_insn_id, "any")])),
            depends_on=updated_depends_on,
            # XXX: scan binary op
            expression=nonlocal_part + local_part)

    kernel = _update_instructions(kernel, (updated_insn,), copy=False)

    # }}}

    return kernel

# }}}


# vim: foldmethod=marker

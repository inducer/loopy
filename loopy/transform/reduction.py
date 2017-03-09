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


from loopy.diagnostic import LoopyError
import loopy as lp

from loopy.kernel.data import auto, temp_var_scope
from pytools import memoize_method, Record
import islpy as isl


import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. currentmodule:: loopy

.. autofunction:: make_two_level_reduction
.. autofunction:: make_two_level_scan
"""


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
    insn_id_gen = kernel.get_instruction_id_generator()

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


def _update_instructions(kernel, id_to_new_insn, copy=True):
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
    # FIXME: stolen from preprocess, should be its own thing...
    v = isl.make_zero_and_vars([iname])
    bs, = (
            v[0].le_set(v[iname])
            &
            v[iname].lt_set(v[0] + size)).get_basic_sets()
    return bs


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

    sweep_idx, = kernel.get_leaf_domain_indices((sweep_iname,))

    domains = list(kernel.domains)
    domains.insert(sweep_idx + 1, subd)

    return kernel.copy(domains=domains)


def _expand_subst_within_expression(kernel, expr):
    from loopy.symbolic import RuleAwareSubstitutionRuleExpander, SubstitutionRuleMappingContext
    from loopy.match import parse_stack_match
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
    within_inames = (
            kernel.id_to_insn[source].within_inames
            & kernel.id_to_insn[sink].within_inames)

    barrier_insn = BarrierInstruction(
            id=barrier_id,
            depends_on=frozenset([source]),
            within_inames = within_inames,
            kind="global")
    
    updated_sink = kernel.id_to_insn[sink]
    updated_sink = updated_sink.copy(
            depends_on=updated_sink.depends_on | frozenset([barrier_id]))

    kernel = _update_instructions(kernel, (barrier_insn, updated_sink), copy=True)

    return kernel


def _get_scan_level(sweep_iname):
    SWEEP_RE = r"l(\d+)_.*"

    import re
    match_result = re.match(SWEEP_RE, sweep_iname)

    if match_result is None:
        return 0

    return int(match_result.group(1))


def _get_base_iname(iname):
    BASE_INAME_RE = r"l\d+_(.*)"

    import re
    match_result = re.match(BASE_INAME_RE, iname)

    if match_result is None:
        return iname

    base_iname = match_result.group(1)

    MODIFIERS = ("inner_", "outer_")

    for modifier in MODIFIERS:
        if base_iname.startswith(modifier):
            base_iname = base_iname[len(modifier):]
            break

    return base_iname


def make_two_level_scan(
        kernel, insn_id,
        scan_iname,
        sweep_iname,
        inner_length,
        local_storage_name=None,
        local_storage_scope=None,
        local_storage_axes=None,
        nonlocal_storage_name=None,
        nonlocal_storage_scope=None,
        nonlocal_tag=None,
        outer_local_tag=None,
        inner_local_tag=None,
        inner_tag=None,
        outer_tag=None,
        inner_iname=None,
        outer_iname=None):
    """
    Two level scan, mediated through a "local" and "nonlocal" array.

    This turns a scan of the form::

         [...,i] result = reduce(j, f(j))

    into::

         [...,l',l''] <scan into local>
         [...,l']     nonlocal[0] = 0
         [...,l']     nonlocal[l'+1] = local[l',-1]
         [...,nl]     <scan into nonlocal>
         [...,i',i''] result = nonlocal[i'] + local[i',i'']
    """

    # {{{ sanity checks

    # FIXME: More sanity checks...

    insn = kernel.id_to_insn[insn_id]
    scan = insn.expression
    assert scan.inames[0] == scan_iname
    assert len(scan.inames) == 1

    # }}}

    # {{{ get stable names for everything

    # XXX: add inner_iname and outer_iname to var_name_gen if not none

    var_name_gen = kernel.get_var_name_generator()
    insn_id_gen = kernel.get_instruction_id_generator()

    level = _get_scan_level(sweep_iname)
    base_scan_iname = _get_base_iname(scan_iname)
    base_sweep_iname = _get_base_iname(sweep_iname)

    format_kwargs = {
            "insn": insn_id, "iname": base_scan_iname, "sweep": base_sweep_iname,
            "level": level, "next_level": level + 1, "prefix": "l"}

    nonlocal_storage_name = var_name_gen(
            "{prefix}{level}_insn".format(**format_kwargs))

    if inner_iname is None:
        inner_iname = var_name_gen(
                "{prefix}{level}_inner2_{sweep}".format(**format_kwargs))

    if outer_iname is None:
        outer_iname = var_name_gen(
                "{prefix}{level}_outer2_{sweep}".format(**format_kwargs))

    nonlocal_iname = var_name_gen(
            "{prefix}{level}_combine_{sweep}".format(**format_kwargs))

    inner_local_iname = var_name_gen(
            "{prefix}{next_level}_inner_{sweep}".format(**format_kwargs))

    inner_scan_iname = var_name_gen(
            "{prefix}{next_level}_{iname}".format(**format_kwargs))

    outer_scan_iname = var_name_gen(
            "{prefix}{level}_{iname}".format(**format_kwargs))

    outer_local_iname = var_name_gen(
            "{prefix}{next_level}_outer_{sweep}".format(**format_kwargs))

    subst_name = var_name_gen(
            "{insn}_inner_subst".format(**format_kwargs))

    local_subst_name = var_name_gen(
            "{insn}_local_subst".format(**format_kwargs))

    if local_storage_name is None:
        local_storage_name = var_name_gen(
            "{prefix}{next_level}l_{insn}".format(**format_kwargs))

    if nonlocal_storage_name is None:
        nonlocal_storage_name = var_name_gen(
            "{prefix}{level}nl_{insn}".format(**format_kwargs))

    local_scan_insn_id = insn_id_gen(
            "{iname}_local_scan".format(**format_kwargs))

    nonlocal_scan_insn_id = insn_id_gen(
            "{iname}_nonlocal_scan".format(**format_kwargs))

    format_kwargs.update({"nonlocal": nonlocal_storage_name})

    nonlocal_init_head_insn_id = insn_id_gen(
            "{nonlocal}_init_head".format(**format_kwargs))

    nonlocal_init_tail_insn_id = insn_id_gen(
            "{nonlocal}_init_tail".format(**format_kwargs))

    # }}}

    # {{{ utils

    if local_storage_axes is None:
        local_storage_axes = (outer_iname, inner_iname)

    def pick_out_relevant_axes(full_indices, strip_scalar=False):
        assert len(full_indices) == 2
        iname_to_index = dict(zip((outer_iname, inner_iname), full_indices))

        result = []
        for iname in local_storage_axes:
            result.append(iname_to_index[iname])

        assert len(result) > 0

        return tuple(result) if not (strip_scalar and len(result) == 1) else result[0]

    # }}}

    # {{{ prepare for two level scan

    # Turn the scan into a substitution rule, replace the original scan with a
    # nop and delete the scan iname.
    #
    # (The presence of the scan iname seems to be making precompute very confused.)

    from loopy.transform.data import reduction_arg_to_subst_rule
    kernel = reduction_arg_to_subst_rule(
            kernel, scan_iname, subst_rule_name=subst_name)

    from loopy.kernel.instruction import NoOpInstruction
    # FIXME: this is stupid
    kernel = _update_instructions(kernel, {insn_id: insn.copy(expression=0)})
    """
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
    local_scan_expr = _expand_subst_within_expression(kernel,
            var(subst_name)(var(outer_local_iname) * inner_length +
                            var(inner_scan_iname)))

    new_inames = ["temp"]

    kernel = lp.duplicate_inames(kernel,
            (sweep_iname),
            within="not id:*",
            new_inames=new_inames)

    kernel = lp.split_iname(kernel, sweep_iname, inner_length,
            inner_iname=inner_iname, outer_iname=outer_iname,
            inner_tag=inner_tag, outer_tag=outer_tag)

    kernel = lp.split_iname(kernel, new_inames[0], inner_length,
            inner_iname=inner_local_iname, outer_iname=outer_local_iname,
            inner_tag=inner_local_tag, outer_tag=outer_local_tag)

    """
    kernel = lp.duplicate_inames(kernel,
            (outer_iname, inner_iname),
            within="not id:*",
            new_inames=[outer_local_iname, inner_local_iname],
            tags={outer_iname: outer_local_tag, inner_iname: inner_local_tag})
    """

    kernel = _add_scan_subdomain(kernel, inner_scan_iname, inner_local_iname)

    from loopy.kernel.data import SubstitutionRule
    from loopy.symbolic import Reduction

    local_subst = SubstitutionRule(
            name=local_subst_name,
            arguments=(outer_iname, inner_iname),
            expression=Reduction(
                scan.operation, (inner_scan_iname,), local_scan_expr))

    substitutions = kernel.substitutions.copy()
    substitutions[local_subst_name] = local_subst

    kernel = kernel.copy(substitutions=substitutions)

    all_precompute_inames = (outer_local_iname, inner_local_iname)

    precompute_inames = pick_out_relevant_axes(all_precompute_inames)
    sweep_inames = pick_out_relevant_axes((outer_iname, inner_iname))

    precompute_outer_inames = (
            frozenset(all_precompute_inames)
            - frozenset(precompute_inames))

    insn = kernel.id_to_insn[insn_id]

    within_inames = insn.within_inames - frozenset([outer_iname, inner_iname])

    from pymbolic import var
    kernel = lp.precompute(kernel,
            [var(local_subst_name)(var(outer_iname), var(inner_iname))],
            sweep_inames=sweep_inames,
            precompute_inames=precompute_inames,
            storage_axes=local_storage_axes,
            precompute_outer_inames=precompute_outer_inames | within_inames,
            temporary_name=local_storage_name,
            compute_insn_id=local_scan_insn_id)

    # }}}

    # {{{ implement local to nonlocal information transfer

    from loopy.symbolic import pw_aff_to_expr
    nonlocal_storage_len_pw_aff = (
            # FIXME: should be 1 + len, bounds check doesnt like this..
            2 + kernel.get_iname_bounds(outer_iname).upper_bound_pw_aff)

    nonlocal_storage_len = pw_aff_to_expr(nonlocal_storage_len_pw_aff)

    if nonlocal_storage_name not in kernel.temporary_variables:
        from loopy.kernel.data import TemporaryVariable
        new_temporary_variables = kernel.temporary_variables.copy()

        new_temporary_variables[nonlocal_storage_name] = (
                TemporaryVariable(
                    nonlocal_storage_name,
                    shape=(nonlocal_storage_len,),
                    scope=nonlocal_storage_scope,
                    base_indices=lp.auto,
                    dtype=lp.auto))

        kernel = kernel.copy(temporary_variables=new_temporary_variables)

    from loopy.kernel.instruction import make_assignment
    nonlocal_init_head = make_assignment(
            id=nonlocal_init_head_insn_id,
            assignees=(var(nonlocal_storage_name)[0],),
            expression=0,
            within_inames=(
                within_inames | frozenset([outer_local_iname,inner_local_iname])),
            predicates=frozenset([var(inner_local_iname).eq(0)]),
            depends_on=frozenset([local_scan_insn_id]))

    final_element_indices = []

    nonlocal_init_tail = make_assignment(
            id=nonlocal_init_tail_insn_id,
            assignees=(var(nonlocal_storage_name)[var(outer_local_iname) + 1],),
            expression=var(local_storage_name)[
                pick_out_relevant_axes(
                    (var(outer_local_iname),var(inner_local_iname)),
                    strip_scalar=True)],
            no_sync_with=frozenset([(local_scan_insn_id, "local")]),
            within_inames=(
                within_inames | frozenset([outer_local_iname,inner_local_iname])),
            depends_on=frozenset([local_scan_insn_id]),
            predicates=frozenset([var(inner_local_iname).eq(inner_length - 1)]))

    kernel = _update_instructions(
            kernel, (nonlocal_init_head, nonlocal_init_tail), copy=False)

    # }}}

    # {{{ implement nonlocal scan

    kernel.domains.append(_make_slab_set(nonlocal_iname, nonlocal_storage_len))

    if nonlocal_tag is not None:
        kernel = lp.tag_inames(kernel, {nonlocal_iname: nonlocal_tag})

    kernel = _add_scan_subdomain(kernel, outer_scan_iname, nonlocal_iname)
    
    nonlocal_scan = make_assignment(
            id=nonlocal_scan_insn_id,
            assignees=(var(nonlocal_storage_name)[var(nonlocal_iname)],),
            expression=Reduction(
                scan.operation,
                (outer_scan_iname,),
                var(nonlocal_storage_name)[var(outer_scan_iname)]),
            within_inames=within_inames | frozenset([nonlocal_iname]),
            depends_on=frozenset([nonlocal_init_tail_insn_id, nonlocal_init_head_insn_id]))

    kernel = _update_instructions(kernel, (nonlocal_scan,), copy=False)

    if nonlocal_storage_scope == lp.temp_var_scope.GLOBAL:
        barrier_id = insn_id_gen("barrier_{insn}".format(**format_kwargs))
        kernel = _add_global_barrier(kernel,
                source=nonlocal_init_tail_insn_id,
                sink=nonlocal_scan_insn_id,
                barrier_id=barrier_id)

    # }}}

    # {{{ replace scan with local + nonlocal

    updated_depends_on = insn.depends_on | frozenset([nonlocal_scan_insn_id])

    if nonlocal_storage_scope == lp.temp_var_scope.GLOBAL:
        barrier_id = insn_id_gen("barrier_{insn}".format(**format_kwargs))
        kernel = (_add_global_barrier(kernel,
                source=nonlocal_scan_insn_id, sink=insn_id, barrier_id=barrier_id))
        updated_depends_on |= frozenset([barrier_id])

    nonlocal_part = var(nonlocal_storage_name)[var(outer_iname)]

    local_part = var(local_storage_name)[
            pick_out_relevant_axes(
                (var(outer_iname), var(inner_iname)), strip_scalar=True)]

    updated_insn = insn.copy(
            depends_on=updated_depends_on,
            # XXX: scan binary op
            expression=nonlocal_part + local_part)

    kernel = _update_instructions(kernel, (updated_insn,), copy=False)

    # }}}

    return kernel


# vim: foldmethod=marker

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
.. autofunction:: precompute_scan
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
    print("ADDING SLAB", bs)
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
        inner_local_iname=None,
        outer_local_iname=None):
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

    insn = kernel.id_to_insn[insn_id]
    scan = insn.expression
    assert scan.inames[0] == scan_iname
    assert len(scan.inames) == 1

    # }}}

    # {{{ get stable names for everything

    var_name_gen = kernel.get_var_name_generator()
    insn_id_gen = kernel.get_instruction_id_generator()

    format_kwargs = {"insn": insn_id, "iname": scan_iname, "sweep": sweep_iname}

    nonlocal_storage_name = var_name_gen(
            "{insn}_nonlocal".format(**format_kwargs))

    inner_iname = var_name_gen(
            "{sweep}_inner".format(**format_kwargs))
    outer_iname = var_name_gen(
            "{sweep}_outer".format(**format_kwargs))
    nonlocal_iname = var_name_gen(
            "{sweep}_nonlocal".format(**format_kwargs))

    if inner_local_iname is None:
        inner_local_iname = var_name_gen(
                "{sweep}_inner_local".format(**format_kwargs))

    inner_scan_iname = var_name_gen(
            "{iname}_inner".format(**format_kwargs))

    outer_scan_iname = var_name_gen(
            "{iname}_outer".format(**format_kwargs))

    if outer_local_iname is None:
        outer_local_iname = var_name_gen(
                "{sweep}_outer_local".format(**format_kwargs))

    subst_name = var_name_gen(
            "{insn}_inner_subst".format(**format_kwargs))

    local_subst_name = var_name_gen(
            "{insn}_local_subst".format(**format_kwargs))

    if local_storage_name is None:
        local_storage_name = var_name_gen(
            "{insn}_local".format(**format_kwargs))

    if nonlocal_storage_name is None:
        nonlocal_storage_name = var_name_gen(
            "{insn}_nonlocal".format(**format_kwargs))

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

    # {{{ implement local scan

    from pymbolic import var
    local_scan_expr = _expand_subst_within_expression(kernel,
            var(subst_name)(var(outer_local_iname) * inner_length +
                            var(inner_scan_iname)))

    kernel = lp.split_iname(kernel, sweep_iname, inner_length,
            inner_iname=inner_iname, outer_iname=outer_iname)

    print("SPLITTING INAME, GOT DOMAINS", kernel.domains)

    from loopy.kernel.data import SubstitutionRule
    from loopy.symbolic import Reduction

    local_subst = SubstitutionRule(
            name=local_subst_name,
            arguments=(outer_iname, inner_iname),
            expression=Reduction(
                scan.operation,
                (inner_scan_iname,),
                local_scan_expr)
            )

    substitutions = kernel.substitutions.copy()
    substitutions[local_subst_name] = local_subst

    kernel = kernel.copy(substitutions=substitutions)

    print(kernel)

    from pymbolic import var
    kernel = lp.precompute(
            kernel,
            [var(local_subst_name)(var(outer_iname), var(inner_iname))],
            storage_axes=(outer_iname, inner_iname),
            sweep_inames=(outer_iname, inner_iname),
            precompute_inames=(outer_local_iname, inner_local_iname),
            temporary_name=local_storage_name,
            compute_insn_id=local_scan_insn_id)

    kernel = _add_scan_subdomain(kernel, inner_scan_iname, inner_local_iname)

    # }}}

    # {{{ implement local to nonlocal information transfer

    from loopy.symbolic import pw_aff_to_expr
    nonlocal_storage_len_pw_aff = (
            # The 2 here is because the first element is 0.
            2 + kernel.get_iname_bounds(outer_iname).upper_bound_pw_aff)

    nonlocal_storage_len = pw_aff_to_expr(nonlocal_storage_len_pw_aff)

    if nonlocal_storage_name not in kernel.temporary_variables:
        from loopy.kernel.data import TemporaryVariable
        new_temporary_variables = kernel.temporary_variables.copy()

        new_temporary_variables[nonlocal_storage_name] = (
                TemporaryVariable(
                    nonlocal_storage_name,
                    shape=(nonlocal_storage_len,),
                    scope=lp.auto,
                    base_indices=lp.auto,
                    dtype=lp.auto))

        kernel = kernel.copy(temporary_variables=new_temporary_variables)

    insn = kernel.id_to_insn[insn_id]

    # XXX: should not include sweep iname?
    within_inames = insn.within_inames

    from loopy.kernel.instruction import make_assignment
    nonlocal_init_head = make_assignment(
            id=nonlocal_init_head_insn_id,
            assignees=(var(nonlocal_storage_name)[0],),
            expression=0,
            within_inames=frozenset([outer_local_iname]),
            depends_on=frozenset([local_scan_insn_id]))

    final_element_indices = []

    nonlocal_init_tail = make_assignment(
            id=nonlocal_init_tail_insn_id,
            assignees=(var(nonlocal_storage_name)[var(outer_local_iname) + 1],),
            expression=var(local_storage_name)[var(outer_local_iname),inner_length - 1],
            within_inames=frozenset([outer_local_iname]),
            depends_on=frozenset([local_scan_insn_id]))

    kernel = _update_instructions(kernel, (nonlocal_init_head, nonlocal_init_tail), copy=False)

    # }}}

    # {{{ implement nonlocal scan

    kernel.domains.append(_make_slab_set(nonlocal_iname, nonlocal_storage_len))

    kernel = _add_scan_subdomain(kernel, outer_scan_iname, nonlocal_iname)
    
    nonlocal_scan = make_assignment(
            id=nonlocal_scan_insn_id,
            assignees=(var(nonlocal_storage_name)[var(nonlocal_iname)],),
            expression=Reduction(
                scan.operation,
                (outer_scan_iname,),
                var(nonlocal_storage_name)[var(outer_scan_iname)]),
            within_inames=frozenset([nonlocal_iname]),
            depends_on=frozenset([nonlocal_init_tail_insn_id, nonlocal_init_head_insn_id]))

    kernel = _update_instructions(kernel, (nonlocal_scan,), copy=False)

    # }}}

    # {{{ replace scan with local + nonlocal

    updated_insn = insn.copy(
        depends_on=insn.depends_on | frozenset([nonlocal_scan_insn_id]),
        expression=var(nonlocal_storage_name)[var(outer_iname)] + var(local_storage_name)[var(outer_iname), var(inner_iname)])

    kernel = _update_instructions(kernel, (updated_insn,), copy=False)

    # }}}

    return kernel


def precompute_scan(
        kernel, insn_id,
        sweep_iname,
        scan_iname,
        outer_inames=(),
        temporary_scope=None,
        temporary_name=None,
        replace_insn_with_nop=False):
    """
    Turn an expression-based scan into an array-based one.

    This takes a reduction of the form::

        [...,sweep_iname] result = reduce(scan_iname, f(scan_iname))

    and does essentially the following transformation::

        [...,sweep_iname'] temp[sweep_iname'] = f(sweep_iname')
        [...,sweep_iname] temp[sweep_iname] = reduce(scan_iname, temp[scan_iname])
        [...,sweep_iname] result = temp[sweep_iname]

    Note: this makes an explicit assumption that the sweep iname shares the
    same bounds as the scan iname and the bounds start at 0.
    """

    # {{{ sanity checks

    insn = kernel.id_to_insn[insn_id]
    scan = insn.expression
    assert scan.inames[0] == scan_iname
    assert len(scan.inames) == 1

    # }}}

    # {{{ get a stable name for things

    var_name_gen = kernel.get_var_name_generator()
    insn_id_gen = kernel.get_instruction_id_generator()

    format_kwargs = {"insn": insn_id, "iname": scan_iname}

    orig_subst_name = var_name_gen(
            "{iname}_orig_subst".format(**format_kwargs))

    scan_subst_name = var_name_gen(
            "{iname}_subst".format(**format_kwargs))

    precompute_insn = insn_id_gen(
            "{insn}_precompute".format(**format_kwargs))

    precompute_reduction_insn = insn_id_gen(
            "{insn}_precompute_reduce".format(**format_kwargs))

    if temporary_name is None:
        temporary_name = var_name_gen(
            "{insn}_precompute".format(**format_kwargs))

    # }}}

    from loopy.transform.data import reduction_arg_to_subst_rule
    kernel = reduction_arg_to_subst_rule(
            kernel, scan_iname, subst_rule_name=orig_subst_name)

    # {{{ create our own variant of the substitution rule

    # FIXME: There has to be a better way of this.

    orig_subst = kernel.substitutions[orig_subst_name]

    from pymbolic.mapper.substitutor import make_subst_func

    from loopy.symbolic import (
        SubstitutionRuleMappingContext, RuleAwareSubstitutionMapper)

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, var_name_gen)

    from pymbolic import var
    mapper = RuleAwareSubstitutionMapper(
            rule_mapping_context,
            make_subst_func({scan_iname: var(sweep_iname)}),
            within=lambda *args: True)

    scan_subst = orig_subst.copy(
            name=scan_subst_name,
            arguments=outer_inames + (sweep_iname,),
            expression=mapper(orig_subst.expression, kernel, None))

    substitutions = kernel.substitutions.copy()

    substitutions[scan_subst_name] = scan_subst

    kernel = kernel.copy(substitutions=substitutions)

    # }}}

    print(kernel)

    # FIXME: multi assignments
    from pymbolic import var

    # FIXME: Make a new precompute iname....

    kernel = lp.precompute(kernel,
            [var(scan_subst_name)(
                *(tuple(var(o) for o in outer_inames) +
                  (var(sweep_iname),)))],
            sweep_inames=outer_inames + (sweep_iname,),
            precompute_inames=(sweep_iname,),
            temporary_name=temporary_name,
            temporary_scope=temporary_scope,
            # FIXME: why on earth is this needed
            compute_insn_id=precompute_insn)

    from loopy.kernel.instruction import make_assignment

    from loopy.symbolic import Reduction
    precompute_reduction = insn.copy(
            id=precompute_reduction_insn,
            assignee=var(temporary_name)[var(sweep_iname)],
            expression=Reduction(
                operation=scan.operation,
                inames=(scan_iname,),
                exprs=(var(temporary_name)[var(scan_iname)],),
                allow_simultaneous=False,
                ),
            depends_on=insn.depends_on | frozenset([precompute_insn]))

    kernel = kernel.copy(instructions=kernel.instructions +
                         [precompute_reduction])

    new_insn = insn.copy(
           expression=var(temporary_name)[var(sweep_iname)],
           depends_on=
           frozenset([precompute_reduction_insn]) | insn.depends_on)

    instructions = list(kernel.instructions)

    for i, insn in enumerate(instructions):
        if insn.id == insn_id:
            instructions[i] = new_insn

    kernel = kernel.copy(instructions=instructions)

    return kernel


# vim: foldmethod=marker

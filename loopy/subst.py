from __future__ import division

from loopy.symbolic import get_dependencies, SubstitutionMapper
from pymbolic.mapper.substitutor import make_subst_func

from pytools import Record
from pymbolic import var





class ExprDescriptor(Record):
    __slots__ = ["insn", "expr", "unif_var_dict"]





def extract_subst(kernel, subst_name, template, parameters):
    """
    :arg subst_name: The name of the substitution rule to be created.
    :arg template: Unification template expression.

    All targeted subexpressions must match ('unify with') *template*
    The template may contain '*' wildcards that will have to match exactly across all
    unifications.
    """

    newly_created_var_names = set()

    if isinstance(template, str):
        from pymbolic import parse
        template = parse(template)

    # {{{ replace any wildcards in template with new variables

    def get_unique_var_name():
        based_on = subst_name+"_wc"

        result = kernel.make_unique_var_name(
                based_on=based_on, extra_used_vars=newly_created_var_names)
        newly_created_var_names.add(result)
        return result

    from loopy.symbolic import WildcardToUniqueVariableMapper
    wc_map = WildcardToUniqueVariableMapper(get_unique_var_name)
    template = wc_map(template)

    # }}}

    # {{{ deal with iname deps of template that are not independent_inames

    # (We call these 'matching_vars', because they have to match exactly in
    # every CSE. As above, they might need to be renamed to make them unique
    # within the kernel.)

    matching_vars = []
    old_to_new = {}

    for iname in (get_dependencies(template)
            - set(parameters)
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
        template = (
                SubstitutionMapper(make_subst_func(old_to_new))
                (template))

    # }}}

    # {{{ gather up expressions

    expr_descriptors = []

    from loopy.symbolic import UnidirectionalUnifier
    unif = UnidirectionalUnifier(
            lhs_mapping_candidates=set(parameters) | set(matching_vars))

    def gather_exprs(expr, mapper):
        urecs = unif(template, expr)

        if urecs:
            if len(urecs) > 1:
                raise RuntimeError("ambiguous unification of '%s' with template '%s'" 
                        % (expr, template))

            urec, = urecs

            expr_descriptors.append(
                    ExprDescriptor(
                        insn=insn,
                        expr=expr,
                        unif_var_dict = dict((lhs.name, rhs)
                            for lhs, rhs in urec.equations)))
        else:
            mapper.fallback_mapper(expr)
            # can't nest, don't recurse

    from loopy.symbolic import (
            CallbackMapper, WalkMapper, IdentityMapper)
    dfmapper = CallbackMapper(gather_exprs, WalkMapper())

    for insn in kernel.instructions:
        dfmapper(insn.expression)

    for sr in kernel.substitutions.itervalues():
        dfmapper(sr.expression)

    # }}}

    if not expr_descriptors:
        raise RuntimeError("no expressions matching '%s'" % template)

    # {{{ substitute rule into instructions

    def replace_exprs(expr, mapper):
        found = False
        for exprd in expr_descriptors:
            if expr is exprd.expr:
                found = True
                break

        if not found:
            return mapper.fallback_mapper(expr)

        args = [exprd.unif_var_dict[arg_name]
                for arg_name in parameters]

        result = var(subst_name)
        if args:
            result = result(*args)

        return result
        # can't nest, don't recurse

    cbmapper = CallbackMapper(replace_exprs, IdentityMapper())

    new_insns = []

    for insn in kernel.instructions:
        new_expr = cbmapper(insn.expression)
        new_insns.append(insn.copy(expression=new_expr))

    from loopy.kernel import SubstitutionRule
    new_substs = {
            subst_name: SubstitutionRule(
                name=subst_name,
                arguments=parameters,
                expression=template,
                )}

    for subst in kernel.substitutions.itervalues():
        new_substs[subst.name] = subst.copy(
                expression=cbmapper(subst.expression))

    # }}}

    return kernel.copy(
            instructions=new_insns,
            substitutions=new_substs)




def expand_subst(kernel, subst_name=None):
    if subst_name is None:
        rules = kernel.substitutions
    else:
        rule = kernel.substitutions[subst_name]
        rules = {rule.name: rule}

    from loopy.symbolic import ParametrizedSubstitutor
    submap = ParametrizedSubstitutor(rules)

    if subst_name:
        new_substs = kernel.substitutions.copy()
        del new_substs[subst_name]
    else:
        new_substs = {}

    return (kernel
            .copy(substitutions=new_substs)
            .map_expressions(submap))




# vim: foldmethod=marker

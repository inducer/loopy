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

from loopy.symbolic import (
        RuleAwareIdentityMapper, SubstitutionRuleMappingContext)
from loopy.diagnostic import LoopyError
from loopy.transform.iname import remove_any_newly_unused_inames

from pytools import ImmutableRecord
from pymbolic import var

from loopy.translation_unit import (for_each_kernel,
                                    TranslationUnit)
from loopy.kernel.function_interface import CallableKernel, ScalarCallable

import logging
logger = logging.getLogger(__name__)


class ExprDescriptor(ImmutableRecord):
    __slots__ = ["insn", "expr", "unif_var_dict"]


# {{{ extract_subst

def extract_subst(kernel, subst_name, template, parameters=(), within=None):
    """
    :arg subst_name: The name of the substitution rule to be created.
    :arg template: Unification template expression.
    :arg parameters: An iterable of parameters used in
        *template*, or a comma-separated string of the same.
    :arg within: An instance of :class:`loopy.match.MatchExpressionBase` or
        :class:`str` as understood by :func:`loopy.match.parse_match`.

    All targeted subexpressions must match ('unify with') *template*
    The template may contain '*' wildcards that will have to match exactly across all
    unifications.
    """

    if isinstance(kernel, TranslationUnit):
        kernel_names = [i for i, clbl in
                kernel.callables_table.items() if isinstance(clbl,
                    CallableKernel)]
        if len(kernel_names) != 1:
            raise LoopyError()

        return kernel.with_kernel(extract_subst(kernel[kernel_names[0]],
            subst_name, template, parameters))

    if isinstance(template, str):
        from pymbolic import parse
        template = parse(template)

    if isinstance(parameters, str):
        parameters = tuple(
                s.strip() for s in parameters.split(","))

    from loopy.match import parse_match
    within = parse_match(within)

    var_name_gen = kernel.get_var_name_generator()

    # {{{ replace any wildcards in template with new variables

    def get_unique_var_name():
        based_on = subst_name+"_wc"

        result = var_name_gen(based_on)
        return result

    from loopy.symbolic import WildcardToUniqueVariableMapper
    wc_map = WildcardToUniqueVariableMapper(get_unique_var_name)
    template = wc_map(template)

    # }}}

    # {{{ gather up expressions

    expr_descriptors = []

    from loopy.symbolic import UnidirectionalUnifier
    unif = UnidirectionalUnifier(
            lhs_mapping_candidates=set(parameters))

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
                        unif_var_dict={lhs.name: rhs
                            for lhs, rhs in urec.equations}))
        else:
            mapper.fallback_mapper(expr)
            # can't nest, don't recurse

    from loopy.symbolic import (
            CallbackMapper, UncachedWalkMapper as WalkMapper,
            IdentityMapper)
    dfmapper = CallbackMapper(gather_exprs, WalkMapper())

    from loopy.kernel.instruction import MultiAssignmentBase
    for insn in kernel.instructions:
        if isinstance(insn, MultiAssignmentBase) and within(kernel, insn):
            dfmapper(insn.assignees)
            dfmapper(insn.expression)

    for sr in kernel.substitutions.values():
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

    def transform_assignee(expr):
        # Assignment LHS's cannot be subst rules. Treat them
        # specially.

        import pymbolic.primitives as prim
        if isinstance(expr, tuple):
            return tuple(
                    transform_assignee(expr_i)
                    for expr_i in expr)

        elif isinstance(expr, prim.Subscript):
            return type(expr)(
                    expr.aggregate,
                    cbmapper(expr.index))

        elif isinstance(expr, prim.Variable):
            return expr
        else:
            raise ValueError("assignment LHS not understood")

    for insn in kernel.instructions:
        if within(kernel, insn):
            new_insns.append(insn.with_transformed_expressions(
                cbmapper, assignee_f=transform_assignee))
        else:
            new_insns.append(insn)

    from loopy.kernel.data import SubstitutionRule
    new_substs = {
            subst_name: SubstitutionRule(
                name=subst_name,
                arguments=tuple(parameters),
                expression=template,
                )}

    for subst in kernel.substitutions.values():
        new_substs[subst.name] = subst.copy(
                expression=cbmapper(subst.expression))

    # }}}

    return kernel.copy(
            instructions=new_insns,
            substitutions=new_substs)


# }}}


# {{{ assignment_to_subst

class AssignmentToSubstChanger(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, lhs_name, definition_insn_ids,
            usage_to_definition, extra_arguments, within):
        self.var_name_gen = rule_mapping_context.make_unique_var_name

        super().__init__(rule_mapping_context)

        self.lhs_name = lhs_name
        self.definition_insn_ids = definition_insn_ids
        self.usage_to_definition = usage_to_definition

        from pymbolic import var
        self.extra_arguments = tuple(var(arg) for arg in extra_arguments)

        self.within = within

        self.definition_insn_id_to_subst_name = {}

        self.saw_unmatched_usage_sites = {}
        for def_id in self.definition_insn_ids:
            self.saw_unmatched_usage_sites[def_id] = False

    def get_subst_name(self, def_insn_id):
        try:
            return self.definition_insn_id_to_subst_name[def_insn_id]
        except KeyError:
            subst_name = self.var_name_gen(self.lhs_name+"_subst")
            self.definition_insn_id_to_subst_name[def_insn_id] = subst_name
            return subst_name

    def map_variable(self, expr, expn_state):
        if (expr.name == self.lhs_name
                and expr.name not in expn_state.arg_context):
            result = self.transform_access(None, expn_state)
            if result is not None:
                return result

        return super().map_variable(
                expr, expn_state)

    def map_subscript(self, expr, expn_state):
        if (expr.aggregate.name == self.lhs_name
                and expr.aggregate.name not in expn_state.arg_context):
            result = self.transform_access(expr.index, expn_state)
            if result is not None:
                return result

        return super().map_subscript(
                expr, expn_state)

    def transform_access(self, index, expn_state):
        my_insn_id = expn_state.insn_id

        if my_insn_id in self.definition_insn_ids:
            return None

        my_def_id = self.usage_to_definition[my_insn_id]

        if not self.within(
                expn_state.kernel,
                expn_state.instruction,
                expn_state.stack):
            self.saw_unmatched_usage_sites[my_def_id] = True
            return None

        subst_name = self.get_subst_name(my_def_id)

        if self.extra_arguments:
            if index is None:
                index = self.extra_arguments
            else:
                index = index + self.extra_arguments

        from pymbolic import var
        if index is None:
            return var(subst_name)
        elif not isinstance(index, tuple):
            return var(subst_name)(index)
        else:
            return var(subst_name)(*index)


@for_each_kernel
@remove_any_newly_unused_inames
def assignment_to_subst(kernel, lhs_name, extra_arguments=(), within=None,
        force_retain_argument=False):
    """Extract an assignment (to a temporary variable or an argument)
    as a :ref:`substitution-rule`. The temporary may be an array, in
    which case the array indices will become arguments to the substitution
    rule.

    :arg within: a stack match as understood by
        :func:`loopy.match.parse_stack_match`.
    :arg force_retain_argument: If True and if *lhs_name* is an argument, it is
        kept even if it is no longer referenced.

    This operation will change all usage sites
    of *lhs_name* matched by *within*. If there
    are further usage sites of *lhs_name*, then
    the original assignment to *lhs_name* as well
    as the temporary variable is left in place.
    """

    if isinstance(extra_arguments, str):
        extra_arguments = tuple(s.strip() for s in extra_arguments.split(","))

    # {{{ establish the relevant definition of lhs_name for each usage site

    dep_kernel = expand_subst(kernel)
    from loopy.kernel.creation import apply_single_writer_depencency_heuristic
    dep_kernel = apply_single_writer_depencency_heuristic(dep_kernel)
    assigning_insn_ids = {insn.id
                          for insn in dep_kernel.instructions
                          if lhs_name in insn.assignee_var_names()}

    id_to_insn = dep_kernel.id_to_insn

    def get_relevant_definition_insn_id(usage_insn_id):
        insn = id_to_insn[usage_insn_id]

        rel_def_ids = set(insn.depends_on & assigning_insn_ids)
        for dep_id in insn.depends_on:
            dep_insn = id_to_insn[dep_id]
            if lhs_name in dep_insn.assignee_var_names():
                if lhs_name in dep_insn.read_dependency_names():
                    raise LoopyError("instruction '%s' both reads *and* "
                            "writes '%s'--cannot transcribe to substitution "
                            "rule" % (dep_id, lhs_name))
            else:
                if rel_def_ids < assigning_insn_ids:
                    rec_result = get_relevant_definition_insn_id(dep_id)
                    if rec_result is not None:
                        rel_def_ids.add(rec_result)

        if len(rel_def_ids) > 1:
            raise LoopyError("more than one write to '%s' found in "
                    "depdendencies of '%s'--definition cannot be resolved "
                    "(writer instructions ids: %s)"
                    % (lhs_name, usage_insn_id, ", ".join(rel_def_ids)))

        if not rel_def_ids:
            return None
        else:
            rel_def_id, = rel_def_ids

        return rel_def_id

    usage_to_definition = {}

    for insn in dep_kernel.instructions:
        if lhs_name not in insn.read_dependency_names():
            continue

        def_id = get_relevant_definition_insn_id(insn.id)
        if def_id is None:
            raise LoopyError("no write to '%s' found in dependency tree "
                    "of '%s'--definition cannot be resolved"
                    % (lhs_name, insn.id))

        usage_to_definition[insn.id] = def_id

    # }}}

    if not assigning_insn_ids:
        raise LoopyError("no assignments to variable '%s' found"
                % lhs_name)

    from loopy.symbolic import SubstitutionMapper
    from pymbolic.mapper.substitutor import make_subst_func
    from loopy.match import parse_stack_match

    within = parse_stack_match(within)
    vng = kernel.get_var_name_generator()

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, vng)
    tts = AssignmentToSubstChanger(rule_mapping_context,
            lhs_name, assigning_insn_ids,
            usage_to_definition, extra_arguments, within)

    def _accesses_lhs(kernel, insn, *args):
        return lhs_name in insn.read_dependency_names()

    kernel = rule_mapping_context.finish_kernel(
        tts.map_kernel(kernel,
                       within=_accesses_lhs,
                       map_tvs=False, map_args=False))

    from loopy.kernel.data import SubstitutionRule

    # {{{ create new substitution rules

    new_substs = kernel.substitutions.copy()
    for def_id, subst_name in tts.definition_insn_id_to_subst_name.items():
        def_insn = kernel.id_to_insn[def_id]

        from loopy.kernel.data import Assignment
        assert isinstance(def_insn, Assignment)

        from pymbolic.primitives import Variable, Subscript
        if isinstance(def_insn.assignee, Subscript):
            indices = def_insn.assignee.index_tuple
        elif isinstance(def_insn.assignee, Variable):
            indices = ()
        else:
            raise LoopyError(
                    "Unrecognized LHS type: %s"
                    % type(def_insn.assignee).__name__)

        arguments = []

        for i in indices:
            if not isinstance(i, Variable):
                raise LoopyError("In defining instruction '%s': "
                        "asignee index '%s' is not a plain variable. "
                        "Perhaps use loopy.affine_map_inames() "
                        "to perform substitution." % (def_id, i))

            arguments.append(i.name)

        # {{{ rename subst arguments

        old_arguments = tuple(arguments) + extra_arguments

        old_arg_to_new_arg = {old_name: vng(old_name)
                              for old_name in old_arguments}

        renamed_arguments = tuple(old_arg_to_new_arg[arg]
                                  for arg in old_arguments)

        renamed_expression = SubstitutionMapper(make_subst_func(
            {k: var(v) for k, v in old_arg_to_new_arg.items()}))(def_insn.expression)

        # }}}

        new_substs[subst_name] = SubstitutionRule(
                name=subst_name,
                arguments=renamed_arguments,
                expression=renamed_expression)

    # }}}

    # {{{ delete temporary variable if possible

    # (copied below if modified)
    new_temp_vars = kernel.temporary_variables
    new_args = kernel.args

    if lhs_name in kernel.temporary_variables:
        if not any(tts.saw_unmatched_usage_sites.values()):
            # All usage sites matched--they're now substitution rules.
            # We can get rid of the variable.

            new_temp_vars = new_temp_vars.copy()
            del new_temp_vars[lhs_name]

    if lhs_name in kernel.arg_dict and not force_retain_argument:
        if not any(tts.saw_unmatched_usage_sites.values()):
            # All usage sites matched--they're now substitution rules.
            # We can get rid of the argument

            new_args = new_args[:]
            for i in range(len(new_args)):
                if new_args[i].name == lhs_name:
                    del new_args[i]
                    break

    # }}}

    import loopy as lp
    kernel = lp.remove_instructions(
            kernel,
            {
                insn_id
                for insn_id, still_used in tts.saw_unmatched_usage_sites.items()
                if not still_used})

    return kernel.copy(
            substitutions=new_substs,
            temporary_variables=new_temp_vars,
            args=new_args,
            )

# }}}


# {{{ expand_subst

@for_each_kernel
def expand_subst(kernel, within=None):
    """
    Returns an instance of :class:`loopy.LoopKernel` with the substitutions
    referenced in instructions of *kernel* matched by *within* expanded.

    :arg within: a stack match as understood by
        :func:`loopy.match.parse_stack_match`.
    """

    if not kernel.substitutions:
        return kernel

    logger.debug("%s: expand subst" % kernel.name)

    from loopy.symbolic import RuleAwareSubstitutionRuleExpander
    from loopy.match import parse_stack_match
    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    submap = RuleAwareSubstitutionRuleExpander(
            rule_mapping_context,
            kernel.substitutions,
            parse_stack_match(within))

    return rule_mapping_context.finish_kernel(submap.map_kernel(kernel))

# }}}


# {{{ find substitution rules by glob patterns

def find_rules_matching(kernel, pattern):
    """
    :pattern: A shell-style glob pattern.
    """

    from loopy.match import re_from_glob
    pattern = re_from_glob(pattern)

    return [r for r in kernel.substitutions if pattern.match(r)]


def find_one_rule_matching(program, pattern):
    rules = []
    for in_knl_callable in program.callables_table.values():
        if isinstance(in_knl_callable, CallableKernel):
            knl = in_knl_callable.subkernel
            rules.extend(find_rules_matching(knl, pattern))
        elif isinstance(in_knl_callable, ScalarCallable):
            pass
        else:
            raise NotImplementedError("Unknown callable types %s." % (
                type(in_knl_callable).__name__))

    if len(rules) > 1:
        raise ValueError("more than one substitution rule matched '%s'"
                % pattern)
    if not rules:
        raise ValueError("no substitution rule matched '%s'" % pattern)

    return rules[0]

# }}}


# vim: foldmethod=marker

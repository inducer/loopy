from __future__ import division
from __future__ import absolute_import
import six

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
        get_dependencies, SubstitutionMapper,
        ExpandingIdentityMapper)
from loopy.diagnostic import LoopyError
from pymbolic.mapper.substitutor import make_subst_func

from pytools import Record
from pymbolic import var


import logging
logger = logging.getLogger(__name__)


class ExprDescriptor(Record):
    __slots__ = ["insn", "expr", "unif_var_dict"]


def extract_subst(kernel, subst_name, template, parameters=()):
    """
    :arg subst_name: The name of the substitution rule to be created.
    :arg template: Unification template expression.
    :arg parameters: An iterable of parameters used in
        *template*, or a comma-separated string of the same.

    All targeted subexpressions must match ('unify with') *template*
    The template may contain '*' wildcards that will have to match exactly across all
    unifications.
    """

    if isinstance(template, str):
        from pymbolic import parse
        template = parse(template)

    if isinstance(parameters, str):
        parameters = tuple(
                s.strip() for s in parameters.split(","))

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
            new_iname = var_name_gen(iname)
            old_to_new[iname] = var(new_iname)
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
                        unif_var_dict=dict((lhs.name, rhs)
                            for lhs, rhs in urec.equations)))
        else:
            mapper.fallback_mapper(expr)
            # can't nest, don't recurse

    from loopy.symbolic import (
            CallbackMapper, WalkMapper, IdentityMapper)
    dfmapper = CallbackMapper(gather_exprs, WalkMapper())

    for insn in kernel.instructions:
        dfmapper(insn.expression)

    for sr in six.itervalues(kernel.substitutions):
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

    from loopy.kernel.data import SubstitutionRule
    new_substs = {
            subst_name: SubstitutionRule(
                name=subst_name,
                arguments=tuple(parameters),
                expression=template,
                )}

    for subst in six.itervalues(kernel.substitutions):
        new_substs[subst.name] = subst.copy(
                expression=cbmapper(subst.expression))

    # }}}

    return kernel.copy(
            instructions=new_insns,
            substitutions=new_substs)


# {{{ temporary_to_subst

class TemporaryToSubstChanger(ExpandingIdentityMapper):
    def __init__(self, kernel, temp_name, definition_insn_ids,
            usage_to_definition, within):
        self.var_name_gen = kernel.get_var_name_generator()

        super(TemporaryToSubstChanger, self).__init__(
                kernel.substitutions, self.var_name_gen)

        self.kernel = kernel
        self.temp_name = temp_name
        self.definition_insn_ids = definition_insn_ids
        self.usage_to_definition = usage_to_definition

        self.within = within

        self.definition_insn_id_to_subst_name = {}

        self.saw_unmatched_usage_sites = {}
        for def_id in self.definition_insn_ids:
            self.saw_unmatched_usage_sites[def_id] = False

    def get_subst_name(self, def_insn_id):
        try:
            return self.definition_insn_id_to_subst_name[def_insn_id]
        except KeyError:
            subst_name = self.var_name_gen(self.temp_name+"_subst")
            self.definition_insn_id_to_subst_name[def_insn_id] = subst_name
            return subst_name

    def map_variable(self, expr, expn_state):
        if expr.name == self.temp_name:
            result = self.transform_access(None, expn_state)
            if result is not None:
                return result

        return super(TemporaryToSubstChanger, self).map_variable(
                expr, expn_state)

    def map_subscript(self, expr, expn_state):
        if expr.aggregate.name == self.temp_name:
            result = self.transform_access(expr.index, expn_state)
            if result is not None:
                return result

        return super(TemporaryToSubstChanger, self).map_subscript(
                expr, expn_state)

    def transform_access(self, index, expn_state):
        my_insn_id = expn_state.stack[0][0]

        if my_insn_id in self.definition_insn_ids:
            return None

        my_def_id = self.usage_to_definition[my_insn_id]

        if not self.within(expn_state.stack):
            self.saw_unmatched_usage_sites[my_def_id] = True
            return None

        my_insn_id = expn_state.stack[0][0]

        subst_name = self.get_subst_name(my_def_id)

        from pymbolic import var
        if index is None:
            return var(subst_name)
        else:
            return var(subst_name)(*index)


def temporary_to_subst(kernel, temp_name, within=None):
    """Extract an assignment to a temporary variable
    as a :ref:`substituion-rule`. The temporary may

    :arg within: a stack match as understood by
        :func:`loopy.context_matching.parse_stack_match`.

    This operation will change all usage sites
    of *temp_name* matched by *within*. If there
    are further usage sites of *temp_name*, then
    the original assignment to *temp_name* as well
    as the temporary variable is left in place.
    """

    # {{{ establish the relevant definition of temp_name for each usage site

    dep_kernel = expand_subst(kernel)
    from loopy.preprocess import add_default_dependencies
    dep_kernel = add_default_dependencies(dep_kernel)

    id_to_insn = dep_kernel.id_to_insn

    def get_relevant_definition_insn_id(usage_insn_id):
        insn = id_to_insn[usage_insn_id]

        def_id = set()
        for dep_id in insn.insn_deps:
            dep_insn = id_to_insn[dep_id]
            if temp_name in dep_insn.write_dependency_names():
                if temp_name in dep_insn.read_dependency_names():
                    raise LoopyError("instruction '%s' both reads *and* "
                            "writes '%s'--cannot transcribe to substitution "
                            "rule" % (dep_id, temp_name))

                def_id.add(dep_id)
            else:
                rec_result = get_relevant_definition_insn_id(dep_id)
                if rec_result is not None:
                    def_id.add(rec_result)

        if len(def_id) > 1:
            raise LoopyError("more than one write to '%s' found in "
                    "depdendencies of '%s'--definition cannot be resolved "
                    "(writer instructions ids: %s)"
                    % (temp_name, usage_insn_id, ", ".join(def_id)))

        if not def_id:
            return None
        else:
            def_id, = def_id

        return def_id

    usage_to_definition = {}

    for insn in kernel.instructions:
        if temp_name not in insn.read_dependency_names():
            continue

        def_id = get_relevant_definition_insn_id(insn.id)
        if def_id is None:
            raise LoopyError("no write to '%s' found in dependency tree "
                    "of '%s'--definition cannot be resolved"
                    % (temp_name, insn.id))

        usage_to_definition[insn.id] = def_id

    definition_insn_ids = set()
    for insn in kernel.instructions:
        if temp_name in insn.write_dependency_names():
            definition_insn_ids.add(insn.id)

    # }}}

    from loopy.context_matching import parse_stack_match
    within = parse_stack_match(within)

    tts = TemporaryToSubstChanger(kernel, temp_name, definition_insn_ids,
            usage_to_definition, within)

    kernel = tts.map_kernel(kernel)

    from loopy.kernel.data import SubstitutionRule

    # {{{ create new substitution rules

    new_substs = kernel.substitutions.copy()
    for def_id, subst_name in six.iteritems(tts.definition_insn_id_to_subst_name):
        def_insn = id_to_insn[def_id]

        (_, indices), = def_insn.assignees_and_indices()

        arguments = []

        from pymbolic.primitives import Variable
        for i in indices:
            if not isinstance(i, Variable):
                raise LoopyError("In defining instruction '%s': "
                        "asignee index '%s' is not a plain variable. "
                        "Perhaps use loopy.affine_map_inames() "
                        "to perform substitution." % (def_id, i))

            arguments.append(i.name)

        new_substs[subst_name] = SubstitutionRule(
                name=subst_name,
                arguments=tuple(arguments),
                expression=def_insn.expression)

    # }}}

    # {{{ delete temporary variable if possible

    new_temp_vars = kernel.temporary_variables
    if not any(six.itervalues(tts.saw_unmatched_usage_sites)):
        # All usage sites matched--they're now substitution rules.
        # We can get rid of the variable.

        new_temp_vars = new_temp_vars.copy()
        del new_temp_vars[temp_name]

    # }}}

    import loopy as lp
    kernel = lp.remove_instructions(
            kernel,
            set(
                insn_id
                for insn_id, still_used in six.iteritems(
                    tts.saw_unmatched_usage_sites)
                if not still_used))

    return kernel.copy(
            substitutions=new_substs,
            temporary_variables=new_temp_vars,
            )

# }}}


def expand_subst(kernel, ctx_match=None):
    logger.debug("%s: expand subst" % kernel.name)

    from loopy.symbolic import SubstitutionRuleExpander
    from loopy.context_matching import parse_stack_match
    submap = SubstitutionRuleExpander(kernel.substitutions,
            kernel.get_var_name_generator(),
            parse_stack_match(ctx_match))

    kernel = submap.map_kernel(kernel)
    if ctx_match is None:
        return kernel.copy(substitutions={})
    else:
        return kernel

# vim: foldmethod=marker

"""Pymbolic mappers for loopy."""

from __future__ import division, absolute_import
import six
from six.moves import range, zip, reduce

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


from six.moves import intern

from pytools import memoize, memoize_method, Record
import pytools.lex

from pymbolic.primitives import (
        Leaf, AlgebraicLeaf, Expression, Variable, CommonSubexpression)

from pymbolic.mapper import (
        CombineMapper as CombineMapperBase,
        IdentityMapper as IdentityMapperBase,
        WalkMapper as WalkMapperBase,
        CallbackMapper as CallbackMapperBase,
        )
from pymbolic.mapper.evaluator import \
        EvaluationMapper as EvaluationMapperBase
from pymbolic.mapper.substitutor import \
        SubstitutionMapper as SubstitutionMapperBase
from pymbolic.mapper.stringifier import \
        StringifyMapper as StringifyMapperBase
from pymbolic.mapper.dependency import \
        DependencyMapper as DependencyMapperBase
from pymbolic.mapper.coefficient import \
        CoefficientCollector as CoefficientCollectorBase
from pymbolic.mapper.unifier import UnidirectionalUnifier \
        as UnidirectionalUnifierBase
from pymbolic.mapper.constant_folder import \
        ConstantFoldingMapper as ConstantFoldingMapperBase

from pymbolic.parser import Parser as ParserBase

import islpy as isl
from islpy import dim_type

import re
import numpy as np


# {{{ mappers with support for loopy-specific primitives

class IdentityMapperMixin(object):
    def map_group_hw_index(self, expr, *args):
        return expr

    def map_local_hw_index(self, expr, *args):
        return expr

    def map_loopy_function_identifier(self, expr, *args):
        return expr

    def map_reduction(self, expr, *args):
        mapped_inames = [self.rec(Variable(iname), *args) for iname in expr.inames]

        new_inames = []
        for iname, new_sym_iname in zip(expr.inames, mapped_inames):
            if not isinstance(new_sym_iname, Variable):
                from loopy.diagnostic import LoopyError
                raise LoopyError("%s did not map iname '%s' to a variable"
                        % (type(self).__name__, iname))

            new_inames.append(new_sym_iname.name)

        return Reduction(
                expr.operation, tuple(new_inames), self.rec(expr.expr, *args),
                allow_simultaneous=expr.allow_simultaneous)

    def map_tagged_variable(self, expr, *args):
        # leaf, doesn't change
        return expr

    def map_type_annotation(self, expr, *args):
        return TypeAnnotation(expr.type, self.rec(expr.child))

    map_linear_subscript = IdentityMapperBase.map_subscript


class IdentityMapper(IdentityMapperBase, IdentityMapperMixin):
    pass


class PartialEvaluationMapper(EvaluationMapperBase, IdentityMapperMixin):
    def map_variable(self, expr):
        return expr


class WalkMapper(WalkMapperBase):
    def map_group_hw_index(self, expr, *args):
        self.visit(expr)

    def map_local_hw_index(self, expr, *args):
        self.visit(expr)

    def map_reduction(self, expr, *args):
        if not self.visit(expr):
            return

        self.rec(expr.expr, *args)

    map_tagged_variable = WalkMapperBase.map_variable

    def map_loopy_function_identifier(self, expr, *args):
        self.visit(expr)

    map_linear_subscript = WalkMapperBase.map_subscript


class CallbackMapper(CallbackMapperBase, IdentityMapper):
    map_reduction = CallbackMapperBase.map_constant


class CombineMapper(CombineMapperBase):
    def map_reduction(self, expr):
        return self.rec(expr.expr)

    map_linear_subscript = CombineMapperBase.map_subscript


class SubstitutionMapper(SubstitutionMapperBase, IdentityMapperMixin):
    pass


class ConstantFoldingMapper(ConstantFoldingMapperBase,
        IdentityMapperMixin):
    pass


class StringifyMapper(StringifyMapperBase):
    def map_group_hw_index(self, expr, enclosing_prec):
        return "grp.%d" % expr.index

    def map_local_hw_index(self, expr, enclosing_prec):
        return "loc.%d" % expr.index

    def map_reduction(self, expr, prec):
        return "%sreduce(%s, [%s], %s)" % (
                "simul_" if expr.allow_simultaneous else "",
                expr.operation, ", ".join(expr.inames), expr.expr)

    def map_tagged_variable(self, expr, prec):
        return "%s$%s" % (expr.name, expr.tag)

    def map_linear_subscript(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_CALL, PREC_NONE
        return self.parenthesize_if_needed(
                self.format("%s[[%s]]",
                    self.rec(expr.aggregate, PREC_CALL),
                    self.rec(expr.index, PREC_NONE)),
                enclosing_prec, PREC_CALL)

    def map_loopy_function_identifier(self, expr, enclosing_prec):
        return "%s<%s>" % (
                type(expr).__name__,
                ", ".join(str(a) for a in expr.__getinitargs__()))


class UnidirectionalUnifier(UnidirectionalUnifierBase):
    def map_reduction(self, expr, other, unis):
        if not isinstance(other, type(expr)):
            return self.treat_mismatch(expr, other, unis)
        if (expr.inames != other.inames
                or type(expr.operation) != type(other.operation)  # noqa
                ):
            return []

        return self.rec(expr.expr, other.expr, unis)

    def map_tagged_variable(self, expr, other, urecs):
        new_uni_record = self.unification_record_from_equation(
                expr, other)
        if new_uni_record is None:
            # Check if the variables match literally--that's ok, too.
            if (isinstance(other, TaggedVariable)
                    and expr.name == other.name
                    and expr.tag == other.tag
                    and expr.name not in self.lhs_mapping_candidates):
                return urecs
            else:
                return []
        else:
            from pymbolic.mapper.unifier import unify_many
            return unify_many(urecs, new_uni_record)


class DependencyMapper(DependencyMapperBase):
    def map_group_hw_index(self, expr):
        return set()

    def map_local_hw_index(self, expr):
        return set()

    def map_call(self, expr, *args):
        # Loopy does not have first-class functions. Do not descend
        # into 'function' attribute of Call.
        return self.combine(
                self.rec(child, *args) for child in expr.parameters)

    def map_reduction(self, expr):
        return (self.rec(expr.expr)
                - set(Variable(iname) for iname in expr.inames))

    def map_tagged_variable(self, expr):
        return set([expr])

    def map_loopy_function_identifier(self, expr):
        return set()

    map_linear_subscript = DependencyMapperBase.map_subscript


class SubstitutionRuleExpander(IdentityMapper):
    def __init__(self, rules):
        self.rules = rules

    def map_variable(self, expr):
        if expr.name in self.rules:
            return self.map_substitution(expr.name, self.rules[expr.name], ())
        else:
            return super(SubstitutionRuleExpander, self).map_variable(expr)

    def map_call(self, expr):
        if expr.function.name in self.rules:
            return self.map_substitution(
                    expr.function.name,
                    self.rules[expr.function.name],
                    expr.parameters)
        else:
            return super(SubstitutionRuleExpander, self).map_call(expr)

    def map_substitution(self, name, rule, arguments):
        if len(rule.arguments) != len(arguments):
            from loopy.diagnostic import LoopyError
            raise LoopyError("number of arguments to '%s' does not match "
                    "definition" % name)

        from pymbolic.mapper.substitutor import make_subst_func
        submap = SubstitutionMapper(
                make_subst_func(
                    dict(zip(rule.arguments, arguments))))

        expr = submap(rule.expression)

        return self.rec(expr)

# }}}


# {{{ loopy-specific primitives

class HardwareAxisIndex(Leaf):
    def __init__(self, axis):
        self.axis = axis

    def stringifier(self):
        return StringifyMapper

    def __getinitargs__(self):
        return (self.axis,)

    init_arg_names = ("axis",)


class GroupHardwareAxisIndex(HardwareAxisIndex):
    mapper_method = "map_group_hw_index"


class LocalHardwareAxisIndex(HardwareAxisIndex):
    mapper_method = "map_local_hw_index"


class FunctionIdentifier(Leaf):
    """A base class for symbols representing functions."""

    init_arg_names = ()

    def stringifier(self):
        return StringifyMapper

    mapper_method = intern("map_loopy_function_identifier")


class TypedCSE(CommonSubexpression):
    """A :class:`pymbolic.primitives.CommonSubexpression` annotated with
    a :class:`numpy.dtype`.
    """

    def __init__(self, child, prefix=None, dtype=None):
        CommonSubexpression.__init__(self, child, prefix)
        self.dtype = dtype

    def __getinitargs__(self):
        return (self.child, self.dtype, self.prefix)

    def get_extra_properties(self):
        return dict(dtype=self.dtype)


class TypeAnnotation(Expression):
    def __init__(self, type, child):
        super(TypeAnnotation, self).__init__()
        self.type = type
        self.child = child

    def __getinitargs__(self):
        return (self.type, self.child)

    mapper_method = intern("map_type_annotation")


class TaggedVariable(Variable):
    """This is an identifier with a tag, such as 'matrix$one', where
    'one' identifies this specific use of the identifier. This mechanism
    may then be used to address these uses--such as by prefetching only
    accesses tagged a certain way.
    """

    init_arg_names = ("name", "tag")

    def __init__(self, name, tag):
        Variable.__init__(self, name)
        self.tag = tag

    def __getinitargs__(self):
        return self.name, self.tag

    def stringifier(self):
        return StringifyMapper

    mapper_method = intern("map_tagged_variable")


class Reduction(AlgebraicLeaf):
    """Represents a reduction operation on :attr:`expr`
    across :attr:`inames`.

    .. attribute:: operation

        an instance of :class:`loopy.library.reduction.ReductionOperation`

    .. attribute:: inames

        a list of inames across which reduction on :attr:`expr` is being
        carried out.

    .. attribute:: expr

        The expression (as a :class:`pymbolic.primitives.Expression`)
        on which reduction is performed.

    .. attribute:: allow_simultaneous

        A :class:`bool`. If not *True*, an iname is allowed to be used
        in precisely one reduction, to avoid mis-nesting errors.
    """

    init_arg_names = ("operation", "inames", "expr", "allow_simultaneous")

    def __init__(self, operation, inames, expr, allow_simultaneous=False):
        if isinstance(inames, str):
            inames = tuple(iname.strip() for iname in inames.split(","))

        elif isinstance(inames, Variable):
            inames = (inames,)

        assert isinstance(inames, tuple)

        def strip_var(iname):
            if isinstance(iname, Variable):
                iname = iname.name

            assert isinstance(iname, str)
            return iname

        inames = tuple(strip_var(iname) for iname in inames)

        if isinstance(operation, str):
            from loopy.library.reduction import parse_reduction_op
            operation = parse_reduction_op(operation)

        from loopy.library.reduction import ReductionOperation
        assert isinstance(operation, ReductionOperation)

        self.operation = operation
        self.inames = inames
        self.expr = expr
        self.allow_simultaneous = allow_simultaneous

    def __getinitargs__(self):
        return (self.operation, self.inames, self.expr, self.allow_simultaneous)

    def get_hash(self):
        return hash((self.__class__, self.operation, self.inames,
            self.expr))

    def is_equal(self, other):
        return (other.__class__ == self.__class__
                and other.operation == self.operation
                and other.inames == self.inames
                and other.expr == self.expr)

    def stringifier(self):
        return StringifyMapper

    @property
    @memoize_method
    def inames_set(self):
        return set(self.inames)

    mapper_method = intern("map_reduction")


class LinearSubscript(AlgebraicLeaf):
    """Represents a linear index into a multi-dimensional array, completely
    ignoring any multi-dimensional layout.
    """

    init_arg_names = ("aggregate", "index")

    def __init__(self, aggregate, index):
        self.aggregate = aggregate
        self.index = index

    def __getinitargs__(self):
        return self.aggregate, self.index

    def stringifier(self):
        return StringifyMapper

    mapper_method = intern("map_linear_subscript")

# }}}


@memoize
def get_dependencies(expr):
    dep_mapper = DependencyMapper(composite_leaves=False)
    return frozenset(dep.name for dep in dep_mapper(expr))


# {{{ rule-aware mappers

def parse_tagged_name(expr):
    if isinstance(expr, TaggedVariable):
        return expr.name, expr.tag
    elif isinstance(expr, Variable):
        return expr.name, None
    else:
        raise RuntimeError("subst rule name not understood: %s" % expr)


class ExpansionState(Record):
    """
    .. attribute:: kernel
    .. attribute:: instruction

    .. attribute:: stack

        a tuple representing the current expansion stack, as a tuple
        of (name, tag) pairs.

    .. attribute:: arg_context

        a dict representing current argument values
    """

    @property
    def insn_id(self):
        return self.instruction.id

    def apply_arg_context(self, expr):
        from pymbolic.mapper.substitutor import make_subst_func
        return SubstitutionMapper(
                make_subst_func(self.arg_context))(expr)


class SubstitutionRuleRenamer(IdentityMapper):
    def __init__(self, renames):
        self.renames = renames

    def map_call(self, expr):
        if not isinstance(expr.function, Variable):
            return IdentityMapper.map_call(self, expr)

        name, tag = parse_tagged_name(expr.function)

        new_name = self.renames.get(name)
        if new_name is None:
            return IdentityMapper.map_call(self, expr)

        if tag is None:
            sym = Variable(new_name)
        else:
            sym = TaggedVariable(new_name, tag)

        return type(expr)(sym, tuple(self.rec(child) for child in expr.parameters))

    def map_variable(self, expr):
        name, tag = parse_tagged_name(expr)

        new_name = self.renames.get(name)
        if new_name is None:
            return IdentityMapper.map_variable(self, expr)

        if tag is None:
            return Variable(new_name)
        else:
            return TaggedVariable(new_name, tag)


def rename_subst_rules_in_instructions(insns, renames):
    subst_renamer = SubstitutionRuleRenamer(renames)

    return [
            insn.with_transformed_expressions(subst_renamer)
            for insn in insns]


class SubstitutionRuleMappingContext(object):
    def __init__(self, old_subst_rules, make_unique_var_name):
        self.old_subst_rules = old_subst_rules
        self.make_unique_var_name = make_unique_var_name

        # maps subst rule (args, bodies) to (names, original_name)
        self.subst_rule_registry = dict(
                ((rule.arguments, rule.expression), (name, name))
                for name, rule in six.iteritems(old_subst_rules))

        # maps subst rule (args, bodies) to use counts
        self.subst_rule_use_count = {}

    def register_subst_rule(self, original_name, args, body):
        """Returns a name (as a string) for a newly created substitution
        rule.
        """
        key = (args, body)
        reg_value = self.subst_rule_registry.get(key)

        if reg_value is None:
            new_name = self.make_unique_var_name(original_name)
            self.subst_rule_registry[key] = (new_name, original_name)
        else:
            new_name, _ = reg_value

        self.subst_rule_use_count[key] = self.subst_rule_use_count.get(key, 0) + 1
        return new_name

    def _get_new_substitutions_and_renames(self):
        """This makes a new dictionary of substitutions from the ones
        encountered in mapping all the encountered expressions.
        It tries hard to keep substitution names the same--i.e.
        if all derivative versions of a substitution rule ended
        up with the same mapped version, then this version should
        retain the name that the substitution rule had previously.
        Unfortunately, this can't be done in a single pass, and so
        the routine returns an additional dictionary *subst_renames*
        of renamings to be performed on the processed expressions.

        The returned substitutions already have the rename applied
        to them.

        :returns: (new_substitutions, subst_renames)
        """

        from loopy.kernel.data import SubstitutionRule

        orig_name_histogram = {}
        for key, (name, orig_name) in six.iteritems(self.subst_rule_registry):
            if self.subst_rule_use_count.get(key, 0):
                orig_name_histogram[orig_name] = \
                        orig_name_histogram.get(orig_name, 0) + 1

        result = {}
        renames = {}

        for key, (name, orig_name) in six.iteritems(self.subst_rule_registry):
            args, body = key

            if self.subst_rule_use_count.get(key, 0):
                if orig_name_histogram[orig_name] == 1 and name != orig_name:
                    renames[name] = orig_name
                    name = orig_name

                result[name] = SubstitutionRule(
                        name=name,
                        arguments=args,
                        expression=body)

        # {{{ perform renames on new substitutions

        subst_renamer = SubstitutionRuleRenamer(renames)

        renamed_result = {}
        for name, rule in six.iteritems(result):
            renamed_result[name] = rule.copy(
                    expression=subst_renamer(rule.expression))

        # }}}

        return renamed_result, renames

    def finish_kernel(self, kernel):
        new_substs, renames = self._get_new_substitutions_and_renames()

        new_insns = rename_subst_rules_in_instructions(kernel.instructions, renames)

        return kernel.copy(
            substitutions=new_substs,
            instructions=new_insns)


class RuleAwareIdentityMapper(IdentityMapper):
    """Note: the third argument dragged around by this mapper is the
    current :class:`ExpansionState`.

    Subclasses of this must be careful to not touch identifiers that
    are in :attr:`ExpansionState.arg_context`.
    """

    def __init__(self, rule_mapping_context):
        self.rule_mapping_context = rule_mapping_context

    def map_variable(self, expr, expn_state):
        name, tag = parse_tagged_name(expr)
        if name not in self.rule_mapping_context.old_subst_rules:
            return IdentityMapper.map_variable(self, expr, expn_state)
        else:
            return self.map_substitution(name, tag, (), expn_state)

    def map_call(self, expr, expn_state):
        if not isinstance(expr.function, Variable):
            return IdentityMapper.map_call(self, expr, expn_state)

        name, tag = parse_tagged_name(expr.function)

        if name not in self.rule_mapping_context.old_subst_rules:
            return super(RuleAwareIdentityMapper, self).map_call(expr, expn_state)
        else:
            return self.map_substitution(name, tag, expr.parameters, expn_state)

    @staticmethod
    def make_new_arg_context(rule_name, arg_names, arguments, arg_context):
        if len(arg_names) != len(arguments):
            raise RuntimeError("Rule '%s' invoked with %d arguments (needs %d)"
                    % (rule_name, len(arguments), len(arg_names), ))

        from pymbolic.mapper.substitutor import make_subst_func
        arg_subst_map = SubstitutionMapper(make_subst_func(arg_context))
        return dict(
                (formal_arg_name, arg_subst_map(arg_value))
                for formal_arg_name, arg_value in zip(arg_names, arguments))

    def map_substitution(self, name, tag, arguments, expn_state):
        rule = self.rule_mapping_context.old_subst_rules[name]

        rec_arguments = self.rec(arguments, expn_state)

        if tag is None:
            tags = None
        else:
            tags = (tag,)

        new_expn_state = expn_state.copy(
                stack=expn_state.stack + ((name, tags),),
                arg_context=self.make_new_arg_context(
                    name, rule.arguments, rec_arguments, expn_state.arg_context))

        result = self.rec(rule.expression, new_expn_state)

        new_name = self.rule_mapping_context.register_subst_rule(
                name, rule.arguments, result)

        if tag is None:
            sym = Variable(new_name)
        else:
            sym = TaggedVariable(new_name, tag)

        if arguments:
            return sym(*rec_arguments)
        else:
            return sym

    def __call__(self, expr, kernel, insn):
        from loopy.kernel.data import InstructionBase
        assert insn is None or isinstance(insn, InstructionBase)

        return IdentityMapper.__call__(self, expr,
                ExpansionState(
                    kernel=kernel,
                    instruction=insn,
                    stack=(),
                    arg_context={}))

    def map_instruction(self, kernel, insn):
        return insn

    def map_kernel(self, kernel):
        new_insns = [
                # While subst rules are not allowed in assignees, the mapper
                # may perform tasks entirely unrelated to subst rules, so
                # we must map assignees, too.
                self.map_instruction(kernel,
                    insn.with_transformed_expressions(self, kernel, insn))
                for insn in kernel.instructions]

        return kernel.copy(instructions=new_insns)


class RuleAwareSubstitutionMapper(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, subst_func, within):
        super(RuleAwareSubstitutionMapper, self).__init__(rule_mapping_context)

        self.subst_func = subst_func
        self.within = within

    def map_variable(self, expr, expn_state):
        if (expr.name in expn_state.arg_context
                or not self.within(
                    expn_state.kernel, expn_state.instruction, expn_state.stack)):
            return super(RuleAwareSubstitutionMapper, self).map_variable(
                    expr, expn_state)

        result = self.subst_func(expr)
        if result is not None:
            return result
        else:
            return super(RuleAwareSubstitutionMapper, self).map_variable(
                    expr, expn_state)


class RuleAwareSubstitutionRuleExpander(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, rules, within):
        super(RuleAwareSubstitutionRuleExpander, self).__init__(rule_mapping_context)

        self.rules = rules
        self.within = within

    def map_substitution(self, name, tag, arguments, expn_state):
        if tag is None:
            tags = None
        else:
            tags = (tag,)

        new_stack = expn_state.stack + ((name, tags),)

        if self.within(expn_state.kernel, expn_state.instruction, new_stack):
            # expand
            rule = self.rules[name]

            new_expn_state = expn_state.copy(
                    stack=new_stack,
                    arg_context=self.make_new_arg_context(
                        name, rule.arguments, arguments, expn_state.arg_context))

            result = self.rec(rule.expression, new_expn_state)

            # substitute in argument values
            from pymbolic.mapper.substitutor import make_subst_func
            subst_map = SubstitutionMapper(make_subst_func(
                new_expn_state.arg_context))

            return subst_map(result)

        else:
            # do not expand
            return super(RuleAwareSubstitutionRuleExpander, self).map_substitution(
                    name, tag, arguments, expn_state)

# }}}


# {{{ functions to primitives, parsing

class VarToTaggedVarMapper(IdentityMapper):
    def map_variable(self, expr):
        dollar_idx = expr.name.find("$")
        if dollar_idx == -1:
            return expr
        else:
            return TaggedVariable(expr.name[:dollar_idx],
                    expr.name[dollar_idx+1:])


class FunctionToPrimitiveMapper(IdentityMapper):
    """Looks for invocations of a function called 'cse' or 'reduce' and
    turns those into the actual pymbolic primitives used for that.
    """

    def _parse_reduction(self, operation, inames, red_expr,
            allow_simultaneous=False):
        if isinstance(inames, Variable):
            inames = (inames,)

        if not isinstance(inames, (tuple)):
            raise TypeError("iname argument to reduce() must be a symbol "
                    "or a tuple of symbols")

        processed_inames = []
        for iname in inames:
            if not isinstance(iname, Variable):
                raise TypeError("iname argument to reduce() must be a symbol "
                        "or a tuple or a tuple of symbols")

            processed_inames.append(iname.name)

        return Reduction(operation, tuple(processed_inames), red_expr,
                allow_simultaneous=allow_simultaneous)

    def map_call(self, expr):
        from loopy.library.reduction import parse_reduction_op

        from pymbolic.primitives import Variable
        if not isinstance(expr.function, Variable):
            return IdentityMapper.map_call(self, expr)

        name = expr.function.name
        if name == "cse":
            from pymbolic.primitives import CommonSubexpression
            if len(expr.parameters) in [1, 2]:
                if len(expr.parameters) == 2:
                    if not isinstance(expr.parameters[1], Variable):
                        raise TypeError("second argument to cse() must be a symbol")
                    tag = expr.parameters[1].name
                else:
                    tag = None

                return CommonSubexpression(
                        self.rec(expr.parameters[0]), tag)
            else:
                raise TypeError("cse takes two arguments")

        elif name in ["reduce", "simul_reduce"]:
            if len(expr.parameters) == 3:
                operation, inames, red_expr = expr.parameters

                if not isinstance(operation, Variable):
                    raise TypeError("operation argument to reduce() "
                            "must be a symbol")

                operation = parse_reduction_op(operation.name)
                return self._parse_reduction(operation, inames, self.rec(red_expr),
                        allow_simultaneous=(name == "simul_reduce"))
            else:
                raise TypeError("invalid 'reduce' calling sequence")

        elif name == "if":
            if len(expr.parameters) == 3:
                from pymbolic.primitives import If
                return If(*expr.parameters)
            else:
                raise TypeError("if takes three arguments")

        else:
            # see if 'name' is an existing reduction op

            operation = parse_reduction_op(name)
            if operation:
                if len(expr.parameters) != 2:
                    raise RuntimeError("invalid invocation of "
                            "reduction operation '%s'" % expr.function.name)

                inames, red_expr = expr.parameters
                return self._parse_reduction(operation, inames, self.rec(red_expr))

            else:
                return IdentityMapper.map_call(self, expr)


# {{{ customization to pymbolic parser

_open_dbl_bracket = intern("open_dbl_bracket")

TRAILING_FLOAT_TAG_RE = re.compile("^(.*?)([a-zA-Z]*)$")


class LoopyParser(ParserBase):
    lex_table = [
            (_open_dbl_bracket, pytools.lex.RE(r"\[\[")),
            ] + ParserBase.lex_table

    def parse_float(self, s):
        match = TRAILING_FLOAT_TAG_RE.match(s)

        val = match.group(1)
        tag = frozenset(match.group(2))
        if tag == frozenset("j"):
            return np.float64(val)*np.complex128(1j)
        elif tag == frozenset("jf"):
            return np.float32(val)*np.complex64(1j)
        elif tag == frozenset("f"):
            return np.float32(val)
        elif tag == frozenset("d"):
            return np.float64(val)
        else:
            return float(val)  # generic float

    def parse_prefix(self, pstate):
        from pymbolic.parser import _PREC_UNARY, _less, _greater, _identifier
        if pstate.is_next(_less):
            pstate.advance()
            if pstate.is_next(_greater):
                typename = None
                pstate.advance()
            else:
                pstate.expect(_identifier)
                typename = pstate.next_str()
                pstate.advance()
                pstate.expect(_greater)
                pstate.advance()

            return TypeAnnotation(
                    typename,
                    self.parse_expression(pstate, _PREC_UNARY))
        else:
            return super(LoopyParser, self).parse_prefix(pstate)

    def parse_postfix(self, pstate, min_precedence, left_exp):
        from pymbolic.parser import _PREC_CALL, _closebracket
        if pstate.next_tag() is _open_dbl_bracket and _PREC_CALL > min_precedence:
            pstate.advance()
            pstate.expect_not_end()
            left_exp = LinearSubscript(left_exp, self.parse_expression(pstate))
            pstate.expect(_closebracket)
            pstate.advance()
            pstate.expect(_closebracket)
            pstate.advance()
            return left_exp, True

        return ParserBase.parse_postfix(self, pstate, min_precedence, left_exp)

# }}}


def parse(expr_str):
    return VarToTaggedVarMapper()(
            FunctionToPrimitiveMapper()(LoopyParser()(expr_str)))

# }}}


# {{{ coefficient collector

class CoefficientCollector(CoefficientCollectorBase):
    map_tagged_variable = CoefficientCollectorBase.map_variable

    def map_subscript(self, expr):
        raise RuntimeError("cannot gather coefficients--indirect addressing in use")

# }}}


# {{{ variable index expression collector

class ArrayAccessFinder(CombineMapper):
    def __init__(self, tgt_vector_name=None):
        self.tgt_vector_name = tgt_vector_name

    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, expr):
        return set()

    def map_algebraic_leaf(self, expr):
        return set()

    def map_subscript(self, expr):
        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        if self.tgt_vector_name is None \
                or expr.aggregate.name == self.tgt_vector_name:
            return set([expr]) | self.rec(expr.index)
        else:
            return CombineMapper.map_subscript(self, expr)

# }}}


# {{{ aff <-> expr conversion

def aff_to_expr(aff, except_name=None, error_on_name=None):
    if except_name is not None and error_on_name is not None:
        raise ValueError("except_name and error_on_name may not be specified "
                "at the same time")
    from pymbolic import var

    except_coeff = 0

    denom = aff.get_denominator_val().to_python()

    result = (aff.get_constant_val()*denom).to_python()
    for dt in [dim_type.in_, dim_type.param]:
        for i in range(aff.dim(dt)):
            coeff = (aff.get_coefficient_val(dt, i)*denom).to_python()
            if coeff:
                dim_name = aff.get_dim_name(dt, i)
                if dim_name == except_name:
                    except_coeff += coeff
                elif dim_name == error_on_name:
                    raise RuntimeError("'%s' occurred in this subexpression--"
                            "this is not allowed" % dim_name)
                else:
                    result += coeff*var(dim_name)

    error_on_name = error_on_name or except_name

    for i in range(aff.dim(dim_type.div)):
        coeff = (aff.get_coefficient_val(dim_type.div, i)*denom).to_python()
        if coeff:
            result += coeff*aff_to_expr(aff.get_div(i), error_on_name=error_on_name)

    if except_name is not None:
        if except_coeff % denom != 0:
            raise RuntimeError("coefficient of '%s' is not divisible by "
                    "aff denominator" % except_name)

        return result // denom, except_coeff // denom
    else:
        return result // denom


def pw_aff_to_expr(pw_aff, int_ok=False):
    if isinstance(pw_aff, int):
        if not int_ok:
            from warnings import warn
            warn("expected PwAff, got int", stacklevel=2)

        return pw_aff

    pieces = pw_aff.get_pieces()

    if len(pieces) != 1:
        raise NotImplementedError("pw_aff_to_expr for multi-piece PwAff instances")

    (set, aff), = pieces
    return aff_to_expr(aff)


def aff_from_expr(space, expr, vars_to_zero=set()):
    zero = isl.Aff.zero_on_domain(isl.LocalSpace.from_space(space))
    context = {}
    for name, (dt, pos) in six.iteritems(space.get_var_dict()):
        if dt == dim_type.set:
            dt = dim_type.in_

        context[name] = zero.set_coefficient_val(dt, pos, 1)

    for name in vars_to_zero:
        context[name] = zero

    from pymbolic import evaluate
    return zero + evaluate(expr, context)

# }}}


# {{{ simplify using aff

def simplify_using_aff(kernel, expr):
    inames = get_dependencies(expr) & kernel.all_inames()

    domain = kernel.get_inames_domain(inames)

    from pymbolic.mapper.evaluator import UnknownVariableError

    try:
        with isl.SuppressedWarnings(kernel.isl_context):
            aff = aff_from_expr(domain.space, expr)
    except isl.Error:
        return expr
    except TypeError:
        return expr
    except UnknownVariableError:
        return expr

    # FIXME: Deal with assumptions, too.
    aff = aff.gist(domain)

    return aff_to_expr(aff)

# }}}


# {{{ expression <-> constraint conversion

def eq_constraint_from_expr(space, expr):
    return isl.Constraint.equality_from_aff(aff_from_expr(space, expr))


def ineq_constraint_from_expr(space, expr):
    return isl.Constraint.inequality_from_aff(aff_from_expr(space, expr))


def constraint_to_expr(cns):
    # Looks like this is ok after all--get_aff() performs some magic.
    # Not entirely sure though... FIXME
    #
    #ls = cns.get_local_space()
    #if ls.dim(dim_type.div):
        #raise RuntimeError("constraint has an existentially quantified variable")

    expr = aff_to_expr(cns.get_aff())

    from pymbolic.primitives import Comparison
    if cns.is_equality():
        return Comparison(expr, "==", 0)
    else:
        return Comparison(expr, ">=", 0)

# }}}


# {{{ Reduction callback mapper

class ReductionCallbackMapper(IdentityMapper):
    def __init__(self, callback):
        self.callback = callback

    def map_reduction(self, expr, **kwargs):
        result = self.callback(expr, self.rec, **kwargs)
        if result is None:
            return IdentityMapper.map_reduction(self, expr, **kwargs)
        return result

# }}}


# {{{ index dependency finding

class IndexVariableFinder(CombineMapper):
    def __init__(self, include_reduction_inames):
        self.include_reduction_inames = include_reduction_inames

    def combine(self, values):
        import operator
        return reduce(operator.or_, values, set())

    def map_constant(self, expr):
        return set()

    def map_algebraic_leaf(self, expr):
        return set()

    def map_subscript(self, expr):
        idx_vars = DependencyMapper()(expr.index)

        from pymbolic.primitives import Variable
        result = set()
        for idx_var in idx_vars:
            if isinstance(idx_var, Variable):
                result.add(idx_var.name)
            else:
                raise RuntimeError("index variable not understood: %s" % idx_var)
        return result

    def map_reduction(self, expr):
        result = self.rec(expr.expr)

        if not (expr.inames_set & result):
            raise RuntimeError("reduction '%s' does not depend on "
                    "reduction inames (%s)" % (expr, ",".join(expr.inames)))
        if self.include_reduction_inames:
            return result
        else:
            return result - expr.inames_set

# }}}


# {{{ wildcard -> unique variable mapper

class WildcardToUniqueVariableMapper(IdentityMapper):
    def __init__(self, unique_var_name_factory):
        self.unique_var_name_factory = unique_var_name_factory

    def map_wildcard(self, expr):
        from pymbolic import var
        return var(self.unique_var_name_factory())

# }}}


# {{{ prime ("'") adder

class PrimeAdder(IdentityMapper):
    def __init__(self, which_vars):
        self.which_vars = which_vars

    def map_variable(self, expr):
        from pymbolic import var
        if expr.name in self.which_vars:
            return var(expr.name+"'")
        else:
            return expr

    def map_tagged_variable(self, expr):
        if expr.name in self.which_vars:
            return TaggedVariable(expr.name+"'", expr.tag)
        else:
            return expr

# }}}


# {{{ get access range

def get_access_range(domain, subscript, assumptions):
    domain, assumptions = isl.align_two(domain,
            assumptions)
    domain = domain & assumptions
    del assumptions

    dims = len(subscript)

    # we build access_map as a set because (idiocy!) Affs
    # cannot live on maps.

    # dims: [domain](dn)[storage]
    access_map = domain

    if isinstance(access_map, isl.BasicSet):
        access_map = isl.Set.from_basic_set(access_map)

    dn = access_map.dim(dim_type.set)
    access_map = access_map.insert_dims(dim_type.set, dn, dims)

    for idim in range(dims):
        idx_aff = aff_from_expr(access_map.get_space(),
                subscript[idim])
        idx_aff = idx_aff.set_coefficient_val(
                dim_type.in_, dn+idim, -1)

        access_map = access_map.add_constraint(
                isl.Constraint.equality_from_aff(idx_aff))

    access_map_as_map = isl.Map.universe(access_map.get_space())
    access_map_as_map = access_map_as_map.intersect_range(access_map)
    access_map = access_map_as_map.move_dims(
            dim_type.in_, 0,
            dim_type.out, 0, dn)
    del access_map_as_map

    return access_map.range()

# }}}


# {{{ access range mapper

class AccessRangeMapper(WalkMapper):
    def __init__(self, kernel, arg_name):
        self.kernel = kernel
        self.arg_name = arg_name
        self.access_range = None
        self.bad_subscripts = []

    def map_subscript(self, expr, inames):
        domain = self.kernel.get_inames_domain(inames)
        WalkMapper.map_subscript(self, expr, inames)

        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        if expr.aggregate.name != self.arg_name:
            return

        subscript = expr.index_tuple

        if not get_dependencies(subscript) <= set(domain.get_var_dict()):
            self.bad_subscripts.append(expr)
            return

        access_range = get_access_range(domain, subscript, self.kernel.assumptions)

        if self.access_range is None:
            self.access_range = access_range
        else:
            if (self.access_range.dim(dim_type.set)
                    != access_range.dim(dim_type.set)):
                raise RuntimeError(
                        "error while determining shape of argument '%s': "
                        "varying number of indices encountered"
                        % self.arg_name)

            self.access_range = self.access_range | access_range

    def map_linear_subscript(self, expr, inames):
        self.rec(expr.index, inames)

        if expr.aggregate.name == self.arg_name:
            self.bad_subscripts.append(expr)

    def map_reduction(self, expr, inames):
        return WalkMapper.map_reduction(self, expr, inames | set(expr.inames))

# }}}


# {{{ is_expression_equal

def is_expression_equal(a, b):
    if a == b:
        return True

    from pymbolic.primitives import Expression
    if isinstance(a, Expression) or isinstance(b, Expression):
        if a is None or b is None:
            return False

        maybe_zero = a - b
        from pymbolic import distribute

        d_result = distribute(maybe_zero)
        return d_result == 0

    else:
        return False


def is_tuple_of_expressions_equal(a, b):
    if a is None or b is None:
        if a is None and b is None:
            return True
        return False

    if not isinstance(a, tuple):
        a = (a,)

    if not isinstance(b, tuple):
        b = (b,)

    if len(a) != len(b):
        return False

    return all(
        is_expression_equal(ai, bi)
        for ai, bi in zip(a, b))

# }}}

# vim: foldmethod=marker

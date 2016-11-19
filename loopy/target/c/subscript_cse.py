"""Common subexpression elimination in array subscripts."""

from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2016 Andreas Kloeckner"

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

import six

from cgen.mapper import IdentityMapper as CASTIdentityMapperBase
import pymbolic.primitives as p
from pytools import Record

from loopy.symbolic import IdentityMapper as ExprIdentityMapper
from loopy.diagnostic import LoopyError
from loopy.target.c import CExpression

import logging
logger = logging.getLogger(__name__)


# {{{ utilities

class CASTIdentityMapper(CASTIdentityMapperBase):
    def map_loopy_scope(self, node, *args, **kwargs):
        return type(node)(
                node.var_subst_map,
                node.available_variables,
                self.rec(node.child, *args, **kwargs))

    def map_loopy_pod(self, node, *args, **kwargs):
        return type(node)(node.ast_builder, node.dtype, node.name)


def generate_all_subsets(l, min_length):
    for bits in range(2**len(l)):
        if bin(bits).count("1") >= min_length:
            yield frozenset(entry for i, entry in enumerate(l) if (1 << i) & bits)


def is_const_product(term):
    return (
            p.is_constant(term)
            or (
                isinstance(term, p.Product)
                and all(is_const_product(ch) for ch in term.children)))


def get_terms(allowable_vars, expr):
    if isinstance(expr, p.Sum):
        terms = expr.children
    else:
        terms = (expr,)

    from loopy.symbolic import get_dependencies

    result = []
    remainder = []
    for term in terms:
        deps = get_dependencies(term)
        if (deps <= allowable_vars
                and not is_const_product(term)):
            result.append(term)
        else:
            remainder.append(term)

    return result, remainder

# }}}


# {{{ counting

class SubscriptSubsetCounter(ExprIdentityMapper):
    def __init__(self, codegen_state, term_set_to_inside_inames_list,
            inside_inames):
        self.codegen_state = codegen_state
        self.term_set_to_inside_inames_list = term_set_to_inside_inames_list
        kernel = codegen_state.kernel
        self.allowable_vars = kernel.all_inames() | kernel.outer_params()
        self.inside_inames = inside_inames

    def map_subscript(self, expr):
        terms, _ = get_terms(self.allowable_vars, expr.index)
        terms = frozenset(terms)
        self.term_set_to_inside_inames_list[terms] = (
                self.term_set_to_inside_inames_list.get(terms, [])
                + [self.inside_inames])


class ASTSubexpressionCollector(CASTIdentityMapper):
    def __init__(self, codegen_state):
        self.term_set_to_inside_inames_list = {}
        self.codegen_state = codegen_state
        self.inside_inames_stack = []

    def map_loopy_scope(self, node):
        if self.inside_inames_stack:
            new_inside_inames = self.inside_inames_stack[-1]
        else:
            new_inside_inames = ()

        new_inside_inames = (
                new_inside_inames + node.available_variables)

        self.inside_inames_stack.append(new_inside_inames)
        result = super(ASTSubexpressionCollector, self).map_loopy_scope(node)
        self.inside_inames_stack.pop()
        return result

    def map_expression(self, expr):
        from pymbolic.primitives import is_constant
        if isinstance(expr, CExpression):
            if self.inside_inames_stack:
                inside_inames = self.inside_inames_stack[-1]
            else:
                inside_inames = ()
            count_mapper = SubscriptSubsetCounter(
                self.codegen_state,
                self.term_set_to_inside_inames_list,
                inside_inames)
            count_mapper(expr.expr)
            return expr
        elif isinstance(expr, str) or is_constant(expr):
            return expr
        else:
            raise LoopyError(
                    "Unexpected expression type: %s" % type(expr).__name__)

# }}}


# {{{ replacing

class SubexpressionReplacementState(Record):
    """
    .. attribute:: codegen_state

    .. attribute:: name_generator

        A callable that can generate new identifiers.

    .. attribute:: term_set_to_inside_inames_list

        A mapping from (summed) sets of subexpressions to a list of tuples of inames
        within which the use is nested, one per use.

    .. attribute:: term_subset_to_inside_inames_list

        A mapping from (summed) subsets of subexpressions to their use counts.

    .. attribute:: available_variables

        A set of variables that subexpressions may refer to.

    .. attribute:: term_set_to_variable

        A mapping from term subsets to their replacement variable names.
    """


def is_simple(term):
    from loopy.symbolic import HardwareAxisIndex

    if p.is_constant(term):
        return True

    if (isinstance(term, p.Variable)
            or isinstance(term, HardwareAxisIndex)):
        return True

    if isinstance(term, p.Product):
        n_constants = 0
        n_simple = 0
        n_other = 0

        for ch in term.children:
            if p.is_constant(ch):
                n_constants += 1
            elif is_simple(ch):
                n_simple += 1
            else:
                n_other += 1

        return n_other == 0 and n_simple <= 1

    return False


def compute_term_subset_to_inside_inames_list(
        term_set_to_inside_inames_list, term_set_to_variable):
    logger.debug("TERM SET TO SUBSET COUNT:")
    for term_set, in_iname_uses in six.iteritems(term_set_to_inside_inames_list):
        logger.debug(
                "%s: %d" % (" + ".join(str(i) for i in term_set),
                    len(in_iname_uses)))

    result = {}
    for code_term_set, in_iname_uses in six.iteritems(
            term_set_to_inside_inames_list):
        logger.debug("CTS: " + " + ".join(str(i) for i in code_term_set))
        interacts_with_var_term_sets = [
                var_term_set
                for var_term_set in six.iterkeys(term_set_to_variable)
                if var_term_set <= code_term_set]

        logger.debug("INTERACTS: " + str(interacts_with_var_term_sets))
        for subset in generate_all_subsets(code_term_set, 1):
            if len(subset) == 1:
                term, = subset
                if is_simple(term):
                    continue

            will_contribute = True

            for var_term_set in interacts_with_var_term_sets:
                if (subset <= var_term_set
                        or (var_term_set & subset
                            and not var_term_set < subset)):
                    will_contribute = False
                    break

            if will_contribute:
                result[subset] = result.get(subset, []) + in_iname_uses

        logger.debug("CTS DONE")

    logger.debug("TERM SUBSET TO COUNT:")
    for term_set, in_iname_uses in six.iteritems(result):
        logger.debug(
                "%s: %d" % (" + ".join(str(i) for i in term_set),
                    len(in_iname_uses)))

    return result


def simplify_terms(terms, term_set_to_variable):
    logger.debug("BUILDING EXPR FOR: " + "+".join(str(s) for s in terms))
    did_something = True
    while did_something:
        did_something = False

        for subset, var_name in sorted(
                six.iteritems(term_set_to_variable),
                # longest first
                key=lambda entry: len(entry[0]), reverse=True):
            if subset <= terms:
                logger.debug("SIMPLIFYING " + "+".join(str(s) for s in subset)
                        + "->" + var_name)
                terms = (
                        (terms - subset)
                        | frozenset([p.Variable(var_name)]))
                did_something = True
                break

    logger.debug("GOT " + "+".join(str(s) for s in terms))

    def term_sort_key(term):
        return str(term)

    return sorted(terms, key=term_sort_key)


class SubscriptSubsetReplacer(ExprIdentityMapper):
    def __init__(self, node_replacer, subex_rep_state):
        self.node_replacer = node_replacer
        self.subex_rep_state = subex_rep_state

    def _process_subscript(self, expr):
        subex_rep_state = self.subex_rep_state

        iname_terms, remainder = get_terms(
            subex_rep_state.codegen_state.kernel.all_inames(),
            expr=expr.index)
        return simplify_terms(
                frozenset(iname_terms),
                subex_rep_state.term_set_to_variable), remainder

    def map_subscript(self, expr):
        iname_terms, remainder = self._process_subscript(expr)

        expr = type(expr)(
                expr.aggregate,
                p.Sum(tuple(iname_terms) + tuple(remainder)))

        return super(SubscriptSubsetReplacer, self).map_subscript(expr)


class ASTSubexpressionReplacer(CASTIdentityMapper):
    def map_loopy_scope(self, node, subex_rep_state):
        codegen_state = subex_rep_state.codegen_state.copy(
                var_subst_map=node.var_subst_map)

        available_variables = (
                subex_rep_state.available_variables
                | frozenset(node.available_variables))

        subex_rep_state = subex_rep_state.copy(
                available_variables=available_variables)

        term_set_to_variable = subex_rep_state.term_set_to_variable.copy()
        term_subset_to_inside_inames_list = \
                subex_rep_state.term_subset_to_inside_inames_list

        from loopy.symbolic import get_dependencies

        from pytools import argmin2
        from cgen import Block
        from loopy.target.c import ScopeASTNode

        initializers = []

        def is_in_deeper_loop(in_iname_uses):
            for iiu in in_iname_uses:
                iiu = frozenset(iiu)

                if available_variables & iiu < iiu:  # note: not equal!
                    return True

            return False

        while True:
            eligible_subsets = frozenset(
                    term_set
                    for term_set, in_iname_uses in six.iteritems(
                        term_subset_to_inside_inames_list)
                    if all(get_dependencies(term) <= available_variables
                        for term in term_set)
                    if len(in_iname_uses) >= 2  # used more than once
                    or is_in_deeper_loop(in_iname_uses))

            if not eligible_subsets:
                break

            def get_name_sort_key(subset):
                return (sorted(str(term) for term in subset))

            # find the shortest, most-used subexpression
            new_var_subset, _ = argmin2(
                    ((subset,
                        (len(subset),
                            -len(term_subset_to_inside_inames_list[subset]),
                            get_name_sort_key(subset)))
                        for subset in eligible_subsets),
                    return_value=True)

            var_name = subex_rep_state.name_generator("ind")

            old_var_expr = p.Sum(tuple(new_var_subset))
            new_var_expr = p.Sum(tuple(
                simplify_terms(new_var_subset, term_set_to_variable)))

            term_set_to_variable[new_var_subset] = var_name

            initializers.append(
                    codegen_state.ast_builder.emit_initializer(
                        codegen_state,
                        codegen_state.kernel.index_dtype,
                        var_name,
                        CExpression(
                            codegen_state,
                            new_var_expr),
                        is_const=True,
                        short_for_expr=old_var_expr))

            term_subset_to_inside_inames_list = \
                    compute_term_subset_to_inside_inames_list(
                            subex_rep_state.term_set_to_inside_inames_list,
                            term_set_to_variable)

        # insert initializer code
        if initializers:
            subnode = node.child
            if isinstance(subnode, Block):
                subnode = Block(initializers + subnode.contents)
            else:
                subnode = Block(initializers+[subnode])
            node = ScopeASTNode(
                    codegen_state, node.available_variables, subnode)

        subex_rep_state = subex_rep_state.copy(
                term_set_to_variable=term_set_to_variable,
                term_subset_to_inside_inames_list=term_subset_to_inside_inames_list)

        return super(ASTSubexpressionReplacer, self).map_loopy_scope(
                node, subex_rep_state)

    def map_expression(self, expr, subex_rep_state):
        from pymbolic.primitives import is_constant
        if isinstance(expr, CExpression):
            ssr = SubscriptSubsetReplacer(self, subex_rep_state)
            return CExpression(
                    expr.codegen_state,
                    ssr(expr.expr))
        elif isinstance(expr, str) or is_constant(expr):
            return expr
        else:
            raise LoopyError(
                    "Unexpected expression type: %s" % type(expr).__name__)

# }}}


def eliminate_common_subscripts(codegen_state, node):
    if not codegen_state.kernel.options.eliminate_common_subscripts:
        return node

    sc = ASTSubexpressionCollector(codegen_state)
    sc(node)

    sr = ASTSubexpressionReplacer()

    term_set_to_variable = {}
    subex_rep_state = SubexpressionReplacementState(
            codegen_state=codegen_state,
            name_generator=codegen_state.kernel.get_var_name_generator(),
            term_set_to_inside_inames_list=sc.term_set_to_inside_inames_list,
            available_variables=codegen_state.kernel.outer_params(),
            term_set_to_variable=term_set_to_variable,
            term_subset_to_inside_inames_list=(
                compute_term_subset_to_inside_inames_list(
                        sc.term_set_to_inside_inames_list, term_set_to_variable)))

    return sr(node, subex_rep_state)

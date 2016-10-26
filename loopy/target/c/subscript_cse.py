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
                node.available_variables,
                self.rec(node.child, *args, **kwargs))

    def map_loopy_pod(self, node, *args, **kwargs):
        return type(node)(node.ast_builder, node.dtype, node.name)


def generate_all_subsets(l, min_length):
    for bits in range(2**len(l)):
        if bin(bits).count("1") >= min_length:
            yield frozenset(entry for i, entry in enumerate(l) if (1 << i) & bits)


def get_terms(allowable_vars, expr):
    if isinstance(expr, p.Sum):
        terms = expr.children
    else:
        terms = (expr,)

    from loopy.symbolic import get_dependencies
    from pymbolic.primitives import is_constant

    result = []
    remainder = []
    for term in terms:
        if get_dependencies(term) <= allowable_vars and not is_constant(term):
            result.append(term)
        elif remainder is not None:
            remainder.append(term)

    return result, remainder

# }}}


# {{{ counting

class SubscriptSubsetCounter(ExprIdentityMapper):
    def __init__(self, kernel, term_set_to_count):
        self.kernel = kernel
        self.term_set_to_count = term_set_to_count
        self.allowable_vars = self.kernel.all_inames() | self.kernel.outer_params()

    def map_subscript(self, expr):
        terms, _ = get_terms(self.allowable_vars, expr.index)
        terms = frozenset(terms)
        self.term_set_to_count[terms] = self.term_set_to_count.get(terms, 0) + 1


class ASTSubexpressionCollector(CASTIdentityMapper):
    def __init__(self, kernel):
        self.term_set_to_count = {}
        self.subset_count_mapper = SubscriptSubsetCounter(
                kernel, self.term_set_to_count)

    def map_expression(self, expr):
        from pymbolic.primitives import is_constant
        if isinstance(expr, CExpression):
            self.subset_count_mapper(expr.expr)
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

    .. attribute:: term_set_to_count

        A mapping from (summed) sets of subexpressions to their use counts.

    .. attribute:: term_subset_to_count

        A mapping from (summed) subsets of subexpressions to their use counts.

    .. attribute:: available_variables

        A set of variables that subexpressions may refer to.

    .. attribute:: term_set_to_variable

        A mapping from term subsets to their replacement variable names.
    """


def compute_term_subset_to_count(term_set_to_count, term_set_to_variable):
    logger.debug("TERM SET TO SUBSET COUNT:")
    for term_set, count in six.iteritems(term_set_to_count):
        logger.debug(" + ".join(str(i) for i in term_set), count)

    result = {}
    for code_term_set, cnt in six.iteritems(term_set_to_count):
        logger.debug("CTS:", " + ".join(str(i) for i in code_term_set))
        interacts_with_var_term_sets = [
                var_term_set
                for var_term_set in six.iterkeys(term_set_to_variable)
                if var_term_set <= code_term_set]

        logger.debug("INTERACTS:", interacts_with_var_term_sets)
        for subset in generate_all_subsets(code_term_set, 2):
            will_contribute = True

            for var_term_set in interacts_with_var_term_sets:
                if (subset <= var_term_set
                        or (var_term_set & subset
                            and not var_term_set < subset)):
                    will_contribute = False
                    break

            if will_contribute:
                result[subset] = result.get(subset, 0) + cnt

        logger.debug("CTS DONE")

    logger.debug("TERM SUBSET TO COUNT:")
    for term_set, count in six.iteritems(result):
        logger.debug(" + ".join(str(i) for i in term_set), count)

    return result


def simplify_terms(terms, term_set_to_variable):
    logger.debug("BUILDING EXPR FOR:", "+".join(str(s) for s in terms))
    did_something = True
    while did_something:
        did_something = False

        for subset, var_name in sorted(
                six.iteritems(term_set_to_variable),
                # longest first
                key=lambda entry: len(entry[0]), reverse=True):
            if subset <= terms:
                logger.debug("SIMPLIFYING", "+".join(str(s) for s in subset),
                        "->", var_name)
                terms = (
                        (terms - subset)
                        | frozenset([p.Variable(var_name)]))
                did_something = True
                break

    logger.debug("GOT", "+".join(str(s) for s in terms))
    return terms


class SubscriptSubsetReplacer(ExprIdentityMapper):
    def __init__(self, node_replacer, subex_rep_state):
        self.node_replacer = node_replacer
        self.subex_rep_state = subex_rep_state

    def map_subscript(self, expr):
        subex_rep_state = self.subex_rep_state

        iname_terms, remainder = get_terms(
            subex_rep_state.codegen_state.kernel.all_inames(),
            expr.index)
        iname_terms = simplify_terms(
                frozenset(iname_terms),
                subex_rep_state.term_set_to_variable)

        expr = type(expr)(
                expr.aggregate,
                p.Sum(tuple(iname_terms) + tuple(remainder)))

        return super(SubscriptSubsetReplacer, self).map_subscript(expr)


class ASTSubexpressionReplacer(CASTIdentityMapper):
    def map_loopy_scope(self, node, subex_rep_state):
        codegen_state = subex_rep_state.codegen_state

        available_variables = (
                subex_rep_state.available_variables
                | frozenset(node.available_variables))
        subex_rep_state = subex_rep_state.copy(
                available_variables=available_variables)

        term_set_to_variable = subex_rep_state.term_set_to_variable.copy()
        term_subset_to_count = subex_rep_state.term_subset_to_count

        from loopy.symbolic import get_dependencies
        from pytools import argmin2
        from cgen import Block
        from loopy.target.c import ScopeASTNode

        initializers = []

        while True:
            eligible_subsets = frozenset(
                    term_set
                    for term_set, count in six.iteritems(term_subset_to_count)
                    if all(get_dependencies(term) <= available_variables
                        for term in term_set)
                    if count >= 2)

            if not eligible_subsets:
                break

            # find the shortest, most-used subexpression
            new_var_subset, _ = argmin2(
                    ((subset,
                        (len(subset), -term_subset_to_count[subset]))
                        for subset in eligible_subsets),
                    return_value=True)

            var_name = subex_rep_state.name_generator("index_subexp")

            new_var_expr = p.Sum(tuple(
                simplify_terms(new_var_subset, term_set_to_variable)))

            term_set_to_variable[new_var_subset] = var_name

            initializers.append(
                    codegen_state.ast_builder.emit_initializer(
                        codegen_state,
                        codegen_state.kernel.index_dtype,
                        var_name,
                        CExpression(
                            subex_rep_state.codegen_state.ast_builder
                            .get_c_expression_to_code_mapper(),
                            new_var_expr),
                        is_const=True))

            term_subset_to_count = compute_term_subset_to_count(
                    subex_rep_state.term_set_to_count,
                    term_set_to_variable)

        # insert initializer code
        if initializers:
            subnode = node.child
            if isinstance(subnode, Block):
                subnode = Block(initializers + subnode.contents)
            else:
                subnode = Block(initializers+[subnode])
            node = ScopeASTNode(node.available_variables, subnode)

        subex_rep_state = subex_rep_state.copy(
                term_set_to_variable=term_set_to_variable,
                term_subset_to_count=term_subset_to_count)

        return super(ASTSubexpressionReplacer, self).map_loopy_scope(
                node, subex_rep_state)

    def map_expression(self, expr, subex_rep_state):
        from pymbolic.primitives import is_constant
        if isinstance(expr, CExpression):
            ssr = SubscriptSubsetReplacer(self, subex_rep_state)
            return CExpression(
                    expr.to_code_mapper,
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

    sc = ASTSubexpressionCollector(codegen_state.kernel)
    sc(node)

    sr = ASTSubexpressionReplacer()

    term_set_to_variable = {}
    subex_rep_state = SubexpressionReplacementState(
            codegen_state=codegen_state,
            name_generator=codegen_state.kernel.get_var_name_generator(),
            term_set_to_count=sc.term_set_to_count,
            available_variables=codegen_state.kernel.outer_params(),
            term_set_to_variable=term_set_to_variable,
            term_subset_to_count=compute_term_subset_to_count(
                sc.term_set_to_count, term_set_to_variable))

    return sr(node, subex_rep_state)

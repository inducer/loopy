__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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

from pymbolic.parser import Parser as ExpressionParserBase
from loopy.frontend.fortran.diagnostic import TranslationError

from sys import intern
import numpy as np

import pytools.lex
import re


_less_than = intern("less_than")
_greater_than = intern("greater_than")
_less_equal = intern("less_equal")
_greater_equal = intern("greater_equal")
_equal = intern("equal")
_not_equal = intern("not_equal")

_not = intern("not")
_and = intern("and")
_or = intern("or")


def tuple_to_complex_literal(expr):
    if len(expr) != 2:
        raise TranslationError("complex literals must have "
                "two entries")

    r, i = expr

    r = np.array(r)[()]
    i = np.array(i)[()]

    dtype = (r.dtype.type(0) + i.dtype.type(0))
    if dtype == np.float32:
        dtype = np.complex64
    else:
        dtype = np.complex128

    return dtype(float(r) + float(i)*1j)


# {{{ expression parser

class FortranExpressionParser(ExpressionParserBase):
    lex_table = [
        (_less_than, pytools.lex.RE(r"\.lt\.", re.I)),
        (_greater_than, pytools.lex.RE(r"\.gt\.", re.I)),
        (_less_equal, pytools.lex.RE(r"\.le\.", re.I)),
        (_greater_equal, pytools.lex.RE(r"\.ge\.", re.I)),
        (_equal, pytools.lex.RE(r"\.eq\.", re.I)),
        (_not_equal, pytools.lex.RE(r"\.ne\.", re.I)),

        (_not, pytools.lex.RE(r"\.not\.", re.I)),
        (_and, pytools.lex.RE(r"\.and\.", re.I)),
        (_or, pytools.lex.RE(r"\.or\.", re.I)),
        ] + ExpressionParserBase.lex_table

    def __init__(self, tree_walker):
        self.tree_walker = tree_walker

    _PREC_FUNC_ARGS = 1

    def parse_terminal(self, pstate):
        scope = self.tree_walker.scope_stack[-1]

        from pymbolic.primitives import Subscript, Call, Variable
        from pymbolic.parser import (
            _identifier, _openpar, _closepar, _float)

        next_tag = pstate.next_tag()
        if next_tag is _float:
            value = pstate.next_str_and_advance().lower()
            if "d" in value:
                dtype = np.float64
            else:
                dtype = np.float32

            value = value.replace("d", "e")
            if value.startswith("."):
                value = "0"+value

            elif value.startswith("-."):
                value = "-0"+value[1:]

            return dtype(float(value))

        elif next_tag is _identifier:
            name = pstate.next_str_and_advance()

            if pstate.is_at_end() or pstate.next_tag() is not _openpar:
                # not a subscript
                scope.use_name(name)

                return Variable(name)

            left_exp = Variable(name)

            pstate.advance()
            pstate.expect_not_end()

            if scope.is_known(name):
                cls = Subscript
            else:
                cls = Call

            if pstate.next_tag is _closepar:
                pstate.advance()
                left_exp = cls(left_exp, ())
            else:
                args = self.parse_expression(pstate, self._PREC_FUNC_ARGS)
                if not isinstance(args, tuple):
                    args = (args,)
                left_exp = cls(left_exp, args)
                pstate.expect(_closepar)
                pstate.advance()

            return left_exp
        else:
            return ExpressionParserBase.parse_terminal(
                    self, pstate)

    COMP_MAP = {
            _less_than: "<",
            _less_equal: "<=",
            _greater_than: ">",
            _greater_equal: ">=",
            _equal: "==",
            _not_equal: "!=",
            }

    def parse_prefix(self, pstate, min_precedence=0):
        from pymbolic.parser import _PREC_UNARY
        import pymbolic.primitives as primitives

        pstate.expect_not_end()

        if pstate.is_next(_not):
            pstate.advance()
            return primitives.LogicalNot(
                    self.parse_expression(pstate, _PREC_UNARY))
        else:
            return ExpressionParserBase.parse_prefix(self, pstate)

    def parse_postfix(self, pstate, min_precedence, left_exp):
        from pymbolic.parser import (
                _PREC_CALL, _PREC_COMPARISON, _openpar,
                _PREC_LOGICAL_OR, _PREC_LOGICAL_AND)
        from pymbolic.primitives import (
                Comparison, LogicalAnd, LogicalOr)

        next_tag = pstate.next_tag()
        if next_tag is _openpar and _PREC_CALL > min_precedence:
            raise TranslationError("parenthesis operator only works on names")

        elif next_tag in self.COMP_MAP and _PREC_COMPARISON > min_precedence:
            pstate.advance()
            left_exp = Comparison(
                    left_exp,
                    self.COMP_MAP[next_tag],
                    self.parse_expression(pstate, _PREC_COMPARISON))
            did_something = True
        elif next_tag is _and and _PREC_LOGICAL_AND > min_precedence:
            pstate.advance()
            left_exp = LogicalAnd((left_exp,
                    self.parse_expression(pstate, _PREC_LOGICAL_AND)))
            did_something = True
        elif next_tag is _or and _PREC_LOGICAL_OR > min_precedence:
            pstate.advance()
            left_exp = LogicalOr((left_exp,
                    self.parse_expression(pstate, _PREC_LOGICAL_OR)))
            did_something = True
        else:
            left_exp, did_something = ExpressionParserBase.parse_postfix(
                    self, pstate, min_precedence, left_exp)

        return left_exp, did_something

    def parse_expression(self, pstate, min_precedence=0):
        left_exp = self.parse_prefix(pstate)

        did_something = True
        while did_something:
            did_something = False
            if pstate.is_at_end():
                return left_exp

            result = self.parse_postfix(
                    pstate, min_precedence, left_exp)
            left_exp, did_something = result

        from pymbolic.parser import FinalizedTuple
        if isinstance(left_exp, FinalizedTuple):
            # View all tuples that survive parsing as complex literals
            # "FinalizedTuple" indicates that this tuple was enclosed
            # in parens.
            return tuple_to_complex_literal(left_exp)

        return left_exp

# }}}


# vim: foldmethod=marker

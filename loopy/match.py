"""Matching functionality for instruction ids and subsitution
rule invocations stacks."""


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

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import FrozenSet, List, Sequence, Tuple, Union, Protocol
from sys import intern

from loopy.kernel import LoopKernel
from loopy.kernel.instruction import InstructionBase


NoneType = type(None)

from pytools.lex import RE
import pytools.tag

__doc__ = """
.. autoclass:: Matchable
.. autoclass:: StackMatchComponent
.. autoclass:: StackMatch

.. autofunction:: parse_match

.. autofunction:: parse_stack_match

Match expressions
^^^^^^^^^^^^^^^^^

.. autoclass:: MatchExpressionBase
.. autoclass:: All
.. autoclass:: And
.. autoclass:: Or
.. autoclass:: Not
.. autoclass:: Id
.. autoclass:: ObjTagged
.. autoclass:: Tagged
.. autoclass:: Writes
.. autoclass:: Reads
.. autoclass:: InKernel
.. autoclass:: Iname
"""


def re_from_glob(s):
    import re
    from fnmatch import translate
    return re.compile("^"+translate(s.strip())+"$")


# {{{ parsing

# {{{ lexer data

_and = intern("and")
_or = intern("or")
_not = intern("not")
_openpar = intern("openpar")
_closepar = intern("closepar")

_id = intern("_id")
_tag = intern("_tag")
_writes = intern("_writes")
_reads = intern("_reads")
_in_kernel = intern("_in_kernel")
_iname = intern("_iname")

_whitespace = intern("_whitespace")

# }}}


_LEX_TABLE = [
    (_and, RE(r"and\b")),
    (_or, RE(r"or\b")),
    (_not, RE(r"not\b")),
    (_openpar, RE(r"\(")),
    (_closepar, RE(r"\)")),

    # TERMINALS
    (_id, RE(r"id:([\w?*]+)")),
    (_tag, RE(r"tag:([\w?*]+)")),
    (_writes, RE(r"writes:([\w?*]+)")),
    (_reads, RE(r"reads:([\w?*]+)")),
    (_in_kernel, RE(r"in_kernel:([\w?*]+)")),
    (_iname, RE(r"iname:([\w?*]+)")),

    (_whitespace, RE("[ \t]+")),
    ]


_TERMINALS = ([_id, _tag, _writes, _reads, _in_kernel, _iname])

# {{{ operator precedence

_PREC_OR = 10
_PREC_AND = 20
_PREC_NOT = 30

# }}}

# }}}


# {{{ match expression

class Matchable(Protocol):
    """
    .. attribute:: tags
    """
    @property
    def tags(self) -> FrozenSet[pytools.tag.Tag]:
        ...


class MatchExpressionBase(ABC):
    @abstractmethod
    def __call__(self, kernel: LoopKernel, matchable: Matchable) -> bool:
        raise NotImplementedError

    def __ne__(self, other):
        return not self.__eq__(other)

    def __and__(self, other):
        return And((self, other))

    def __or__(self, other):
        return Or((self, other))

    def __inv__(self):
        return Not(self)


class All(MatchExpressionBase):
    def __call__(self, kernel: LoopKernel, matchable: Matchable) -> bool:
        return True

    def __str__(self):
        return "all"

    def __repr__(self):
        return "%s()" % (type(self).__name__)

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.rec(key_hash, "all_match_expr")

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


@dataclass(frozen=True, eq=True)
class MultiChildMatchExpressionBase(MatchExpressionBase):
    children: Sequence[MatchExpressionBase]

    def __str__(self):
        joiner = " %s " % type(self).__name__.lower()
        return "(%s)" % (joiner.join(str(ch) for ch in self.children))

    def __repr__(self):
        return "{}({})".format(
                type(self).__name__,
                ", ".join(repr(ch) for ch in self.children))


class And(MultiChildMatchExpressionBase):
    def __call__(self, kernel: LoopKernel, matchable: Matchable) -> bool:
        return all(ch(kernel, matchable) for ch in self.children)


class Or(MultiChildMatchExpressionBase):
    def __call__(self, kernel: LoopKernel, matchable: Matchable) -> bool:
        return any(ch(kernel, matchable) for ch in self.children)


@dataclass(frozen=True, eq=True)
class Not(MatchExpressionBase):
    child: MatchExpressionBase

    def __call__(self, kernel: LoopKernel, matchable: Matchable) -> bool:
        return not self.child(kernel, matchable)

    def __str__(self):
        return "(not %s)" % str(self.child)

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.child)


@dataclass(frozen=True, eq=True)
class ObjTagged(MatchExpressionBase):
    """Match if the object is tagged with a given :class:`~pytools.tag.Tag`.

    .. note::

        These instance-based tags will, in the not-too-distant future, replace
        the string-based tags matched by :class:`Tagged`.
    """
    tag: pytools.tag.Tag

    def __call__(self, kernel: LoopKernel, matchable: Matchable) -> bool:
        return self.tag in matchable.tags


class GlobMatchExpressionBase(MatchExpressionBase):
    def __init__(self, glob: str) -> None:
        self.glob = glob

        import re
        from fnmatch import translate
        self.re = re.compile("^"+translate(glob.strip())+"$")

    def __str__(self):
        descr = type(self).__name__
        return descr.lower() + ":" + self.glob

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self. glob)

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.rec(key_hash, type(self).__name__)
        key_builder.rec(key_hash, self.glob)

    def __eq__(self, other):
        return (type(self) is type(other)
                and self.glob == other.glob)

    def __hash__(self):
        return hash((type(self), self.glob))


class Id(GlobMatchExpressionBase):
    def __call__(self, kernel, matchable):
        return self.re.match(matchable.id)


class Tagged(GlobMatchExpressionBase):
    """Match a string-based tagged using a glob expression.

    .. note::

        These string-based tags will, in the not-too-distant future, be replace
        by instance-based tags matched by :class:`ObjTagged`.
    """

    def __call__(self, kernel: LoopKernel, matchable: Matchable) -> bool:
        from loopy.kernel.instruction import LegacyStringInstructionTag
        if matchable.tags:
            return any(
                    self.re.match(tag.value)
                    if isinstance(tag, LegacyStringInstructionTag)
                    else

                    False

                    for tag in matchable.tags)
        else:
            return False


class Writes(GlobMatchExpressionBase):
    def __call__(self, kernel: LoopKernel, matchable: Matchable) -> bool:
        if not isinstance(matchable, InstructionBase):
            return False
        return any(self.re.match(name)
                for name in matchable.assignee_var_names())


class Reads(GlobMatchExpressionBase):
    def __call__(self, kernel: LoopKernel, matchable: Matchable) -> bool:
        if not isinstance(matchable, InstructionBase):
            return False
        return any(self.re.match(name)
                for name in matchable.read_dependency_names())


class InKernel(GlobMatchExpressionBase):
    def __call__(self, kernel, matchable):
        return self.re.match(kernel.name)


class Iname(GlobMatchExpressionBase):
    def __call__(self, kernel: LoopKernel, matchable: Matchable) -> bool:
        if not isinstance(matchable, InstructionBase):
            return False

        return any(self.re.match(name)
                for name in matchable.within_inames)

# }}}


# {{{ parser

def parse_match(expr):
    """Syntax examples::

    * ``id:yoink and writes:a_temp``
    * ``id:yoink and (not writes:a_temp or tag:input)``
    """
    if not expr:
        return All()

    def parse_terminal(pstate):
        next_tag = pstate.next_tag()
        if next_tag is _id:
            result = Id(pstate.next_match_obj().group(1))
            pstate.advance()
            return result
        elif next_tag is _tag:
            result = Tagged(pstate.next_match_obj().group(1))
            pstate.advance()
            return result
        elif next_tag is _writes:
            result = Writes(pstate.next_match_obj().group(1))
            pstate.advance()
            return result
        elif next_tag is _reads:
            result = Reads(pstate.next_match_obj().group(1))
            pstate.advance()
            return result
        elif next_tag is _in_kernel:
            result = InKernel(pstate.next_match_obj().group(1))
            pstate.advance()
            return result
        elif next_tag is _iname:
            result = Iname(pstate.next_match_obj().group(1))
            pstate.advance()
            return result
        else:
            pstate.expected("terminal")

    def inner_parse(pstate, min_precedence=0):
        pstate.expect_not_end()

        if pstate.is_next(_not):
            pstate.advance()
            left_query = Not(inner_parse(pstate, _PREC_NOT))
        elif pstate.is_next(_openpar):
            pstate.advance()
            left_query = inner_parse(pstate)
            pstate.expect(_closepar)
            pstate.advance()
        else:
            left_query = parse_terminal(pstate)

        did_something = True
        while did_something:
            did_something = False
            if pstate.is_at_end():
                return left_query

            next_tag = pstate.next_tag()

            if next_tag is _and and _PREC_AND > min_precedence:
                pstate.advance()
                left_query = And(
                        (left_query, inner_parse(pstate, _PREC_AND)))
                did_something = True
            elif next_tag is _or and _PREC_OR > min_precedence:
                pstate.advance()
                left_query = Or(
                        (left_query, inner_parse(pstate, _PREC_OR)))
                did_something = True

        return left_query

    if isinstance(expr, MatchExpressionBase):
        return expr

    from pytools.lex import LexIterator, lex, InvalidTokenError
    try:
        pstate = LexIterator(
            [(tag, s, idx, matchobj)
             for (tag, s, idx, matchobj) in lex(_LEX_TABLE, expr,
                 match_objects=True)
             if tag is not _whitespace], expr)
    except InvalidTokenError as e:
        from loopy.diagnostic import LoopyError
        raise LoopyError(
                "invalid match expression: '{match_expr}' ({err_type}: {err_str})"
                .format(
                    match_expr=expr,
                    err_type=type(e).__name__,
                    err_str=str(e)))

    if pstate.is_at_end():
        pstate.raise_parse_error("unexpected end of input")

    result = inner_parse(pstate)
    if not pstate.is_at_end():
        pstate.raise_parse_error("leftover input after completed parse")

    return result

# }}}


# {{{ stack match objects

class StackMatchComponent(ABC):
    """
    .. automethod:: __call__
    """

    @abstractmethod
    def __call__(self, kernel: LoopKernel, stack: Sequence[Matchable]) -> bool:
        pass

    def __ne__(self, other):
        return not self.__eq__(other)


class StackAllMatchComponent(StackMatchComponent):
    def __call__(self, kernel: LoopKernel, stack: Sequence[Matchable]) -> bool:
        return True

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.rec(key_hash, "all_match")

    def __eq__(self, other):
        return type(self) is type(other)


class StackBottomMatchComponent(StackMatchComponent):
    def __call__(self, kernel: LoopKernel, stack: Sequence[Matchable]) -> bool:
        return not stack

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.rec(key_hash, "bottom_match")

    def __eq__(self, other):
        return type(self) is type(other)


@dataclass(eq=True, frozen=True)
class StackItemMatchComponent(StackMatchComponent):
    match_expr: MatchExpressionBase
    inner_match: StackMatchComponent

    def __call__(self, kernel: LoopKernel, stack: Sequence[Matchable]) -> bool:
        if not stack:
            return False

        outer = stack[0]
        if not self.match_expr(kernel, outer):
            return False

        return self.inner_match(kernel, stack[1:])


@dataclass(eq=True, frozen=True)
class StackWildcardMatchComponent(StackMatchComponent):
    inner_match: StackMatchComponent

    def __call__(self, kernel: LoopKernel, stack: Sequence[Matchable]) -> bool:
        for i in range(0, len(stack)):
            if self.inner_match(kernel, stack[i:]):
                return True

        return False

# }}}


# {{{ stack matcher

@dataclass(eq=True, frozen=True)
class RuleInvocationMatchable:
    id: str
    tags: FrozenSet[pytools.tag.Tag]

    def write_dependency_names(self):
        raise TypeError("writes: query may not be applied to rule invocations")

    def read_dependency_names(self):
        raise TypeError("reads: query may not be applied to rule invocations")

    def inames(self, kernel):
        raise TypeError("inames: query may not be applied to rule invocations")


@dataclass(eq=True, frozen=True)
class StackMatch:
    """
    .. automethod:: __call__
    """

    root_component: StackMatchComponent

    def __call__(
            self, kernel: LoopKernel, insn: InstructionBase,
            rule_stack: Sequence[Tuple[str, FrozenSet[pytools.tag.Tag]]]) -> bool:
        """
        :arg rule_stack: a tuple of (name, tags) rule invocation, outermost first
        """
        stack_of_matchables: List[Matchable] = [insn]
        for id, tags in rule_stack:
            stack_of_matchables.append(RuleInvocationMatchable(id, tags))

        return self.root_component(kernel, stack_of_matchables)

# }}}


# {{{ stack match parsing

ToStackMatchCovertible = Union[StackMatch, str, None]


def parse_stack_match(smatch: ToStackMatchCovertible) -> StackMatch:
    """Syntax example::

        ... > outer > ... > next > innermost $
        insn > next
        insn > ... > next > innermost $

    ``...`` matches an arbitrary number of intervening stack levels.

    Each of the entries is a match expression as understood by
    :func:`parse_match`.
    """

    if isinstance(smatch, StackMatch):
        return smatch
    if isinstance(smatch, MatchExpressionBase):
        return StackMatch(
                StackItemMatchComponent(
                    smatch, StackAllMatchComponent()))

    if smatch is None:
        return StackMatch(StackAllMatchComponent())

    smatch = smatch.strip()

    match: StackMatchComponent = StackAllMatchComponent()
    if smatch[-1] == "$":
        match = StackBottomMatchComponent()
        smatch = smatch[:-1]

    smatch = smatch.strip()

    components = smatch.split(">")

    for comp in components[::-1]:
        comp = comp.strip()
        if comp == "...":
            match = StackWildcardMatchComponent(match)
        else:
            match = StackItemMatchComponent(parse_match(comp), match)

    return StackMatch(match)

# }}}


# vim: foldmethod=marker

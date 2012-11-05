"""Matching functionality for instruction ids and subsitution
rule invocations stacks."""

from __future__ import division

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




# {{{ id match objects

class AllMatch(object):
    def __call__(self, identifier, tag):
        return True

class RegexIdentifierMatch(object):
    def __init__(self, id_re, tag_re=None):
        self.id_re = id_re
        self.tag_re = tag_re

    def __call__(self, identifier, tag):
        if self.tag_re is None:
            return self.id_re.match(identifier) is not None
        else:
            if tag is None:
                tag = ""

            return (
                    self.id_re.match(identifier) is not None
                    and self.tag_re.match(tag) is not None)

class AlternativeMatch(object):
    def __init__(self, matches):
        self.matches = matches

    def __call__(self, identifier, tag):
        from pytools import any
        return any(
                mtch(identifier, tag) for mtch in self.matches)

# }}}

# {{{ single id match parsing

def parse_id_match(id_matches):
    """Syntax examples:

    my_insn
    compute_*
    fetch*$first
    fetch*$first,store*$first

    Alternatively, a list of *(name_glob, tag_glob)* tuples.
    """

    if id_matches is None:
        return AllMatch()

    if isinstance(id_matches, str):
        id_matches = id_matches.split(",")

    if len(id_matches) > 1:
        return AlternativeMatch(parse_id_match(im) for im in id_matches)

    if len(id_matches) == 0:
        return AllMatch()

    id_match, = id_matches
    del id_matches

    def re_from_glob(s):
        import re
        from fnmatch import translate
        return re.compile(translate(s.strip()))

    if not isinstance(id_match, tuple):
        components = id_match.split("$")

    if len(components) == 1:
        return RegexIdentifierMatch(re_from_glob(components[0]))
    elif len(components) == 2:
        return RegexIdentifierMatch(
                re_from_glob(components[0]),
                re_from_glob(components[1]))
    else:
        raise RuntimeError("too many (%d) $-separated components in id match"
                % len(components))

# }}}

# {{{ stack match objects

# these match from the tail of the stack

class StackMatchBase(object):
    pass

class AllStackMatch(StackMatchBase):
    def __call__(self, stack):
        return True

class StackIdMatch(StackMatchBase):
    def __init__(self, id_match, up_match):
        self.id_match = id_match
        self.up_match = up_match

    def __call__(self, stack):
        if not stack:
            return False

        last = stack[-1]
        if not self.id_match(*last):
            return False

        if self.up_match is None:
            return True
        else:
            return self.up_match(stack[:-1])

class StackWildcardMatch(StackMatchBase):
    def __init__(self, up_match):
        self.up_match = up_match

    def __call__(self, stack):
        if self.up_match is None:
            return True

        n = len(stack)

        for i in xrange(n):
            if self.up_match(stack[:-i]):
                return True

        return False

# }}}

# {{{ stack match parsing

def parse_stack_match(smatch):
    """Syntax example::

        lowest < next < ... < highest

    where `lowest` is necessarily the bottom of the stack.  `...` matches an
    arbitrary number of intervening stack levels. There is currently no way to
    match the top of the stack.

    Each of the entries is an identifier match as understood by :func:`parse_id_match`.
    """

    if isinstance(smatch, StackMatchBase):
        return smatch

    match = AllStackMatch()

    if smatch is None:
        return match

    components = smatch.split("<")

    for comp in components[::-1]:
        comp = comp.strip()
        if comp == "...":
            match = StackWildcardMatch(match)
        else:
            match = StackIdMatch(parse_id_match(comp), match)

    return match

# }}}



# vim: foldmethod=marker

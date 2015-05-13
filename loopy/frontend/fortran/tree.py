from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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

import re

from loopy.diagnostic import LoopyError


class FTreeWalkerBase(object):
    def __init__(self):
        from loopy.frontend.fortran.expression import FortranExpressionParser
        self.expr_parser = FortranExpressionParser(self)

    def rec(self, expr, *args, **kwargs):
        mro = list(type(expr).__mro__)
        dispatch_class = kwargs.pop("dispatch_class", type(self))

        while mro:
            method_name = "map_"+mro.pop(0).__name__

            try:
                method = getattr(dispatch_class, method_name)
            except AttributeError:
                pass
            else:
                return method(self, expr, *args, **kwargs)

        raise NotImplementedError(
                "%s does not know how to map type '%s'"
                % (type(self).__name__,
                    type(expr)))

    ENTITY_RE = re.compile(
            r"^(?P<name>[_0-9a-zA-Z]+)"
            "(\((?P<shape>[-+*0-9:a-zA-Z, \t]+)\))?$")

    def parse_dimension_specs(self, node, dim_decls):
        def parse_bounds(bounds_str):
            start_end = bounds_str.split(":")

            assert 1 <= len(start_end) <= 2

            return [self.parse_expr(node, s) for s in start_end]

        for decl in dim_decls:
            entity_match = self.ENTITY_RE.match(decl)
            assert entity_match

            groups = entity_match.groupdict()
            name = groups["name"]
            assert name

            if groups["shape"]:
                shape = [parse_bounds(s) for s in groups["shape"].split(",")]
            else:
                shape = None

            yield name, shape

    def __call__(self, expr, *args, **kwargs):
        return self.rec(expr, *args, **kwargs)

    # {{{ expressions

    def parse_expr(self, node, expr_str, **kwargs):
        try:
            return self.expr_parser(expr_str, **kwargs)
        except Exception as e:
            raise LoopyError(
                    "Error parsing expression '%s' on line %d of '%s': %s"
                    % (expr_str, node.item.span[0], self.filename, str(e)))

    # }}}

# vim: foldmethod=marker

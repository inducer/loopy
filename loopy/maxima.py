"""Export to maxima."""

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


from pymbolic.interop.maxima import \
        MaximaStringifyMapper as MaximaStringifyMapperBase


class MaximaStringifyMapper(MaximaStringifyMapperBase):
    def map_subscript(self, expr, enclosing_prec):
        res = self.rec(expr.aggregate, enclosing_prec)
        idx = expr.index
        if not isinstance(idx, tuple):
            idx = (idx,)
        for i in idx:
            if isinstance(i, int):
                res += "_%d" % i

        return res


def get_loopy_statements_as_maxima(kernel, prefix):
    """Sample use for code comparison::

        load("knl-optFalse.mac");
        load("knl-optTrue.mac");

        vname: bessel_j_8;

        un_name : concat(''un_, vname);
        opt_name : concat(''opt_, vname);

        print(ratsimp(ev(un_name - opt_name)));
    """
    from loopy.preprocess import add_boostability_and_automatic_dependencies
    kernel = add_boostability_and_automatic_dependencies(kernel)

    my_variable_names = (
            avn
            for stmt in kernel.statements
            for avn in stmt.assignee_var_names()
            )

    from pymbolic import var
    subst_dict = dict(
            (vn, var(prefix+vn)) for vn in my_variable_names)

    mstr = MaximaStringifyMapper()
    from loopy.symbolic import SubstitutionMapper
    from pymbolic.mapper.substitutor import make_subst_func
    substitute = SubstitutionMapper(make_subst_func(subst_dict))

    result = ["ratprint:false;"]

    written_stmt_ids = set()

    from loopy.kernel import StatementBase, Assignment

    def write_stmt(stmt):
        if not isinstance(stmt, StatementBase):
            stmt = kernel.id_to_stmt[stmt]
        if not isinstance(stmt, Assignment):
            raise RuntimeError("non-single-output assignment not supported "
                    "in maxima export")

        for dep in stmt.depends_on:
            if dep not in written_stmt_ids:
                write_stmt(dep)

        aname, = stmt.assignee_var_names()
        result.append("%s%s : %s;" % (
            prefix, aname,
            mstr(substitute(stmt.expression))))

        written_stmt_ids.add(stmt.id)

    for stmt in kernel.statements:
        if stmt.id not in written_stmt_ids:
            write_stmt(stmt)

    return "\n".join(result)

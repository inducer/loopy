from __future__ import annotations


__copyright__ = "Copyright (C) 2012-15 Andreas Kloeckner"

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
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np

import pymbolic.primitives as p
from pymbolic import ArithmeticExpression
from pymbolic.mapper import Mapper

from loopy.codegen import UnvectorizableError
from loopy.diagnostic import LoopyError
from loopy.symbolic import simplify_using_aff


if TYPE_CHECKING:
    from loopy.kernel import LoopKernel
    from loopy.symbolic import LinearSubscript, Reduction


TypeContext: TypeAlias = Literal[
    "f",  # single-precision floating point
    "d",  # double-precision floating point
    "i",  # integer
    "b",  # boolean
] | None  # "no known context"


def dtype_to_type_context(target, dtype) -> TypeContext:
    from loopy.types import NumpyType

    if dtype.is_integral():
        return "i"
    if isinstance(dtype, NumpyType) and dtype.dtype in [np.float64, np.complex128]:
        return "d"
    if isinstance(dtype, NumpyType) and dtype.dtype in [np.float32, np.complex64]:
        return "f"
    if isinstance(dtype, NumpyType) and dtype.dtype == np.bool_:
        return "b"
    if target.is_vector_dtype(dtype):
        return dtype_to_type_context(
                target, NumpyType(dtype.numpy_dtype.fields["x"][0]))

    return None


# {{{ vectorizability checker

class VectorizabilityChecker(Mapper[bool, []]):
    """The return value from this mapper is a :class:`bool` indicating whether
    the result of the expression is vectorized along :attr:`vec_iname`.
    If the expression is not vectorizable, the mapper raises
    :class:`UnvectorizableError`.

    .. attribute:: vec_iname
    """

    def __init__(self,
                kernel: LoopKernel,
                vec_iname: str,
                vec_iname_length: int
            ) -> None:
        self.kernel: LoopKernel = kernel
        self.vec_iname: str = vec_iname
        self.vec_iname_length: int = vec_iname_length

    def map_sum(self, expr: p.Sum) -> bool:
        return any(self.rec(child) for child in expr.children)

    def map_product(self, expr: p.Product) -> bool:
        return any(self.rec(child) for child in expr.children)

    def map_quotient(self, expr: p.QuotientBase) -> bool:
        return (self.rec(expr.numerator)
                or
                self.rec(expr.denominator))

    map_remainder = map_quotient

    def map_linear_subscript(self, expr: LinearSubscript) -> bool:
        raise UnvectorizableError("linear subscripts cannot be vectorized")

    def map_call(self, expr: p.Call) -> bool:
        # FIXME: Should implement better vectorization check for function calls

        rec_pars = [
                self.rec(child) for child in expr.parameters]
        if any(rec_pars):
            raise UnvectorizableError("function calls cannot yet be vectorized")

        return False

    def map_subscript(self, expr: p.Subscript) -> bool:
        assert isinstance(expr.aggregate, p.Variable)
        name = expr.aggregate.name

        var = self.kernel.get_var_descriptor(name)

        from loopy.kernel.array import ArrayBase
        if not isinstance(var, ArrayBase):
            raise LoopyError("non-array subscript '%s'" % expr)

        index = expr.index_tuple

        index = tuple(
            simplify_using_aff(self.kernel, cast("ArithmeticExpression", idx_i))
            for idx_i in index)

        from pymbolic.primitives import Variable

        from loopy.kernel.array import VectorArrayDimTag
        from loopy.symbolic import get_dependencies

        possible = None

        assert isinstance(var.shape, tuple)
        assert var.dim_tags is not None

        for i in range(len(var.shape)):
            idx_i = index[i]
            if (
                    isinstance(var.dim_tags[i], VectorArrayDimTag)
                    and isinstance(idx_i, Variable)
                    and idx_i.name == self.vec_iname):
                if var.shape[i] != self.vec_iname_length:
                    raise UnvectorizableError("vector length was mismatched")

                if possible is None:
                    possible = True

            else:
                if self.vec_iname in get_dependencies(idx_i):
                    raise UnvectorizableError("vectorizing iname '%s' occurs in "
                            "unvectorized subscript axis %d (1-based) of "
                            "expression '%s'"
                            % (self.vec_iname, i+1, expr))
                    break

        return bool(possible)

    def map_constant(self, expr: object) -> bool:
        # Loopy does not have vector literals.
        return False

    def map_variable(self, expr: p.Variable) -> bool:
        if expr.name == self.vec_iname:
            # Technically, this is doable. But we're not going there.
            raise UnvectorizableError()

        # A single variable is always a scalar.
        return False

    map_tagged_variable = map_variable

    def map_lookup(self, expr: p.Lookup) -> bool:
        if self.rec(expr.aggregate):
            raise UnvectorizableError()

        return False

    def map_comparison(self, expr: p.Comparison) -> bool:
        return any(self.rec(child) for child in [expr.left, expr.right])

    def map_logical_not(self, expr: object) -> bool:
        raise UnvectorizableError()

    map_logical_and = map_logical_not
    map_logical_or = map_logical_not

    def map_reduction(self, expr: Reduction) -> bool:
        # FIXME: Do this more carefully
        raise UnvectorizableError()

    def map_if(self, expr: p.If) -> bool:
        return any(self.rec(child) for child in [expr.condition, expr.then, expr.else_])

# }}}

# vim: fdm=marker

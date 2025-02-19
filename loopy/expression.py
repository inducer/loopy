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


from typing import (
    Iterable,
)

import numpy as np

from pymbolic import primitives
from pymbolic.mapper import Mapper

from loopy.codegen import UnvectorizableError
from loopy.diagnostic import LoopyError


# type_context may be:
# - "i" for integer -
# - "f" for single-precision floating point
# - "d" for double-precision floating point
# or None for 'no known context'.


def dtype_to_type_context(target, dtype):
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

class VectorizabilityChecker(Mapper[bool | None, []]):
    """The return value from this mapper is a :class:`bool` indicating whether
    the result of the expression is vectorized along :attr:`vec_iname`.
    If the expression is not vectorizable, the mapper raises
    :class:`UnvectorizableError`.

    .. attribute:: vec_iname
    """

    def __init__(self, kernel, vec_iname: str, vec_iname_length: int):
        self.kernel = kernel
        self.vec_iname = vec_iname
        self.vec_iname_length = vec_iname_length

    @staticmethod
    def combine(vectorizabilities: Iterable[bool]):
        from functools import reduce
        from operator import and_
        return reduce(and_, vectorizabilities)

    def map_sum(self, expr: primitives.Sum | primitives.Product) -> bool | None:
        return any(self.rec(child) for child in expr.children)

    map_product = map_sum

    def map_quotient(self, expr: primitives.Quotient) -> bool | None:
        return (self.rec(expr.numerator)
                or
                self.rec(expr.denominator))

    def map_linear_subscript(self, expr: primitives.LinearSubscript) -> bool:
        return False

    def map_call(self, expr) -> bool | None:
        # FIXME: Should implement better vectorization check for function calls

        rec_pars = [
                self.rec(child) for child in expr.parameters]
        if any(rec_pars):
            raise UnvectorizableError("function calls cannot yet be vectorized")

        return False

    def map_subscript(self, expr):
        assert isinstance(expr.aggregate, primitives.Variable)
        name = expr.aggregate.name

        var = self.kernel.arg_dict.get(name)
        if var is None:
            var = self.kernel.temporary_variables.get(name)

        if var is None:
            raise LoopyError("unknown array variable in subscript: %s"
                    % name)

        from loopy.kernel.array import ArrayBase
        if not isinstance(var, ArrayBase):
            raise LoopyError("non-array subscript '%s'" % expr)

        index = expr.index_tuple

        from pymbolic.primitives import Variable

        from loopy.kernel.array import VectorArrayDimTag
        from loopy.symbolic import get_dependencies

        possible = None
        for i in range(len(var.shape)):
            if (isinstance(list(var.dim_tags)[i], VectorArrayDimTag)):
                if isinstance(index[i], Variable):
                    if index[i].name == self.vec_iname:
                        if var.shape[i] != self.vec_iname_length:
                            raise UnvectorizableError("vector length was mismatched")

                        if possible is None:
                            possible = True

            else:
                if self.vec_iname in get_dependencies(index[i]):
                    raise UnvectorizableError("vectorizing iname '%s' occurs in "
                            "unvectorized subscript axis %d (1-based) of "
                            "expression '%s'"
                            % (self.vec_iname, i+1, expr))
                    break

        return bool(possible)

    def map_constant(self, expr: primitives.Constant) -> bool | None:
        # A constant can be vectorized always.
        # One just may not want to vectorize it.
        return True

    def map_variable(self, expr):
        if expr.name == self.vec_iname:
            return True

        name = expr.name
        var = self.kernel.arg_dict.get(name)
        if var is None:
            var = self.kernel.temporary_variables.get(name)

        if var is None:
            raise LoopyError("unknown array variable in subscript: %s"
                    % name)

        from loopy.kernel.array import ArrayBase
        from loopy.kernel.data import ValueArg
        if isinstance(var, ValueArg):
            # Just a simple scalar argument which will get broadcast as necessary.
            return True

        if not isinstance(var, ArrayBase):
            raise LoopyError("non-array variable '%s'" % expr)

        from loopy.kernel.array import VectorArrayDimTag

        possible = None
        for i in range(len(var.shape)):
            if (isinstance(var.dim_tags[i], VectorArrayDimTag)):
                if var.shape[i] != self.vec_iname_length:
                    raise UnvectorizableError("vector length was mismatched")
                elif self.vec_iname_length in [2, 3, 4, 8, 16]:
                    possible = True
                else:
                    raise UnvectorizableError("Vector length not supported.")
        return possible

    map_tagged_variable = map_variable

    def map_lookup(self, expr: primitives.Lookup) -> bool:
        if self.rec(expr.aggregate):
            raise UnvectorizableError()

        return False

    def map_comparison(self, expr: primitives.Comparision) -> bool | None:
        # FIXME: These actually can be vectorized:
        # https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/relationalFunctions.html

        left = self.rec(expr.left)
        right = self.rec(expr.right)
        return all([left, right])

    def map_logical_not(self, expr) -> bool | None:
        raise UnvectorizableError()

    map_logical_and = map_logical_not
    map_logical_or = map_logical_not

    def map_reduction(self, expr: primitives.Reduction) -> bool | None:
        # FIXME: Do this more carefully
        raise UnvectorizableError()

    def map_if(self, expr: primitives.If) -> bool | None:
        condition_vector = self.rec(expr.condition)
        then_case = self.rec(expr.then)
        else_case = self.rec(expr.else_)

        return all([condition_vector, then_case, else_case])
        raise UnvectorizableError()
# }}}

# vim: fdm=marker

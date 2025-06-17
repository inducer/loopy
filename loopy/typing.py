"""
.. autoclass:: Expression
.. autoclass:: ShapeType
.. autodata:: InameStr
.. autodata:: InameStrSet

.. currentmodule:: loopy

.. autoclass:: auto
"""


from __future__ import annotations


__copyright__ = "Copyright (C) 2022 University of Illinois Board of Trustees"

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


from typing import TYPE_CHECKING, TypeAlias, TypeVar, cast

import numpy as np
from typing_extensions import TypeIs

from pymbolic.primitives import ExpressionNode
from pymbolic.typing import ArithmeticExpression, Expression, Integer


if TYPE_CHECKING:
    from collections.abc import Iterable


# The Fortran parser may insert dimensions of 'None', but I'd like to phase
# that out, so we're not encoding that in the type.
ShapeType: TypeAlias = tuple[ArithmeticExpression, ...]
StridesType: TypeAlias = ShapeType

InameStr: TypeAlias = str
InameStrSet: TypeAlias = frozenset[InameStr]

InsnId: TypeAlias = str


class auto:  # noqa
    """A generic placeholder object for something that should be automatically
    determined.  See, for example, the *shape* or *strides* argument of
    :class:`~loopy.ArrayArg`.
    """


T = TypeVar("T")


def not_none(obj: T | None) -> T:
    assert obj is not None
    return obj


def is_integer(obj: object) -> TypeIs[int | np.integer]:
    return isinstance(obj, (int, np.integer))


def integer_or_err(expr: Expression) -> Integer:
    if isinstance(expr, (int, np.integer)):
        return expr
    else:
        raise ValueError(f"expected integer, got {type(expr)}")


def integer_expr_or_err(expr: Expression) -> Integer | ExpressionNode:
    if isinstance(expr, (int, np.integer, ExpressionNode)):
        return expr
    else:
        raise ValueError(f"expected integer or expression, got {type(expr)}")


ElT = TypeVar("ElT")


def assert_tuple(obj: tuple[ElT, ...] | object) -> tuple[ElT, ...]:
    assert isinstance(obj, tuple)
    return obj


def set_union(iterable: Iterable[Iterable[T]]):
    return cast("set[T]", set()).union(*iterable)


def fset_union(iterable: Iterable[Iterable[T]]):
    return cast("frozenset[T]", frozenset()).union(*iterable)


def set_intersection(iterable: Iterable[Iterable[T]]):
    return cast("set[T]", set()).intersection(*iterable)


def fset_intersection(iterable: Iterable[Iterable[T]]):
    return cast("frozenset[T]", frozenset()).intersection(*iterable)

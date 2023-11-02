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


from typing import Union, Tuple, TypeVar, Optional

import numpy as np

from pymbolic.primitives import Expression

IntegralT = Union[int, np.int8, np.int16, np.int32, np.int64, np.uint8,
                  np.uint16, np.uint32, np.uint64]
FloatT = Union[float, complex, np.float32, np.float64, np.complex64,
        np.complex128]


ExpressionT = Union[IntegralT, FloatT, Expression]
ShapeType = Tuple[ExpressionT, ...]
StridesType = ShapeType


class auto:  # noqa
    """A generic placeholder object for something that should be automatically
    determined.  See, for example, the *shape* or *strides* argument of
    :class:`ArrayArg`.
    """


T = TypeVar("T")


def not_none(obj: Optional[T]) -> T:
    assert obj is not None
    return obj

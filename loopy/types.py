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

from warnings import warn
import numpy as np

from loopy.diagnostic import LoopyError

__doc__ = """
.. currentmodule:: loopy.types

.. autoclass:: LoopyType

.. autoclass:: NumpyType

.. autoclass:: AtomicType

.. autoclass:: AtomicNumpyType
"""


class LoopyType:
    """
    Abstract class for dtypes of variables encountered in a
    :class:`loopy.LoopKernel`.
    """
    def is_integral(self):
        raise NotImplementedError()

    def is_complex(self):
        raise NotImplementedError()

    def uses_complex(self):
        raise NotImplementedError()

    def is_composite(self):
        raise NotImplementedError()

    @property
    def itemsize(self):
        raise NotImplementedError()

    @property
    def numpy_dtype(self):
        raise ValueError("'%s' is not a numpy type"
                % str(self))


class AtomicType(LoopyType):
    """
    Abstract class for dtypes of variables encountered in a :class:`loopy.LoopKernel`
    on which atomic operations are performed .
    """


# {{{ numpy-based dtype

class NumpyType(LoopyType):
    def __init__(self, dtype, target=None):
        assert not isinstance(dtype, NumpyType)

        if dtype is None:
            raise TypeError("may not pass None to construct NumpyType")

        if dtype == object:
            raise TypeError("loopy does not directly support object arrays")

        if target is not None:
            warn("Passing target is deprecated and will stop working in 2022.",
                    DeprecationWarning, stacklevel=2)

        self.dtype = np.dtype(dtype)

    def __hash__(self):
        return hash(self.dtype)

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.rec(key_hash, self.dtype)

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.dtype == other.dtype)

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_integral(self):
        return self.dtype.kind in "iu"

    def is_complex(self):
        return self.dtype.kind == "c"

    def involves_complex(self):
        def dtype_involves_complex(dtype):
            if dtype.kind == "c":
                return True

            if dtype.fields is None:
                return False
            else:
                return any(
                        dtype_involves_complex(f[0])
                        for f in dtype.fields.values())

        return dtype_involves_complex(self.dtype)

    def is_composite(self):
        return self.dtype.kind == "V"

    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def numpy_dtype(self):
        return self.dtype

    def __repr__(self):
        return "np:" + repr(self.dtype)

# }}}


# {{{ atomic dtype

class AtomicNumpyType(NumpyType, AtomicType):
    """A dtype wrapper that indicates that the described type should be capable
    of atomic operations.
    """
    def __hash__(self):
        return 0xa7031c ^ hash(self.dtype)

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.rec(key_hash, 0xa7031c)
        key_builder.rec(key_hash, self.dtype)

    def __repr__(self):
        return "np_atomic:%s" % repr(self.dtype)

# }}}


# {{{ opaque type

class OpaqueType(LoopyType):
    """An opaque data type is truly opaque - it has no allocations, no
    temporaries of that type, etc. The only thing allowed is to be pass in
    through one ValueArg and go out to another. It is introduced to accomodate
    functional calls to external libraries.
    """
    def __init__(self, name):
        assert isinstance(name, str)
        self.name = name

    def is_integral(self):
        return False

    def is_complex(self):
        return False

    def involves_complex(self):
        return False

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.rec(key_hash, self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.name == other.name)

    def __ne__(self, other):
        return not self.__eq__(other)

# }}}


def to_loopy_type(dtype, allow_auto=False, allow_none=False, for_atomic=False,
        target=None):
    if target is not None:
        warn("Passing target is deprecated and will stop working in 2022.",
                DeprecationWarning, stacklevel=2)

    from loopy.kernel.data import auto
    if dtype is None:
        if allow_none:
            return None
        else:
            raise LoopyError("dtype may not be none")

    elif dtype is auto:
        if allow_auto:
            return dtype
        else:
            raise LoopyError("dtype may not be auto")

    numpy_dtype = None

    if dtype is not None:
        try:
            numpy_dtype = np.dtype(dtype)
        except Exception:
            pass

    if isinstance(dtype, LoopyType):
        if for_atomic:
            if isinstance(dtype, NumpyType):
                return AtomicNumpyType(dtype.dtype)
            elif not isinstance(dtype, AtomicType):
                raise LoopyError("do not know how to convert '%s' to an atomic type"
                        % dtype)

        return dtype

    elif numpy_dtype is not None:
        if for_atomic:
            return AtomicNumpyType(numpy_dtype)
        else:
            return NumpyType(numpy_dtype)

    else:
        raise TypeError("dtype must be a LoopyType, or convertible to one, "
                "found '%s' instead" % type(dtype))

# vim: foldmethod=marker

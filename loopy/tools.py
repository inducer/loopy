from __future__ import division
from __future__ import absolute_import
import six

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

import numpy as np
from pytools.persistent_dict import KeyBuilder as KeyBuilderBase
from loopy.symbolic import WalkMapper as LoopyWalkMapper
from pymbolic.mapper.persistent_hash import (
        PersistentHashWalkMapper as PersistentHashWalkMapperBase)
import six  # noqa
from six.moves import intern


if six.PY2:
    def is_integer(obj):
        return isinstance(obj, (int, long, np.integer))
else:
    def is_integer(obj):
        return isinstance(obj, (int, np.integer))


# {{{ custom KeyBuilder subclass

class PersistentHashWalkMapper(LoopyWalkMapper, PersistentHashWalkMapperBase):
    """A subclass of :class:`loopy.symbolic.WalkMapper` for constructing
    persistent hash keys for use with
    :class:`pytools.persistent_dict.PersistentDict`.

    See also :meth:`LoopyKeyBuilder.update_for_pymbolic_expression`.
    """

    # <empty implementation>


class LoopyKeyBuilder(KeyBuilderBase):
    """A custom :class:`pytools.persistent_dict.KeyBuilder` subclass
    for objects within :mod:`loopy`.
    """

    # Lists, sets and dicts aren't immutable. But loopy kernels are, so we're
    # simply ignoring that fact here.
    update_for_list = KeyBuilderBase.update_for_tuple
    update_for_set = KeyBuilderBase.update_for_frozenset

    def update_for_dict(self, key_hash, key):
        # Order matters for the hash--insert in sorted order.
        for dict_key in sorted(six.iterkeys(key)):
            self.rec(key_hash, (dict_key, key[dict_key]))

    def update_for_BasicSet(self, key_hash, key):  # noqa
        from islpy import Printer
        prn = Printer.to_str(key.get_ctx())
        getattr(prn, "print_"+key._base_name)(key)
        key_hash.update(prn.get_str().encode("utf8"))

    def update_for_type(self, key_hash, key):
        try:
            method = getattr(self, "update_for_type_"+key.__name__)
        except AttributeError:
            pass
        else:
            method(key_hash, key)
            return

        raise TypeError("unsupported type for persistent hash keying: %s"
                % type(key))

    def update_for_type_auto(self, key_hash, key):
        key_hash.update("auto".encode("utf8"))

    def update_for_pymbolic_expression(self, key_hash, key):
        if key is None:
            self.update_for_NoneType(key_hash, key)
        else:
            PersistentHashWalkMapper(key_hash)(key)


class PymbolicExpressionHashWrapper(object):
    def __init__(self, expression):
        self.expression = expression

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.expression == other.expression)

    def __ne__(self, other):
        return not self.__eq__(other)

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.update_for_pymbolic_expression(key_hash, self.expression)

# }}}


# {{{ picklable dtype

class PicklableDtype(object):
    """This object works around several issues with pickling :class:`numpy.dtype`
    objects. It does so by serving as a picklable wrapper around the original
    dtype.

    The issues are the following

    - :class:`numpy.dtype` objects for custom types in :mod:`loopy` are usually
      registered in the target's dtype registry. This registration may
      have been lost after unpickling. This container restores it implicitly,
      as part of unpickling.

    - There is a`numpy bug <https://github.com/numpy/numpy/issues/4317>`_
      that prevents unpickled dtypes from hashing properly. This is solved
      by retrieving the 'canonical' type from the dtype registry.
    """

    def __init__(self, dtype, target=None):
        assert not isinstance(dtype, PicklableDtype)

        if dtype is None:
            raise TypeError("may not pass None to construct PicklableDtype")

        self.target = target
        self.dtype = np.dtype(dtype)

    def __hash__(self):
        return hash(self.dtype)

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.dtype == other.dtype)

    def __ne__(self, other):
        return not self.__eq__(self, other)

    def __getstate__(self):
        if self.target is None:
            raise RuntimeError("unable to pickle dtype: target not known")

        c_name = self.target.dtype_to_typename(self.dtype)
        return (self.target, c_name, self.dtype)

    def __setstate__(self, state):
        target, name, dtype = state
        self.target = target
        self.dtype = self.target.get_or_register_dtype([name], dtype)

    def with_target(self, target):
        if (self.target is not None
                and target is not self.target):
            raise RuntimeError("target already set to different value")

        return PicklableDtype(self.dtype, target)

    def assert_has_target(self):
        assert self.target is not None

# }}}


# {{{ remove common indentation

def remove_common_indentation(code, require_leading_newline=True,
        ignore_lines_starting_with=None, strip_empty_lines=True):
    if "\n" not in code:
        return code

    # accommodate pyopencl-ish syntax highlighting
    code = code.lstrip("//CL//")

    if require_leading_newline and not code.startswith("\n"):
        return code

    lines = code.split("\n")

    if strip_empty_lines:
        while lines[0].strip() == "":
            lines.pop(0)
        while lines[-1].strip() == "":
            lines.pop(-1)

    test_line = None
    if ignore_lines_starting_with:
        for l in lines:
            strip_l = l.lstrip()
            if (strip_l
                    and not strip_l.startswith(ignore_lines_starting_with)):
                test_line = l
                break

    else:
        test_line = lines[0]

    base_indent = 0
    if test_line:
        while test_line[base_indent] in " \t":
            base_indent += 1

    new_lines = []
    for line in lines:
        if (ignore_lines_starting_with
                and line.lstrip().startswith(ignore_lines_starting_with)):
            new_lines.append(line)
            continue

        if line[:base_indent].strip():
            raise ValueError("inconsistent indentation: '%s'" % line)

        new_lines.append(line[base_indent:])

    return "\n".join(new_lines)

# }}}


def is_interned(s):
    return s is None or intern(s) is s


def intern_frozenset_of_ids(fs):
    return frozenset(intern(s) for s in fs)


# vim: foldmethod=marker

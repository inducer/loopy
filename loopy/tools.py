from __future__ import division, absolute_import

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

import six

try:
    import collections.abc as abc
except ImportError:
    # Python 2
    import collections as abc

import numpy as np
from pytools import memoize_method
from pytools.persistent_dict import KeyBuilder as KeyBuilderBase
from loopy.symbolic import WalkMapper as LoopyWalkMapper
from pymbolic.mapper.persistent_hash import (
        PersistentHashWalkMapper as PersistentHashWalkMapperBase)
import six  # noqa
from six.moves import intern
import re
from mako.template import Template
import loopy as lp

if six.PY2:
    def is_integer(obj):
        return isinstance(obj, (int, long, np.integer))  # noqa pylint:disable=undefined-variable
else:
    def is_integer(obj):
        return isinstance(obj, (int, np.integer))


def update_persistent_hash(obj, key_hash, key_builder):
    """
    Custom hash computation function for use with
    :class:`pytools.persistent_dict.PersistentDict`.

    Only works in conjunction with :class:`loopy.tools.KeyBuilder`.
    """
    for field_name in obj.hash_fields:
        key_builder.rec(key_hash, getattr(obj, field_name))


# {{{ custom KeyBuilder subclass

class PersistentHashWalkMapper(LoopyWalkMapper, PersistentHashWalkMapperBase):
    """A subclass of :class:`loopy.symbolic.WalkMapper` for constructing
    persistent hash keys for use with
    :class:`pytools.persistent_dict.PersistentDict`.

    See also :meth:`LoopyKeyBuilder.update_for_pymbolic_expression`.
    """

    def map_reduction(self, expr, *args):
        if not self.visit(expr):
            return

        self.key_hash.update(type(expr.operation).__name__.encode("utf-8"))
        self.rec(expr.expr, *args)


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
        for dict_key in sorted(six.iterkeys(key), key=lambda obj:
                type(obj).__name__ + str(obj)):
            self.rec(key_hash, (dict_key, key[dict_key]))

    update_for_defaultdict = update_for_dict

    def update_for_frozenset(self, key_hash, key):
        for set_key in sorted(key,
                key=lambda obj: type(obj).__name__ + str(obj)):
            self.rec(key_hash, set_key)

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


# {{{ eq key builder

class LoopyEqKeyBuilder(object):
    """Unlike :class:`loopy.tools.LoopyKeyBuilder`, this builds keys for use in
    equality comparison, such that `key(a) == key(b)` if and only if `a == b`.
    The types of objects being compared should satisfy structural equality.

    The output is suitable for use with :class:`loopy.tools.LoopyKeyBuilder`
    provided all fields are persistent hashable.

    As an optimization, top-level pymbolic expression fields are stringified for
    faster comparisons / hash calculations.

    Usage::

        kb = LoopyEqKeyBuilder()
        kb.update_for_class(insn.__class__)
        kb.update_for_field("field", insn.field)
        ...
        key = kb.key()

    """

    def __init__(self):
        self.field_dict = {}

    def update_for_class(self, class_):
        self.class_ = class_

    def update_for_field(self, field_name, value):
        self.field_dict[field_name] = value

    def update_for_pymbolic_field(self, field_name, value):
        from loopy.symbolic import EqualityPreservingStringifyMapper
        self.field_dict[field_name] = \
                EqualityPreservingStringifyMapper()(value).encode("utf-8")

    def key(self):
        """A key suitable for equality comparison."""
        return (self.class_.__name__.encode("utf-8"), self.field_dict)

    @memoize_method
    def hash_key(self):
        """A key suitable for hashing.
        """
        # To speed up any calculations that repeatedly use the return value,
        # this method returns a hash.

        kb = LoopyKeyBuilder()
        # Build the key. For faster hashing, avoid hashing field names.
        key = (
            (self.class_.__name__.encode("utf-8"),) +
            tuple(self.field_dict[k] for k in sorted(self.field_dict.keys())))

        return kb(key)

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


# {{{ build_ispc_shared_lib

# DO NOT RELY ON THESE: THEY WILL GO AWAY

def build_ispc_shared_lib(
        cwd, ispc_sources, cxx_sources,
        ispc_options=[], cxx_options=[],
        ispc_bin="ispc",
        cxx_bin="g++",
        quiet=True):
    from os.path import join

    ispc_source_names = []
    for name, contents in ispc_sources:
        ispc_source_names.append(name)

        with open(join(cwd, name), "w") as srcf:
            srcf.write(contents)

    cxx_source_names = []
    for name, contents in cxx_sources:
        cxx_source_names.append(name)

        with open(join(cwd, name), "w") as srcf:
            srcf.write(contents)

    from subprocess import check_call

    ispc_cmd = ([ispc_bin,
                "--pic",
                "-o", "ispc.o"]
            + ispc_options
            + list(ispc_source_names))
    if not quiet:
        print(" ".join(ispc_cmd))

    check_call(ispc_cmd, cwd=cwd)

    cxx_cmd = ([
                cxx_bin,
                "-shared", "-Wl,--export-dynamic",
                "-fPIC",
                "-oshared.so",
                "ispc.o",
                ]
            + cxx_options
            + list(cxx_source_names))

    check_call(cxx_cmd, cwd=cwd)

    if not quiet:
        print(" ".join(cxx_cmd))

# }}}


# {{{ numpy address munging

# DO NOT RELY ON THESE: THEY WILL GO AWAY

def address_from_numpy(obj):
    ary_intf = getattr(obj, "__array_interface__", None)
    if ary_intf is None:
        raise RuntimeError("no array interface")

    buf_base, is_read_only = ary_intf["data"]
    return buf_base + ary_intf.get("offset", 0)


def cptr_from_numpy(obj):
    import ctypes
    return ctypes.c_void_p(address_from_numpy(obj))


# https://github.com/hgomersall/pyFFTW/blob/master/pyfftw/utils.pxi#L172
def empty_aligned(shape, dtype, order='C', n=64):
    '''empty_aligned(shape, dtype='float64', order='C', n=None)
    Function that returns an empty numpy array that is n-byte aligned,
    where ``n`` is determined by inspecting the CPU if it is not
    provided.
    The alignment is given by the final optional argument, ``n``. If
    ``n`` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.empty`.
    '''
    itemsize = np.dtype(dtype).itemsize

    # Apparently there is an issue with numpy.prod wrapping around on 32-bits
    # on Windows 64-bit. This shouldn't happen, but the following code
    # alleviates the problem.
    if not isinstance(shape, (int, np.integer)):
        array_length = 1
        for each_dimension in shape:
            array_length *= each_dimension

    else:
        array_length = shape

    base_ary = np.empty(array_length*itemsize+n, dtype=np.int8)

    # We now need to know how to offset base_ary
    # so it is correctly aligned
    _array_aligned_offset = (n-address_from_numpy(base_ary)) % n

    array = np.frombuffer(
            base_ary[_array_aligned_offset:_array_aligned_offset-n].data,
            dtype=dtype).reshape(shape, order=order)

    return array

# }}}


# {{{ compute SCCs with Tarjan's algorithm

def compute_sccs(graph):
    to_search = set(graph.keys())
    visit_order = {}
    scc_root = {}
    sccs = []

    while to_search:
        top = next(iter(to_search))
        call_stack = [(top, iter(graph[top]), None)]
        visit_stack = []
        visiting = set()

        scc = []

        while call_stack:
            top, children, last_popped_child = call_stack.pop()

            if top not in visiting:
                # Unvisited: mark as visited, initialize SCC root.
                count = len(visit_order)
                visit_stack.append(top)
                visit_order[top] = count
                scc_root[top] = count
                visiting.add(top)
                to_search.discard(top)

            # Returned from a recursion, update SCC.
            if last_popped_child is not None:
                scc_root[top] = min(
                    scc_root[top],
                    scc_root[last_popped_child])

            for child in children:
                if child not in visit_order:
                    # Recurse.
                    call_stack.append((top, children, child))
                    call_stack.append((child, iter(graph[child]), None))
                    break
                if child in visiting:
                    scc_root[top] = min(
                        scc_root[top],
                        visit_order[child])
            else:
                if scc_root[top] == visit_order[top]:
                    scc = []
                    while visit_stack[-1] != top:
                        scc.append(visit_stack.pop())
                    scc.append(visit_stack.pop())
                    for item in scc:
                        visiting.remove(item)
                    sccs.append(scc)

    return sccs

# }}}


# {{{ pickled container value

class _PickledObject(object):
    """A class meant to wrap a pickled value (for :class:`LazilyUnpicklingDict` and
    :class:`LazilyUnpicklingList`).
    """

    def __init__(self, obj):
        if isinstance(obj, _PickledObject):
            self.objstring = obj.objstring
        else:
            from pickle import dumps
            self.objstring = dumps(obj)

    def unpickle(self):
        from pickle import loads
        return loads(self.objstring)

    def __getstate__(self):
        return {"objstring": self.objstring}


class _PickledObjectWithEqAndPersistentHashKeys(_PickledObject):
    """Like :class:`_PickledObject`, with two additional attributes:

        * `eq_key`
        * `persistent_hash_key`

    This allows for comparison and for persistent hashing without unpickling.
    """

    def __init__(self, obj, eq_key, persistent_hash_key):
        _PickledObject.__init__(self, obj)
        self.eq_key = eq_key
        self.persistent_hash_key = persistent_hash_key

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.rec(key_hash, self.persistent_hash_key)

    def __getstate__(self):
        return {"objstring": self.objstring,
                "eq_key": self.eq_key,
                "persistent_hash_key": self.persistent_hash_key}

# }}}


# {{{ lazily unpickling dictionary

class LazilyUnpicklingDict(abc.MutableMapping):
    """A dictionary-like object which lazily unpickles its values.
    """

    def __init__(self, *args, **kwargs):
        self._map = dict(*args, **kwargs)

    def __getitem__(self, key):
        value = self._map[key]
        if isinstance(value, _PickledObject):
            value = self._map[key] = value.unpickle()
        return value

    def __setitem__(self, key, value):
        self._map[key] = value

    def __delitem__(self, key):
        del self._map[key]

    def __len__(self):
        return len(self._map)

    def __iter__(self):
        return iter(self._map)

    def __getstate__(self):
        return {"_map": dict(
            (key, _PickledObject(val))
            for key, val in six.iteritems(self._map))}

# }}}


# {{{ lazily unpickling list

class LazilyUnpicklingList(abc.MutableSequence):
    """A list which lazily unpickles its values."""

    def __init__(self, *args, **kwargs):
        self._list = list(*args, **kwargs)

    def __getitem__(self, key):
        item = self._list[key]
        if isinstance(item, _PickledObject):
            item = self._list[key] = item.unpickle()
        return item

    def __setitem__(self, key, value):
        self._list[key] = value

    def __delitem__(self, key):
        del self._list[key]

    def __len__(self):
        return len(self._list)

    def insert(self, key, value):
        self._list.insert(key, value)

    def __getstate__(self):
        return {"_list": [_PickledObject(val) for val in self._list]}


class LazilyUnpicklingListWithEqAndPersistentHashing(LazilyUnpicklingList):
    """A list which lazily unpickles its values, and supports equality comparison
    and persistent hashing without unpickling.

    Persistent hashing only works in conjunction with :class:`LoopyKeyBuilder`.

    Equality comparison and persistent hashing are implemented by supplying
    functions `eq_key_getter` and `persistent_hash_key_getter` to the
    constructor. These functions should return keys that can be used in place of
    the original object for the respective purposes of equality comparison and
    persistent hashing.
    """

    def __init__(self, *args, **kwargs):
        self.eq_key_getter = kwargs.pop("eq_key_getter")
        self.persistent_hash_key_getter = kwargs.pop("persistent_hash_key_getter")
        LazilyUnpicklingList.__init__(self, *args, **kwargs)

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.update_for_list(key_hash, self._list)

    def _get_eq_key(self, obj):
        if isinstance(obj, _PickledObjectWithEqAndPersistentHashKeys):
            return obj.eq_key
        return self.eq_key_getter(obj)

    def _get_persistent_hash_key(self, obj):
        if isinstance(obj, _PickledObjectWithEqAndPersistentHashKeys):
            return obj.persistent_hash_key
        return self.persistent_hash_key_getter(obj)

    def __eq__(self, other):
        if not isinstance(other, (list, LazilyUnpicklingList)):
            return NotImplemented

        if isinstance(other, LazilyUnpicklingList):
            other = other._list

        if len(self) != len(other):
            return False

        for a, b in zip(self._list, other):
            if self._get_eq_key(a) != self._get_eq_key(b):
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getstate__(self):
        return {"_list": [
                _PickledObjectWithEqAndPersistentHashKeys(
                    val,
                    self._get_eq_key(val),
                    self._get_persistent_hash_key(val))
                for val in self._list],
                "eq_key_getter": self.eq_key_getter,
                "persistent_hash_key_getter": self.persistent_hash_key_getter}

# }}}


# {{{ optional object

class _no_value(object):  # noqa
    pass


class Optional(object):
    """A wrapper for an optionally present object.

    .. attribute:: has_value

        *True* if and only if this object contains a value.

    .. attribute:: value

        The value, if present.
    """

    __slots__ = ("has_value", "_value")

    def __init__(self, value=_no_value):
        self.has_value = value is not _no_value
        if self.has_value:
            self._value = value

    def __str__(self):
        if not self.has_value:
            return "Optional()"
        return "Optional(%s)" % self._value

    def __repr__(self):
        if not self.has_value:
            return "Optional()"
        return "Optional(%r)" % self._value

    def __getstate__(self):
        if not self.has_value:
            return _no_value

        return (self._value,)

    def __setstate__(self, state):
        if state is _no_value:
            self.has_value = False
            return

        self.has_value = True
        self._value, = state

    def __eq__(self, other):
        if not self.has_value:
            return not other.has_value

        return self.value == other.value if other.has_value else False

    def __neq__(self, other):
        return not self.__eq__(other)

    @property
    def value(self):
        if not self.has_value:
            raise AttributeError("optional value not present")
        return self._value

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.rec(
                key_hash,
                (self._value,) if self.has_value else ())

# }}}


def unpickles_equally(obj):
    from six.moves.cPickle import loads, dumps
    return loads(dumps(obj)) == obj


def is_interned(s):
    return s is None or intern(s) is s


def intern_frozenset_of_ids(fs):
    return frozenset(intern(s) for s in fs)


def natorder(key):
    # Return natural ordering for strings, as opposed to dictionary order.
    # E.g. will result in
    #  'abc1' < 'abc9' < 'abc10'
    # rather than
    #  'abc1' < 'abc10' < 'abc9'
    # Based on
    # http://code.activestate.com/recipes/285264-natural-string-sorting/#c7
    import re
    return [int(n) if n else s for n, s in re.findall(r'(\d+)|(\D+)', key)]


def natsorted(seq, key=lambda x: x):
    return sorted(seq, key=lambda y: natorder(key(y)))


def dump_as_python(kernel, filename=None):
    """
    Generates a python code for generating *kernel* for sharing kernels.

    :arg kernel: An instance of :class:`loopy.LoopKernel`
    :arg filename: An instance of :class:`str`. If *None*, then prints the
        python file to *stdout*.
    """

    options = []

    printed_insn_ids = set()
    printed_insn_order = []

    def insert_insn_into_order(insn):
        if insn.id in printed_insn_ids:
            return
        printed_insn_ids.add(insn.id)

        for dep_id in natsorted(insn.depends_on):
            insert_insn_into_order(kernel.id_to_insn[dep_id])

        printed_insn_order.append(insn)

    for insn in kernel.instructions:
        insert_insn_into_order(insn)

    for insn in printed_insn_order:
        option = 'id=%s, ' % insn.id
        if insn.depends_on:
            option += ("dep="+":".join(insn.depends_on)+", ")
        if insn.tags:
            option += ("tags="+":".join(insn.tags)+", ")
        if insn.within_inames:
            option += ("inames="+":".join(insn.within_inames)+", ")
        if isinstance(insn, lp.MultiAssignmentBase):
            if insn.atomicity:
                option += "atomic, "
        elif isinstance(insn, lp.BarrierInstruction):
            option += ("mem_kind=%s, " % insn.mem_kind)
        options.append(option[:-2])

    insn_x_options = zip(printed_insn_order, options)

    python_code = r'''<%! import loopy as lp %>import loopy as lp
    import numpy as np
    <%! tv_scope = {0: 'lp.AddressSpace.PRIVATE', 1: 'lp.AddressSpace.LOCAL',
    2: 'lp.AddressSpace.GLOBAL', lp.auto: 'lp.auto' } %>
    knl = lp.make_kernel(
        [
        % for dom in domains:
        "${str(dom)}",
        % endfor
        ],
        """
        % for insn, opts in insn_x_opts:
        % if isinstance(insn, lp.Assignment):
        ${insn.assignee} = ${insn.expression} {${opts}}
        % elif isinstance(insn, lp.BarrierInstruction):
        ... ${insn.synchronization_kind[0]}barrier{${opts}}
        % elif isinstance(insn, lp.NoOpInstruction):
        ... nop {${opts}}
        % else:
        **Not implemented for ${type(insn)}**
        % endif
        %endfor
        """, [
            % for arg in args:
            % if isinstance(arg, lp.ValueArg):
            lp.ValueArg(
                name='${arg.name}', dtype=np.${arg.dtype.numpy_dtype.name}),
            % else:
            lp.GlobalArg(
                name='${arg.name}', dtype=np.${arg.dtype.numpy_dtype.name},
                shape=${arg.shape}, for_atomic=${arg.for_atomic}),
            % endif
            % endfor
            % for tv in temp_vars:
            lp.TemporaryVariable(
                name='${tv.name}', dtype=np.${tv.dtype.numpy_dtype.name},
                shape=${tv.shape}, for_atomic=${tv.for_atomic},
                address_space=${tv_scope[tv.address_space]},
                read_only=${tv.read_only},
                % if tv.initializer is not None:
                initializer=${"np."+str((tv.initializer).__repr__())},
                % endif
                ),
            % endfor
            ], lang_version=${lp.MOST_RECENT_LANGUAGE_VERSION})'''

    python_code = Template(python_code).render(insn_x_opts=insn_x_options,
            domains=kernel.domains, args=kernel.args,
            temp_vars=[k for k in kernel.temporary_variables.values()])

    python_code = re.sub("\\n    ", "\n", python_code)
    if filename:
        with open(filename, 'w') as f:
            f.write(python_code)
    else:
        print(python_code)


# vim: foldmethod=marker

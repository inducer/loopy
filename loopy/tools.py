from __future__ import annotations


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
import collections.abc as abc
import logging
from functools import cached_property
from sys import intern
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeVar, cast, overload

import numpy as np
from constantdict import constantdict
from typing_extensions import override

import islpy as isl
from pytools import Hash, ProcessLogger, memoize_method
from pytools.persistent_dict import (
    KeyBuilder as KeyBuilderBase,
    WriteOncePersistentDict,
)

from .symbolic import (
    RuleAwareIdentityMapper,
)


if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    from loopy.kernel import LoopKernel
    from loopy.translation_unit import TranslationUnit
    from loopy.typing import InsnId


logger = logging.getLogger(__name__)


K = TypeVar("K")
V = TypeVar("V")


T_co = TypeVar("T_co", covariant=True)


def update_persistent_hash(obj, key_hash, key_builder):
    """
    Custom hash computation function for use with
    :class:`pytools.persistent_dict.PersistentDict`.

    Only works in conjunction with :class:`loopy.tools.KeyBuilder`.
    """
    for field_name in obj.hash_fields:
        key_builder.rec(key_hash, getattr(obj, field_name))


# {{{ custom KeyBuilder subclass

class LoopyKeyBuilder(KeyBuilderBase):
    """A custom :class:`pytools.persistent_dict.KeyBuilder` subclass
    for objects within :mod:`loopy`.
    """

    # Lists, sets and dicts aren't immutable. But loopy kernels are, so we're
    # simply ignoring that fact here.
    update_for_list = KeyBuilderBase.update_for_tuple
    update_for_set = KeyBuilderBase.update_for_frozenset

    update_for_dict = KeyBuilderBase.update_for_constantdict
    update_for_defaultdict = KeyBuilderBase.update_for_constantdict

    def update_for_BasicSet(self, key_hash, key):  # noqa
        from islpy import Printer
        prn = Printer.to_str(key.get_ctx())
        getattr(prn, "print_"+key._base_name)(key)
        key_hash.update(prn.get_str().encode("utf8"))

    def update_for_Map(self, key_hash, key):  # noqa
        if isinstance(key, isl.Map):
            self.update_for_BasicSet(key_hash, key)
        else:
            raise AssertionError()

# }}}


# {{{ eq key builder

class LoopyEqKeyBuilder:
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
            (self.class_.__name__.encode("utf-8"),
                *(self.field_dict[k] for k in sorted(self.field_dict.keys()))))

        return kb(key)

# }}}


# {{{ remove common indentation

def remove_common_indentation(code, require_leading_newline=True,
        ignore_lines_starting_with=None, strip_empty_lines=True):
    if "\n" not in code:
        return code

    # accommodate pyopencl-ish syntax highlighting
    cl_prefix = "//CL//"
    if code.startswith(cl_prefix):
        code = code[len(cl_prefix):]

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
        for line in lines:
            strip_l = line.lstrip()
            if (strip_l
                    and not strip_l.startswith(ignore_lines_starting_with)):
                test_line = line
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


# {{{ remove_lines_with_only_spaces

def remove_lines_with_only_spaces(code):
    return "\n".join(line for line in code.split("\n") if set(line) != {" "})

# }}}


# {{{ build_ispc_shared_lib

# DO NOT RELY ON THESE: THEY WILL GO AWAY

def build_ispc_shared_lib(
        cwd, ispc_sources, cxx_sources,
        ispc_options=None, cxx_options=None,
        ispc_bin="ispc",
        cxx_bin="g++",
        quiet=True):
    if ispc_options is None:
        ispc_options = []
    if cxx_options is None:
        cxx_options = []

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

    ispc_cmd = ([ispc_bin, "--pic", "-o", "ispc.o", *ispc_options, *ispc_source_names])
    if not quiet:
        print(" ".join(ispc_cmd))

    check_call(ispc_cmd, cwd=cwd)

    cxx_cmd = ([cxx_bin, "-shared", "-Wl,--export-dynamic", "-fPIC", "-oshared.so",
        "ispc.o", *cxx_options, *cxx_source_names])

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

    buf_base, _is_read_only = ary_intf["data"]
    return buf_base + ary_intf.get("offset", 0)


def cptr_from_numpy(obj):
    import ctypes
    return ctypes.c_void_p(address_from_numpy(obj))


# https://github.com/hgomersall/pyFFTW/blob/master/pyfftw/utils.pxi#L172
def empty_aligned(
            shape: tuple[int, ...] | int,
            dtype: DTypeLike,
            order: np._OrderACF = "C",
            n: int = 64
         ) -> NDArray[np.generic]:
    """empty_aligned(shape, dtype='float64', order="C", n=None)
    Function that returns an empty numpy array that is n-byte aligned,
    where ``n`` is determined by inspecting the CPU if it is not
    provided.
    The alignment is given by the final optional argument, ``n``. If
    ``n`` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.empty`.
    """
    shape_tup = shape if isinstance(shape, tuple) else (shape,)

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
    array_aligned_offset = (n-address_from_numpy(base_ary)) % n

    array = np.frombuffer(
            base_ary[array_aligned_offset:array_aligned_offset-n].data,
            dtype=dtype).reshape(*shape_tup, order=order)

    return array

# }}}


# {{{ pickled container value

class _PickledObject:
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

    def __repr__(self) -> str:
        return type(self).__name__ + "(" + repr(self.unpickle()) + ")"


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

class LazilyUnpicklingDict(abc.MutableMapping[K, V]):
    """A dictionary-like object which lazily unpickles its values.
    """

    _map: dict[K, V]

    def __init__(self, *args: object):
        self._map = dict(*args)

    @override
    def __getitem__(self, key: K) -> V:
        value = self._map[key]
        if isinstance(value, _PickledObject):
            value = self._map[key] = cast("V", value.unpickle())
        return value

    @override
    def __setitem__(self, key, value):
        self._map[key] = value

    @override
    def __delitem__(self, key):
        del self._map[key]

    @override
    def __len__(self):
        return len(self._map)

    @override
    def __iter__(self) -> abc.Iterator[K]:
        return iter(self._map)

    def __getstate__(self):
        return {"_map": {
            key: _PickledObject(val)
            for key, val in self._map.items()}}

    @override
    def __repr__(self) -> str:
        return type(self).__name__ + "(" + repr(self._map) + ")"

# }}}


# {{{ lazily unpickling list

class LazilyUnpicklingList(abc.MutableSequence[V]):
    """A list which lazily unpickles its values."""

    _list: list[V]

    def __init__(self, *args, **kwargs):
        self._list = list(*args, **kwargs)

    # incompatible because no slice support
    @override
    def __getitem__(self, key: int) -> V:  # pyright: ignore[reportIncompatibleMethodOverride]
        item = self._list[key]
        if isinstance(item, _PickledObject):
            item = self._list[key] = cast("V", item.unpickle())
        return item

    # incompatible because no slice support
    @override
    def __setitem__(self, key: int, value: V):  # pyright: ignore[reportIncompatibleMethodOverride]
        self._list[key] = value

    # incompatible because no slice support
    @override
    def __delitem__(self, key: int):  # pyright: ignore[reportIncompatibleMethodOverride]
        del self._list[key]

    @override
    def __len__(self):
        return len(self._list)

    @override
    def insert(self, key: int, value: V):
        self._list.insert(key, value)

    def __getstate__(self):
        return {"_list": [_PickledObject(val) for val in self._list]}

    def __add__(self, other):
        return self._list + other

    def __mul__(self, other):
        return self._list * other

    @override
    def __repr__(self) -> str:
        return type(self).__name__ + "(" + repr(self._list) + ")"


class LazilyUnpicklingListWithEqAndPersistentHashing(LazilyUnpicklingList[V]):
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

class _no_value:  # noqa
    pass


class Optional(Generic[T_co]):
    """A wrapper for an optionally present object.

    .. attribute:: has_value

        *True* if and only if this object contains a value.

    .. attribute:: value

        The value, if present.
    """

    has_value: bool
    _value: T_co

    __slots__: ClassVar[tuple[str, ...]] = ("_value", "has_value")

    def __init__(self, value: T_co | type[_no_value] = _no_value):
        self.has_value = value is not _no_value
        if self.has_value:
            self._value = cast("T_co", value)

    @override
    def __str__(self) -> str:
        if not self.has_value:
            return "Optional()"
        return "Optional(%s)" % self._value

    @override
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

    @override
    def __eq__(self, other: object):
        if not isinstance(other, Optional):
            return False

        if not self.has_value:
            return not other.has_value

        return self.value == other.value if other.has_value else False

    @property
    def value(self):
        if not self.has_value:
            raise AttributeError("optional value not present")
        return self._value

    def update_persistent_hash(self,
            key_hash: Hash,
            key_builder: KeyBuilderBase) -> None:
        key_builder.rec(
                key_hash,
                (self._value,) if self.has_value else ())

    @override
    def __hash__(self):
        if not self.has_value:
            return hash((type(self), False))
        else:
            return hash((self.has_value, self._value))

# }}}


def unpickles_equally(obj: object) -> bool:
    from pickle import dumps, loads
    return loads(dumps(obj)) == obj  # pyright: ignore[reportAny]


def is_interned(s: str | None) ->  bool:
    return s is None or intern(s) is s


def intern_frozenset_of_ids(fs: abc.Iterable[str]) -> frozenset[str]:
    return frozenset(intern(s) for s in fs)


# {{{ t_unit_to_python

def _is_generated_t_unit_the_same(python_code, var_name, ref_t_unit):
    """
    Helper for :func:`kernel_to_python`. Returns *True* only if the variable
    referenced by *var_name* in *python_code* is equal to *kernel*, else
    returns *False*.
    """
    reproducer_variables = {}
    exec(python_code, reproducer_variables)
    t_unit = reproducer_variables[var_name]
    return ref_t_unit == t_unit


# {{{ CallablesUnresolver

class _CallablesUnresolver(RuleAwareIdentityMapper[[]]):
    def __init__(self, rule_mapping_context, callables_table, target):
        super().__init__(rule_mapping_context)
        self.callables_table = callables_table
        self.target = target

    @cached_property
    def known_callables(self):
        from loopy.kernel.function_interface import CallableKernel
        return (frozenset(self.target.get_device_ast_builder().known_callables)
                | {name
                   for name, clbl in self.callables_table.items()
                   if isinstance(clbl, CallableKernel)})

    def map_call(self, expr, expn_state):
        from loopy.symbolic import ResolvedFunction
        if isinstance(expr.function, ResolvedFunction):
            if expr.function.name not in self.known_callables:
                raise NotImplementedError("User-provided scalar callables not"
                                          " supported yet.")

            from pymbolic.primitives import Call
            return Call(expr.function.function, tuple(self.rec(par, expn_state)
                                                      for par in expr.parameters))
        else:
            return super().map_call(expr, expn_state)


def _unresolve_callables(kernel, callables_table):
    from loopy.kernel import KernelState
    from loopy.symbolic import SubstitutionRuleMappingContext

    vng = kernel.get_var_name_generator()
    rule_mapping_context = SubstitutionRuleMappingContext(kernel.substitutions,
                                                          vng)
    mapper = _CallablesUnresolver(rule_mapping_context,
                                  callables_table,
                                  kernel.target)
    return (rule_mapping_context.finish_kernel(mapper.map_kernel(kernel))
            .copy(state=KernelState.INITIAL))

# }}}


def _kernel_to_python(
            kernel: LoopKernel,
            is_entrypoint: bool = False,
            var_name: str = "kernel"
        ) -> str:
    from mako.template import Template

    from loopy.kernel.instruction import BarrierInstruction, MultiAssignmentBase

    options: dict[InsnId, str] = {}  # options: mapping from insn_id to str of options

    for insn in kernel.instructions:
        option = f"id={insn.id}, "
        if insn.depends_on:
            option += ("dep="+":".join(insn.depends_on)+", ")
        if insn.tags:
            option += ("tags="+":".join(str(tag) for tag in insn.tags)+", ")
        if insn.within_inames is not None:
            if insn.within_inames_is_final:
                option += ("inames="+":".join(insn.within_inames)+", ")
            else:
                option += ("inames=+"+":".join(insn.within_inames)+", ")

        if isinstance(insn, MultiAssignmentBase):
            if insn.atomicity:
                option += "atomic, "
        elif isinstance(insn, BarrierInstruction):
            option += (f"mem_kind={insn.mem_kind}, ")

        options[insn.id] = option[:-2]  # get rid of the trailing ", "

    make_kernel = "make_kernel" if is_entrypoint else "make_function"

    python_code = r"""
    <%! import loopy as lp %>

    <%! tv_aspace = {0: 'lp.AddressSpace.PRIVATE', 1: 'lp.AddressSpace.LOCAL',
    2: 'lp.AddressSpace.GLOBAL', lp.auto: 'lp.auto' } %>
    ${var_name} = lp.${make_kernel}(
        [
        % for dom in kernel.domains:
        "${str(dom)}",
        % endfor
        ],
        '''
        % for name, rule in sorted(kernel.substitutions.items(), key=lambda x: x[0]):
        ${name}(${", ".join(rule.arguments)}) := ${str(rule.expression)}
        %endfor

        % for id, opts in options.items():
        <% insn = kernel.id_to_insn[id] %>
        % if isinstance(insn, lp.MultiAssignmentBase):
        ${','.join([str(a) for a in insn.assignees])} = ${insn.expression} {${opts}}
        % elif isinstance(insn, lp.BarrierInstruction):
        ... ${insn.synchronization_kind[0]}barrier {${opts}}
        % elif isinstance(insn, lp.NoOpInstruction):
        ... nop {${opts}}
        % else:
        <% raise NotImplementedError(f"Not implemented for {type(insn)}.")%>
        % endif
        %endfor
        ''', [
            % for arg in kernel.args:
            % if isinstance(arg, lp.ValueArg):
            lp.ValueArg(
                name="${arg.name}",
                dtype=${('np.'+arg.dtype.numpy_dtype.name
                            if arg.dtype else 'None')}),
            % else:
            lp.GlobalArg(
                name="${arg.name}", dtype=${('np.'+arg.dtype.numpy_dtype.name
                                                if arg.dtype else 'None')},
                shape=${arg.shape}, for_atomic=${arg.for_atomic}),
            % endif
            % endfor
            % for tv in kernel.temporary_variables.values():
            lp.TemporaryVariable(
                name="${tv.name}",
                dtype=${'np.'+tv.dtype.numpy_dtype.name if tv.dtype else 'lp.auto'},
                shape=${tv.shape}, for_atomic=${tv.for_atomic},
                address_space=${tv_aspace[tv.address_space]},
                read_only=${tv.read_only},
                % if tv.initializer is not None:
                initializer=${"np."+repr(tv.initializer)},
                % endif
                ),
            % endfor
            ],
            lang_version=${lp.MOST_RECENT_LANGUAGE_VERSION},
    % if kernel.iname_slab_increments:
            iname_slab_increments=${repr(kernel.iname_slab_increments)},
    % endif
    % if kernel.applied_iname_rewrites:
            applied_iname_rewrites=${repr(kernel.applied_iname_rewrites)},
    % endif
    % if kernel.name != "loopy_kernel":
            name="${kernel.name}",
    % endif
            )

    % for iname in kernel.inames.values():
    % for tag in iname.tags:
    ${var_name} = lp.tag_inames(${var_name}, "${"%s:%s" %(iname.name, tag)}")
    % endfor

    % endfor
    """

    python_code = Template(python_code,
                           strict_undefined=True).render(options=options,
                                                         kernel=kernel,
                                                         make_kernel=make_kernel,
                                                         var_name=var_name)
    python_code = remove_lines_with_only_spaces(
            remove_common_indentation(python_code))

    return python_code


@overload
def t_unit_to_python(
            t_unit: TranslationUnit,
            *, var_name: str = "t_unit",
            return_preamble_and_body_separately: Literal[False] = False
        ) -> str: ...


@overload
def t_unit_to_python(
            t_unit: TranslationUnit,
            *, var_name: str = "t_unit",
            return_preamble_and_body_separately: Literal[True]
        ) -> tuple[str, str]: ...


def t_unit_to_python(
            t_unit: TranslationUnit,
            *, var_name: str = "t_unit",
            return_preamble_and_body_separately: bool = False
        ) -> tuple[str, str] | str:
    """"
    Returns a :class:`str` of a python code that instantiates *kernel*.

    :arg kernel: An instance of :class:`loopy.LoopKernel`
    :arg var_name: A :class:`str` of the kernel variable name in the generated
        python script.
    :arg return_preamble_and_body_separately: A :class:`bool`.
        If *True* returns ``(preamble, body)``, where ``preamble`` includes the
        import statements and ``body`` includes the kernel, translation unit
        instantiation code.

    .. note::

        The implementation is partially complete and a :class:`AssertionError`
        is raised if the returned python script does not exactly reproduce
        *kernel*. Contributions are welcome to fill in the missing voids.
    """
    from loopy.kernel.function_interface import CallableKernel

    new_callables = {name: CallableKernel(_unresolve_callables(clbl.subkernel,
                                                               t_unit
                                                               .callables_table))
                     for name, clbl in t_unit.callables_table.items()
                     if isinstance(clbl, CallableKernel)}
    t_unit = t_unit.copy(callables_table=constantdict(new_callables))

    knl_python_code_srcs = [_kernel_to_python(cast("CallableKernel", clbl).subkernel,
                                              name in t_unit.entrypoints,
                                              f"{name}_knl"
                                              )
                            for name, clbl in t_unit.callables_table.items()]

    knl_args = ", ".join(f"{name}_knl" for name in t_unit.callables_table)
    merge_stmt = f"{var_name} = lp.merge([{knl_args}])"

    preamble_str = "\n".join([
        "import loopy as lp",
        "import numpy as np",
        "from pymbolic.primitives import *",
        "from constantdict import constantdict",
        ])
    body_str = "\n".join([*knl_python_code_srcs, "\n", merge_stmt])

    python_code = "\n".join([preamble_str, "\n", body_str])
    assert _is_generated_t_unit_the_same(python_code, var_name, t_unit)

    if return_preamble_and_body_separately:
        return preamble_str, body_str
    else:
        return python_code

# }}}


# {{{ cache management

caches: list[WriteOncePersistentDict] = []


def clear_in_mem_caches() -> None:
    for cache in caches:
        cache.clear_in_mem_cache()

# }}}


# {{{ memoize_on_disk

def memoize_on_disk(func, key_builder_t=LoopyKeyBuilder):
    from functools import wraps

    from pytools.persistent_dict import WriteOncePersistentDict

    from loopy.kernel import LoopKernel
    from loopy.translation_unit import TranslationUnit
    from loopy.version import DATA_MODEL_VERSION

    transform_cache = WriteOncePersistentDict(
        ("loopy-memoize-cache-"
            f"{func.__name__}-"
            f"{key_builder_t.__qualname__}.{key_builder_t.__name__}"
            f"-v0-{DATA_MODEL_VERSION}"),
        key_builder=key_builder_t(),
        safe_sync=False)

    caches.append(transform_cache)

    @wraps(func)
    def wrapper(*args, **kwargs):
        from loopy import CACHING_ENABLED

        if (not CACHING_ENABLED
                or kwargs.pop("_no_memoize_on_disk", False)):
            return func(*args, **kwargs)

        cache_key = (func.__qualname__, func.__name__, args, kwargs)

        try:
            result = transform_cache[cache_key]
            logger.debug(f"Function {func.__name__} returned from"
                         " memoized result on disk.")
            return result
        except KeyError:
            logger.debug(f"Function {func.__name__} not present"
                         " on disk.")
            if args and isinstance(args[0], LoopKernel):
                proc_log_str = f"{func.__name__} on '{args[0].name}'"
            elif args and isinstance(args[0], TranslationUnit):
                entrypoints_str = ", ".join(args[0].entrypoints)
                proc_log_str = f"{func.__name__} on '{entrypoints_str}'"
            else:
                proc_log_str = f"{func.__name__}"

            with ProcessLogger(logger, proc_log_str):
                result = func(*args, **kwargs)

            transform_cache.store_if_not_present(cache_key, result)
            return result

    return wrapper

# }}}


def is_hashable(o: object) -> bool:
    try:
        hash(o)
    except TypeError:
        return False
    return True


# vim: fdm=marker

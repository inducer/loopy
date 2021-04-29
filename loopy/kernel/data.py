"""Data used by the kernel object."""


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


from sys import intern
import numpy as np  # noqa
from pytools import ImmutableRecord
from pytools.tag import Taggable
from pytools.tag import UniqueTag as UniqueTagBase
from loopy.kernel.array import ArrayBase
from loopy.diagnostic import LoopyError
from loopy.kernel.instruction import (  # noqa
        InstructionBase,
        MemoryOrdering,
        MemoryScope,
        VarAtomicity,
        AtomicInit,
        AtomicUpdate,
        MultiAssignmentBase,
        Assignment,
        ExpressionInstruction,
        CallInstruction,
        make_assignment,
        CInstruction)
from warnings import warn

__doc__ = """
.. currentmodule:: loopy.kernel.data

.. autofunction:: filter_iname_tags_by_type

.. autoclass:: IndexTag

.. autoclass:: ConcurrentTag

.. autoclass:: UniqueTag

.. autoclass:: AxisTag

.. autoclass:: LocalIndexTag

.. autoclass:: GroupIndexTag

.. autoclass:: VectorizeTag

.. autoclass:: UnrollTag

.. autoclass:: Iname

.. autoclass:: KernelArgument
"""


class auto:  # noqa
    """A generic placeholder object for something that should be automatically
    determined.  See, for example, the *shape* or *strides* argument of
    :class:`ArrayArg`.
    """


# {{{ iname tags


def filter_iname_tags_by_type(tags, tag_type, max_num=None, min_num=None):
    """Return a subset of *tags* that matches type *tag_type*. Raises exception
    if the number of tags found were greater than *max_num* or less than
    *min_num*.

    :arg tags: An iterable of tags.
    :arg tag_type: a subclass of :class:`loopy.kernel.data.IndexTag`.
    :arg max_num: the maximum number of tags expected to be found.
    :arg min_num: the minimum number of tags expected to be found.
    """

    result = {tag for tag in tags if isinstance(tag, tag_type)}

    def strify_tag_type():
        if isinstance(tag_type, tuple):
            return ", ".join(t.__name__ for t in tag_type)
        else:
            return tag_type.__name__

    if max_num is not None:
        if len(result) > max_num:
            raise LoopyError("cannot have more than {} tags "
                    "of type(s): {}".format(max_num, strify_tag_type()))
    if min_num is not None:
        if len(result) < min_num:
            raise LoopyError("must have more than {} tags "
                    "of type(s): {}".format(max_num, strify_tag_type()))
    return result


class IndexTag(ImmutableRecord, UniqueTagBase):
    __slots__ = []

    def __hash__(self):
        return hash(self.key)

    def __lt__(self, other):
        return self.__hash__() < other.__hash__()

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        return key_builder.rec(key_hash, self.key)

    @property
    def key(self):
        """Return a hashable, comparable value that is used to ensure
        per-instruction uniqueness of this unique iname tag.

        Also used for persistent hash construction.
        """
        return type(self).__name__


class ConcurrentTag(IndexTag):
    pass


class HardwareConcurrentTag(ConcurrentTag):
    pass


# deprecated aliases
ParallelTag = ConcurrentTag
HardwareParallelTag = HardwareConcurrentTag


class UniqueTag(IndexTag):
    pass


class AxisTag(UniqueTag):
    __slots__ = ["axis"]

    def __init__(self, axis):
        ImmutableRecord.__init__(self,
                axis=axis)

    @property
    def key(self):
        return (type(self).__name__, self.axis)

    def __str__(self):
        return "%s.%d" % (
                self.print_name, self.axis)


class GroupIndexTag(HardwareConcurrentTag, AxisTag):
    print_name = "g"


class LocalIndexTagBase(HardwareConcurrentTag):
    pass


class LocalIndexTag(LocalIndexTagBase, AxisTag):
    print_name = "l"


class AutoLocalIndexTagBase(LocalIndexTagBase):
    @property
    def key(self):
        return type(self).__name__


class AutoFitLocalIndexTag(AutoLocalIndexTagBase):
    def __str__(self):
        return "l.auto"


# {{{ ilp-like

class IlpBaseTag(ConcurrentTag):
    pass


class UnrolledIlpTag(IlpBaseTag):
    def __str__(self):
        return "ilp.unr"


class LoopedIlpTag(IlpBaseTag):
    def __str__(self):
        return "ilp.seq"

# }}}


class VectorizeTag(UniqueTag, HardwareConcurrentTag):
    def __str__(self):
        return "vec"


class UnrollTag(IndexTag):
    def __str__(self):
        return "unr"


class ForceSequentialTag(IndexTag):
    def __str__(self):
        return "forceseq"


class InOrderSequentialSequentialTag(IndexTag):
    def __str__(self):
        return "ord"


def parse_tag(tag):
    from pytools.tag import Tag as TagBase
    if tag is None:
        return tag

    if isinstance(tag, TagBase):
        return tag

    if not isinstance(tag, str):
        raise ValueError("cannot parse tag: %s" % tag)

    if tag == "for":
        return None
    elif tag == "ord":
        return InOrderSequentialSequentialTag()
    elif tag in ["unr"]:
        return UnrollTag()
    elif tag in ["vec"]:
        return VectorizeTag()
    elif tag in ["ilp", "ilp.unr"]:
        return UnrolledIlpTag()
    elif tag == "ilp.seq":
        return LoopedIlpTag()
    elif tag.startswith("g."):
        return GroupIndexTag(int(tag[2:]))
    elif tag.startswith("l."):
        axis = tag[2:]
        if axis == "auto":
            return AutoFitLocalIndexTag()
        else:
            return LocalIndexTag(int(axis))
    else:
        raise ValueError("cannot parse tag: %s" % tag)

# }}}


# {{{ memory address space

class AddressSpace:
    """Storage location of a variable.

    .. attribute:: PRIVATE
    .. attribute:: LOCAL
    .. attribute:: GLOBAL
    """

    # These must occur in ascending order of 'globality' so that
    # max(scope) does the right thing.

    PRIVATE = 0
    LOCAL = 1
    GLOBAL = 2

    @classmethod
    def stringify(cls, val):
        if val == cls.PRIVATE:
            return "private"
        elif val == cls.LOCAL:
            return "local"
        elif val == cls.GLOBAL:
            return "global"
        else:
            raise ValueError("unexpected value of AddressSpace")


class _deprecated_temp_var_scope_class_method:  # noqa
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, klass):
        warn("'temp_var_scope' is deprecated. Use 'AddressSpace'.",
                DeprecationWarning, stacklevel=2)
        return self.f()


class temp_var_scope:  # noqa
    """Deprecated. Use :class:`loopy.AddressSpace` instead.
    """

    @_deprecated_temp_var_scope_class_method
    def PRIVATE():  # pylint:disable=no-method-argument
        return AddressSpace.PRIVATE

    @_deprecated_temp_var_scope_class_method
    def LOCAL():  # pylint:disable=no-method-argument
        return AddressSpace.LOCAL

    @_deprecated_temp_var_scope_class_method
    def GLOBAL():  # pylint:disable=no-method-argument
        return AddressSpace.GLOBAL

    @classmethod
    def stringify(cls, val):
        warn("'temp_var_scope' is deprecated. Use 'AddressSpace'.",
                DeprecationWarning, stacklevel=2)
        return AddressSpace.stringify(val)

# }}}


# {{{ arguments

class KernelArgument(ImmutableRecord):
    """Base class for all argument types"""

    def __init__(self, **kwargs):
        kwargs["name"] = intern(kwargs.pop("name"))

        target = kwargs.pop("target", None)

        dtype = kwargs.pop("dtype", None)

        if "for_atomic" in kwargs:
            for_atomic = kwargs["for_atomic"]
        else:
            for_atomic = False

        from loopy.types import to_loopy_type
        dtype = to_loopy_type(
                dtype, allow_auto=True, allow_none=True, for_atomic=for_atomic,
                target=target)

        import loopy as lp
        if dtype is lp.auto:
            warn("Argument/temporary data type for '%s' should be None if "
                   "unspecified, not auto. This usage will be disallowed in 2018."
                    % kwargs["name"],
                    DeprecationWarning, stacklevel=2)

            dtype = None
        kwargs["dtype"] = dtype
        kwargs["is_output"] = kwargs.pop("is_output", None)
        kwargs["is_input"] = kwargs.pop("is_input", None)

        ImmutableRecord.__init__(self, **kwargs)


class ArrayArg(ArrayBase, KernelArgument):
    __doc__ = ArrayBase.__doc__ + (
        """
        .. attribute:: address_space

            An attribute of :class:`AddressSpace` defining the address
            space in which the array resides.

        .. attribute:: is_output

            An instance of :class:`bool`. If set to *True*, the array is used to
            return information to the caller. If set to *False*, the callee does not
            write to the array during a call.

        .. attribute:: is_input

            An instance of :class:`bool`. If set to *True*, expected to be provided
            by the caller. If *False*, the callee does not depend on the array
            at kernel entry.
        """)

    allowed_extra_kwargs = [
            "address_space",
            "is_output",
            "is_input",
            "tags"]

    def __init__(self, *args, **kwargs):
        if "address_space" not in kwargs:
            raise TypeError("'address_space' must be specified")

        is_output_only = kwargs.pop("is_output_only", None)
        if is_output_only is not None:
            warn("'is_output_only' is deprecated. Use 'is_output', 'is_input'"
                    " instead.", DeprecationWarning, stacklevel=2)
            kwargs["is_output"] = is_output_only
            kwargs["is_input"] = not is_output_only
        else:
            kwargs["is_output"] = kwargs.pop("is_output", None)
            kwargs["is_input"] = kwargs.pop("is_input", None)

        super().__init__(*args, **kwargs)

    min_target_axes = 0
    max_target_axes = 1

    def get_arg_decl(self, ast_builder, name_suffix, shape, dtype, is_written):
        return ast_builder.get_array_arg_decl(self.name + name_suffix,
                self.address_space, shape, dtype, is_written)

    def __str__(self):
        # dont mention the type name if shape is known
        include_typename = self.shape in (None, auto)

        aspace_str = AddressSpace.stringify(self.address_space)

        return (
                self.stringify(include_typename=include_typename)
                +
                " aspace: %s" % aspace_str)

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """
        super().update_persistent_hash(key_hash, key_builder)
        key_builder.rec(key_hash, self.address_space)
        key_builder.rec(key_hash, self.is_output)
        key_builder.rec(key_hash, self.is_input)


# Making this a function prevents incorrect use in isinstance.
# Note: This is *not* deprecated, as it is super-common and
# incrementally more convenient to use than ArrayArg directly.
def GlobalArg(*args, **kwargs):
    address_space = kwargs.pop("address_space", None)
    if address_space is not None:
        raise TypeError("may not pass 'address_space' to GlobalArg")
    kwargs["address_space"] = AddressSpace.GLOBAL

    return ArrayArg(*args, **kwargs)


class ConstantArg(ArrayBase, KernelArgument):
    __doc__ = ArrayBase.__doc__

    def __init__(self, *args, **kwargs):
        if kwargs.pop("address_space", AddressSpace.GLOBAL) != AddressSpace.GLOBAL:
            raise LoopyError("'address_space' for ConstantArg must be GLOBAL.")
        super().__init__(*args, **kwargs)

    # Constant Arg cannot be an output
    is_output = False
    is_input = True
    address_space = AddressSpace.GLOBAL

    min_target_axes = 0
    max_target_axes = 1

    def get_arg_decl(self, ast_builder, name_suffix, shape, dtype, is_written):
        return ast_builder.get_constant_arg_decl(self.name + name_suffix, shape,
                dtype, is_written)


class ImageArg(ArrayBase, KernelArgument):
    __doc__ = ArrayBase.__doc__

    def __init__(self, *args, **kwargs):
        if kwargs.pop("address_space", AddressSpace.GLOBAL) != AddressSpace.GLOBAL:
            raise LoopyError("'address_space' for ImageArg must be GLOBAL.")
        super().__init__(*args, **kwargs)

    min_target_axes = 1
    max_target_axes = 3

    # ImageArg cannot be an output (for now)
    is_output = False
    is_input = True
    address_space = AddressSpace.GLOBAL

    @property
    def dimensions(self):
        return len(self.dim_tags)

    def get_arg_decl(self, ast_builder, name_suffix, shape, dtype, is_written):
        return ast_builder.get_image_arg_decl(self.name + name_suffix, shape,
                self.num_target_axes(), dtype, is_written)


"""
    :attribute tags: A (possibly empty) frozenset of instances of
        :class:`pytools.tag.Tag` intended for consumption by an
        application.

        ..versionadded: 2020.2.2
"""


class ValueArg(KernelArgument, Taggable):
    def __init__(self, name, dtype=None, approximately=1000, target=None,
            is_output=False, is_input=True, tags=None):
        """
        :arg tags: A an instance of or Iterable of instances of
            :class:`pytools.tag.Tag` intended for consumption by an
            application.
        """

        KernelArgument.__init__(self, name=name,
                dtype=dtype,
                approximately=approximately,
                target=target,
                is_output=is_output,
                is_input=is_input,
                tags=tags)

    def __str__(self):
        import loopy as lp
        assert self.dtype is not lp.auto

        if self.dtype is None:
            type_str = "<auto/runtime>"
        else:
            type_str = str(self.dtype)

        return f"{self.name}: ValueArg, type: {type_str}"

    def __repr__(self):
        return "<%s>" % self.__str__()

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        key_builder.rec(key_hash, self.name)
        key_builder.rec(key_hash, self.dtype)

    def get_arg_decl(self, ast_builder):
        return ast_builder.get_value_arg_decl(self.name, (),
                self.dtype, False)


class InameArg(ValueArg):
    pass

# }}}


# {{{ temporary variable


class TemporaryVariable(ArrayBase):
    __doc__ = ArrayBase.__doc__ + """
    .. attribute:: storage_shape
    .. attribute:: base_indices
    .. attribute:: address_space

        What memory this temporary variable lives in.
        One of the values in :class:`AddressSpace`,
        or :class:`loopy.auto` if this is
        to be automatically determined.

    .. attribute:: base_storage

        The name of a storage array that is to be used to actually
        hold the data in this temporary. Note that this storage
        array must not match any existing variable names.

    .. attribute:: initializer

        *None* or a :class:`numpy.ndarray` of data to be used to initialize the
        array.

    .. attribute:: read_only

        A :class:`bool` indicating whether the variable may be written during
        its lifetime. If *True*, *initializer* must be given.

    .. attribute:: _base_storage_access_may_be_aliasing

        Whether the temporary is used to alias the underlying base storage.
        Defaults to *False*. If *False*, C-based code generators will declare
        the temporary as a ``restrict`` const pointer to the base storage
        memory location. If *True*, the restrict part is omitted on this
        declaration.
    """

    min_target_axes = 0
    max_target_axes = 1

    allowed_extra_kwargs = [
            "storage_shape",
            "base_indices",
            "address_space",
            "base_storage",
            "initializer",
            "read_only",
            "_base_storage_access_may_be_aliasing",
            ]

    def __init__(self, name, dtype=None, shape=auto, address_space=None,
            dim_tags=None, offset=0, dim_names=None, strides=None, order=None,
            base_indices=None, storage_shape=None,
            base_storage=None, initializer=None, read_only=False,
            _base_storage_access_may_be_aliasing=False, **kwargs):
        """
        :arg dtype: :class:`loopy.auto` or a :class:`numpy.dtype`
        :arg shape: :class:`loopy.auto` or a shape tuple
        :arg base_indices: :class:`loopy.auto` or a tuple of base indices
        """

        scope = kwargs.pop("scope", None)
        if scope is not None:
            warn("Passing 'scope' is deprecated. Use 'address_space' instead.",
                    DeprecationWarning, stacklevel=2)

            if address_space is not None:
                raise ValueError("only one of 'scope' and 'address_space' "
                        "may be specified")
            else:
                address_space = scope

        del scope

        if address_space is None:
            address_space = auto

        if address_space is None:
            raise LoopyError(
                    "temporary variable '%s': "
                    "address_space must not be None"
                    % name)

        if initializer is None:
            pass
        elif isinstance(initializer, np.ndarray):
            if offset != 0:
                raise LoopyError(
                        "temporary variable '%s': "
                        "offset must be 0 if initializer specified"
                        % name)

            from loopy.types import NumpyType, to_loopy_type
            if dtype is auto or dtype is None:
                dtype = NumpyType(initializer.dtype)
            elif to_loopy_type(dtype) != to_loopy_type(initializer.dtype):
                raise LoopyError(
                        "temporary variable '%s': "
                        "dtype of initializer does not match "
                        "dtype of array."
                        % name)

            if shape is auto:
                shape = initializer.shape
            else:
                if shape != initializer.shape:
                    raise LoopyError("Shape of '{}' does not match that of the"
                            " initializer.".format(name))
        else:
            raise LoopyError(
                    "temporary variable '%s': "
                    "initializer must be None or a numpy array"
                    % name)

        if order is None:
            order = "C"

        if base_indices is None and shape is not auto:
            base_indices = (0,) * len(shape)

        if not read_only and initializer is not None:
            raise LoopyError(
                    "temporary variable '%s': "
                    "read-write variables with initializer "
                    "are not currently supported "
                    "(did you mean to set read_only=True?)"
                    % name)

        if base_storage is not None and initializer is not None:
            raise LoopyError(
                    "temporary variable '%s': "
                    "base_storage and initializer are "
                    "mutually exclusive"
                    % name)

        if base_storage is None and _base_storage_access_may_be_aliasing:
            raise LoopyError(
                    "temporary variable '%s': "
                    "_base_storage_access_may_be_aliasing option, but no "
                    "base_storage given!"
                    % name)

        ArrayBase.__init__(self, name=intern(name),
                dtype=dtype, shape=shape, strides=strides,
                dim_tags=dim_tags, offset=offset, dim_names=dim_names,
                order=order,
                base_indices=base_indices,
                address_space=address_space,
                storage_shape=storage_shape,
                base_storage=base_storage,
                initializer=initializer,
                read_only=read_only,
                _base_storage_access_may_be_aliasing=(
                    _base_storage_access_may_be_aliasing),
                **kwargs)

    @property
    def scope(self):
        warn("Use of 'TemporaryVariable.scope' is deprecated, "
                "use 'TemporaryVariable.address_space' instead.",
                DeprecationWarning, stacklevel=2)

        return self.address_space

    def copy(self, **kwargs):
        address_space = kwargs.pop("address_space", None)
        scope = kwargs.pop("scope", None)

        if scope is not None:
            warn("Passing 'scope' is deprecated. Use 'address_space' instead.",
                    DeprecationWarning, stacklevel=2)

            if address_space is not None:
                raise ValueError("only one of 'scope' and 'address_space' "
                        "may be specified")
            else:
                address_space = scope

        del scope

        if address_space is not None:
            kwargs["address_space"] = address_space

        return super().copy(**kwargs)

    @property
    def nbytes(self):
        shape = self.shape
        if self.storage_shape is not None:
            shape = self.storage_shape

        from pytools import product
        return product(si for si in shape)*self.dtype.itemsize

    def decl_info(self, target, index_dtype):
        return super().decl_info(
                target, is_written=True, index_dtype=index_dtype,
                shape_override=self.storage_shape)

    def get_arg_decl(self, ast_builder, name_suffix, shape, dtype, is_written):
        if self.address_space == AddressSpace.GLOBAL:
            return ast_builder.get_array_arg_decl(self.name + name_suffix,
                    AddressSpace.GLOBAL, shape, dtype, is_written)
        else:
            raise LoopyError("unexpected request for argument declaration of "
                    "non-global temporary")

    def __str__(self):
        if self.address_space is auto:
            scope_str = "auto"
        else:
            scope_str = AddressSpace.stringify(self.address_space)

        return (
                self.stringify(include_typename=False)
                +
                " scope:%s" % scope_str)

    def __eq__(self, other):
        return (
                super().__eq__(other)
                and self.storage_shape == other.storage_shape
                and self.base_indices == other.base_indices
                and self.address_space == other.address_space
                and self.base_storage == other.base_storage
                and (
                    (self.initializer is None and other.initializer is None)
                    or np.array_equal(self.initializer, other.initializer))
                and self.read_only == other.read_only
                and (self._base_storage_access_may_be_aliasing
                    == other._base_storage_access_may_be_aliasing)
                )

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        super().update_persistent_hash(key_hash, key_builder)
        self.update_persistent_hash_for_shape(key_hash, key_builder,
                self.storage_shape)
        key_builder.rec(key_hash, self.base_indices)
        key_builder.rec(key_hash, self.address_space)
        key_builder.rec(key_hash, self.base_storage)

        initializer = self.initializer
        if initializer is not None:
            initializer = (initializer.tolist(), initializer.dtype)
        key_builder.rec(key_hash, initializer)

        key_builder.rec(key_hash, self.read_only)
        key_builder.rec(key_hash, self._base_storage_access_may_be_aliasing)

# }}}


def iname_tag_to_temp_var_scope(iname_tag):
    iname_tag = parse_tag(iname_tag)

    if isinstance(iname_tag, GroupIndexTag):
        return AddressSpace.GLOBAL
    elif isinstance(iname_tag, LocalIndexTag):
        return AddressSpace.LOCAL
    else:
        return AddressSpace.PRIVATE


# {{{ substitution rule

class SubstitutionRule(ImmutableRecord):
    """
    .. attribute:: name
    .. attribute:: arguments

        A tuple of strings

    .. attribute:: expression
    """

    def __init__(self, name, arguments, expression):
        assert isinstance(arguments, tuple)

        ImmutableRecord.__init__(self,
                name=name, arguments=arguments, expression=expression)

    def __str__(self):
        return "{}({}) := {}".format(
                self.name, ", ".join(self.arguments), self.expression)

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        key_builder.rec(key_hash, self.name)
        key_builder.rec(key_hash, self.arguments)
        key_builder.update_for_pymbolic_expression(key_hash, self.expression)

# }}}


# {{{ function call mangling

class CallMangleInfo(ImmutableRecord):
    """
    .. attribute:: target_name

        A string. The name of the function to be called in the
        generated target code.

    .. attribute:: result_dtypes

        A tuple of :class:`loopy.types.LoopyType` instances indicating what
        types of values the function returns.

    .. attribute:: arg_dtypes

        A tuple of :class:`loopy.types.LoopyType` instances indicating what
        types of arguments the function actually receives.
    """

    def __init__(self, target_name, result_dtypes, arg_dtypes):
        assert isinstance(result_dtypes, tuple)

        super().__init__(
                target_name=target_name,
                result_dtypes=result_dtypes,
                arg_dtypes=arg_dtypes)

# }}}


# {{{ Iname class

class Iname(Taggable):
    """
    Records an iname in a :class:`~loopy.LoopKernel`. See :ref:`domain-tree` for
    semantics of *inames* in :mod:`loopy`.

    This class records the metadata attached to an iname as instances of
    :class:pytools.tag.Tag`. A tag maybe a builtin tag like
    :class:`loopy.kernel.data.IndexTag` or a user-defined custom tag. Custom tags
    may be attached to inames to be used in targeting later during transformations.

    .. attribute:: name

        An instance of :class:`str`, denoting the iname's name.

    .. attribute:: tas

        An instance of :class:`frozenset` of :class:`pytools.tag.Tag`.
    """
    def __init__(self, name, tags=frozenset()):
        super().__init__(tags=tags)

        assert isinstance(name, str)
        self.name = name

    def copy(self, *, name=None, tags=None):
        if name is None:
            name = self.name
        if tags is None:
            tags = self.tags

        return type(self)(name=name, tags=tags)

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """
        key_builder.rec(key_hash, type(self).__name__.encode("utf-8"))
        key_builder.rec(key_hash, self.name)
        key_builder.rec(key_hash, self.tags)

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.name == other.name
                and self.tags == other.tags)

# }}}

# vim: foldmethod=marker

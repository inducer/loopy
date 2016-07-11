"""Data used by the kernel object."""

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


from six.moves import intern
import numpy as np  # noqa
from pytools import Record, memoize_method
from loopy.kernel.array import ArrayBase
from loopy.diagnostic import LoopyError


class auto(object):  # noqa
    """A generic placeholder object for something that should be automatically
    detected.  See, for example, the *shape* or *strides* argument of
    :class:`GlobalArg`.
    """


# {{{ iname tags

class IndexTag(Record):
    __slots__ = []

    def __hash__(self):
        raise RuntimeError("use .key to hash index tags")

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


class ParallelTag(IndexTag):
    pass


class HardwareParallelTag(ParallelTag):
    pass


class UniqueTag(IndexTag):
    pass


class AxisTag(UniqueTag):
    __slots__ = ["axis"]

    def __init__(self, axis):
        Record.__init__(self,
                axis=axis)

    @property
    def key(self):
        return (type(self).__name__, self.axis)

    def __str__(self):
        return "%s.%d" % (
                self.print_name, self.axis)


class GroupIndexTag(HardwareParallelTag, AxisTag):
    print_name = "g"


class LocalIndexTagBase(HardwareParallelTag):
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

class IlpBaseTag(ParallelTag):
    pass


class UnrolledIlpTag(IlpBaseTag):
    def __str__(self):
        return "ilp.unr"


class LoopedIlpTag(IlpBaseTag):
    def __str__(self):
        return "ilp.seq"

# }}}


class VectorizeTag(UniqueTag):
    def __str__(self):
        return "vec"


class UnrollTag(IndexTag):
    def __str__(self):
        return "unr"


class ForceSequentialTag(IndexTag):
    def __str__(self):
        return "forceseq"


def parse_tag(tag):
    if tag is None:
        return tag

    if isinstance(tag, IndexTag):
        return tag

    if not isinstance(tag, str):
        raise ValueError("cannot parse tag: %s" % tag)

    if tag == "for":
        return None
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


# {{{ arguments

class KernelArgument(Record):
    """Base class for all argument types"""

    def __init__(self, **kwargs):
        kwargs["name"] = intern(kwargs.pop("name"))

        dtype = kwargs.pop("dtype", None)
        from loopy.types import to_loopy_type
        kwargs["dtype"] = to_loopy_type(
                dtype, allow_auto=True, allow_none=True)

        Record.__init__(self, **kwargs)


class GlobalArg(ArrayBase, KernelArgument):
    min_target_axes = 0
    max_target_axes = 1

    def get_arg_decl(self, ast_builder, name_suffix, shape, dtype, is_written):
        return ast_builder.get_global_arg_decl(self.name + name_suffix, shape,
                dtype, is_written)


class ConstantArg(ArrayBase, KernelArgument):
    min_target_axes = 0
    max_target_axes = 1

    def get_arg_decl(self, ast_builder, name_suffix, shape, dtype, is_written):
        return ast_builder.get_constant_arg_decl(self.name + name_suffix, shape,
                dtype, is_written)


class ImageArg(ArrayBase, KernelArgument):
    min_target_axes = 1
    max_target_axes = 3

    @property
    def dimensions(self):
        return len(self.dim_tags)

    def get_arg_decl(self, ast_builder, name_suffix, shape, dtype, is_written):
        return ast_builder.get_image_arg_decl(self.name + name_suffix, shape,
                self.num_target_axes(), dtype, is_written)


class ValueArg(KernelArgument):
    def __init__(self, name, dtype=None, approximately=1000):
        from loopy.types import to_loopy_type
        KernelArgument.__init__(self, name=name,
                dtype=to_loopy_type(dtype, allow_auto=True, allow_none=True),
                approximately=approximately)

    def __str__(self):
        import loopy as lp
        if self.dtype is lp.auto:
            type_str = "<auto>"
        elif self.dtype is None:
            type_str = "<runtime>"
        else:
            type_str = str(self.dtype)

        return "%s: ValueArg, type: %s" % (self.name, type_str)

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

class temp_var_scope:
    """Storage location of a temporary

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
            raise ValueError("unexpected value of temp_var_scope")


class TemporaryVariable(ArrayBase):
    __doc__ = ArrayBase.__doc__ + """
    .. attribute:: storage_shape
    .. attribute:: base_indices
    .. attribute:: scope

        What memory this temporary variable lives in.
        One of the values in :class:`temp_var_scope`,
        or :class:`loopy.auto` if this is
        to be automatically determined.

    .. attribute:: base_storage

        The name of a storage array that is to be used to actually
        hold the data in this temporary.

    .. attribute:: scope

        One of :class:`temp_var_scope`.

    .. attribute:: initializer

        *None* or a :class:`numpy.ndarray` of data to be used to initialize the
        array.

    .. attribute:: read_only

        A :class:`bool` indicating whether the variable may be written during
        its lifetime. If *True*, *initializer* must be given.
    """

    min_target_axes = 0
    max_target_axes = 1

    allowed_extra_kwargs = [
            "storage_shape",
            "base_indices",
            "scope",
            "base_storage",
            "initializer",
            "read_only",
            ]

    def __init__(self, name, dtype=None, shape=(), scope=auto,
            dim_tags=None, offset=0, dim_names=None, strides=None, order=None,
            base_indices=None, storage_shape=None,
            base_storage=None, initializer=None, read_only=False):
        """
        :arg dtype: :class:`loopy.auto` or a :class:`numpy.dtype`
        :arg shape: :class:`loopy.auto` or a shape tuple
        :arg base_indices: :class:`loopy.auto` or a tuple of base indices
        """

        if initializer is None:
            pass
        elif isinstance(initializer, np.ndarray):
            if offset != 0:
                raise LoopyError(
                        "temporary variable '%s': "
                        "offset must be 0 if initializer specified"
                        % name)

            if dtype is auto or dtype is None:
                from loopy.types import NumpyType
                dtype = NumpyType(initializer.dtype)
            elif dtype.numpy_dtype != initializer.dtype:
                raise LoopyError(
                        "temporary variable '%s': "
                        "dtype of initializer does not match "
                        "dtype of array."
                        % name)

            if shape is auto:
                shape = initializer.shape

        else:
            raise LoopyError(
                    "temporary variable '%s': "
                    "initializer must be None or a numpy array"
                    % name)

        if base_indices is None:
            base_indices = (0,) * len(shape)

        if (not read_only
                and initializer is not None
                and scope == temp_var_scope.GLOBAL):
            raise LoopyError(
                    "temporary variable '%s': "
                    "read-write global variables with initializer "
                    "are not currently supported "
                    "(did you mean to set read_only=True?)"
                    % name)

        if base_storage is not None and initializer is not None:
            raise LoopyError(
                    "temporary variable '%s': "
                    "base_storage and initializer are "
                    "mutually exclusive"
                    % name)

        ArrayBase.__init__(self, name=intern(name),
                dtype=dtype, shape=shape,
                dim_tags=dim_tags, offset=offset, dim_names=dim_names,
                order="C",
                base_indices=base_indices, scope=scope,
                storage_shape=storage_shape,
                base_storage=base_storage,
                initializer=initializer,
                read_only=read_only)

    @property
    def is_local(self):
        """One of :class:`loopy.temp_var_scope`."""

        if self.scope is auto:
            return auto
        elif self.scope == temp_var_scope.LOCAL:
            return True
        elif self.scope == temp_var_scope.PRIVATE:
            return False
        elif self.scope == temp_var_scope.GLOBAL:
            raise LoopyError("TemporaryVariable.is_local called on "
                    "global temporary variable '%s'" % self.name)
        else:
            raise LoopyError("unexpected value of TemporaryVariable.scope")

    @property
    def nbytes(self):
        shape = self.shape
        if self.storage_shape is not None:
            shape = self.storage_shape

        from pytools import product
        return product(si for si in shape)*self.dtype.itemsize

    def decl_info(self, target, index_dtype):
        return super(TemporaryVariable, self).decl_info(
                target, is_written=True, index_dtype=index_dtype,
                shape_override=self.storage_shape)

    def get_arg_decl(self, ast_builder, name_suffix, shape, dtype, is_written):
        if self.scope == temp_var_scope.GLOBAL:
            return ast_builder.get_global_arg_decl(self.name + name_suffix, shape,
                    dtype, is_written)
        else:
            raise LoopyError("unexpected request for argument declaration of "
                    "non-global temporary")

    def __str__(self):
        if self.scope is auto:
            scope_str = "auto"
        else:
            scope_str = temp_var_scope.stringify(self.scope)

        return (
                self.stringify(include_typename=False)
                +
                " scope:%s" % scope_str)

    def __eq__(self, other):
        return (
                super(TemporaryVariable, self).__eq__(other)
                and self.storage_shape == other.storage_shape
                and self.base_indices == other.base_indices
                and self.scope == other.scope
                and self.base_storage == other.base_storage
                and (
                    (self.initializer is None and other.initializer is None)
                    or np.array_equal(self.initializer, other.initializer))
                and self.read_only == other.read_only)

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        super(TemporaryVariable, self).update_persistent_hash(key_hash, key_builder)
        key_builder.rec(key_hash, self.storage_shape)
        key_builder.rec(key_hash, self.base_indices)

        initializer = self.initializer
        if initializer is not None:
            initializer = (initializer.tolist(), initializer.dtype)
        key_builder.rec(key_hash, initializer)

        key_builder.rec(key_hash, self.read_only)

# }}}


# {{{ subsitution rule

class SubstitutionRule(Record):
    """
    .. attribute:: name
    .. attribute:: arguments

        A tuple of strings

    .. attribute:: expression
    """

    def __init__(self, name, arguments, expression):
        assert isinstance(arguments, tuple)

        Record.__init__(self,
                name=name, arguments=arguments, expression=expression)

    def __str__(self):
        return "%s(%s) := %s" % (
                self.name, ", ".join(self.arguments), self.expression)

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        key_builder.rec(key_hash, self.name)
        key_builder.rec(key_hash, self.arguments)
        key_builder.update_for_pymbolic_expression(key_hash, self.expression)

# }}}


# {{{ instructions: base class

class InstructionBase(Record):
    """A base class for all types of instruction that can occur in
    a kernel.

    .. attribute:: id

        An (otherwise meaningless) identifier that is unique within
        a :class:`loopy.kernel.LoopKernel`.

    .. rubric:: Instruction ordering

    .. attribute:: depends_on

        a :class:`frozenset` of :attr:`id` values of :class:`Instruction` instances
        that *must* be executed before this one. Note that
        :func:`loopy.preprocess_kernel` (usually invoked automatically)
        augments this by adding dependencies on any writes to temporaries read
        by this instruction.

        May be *None* to invoke the default.

    .. attribute:: depends_on_is_final

        A :class:`bool` determining whether :attr:`depends_on` constitutes
        the *entire* list of iname dependencies.

        Defaults to *False*.

    .. attribute:: groups

        A :class:`frozenset` of strings indicating the names of 'instruction
        groups' of which this instruction is a part. An instruction group is
        considered 'active' as long as one (but not all) instructions of the
        group have been executed.

    .. attribute:: conflicts_with_groups

        A :class:`frozenset` of strings indicating which instruction groups
        (see :class:`InstructionBase.groups`) may not be active when this
        instruction is scheduled.

    .. attribute:: priority

        Scheduling priority, an integer. Higher means 'execute sooner'.
        Default 0.

    .. rubric :: Synchronization

    .. attribute:: no_sync_with

        a :class:`frozenset` of :attr:`id` values of :class:`Instruction` instances
        with which no barrier synchronization is necessary, even given the existence
        of a dependency chain and apparently conflicting access

    .. rubric:: Conditionals

    .. attribute:: predicates

        a :class:`frozenset` of variable names the conjunction (logical and) of
        whose truth values (as defined by C) determine whether this instruction
        should be run. Each variable name may, optionally, be preceded by
        an exclamation point, indicating negation.

    .. rubric:: Iname dependencies

    .. attribute:: forced_iname_deps_is_final

        A :class:`bool` determining whether :attr:`forced_iname_deps` constitutes
        the *entire* list of iname dependencies.

    .. attribute:: forced_iname_deps

        A :class:`frozenset` of inames that are added to the list of iname
        dependencies *or* constitute the entire list of iname dependencies,
        depending on the value of :attr:`forced_iname_deps_is_final`.

    .. rubric:: Iname dependencies

    .. attribute:: boostable

        Whether the instruction may safely be executed inside more loops than
        advertised without changing the meaning of the program. Allowed values
        are *None* (for unknown), *True*, and *False*.

    .. attribute:: boostable_into

        A :class:`set` of inames into which the instruction
        may need to be boosted, as a heuristic help for the scheduler.
        Also allowed to be *None* to indicate that this hasn't been
        decided yet.

    .. rubric:: Tagging

    .. attribute:: tags

        A :class:`frozenset` of string identifiers that can be used to
        identify groups of instructions.

    .. automethod:: __init__
    .. automethod:: assignee_var_names
    .. automethod:: assignee_subscript_deps
    .. automethod:: with_transformed_expressions
    .. automethod:: write_dependency_names
    .. automethod:: dependency_names
    .. automethod:: copy
    """

    fields = set("id depends_on depends_on_is_final "
            "groups conflicts_with_groups "
            "no_sync_with "
            "predicates "
            "forced_iname_deps_is_final forced_iname_deps "
            "priority boostable boostable_into".split())

    def __init__(self, id, depends_on, depends_on_is_final,
            groups, conflicts_with_groups,
            no_sync_with,
            forced_iname_deps_is_final, forced_iname_deps,
            priority,
            boostable, boostable_into, predicates, tags,
            insn_deps=None, insn_deps_is_final=None):

        if depends_on is not None and insn_deps is not None:
            raise ValueError("may not specify both insn_deps and depends_on")
        elif insn_deps is not None:
            from warnings import warn
            warn("insn_deps is deprecated, use depends_on",
                    DeprecationWarning, stacklevel=2)

            depends_on = insn_deps
            depends_on_is_final = insn_deps_is_final

        if depends_on is None:
            depends_on = frozenset()

        if groups is None:
            groups = frozenset()

        if conflicts_with_groups is None:
            conflicts_with_groups = frozenset()

        if no_sync_with is None:
            no_sync_with = frozenset()

        if forced_iname_deps_is_final is None:
            forced_iname_deps_is_final = False

        if depends_on_is_final is None:
            depends_on_is_final = False

        if depends_on_is_final and not isinstance(depends_on, frozenset):
            raise LoopyError("Setting depends_on_is_final to True requires "
                    "actually specifying depends_on")

        if tags is None:
            tags = frozenset()

        if not isinstance(tags, frozenset):
            # was previously allowed to be tuple
            tags = frozenset(tags)

        # Periodically reenable these and run the tests to ensure all
        # performance-relevant identifiers are interned.
        #
        # from loopy.tools import is_interned
        # assert is_interned(id)
        # assert all(is_interned(dep) for dep in depends_on)
        # assert all(is_interned(grp) for grp in groups)
        # assert all(is_interned(grp) for grp in conflicts_with_groups)
        # assert all(is_interned(iname) for iname in forced_iname_deps)
        # assert all(is_interned(pred) for pred in predicates)

        assert isinstance(forced_iname_deps, frozenset)
        assert isinstance(depends_on, frozenset) or depends_on is None
        assert isinstance(groups, frozenset)
        assert isinstance(conflicts_with_groups, frozenset)

        Record.__init__(self,
                id=id,
                depends_on=depends_on,
                depends_on_is_final=depends_on_is_final,
                no_sync_with=no_sync_with,
                groups=groups, conflicts_with_groups=conflicts_with_groups,
                forced_iname_deps_is_final=forced_iname_deps_is_final,
                forced_iname_deps=forced_iname_deps,
                priority=priority,
                boostable=boostable,
                boostable_into=boostable_into,
                predicates=predicates,
                tags=tags)

    # legacy
    @property
    def insn_deps(self):
        return self.depends_on

    # legacy
    @property
    def insn_deps_is_final(self):
        return self.depends_on_is_final

    # {{{ abstract interface

    def read_dependency_names(self):
        raise NotImplementedError

    def reduction_inames(self):
        raise NotImplementedError

    def assignee_var_names(self):
        """Return a tuple of tuples of assignee variable names, one
        for each quantity being assigned to.
        """
        raise NotImplementedError

    def assignee_subscript_deps(self):
        """Return a list of sets of variable names referred to in the subscripts
        of the quantities being assigned to, one for each assignee.
        """
        raise NotImplementedError

    def with_transformed_expressions(self, f, *args):
        """Return a new copy of *self* where *f* has been applied to every
        expression occurring in *self*. *args* will be passed as extra
        arguments (in addition to the expression) to *f*.
        """
        raise NotImplementedError

    # }}}

    @property
    def assignee_name(self):
        """A convenience wrapper around :meth:`assignee_var_names`
        that returns the the name of the variable being assigned.
        If more than one variable is being modified in the instruction,
        :raise:`ValueError` is raised.
        """

        names = self.assignee_var_names()

        if len(names) != 1:
            raise ValueError("expected exactly one assignment in instruction "
                    "on which assignee_name is being called, found %d"
                    % len(names))

        name, = names
        return name

    @memoize_method
    def write_dependency_names(self):
        """Return a set of dependencies of the left hand side of the
        assignments performed by this instruction, including written variables
        and indices.
        """

        result = set()
        for assignee in self.assignees:
            from loopy.symbolic import get_dependencies
            result.update(get_dependencies(assignee))

        return frozenset(result)

    def dependency_names(self):
        return self.read_dependency_names() | self.write_dependency_names()

    def get_str_options(self):
        result = []

        if self.boostable is True:
            if self.boostable_into:
                result.append("boostable into '%s'" % ",".join(self.boostable_into))
            else:
                result.append("boostable")
        elif self.boostable is False:
            result.append("not boostable")
        elif self.boostable is None:
            pass
        else:
            raise RuntimeError("unexpected value for Instruction.boostable")

        if self.depends_on:
            result.append("deps="+":".join(self.depends_on))
        if self.no_sync_with:
            result.append("nosync="+":".join(self.no_sync_with))
        if self.groups:
            result.append("groups=%s" % ":".join(self.groups))
        if self.conflicts_with_groups:
            result.append("conflicts=%s" % ":".join(self.conflicts_with_groups))
        if self.priority:
            result.append("priority=%d" % self.priority)
        if self.tags:
            result.append("tags=%s" % ":".join(self.tags))
        if hasattr(self, "atomicity") and self.atomicity:
            result.append("atomic=%s" % ":".join(str(a) for a in self.atomicity))

        return result

    # {{{ comparison, hashing

    def __eq__(self, other):
        if not type(self) == type(other):
            return False

        for field_name in self.fields:
            if getattr(self, field_name) != getattr(other, field_name):
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.

        Only works in conjunction with :class:`loopy.tools.KeyBuilder`.
        """

        # Order matters for hash forming--sort the field names
        for field_name in sorted(self.fields):
            key_builder.rec(key_hash, getattr(self, field_name))

    # }}}

    def copy(self, **kwargs):
        if "insn_deps" in kwargs:
            from warnings import warn
            warn("insn_deps is deprecated, use depends_on",
                    DeprecationWarning, stacklevel=2)

            kwargs["depends_on"] = kwargs.pop("insn_deps")

        if "insn_deps_is_final" in kwargs:
            from warnings import warn
            warn("insn_deps_is_final is deprecated, use depends_on",
                    DeprecationWarning, stacklevel=2)

            kwargs["depends_on_is_final"] = kwargs.pop("insn_deps_is_final")

        return super(InstructionBase, self).copy(**kwargs)

    def __setstate__(self, val):
        super(InstructionBase, self).__setstate__(val)

        from loopy.tools import intern_frozenset_of_ids

        self.id = intern(self.id)
        self.depends_on = intern_frozenset_of_ids(self.depends_on)
        self.groups = intern_frozenset_of_ids(self.groups)
        self.conflicts_with_groups = (
                intern_frozenset_of_ids(self.conflicts_with_groups))
        self.forced_iname_deps = (
                intern_frozenset_of_ids(self.forced_iname_deps))
        self.predicates = (
                intern_frozenset_of_ids(self.predicates))

# }}}


def _get_assignee_var_name(expr):
    from pymbolic.primitives import Variable, Subscript
    from loopy.symbolic import LinearSubscript

    if isinstance(expr, Variable):
        return expr.name

    elif isinstance(expr, Subscript):
        agg = expr.aggregate
        assert isinstance(agg, Variable)

        return agg.name

    elif isinstance(expr, LinearSubscript):
        agg = expr.aggregate
        assert isinstance(agg, Variable)

        return agg.name
    else:
        raise RuntimeError("invalid lvalue '%s'" % expr)


def _get_assignee_subscript_deps(expr):
    from pymbolic.primitives import Variable, Subscript
    from loopy.symbolic import LinearSubscript, get_dependencies

    if isinstance(expr, Variable):
        return frozenset()
    elif isinstance(expr, Subscript):
        return get_dependencies(expr.index)
    elif isinstance(expr, LinearSubscript):
        return get_dependencies(expr.index)
    else:
        raise RuntimeError("invalid lvalue '%s'" % expr)


# {{{ atomic ops

class memory_ordering:
    """Ordering of atomic operations, defined as in C11 and OpenCL.

    .. attribute:: relaxed
    .. attribute:: acquire
    .. attribute:: release
    .. attribute:: acq_rel
    .. attribute:: seq_cst
    """

    relaxed = 0
    acquire = 1
    release = 2
    acq_rel = 3
    seq_cst = 4

    @staticmethod
    def to_string(v):
        for i in dir(memory_ordering):
            if i.startswith("_"):
                continue

            if getattr(memory_ordering, i) == v:
                return i

        raise ValueError("Unknown value of memory_ordering")


class memory_scope:
    """Scope of atomicity, defined as in OpenCL.

    .. attribute:: auto

        Scope matches the accessibility of the variable.

    .. attribute:: work_item
    .. attribute:: work_group
    .. attribute:: work_device
    .. attribute:: all_svm_devices
    """

    work_item = 0
    work_group = 1
    device = 2
    all_svm_devices = 2

    auto = -1

    @staticmethod
    def to_string(v):
        for i in dir(memory_scope):
            if i.startswith("_"):
                continue

            if getattr(memory_scope, i) == v:
                return i

        raise ValueError("Unknown value of memory_scope")


class VarAtomicity(object):
    """A base class for the description of how atomic access to :attr:`var_name`
    shall proceed.

    .. attribute:: var_name
    """

    def __init__(self, var_name):
        self.var_name = var_name

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        key_builder.rec(key_hash, self.var_name)

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.var_name == other.var_name)

    def __ne__(self, other):
        return not self.__eq__(other)


class AtomicInit(VarAtomicity):
    """Describes initialization of an atomic variable. A subclass of
    :class:`VarAtomicity`.
    """

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        super(AtomicInit, self).update_persistent_hash(key_hash, key_builder)
        key_builder.rec(key_hash, "AtomicInit")

    def __str__(self):
        return "update[%s]%s/%s" % (
                self.var_name,
                memory_ordering.to_string(self.ordering),
                memory_scope.to_string(self.scope))


class AtomicUpdate(VarAtomicity):
    """Properties of an atomic operation. A subclass of :class:`VarAtomicity`.

    .. attribute:: ordering

        One of the values from :class:`memory_ordering`

    .. attribute:: scope

        One of the values from :class:`memory_scope`
    """

    ordering = memory_ordering.seq_cst
    scope = memory_scope.auto

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        super(AtomicUpdate, self).update_persistent_hash(key_hash, key_builder)
        key_builder.rec(key_hash, "AtomicUpdate")
        key_builder.rec(key_hash, self.ordering)
        key_builder.rec(key_hash, self.scope)

    def __eq__(self, other):
        return (super(AtomicUpdate, self).__eq__(other)
                and self.ordering == other.ordering
                and self.scope == other.scope)

    def __str__(self):
        return "update[%s]%s/%s" % (
                self.var_name,
                memory_ordering.to_string(self.ordering),
                memory_scope.to_string(self.scope))

# }}}


# {{{ instruction base class: expression rhs

class MultiAssignmentBase(InstructionBase):
    """An assignment instruction with an expression as a right-hand side."""

    fields = InstructionBase.fields | set(["expression"])

    @memoize_method
    def read_dependency_names(self):
        from loopy.symbolic import get_dependencies
        result = get_dependencies(self.expression)
        for subscript_deps in self.assignee_subscript_deps():
            result = result | subscript_deps

        processed_predicates = frozenset(
                pred.lstrip("!") for pred in self.predicates)

        result = result | processed_predicates

        return result

    @memoize_method
    def reduction_inames(self):
        def map_reduction(expr, rec):
            rec(expr.expr)
            for iname in expr.inames:
                result.add(iname)

        from loopy.symbolic import ReductionCallbackMapper
        cb_mapper = ReductionCallbackMapper(map_reduction)

        result = set()
        cb_mapper(self.expression)

        return result

# }}}


# {{{ instruction: assignment

class Assignment(MultiAssignmentBase):
    """
    .. attribute:: assignee

    .. attribute:: expression

    The following attributes are only used until
    :func:`loopy.make_kernel` is finished:

    .. attribute:: temp_var_type

        if not *None*, a type that will be assigned to the new temporary variable
        created from the assignee

    .. attribute:: atomicity

        A tuple of instances of :class:`VarAtomicity`. Together, they describe
        to what extent the assignment is to be carried out in a way that
        involves atomic operations.

        To describe an atomic update, any memory reads of *exact* occurrences
        of the left-hand side expression of the assignment in the right hand
        side are treated , together with the "memory write" part of the
        assignment, as part of one single atomic update.

        .. note::

            Exact identity of the LHS with RHS subexpressions is required for
            an atomic update to be recognized. For example, the following update
            will not be recognized as an update::

                z[i] = z[i+1-1] + a {atomic}

        :mod:`loopy` may to evaluate the right-hand side *multiple times*
        as part of a single assignment. It is up to the user to ensure that
        this retains correct semantics.

        For example, the following assignment::

            z[i] = f(z[i]) + a {atomic}

        may generate the following (pseudo-)code::

            DO
                READ ztemp_old = z[i]
                EVALUATE ztemp_new = f(ztemp_old) + a
            WHILE compare_and_swap(z[i], ztemp_new, ztemp_old) did not succeed

    .. automethod:: __init__
    """

    fields = MultiAssignmentBase.fields | \
            set("assignee temp_var_type atomicity".split())

    def __init__(self,
            assignee, expression,
            id=None,
            depends_on=None,
            depends_on_is_final=None,
            groups=None,
            conflicts_with_groups=None,
            no_sync_with=None,
            forced_iname_deps_is_final=None,
            forced_iname_deps=frozenset(),
            boostable=None, boostable_into=None, tags=None,
            temp_var_type=None, atomicity=(),
            priority=0, predicates=frozenset(),
            insn_deps=None, insn_deps_is_final=None):

        super(Assignment, self).__init__(
                id=id,
                depends_on=depends_on,
                depends_on_is_final=depends_on_is_final,
                groups=groups,
                conflicts_with_groups=conflicts_with_groups,
                no_sync_with=no_sync_with,
                forced_iname_deps_is_final=forced_iname_deps_is_final,
                forced_iname_deps=forced_iname_deps,
                boostable=boostable,
                boostable_into=boostable_into,
                priority=priority,
                predicates=predicates,
                tags=tags,
                insn_deps=insn_deps,
                insn_deps_is_final=insn_deps_is_final)

        from loopy.symbolic import parse
        if isinstance(assignee, str):
            assignee = parse(assignee)
        if isinstance(expression, str):
            expression = parse(expression)

        # FIXME: It may be worth it to enable this check eventually.
        # For now, it causes grief with certain 'checky' uses of the
        # with_transformed_expressions(). (notably the access checker)
        #
        # from pymbolic.primitives import Variable, Subscript
        # if not isinstance(assignee, (Variable, Subscript)):
        #     raise LoopyError("invalid lvalue '%s'" % assignee)

        self.assignee = assignee
        self.expression = expression
        self.temp_var_type = temp_var_type
        self.atomicity = atomicity

    # {{{ implement InstructionBase interface

    @memoize_method
    def assignee_var_names(self):
        return (_get_assignee_var_name(self.assignee),)

    def assignee_subscript_deps(self):
        return (_get_assignee_subscript_deps(self.assignee),)

    def with_transformed_expressions(self, f, *args):
        return self.copy(
                assignee=f(self.assignee, *args),
                expression=f(self.expression, *args))

    # }}}

    def __str__(self):
        result = "%s: %s <- %s" % (self.id,
                self.assignee, self.expression)

        options = self.get_str_options()
        if options:
            result += " (%s)" % (": ".join(options))

        if self.predicates:
            result += "\n" + 10*" " + "if (%s)" % " && ".join(self.predicates)
        return result

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.

        Only works in conjunction with :class:`loopy.tools.KeyBuilder`.
        """

        # Order matters for hash forming--sort the fields.
        for field_name in sorted(self.fields):
            if field_name in ["assignee", "expression"]:
                key_builder.update_for_pymbolic_expression(
                        key_hash, getattr(self, field_name))
            else:
                key_builder.rec(key_hash, getattr(self, field_name))

    # {{{ for interface uniformity with CallInstruction

    @property
    def temp_var_types(self):
        return (self.temp_var_type,)

    @property
    def assignees(self):
        return (self.assignee,)

    # }}}


class ExpressionInstruction(Assignment):
    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("ExpressionInstruction is deprecated. Use Assignment instead",
                DeprecationWarning, stacklevel=2)

        super(ExpressionInstruction, self).__init__(*args, **kwargs)

# }}}


# {{{ instruction: function call

class CallInstruction(MultiAssignmentBase):
    """An instruction capturing a function call. Unlike :class:`Assignment`,
    this instruction supports functions with multiple return values.

    .. attribute:: assignees

    .. attribute:: expression

    The following attributes are only used until
    :func:`loopy.make_kernel` is finished:

    .. attribute:: temp_var_types

        if not *None*, a type that will be assigned to the new temporary variable
        created from the assignee

    .. automethod:: __init__
    """

    fields = MultiAssignmentBase.fields | \
            set("assignees temp_var_types".split())

    def __init__(self,
            assignees, expression,
            id=None,
            depends_on=None,
            depends_on_is_final=None,
            groups=None,
            conflicts_with_groups=None,
            no_sync_with=None,
            forced_iname_deps_is_final=None,
            forced_iname_deps=frozenset(),
            boostable=None, boostable_into=None, tags=None,
            temp_var_types=None,
            priority=0, predicates=frozenset(),
            insn_deps=None, insn_deps_is_final=None):

        super(CallInstruction, self).__init__(
                id=id,
                depends_on=depends_on,
                depends_on_is_final=depends_on_is_final,
                groups=groups,
                conflicts_with_groups=conflicts_with_groups,
                no_sync_with=no_sync_with,
                forced_iname_deps_is_final=forced_iname_deps_is_final,
                forced_iname_deps=forced_iname_deps,
                boostable=boostable,
                boostable_into=boostable_into,
                priority=priority,
                predicates=predicates,
                tags=tags,
                insn_deps=insn_deps,
                insn_deps_is_final=insn_deps_is_final)

        from pymbolic.primitives import Call
        from loopy.symbolic import Reduction
        if not isinstance(expression, (Call, Reduction)) and expression is not None:
            raise LoopyError("'expression' argument to CallInstruction "
                    "must be a function call")

        from loopy.symbolic import parse
        if isinstance(assignees, str):
            assignees = parse(assignees)
        if isinstance(expression, str):
            expression = parse(expression)

        # FIXME: It may be worth it to enable this check eventually.
        # For now, it causes grief with certain 'checky' uses of the
        # with_transformed_expressions(). (notably the access checker)
        #
        # from pymbolic.primitives import Variable, Subscript
        # if not isinstance(assignee, (Variable, Subscript)):
        #     raise LoopyError("invalid lvalue '%s'" % assignee)

        self.assignees = assignees
        self.expression = expression
        self.temp_var_types = temp_var_types

    # {{{ implement InstructionBase interface

    @memoize_method
    def assignee_var_names(self):
        return tuple(_get_assignee_var_name(a) for a in self.assignees)

    def assignee_subscript_deps(self):
        return tuple(
                _get_assignee_subscript_deps(a)
                for a in self.assignees)

    def with_transformed_expressions(self, f, *args):
        return self.copy(
                assignees=f(self.assignees, *args),
                expression=f(self.expression, *args))

    # }}}

    def __str__(self):
        result = "%s: %s <- %s" % (self.id,
                ", ".join(str(a) for a in self.assignees),
                self.expression)

        options = self.get_str_options()
        if options:
            result += " (%s)" % (": ".join(options))

        if self.predicates:
            result += "\n" + 10*" " + "if (%s)" % " && ".join(self.predicates)
        return result

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.

        Only works in conjunction with :class:`loopy.tools.KeyBuilder`.
        """

        # Order matters for hash forming--sort the fields.
        for field_name in sorted(self.fields):
            if field_name in ["assignees", "expression"]:
                key_builder.update_for_pymbolic_expression(
                        key_hash, getattr(self, field_name))
            else:
                key_builder.rec(key_hash, getattr(self, field_name))

# }}}


def make_assignment(assignees, expression, temp_var_types=None, **kwargs):
    if len(assignees) < 1:
        raise LoopyError("every instruction must have a left-hand side")
    elif len(assignees) > 1:
        atomicity = kwargs.pop("atomicity", ())
        if atomicity:
            raise LoopyError("atomic operations with more than one "
                    "left-hand side not supported")

        from pymbolic.primitives import Call
        from loopy.symbolic import Reduction
        if not isinstance(expression, (Call, Reduction)):
            raise LoopyError("right-hand side in multiple assignment must be "
                    "function call or reduction")

        return CallInstruction(
                assignees=assignees,
                expression=expression,
                temp_var_types=temp_var_types,
                **kwargs)

    else:
        return Assignment(
                assignee=assignees[0],
                expression=expression,
                temp_var_type=(
                    temp_var_types[0]
                    if temp_var_types is not None
                    else None),
                **kwargs)


# {{{ c instruction

class CInstruction(InstructionBase):
    """
    .. attribute:: iname_exprs

        A list of tuples *(name, expr)* of inames or expressions based on them
        that the instruction needs access to.

    .. attribute:: code

        The C code to be executed.

        The code should obey the following rules:

        * It should only write to temporary variables, specifically the
          temporary variables

        .. note::

            Of course, nothing in :mod:`loopy` will prevent you from doing
            'forbidden' things in your C code. If you ignore the rules and
            something breaks, you get to keep both pieces.

    .. attribute:: read_variables

        A :class:`frozenset` of variable names that :attr:`code` reads. This is
        optional and only used for figuring out dependencies.

    .. attribute:: assignees

        A sequence (typically a :class:`tuple`) of variable references (with or
        without subscript) as :class:`pymbolic.primitives.Expression` instances
        that :attr:`code` writes to. This is optional and only used for
        figuring out dependencies.
    """

    fields = InstructionBase.fields | \
            set("iname_exprs code read_variables assignees".split())

    def __init__(self,
            iname_exprs, code,
            read_variables=frozenset(), assignees=tuple(),
            id=None, depends_on=None, depends_on_is_final=None,
            groups=None, conflicts_with_groups=None,
            no_sync_with=None,
            forced_iname_deps_is_final=None, forced_iname_deps=frozenset(),
            priority=0, boostable=None, boostable_into=None,
            predicates=frozenset(), tags=None,
            insn_deps=None, insn_deps_is_final=None):
        """
        :arg iname_exprs: Like :attr:`iname_exprs`, but instead of tuples,
            simple strings pepresenting inames are also allowed. A single
            string is also allowed, which should consists of comma-separated
            inames.
        :arg assignees: Like :attr:`assignees`, but may also be a
            semicolon-separated string of such expressions or a
            sequence of strings parseable into the desired format.
        """

        InstructionBase.__init__(self,
                id=id,
                depends_on=depends_on,
                depends_on_is_final=depends_on_is_final,
                groups=groups, conflicts_with_groups=conflicts_with_groups,
                no_sync_with=no_sync_with,
                forced_iname_deps_is_final=forced_iname_deps_is_final,
                forced_iname_deps=forced_iname_deps,
                boostable=boostable,
                boostable_into=boostable_into,
                priority=priority, predicates=predicates, tags=tags,
                insn_deps=insn_deps,
                insn_deps_is_final=insn_deps_is_final)

        # {{{ normalize iname_exprs

        if isinstance(iname_exprs, str):
            iname_exprs = [i.strip() for i in iname_exprs.split(",")]
            iname_exprs = [i for i in iname_exprs if i]

        from pymbolic import var
        new_iname_exprs = []
        for i in iname_exprs:
            if isinstance(i, str):
                new_iname_exprs.append((i, var(i)))
            else:
                new_iname_exprs.append(i)

        # }}}

        # {{{ normalize assignees

        if isinstance(assignees, str):
            assignees = [i.strip() for i in assignees.split(";")]
            assignees = [i for i in assignees if i]

        new_assignees = []
        from loopy.symbolic import parse
        for i in assignees:
            if isinstance(i, str):
                new_assignees.append(parse(i))
            else:
                new_assignees.append(i)
        # }}}

        self.iname_exprs = new_iname_exprs
        from loopy.tools import remove_common_indentation
        self.code = remove_common_indentation(code)
        self.read_variables = read_variables
        self.assignees = new_assignees

    # {{{ abstract interface

    def read_dependency_names(self):
        result = set(self.read_variables)

        from loopy.symbolic import get_dependencies
        for name, iname_expr in self.iname_exprs:
            result.update(get_dependencies(iname_expr))

        for subscript_deps in self.assignee_subscript_deps():
            result.update(subscript_deps)

        return frozenset(result) | self.predicates

    def reduction_inames(self):
        return set()

    def assignee_var_names(self):
        return tuple(_get_assignee_var_name(expr) for expr in self.assignees)

    def assignee_subscript_deps(self):
        return tuple(
                _get_assignee_subscript_deps(a)
                for a in self.assignees)

    def with_transformed_expressions(self, f, *args):
        return self.copy(
                iname_exprs=[
                    (name, f(expr, *args))
                    for name, expr in self.iname_exprs],
                assignees=[f(a, *args) for a in self.assignees])

    # }}}

    def __str__(self):
        first_line = "%s: %s <- CODE(%s|%s)" % (self.id,
                ", ".join(str(a) for a in self.assignees),
                ", ".join(str(x) for x in self.read_variables),
                ", ".join("%s=%s" % (name, expr)
                    for name, expr in self.iname_exprs))

        options = self.get_str_options()
        if options:
            first_line += " (%s)" % (": ".join(options))

        return first_line + "\n    " + "\n    ".join(
                self.code.split("\n"))

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.

        Only works in conjunction with :class:`loopy.tools.KeyBuilder`.
        """

        # Order matters for hash forming--sort the fields.
        for field_name in sorted(self.fields):
            if field_name == "assignees":
                for a in self.assignees:
                    key_builder.update_for_pymbolic_expression(key_hash, a)
            elif field_name == "iname_exprs":
                for name, val in self.iname_exprs:
                    key_builder.rec(key_hash, name)
                    key_builder.update_for_pymbolic_expression(key_hash, val)
            else:
                key_builder.rec(key_hash, getattr(self, field_name))

# }}}


# {{{ function call mangling

class CallMangleInfo(Record):
    """
    .. attribute:: target_name

        A string. The name of the function to be called in the
        generated target code.

    .. attribute:: result_dtypes

        A tuple of :class:`LoopyType` instances indicating what
        types of values the function returns.

    .. attribute:: arg_dtypes

        A tuple of :class:`LoopyType` instances indicating what
        types of arguments the function actually receives.
    """

    def __init__(self, target_name, result_dtypes, arg_dtypes):
        assert isinstance(result_dtypes, tuple)

        super(CallMangleInfo, self).__init__(
                target_name=target_name,
                result_dtypes=result_dtypes,
                arg_dtypes=arg_dtypes)

# }}}

# vim: foldmethod=marker

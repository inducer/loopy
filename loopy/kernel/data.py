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
import numpy as np
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
    def __init__(self, **kwargs):
        kwargs["name"] = intern(kwargs.pop("name"))

        dtype = kwargs.pop("dtype", None)

        if isinstance(dtype, np.dtype):
            from loopy.tools import PicklableDtype
            kwargs["picklable_dtype"] = PicklableDtype(dtype)
        else:
            kwargs["picklable_dtype"] = dtype

        Record.__init__(self, **kwargs)

    def get_copy_kwargs(self, **kwargs):
        result = Record.get_copy_kwargs(self, **kwargs)
        if "dtype" not in result:
            result["dtype"] = self.dtype

        del result["picklable_dtype"]

        return result

    @property
    def dtype(self):
        from loopy.tools import PicklableDtype
        if isinstance(self.picklable_dtype, PicklableDtype):
            return self.picklable_dtype.dtype
        else:
            return self.picklable_dtype


class GlobalArg(ArrayBase, KernelArgument):
    min_target_axes = 0
    max_target_axes = 1

    def get_arg_decl(self, target, name_suffix, shape, dtype, is_written):
        return target.get_global_arg_decl(self.name + name_suffix, shape,
                dtype, is_written)


class ConstantArg(ArrayBase, KernelArgument):
    min_target_axes = 0
    max_target_axes = 1

    def get_arg_decl(self, target, name_suffix, shape, dtype, is_written):
        return target.get_constant_arg_decl(self.name + name_suffix, shape,
                dtype, is_written)


class ImageArg(ArrayBase, KernelArgument):
    min_target_axes = 1
    max_target_axes = 3

    @property
    def dimensions(self):
        return len(self.dim_tags)

    def get_arg_decl(self, target, name_suffix, shape, dtype, is_written):
        return target.get_image_arg_decl(self.name + name_suffix, shape,
                self.num_target_axes(), dtype, is_written)


class ValueArg(KernelArgument):
    def __init__(self, name, dtype=None, approximately=1000):
        from loopy.tools import PicklableDtype
        if dtype is not None and not isinstance(dtype, PicklableDtype):
            dtype = np.dtype(dtype)

        KernelArgument.__init__(self, name=name, dtype=dtype,
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

    def get_arg_decl(self, target):
        return target.get_value_arg_decl(self.name, (),
                self.dtype, False)

# }}}


# {{{ temporary variable

class TemporaryVariable(ArrayBase):
    __doc__ = ArrayBase.__doc__ + """
    .. attribute:: storage_shape
    .. attribute:: base_indices
    .. attribute:: is_local

        Whether this is temporary lives in ``local`` memory.
        May be *True*, *False*, or :class:`loopy.auto` if this is
        to be automatically determined.

    .. attribute:: base_storage

        The name of a storage array that is to be used to actually
        hold the data in this temporary.
    """

    min_target_axes = 0
    max_target_axes = 1

    allowed_extra_kwargs = [
            "storage_shape",
            "base_indices",
            "is_local",
            "base_storage"
            ]

    def __init__(self, name, dtype=None, shape=(), is_local=auto,
            dim_tags=None, offset=0, dim_names=None, strides=None, order=None,
            base_indices=None, storage_shape=None,
            base_storage=None):
        """
        :arg dtype: :class:`loopy.auto` or a :class:`numpy.dtype`
        :arg shape: :class:`loopy.auto` or a shape tuple
        :arg base_indices: :class:`loopy.auto` or a tuple of base indices
        """

        if is_local is None:
            raise ValueError("is_local is None is no longer supported. "
                    "Use loopy.auto.")

        if base_indices is None:
            base_indices = (0,) * len(shape)

        ArrayBase.__init__(self, name=intern(name),
                dtype=dtype, shape=shape,
                dim_tags=dim_tags, offset=offset, dim_names=dim_names,
                order="C",
                base_indices=base_indices, is_local=is_local,
                storage_shape=storage_shape,
                base_storage=base_storage)

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

    def get_arg_decl(self, target, name_suffix, shape, dtype, is_written):
        return None

    def __str__(self):
        return self.stringify(include_typename=False)

    def __eq__(self, other):
        return (
                super(TemporaryVariable, self).__eq__(other)
                and self.storage_shape == other.storage_shape
                and self.base_indices == other.base_indices
                and self.is_local == other.is_local
                and self.base_storage == other.base_storage)

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        super(TemporaryVariable, self).update_persistent_hash(key_hash, key_builder)
        key_builder.rec(key_hash, self.storage_shape)
        key_builder.rec(key_hash, self.base_indices)
        key_builder.rec(key_hash, self.is_local)

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


# {{{ instruction

# {{{ base class

class InstructionBase(Record):
    """
    .. attribute:: id

        An (otherwise meaningless) identifier that is unique within
        a :class:`loopy.kernel.LoopKernel`.

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

    .. attribute:: predicates

        a :class:`frozenset` of variable names the conjunction (logical and) of
        whose truth values (as defined by C) determine whether this instruction
        should be run. Each variable name may, optionally, be preceded by
        an exclamation point, indicating negation.

    .. attribute:: forced_iname_deps_is_final

        A :class:`bool` determining whether :attr:`forced_iname_deps` constitutes
        the *entire* list of iname dependencies.

    .. attribute:: forced_iname_deps

        A :class:`frozenset` of inames that are added to the list of iname
        dependencies *or* constitute the entire list of iname dependencies,
        depending on the value of :attr:`forced_iname_deps_is_final`.

    .. attribute:: priority

        Scheduling priority, an integer. Higher means 'execute sooner'.
        Default 0.

    .. attribute:: boostable

        Whether the instruction may safely be executed inside more loops than
        advertised without changing the meaning of the program. Allowed values
        are *None* (for unknown), *True*, and *False*.

    .. attribute:: boostable_into

        A :class:`set` of inames into which the instruction
        may need to be boosted, as a heuristic help for the scheduler.
        Also allowed to be *None*.

    .. attribute:: tags

        A tuple of string identifiers that can be used to identify groups
        of instructions.
    """

    fields = set("id depends_on depends_on_is_final "
            "groups conflicts_with_groups "
            "predicates "
            "forced_iname_deps_is_final forced_iname_deps "
            "priority boostable boostable_into".split())

    def __init__(self, id, depends_on, depends_on_is_final,
            groups, conflicts_with_groups,
            forced_iname_deps_is_final, forced_iname_deps, priority,
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

        if forced_iname_deps_is_final is None:
            forced_iname_deps_is_final = False

        if depends_on_is_final is None:
            depends_on_is_final = False

        if depends_on_is_final and not isinstance(depends_on, frozenset):
            raise LoopyError("Setting depends_on_is_final to True requires "
                    "actually specifying depends_on")

        if tags is None:
            tags = ()

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
                groups=groups, conflicts_with_groups=conflicts_with_groups,
                forced_iname_deps_is_final=forced_iname_deps_is_final,
                forced_iname_deps=forced_iname_deps,
                priority=priority,
                boostable=boostable,
                boostable_into=boostable_into,
                predicates=predicates,
                tags=tags)

    @property
    def insn_deps(self):
        return self.depends_on

    @property
    def insn_deps_is_final(self):
        return self.depends_on_is_final

    # {{{ abstract interface

    def read_dependency_names(self):
        raise NotImplementedError

    def reduction_inames(self):
        raise NotImplementedError

    def assignees_and_indices(self):
        """Return a list of tuples *(assignee_var_name, subscript)*
        where assignee_var_name is a string representing an assigned
        variable name and subscript is a :class:`tuple`.
        """
        raise NotImplementedError

    def with_transformed_expressions(self, f, *args):
        """Return a new copy of *self* where *f* has been applied to every
        expression occurring in *self*. *args* will be passed as extra
        arguments (in addition to the expression) to *f*.
        """
        raise NotImplementedError

    # }}}

    @memoize_method
    def write_dependency_names(self):
        """Return a set of dependencies of the left hand side of the
        assignments performed by this instruction, including written variables
        and indices.
        """

        result = set()
        for assignee, indices in self.assignees_and_indices():
            result.add(assignee)
            from loopy.symbolic import get_dependencies
            result.update(get_dependencies(indices))

        return frozenset(result)

    def dependency_names(self):
        return self.read_dependency_names() | self.write_dependency_names()

    def assignee_var_names(self):
        return (var_name for var_name, _ in self.assignees_and_indices())

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
        if self.groups:
            result.append("groups=%s" % ":".join(self.groups))
        if self.conflicts_with_groups:
            result.append("conflicts=%s" % ":".join(self.conflicts_with_groups))
        if self.priority:
            result.append("priority=%d" % self.priority)
        if self.tags:
            result.append("tags=%s" % ":".join(self.tags))

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


def _get_assignee_and_index(expr):
    from pymbolic.primitives import Variable, Subscript
    if isinstance(expr, Variable):
        return (expr.name, ())
    elif isinstance(expr, Subscript):
        agg = expr.aggregate
        assert isinstance(agg, Variable)

        return (agg.name, expr.index_tuple)
    else:
        raise RuntimeError("invalid lvalue '%s'" % expr)


# {{{ assignment

class Assignment(InstructionBase):
    """
    .. attribute:: assignee

    .. attribute:: expression

    The following attributes are only used until
    :func:`loopy.make_kernel` is finished:

    .. attribute:: temp_var_type

        if not *None*, a type that will be assigned to the new temporary variable
        created from the assignee
    """

    fields = InstructionBase.fields | \
            set("assignee expression temp_var_type".split())

    def __init__(self,
            assignee, expression,
            id=None,
            depends_on=None,
            depends_on_is_final=None,
            groups=None,
            conflicts_with_groups=None,
            forced_iname_deps_is_final=None,
            forced_iname_deps=frozenset(),
            boostable=None, boostable_into=None, tags=None,
            temp_var_type=None, priority=0, predicates=frozenset(),
            insn_deps=None, insn_deps_is_final=None):

        InstructionBase.__init__(self,
                id=id,
                depends_on=depends_on,
                depends_on_is_final=depends_on_is_final,
                groups=groups,
                conflicts_with_groups=conflicts_with_groups,
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
            assignee = parse(expression)

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

    # {{{ implement InstructionBase interface

    @memoize_method
    def read_dependency_names(self):
        from loopy.symbolic import get_dependencies
        result = get_dependencies(self.expression)
        for _, subscript in self.assignees_and_indices():
            result = result | get_dependencies(subscript)

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

    @memoize_method
    def assignees_and_indices(self):
        return [_get_assignee_and_index(self.assignee)]

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


class ExpressionInstruction(Assignment):
    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("ExpressionInstruction is deprecated. Use Assignment instead",
                DeprecationWarning, stacklevel=2)

        super(ExpressionInstruction, self).__init__(*args, **kwargs)

# }}}


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

        A sequence of variable references (with or without subscript) as
        :class:`pymbolic.primitives.Expression` instances that :attr:`code`
        writes to. This is optional and only used for figuring out dependencies.
    """

    fields = InstructionBase.fields | \
            set("iname_exprs code read_variables assignees".split())

    def __init__(self,
            iname_exprs, code,
            read_variables=frozenset(), assignees=frozenset(),
            id=None, depends_on=None, depends_on_is_final=None,
            groups=None, conflicts_with_groups=None,
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

        for _, subscript in self.assignees_and_indices():
            result.update(get_dependencies(subscript))

        return frozenset(result) | self.predicates

    def reduction_inames(self):
        return set()

    def assignees_and_indices(self):
        return [_get_assignee_and_index(expr)
                for expr in self.assignees]

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

# }}}

# vim: foldmethod=marker

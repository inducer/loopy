from __future__ import annotations


__copyright__ = "Copyright (C) 2016 Andreas Kloeckner"

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

from collections.abc import (
    Callable,
    Mapping,
    Mapping as abc_Mapping,
    Sequence,
    Set as abc_Set,
)
from dataclasses import dataclass
from functools import cached_property
from sys import intern
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    TypeAlias,
    cast,
)
from warnings import warn

from typing_extensions import Self, override

import islpy as isl
import pymbolic.primitives as p
from pytools import ImmutableRecord, memoize_method
from pytools.tag import Tag, Taggable, tag_dataclass

from loopy.diagnostic import LoopyError
from loopy.symbolic import LinearSubscript, SubArrayRef
from loopy.tools import Optional as LoopyOptional
from loopy.types import LoopyType, ToLoopyTypeConvertible, to_loopy_type


if TYPE_CHECKING:
    from pymbolic import Expression

    from loopy.kernel import LoopKernel
    from loopy.typing import InameStr


Assignable: TypeAlias = (
    p.Variable
    | p.Subscript
    | p.Lookup
    | SubArrayRef
    | LinearSubscript)


# {{{ instruction tags

@tag_dataclass
class LegacyStringInstructionTag(Tag):
    """A subclass of :class:`pytools.tag.Tag` for use in
    :attr:`InstructionBase.tags` used for forward compatibility of the old
    string-based tagging mechanism. String-based tags are automatically converted
    to this type.

    .. attribute:: value
    """
    value: str

    # FIXME: This class should be deprecated as soon as there is a viable
    # alternative. For now, pattern matching and the textual syntax are
    # only able to generate string tags, which is why the deprecation is not
    # yet in effect.


@tag_dataclass
class UseStreamingStoreTag(Tag):
    """A subclass of :class:`pytools.tag.Tag` for use in
    :attr:`InstructionBase.tags` used to indicate that if the instruction is an
    :class:`Assignment` or a :class:`CallInstruction`, then the 'store' part of
    the assignment should be realized using streaming stores.

    .. note::

        This tag is advisory in nature and may be ignored by targets
        that do not understand it or in situations where it does not
        apply.

    .. warning::

        This is a dodgy shortcut, and no promise is made that this will
        continue to work. Whether this is safe is target-dependent and
        program-dependent. No promise of safety is made.
    """
    pass

# }}}


# {{{ HappensAfter

@dataclass(frozen=True)
class HappensAfter:
    """A class representing a "happens-after" relationship between two
    statements found in a :class:`loopy.LoopKernel`. Used to validate that a
    given kernel transformation respects the data dependencies in a given
    program.

    .. attribute:: variable_name

       The name of the variable responsible for the dependency. For
       backward compatibility purposes, this may be *None*. In this case, the
       dependency semantics revert to the deprecated, statement-level
       dependencies of prior versions of :mod:`loopy`.

    .. attribute:: instances_rel

        An :class:`islpy.Map` representing the precise happens-after
        relationship. The domain and range are sets of statement instances. The
        instances in the domain are required to execute before the instances in
        the range.

        Map dimensions are named according to the order of appearance of the
        inames in a :mod:`loopy` program. The dimension names in the range are
        appended with a prime to signify that the mapped instances are distinct.

        As a (deprecated) matter of backward compatibility, this may be *None*,
        in which case the semantics revert to the (underspecified)
        statement-level dependencies of prior versions of :mod:`loopy`.
    """

    variable_name: str | None
    instances_rel: isl.Map | None

# }}}


# {{{ instructions: base class

class InstructionBase(ImmutableRecord, Taggable):
    """A base class for all types of instruction that can occur in
    a kernel.

    .. attribute:: id

        An (otherwise meaningless) identifier that is unique within
        a :class:`loopy.LoopKernel`.

    .. rubric:: Instruction ordering

    .. attribute:: depends_on

        a :class:`frozenset` of :attr:`id` values of :class:`InstructionBase`
        instances that *must* be executed before this one. Note that
        :func:`loopy.preprocess_kernel` (usually invoked automatically)
        augments this by adding dependencies on any writes to temporaries read
        by this instruction.

        May be *None* to invoke the default.

        There are two extensions to this:

        - You may use `*` as a wildcard in the given IDs. This will be expanded
          to all matching instruction IDs during :func:`loopy.make_kernel`.
        - Instead of an instruction ID, you may pass an instance of
          :class:`loopy.match.MatchExpressionBase` into the :attr:`depends_on`
          :class:`frozenset`. The given expression will be used to add any
          matching instructions in the kernel to :attr:`depends_on` during
          :func:`loopy.make_kernel`. Note, that this is not meant as a user-facing
          interface.

    .. attribute:: depends_on_is_final

        A :class:`bool` determining whether :attr:`depends_on` constitutes
        the *entire* list of iname dependencies. If *not* marked final,
        various semi-broken heuristics will try to add further dependencies.

        Defaults to *False*.

    .. attribute:: groups

        A :class:`frozenset` of strings indicating the names of 'instruction
        groups' of which this instruction is a part. An instruction group is
        considered 'active' as long as one (but not all) instructions of the
        group have been executed.

    .. attribute:: conflicts_with_groups

        A :class:`frozenset` of strings indicating which instruction groups
        (see :attr:`groups`) may not be active when this
        instruction is scheduled.

    .. attribute:: priority

        Scheduling priority, an integer. Higher means 'execute sooner'.
        Default 0.

    .. rubric :: Synchronization

    .. attribute:: no_sync_with

        a :class:`frozenset` of tuples of the form ``(insn_id, scope)``, where
        ``insn_id`` refers to :attr:`id` of :class:`InstructionBase` instances
        and `scope` is one of the following strings:

           - `"local"`
           - `"global"`
           - `"any"`.

        An element ``(insn_id, scope)`` means "do not consider any variable
        access conflicting for variables of ``scope`` between this instruction
        and ``insn_id``".
        Specifically, loopy will not complain even if it detects that accesses
        potentially requiring ordering (e.g. by dependencies) exist, and it
        will not emit barriers to guard any dependencies from this
        instruction on ``insn_id`` that may exist.

        Note, that :attr:`no_sync_with` allows instruction matching through wildcards
        and match expression, just like :attr:`depends_on`.

        This data is used specifically by barrier insertion and
        :func:`loopy.check.check_variable_access_ordered`.

    .. rubric:: Conditionals

    .. attribute:: predicates

        a :class:`frozenset` of expressions. The conjunction (logical and) of
        their truth values (as defined by C) determines whether this instruction
        should be run.

    .. rubric:: Iname dependencies

    .. attribute:: within_inames

        A :class:`frozenset` of inames identifying the loops within which this
        instruction will be executed.

    .. rubric:: Iname dependencies

    .. rubric:: Tagging

    .. attribute:: tags

        A :class:`frozenset` of subclasses of :class:`pytools.tag.Tag` used to
        provide metadata on this object. Legacy string tags are converted to
        :class:`LegacyStringInstructionTag` or, if they used to carry
        a functional meaning, the tag carrying that same functional meaning
        (e.g. :class:`UseStreamingStoreTag`).

    .. automethod:: __init__
    .. automethod:: assignee_var_names
    .. automethod:: assignee_subscript_deps
    .. automethod:: with_transformed_expressions
    .. automethod:: write_dependency_names
    .. automethod:: dependency_names
    .. automethod:: copy

    Inherits from :class:`pytools.tag.Taggable`.
    """
    # None-able during kernel creation
    id: str

    happens_after: Mapping[str, HappensAfter]
    depends_on_is_final: bool
    groups: frozenset[str]
    conflicts_with_groups: frozenset[str]
    no_sync_with: frozenset[tuple[str, str]]
    predicates: frozenset[Expression]
    within_inames: frozenset[InameStr]
    within_inames_is_final: bool
    priority: int

    # within_inames_is_final is deprecated and will be removed in version 2017.x.

    fields: ClassVar[set[str]] = {"id", "depends_on_is_final", "groups",
        "conflicts_with_groups", "no_sync_with", "predicates",
        "within_inames_is_final", "within_inames", "priority"}

    def __init__(self,
                 id: str | None,
                 happens_after: (
                     Mapping[str, HappensAfter] | frozenset[str] | str | None),
                 depends_on_is_final: bool | None,
                 groups: frozenset[str] | None,
                 conflicts_with_groups: frozenset[str] | None,
                 no_sync_with: frozenset[tuple[str, str]] | None,
                 within_inames_is_final: bool | None,
                 within_inames: frozenset[str] | None,
                 priority: int | None,
                 predicates: frozenset[str] | None,
                 tags: frozenset[Tag] | None,
                 *,
                 depends_on: frozenset[str] | str | None = None,
                 ) -> None:
        from constantdict import constantdict

        if predicates is None:
            predicates = frozenset()

        new_predicates = set()
        for pred in predicates:
            if isinstance(pred, str):
                from loopy.symbolic import parse
                pred = parse(pred)

            new_predicates.add(pred)

        predicates = frozenset(new_predicates)
        del new_predicates

        # {{{ process happens_after/depends_on

        if happens_after is not None and depends_on is not None:
            raise TypeError("may not pass both happens_after and depends_on")
        elif depends_on is not None:
            # FIXME Enable once we realistically check detailed dependencies.
            # warn("depends_on is deprecated and will stop working in 2026. "
            #      "Pass happens_after instead.", DeprecationWarning, stacklevel=2)
            happens_after = depends_on

        del depends_on

        if depends_on_is_final and happens_after is None:
            raise LoopyError("Setting depends_on_is_final to True requires "
                    "actually specifying happens_after/depends_on")

        if isinstance(happens_after, constantdict):
            pass
        elif happens_after is None:
            happens_after = constantdict()
        elif isinstance(happens_after, str):
            warn("Passing a string for happens_after/depends_on is deprecated and "
                 "will stop working in 2025. Instead, pass a full-fledged "
                 "happens_after data structure.", DeprecationWarning, stacklevel=2)

            happens_after = constantdict({
                    after_id.strip(): HappensAfter(
                        variable_name=None,
                        instances_rel=None)
                    for after_id in happens_after.split(",")
                    if after_id.strip()})
        elif isinstance(happens_after, frozenset):
            happens_after = constantdict({
                    after_id: HappensAfter(
                        variable_name=None,
                        instances_rel=None)
                    for after_id in happens_after})
        elif isinstance(happens_after, dict):
            happens_after = constantdict(happens_after)
        else:
            raise TypeError("'happens_after' has unexpected type: "
                            f"{type(happens_after)}")

        # }}}

        if groups is None:
            groups = frozenset()

        if conflicts_with_groups is None:
            conflicts_with_groups = frozenset()

        if no_sync_with is None:
            no_sync_with = frozenset()

        if within_inames is None:
            within_inames = frozenset()

        if within_inames_is_final is None:
            within_inames_is_final = False

        if depends_on_is_final is None:
            depends_on_is_final = False

        if depends_on_is_final and not isinstance(happens_after, abc_Mapping):
            raise LoopyError("Setting depends_on_is_final to True requires "
                    "actually specifying happens_after/depends_on")

        if tags is None:
            tags = frozenset()

        if priority is None:
            priority = 0

        if not isinstance(tags, abc_Set):
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
        # assert all(is_interned(iname) for iname in within_inames)
        # assert all(is_interned(pred) for pred in predicates)

        assert isinstance(within_inames, abc_Set)
        assert isinstance(happens_after, abc_Mapping) or happens_after is None
        assert isinstance(groups, abc_Set)
        assert isinstance(conflicts_with_groups, abc_Set)

        from loopy.tools import is_hashable
        assert is_hashable(happens_after)

        ImmutableRecord.__init__(self,
                id=id,
                happens_after=happens_after,
                depends_on_is_final=depends_on_is_final,
                no_sync_with=no_sync_with,
                groups=groups, conflicts_with_groups=conflicts_with_groups,
                within_inames_is_final=within_inames_is_final,
                within_inames=within_inames,
                priority=priority,
                predicates=predicates,
                # Yes, tags is set by both this and the Taggable constructor.
                # Here, we set it so that ImmutableRecord knows about it.
                # The Taggable constructor call does extra validation.
                tags=tags)

    def get_copy_kwargs(self, **kwargs):
        passed_depends_on = "depends_on" in kwargs

        if passed_depends_on:
            assert "happens_after" not in kwargs

        kwargs = super().get_copy_kwargs(**kwargs)

        if passed_depends_on:
            # FIXME Enable once we realistically check detailed dependencies.
            # warn("depends_on is deprecated and will stop working in 2026. "
            #      "Instead, use happens_after.", DeprecationWarning, stacklevel=2)
            del kwargs["happens_after"]

        return kwargs

    # {{{ abstract interface

    def read_dependency_names(self) -> abc_Set[str]:
        from loopy.symbolic import get_dependencies
        result: frozenset[str] = frozenset()

        for pred in self.predicates:
            result = result | get_dependencies(pred)

        return result

    def reduction_inames(self) -> abc_Set[str]:
        raise NotImplementedError

    def sub_array_ref_inames(self) -> abc_Set[str]:
        raise NotImplementedError

    def assignee_var_names(self) -> Sequence[str]:
        """Return a tuple of assignee variable names, one
        for each quantity being assigned to.
        """
        raise NotImplementedError

    def assignee_subscript_deps(self):
        """Return a list of sets of variable names referred to in the subscripts
        of the quantities being assigned to, one for each assignee.
        """
        raise NotImplementedError

    def with_transformed_expressions(self,
                f: Callable[[Expression], Expression],
                assignee_f: Callable[[Expression], Expression] | None = None
            ) -> Self:
        """Return a new copy of *self* where *f* has been applied to every
        expression occurring in *self*. *args* will be passed as extra
        arguments (in addition to the expression) to *f*.

        If *assignee_f* is passed, then left-hand sides of assignments are
        passed to it. If it is not given, it defaults to the same as *f*.
        """
        raise NotImplementedError

    # }}}

    @property
    def depends_on(self):
        # FIXME Enable once we realistically check detailed dependencies.
        # warn("depends_on is deprecated and will stop working in 2026. "
        #      "Use happens_after instead.", DeprecationWarning, stacklevel=2)
        return frozenset(self.happens_after)

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

        result = frozenset(self.assignee_var_names())
        asd = self.assignee_subscript_deps()
        if asd:
            result = result | frozenset.union(*self.assignee_subscript_deps())

        return result

    @memoize_method
    def dependency_names(self):
        return self.read_dependency_names() | self.write_dependency_names()

    def get_str_options(self):
        result = []

        if self.depends_on:
            result.append("dep="+":".join(self.depends_on))
        if self.no_sync_with:
            result.append("nosync="+":".join(
                    "%s@%s" % entry for entry in self.no_sync_with))
        if self.groups:
            result.append("groups=%s" % ":".join(self.groups))
        if self.conflicts_with_groups:
            result.append("conflicts=%s" % ":".join(self.conflicts_with_groups))
        if self.priority:
            result.append("priority=%d" % self.priority)
        if self.tags:
            from loopy.kernel.tools import stringify_instruction_tag
            result.append("tags=%s" % ":".join(
                stringify_instruction_tag(t) for t in self.tags))
        if hasattr(self, "atomicity") and self.atomicity:
            result.append("atomic=%s" % ":".join(str(a) for a in self.atomicity))

        return result

    # {{{ hashing and key building

    @cached_property
    def _key_builder(self):
        from loopy.tools import LoopyEqKeyBuilder
        key_builder = LoopyEqKeyBuilder()
        key_builder.update_for_class(self.__class__)

        for field_name in self.fields:
            key_builder.update_for_field(field_name, getattr(self, field_name))

        return key_builder

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.

        Only works in conjunction with :class:`loopy.tools.KeyBuilder`.
        """

        key_builder.rec(key_hash, self._key_builder.hash_key())

    # }}}

    def __setstate__(self, val):
        super().__setstate__(val)

        from constantdict import constantdict

        from loopy.tools import intern_frozenset_of_ids

        if self.id is not None:  # pylint:disable=access-member-before-definition
            self.id = intern(self.id)
        self.happens_after = constantdict({
                intern(after_id): ha
                for after_id, ha in self.happens_after.items()})
        self.groups = intern_frozenset_of_ids(self.groups)
        self.conflicts_with_groups = (
                intern_frozenset_of_ids(self.conflicts_with_groups))
        self.within_inames = (
                intern_frozenset_of_ids(self.within_inames))

    def _with_new_tags(self, tags: frozenset[Tag]):
        return self.copy(tags=tags)

# }}}


def _get_assignee_var_name(expr: Assignable) -> str:
    from pymbolic.primitives import Lookup, Subscript, Variable

    from loopy.symbolic import LinearSubscript, SubArrayRef

    if isinstance(expr, Lookup):
        expr = cast("Assignable", expr.aggregate)

    if isinstance(expr, Variable):
        return expr.name

    elif isinstance(expr, (Subscript, LinearSubscript)):
        agg = expr.aggregate
        assert isinstance(agg, Variable)

        return agg.name

    elif isinstance(expr, SubArrayRef):
        agg = expr.subscript.aggregate
        assert isinstance(agg, Variable)

        return agg.name

    else:
        raise RuntimeError("invalid lvalue '%s'" % expr)


def _get_assignee_subscript_deps(expr: Expression) -> frozenset[str]:
    from pymbolic.primitives import Lookup, Subscript, Variable

    from loopy.symbolic import LinearSubscript, SubArrayRef, get_dependencies

    if isinstance(expr, Lookup):
        expr = expr.aggregate

    if isinstance(expr, Variable):
        return frozenset()
    elif isinstance(expr, (Subscript, LinearSubscript)):
        return get_dependencies(expr.index)
    elif isinstance(expr, SubArrayRef):
        return get_dependencies(expr.subscript.index) - (
                frozenset(iname.name for iname in expr.swept_inames))
    else:
        raise RuntimeError("invalid lvalue '%s'" % expr)


# {{{ atomic ops

class MemoryOrdering:
    """Ordering of atomic operations, defined as in C11 and OpenCL.

    .. attribute:: RELAXED
    .. attribute:: ACQUIRE
    .. attribute:: RELEASE
    .. attribute:: ACQ_REL
    .. attribute:: SEQ_CST
    """

    RELAXED = 0
    ACQUIRE = 1
    RELEASE = 2
    ACQ_REL = 3
    SEQ_CST = 4

    @staticmethod
    def to_string(v):
        for i in dir(MemoryOrdering):
            if i.startswith("_"):
                continue

            if getattr(MemoryOrdering, i) == v:
                return i

        raise ValueError("Unknown value of MemoryOrdering")


class MemoryScope:
    """Scope of atomicity, defined as in OpenCL.

    .. attribute:: auto

        Scope matches the accessibility of the variable.

    .. attribute:: WORK_ITEM
    .. attribute:: WORK_GROUP
    .. attribute:: WORK_DEVICE
    .. attribute:: ALL_SVM_DEVICES
    """

    WORK_ITEM = 0
    WORK_GROUP = 1
    DEVICE = 2
    ALL_SVM_DEVICES = 2

    auto = -1

    @staticmethod
    def to_string(v):
        for i in dir(MemoryScope):
            if i.startswith("_"):
                continue

            if getattr(MemoryScope, i) == v:
                return i

        raise ValueError("Unknown value of MemoryScope")


class VarAtomicity:
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
        return (type(self) is type(other)
                and self.var_name == other.var_name)

    def __ne__(self, other):
        return not self.__eq__(other)


class OrderedAtomic(VarAtomicity):
    """Properties of an atomic operation. A subclass of :class:`VarAtomicity`.

    .. attribute:: ordering

        One of the values from :class:`MemoryOrdering`

    .. attribute:: scope

        One of the values from :class:`MemoryScope`
    """

    ordering = MemoryOrdering.SEQ_CST
    scope = MemoryScope.auto

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        super().update_persistent_hash(key_hash, key_builder)
        key_builder.rec(key_hash, str(self.__class__.__name__))
        key_builder.rec(key_hash, self.ordering)
        key_builder.rec(key_hash, self.scope)

    def __eq__(self, other):
        return (super().__eq__(other)
                and self.ordering == other.ordering
                and self.scope == other.scope)

    def __hash__(self):
        return hash((type(self), self.var_name,
                     self.ordering, self.scope))

    @property
    def op_name(self):
        raise NotImplementedError

    def __str__(self):
        return "{}[{}]{}/{}".format(
                self.op_name,
                self.var_name,
                MemoryOrdering.to_string(self.ordering),
                MemoryScope.to_string(self.scope))


class AtomicInit(OrderedAtomic):
    """Describes initialization of an atomic variable. A subclass of
    :class:`OrderedAtomic`.

    .. attribute:: ordering

        One of the values from :class:`MemoryOrdering`

    .. attribute:: scope

        One of the values from :class:`MemoryScope`
    """
    op_name = "init"


class AtomicUpdate(OrderedAtomic):
    """Properties of an atomic update. A subclass of
    :class:`OrderedAtomic`.

    .. attribute:: ordering

        One of the values from :class:`MemoryOrdering`

    .. attribute:: scope

        One of the values from :class:`MemoryScope`
    """
    op_name = "update"


class AtomicLoad(OrderedAtomic):
    """Properties of an atomic load. A subclass of :class:`OrderedAtomic`.

    .. attribute:: ordering

        One of the values from :class:`MemoryOrdering`

    .. attribute:: scope

        One of the values from :class:`MemoryScope`
    """
    op_name = "load"

# }}}


# {{{ instruction base class: expression rhs

class MultiAssignmentBase(InstructionBase):
    """An assignment instruction with an expression as a right-hand side."""

    fields = InstructionBase.fields | {"expression"}

    assignees: tuple[Assignable, ...]  # pyright: ignore[reportUninitializedInstanceVariable]

    @memoize_method
    def read_dependency_names(self):
        from loopy.symbolic import get_dependencies
        result = (
                super().read_dependency_names()
                | get_dependencies(self.expression))

        for subscript_deps in self.assignee_subscript_deps():
            result = result | subscript_deps

        return result

    @memoize_method
    def reduction_inames(self):
        from loopy.symbolic import get_reduction_inames
        return frozenset(get_reduction_inames(self.expression))

    @memoize_method
    def sub_array_ref_inames(self):
        from loopy.symbolic import get_sub_array_ref_swept_inames
        return get_sub_array_ref_swept_inames((self.assignees, self.expression))

# }}}


# {{{ instruction: assignment

class _not_provided:  # noqa: N801
    pass


class Assignment(MultiAssignmentBase):
    """
    .. attribute:: assignee

    .. attribute:: expression

    The following attributes are only used until
    :func:`loopy.make_kernel` is finished:

    .. attribute:: temp_var_type

        A :class:`loopy.Optional`. If not empty, contains the type that
        will be assigned to the new temporary variable created from the
        assignment.

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

        :mod:`loopy` may choose to evaluate the right-hand side *multiple times*
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

    assignee: Assignable
    expression: Expression
    temp_var_type: LoopyOptional[LoopyType | None]
    atomicity: tuple[VarAtomicity, ...]

    fields = MultiAssignmentBase.fields | \
            {"assignee", "temp_var_type", "atomicity"}

    def __init__(self,
                 assignee: str | Assignable,
                 expression: str | Expression,
                 id: str | None = None,
                 happens_after:
                     Mapping[str, HappensAfter] | frozenset[str] | str | None = None,
                 depends_on_is_final: bool | None = None,
                 groups: frozenset[str] | None = None,
                 conflicts_with_groups: frozenset[str] | None = None,
                 no_sync_with: frozenset[tuple[str, str]] | None = None,
                 within_inames_is_final: bool | None = None,
                 within_inames: frozenset[str] | None = None,
                 priority: int | None = None,
                 predicates: frozenset[str] | None = None,
                 tags: frozenset[Tag] | None = None,
                 temp_var_type:
                    type[_not_provided]
                        | LoopyOptional[ToLoopyTypeConvertible | None]
                    = _not_provided,
                 atomicity: tuple[VarAtomicity, ...] = (),
                 *,
                 depends_on: frozenset[str] | str | None = None,
                 ) -> None:

        if temp_var_type is _not_provided:
            temp_var_type = LoopyOptional[LoopyType]()

        super().__init__(
                id=id,
                happens_after=happens_after,
                depends_on_is_final=depends_on_is_final,
                groups=groups,
                conflicts_with_groups=conflicts_with_groups,
                no_sync_with=no_sync_with,
                within_inames_is_final=within_inames_is_final,
                within_inames=within_inames,
                priority=priority,
                predicates=predicates,
                tags=tags,
                depends_on=depends_on)

        from loopy.symbolic import parse
        if isinstance(assignee, str):
            assignee_expr = parse(assignee)
            if isinstance(assignee_expr, (p.Subscript, p.Variable)):
                assignee = assignee_expr
            else:
                raise LoopyError(f"not assignable: {type(assignee_expr)}")

        if isinstance(expression, str):
            parsed_expression = parse(expression)
        else:
            parsed_expression = expression

        from pymbolic.primitives import Lookup, Subscript, Variable

        from loopy.symbolic import LinearSubscript
        if not isinstance(assignee, (Variable, Subscript, LinearSubscript, Lookup)):
            raise LoopyError("invalid lvalue '%s'" % assignee)

        self.assignee = assignee
        self.expression = parsed_expression

        self.temp_var_type = _check_and_fix_temp_var_type(temp_var_type)
        self.atomicity = atomicity

    # {{{ implement InstructionBase interface

    @memoize_method
    def assignee_var_names(self):
        return (_get_assignee_var_name(self.assignee),)

    def assignee_subscript_deps(self):
        return frozenset({_get_assignee_subscript_deps(self.assignee)})

    @override
    def with_transformed_expressions(self,
                f: Callable[[Expression], Expression],
                assignee_f: Callable[[Expression], Expression] | None = None
            ) -> Self:
        if assignee_f is None:
            assignee_f = f

        assignee = assignee_f(self.assignee)
        expression = f(self.expression)
        predicates = []
        changed_predicates = False
        for pred in self.predicates:
            new_pred = f(pred)
            if new_pred is not pred:
                changed_predicates = True
            predicates.append(new_pred)
        predicates = frozenset(predicates) if changed_predicates else self.predicates

        if assignee is self.assignee and expression is self.expression and \
                predicates is self.predicates:
            return self

        return self.copy(
                assignee=assignee,
                expression=expression,
                predicates=predicates)

    # }}}

    def __str__(self):
        result = f"{self.assignee} <- {self.expression}"

        if self.id is not None:
            result = "%s: " % self.id + result

        options = self.get_str_options()
        if options:
            result += " {%s}" % (": ".join(options))

        if self.predicates:
            result += "\n" + 10*" " + "if (%s)" % " and ".join(
                    str(p) for p in self.predicates)
        return result

    # {{{ for interface uniformity with CallInstruction

    @property
    def temp_var_types(self):
        return (self.temp_var_type,)

    @property
    @override
    def assignees(self) -> tuple[Assignable]:  # pyright: ignore[reportIncompatibleVariableOverride]
        return (self.assignee,)

    @memoize_method
    def sub_array_ref_inames(self):
        assert super().sub_array_ref_inames() == frozenset()
        return frozenset()

    # }}}

# }}}


# {{{ instruction: function call

class CallInstruction(MultiAssignmentBase):
    """An instruction capturing a function call. Unlike :class:`Assignment`,
    this instruction supports functions with multiple return values.

    .. attribute:: assignees

        A :class:`tuple` of left-hand sides for the assignment

    .. attribute:: expression

    The following attributes are only used until
    :func:`loopy.make_kernel` is finished:

    .. attribute:: temp_var_types

        A tuple of `:class:loopy.Optional`. If an entry is not empty, it
        contains the type that will be assigned to the new temporary variable
        created from the assignment.

    .. automethod:: __init__
    """

    expression: p.Call

    fields = MultiAssignmentBase.fields | \
            {"assignees", "temp_var_types"}

    def __init__(self,
            assignees: tuple[Assignable, ...] | str,
            expression: Expression,
            id: str | None = None,
            happens_after=None,
            depends_on_is_final=None,
            groups=None,
            conflicts_with_groups=None,
            no_sync_with=None,
            within_inames_is_final=None,
            within_inames=None,
            tags=None,
            temp_var_types=None,
            priority=0, predicates=frozenset(),
            depends_on=None
        ) -> None:

        super().__init__(
                id=id,
                happens_after=happens_after,
                depends_on_is_final=depends_on_is_final,
                groups=groups,
                conflicts_with_groups=conflicts_with_groups,
                no_sync_with=no_sync_with,
                within_inames_is_final=within_inames_is_final,
                within_inames=within_inames,
                priority=priority,
                predicates=predicates,
                tags=tags,
                depends_on=depends_on)

        from pymbolic.primitives import Call

        from loopy.symbolic import Reduction
        if not isinstance(expression, (Call, Reduction)) and (
                expression is not None):
            raise LoopyError("'expression' argument to CallInstruction "
                    "must be a function call")

        from loopy.symbolic import parse
        if isinstance(assignees, str):
            assignees_expr = parse(assignees)
            if isinstance(assignees_expr, tuple):
                assignees = cast(
                                 "tuple[Assignable, ...]", assignees_expr)
            else:
                raise LoopyError(f"not assignable: {type(assignees_expr)}")
        if not isinstance(assignees, tuple):
            raise LoopyError("'assignees' argument to CallInstruction "
                    "must be a tuple or a string parseable to a tuple"
                    "--got '%s'" % type(assignees).__name__)

        if isinstance(expression, str):
            expression = parse(expression)

        from pymbolic.primitives import Subscript, Variable

        from loopy.symbolic import LinearSubscript, SubArrayRef
        for assignee in assignees:
            if not isinstance(assignee, (Variable, Subscript, LinearSubscript,
                    SubArrayRef)):
                raise LoopyError("invalid lvalue '%s'" % assignee)

        self.assignees = assignees
        self.expression = expression

        if temp_var_types is None:
            self.temp_var_types = (LoopyOptional(),) * len(self.assignees)
        else:
            self.temp_var_types = tuple(
                    _check_and_fix_temp_var_type(tvt, stacklevel=3)
                    for tvt in temp_var_types)

    # {{{ implement InstructionBase interface

    @memoize_method
    def assignee_var_names(self):
        return tuple(_get_assignee_var_name(a) for a in self.assignees)

    def assignee_subscript_deps(self):
        return tuple(
                _get_assignee_subscript_deps(a)
                for a in self.assignees)

    @override
    def with_transformed_expressions(self,
                f: Callable[[Expression], Expression],
                assignee_f: Callable[[Expression], Expression] | None = None
            ) -> Self:
        if assignee_f is None:
            assignee_f = f

        assignees = cast("tuple[Assignable]", assignee_f(self.assignees))

        expression = f(self.expression)
        predicates = []
        changed_predicates = False
        for pred in self.predicates:
            new_pred = f(pred)
            if new_pred is not pred:
                changed_predicates = True
            predicates.append(new_pred)
        predicates = frozenset(predicates) if changed_predicates else self.predicates

        if len(assignees) == len(self.assignees) and \
                all(assignee is orig_assignee for assignee, orig_assignee in
                    zip(assignees, self.assignees)) \
                and expression is self.expression and \
                predicates is self.predicates:
            return self

        return self.copy(
                assignees=assignees,
                expression=expression,
                predicates=predicates)

    # }}}

    def __str__(self):
        result = "{}: {} <- {}".format(self.id,
                ", ".join(str(a) for a in self.assignees),
                self.expression)

        options = self.get_str_options()
        if options:
            result += " {%s}" % (": ".join(options))

        if self.predicates:
            result += "\n" + 10*" " + "if (%s)" % " && ".join(
                    str(pred) for pred in self.predicates)
        return result

    def arg_id_to_arg(self) -> Mapping[int, Expression]:
        """:returns: a :class:`dict` mapping argument identifiers (non-negative
            numbers for positional arguments and negative numbers for assignees) to
            their respective values
        """
        arg_id_to_arg = dict(enumerate(self.expression.parameters))
        for i, arg in enumerate(self.assignees):
            arg_id_to_arg[-i-1] = arg

        return arg_id_to_arg

    @property
    def atomicity(self):
        # Function calls can impossibly be atomic, and even the result assignment
        # is troublesome, especially in the case of multiple results. Avoid the
        # issue altogether by disallowing atomicity.
        return ()

# }}}


def subscript_contains_slice(subscript):
    """Return *True* if the *subscript* contains an instance of
    :class:`pymbolic.primitives.Slice` as of its indices.
    """
    from pymbolic.primitives import Slice, Subscript
    assert isinstance(subscript, Subscript)
    return any(isinstance(index, Slice) for index in subscript.index_tuple)


def is_array_call(assignees, expression):
    """
    Returns *True* is the instruction is an array call.

    An array call is a function call applied to array type objects. If any of
    the arguments or assignees to the function is an array,
    :meth:`is_array_call` will return *True*.
    """
    from pymbolic.primitives import Call, Subscript

    from loopy.symbolic import SubArrayRef

    if not isinstance(expression, Call):
        return False

    for par in expression.parameters+assignees:
        if isinstance(par, SubArrayRef) or (
                isinstance(par, Subscript) and subscript_contains_slice(par)):
            return True

    # did not encounter SubArrayRef/Slice, hence must be a normal call
    return False


def modify_assignee_for_array_call(
            assignee: p.Variable | p.Subscript | SubArrayRef
        ) -> p.Subscript | SubArrayRef:
    """
    Converts the assignee subscript or variable as a SubArrayRef.
    """
    from pymbolic.primitives import Subscript, Variable

    from loopy.symbolic import SubArrayRef
    if isinstance(assignee, SubArrayRef):
        return assignee
    elif isinstance(assignee, Subscript):
        if subscript_contains_slice(assignee):
            # Slice subscripted array are treated as SubArrayRef in the kernel
            # Hence, making the behavior similar to that of `SubArrayref`
            return assignee
        else:
            return SubArrayRef((), assignee)
    elif isinstance(assignee, Variable):
        return SubArrayRef((), Subscript(assignee, 0))
    else:
        raise LoopyError("ArrayCall only takes Variable, Subscript or "
                "SubArrayRef as its inputs")


def make_assignment(assignees: tuple[Assignable, ...],
                    expression: Expression,
                    temp_var_types: (
                        Sequence[ToLoopyTypeConvertible | None] | None) = None,
                    **kwargs: Any) -> Assignment | CallInstruction:

    tv_types: Sequence[LoopyOptional[ToLoopyTypeConvertible] | None]
    if temp_var_types is None:
        tv_types = (LoopyOptional(),) * len(assignees)
    else:
        tv_types = [
            t if isinstance(t, LoopyOptional) else LoopyOptional(t)
            for t in temp_var_types]

    if len(assignees) != 1 or is_array_call(assignees, expression):
        atomicity = kwargs.pop("atomicity", ())
        if atomicity:
            raise LoopyError("atomic operations with more than one "
                    "left-hand side not supported")

        from pymbolic.primitives import Call

        from loopy.symbolic import Reduction
        if not isinstance(expression, (Call, Reduction)):
            raise LoopyError("right-hand side in multiple assignment must be "
                    "function call or reduction, got: "
                    f"'{type(expression).__name__}'")

        if not is_array_call(assignees, expression):
            return CallInstruction(
                    assignees=assignees,
                    expression=expression,
                    temp_var_types=tv_types,
                    **kwargs)
        else:
            # In the case of an array call, it is important to have each
            # assignee as an instance of SubArrayRef. If not given as a
            # SubArrayRef
            return CallInstruction(
                    assignees=tuple(modify_assignee_for_array_call(
                        # FIXME: It's got a point. But LinearSubscript is deprecated.
                        assignee) for assignee in assignees),  # pyright: ignore[reportArgumentType]
                    expression=expression,
                    temp_var_types=tuple(tv_types),
                    **kwargs)
    else:
        def _is_array(expr):
            from pymbolic.primitives import Slice, Subscript

            from loopy.symbolic import SubArrayRef
            if isinstance(expr, SubArrayRef):
                return True
            if isinstance(expr, Subscript):
                return any(isinstance(idx, Slice) for idx in
                        expr.index_tuple)
            return False

        from loopy.symbolic import DependencyMapper
        if any(_is_array(dep) for dep in DependencyMapper()((assignees,
                expression))):
            raise LoopyError("Array calls only supported as instructions"
                    " with function call as RHS for now.")

        assignee, = assignees
        tv_type, = tv_types

        return Assignment(
                assignee=assignee,
                expression=expression,
                temp_var_type=tv_type,
                **kwargs)


# {{{ c instruction

class CInstruction(InstructionBase):
    """
    .. attribute:: iname_exprs

        A tuple of tuples *(name, expr)* of inames or expressions based on them
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
        without subscript) as :data:`pymbolic.typing.Expression` instances
        that :attr:`code` writes to. This is optional and only used for
        figuring out dependencies.
    """

    fields = InstructionBase.fields | \
            {"iname_exprs", "code", "read_variables", "assignees"}

    def __init__(self,
            iname_exprs, code,
            read_variables=frozenset(), assignees=(),
            id=None, happens_after=None, depends_on_is_final=None,
            groups=None, conflicts_with_groups=None,
            no_sync_with=None,
            within_inames_is_final=None, within_inames=None,
            priority=0,
            predicates=frozenset(), tags=None,
            depends_on=None):
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
                happens_after=happens_after,
                depends_on_is_final=depends_on_is_final,
                groups=groups, conflicts_with_groups=conflicts_with_groups,
                no_sync_with=no_sync_with,
                within_inames_is_final=within_inames_is_final,
                within_inames=within_inames,
                priority=priority, predicates=predicates, tags=tags,
                depends_on=depends_on)

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

        self.iname_exprs = tuple(new_iname_exprs)
        from loopy.tools import remove_common_indentation
        self.code = remove_common_indentation(code)
        self.read_variables = read_variables
        self.assignees = tuple(new_assignees)

    # {{{ abstract interface

    def read_dependency_names(self):
        result = (
                super().read_dependency_names()
                | frozenset(self.read_variables))

        from loopy.symbolic import get_dependencies
        for _name, iname_expr in self.iname_exprs:
            result = result | get_dependencies(iname_expr)

        for subscript_deps in self.assignee_subscript_deps():
            result = result | subscript_deps

        for pred in self.predicates:
            result = result | get_dependencies(pred)

        return frozenset(result)

    def reduction_inames(self):
        return frozenset()

    def sub_array_ref_inames(self):
        return frozenset()

    def assignee_var_names(self):
        return tuple(_get_assignee_var_name(expr) for expr in self.assignees)

    def assignee_subscript_deps(self):
        return tuple(
                _get_assignee_subscript_deps(a)
                for a in self.assignees)

    @override
    def with_transformed_expressions(self,
                f: Callable[[Expression], Expression],
                assignee_f: Callable[[Expression], Expression] | None = None
            ) -> Self:
        if assignee_f is None:
            assignee_f = f

        return self.copy(
                iname_exprs=[
                    (name, f(expr))
                    for name, expr in self.iname_exprs],
                assignees=[assignee_f(a) for a in self.assignees],
                predicates=frozenset(
                    f(pred) for pred in self.predicates))

    # }}}

    def __str__(self):
        first_line = "{}: {} <- CODE({}|{})".format(self.id,
                ", ".join(str(a) for a in self.assignees),
                ", ".join(str(x) for x in self.read_variables),
                ", ".join(f"{name}={expr}"
                    for name, expr in self.iname_exprs))

        options = self.get_str_options()
        if options:
            first_line += " {%s}" % (": ".join(options))

        return first_line + "\n    " + "\n    ".join(
                self.code.split("\n"))

# }}}


class _DataObliviousInstruction(InstructionBase):
    # {{{ abstract interface

    # read_dependency_names inherited

    @override
    def reduction_inames(self):
        return frozenset()

    @override
    def sub_array_ref_inames(self):
        return frozenset()

    @override
    def assignee_var_names(self) -> Sequence[str]:
        return ()

    @override
    def assignee_subscript_deps(self):
        return frozenset()

    @override
    def with_transformed_expressions(self,
                f: Callable[[Expression], Expression],
                assignee_f: Callable[[Expression], Expression] | None = None
            ) -> Self:
        return self.copy(
                predicates=frozenset(
                    f(pred) for pred in self.predicates))

    # }}}

    @property
    def assignees(self):
        return ()


# {{{ barrier instruction

class NoOpInstruction(_DataObliviousInstruction):
    """An instruction that carries out no operation. It is mainly
    useful as a way to structure dependencies between other
    instructions.

    The textual syntax in a :mod:`loopy` kernel is::

        ... nop
    """

    def __init__(self, id=None, happens_after=None, depends_on_is_final=None,
            groups=None, conflicts_with_groups=None,
            no_sync_with=None,
            within_inames_is_final=None, within_inames=None,
            priority=None,
            predicates=None, tags=None, depends_on=None):
        super().__init__(
                id=id,
                happens_after=happens_after,
                depends_on_is_final=depends_on_is_final,
                groups=groups,
                conflicts_with_groups=conflicts_with_groups,
                no_sync_with=no_sync_with,
                within_inames_is_final=within_inames_is_final,
                within_inames=within_inames,
                priority=priority,
                predicates=predicates,
                tags=tags,
                depends_on=depends_on)

    def __str__(self):
        first_line = "%s: ... nop" % self.id

        options = self.get_str_options()
        if options:
            first_line += " {%s}" % (": ".join(options))

        return first_line

# }}}


# {{{ barrier instruction

class BarrierInstruction(_DataObliviousInstruction):
    """An instruction that requires synchronization with all
    concurrent work items of :attr:`synchronization_kind`.

    .. attribute:: synchronization_kind

        A string, ``"global"`` or ``"local"``.

    .. attribute:: mem_kind

        A string, ``"global"`` or ``"local"``. Chooses which memory type to
        synchronize, for targets that require this (e.g. OpenCL)

    The textual syntax in a :mod:`loopy` kernel is::

        ... gbarrier
        ... lbarrier

    Note that the memory type :attr:`mem_kind` can be specified for local barriers::

        ... lbarrier {mem_kind=global}
    """

    fields = _DataObliviousInstruction.fields | {"synchronization_kind",
                                                     "mem_kind"}

    def __init__(self, id, happens_after=None, depends_on_is_final=None,
            groups=None, conflicts_with_groups=None,
            no_sync_with=None,
            within_inames_is_final=None, within_inames=None,
            priority=None,
            predicates=None, tags=None, synchronization_kind="global",
            mem_kind="local",
            depends_on=None):

        if predicates:
            raise LoopyError("conditional barriers are not supported")

        super().__init__(
                id=id,
                happens_after=happens_after,
                depends_on_is_final=depends_on_is_final,
                groups=groups,
                conflicts_with_groups=conflicts_with_groups,
                no_sync_with=no_sync_with,
                within_inames_is_final=within_inames_is_final,
                within_inames=within_inames,
                priority=priority,
                predicates=predicates,
                tags=tags,
                depends_on=depends_on)

        self.synchronization_kind = synchronization_kind
        self.mem_kind = mem_kind

    def __str__(self):
        first_line = \
                "{}: ... {}barrier".format(self.id, self.synchronization_kind[0])

        options = self.get_str_options()
        if self.synchronization_kind == "local":
            # add the memory kind
            options += [f"mem_kind={self.mem_kind}"]
        if options:
            first_line += " {%s}" % (": ".join(options))

        return first_line

# }}}


# {{{ key getters

def _get_insn_eq_key(insn):
    return insn._key_builder.key()


def _get_insn_hash_key(insn):
    return insn._key_builder.hash_key()

# }}}


# {{{ _check_and_fix_temp_var_type

def _check_and_fix_temp_var_type(
            temp_var_type:
                type[_not_provided]
                | ToLoopyTypeConvertible
                | LoopyOptional[None]
                | LoopyOptional[ToLoopyTypeConvertible],
            stacklevel: int = 2
        ) -> LoopyOptional[LoopyType | None]:
    """Check temp_var_type for deprecated usage, and convert to the right value.
    """

    import loopy as lp

    assert temp_var_type is not _not_provided

    if temp_var_type is None:
        warn("temp_var_type should be Optional() if no temporary, not None. "
             "This usage will be disallowed soon.",
             DeprecationWarning, stacklevel=1 + stacklevel)
        return lp.Optional()

    elif temp_var_type is lp.auto:
        warn("temp_var_type should be Optional(None) if "
             "unspecified, not auto. This usage will be disallowed soon.",
             DeprecationWarning, stacklevel=1 + stacklevel)
        return lp.Optional(None)

    elif not isinstance(temp_var_type, lp.Optional):
        warn("temp_var_type should be an instance of Optional. "
             "Other values for temp_var_type will be disallowed soon.",
             DeprecationWarning, stacklevel=1 + stacklevel)
        return lp.Optional(to_loopy_type(temp_var_type))

    if not temp_var_type.has_value:
        return LoopyOptional()
    else:
        return LoopyOptional(to_loopy_type(temp_var_type.value, allow_none=True))

# }}}


def get_insn_domain(insn: InstructionBase, kernel: LoopKernel) -> isl.Set:
    """
    Returns an instance of :class:`islpy.Set` for the *insn*'s domain.

    .. note::

        Does not take into account additional hints available through
        :attr:`loopy.LoopKernel.assumptions`.
    """
    domain = kernel.get_inames_domain(insn.within_inames)

    # {{{ add read-only ValueArgs to domain

    from loopy.kernel.data import ValueArg

    valueargs_to_add = ({arg.name for arg in kernel.args
                         if isinstance(arg, ValueArg)
                         and arg.name not in kernel.get_written_variables()}
                        - set(domain.get_var_names(isl.dim_type.param)))

    # only consider valueargs relevant to *insn*
    valueargs_to_add = valueargs_to_add & insn.read_dependency_names()

    for arg_to_add in valueargs_to_add:
        idim = domain.dim(isl.dim_type.param)
        domain = domain.add_dims(isl.dim_type.param, 1)
        domain = domain.set_dim_name(isl.dim_type.param, idim, arg_to_add)

    # }}}

    # {{{ enforce restriction from predicates

    insn_preds_set = isl.Set.universe(domain.space)

    for predicate in insn.predicates:
        from loopy.symbolic import condition_to_set
        predicate_as_isl_set = condition_to_set(domain.space, predicate)
        if predicate_as_isl_set is not None:
            insn_preds_set = insn_preds_set & predicate_as_isl_set

    # }}}

    return domain & insn_preds_set


# vim: foldmethod=marker

from __future__ import division, absolute_import, print_function

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

from six.moves import intern
from pytools import ImmutableRecord, memoize_method
from loopy.diagnostic import LoopyError
from loopy.tools import Optional
from warnings import warn


# {{{ instructions: base class

class InstructionBase(ImmutableRecord):
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
        (see :class:`InstructionBase.groups`) may not be active when this
        instruction is scheduled.

    .. attribute:: priority

        Scheduling priority, an integer. Higher means 'execute sooner'.
        Default 0.

    .. rubric :: Synchronization

    .. attribute:: no_sync_with

        a :class:`frozenset` of tuples of the form ``(insn_id, scope)``, where
        `insn_id` refers to :attr:`id` of :class:`Instruction` instances
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
        :func:`loopy.check.enforce_variable_access_ordered`.

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

        A :class:`frozenset` of string identifiers that can be used to
        identify groups of instructions.

        Tags starting with exclamation marks (``!``) are reserved and may have
        specific meanings defined by :mod:`loopy` or its targets.

    .. automethod:: __init__
    .. automethod:: assignee_var_names
    .. automethod:: assignee_subscript_deps
    .. automethod:: with_transformed_expressions
    .. automethod:: write_dependency_names
    .. automethod:: dependency_names
    .. automethod:: copy
    """

    # within_inames_is_final is deprecated and will be removed in version 2017.x.

    fields = set("id depends_on depends_on_is_final "
            "groups conflicts_with_groups "
            "no_sync_with "
            "predicates "
            "within_inames_is_final within_inames "
            "priority".split())

    # Names of fields that are pymbolic expressions. Needed for key building
    pymbolic_fields = set("")

    # Names of fields that are sets of pymbolic expressions. Needed for key building
    pymbolic_set_fields = set(["predicates"])

    def __init__(self, id, depends_on, depends_on_is_final,
            groups, conflicts_with_groups,
            no_sync_with,
            within_inames_is_final, within_inames,
            priority,
            predicates, tags):

        if predicates is None:
            predicates = frozenset()

        new_predicates = set()
        for pred in predicates:
            if isinstance(pred, str):
                from pymbolic.primitives import LogicalNot
                from loopy.symbolic import parse
                if pred.startswith("!"):
                    warn("predicates starting with '!' are deprecated. "
                            "Simply use 'not' instead")
                    pred = LogicalNot(parse(pred[1:]))
                else:
                    pred = parse(pred)

            new_predicates.add(pred)

        predicates = frozenset(new_predicates)
        del new_predicates

        if depends_on is None:
            depends_on = frozenset()

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

        if isinstance(depends_on, str):
            depends_on = frozenset(
                    s.strip() for s in depends_on.split(",") if s.strip())

        if depends_on_is_final is None:
            depends_on_is_final = False

        if depends_on_is_final and not isinstance(depends_on, frozenset):
            raise LoopyError("Setting depends_on_is_final to True requires "
                    "actually specifying depends_on")

        if tags is None:
            tags = frozenset()

        if priority is None:
            priority = 0

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
        # assert all(is_interned(iname) for iname in within_inames)
        # assert all(is_interned(pred) for pred in predicates)

        assert isinstance(within_inames, frozenset)
        assert isinstance(depends_on, frozenset) or depends_on is None
        assert isinstance(groups, frozenset)
        assert isinstance(conflicts_with_groups, frozenset)

        ImmutableRecord.__init__(self,
                id=id,
                depends_on=depends_on,
                depends_on_is_final=depends_on_is_final,
                no_sync_with=no_sync_with,
                groups=groups, conflicts_with_groups=conflicts_with_groups,
                within_inames_is_final=within_inames_is_final,
                within_inames=within_inames,
                priority=priority,
                predicates=predicates,
                tags=tags)

    # {{{ abstract interface

    def read_dependency_names(self):
        from loopy.symbolic import get_dependencies
        result = frozenset()

        for pred in self.predicates:
            result = result | get_dependencies(pred)

        return result

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

    def with_transformed_expressions(self, f, assignee_f=None):
        """Return a new copy of *self* where *f* has been applied to every
        expression occurring in *self*. *args* will be passed as extra
        arguments (in addition to the expression) to *f*.

        If *assignee_f* is passed, then left-hand sides of assignments are
        passed to it. If it is not given, it defaults to the same as *f*.
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

        result = frozenset(self.assignee_var_names())
        asd = self.assignee_subscript_deps()
        if asd:
            result = result | frozenset.union(*self.assignee_subscript_deps())

        return result

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
            result.append("tags=%s" % ":".join(self.tags))
        if hasattr(self, "atomicity") and self.atomicity:
            result.append("atomic=%s" % ":".join(str(a) for a in self.atomicity))

        return result

    # {{{ hashing and key building

    @property
    @memoize_method
    def _key_builder(self):
        from loopy.tools import LoopyEqKeyBuilder
        key_builder = LoopyEqKeyBuilder()
        key_builder.update_for_class(self.__class__)

        for field_name in self.fields:
            field_value = getattr(self, field_name)
            if field_name in self.pymbolic_fields:
                key_builder.update_for_pymbolic_field(field_name, field_value)
            elif field_name in self.pymbolic_set_fields:
                # First sort the fields, as a canonical form
                items = tuple(sorted(field_value, key=str))
                key_builder.update_for_pymbolic_field(field_name, items)

            # from CExpression
            elif field_name == "iname_exprs":
                from loopy.symbolic import EqualityPreservingStringifyMapper
                key_builder.field_dict[field_name] = [
                        (iname, EqualityPreservingStringifyMapper()(expr)
                            .encode("utf-8"))
                        for iname, expr in self.iname_exprs
                        ]

            else:
                key_builder.update_for_field(field_name, field_value)

        return key_builder

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.

        Only works in conjunction with :class:`loopy.tools.KeyBuilder`.
        """

        key_builder.rec(key_hash, self._key_builder.hash_key())

    # }}}

    def __setstate__(self, val):
        super(InstructionBase, self).__setstate__(val)

        from loopy.tools import intern_frozenset_of_ids

        if self.id is not None:  # pylint:disable=access-member-before-definition
            self.id = intern(self.id)
        self.depends_on = intern_frozenset_of_ids(self.depends_on)
        self.groups = intern_frozenset_of_ids(self.groups)
        self.conflicts_with_groups = (
                intern_frozenset_of_ids(self.conflicts_with_groups))
        self.within_inames = (
                intern_frozenset_of_ids(self.within_inames))

# }}}


def _get_assignee_var_name(expr):
    from pymbolic.primitives import Variable, Subscript, Lookup
    from loopy.symbolic import LinearSubscript

    if isinstance(expr, Lookup):
        expr = expr.aggregate

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
    from pymbolic.primitives import Variable, Subscript, Lookup
    from loopy.symbolic import LinearSubscript, get_dependencies

    if isinstance(expr, Lookup):
        expr = expr.aggregate

    if isinstance(expr, Variable):
        return frozenset()
    elif isinstance(expr, Subscript):
        return get_dependencies(expr.index)
    elif isinstance(expr, LinearSubscript):
        return get_dependencies(expr.index)
    else:
        raise RuntimeError("invalid lvalue '%s'" % expr)


# {{{ atomic ops

class MemoryOrdering:  # noqa
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


# {{{ memory_ordering, MemoryOrdering compatibility

class _deprecated_memory_ordering_class_method(object):  # noqa
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, klass):
        warn("'memory_ordering' is deprecated. Use 'MemoryOrdering'.",
                DeprecationWarning, stacklevel=2)
        return self.f()


class memory_ordering(object):  # noqa
    """Deprecated. Use :class:`MemoryOrdering` instead.
    """

    @_deprecated_memory_ordering_class_method
    def RELAXED():  # pylint:disable=no-method-argument
        return MemoryOrdering.RELAXED

    @_deprecated_memory_ordering_class_method
    def ACQUIRE():  # pylint:disable=no-method-argument
        return MemoryOrdering.ACQUIRE

    @_deprecated_memory_ordering_class_method
    def RELEASE():  # pylint:disable=no-method-argument
        return MemoryOrdering.RELEASE

    @_deprecated_memory_ordering_class_method
    def ACQ_REL():  # pylint:disable=no-method-argument
        return MemoryOrdering.ACQ_REL

    @_deprecated_memory_ordering_class_method
    def SEQ_CST():  # pylint:disable=no-method-argument
        return MemoryOrdering.SEQ_CST

    @staticmethod
    def to_string(v):
        warn("'memory_ordering' is deprecated. Use 'MemoryOrdering'.",
                DeprecationWarning, stacklevel=2)
        return MemoryOrdering.to_string(v)

# }}}


class MemoryScope:  # noqa
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


# {{{ memory_scope, MemoryScope compatiability

class _deprecated_memory_scope_class_method(object):  # noqa
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, klass):
        warn("'memory_scope' is deprecated. Use 'MemoryScope'.",
                DeprecationWarning, stacklevel=2)
        return self.f()


class memory_scope(object):  # noqa
    """Deprecated. Use :class:`MemoryScope` instead.
    """

    @_deprecated_memory_scope_class_method
    def WORK_ITEM():  # pylint:disable=no-method-argument
        return MemoryScope.WORK_ITEM

    @_deprecated_memory_scope_class_method
    def WORK_GROUP():  # pylint:disable=no-method-argument
        return MemoryScope.WORK_GROUP

    @_deprecated_memory_scope_class_method
    def DEVICE():  # pylint:disable=no-method-argument
        return MemoryScope.DEVICE

    @_deprecated_memory_scope_class_method
    def ALL_SVM_DEVICES():  # pylint:disable=no-method-argument
        return MemoryScope.ALL_SVM_DEVICES

    @_deprecated_memory_scope_class_method
    def auto():  # pylint:disable=no-method-argument
        return MemoryScope.auto

    @staticmethod
    def to_string(v):
        warn("'memory_scope' is deprecated. Use 'MemoryScope'.",
                DeprecationWarning, stacklevel=2)
        return MemoryScope.to_string(v)

# }}}


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

        super(OrderedAtomic, self).update_persistent_hash(key_hash, key_builder)
        key_builder.rec(key_hash, str(self.__class__.__name__))
        key_builder.rec(key_hash, self.ordering)
        key_builder.rec(key_hash, self.scope)

    def __eq__(self, other):
        return (super(OrderedAtomic, self).__eq__(other)
                and self.ordering == other.ordering
                and self.scope == other.scope)

    @property
    def op_name(self):
        raise NotImplementedError

    def __str__(self):
        return "%s[%s]%s/%s" % (
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
    op_name = 'init'


class AtomicUpdate(OrderedAtomic):
    """Properties of an atomic update. A subclass of :class:`OrderedAtomic`.

    .. attribute:: ordering

        One of the values from :class:`MemoryOrdering`

    .. attribute:: scope

        One of the values from :class:`MemoryScope`
    """
    op_name = 'update'


class AtomicLoad(OrderedAtomic):
    """Properties of an atomic load. A subclass of :class:`OrderedAtomic`.

    .. attribute:: ordering

        One of the values from :class:`MemoryOrdering`

    .. attribute:: scope

        One of the values from :class:`MemoryScope`
    """
    op_name = 'load'

# }}}


# {{{ instruction base class: expression rhs

class MultiAssignmentBase(InstructionBase):
    """An assignment instruction with an expression as a right-hand side."""

    fields = InstructionBase.fields | set(["expression"])
    pymbolic_fields = InstructionBase.pymbolic_fields | set(["expression"])

    @memoize_method
    def read_dependency_names(self):
        from loopy.symbolic import get_dependencies
        result = (
                super(MultiAssignmentBase, self).read_dependency_names()
                | get_dependencies(self.expression))

        for subscript_deps in self.assignee_subscript_deps():
            result = result | subscript_deps

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

    fields = MultiAssignmentBase.fields | \
            set("assignee temp_var_type atomicity".split())
    pymbolic_fields = MultiAssignmentBase.pymbolic_fields | set(["assignee"])

    def __init__(self,
            assignee, expression,
            id=None,
            depends_on=None,
            depends_on_is_final=None,
            groups=None,
            conflicts_with_groups=None,
            no_sync_with=None,
            within_inames_is_final=None,
            within_inames=None,
            tags=None,
            temp_var_type=Optional(), atomicity=(),
            priority=0, predicates=frozenset()):

        super(Assignment, self).__init__(
                id=id,
                depends_on=depends_on,
                depends_on_is_final=depends_on_is_final,
                groups=groups,
                conflicts_with_groups=conflicts_with_groups,
                no_sync_with=no_sync_with,
                within_inames_is_final=within_inames_is_final,
                within_inames=within_inames,
                priority=priority,
                predicates=predicates,
                tags=tags)

        from loopy.symbolic import parse
        if isinstance(assignee, str):
            assignee = parse(assignee)
        if isinstance(expression, str):
            expression = parse(expression)

        from pymbolic.primitives import Variable, Subscript, Lookup
        from loopy.symbolic import LinearSubscript
        if not isinstance(assignee, (Variable, Subscript, LinearSubscript, Lookup)):
            raise LoopyError("invalid lvalue '%s'" % assignee)

        self.assignee = assignee
        self.expression = expression

        self.temp_var_type = _check_and_fix_temp_var_type(temp_var_type)
        self.atomicity = atomicity

    # {{{ implement InstructionBase interface

    @memoize_method
    def assignee_var_names(self):
        return (_get_assignee_var_name(self.assignee),)

    def assignee_subscript_deps(self):
        return (_get_assignee_subscript_deps(self.assignee),)

    def with_transformed_expressions(self, f, assignee_f=None):
        if assignee_f is None:
            assignee_f = f

        return self.copy(
                assignee=assignee_f(self.assignee),
                expression=f(self.expression),
                predicates=frozenset(
                    f(pred) for pred in self.predicates))

    # }}}

    def __str__(self):
        result = "%s <- %s" % (self.assignee, self.expression)

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
    def assignees(self):
        return (self.assignee,)

    # }}}


class ExpressionInstruction(Assignment):
    def __init__(self, *args, **kwargs):
        warn("ExpressionInstruction is deprecated. Use Assignment instead",
                DeprecationWarning, stacklevel=2)

        super(ExpressionInstruction, self).__init__(*args, **kwargs)

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
        created from the assigment.

    .. automethod:: __init__
    """

    fields = MultiAssignmentBase.fields | \
            set("assignees temp_var_types".split())
    pymbolic_fields = MultiAssignmentBase.pymbolic_fields | set(["assignees"])

    def __init__(self,
            assignees, expression,
            id=None,
            depends_on=None,
            depends_on_is_final=None,
            groups=None,
            conflicts_with_groups=None,
            no_sync_with=None,
            within_inames_is_final=None,
            within_inames=None,
            tags=None,
            temp_var_types=None,
            priority=0, predicates=frozenset()):

        super(CallInstruction, self).__init__(
                id=id,
                depends_on=depends_on,
                depends_on_is_final=depends_on_is_final,
                groups=groups,
                conflicts_with_groups=conflicts_with_groups,
                no_sync_with=no_sync_with,
                within_inames_is_final=within_inames_is_final,
                within_inames=within_inames,
                priority=priority,
                predicates=predicates,
                tags=tags)

        from pymbolic.primitives import Call
        from loopy.symbolic import Reduction
        if not isinstance(expression, (Call, Reduction)) and expression is not None:
            raise LoopyError("'expression' argument to CallInstruction "
                    "must be a function call")

        from loopy.symbolic import parse
        if isinstance(assignees, str):
            assignees = parse(assignees)
        if not isinstance(assignees, tuple):
            raise LoopyError("'assignees' argument to CallInstruction "
                    "must be a tuple or a string parseable to a tuple"
                    "--got '%s'" % type(assignees).__name__)

        if isinstance(expression, str):
            expression = parse(expression)

        from pymbolic.primitives import Variable, Subscript
        from loopy.symbolic import LinearSubscript
        for assignee in assignees:
            if not isinstance(assignee, (Variable, Subscript, LinearSubscript)):
                raise LoopyError("invalid lvalue '%s'" % assignee)

        self.assignees = assignees
        self.expression = expression

        if temp_var_types is None:
            self.temp_var_types = (Optional(),) * len(self.assignees)
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

    def with_transformed_expressions(self, f, assignee_f=None):
        if assignee_f is None:
            assignee_f = f

        return self.copy(
                assignees=assignee_f(self.assignees),
                expression=f(self.expression),
                predicates=frozenset(
                    f(pred) for pred in self.predicates))

    # }}}

    def __str__(self):
        result = "%s: %s <- %s" % (self.id,
                ", ".join(str(a) for a in self.assignees),
                self.expression)

        options = self.get_str_options()
        if options:
            result += " {%s}" % (": ".join(options))

        if self.predicates:
            result += "\n" + 10*" " + "if (%s)" % " && ".join(self.predicates)
        return result

    @property
    def atomicity(self):
        # Function calls can impossibly be atomic, and even the result assignment
        # is troublesome, especially in the case of multiple results. Avoid the
        # issue altogether by disallowing atomicity.
        return ()

# }}}


def make_assignment(assignees, expression, temp_var_types=None, **kwargs):
    if temp_var_types is None:
        temp_var_types = (Optional(),) * len(assignees)

    if len(assignees) == 1:
        return Assignment(
                assignee=assignees[0],
                expression=expression,
                temp_var_type=temp_var_types[0],
                **kwargs)

    atomicity = kwargs.pop("atomicity", ())
    if atomicity:
        raise LoopyError("atomic operations with more than one "
                "left-hand side not supported")

    from pymbolic.primitives import Call
    from loopy.symbolic import Reduction
    if not isinstance(expression, (Call, Reduction)):
        raise LoopyError("right-hand side in multiple assignment must be "
                "function call or reduction, got: '%s'" % expression)

    return CallInstruction(
            assignees=assignees,
            expression=expression,
            temp_var_types=temp_var_types,
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
    pymbolic_fields = InstructionBase.pymbolic_fields | \
            set("assignees".split())

    def __init__(self,
            iname_exprs, code,
            read_variables=frozenset(), assignees=tuple(),
            id=None, depends_on=None, depends_on_is_final=None,
            groups=None, conflicts_with_groups=None,
            no_sync_with=None,
            within_inames_is_final=None, within_inames=None,
            priority=0,
            predicates=frozenset(), tags=None):
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
                within_inames_is_final=within_inames_is_final,
                within_inames=within_inames,
                priority=priority, predicates=predicates, tags=tags)

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
        result = (
                super(CInstruction, self).read_dependency_names()
                | frozenset(self.read_variables))

        from loopy.symbolic import get_dependencies
        for name, iname_expr in self.iname_exprs:
            result = result | get_dependencies(iname_expr)

        for subscript_deps in self.assignee_subscript_deps():
            result = result | subscript_deps

        return frozenset(result) | self.predicates

    def reduction_inames(self):
        return set()

    def assignee_var_names(self):
        return tuple(_get_assignee_var_name(expr) for expr in self.assignees)

    def assignee_subscript_deps(self):
        return tuple(
                _get_assignee_subscript_deps(a)
                for a in self.assignees)

    def with_transformed_expressions(self, f, assignee_f=None):
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
        first_line = "%s: %s <- CODE(%s|%s)" % (self.id,
                ", ".join(str(a) for a in self.assignees),
                ", ".join(str(x) for x in self.read_variables),
                ", ".join("%s=%s" % (name, expr)
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

    def reduction_inames(self):
        return frozenset()

    def assignee_var_names(self):
        return frozenset()

    def assignee_subscript_deps(self):
        return frozenset()

    def with_transformed_expressions(self, f, assignee_f=None):
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

    def __init__(self, id=None, depends_on=None, depends_on_is_final=None,
            groups=None, conflicts_with_groups=None,
            no_sync_with=None,
            within_inames_is_final=None, within_inames=None,
            priority=None,
            predicates=None, tags=None):
        super(NoOpInstruction, self).__init__(
                id=id,
                depends_on=depends_on,
                depends_on_is_final=depends_on_is_final,
                groups=groups,
                conflicts_with_groups=conflicts_with_groups,
                no_sync_with=no_sync_with,
                within_inames_is_final=within_inames_is_final,
                within_inames=within_inames,
                priority=priority,
                predicates=predicates,
                tags=tags)

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
        sychronize, for targets that require this (e.g. OpenCL)

    The textual syntax in a :mod:`loopy` kernel is::

        ... gbarrier
        ... lbarrier

    Note that the memory type :attr:`mem_kind` can be specified for local barriers::

        ... lbarrier {mem_kind=global}
    """

    fields = _DataObliviousInstruction.fields | set(["synchronization_kind",
                                                     "mem_kind"])

    def __init__(self, id, depends_on=None, depends_on_is_final=None,
            groups=None, conflicts_with_groups=None,
            no_sync_with=None,
            within_inames_is_final=None, within_inames=None,
            priority=None,
            predicates=None, tags=None, synchronization_kind="global",
            mem_kind="local"):

        if predicates:
            raise LoopyError("conditional barriers are not supported")

        super(BarrierInstruction, self).__init__(
                id=id,
                depends_on=depends_on,
                depends_on_is_final=depends_on_is_final,
                groups=groups,
                conflicts_with_groups=conflicts_with_groups,
                no_sync_with=no_sync_with,
                within_inames_is_final=within_inames_is_final,
                within_inames=within_inames,
                priority=priority,
                predicates=predicates,
                tags=tags
                )

        self.synchronization_kind = synchronization_kind
        self.mem_kind = mem_kind

    def __str__(self):
        first_line = "%s: ... %sbarrier" % (self.id, self.synchronization_kind[0])

        options = self.get_str_options()
        if self.synchronization_kind == "local":
            # add the memory kind
            options += ['mem_kind={}'.format(self.mem_kind)]
        if options:
            first_line += " {%s}" % (": ".join(options))

        return first_line

    @property
    def kind(self):
        from warnings import warn
        warn("BarrierInstruction.kind is deprecated, use synchronization_kind "
             "instead", DeprecationWarning, stacklevel=2)
        return self.synchronization_kind

# }}}


# {{{ key getters

def _get_insn_eq_key(insn):
    return insn._key_builder.key()


def _get_insn_hash_key(insn):
    return insn._key_builder.hash_key()

# }}}


# {{{ _check_and_fix_temp_var_type

def _check_and_fix_temp_var_type(temp_var_type, stacklevel=2):
    """Check temp_var_type for deprecated usage, and convert to the right value.
    """

    import loopy as lp

    if temp_var_type is None:
        warn("temp_var_type should be Optional() if no temporary, not None. "
             "This usage will be disallowed soon.",
             DeprecationWarning, stacklevel=1 + stacklevel)
        temp_var_type = lp.Optional()

    elif temp_var_type is lp.auto:
        warn("temp_var_type should be Optional(None) if "
             "unspecified, not auto. This usage will be disallowed soon.",
             DeprecationWarning, stacklevel=1 + stacklevel)
        temp_var_type = lp.Optional(None)

    elif not isinstance(temp_var_type, lp.Optional):
        warn("temp_var_type should be an instance of Optional. "
             "Other values for temp_var_type will be disallowed soon.",
             DeprecationWarning, stacklevel=1 + stacklevel)
        temp_var_type = lp.Optional(temp_var_type)

    return temp_var_type

# }}}


# vim: foldmethod=marker

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


import numpy as np
from pytools import Record, memoize_method
from loopy.kernel.array import ArrayBase
from loopy.diagnostic import LoopyError  # noqa


class auto(object):
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


class ParallelTag(IndexTag):
    pass


class HardwareParallelTag(ParallelTag):
    pass


class UniqueTag(IndexTag):
    @property
    def key(self):
        return type(self).__name__


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


class IlpBaseTag(ParallelTag):
    @property
    def key(self):
        return type(self).__name__


class UnrolledIlpTag(IlpBaseTag):
    def __str__(self):
        return "ilp.unr"


class LoopedIlpTag(IlpBaseTag):
    def __str__(self):
        return "ilp.seq"


class UnrollTag(IndexTag):
    def __str__(self):
        return "unr"

    @property
    def key(self):
        return type(self).__name__


class ForceSequentialTag(IndexTag):
    def __str__(self):
        return "forceseq"

    @property
    def key(self):
        return type(self).__name__


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
        from loopy.codegen import POD  # uses the correct complex type
        from cgen import RestrictPointer, Const
        from cgen.opencl import CLGlobal

        arg_decl = RestrictPointer(
                POD(target, dtype, self.name + name_suffix))

        if not is_written:
            arg_decl = Const(arg_decl)

        return CLGlobal(arg_decl)


class ConstantArg(ArrayBase, KernelArgument):
    min_target_axes = 0
    max_target_axes = 1

    def get_arg_decl(self, target, name_suffix, shape, dtype, is_written):
        from loopy.codegen import POD  # uses the correct complex type
        from cgen import RestrictPointer, Const
        from cgen.opencl import CLConstant

        arg_decl = RestrictPointer(
                POD(dtype, self.name + name_suffix))

        if not is_written:
            arg_decl = Const(arg_decl)

        return CLConstant(arg_decl)


class ImageArg(ArrayBase, KernelArgument):
    min_target_axes = 1
    max_target_axes = 3

    @property
    def dimensions(self):
        return len(self.dim_tags)

    def get_arg_decl(self, target, name_suffix, shape, dtype, is_written):
        if is_written:
            mode = "w"
        else:
            mode = "r"

        from cgen.opencl import CLImage
        return CLImage(self.num_target_axes(), mode, self.name+name_suffix)


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
    """

    min_target_axes = 0
    max_target_axes = 1

    allowed_extra_kwargs = [
            "storage_shape",
            "base_indices",
            "is_local"
            ]

    def __init__(self, name, dtype=None, shape=(), is_local=auto,
            dim_tags=None, offset=0, strides=None, order=None,
            base_indices=None, storage_shape=None):
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

        ArrayBase.__init__(self, name=name,
                dtype=dtype, shape=shape,
                dim_tags=dim_tags, order="C",
                base_indices=base_indices, is_local=is_local,
                storage_shape=storage_shape)

    @property
    def nbytes(self):
        from pytools import product
        return product(si for si in self.shape)*self.dtype.itemsize

    def get_arg_decl(self, target, name_suffix, shape, dtype, is_written):
        from cgen import ArrayOf
        from loopy.codegen import POD  # uses the correct complex type
        from cgen.opencl import CLLocal

        temp_var_decl = POD(target, self.dtype, self.name)

        # FIXME take into account storage_shape, or something like it
        storage_shape = self.shape

        if storage_shape:
            temp_var_decl = ArrayOf(temp_var_decl,
                    " * ".join(str(s) for s in storage_shape))

        if self.is_local:
            temp_var_decl = CLLocal(temp_var_decl)

        return temp_var_decl

    def __str__(self):
        return self.stringify(include_typename=False)

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

    .. attribute:: insn_deps

        a :class:`frozenset` of :attr:`id` values of :class:`Instruction` instances
         that *must* be executed before this one. Note that
        :func:`loopy.preprocess_kernel` (usually invoked automatically)
        augments this by adding dependencies on any writes to temporaries read
        by this instruction.

        May be *None* to invoke the default.

    .. attribute:: insn_deps_is_final

        A :class:`bool` determining whether :attr:`insn_deps` constitutes
        the *entire* list of iname dependencies.

        Defaults to *False*.

    .. attribute:: predicates

        a :class:`frozenset` of variable names whose truth values (as defined
        by C) determine whether this instruction should be run

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
        of statements.
    """

    fields = set("id insn_deps insn_deps_is_final predicates "
            "forced_iname_deps_is_final forced_iname_deps "
            "priority boostable boostable_into".split())

    def __init__(self, id, insn_deps, insn_deps_is_final,
            forced_iname_deps_is_final, forced_iname_deps, priority,
            boostable, boostable_into, predicates, tags):

        if forced_iname_deps_is_final is None:
            forced_iname_deps_is_final = False

        if insn_deps_is_final is None:
            insn_deps_is_final = False

        if tags is None:
            tags = ()

        assert isinstance(forced_iname_deps, frozenset)
        assert isinstance(insn_deps, frozenset) or insn_deps is None

        Record.__init__(self,
                id=id,
                insn_deps=insn_deps,
                insn_deps_is_final=insn_deps_is_final,
                forced_iname_deps_is_final=forced_iname_deps_is_final,
                forced_iname_deps=forced_iname_deps,
                priority=priority,
                boostable=boostable,
                boostable_into=boostable_into,
                predicates=predicates,
                tags=tags)

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

        if self.insn_deps:
            result.append("deps="+":".join(self.insn_deps))
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

# }}}


def _get_assignee_and_index(expr):
    from pymbolic.primitives import Variable, Subscript
    if isinstance(expr, Variable):
        return (expr.name, ())
    elif isinstance(expr, Subscript):
        agg = expr.aggregate
        assert isinstance(agg, Variable)

        idx = expr.index
        if not isinstance(idx, tuple):
            idx = (idx,)

        return (agg.name, idx)
    else:
        raise RuntimeError("invalid lvalue '%s'" % expr)


# {{{ expression instruction

class ExpressionInstruction(InstructionBase):
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
            forced_iname_deps_is_final=None,
            forced_iname_deps=frozenset(),
            insn_deps=None,
            insn_deps_is_final=None,
            boostable=None, boostable_into=None, tags=None,
            temp_var_type=None, priority=0, predicates=frozenset()):

        InstructionBase.__init__(self,
                id=id,
                forced_iname_deps_is_final=forced_iname_deps_is_final,
                forced_iname_deps=forced_iname_deps,
                insn_deps=insn_deps,
                insn_deps_is_final=insn_deps_is_final,
                boostable=boostable,
                boostable_into=boostable_into,
                priority=priority,
                predicates=predicates,
                tags=tags)

        from loopy.symbolic import parse
        if isinstance(assignee, str):
            assignee = parse(assignee)
        if isinstance(expression, str):
            assignee = parse(expression)

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

        result = result | self.predicates

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

# }}}


def _remove_common_indentation(code):
    if "\n" not in code:
        return code

    # accommodate pyopencl-ish syntax highlighting
    code = code.lstrip("//CL//")

    if not code.startswith("\n"):
        return code

    lines = code.split("\n")
    while lines[0].strip() == "":
        lines.pop(0)
    while lines[-1].strip() == "":
        lines.pop(-1)

    if lines:
        base_indent = 0
        while lines[0][base_indent] in " \t":
            base_indent += 1

        for line in lines[1:]:
            if line[:base_indent].strip():
                raise ValueError("inconsistent indentation")

    return "\n".join(line[base_indent:] for line in lines)


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
            id=None, insn_deps=None, insn_deps_is_final=None,
            forced_iname_deps_is_final=None, forced_iname_deps=frozenset(),
            priority=0, boostable=None, boostable_into=None,
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
                forced_iname_deps_is_final=forced_iname_deps_is_final,
                forced_iname_deps=forced_iname_deps,
                insn_deps=insn_deps,
                insn_deps_is_final=insn_deps_is_final,
                boostable=boostable,
                boostable_into=boostable_into,
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
        self.code = _remove_common_indentation(code)
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

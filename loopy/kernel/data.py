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


# {{{ iname tags

class IndexTag(Record):
    __slots__ = []

    def __hash__(self):
        raise RuntimeError("use .key to hash index tags")


class ParallelTag(IndexTag):
    pass


class HardwareParallelTag(ParallelTag):
    pass


class UniqueTag(IndexTag):
    @property
    def key(self):
        return type(self)


class AxisTag(UniqueTag):
    __slots__ = ["axis"]

    def __init__(self, axis):
        Record.__init__(self,
                axis=axis)

    @property
    def key(self):
        return (type(self), self.axis)

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
    pass


class AutoFitLocalIndexTag(AutoLocalIndexTagBase):
    def __str__(self):
        return "l.auto"


class IlpBaseTag(ParallelTag):
    pass


class UnrolledIlpTag(IlpBaseTag):
    def __str__(self):
        return "ilp.unr"


class LoopedIlpTag(IlpBaseTag):
    def __str__(self):
        return "ilp.seq"


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
    pass


class GlobalArg(ArrayBase, KernelArgument):
    min_target_axes = 0
    max_target_axes = 1

    def get_arg_decl(self, name_suffix, shape, dtype, is_written):
        from loopy.codegen import POD  # uses the correct complex type
        from cgen import RestrictPointer, Const
        from cgen.opencl import CLGlobal

        arg_decl = RestrictPointer(
                POD(dtype, self.name + name_suffix))

        if not is_written:
            arg_decl = Const(arg_decl)

        return CLGlobal(arg_decl)


class ConstantArg(ArrayBase, KernelArgument):
    min_target_axes = 0
    max_target_axes = 1

    def get_arg_decl(self, name_suffix, shape, dtype, is_written):
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

    def get_arg_decl(self, name_suffix, shape, dtype, is_written):
        if is_written:
            mode = "w"
        else:
            mode = "r"

        from cgen.opencl import CLImage
        return CLImage(self.num_target_axes(), mode, self.name+name_suffix)


class ValueArg(KernelArgument):
    def __init__(self, name, dtype=None, approximately=1000):
        if dtype is not None:
            dtype = np.dtype(dtype)

        Record.__init__(self, name=name, dtype=dtype,
                approximately=approximately)

    def __str__(self):
        return "%s: ValueArg, type %s" % (self.name, self.dtype)

    def __repr__(self):
        return "<%s>" % self.__str__()

# }}}


# {{{ temporary variable

class TemporaryVariable(ArrayBase):
    __doc__ = ArrayBase.__doc__ + """
    .. attribute:: storage_shape
    .. attribute:: base_indices
    .. attribute:: is_local
    """

    min_target_axes = 0
    max_target_axes = 1

    allowed_extra_kwargs = [
            "storage_shape",
            "base_indices",
            "is_local"
            ]

    def __init__(self, name, dtype, shape, is_local,
            dim_tags=None, offset=0, strides=None, order=None,
            base_indices=None, storage_shape=None):
        if base_indices is None:
            base_indices = (0,) * len(shape)

        ArrayBase.__init__(self, name=name, dtype=dtype, shape=shape,
                dim_tags=dim_tags, order="C",
                base_indices=base_indices, is_local=is_local,
                storage_shape=storage_shape)

    @property
    def nbytes(self):
        from pytools import product
        return product(si for si in self.shape)*self.dtype.itemsize

    def get_arg_decl(self, name_suffix, shape, dtype, is_written):
        from cgen import ArrayOf
        from loopy.codegen import POD  # uses the correct complex type
        from cgen.opencl import CLLocal

        temp_var_decl = POD(self.dtype, self.name)

        # FIXME take into account storage_shape, or something like it
        storage_shape = self.shape

        if storage_shape:
            temp_var_decl = ArrayOf(temp_var_decl,
                    " * ".join(str(s) for s in storage_shape))

        if self.is_local:
            temp_var_decl = CLLocal(temp_var_decl)

        return temp_var_decl

# }}}


# {{{ subsitution rule

class SubstitutionRule(Record):
    """
    :ivar name:
    :ivar arguments:
    :ivar expression:
    """

    def __init__(self, name, arguments, expression):
        assert isinstance(arguments, tuple)

        Record.__init__(self,
                name=name, arguments=arguments, expression=expression)

    def __str__(self):
        return "%s(%s) := %s" % (
                self.name, ", ".join(self.arguments), self.expression)

# }}}


# {{{ instruction

class Instruction(Record):
    """
    .. attribute:: id

        An (otherwise meaningless) identifier that is unique within
        a :class:`LoopKernel`.

    .. attribute:: assignee

    .. attribute:: expression

    .. attribute:: forced_iname_deps

        a set of inames that are added to the list of iname
        dependencies

    .. attribute:: insn_deps

        a list of ids of :class:`Instruction` instances that
        *must* be executed before this one. Note that loop scheduling augments this
        by adding dependencies on any writes to temporaries read by this instruction.
    .. attribute:: boostable

        Whether the instruction may safely be executed inside more loops than
        advertised without changing the meaning of the program. Allowed values
        are *None* (for unknown), *True*, and *False*.

    .. attribute:: boostable_into

        a set of inames into which the instruction
        may need to be boosted, as a heuristic help for the scheduler.

    .. attribute:: priority: scheduling priority

    The following instance variables are only used until
    :func:`loopy.make_kernel` is finished:

    .. attribute:: temp_var_type

        if not *None*, a type that will be assigned to the new temporary variable
        created from the assignee
    """

    def __init__(self,
            id, assignee, expression,
            forced_iname_deps=frozenset(), insn_deps=set(), boostable=None,
            boostable_into=None,
            temp_var_type=None, priority=0):

        from loopy.symbolic import parse
        if isinstance(assignee, str):
            assignee = parse(assignee)
        if isinstance(expression, str):
            assignee = parse(expression)

        assert isinstance(forced_iname_deps, frozenset)
        assert isinstance(insn_deps, set)

        Record.__init__(self,
                id=id, assignee=assignee, expression=expression,
                forced_iname_deps=forced_iname_deps,
                insn_deps=insn_deps, boostable=boostable,
                boostable_into=boostable_into,
                temp_var_type=temp_var_type,
                priority=priority)

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

    def __str__(self):
        result = "%s: %s <- %s" % (self.id,
                self.assignee, self.expression)

        if self.boostable is True:
            if self.boostable_into:
                result += " (boostable into '%s')" % ",".join(self.boostable_into)
            else:
                result += " (boostable)"
        elif self.boostable is False:
            result += " (not boostable)"
        elif self.boostable is None:
            pass
        else:
            raise RuntimeError("unexpected value for Instruction.boostable")

        options = []

        if self.insn_deps:
            options.append("deps="+":".join(self.insn_deps))
        if self.priority:
            options.append("priority=%d" % self.priority)

        return result

    @memoize_method
    def get_assignee_var_name(self):
        from pymbolic.primitives import Variable, Subscript

        if isinstance(self.assignee, Variable):
            var_name = self.assignee.name
        elif isinstance(self.assignee, Subscript):
            agg = self.assignee.aggregate
            assert isinstance(agg, Variable)
            var_name = agg.name
        else:
            raise RuntimeError("invalid lvalue '%s'" % self.assignee)

        return var_name

    @memoize_method
    def get_assignee_indices(self):
        from pymbolic.primitives import Variable, Subscript

        if isinstance(self.assignee, Variable):
            return ()
        elif isinstance(self.assignee, Subscript):
            result = self.assignee.index
            if not isinstance(result, tuple):
                result = (result,)
            return result
        else:
            raise RuntimeError("invalid lvalue '%s'" % self.assignee)

    @memoize_method
    def get_read_var_names(self):
        from loopy.symbolic import get_dependencies
        return get_dependencies(self.expression)

# }}}

# vim: foldmethod=marker

"""Elements of loopy's user-facing language."""

from __future__ import division

import numpy as np
from pytools import Record, memoize_method
import islpy as isl
from islpy import dim_type
from pymbolic import var





# {{{ index tags

class IndexTag(Record):
    __slots__ = []

    def __hash__(self):
        raise RuntimeError("use .key to hash index tags")




class SequentialTag(IndexTag):
    def __str__(self):
        return "seq"

class ParallelTag(IndexTag):
    pass

class UniqueTag(IndexTag):
    @property
    def key(self):
        return type(self)

class ParallelTagWithAxis(ParallelTag, UniqueTag):
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

class TAG_GROUP_IDX(ParallelTagWithAxis):
    print_name = "g"

class TAG_WORK_ITEM_IDX(ParallelTagWithAxis):
    print_name = "l"

class TAG_AUTO_WORK_ITEM_IDX(ParallelTag):
    def __str__(self):
        return "l.auto"

class TAG_ILP(ParallelTag):
    def __str__(self):
        return "ilp"

class BaseUnrollTag(IndexTag):
    pass

class TAG_UNROLL_STATIC(BaseUnrollTag):
    def __str__(self):
        return "unr"

class TAG_UNROLL_INCR(BaseUnrollTag):
    def __str__(self):
        return "unri"

def parse_tag(tag):
    if tag is None:
        return tag

    if isinstance(tag, IndexTag):
        return tag

    if not isinstance(tag, str):
        raise ValueError("cannot parse tag: %s" % tag)

    if tag in ["unrs", "unr"]:
        return TAG_UNROLL_STATIC()
    elif tag == "unri":
        return TAG_UNROLL_INCR()
    elif tag == "ilp":
        return TAG_ILP()
    elif tag.startswith("g."):
        return TAG_GROUP_IDX(int(tag[2:]))
    elif tag.startswith("l."):
        return TAG_WORK_ITEM_IDX(int(tag[2:]))
    else:
        raise ValueError("cannot parse tag: %s" % tag)

# }}}

# {{{ arguments

class ArrayArg:
    def __init__(self, name, dtype, strides=None, shape=None, order="C",
            offset=0, constant_mem=False):
        """
        All of the following are optional. Specify either strides or shape.

        :arg strides: like numpy strides, but in multiples of
            data type size
        :arg shape:
        :arg order:
        :arg offset: Offset from the beginning of the vector from which
            the strides are counted.
        """
        self.name = name
        self.dtype = np.dtype(dtype)

        if strides is not None and shape is not None:
            raise ValueError("can only specify one of shape and strides")

        if strides is not None:
            if isinstance(strides, str):
                from pymbolic import parse
                strides = parse(strides)

            strides = tuple(strides)

        if shape is not None:
            if isinstance(shape, str):
                from pymbolic import parse
                shape = parse(shape)

            from pyopencl.compyte.array import (
                    f_contiguous_strides,
                    c_contiguous_strides)

            if order == "F":
                strides = f_contiguous_strides(1, shape)
            elif order == "C":
                strides = c_contiguous_strides(1, shape)
            else:
                raise ValueError("invalid order: %s" % order)

        self.strides = strides
        self.offset = offset

        self.constant_mem = constant_mem

    def __repr__(self):
        return "<ArrayArg '%s' of type %s>" % (self.name, self.dtype)



class ImageArg:
    def __init__(self, name, dtype, dimensions):
        self.name = name
        self.dtype = np.dtype(dtype)
        self.dimensions = dimensions

    def __repr__(self):
        return "<ImageArg '%s' of type %s>" % (self.name, self.dtype)


class ScalarArg:
    def __init__(self, name, dtype, approximately=None):
        self.name = name
        self.dtype = np.dtype(dtype)
        self.approximately = approximately

    def __repr__(self):
        return "<ScalarArg '%s' of type %s>" % (self.name, self.dtype)

# }}}

# {{{ temporary variable

class TemporaryVariable(Record):
    """
    :ivar name:
    :ivar dtype:
    :ivar shape:
    :ivar base_indices:
    :ivar is_local:
    """

    def __init__(self, name, dtype, shape, is_local, base_indices=None):
        if base_indices is None:
            base_indices = (0,) * len(shape)

        Record.__init__(self, name=name, dtype=dtype, shape=shape, is_local=is_local)

    @property
    def nbytes(self):
        from pytools import product
        return product(self.shape)*self.dtype.itemsize

# }}}

# {{{ instruction

class Instruction(Record):
    #:ivar kernel: handle to the :class:`LoopKernel` of which this instruction
        #is a part. (not yet)
    """
    :ivar id: An (otherwise meaningless) identifier that is unique within 
        a :class:`LoopKernel`.
    :ivar assignee:
    :ivar expression:
    :ivar forced_iname_deps: a list of inames that are added to the list of iname
        dependencies
    :ivar insn_deps: a list of ids of :class:`Instruction` instances that
        *must* be executed before this one. Note that loop scheduling augments this
        by adding dependencies on any writes to temporaries read by this instruction.
    :ivar iname_to_tag: a map from loop domain variables to subclasses
        of :class:`IndexTag`
    """
    def __init__(self,
            id, assignee, expression,
            forced_iname_deps=[], insn_deps=[],
            iname_to_tag={}):

        # {{{ find and properly tag reduction inames

        reduction_inames = set()

        from loopy.symbolic import ReductionCallbackMapper

        def map_reduction(expr, rec):
            rec(expr.expr)
            reduction_inames.update(expr.inames)

        ReductionCallbackMapper(map_reduction)(expression)

        if reduction_inames:
            iname_to_tag = iname_to_tag.copy()

            for iname in reduction_inames:
                tag = iname_to_tag.get(iname)
                if not (tag is None or isinstance(tag, SequentialTag)):
                    raise RuntimeError("inconsistency detected: "
                            "sequential/reduction iname '%s' was "
                            "tagged otherwise" % iname)

                iname_to_tag[iname] = SequentialTag()

        # }}}

        Record.__init__(self,
                id=id, assignee=assignee, expression=expression,
                forced_iname_deps=forced_iname_deps,
                insn_deps=insn_deps,
                iname_to_tag=dict(
                    (iname, parse_tag(tag))
                    for iname, tag in iname_to_tag.iteritems()))

        unused_tags = set(self.iname_to_tag.iterkeys()) - self.all_inames()
        if unused_tags:
            raise RuntimeError("encountered tags for unused inames: "
                    + ", ".join(unused_tags))

    @memoize_method
    def all_inames(self):
        from loopy.symbolic import IndexVariableFinder
        index_vars = (
                IndexVariableFinder()(self.expression)
                | IndexVariableFinder()(self.assignee))

        return index_vars | set(self.forced_iname_deps)

    @memoize_method
    def sequential_inames(self):
        result = set()

        for iname, tag in self.iname_to_tag.iteritems():
            if isinstance(tag, SequentialTag):
                result.add(iname)

        return result

    def __str__(self):
        loop_descrs = []
        for iname in sorted(self.all_inames()):
            tag = self.iname_to_tag.get(iname)

            if tag is None:
                loop_descrs.append(iname)
            else:
                loop_descrs.append("%s: %s" % (iname, tag))

        result = "%s: %s <- %s\n    [%s]" % (self.id,
                self.assignee, self.expression, ", ".join(loop_descrs))

        if self.insn_deps:
            result += "\n    : " + ", ".join(self.insn_deps)

        return result

# }}}

# {{{ reduction operations

class ReductionOperation(object):
    """
    :ivar neutral_element:
    :ivar dtype:
    """

    def __call__(self, operand1, operand2):
        raise NotImplementedError

class TypedReductionOperation(ReductionOperation):
    def __init__(self, dtype):
        self.dtype = dtype

    def __str__(self):
        return (type(self).__name__.replace("ReductionOperation", "").lower()
                + "_" + str(self.dtype))

class SumReductionOperation(TypedReductionOperation):
    neutral_element = 0

    def __call__(self, operand1, operand2):
        return operand1 + operand2

class ProductReductionOperation(TypedReductionOperation):
    neutral_element = 1

    def __call__(self, operand1, operand2):
        return operand1 * operand2

class FloatingPointMaxOperation(TypedReductionOperation):
    neutral_element = -var("INFINITY")

    def __call__(self, operand1, operand2):
        return var("max")(operand1, operand2)

class FloatingPointMaxOperation(TypedReductionOperation):
    # OpenCL 1.1, section 6.11.2
    neutral_element = -var("INFINITY")

    def __call__(self, operand1, operand2):
        from pymbolic.primitives import FunctionSymbol
        return FunctionSymbol("max")(operand1, operand2)

class FloatingPointMinOperation(TypedReductionOperation):
    # OpenCL 1.1, section 6.11.2
    neutral_element = var("INFINITY")

    def __call__(self, operand1, operand2):
        from pymbolic.primitives import FunctionSymbol
        return FunctionSymbol("min")(operand1, operand2)




_REDUCTION_OPS = {
        "sum": SumReductionOperation,
        "product": ProductReductionOperation,
        "fpmax": FloatingPointMaxOperation,
        "fpmin": FloatingPointMinOperation,
        }

_REDUCTION_OP_PARSERS = [
        ]


def register_reduction_parser(parser):
    """Register a new :class:`ReductionOperation`.

    :arg parser: A function that receives a string and returns
        a subclass of ReductionOperation.
    """
    _REDUCTION_OP_PARSERS.append(parser)

def parse_reduction_op(name):
    import re
    red_op_match = re.match(r"^([a-z]+)_([a-z0-9]+)$", name)
    if red_op_match:
        op_name = red_op_match.group(1)
        op_type = red_op_match.group(2)
        try:
            op_dtype = np.dtype(op_type)
        except TypeError:
            op_dtype = None

        if op_name in _REDUCTION_OPS and op_dtype is not None:
            return _REDUCTION_OPS[op_name](op_dtype)

    for parser in _REDUCTION_OP_PARSERS:
        result = parser(name)
        if result is not None:
            return result

    raise RuntimeError("could not parse reudction operation '%s'" % name)

# }}}

# {{{ loop kernel object

class LoopKernel(Record):
    """
    :ivar device: :class:`pyopencl.Device`
    :ivar domain: :class:`islpy.BasicSet`
    :ivar instructions:
    :ivar args:
    :ivar schedule:
    :ivar name:
    :ivar preamble:
    :ivar assumptions: the initial implemented_domain, captures assumptions
        on the parameters. (an isl.Set)
    :ivar iname_slab_increments: a dictionary mapping inames to (lower_incr,
        upper_incr) tuples that will be separated out in the execution to generate
        'bulk' slabs with fewer conditionals.
    :ivar temporary_variables:
    :ivar workgroup_size:
    :ivar name_to_dim: A lookup table from inames to ISL-style
        (dim_type, index) tuples
    """

    def __init__(self, device, domain, instructions, args=None, schedule=None,
            name="loopy_kernel",
            preamble=None, assumptions=None,
            iname_slab_increments={},
            temporary_variables=[],
            workgroup_size=None,
            name_to_dim=None):
        """
        :arg domain: a :class:`islpy.BasicSet`, or a string parseable to a basic set by the isl.
            Example: "{[i,j]: 0<=i < 10 and 0<= j < 9}"
        """

        def parse_if_necessary(insn):
            from pymbolic import parse

            if isinstance(insn, Instruction):
                return insn
            if isinstance(insn, str):
                lhs, rhs = insn.split("=")
            elif isinstance(insn, tuple):
                lhs, rhs = insn

            if isinstance(lhs, str):
                lhs = parse(lhs)

            if isinstance(rhs, str):
                from loopy.symbolic import FunctionToPrimitiveMapper
                rhs = parse(rhs)
                rhs = FunctionToPrimitiveMapper()(rhs)

            return Instruction(
                    id=self.make_unique_instruction_id(insns),
                    assignee=lhs, expression=rhs)

        if isinstance(domain, str):
            ctx = isl.Context()
            domain = isl.Set.read_from_str(ctx, domain, nparam=-1)

        if name_to_dim is None:
            name_to_dim = domain.get_space().get_var_dict()

        insns = []
        for insn in instructions:
            # must construct list one-by-one to facilitate unique id generation
            insns.append(parse_if_necessary(insn))

        if len(set(insn.id for insn in insns)) != len(insns):
            raise RuntimeError("instruction ids do not appear to be unique")

        if assumptions is None:
            assumptions = isl.Set.universe(domain.get_space())
        elif isinstance(assumptions, str):
            s = domain.get_space()
            assumptions = isl.BasicSet.read_from_str(domain.get_ctx(),
                    "[%s] -> {[%s]: %s}"
                    % (",".join(s.get_name(dim_type.param, i)
                        for i in range(s.size(dim_type.param))),
                       ",".join(s.get_name(dim_type.set, i) 
                           for i in range(s.size(dim_type.set))),
                       assumptions),
                       nparam=-1)

        Record.__init__(self,
                device=device,  domain=domain, instructions=insns,
                args=args,
                schedule=schedule,
                name=name,
                preamble=preamble,
                assumptions=assumptions,
                iname_slab_increments=iname_slab_increments,
                temporary_variables=temporary_variables,
                workgroup_size=workgroup_size,
                name_to_dim=name_to_dim)

    def make_unique_instruction_id(self, insns=None, based_on="insn", extra_used_ids=set()):
        if insns is None:
            insns = self.instructions

        used_ids = set(insn.id for insn in insns) | extra_used_ids

        from loopy.tools import generate_unique_possibilities
        for id_str in generate_unique_possibilities(based_on):
            if id_str not in used_ids:
                return id_str

    def make_unique_var_name(self, based_on="var", extra_used_vars=set()):
        used_vars = (
                set(lv.name for lv in self.temporary_variables)
                | set(arg.name for arg in self.args)
                | set(self.name_to_dim.keys())
                | extra_used_vars)

        from loopy.tools import generate_unique_possibilities
        for var_name in generate_unique_possibilities(based_on):
            if var_name not in used_vars:
                return var_name

    @property
    @memoize_method
    def dim_to_name(self):
        from pytools import reverse_dict
        return reverse_dict(self.name_to_dim)

    @property
    @memoize_method
    def id_to_insn(self):
        return dict((insn.id, insn) for insn in self.instructions)

    @property
    @memoize_method
    def space(self):
        return self.domain.get_space()

    @property
    @memoize_method
    def tag_key_to_iname(self):
        return dict(
                (tag.key, iname)
                for iname, tag in self.iname_to_tag.iteritems()
                if isinstance(tag, UniqueTag))

    @property
    @memoize_method
    def arg_dict(self):
        return dict((arg.name, arg) for arg in self.args)

    @property
    @memoize_method
    def scalar_loop_args(self):
        if self.args is None:
            return []
        else:
            loop_arg_names = [self.space.get_name(dim_type.param, i)
                    for i in range(self.space.size(dim_type.param))]
            return [arg.name for arg in self.args if isinstance(arg, ScalarArg)
                    if arg.name in loop_arg_names]

    @memoize_method
    def all_inames(self):
        from islpy import dim_type
        return set(self.space.get_var_dict(dim_type.set).iterkeys())

    def inames_by_tag_type(self, tag_type):
        return [iname for iname in self.all_inames()
                if isinstance(self.iname_to_tag.get(iname), tag_type)]

    def ordered_inames_by_tag_type(self, tag_type):
        result = []
        from itertools import count
        for i in count():
            try:
                dim = self.tag_key_to_iname[tag_type(i).key]
            except KeyError:
                return result
            else:
                result.append(dim)

    @memoize_method
    def get_bounds_constraints(self, iname, admissible_vars, allow_parameters):
        """Get an overapproximation of the loop bounds for the variable *iname*."""

        from loopy.codegen.bounds import get_bounds_constraints
        return get_bounds_constraints(self.domain, iname, admissible_vars,
                allow_parameters)

    @memoize_method
    def get_bounds(self, iname, admissible_vars, allow_parameters):
        """Get an overapproximation of the loop bounds for the variable *iname*."""

        from loopy.codegen.bounds import get_bounds
        return get_bounds(self.domain, iname, admissible_vars, allow_parameters)

    def tag_type_lengths(self, tag_cls, allow_parameters):
        def get_length(iname):
            tag = self.iname_to_tag[iname]
            if tag.forced_length is not None:
                return tag.forced_length

            lower, upper, equality = self.get_bounds(iname, (iname,), 
                    allow_parameters=allow_parameters)
            return upper-lower

        return [get_length(iname)
                for iname in self.ordered_inames_by_tag_type(tag_cls)]

    def tag_or_iname_to_iname(self, s):
        try:
            tag = parse_tag(s)
        except ValueError:
            pass
        else:
            return self.tag_key_to_iname[tag.key]

        if s not in self.all_inames():
            raise RuntimeError("invalid index name '%s'" % s)

        return s

    def local_mem_use(self):
        return sum(lv.nbytes for lv in self.temporary_variables
                if lv.is_local)

# }}}


# vim: foldmethod=marker

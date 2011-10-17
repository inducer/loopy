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

class IlpTag(ParallelTag):
    def __str__(self):
        return "ilp"

class UnrollTag(IndexTag):
    def __str__(self):
        return "unr"

def parse_tag(tag):
    if tag is None:
        return tag

    if isinstance(tag, IndexTag):
        return tag

    if not isinstance(tag, str):
        raise ValueError("cannot parse tag: %s" % tag)

    if tag in ["unr"]:
        return UnrollTag()
    elif tag == "ilp":
        return IlpTag()
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
    :ivar storage_shape:
    :ivar base_indices:
    :ivar is_local:
    """

    def __init__(self, name, dtype, shape, is_local, base_indices=None,
            storage_shape=None):
        if base_indices is None:
            base_indices = (0,) * len(shape)

        Record.__init__(self, name=name, dtype=dtype, shape=shape, is_local=is_local,
                base_indices=base_indices,
                storage_shape=storage_shape)

    @property
    def nbytes(self):
        from pytools import product
        return product(self.shape)*self.dtype.itemsize

# }}}

# {{{ instruction

class Instruction(Record):
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
    :ivar idempotent: Whether the instruction may be executed repeatedly (while obeying
        dependencies) without changing the meaning of the program.
    """
    def __init__(self,
            id, assignee, expression, idempotent,
            forced_iname_deps=[], insn_deps=[]):

        assert isinstance(idempotent, bool)

        Record.__init__(self,
                id=id, assignee=assignee, expression=expression,
                forced_iname_deps=forced_iname_deps,
                insn_deps=insn_deps, idempotent=idempotent)

    @memoize_method
    def all_inames(self):
        from loopy.symbolic import IndexVariableFinder
        index_vars = (
                IndexVariableFinder()(self.expression)
                | IndexVariableFinder()(self.assignee))

        return index_vars | set(self.forced_iname_deps)

    @memoize_method
    def sequential_inames(self, iname_to_tag):
        result = set()

        for iname in self.all_inames():
            tag = iname_to_tag.get(iname)
            if isinstance(tag, SequentialTag):
                result.add(iname)

        return result

    def __str__(self):
        result = "%s: %s <- %s\n    [%s]" % (self.id,
                self.assignee, self.expression, ", ".join(sorted(self.all_inames())))

        if self.insn_deps:
            result += "\n    : " + ", ".join(self.insn_deps)

        return result

    def get_assignee_var_name(self):
        from pymbolic.primitives import Variable, Subscript

        if isinstance(self.assignee, Variable):
            var_name = self.assignee.name
        elif isinstance(self.assignee, Subscript):
            var = self.assignee.aggregate
            assert isinstance(var, Variable)
            var_name = var.name
        else:
            raise RuntimeError("invalid lvalue '%s'" % self.assignee)

        return var_name


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

    return None

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
    :ivar iname_to_dim: A lookup table from inames to ISL-style
        (dim_type, index) tuples
    :ivar iname_to_tag:
    """

    def __init__(self, device, domain, instructions, args=None, schedule=None,
            name="loopy_kernel",
            preamble=None, assumptions=None,
            iname_slab_increments={},
            temporary_variables={},
            workgroup_size=None,
            iname_to_dim=None,
            iname_to_tag={}):
        """
        :arg domain: a :class:`islpy.BasicSet`, or a string parseable to a basic set by the isl.
            Example: "{[i,j]: 0<=i < 10 and 0<= j < 9}"
        """
        import re
        LABEL_DEP_RE = re.compile(
                r"^\s*(?:(?P<label>\w+):)?"
                "\s*(?:\[(?P<iname_deps>[\s\w,]+)\])?"
                "\s*(?P<lhs>.+)\s*=\s*(?P<rhs>.+?)\s*?"
                "(?:\:\s*(?P<insn_deps>[\s\w,]+))?$"
                )

        def parse_if_necessary(insn):
            from pymbolic import parse

            insn_deps = []
            forced_iname_deps = []
            label = "insn"

            if isinstance(insn, Instruction):
                return insn
            if isinstance(insn, str):
                label_dep_match = LABEL_DEP_RE.match(insn)
                if label_dep_match is None:
                    raise RuntimeError("insn parse error")

                groups = label_dep_match.groupdict()
                if groups["label"] is not None:
                    label = groups["label"]
                if groups["insn_deps"] is not None:
                    insn_deps = [dep.strip() for dep in groups["insn_deps"].split(",")]
                if groups["iname_deps"] is not None:
                    forced_iname_deps = [dep.strip() for dep in groups["iname_deps"].split(",")]

                lhs = parse(groups["lhs"])
                from loopy.symbolic import FunctionToPrimitiveMapper
                rhs = FunctionToPrimitiveMapper()(parse(groups["rhs"]))

            return Instruction(
                    id=self.make_unique_instruction_id(insns, based_on=label),
                    insn_deps=insn_deps,
                    forced_iname_deps=forced_iname_deps,
                    assignee=lhs, expression=rhs,
                    idempotent=True)

        if isinstance(domain, str):
            ctx = isl.Context()
            domain = isl.Set.read_from_str(ctx, domain)

        if iname_to_dim is None:
            iname_to_dim = domain.get_space().get_var_dict()

        insns = []
        for insn in instructions:
            # must construct list one-by-one to facilitate unique id generation
            insns.append(parse_if_necessary(insn))

        if len(set(insn.id for insn in insns)) != len(insns):
            raise RuntimeError("instruction ids do not appear to be unique")

        # {{{ find and properly tag reduction inames

        reduction_inames = set()

        from loopy.symbolic import ReductionCallbackMapper

        def map_reduction(expr, rec):
            rec(expr.expr)
            reduction_inames.update(expr.inames)

        for insn in insns:
            ReductionCallbackMapper(map_reduction)(insn.expression)

        iname_to_tag = iname_to_tag.copy()

        if reduction_inames:
            for iname in reduction_inames:
                tag = iname_to_tag.get(iname)
                if not (tag is None or isinstance(tag, SequentialTag)):
                    raise RuntimeError("inconsistency detected: "
                            "sequential/reduction iname '%s' was "
                            "tagged otherwise" % iname)

                iname_to_tag[iname] = SequentialTag()

        # }}}

        if assumptions is None:
            assumptions_space = domain.get_space()
            assumptions_space = assumptions_space.remove_dims(
                    dim_type.set, 0, assumptions_space.dim(dim_type.set))
            assumptions = isl.Set.universe(assumptions_space)
        elif isinstance(assumptions, str):
            s = domain.get_space()
            assumptions = isl.BasicSet.read_from_str(domain.get_ctx(),
                    "[%s] -> { : %s}"
                    % (",".join(s.get_dim_name(dim_type.param, i)
                        for i in range(s.dim(dim_type.param))),
                        assumptions))

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
                iname_to_dim=iname_to_dim,
                iname_to_tag=iname_to_tag)

    def make_unique_instruction_id(self, insns=None, based_on="insn", extra_used_ids=set()):
        if insns is None:
            insns = self.instructions

        used_ids = set(insn.id for insn in insns) | extra_used_ids

        from loopy.tools import generate_unique_possibilities
        for id_str in generate_unique_possibilities(based_on):
            if id_str not in used_ids:
                return id_str

    @memoize_method
    def get_written_variables(self):
        return set(
            insn.get_assignee_var_name()
            for insn in self.instructions)

    def make_unique_var_name(self, based_on="var", extra_used_vars=set()):
        used_vars = (
                set(self.temporary_variables.iterkeys())
                | set(arg.name for arg in self.args)
                | set(self.iname_to_dim.keys())
                | extra_used_vars)

        from loopy.tools import generate_unique_possibilities
        for var_name in generate_unique_possibilities(based_on):
            if var_name not in used_vars:
                return var_name

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
    def arg_dict(self):
        return dict((arg.name, arg) for arg in self.args)

    @property
    @memoize_method
    def scalar_loop_args(self):
        if self.args is None:
            return []
        else:
            loop_arg_names = [self.space.get_dim_name(dim_type.param, i)
                    for i in range(self.space.dim(dim_type.param))]
            return [arg.name for arg in self.args if isinstance(arg, ScalarArg)
                    if arg.name in loop_arg_names]

    @memoize_method
    def all_inames(self):
        from islpy import dim_type
        return set(self.space.get_var_dict(dim_type.set).iterkeys())

    @memoize_method
    def get_iname_bounds(self, iname):
        lower_bound_pw_aff = (self.domain
                .intersect(self.assumptions)
                .dim_min(self.iname_to_dim[iname][1])
                .coalesce())
        upper_bound_pw_aff = (self.domain
                .intersect(self.assumptions)
                .dim_max(self.iname_to_dim[iname][1])
                .coalesce())

        class BoundsRecord(Record):
            pass

        size = (upper_bound_pw_aff - lower_bound_pw_aff + 1)
        size = size.intersect_domain(self.assumptions)

        return BoundsRecord(
                lower_bound_pw_aff=lower_bound_pw_aff,
                upper_bound_pw_aff=upper_bound_pw_aff,
                size=size)

    @memoize_method
    def get_constant_iname_length(self, iname):
        from loopy.isl_helpers import static_max_of_pw_aff
        from loopy.symbolic import aff_to_expr
        return int(aff_to_expr(static_max_of_pw_aff(
                self.get_iname_bounds(iname).size,
                constants_only=True)))

    @memoize_method
    def get_grid_sizes(self, ignore_auto=False):
        all_inames_by_insns = set()
        for insn in self.instructions:
            all_inames_by_insns |= insn.all_inames()

        if all_inames_by_insns != self.all_inames():
            raise RuntimeError("inames collected from instructions "
                    "do not match domain inames")

        global_sizes = {}
        local_sizes = {}

        from loopy.kernel import (
                GroupIndexTag, LocalIndexTag,
                AutoLocalIndexTagBase)

        for iname in self.all_inames():
            tag = self.iname_to_tag.get(iname)

            if isinstance(tag, GroupIndexTag):
                tgt_dict = global_sizes
            elif isinstance(tag, LocalIndexTag):
                tgt_dict = local_sizes
            elif isinstance(tag, AutoLocalIndexTagBase) and not ignore_auto:
                raise RuntimeError("cannot find grid sizes if automatic local index tags are "
                        "present")
            else:
                tgt_dict = None

            if tgt_dict is None:
                continue

            size = self.get_iname_bounds(iname).size

            if tag.axis in tgt_dict:
                size = tgt_dict[tag.axis].max(size)

            from loopy.isl_helpers import static_max_of_pw_aff
            try:
                # insist block size is constant
                size = static_max_of_pw_aff(size, 
                        constants_only=isinstance(tag, LocalIndexTag))
            except ValueError:
                pass

            tgt_dict[tag.axis] = size

        max_dims = self.device.max_work_item_dimensions

        def to_dim_tuple(size_dict, which):
            size_list = []
            sorted_axes = sorted(size_dict.iterkeys())
            while sorted_axes:
                cur_axis = sorted_axes.pop(0)
                while cur_axis > len(size_list):
                    from loopy import LoopyAdvisory
                    from warnings import warn
                    warn("%s axis %d unassigned--assuming length 1" % len(size_list),
                            LoopyAdvisory)
                    size_list.append(1)

                size_list.append(size_dict[cur_axis])

            if len(size_list) > max_dims:
                raise ValueError("more %s dimensions assigned than supported "
                        "by hardware (%d > %d)" % (which, len(size_list), max_dims))

            return tuple(size_list)

        return (to_dim_tuple(global_sizes, "global"),
                to_dim_tuple(local_sizes, "local"))

    def get_grid_sizes_as_exprs(self, ignore_auto=False):
        grid_size, group_size = self.get_grid_sizes(ignore_auto=ignore_auto)

        def tup_to_exprs(tup):
            from loopy.symbolic import pw_aff_to_expr
            return tuple(pw_aff_to_expr(i) for i in tup)

        return tup_to_exprs(grid_size), tup_to_exprs(group_size)

    def local_mem_use(self):
        return sum(lv.nbytes for lv in self.temporary_variables.itervalues()
                if lv.is_local)

    def __str__(self):
        lines = []

        for insn in self.instructions:
            lines.append(str(insn))
        lines.append("")
        for iname in sorted(self.all_inames()):
            lines.append("%s: %s" % (iname, self.iname_to_tag.get(iname)))

        return "\n".join(lines)

# }}}


# vim: foldmethod=marker

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

    if tag == "for":
        return None
    elif tag in ["unr"]:
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
        from loopy.symbolic import pw_aff_to_expr
        return product(pw_aff_to_expr(si) for si in self.shape)*self.dtype.itemsize

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
    :ivar boostable: Whether the instruction may safely be executed
        inside more loops than advertised without changing the meaning
        of the program. Allowed values are *None* (for unknwon), *True*, and *False*.

    The following two instance variables are only used until :func:`loopy.kernel.make_kernel` is
    finished:

    :ivar temp_var_type: if not None, a type that will be assigned to the new temporary variable
        created from the assignee
    :ivar duplicate_inames_and_tags: a list of inames used in the instruction that will be duplicated onto
        different inames.
    """
    def __init__(self,
            id, assignee, expression,
            forced_iname_deps=[], insn_deps=[], boostable=None,
            temp_var_type=None, duplicate_inames_and_tags=[]):

        Record.__init__(self,
                id=id, assignee=assignee, expression=expression,
                forced_iname_deps=forced_iname_deps,
                insn_deps=insn_deps, boostable=boostable,
                temp_var_type=temp_var_type, duplicate_inames_and_tags=duplicate_inames_and_tags)

    @memoize_method
    def all_inames(self):
        from loopy.symbolic import IndexVariableFinder
        ivarf = IndexVariableFinder(include_reduction_inames=False)
        index_vars = (ivarf(self.expression) | ivarf(self.assignee))

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

        if self.boostable == True:
            result += " (boostable)"
        elif self.boostable == False:
            result += " (not boostable)"
        elif self.boostable is None:
            result += " (boostability unknown)"
        else:
            raise RuntimeError("unexpected value for Instruction.boostable")

        if self.insn_deps:
            result += "\n    : " + ", ".join(self.insn_deps)

        return result

    @memoize_method
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

    @memoize_method
    def get_assignee_indices(self):
        from pymbolic.primitives import Variable, Subscript

        if isinstance(self.assignee, Variable):
            result = ()
        elif isinstance(self.assignee, Subscript):
            result = self.assignee.index
        else:
            raise RuntimeError("invalid lvalue '%s'" % self.assignee)

        return result

    @memoize_method
    def get_read_var_names(self):
        from loopy.symbolic import DependencyMapper
        return set(var.name for var in
                DependencyMapper(composite_leaves=False)(self.expression))

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
        assert iname_to_dim is None

        import re

        if isinstance(domain, str):
            ctx = isl.Context()
            domain = isl.Set.read_from_str(ctx, domain)

        DUP_ENTRY_RE = re.compile(
                r"^\s*(?P<iname>\w+)\s*(?:\:\s*(?P<tag>[\w.]+))?\s*$")
        LABEL_DEP_RE = re.compile(
                r"^\s*(?:(?P<label>\w+):)?"
                "\s*(?:\["
                    "(?P<iname_deps>[\s\w,]*)"
                    "(?:\|(?P<duplicate_inames_and_tags>[\s\w,:.]*))?"
                "\])?"
                "\s*(?:\<(?P<temp_var_type>.+)\>)?"
                "\s*(?P<lhs>.+)\s*=\s*(?P<rhs>.+?)"
                "\s*?(?:\:\s*(?P<insn_deps>[\s\w,]+))?$"
                )

        def parse_if_necessary(insn):
            from pymbolic import parse


            if isinstance(insn, Instruction):
                return insn
            if isinstance(insn, str):
                label_dep_match = LABEL_DEP_RE.match(insn)
                if label_dep_match is None:
                    raise RuntimeError("insn parse error")

                groups = label_dep_match.groupdict()
                if groups["label"] is not None:
                    label = groups["label"]
                else:
                    label = "insn"
                if groups["insn_deps"] is not None:
                    insn_deps = [dep.strip() for dep in groups["insn_deps"].split(",")]
                else:
                    insn_deps = []

                if groups["iname_deps"] is not None:
                    forced_iname_deps = [dep.strip()
                            for dep in groups["iname_deps"].split(",")
                            if dep.strip()]
                else:
                    forced_iname_deps = []

                if groups["duplicate_inames_and_tags"] is not None:
                    dup_entries = [
                            dep.strip() for dep in groups["duplicate_inames_and_tags"].split(",")]
                    duplicate_inames_and_tags = []
                    for dup_entry in dup_entries:
                        if not dup_entry:
                            continue

                        dup_entry_match = DUP_ENTRY_RE.match(dup_entry)
                        if dup_entry_match is None:
                            raise RuntimeError(
                                    "could not parse iname duplication entry '%s'"
                                    % dup_entry)

                        dup_groups = dup_entry_match.groupdict()
                        dup_iname = dup_groups["iname"]
                        assert dup_iname
                        dup_tag = AutoFitLocalIndexTag()
                        if dup_groups["tag"] is not None:
                            dup_tag = parse_tag(dup_groups["tag"])

                        duplicate_inames_and_tags.append((dup_iname, dup_tag))
                else:
                    duplicate_inames_and_tags = []

                if groups["temp_var_type"] is not None:
                    temp_var_type = groups["temp_var_type"]
                else:
                    temp_var_type = None

                lhs = parse(groups["lhs"])
                from loopy.symbolic import FunctionToPrimitiveMapper
                rhs = FunctionToPrimitiveMapper()(parse(groups["rhs"]))

            return Instruction(
                    id=self.make_unique_instruction_id(insns, based_on=label),
                    insn_deps=insn_deps,
                    forced_iname_deps=forced_iname_deps,
                    assignee=lhs, expression=rhs,
                    temp_var_type=temp_var_type,
                    duplicate_inames_and_tags=duplicate_inames_and_tags)

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
            assumptions_space = domain.get_space().params()
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
                iname_to_tag=iname_to_tag)

    def make_unique_instruction_id(self, insns=None, based_on="insn", extra_used_ids=set()):
        if insns is None:
            insns = self.instructions

        used_ids = set(insn.id for insn in insns) | extra_used_ids

        from loopy.tools import generate_unique_possibilities
        for id_str in generate_unique_possibilities(based_on):
            if id_str not in used_ids:
                return id_str

    @property
    @memoize_method
    def iname_to_dim(self):
        return self.domain.get_space().get_var_dict()

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
        dom_intersect_assumptions = (
                isl.align_spaces(self.assumptions, self.domain)
                & self.domain)
        lower_bound_pw_aff = (
                dom_intersect_assumptions
                .dim_min(self.iname_to_dim[iname][1])
                .coalesce())
        upper_bound_pw_aff = (
                dom_intersect_assumptions
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
            raise RuntimeError("inames collected from instructions (%s) "
                    "do not match domain inames (%s)"
                    % (", ".join(sorted(all_inames_by_insns)), 
                        ", ".join(sorted(self.all_inames()))))

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
                    warn("%s axis %d unassigned--assuming length 1" % (
                        which, len(size_list)), LoopyAdvisory)
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

    @memoize_method
    def local_var_names(self):
        return set(
                tv.name
            for tv in self.temporary_variables.itervalues()
            if tv.is_local)

    def local_mem_use(self):
        return sum(lv.nbytes for lv in self.temporary_variables.itervalues()
                if lv.is_local)

    def __str__(self):
        lines = []

        for iname in sorted(self.all_inames()):
            lines.append("%s: %s" % (iname, self.iname_to_tag.get(iname)))
        lines.append("")
        lines.append(str(self.domain))
        lines.append("")
        for insn in self.instructions:
            lines.append(str(insn))

        return "\n".join(lines)

# }}}




def find_var_base_indices_and_shape_from_inames(domain, inames):
    base_indices = []
    shape = []

    iname_to_dim = domain.get_space().get_var_dict()
    for iname in inames:
        lower_bound_pw_aff = domain.dim_min(iname_to_dim[iname][1])
        upper_bound_pw_aff = domain.dim_max(iname_to_dim[iname][1])

        from loopy.isl_helpers import static_max_of_pw_aff
        from loopy.symbolic import pw_aff_to_expr

        shape.append(static_max_of_pw_aff(
                upper_bound_pw_aff - lower_bound_pw_aff + 1, constants_only=True))
        base_indices.append(pw_aff_to_expr(lower_bound_pw_aff))

    return base_indices, shape




# {{{ count number of uses of each reduction iname

def count_reduction_iname_uses(insn):

    def count_reduction_iname_uses(expr, rec):
        rec(expr.expr)
        for iname in expr.inames:
            reduction_iname_uses[iname] = (
                    reduction_iname_uses.get(iname, 0)
                    + 1)

    from loopy.symbolic import ReductionCallbackMapper
    cb_mapper = ReductionCallbackMapper(count_reduction_iname_uses)

    reduction_iname_uses = {}
    cb_mapper(insn.expression)

    return reduction_iname_uses




def make_kernel(*args, **kwargs):
    """Second pass of kernel creation. Think about requests for iname duplication
    and temporary variable declaration received as part of string instructions.
    """

    knl = LoopKernel(*args, **kwargs)

    new_insns = []
    new_domain = knl.domain
    new_temp_vars = knl.temporary_variables.copy()
    new_iname_to_tag = knl.iname_to_tag.copy()

    newly_created_vars = set()

    # {{{ reduction iname duplication helper function

    def duplicate_reduction_inames(reduction_expr, rec):
        duplicate_inames = [iname
                for iname, tag in insn.duplicate_inames_and_tags]

        child = rec(reduction_expr.expr)
        new_red_inames = []
        did_something = False

        for iname in reduction_expr.inames:
            if iname in duplicate_inames:
                new_iname = knl.make_unique_var_name(iname, newly_created_vars)

                old_insn_inames.append(iname)
                new_insn_inames.append(new_iname)
                newly_created_vars.add(new_iname)
                new_red_inames.append(new_iname)
                reduction_iname_uses[iname] -= 1
                did_something = True
            else:
                new_red_inames.append(iname)

        if did_something:
            from loopy.symbolic import SubstitutionMapper
            from pymbolic.mapper.substitutor import make_subst_func
            from pymbolic import var
            subst_dict = dict(
                    (old_iname, var(new_iname))
                    for old_iname, new_iname in zip(
                        reduction_expr.inames, new_red_inames))
            subst_map = SubstitutionMapper(make_subst_func(subst_dict))

            child = subst_map(child)

        from loopy.symbolic import Reduction
        return Reduction(
                operation=reduction_expr.operation,
                inames=tuple(new_red_inames),
                expr=child)

    # }}}

    for insn in knl.instructions:
        # {{{ iname duplication

        if insn.duplicate_inames_and_tags:
            # {{{ duplicate non-reduction inames

            reduction_iname_uses = count_reduction_iname_uses(insn)

            duplicate_inames = [iname
                    for iname, tag in insn.duplicate_inames_and_tags
                    if iname not in reduction_iname_uses]
            new_iname_tags = [tag for iname, tag in insn.duplicate_inames_and_tags
                    if iname not in reduction_iname_uses]

            new_inames = [
                    knl.make_unique_var_name(
                        iname,
                        extra_used_vars=
                        newly_created_vars)
                    for iname in duplicate_inames]

            for iname, tag in zip(new_inames, new_iname_tags):
                new_iname_to_tag[iname] = tag

            newly_created_vars.update(new_inames)

            from loopy.isl_helpers import duplicate_axes
            new_domain = duplicate_axes(new_domain, duplicate_inames, new_inames)

            from loopy.symbolic import SubstitutionMapper
            from pymbolic.mapper.substitutor import make_subst_func
            old_to_new = dict(
                    (old_iname, var(new_iname))
                    for old_iname, new_iname in zip(duplicate_inames, new_inames))
            subst_map = SubstitutionMapper(make_subst_func(old_to_new))
            new_expression = subst_map(insn.expression)

            # }}}

            # {{{ duplicate reduction inames

            if len(duplicate_inames) < len(insn.duplicate_inames_and_tags):
                # there must've been requests to duplicate reduction inames
                old_insn_inames = []
                new_insn_inames = []

                from loopy.symbolic import ReductionCallbackMapper
                new_expression = (
                        ReductionCallbackMapper(duplicate_reduction_inames)
                        (new_expression))

                from loopy.isl_helpers import duplicate_axes
                for old, new in zip(old_insn_inames, new_insn_inames):
                    new_domain = duplicate_axes(new_domain, [old], [new])

            # }}}

            insn = insn.copy(
                    assignee=subst_map(insn.assignee),
                    expression=new_expression,
                    forced_iname_deps=[
                        old_to_new.get(iname, iname) for iname in insn.forced_iname_deps],
                    )

        # }}}

        # {{{ temporary variable creation

        if insn.temp_var_type is not None:
            assignee_name = insn.get_assignee_var_name()

            assignee_indices = []
            from pymbolic.primitives import Variable
            for index_expr in insn.get_assignee_indices():
                if (not isinstance(index_expr, Variable)
                        or not index_expr.name in insn.all_inames()):
                    raise RuntimeError(
                            "only plain inames are allowed in "
                            "the lvalue index when declaring the "
                            "variable '%s' in an instruction"
                            % assignee_name)

                assignee_indices.append(index_expr.name)

            from loopy.kernel import LocalIndexTagBase
            from pytools import any
            is_local = any(
                    isinstance(new_iname_to_tag.get(iname), LocalIndexTagBase)
                    for iname in assignee_indices)

            base_indices, shape = \
                    find_var_base_indices_and_shape_from_inames(
                            new_domain, assignee_indices)

            new_temp_vars[assignee_name] = TemporaryVariable(
                    name=assignee_name,
                    dtype=np.dtype(insn.temp_var_type),
                    is_local=is_local,
                    base_indices=base_indices,
                    shape=shape)

            newly_created_vars.add(assignee_name)

            insn = insn.copy(temp_var_type=None)

        # }}}

        new_insns.append(insn)

    return knl.copy(
            instructions=new_insns,
            domain=new_domain,
            temporary_variables=new_temp_vars,
            iname_to_tag=new_iname_to_tag)




# vim: foldmethod=marker

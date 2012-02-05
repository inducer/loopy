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
        self.shape = shape
        self.order = order

        self.constant_mem = constant_mem

    @property
    def dimensions(self):
        return len(self.shape)

    def __repr__(self):
        return "<ArrayArg '%s' of type %s>" % (self.name, self.dtype)

class ConstantArrayArg(ArrayArg):
    pass

class ImageArg:
    def __init__(self, name, dtype, dimensions=None, shape=None):
        self.name = name
        self.dtype = np.dtype(dtype)
        if shape is not None:
            if dimensions is not None:
                raise RuntimeError("cannot specify both shape and dimensions "
                        "in ImageArg")
            self.dimensions = len(shape)
            self.shape = shape
        else:
            if not isinstance(dimensions, int):
                raise RuntimeError("ImageArg: dimensions must be an integer")
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
        return product(si for si in self.shape)*self.dtype.itemsize

# }}}

# {{{ subsitution rule

class SubstitutionRule(Record):
    """
    :ivar name:
    :ivar arguments:
    :ivar expression:
    """

    def __init__(self, name, arguments, expression):
        Record.__init__(self,
                name=name, arguments=arguments, expression=expression)

    def __str__(self):
        return "%s(%s) := %s" % (
                self.name, ", ".join(self.arguments), self.expression)


# }}}

# {{{ instruction

class Instruction(Record):
    """
    :ivar id: An (otherwise meaningless) identifier that is unique within
        a :class:`LoopKernel`.
    :ivar assignee:
    :ivar expression:
    :ivar forced_iname_deps: a set of inames that are added to the list of iname
        dependencies
    :ivar insn_deps: a list of ids of :class:`Instruction` instances that
        *must* be executed before this one. Note that loop scheduling augments this
        by adding dependencies on any writes to temporaries read by this instruction.
    :ivar boostable: Whether the instruction may safely be executed
        inside more loops than advertised without changing the meaning
        of the program. Allowed values are *None* (for unknown), *True*, and *False*.
    :ivar boostable_into: a set of inames into which the instruction
        may need to be boosted, as a heuristic help for the scheduler.

    The following two instance variables are only used until :func:`loopy.make_kernel` is
    finished:

    :ivar temp_var_type: if not None, a type that will be assigned to the new temporary variable
        created from the assignee
    :ivar duplicate_inames_and_tags: a list of inames used in the instruction that will be duplicated onto
        different inames.
    """
    def __init__(self,
            id, assignee, expression,
            forced_iname_deps=set(), insn_deps=set(), boostable=None,
            boostable_into=None,
            temp_var_type=None, duplicate_inames_and_tags=[]):

        assert isinstance(forced_iname_deps, set)
        assert isinstance(insn_deps, set)

        Record.__init__(self,
                id=id, assignee=assignee, expression=expression,
                forced_iname_deps=forced_iname_deps,
                insn_deps=insn_deps, boostable=boostable,
                boostable_into=boostable_into,
                temp_var_type=temp_var_type, duplicate_inames_and_tags=duplicate_inames_and_tags)

    @memoize_method
    def reduction_inames(self):
        def map_reduction(expr, rec):
            rec(expr.expr)
            for iname in expr.untagged_inames:
                result.add(iname)

        from loopy.symbolic import ReductionCallbackMapper
        cb_mapper = ReductionCallbackMapper(map_reduction)

        result = set()
        cb_mapper(self.expression)

        return result

    def __str__(self):
        result = "%s: %s <- %s" % (self.id,
                self.assignee, self.expression)

        if self.boostable == True:
            if self.boostable_into:
                result += " (boostable into '%s')" % ",".join(self.boostable_into)
            else:
                result += " (boostable)"
        elif self.boostable == False:
            result += " (not boostable)"
        elif self.boostable is None:
            pass
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
    red_op_match = re.match(r"^([a-z]+)_([a-z0-9_]+)$", name)
    if red_op_match:
        op_name = red_op_match.group(1)
        op_type = red_op_match.group(2)

        try:
            op_dtype = np.dtype(op_type)
        except TypeError:
            op_dtype = None

        if op_dtype is None and op_type.startswith("vec_"):
            import pyopencl.array as cl_array
            try:
                op_dtype = getattr(cl_array.vec, op_type[4:])
            except AttributeError:
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
    :ivar local_sizes: A dictionary from integers to integers, mapping
        workgroup axes to ther sizes, e.g. *{0: 16}* forces axis 0 to be
        length 16.
    :ivar substitutions: a mapping from substitution names to :class:`SubstitutionRule`
        objects

    :ivar cache_manager:

    The following instance variables are only used until :func:`loopy.make_kernel` is
    finished:

    :ivar iname_to_tag_requests:
    """

    def __init__(self, device, domain, instructions, args=None, schedule=None,
            name="loopy_kernel",
            preamble=None, assumptions=None,
            iname_slab_increments={},
            temporary_variables={},
            local_sizes={},
            iname_to_tag={}, iname_to_tag_requests=None, substitutions={},
            cache_manager=None):
        """
        :arg domain: a :class:`islpy.BasicSet`, or a string parseable to a basic set by the isl.
            Example: "{[i,j]: 0<=i < 10 and 0<= j < 9}"
        """
        assert not iname_to_tag_requests

        import re

        if cache_manager is None:
            cache_manager = SetOperationCacheManager()

        if isinstance(domain, str):
            ctx = isl.Context()
            domain = isl.Set.read_from_str(ctx, domain)

        iname_to_tag_requests = {}

        INAME_ENTRY_RE = re.compile(
                r"^\s*(?P<iname>\w+)\s*(?:\:\s*(?P<tag>[\w.]+))?\s*$")
        INSN_RE = re.compile(
                r"^\s*(?:(?P<label>\w+):)?"
                "\s*(?:\["
                    "(?P<iname_deps_and_tags>[\s\w,:.]*)"
                    "(?:\|(?P<duplicate_inames_and_tags>[\s\w,:.]*))?"
                "\])?"
                "\s*(?:\<(?P<temp_var_type>.+?)\>)?"
                "\s*(?P<lhs>.+?)\s*(?<!\:)=\s*(?P<rhs>.+?)"
                "\s*?(?:\:\s*(?P<insn_deps>[\s\w,]+))?$"
                )
        SUBST_RE = re.compile(
                r"^\s*(?P<lhs>.+?)\s*:=\s*(?P<rhs>.+)\s*$"
                )

        def parse_iname_and_tag_list(s):
            dup_entries = [
                    dep.strip() for dep in s.split(",")]
            result = []
            for entry in dup_entries:
                if not entry:
                    continue

                entry_match = INAME_ENTRY_RE.match(entry)
                if entry_match is None:
                    raise RuntimeError(
                            "could not parse iname:tag entry '%s'"
                            % entry)

                groups = entry_match.groupdict()
                iname = groups["iname"]
                assert iname

                tag = None
                if groups["tag"] is not None:
                    tag = parse_tag(groups["tag"])

                result.append((iname, tag))

            return result

        # {{{ instruction parser

        def parse_if_necessary(insn):
            from pymbolic import parse

            if isinstance(insn, Instruction):
                insns.append(insn)
                return

            if not isinstance(insn, str):
                raise TypeError("Instructions must be either an Instruction "
                        "instance or a parseable string. got '%s' instead."
                        % type(insn))

            insn_match = INSN_RE.match(insn)
            subst_match = SUBST_RE.match(insn)
            if insn_match is not None and subst_match is not None:
                raise RuntimeError("insn parse error")

            if insn_match is not None:
                groups = insn_match.groupdict()
            elif subst_match is not None:
                groups = subst_match.groupdict()
            else:
                raise RuntimeError("insn parse error")

            lhs = parse(groups["lhs"])
            from loopy.symbolic import FunctionToPrimitiveMapper
            rhs = FunctionToPrimitiveMapper()(parse(groups["rhs"]))

            if insn_match is not None:
                if groups["label"] is not None:
                    label = groups["label"]
                else:
                    label = "insn"

                if groups["insn_deps"] is not None:
                    insn_deps = set(dep.strip() for dep in groups["insn_deps"].split(","))
                else:
                    insn_deps = set()

                if groups["iname_deps_and_tags"] is not None:
                    inames_and_tags = parse_iname_and_tag_list(
                            groups["iname_deps_and_tags"])
                    forced_iname_deps = set(iname for iname, tag in inames_and_tags)
                    iname_to_tag_requests.update(dict(inames_and_tags))
                else:
                    forced_iname_deps = set()

                if groups["duplicate_inames_and_tags"] is not None:
                    duplicate_inames_and_tags = parse_iname_and_tag_list(
                            groups["duplicate_inames_and_tags"])
                else:
                    duplicate_inames_and_tags = []

                if groups["temp_var_type"] is not None:
                    temp_var_type = groups["temp_var_type"]
                else:
                    temp_var_type = None

                insns.append(
                        Instruction(
                            id=self.make_unique_instruction_id(insns, based_on=label),
                            insn_deps=insn_deps,
                            forced_iname_deps=forced_iname_deps,
                            assignee=lhs, expression=rhs,
                            temp_var_type=temp_var_type,
                            duplicate_inames_and_tags=duplicate_inames_and_tags))
            elif subst_match is not None:
                from pymbolic.primitives import Variable, Call

                if isinstance(lhs, Variable):
                    subst_name = lhs.name
                    arg_names = []
                elif isinstance(lhs, Call):
                    if not isinstance(lhs.function, Variable):
                        raise RuntimeError("Invalid substitution rule left-hand side")
                    subst_name = lhs.function.name
                    arg_names = []

                    for arg in lhs.parameters:
                        if not isinstance(arg, Variable):
                            raise RuntimeError("Invalid substitution rule left-hand side")
                        arg_names.append(arg.name)
                else:
                    raise RuntimeError("Invalid substitution rule left-hand side")

                substitutions[subst_name] = SubstitutionRule(
                        name=subst_name,
                        arguments=arg_names,
                        expression=rhs)

        # }}}

        insns = []

        substitutions = substitutions.copy()

        for insn in instructions:
            # must construct list one-by-one to facilitate unique id generation
            parse_if_necessary(insn)

        if len(set(insn.id for insn in insns)) != len(insns):
            raise RuntimeError("instruction ids do not appear to be unique")

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
                local_sizes=local_sizes,
                iname_to_tag=iname_to_tag,
                iname_to_tag_requests=iname_to_tag_requests,
                substitutions=substitutions,
                cache_manager=cache_manager)

    def make_unique_instruction_id(self, insns=None, based_on="insn", extra_used_ids=set()):
        if insns is None:
            insns = self.instructions

        used_ids = set(insn.id for insn in insns) | extra_used_ids

        from loopy.tools import generate_unique_possibilities
        for id_str in generate_unique_possibilities(based_on):
            if id_str not in used_ids:
                return id_str

    @memoize_method
    def all_inames(self):
        from islpy import dim_type
        return set(self.space.get_var_dict(dim_type.set).iterkeys())

    @memoize_method
    def non_iname_variable_names(self):
        return (set(self.arg_dict.iterkeys())
                | set(self.temporary_variables.iterkeys()))

    @memoize_method
    def all_insn_inames(self):
        from loopy.symbolic import get_dependencies

        insn_id_to_inames = {}
        insn_assignee_inames = {}

        for insn in self.instructions:
            read_deps = get_dependencies(insn.expression)
            write_deps = get_dependencies(insn.assignee)
            deps = read_deps | write_deps

            iname_deps = (
                    deps & self.all_inames()
                    | insn.forced_iname_deps)

            insn_id_to_inames[insn.id] = iname_deps
            insn_assignee_inames[insn.id] = write_deps & self.all_inames()

        writers = self.find_writers()
        temp_var_names = set(self.temporary_variables.iterkeys())

        # fixed point iteration until all iname dep sets have converged

        while True:
            did_something = False
            for insn in self.instructions:

                # For all variables that insn depends on, find the intersection
                # of iname deps of all writers, and add those to insn's
                # dependencies.

                for tv_name in (get_dependencies(insn.expression)
                        & temp_var_names):
                    implicit_inames = None

                    for writer_id in writers[tv_name]:
                        #writer_insn = self.id_to_insn[writer_id]

                        writer_implicit_inames = (
                                insn_id_to_inames[writer_id]
                                - insn_assignee_inames[writer_id])
                        if implicit_inames is None:
                            implicit_inames = writer_implicit_inames
                        else:
                            implicit_inames = (implicit_inames
                                    & writer_implicit_inames)

                    inames_old = insn_id_to_inames[insn.id]
                    inames_new = (inames_old | implicit_inames) \
                                - insn.reduction_inames()
                    insn_id_to_inames[insn.id] = inames_new

                    if inames_new != inames_old:
                        did_something = True

            if not did_something:
                break

        return insn_id_to_inames

    @memoize_method
    def all_referenced_inames(self):
        result = set()
        for inames in self.all_insn_inames().itervalues():
            result.update(inames)
        return result

    def insn_inames(self, insn):
        if isinstance(insn, str):
            return self.all_insn_inames()[insn]
        else:
            return self.all_insn_inames()[insn.id]

    @memoize_method
    def iname_to_insns(self):
        result = dict(
                (iname, set()) for iname in self.all_inames())
        for insn in self.instructions:
            for iname in self.insn_inames(insn):
                result[iname].add(insn.id)

        return result

    @property
    @memoize_method
    def sequential_inames(self):
        result = set()

        def map_reduction(red_expr, rec):
            rec(red_expr.expr)
            result.update(red_expr.inames)

        from loopy.symbolic import ReductionCallbackMapper
        for insn in self.instructions:
            ReductionCallbackMapper(map_reduction)(insn.expression)

        for iname in result:
            tag = self.iname_to_tag.get(iname)
            if tag is not None and isinstance(tag, ParallelTag):
                raise RuntimeError("inconsistency detected: "
                        "sequential/reduction iname '%s' has "
                        "a parallel tag" % iname)

        return result

    def find_readers(self):
        """
        :return: a dict that maps variable names to ids of insns that read that variable.
        """
        result = {}

        admissible_vars = (
                set(arg.name for arg in self.args)
                | set(self.temporary_variables.iterkeys()))

        for insn in self.instructions:
            for var_name in insn.get_read_var_names() & admissible_vars:
                result.setdefault(var_name, set()).add(insn.id)

    def find_writers(self):
        """
        :return: a dict that maps variable names to ids of insns that write to that variable.
        """
        result = {}

        admissible_vars = (
                set(arg.name for arg in self.args)
                | set(self.temporary_variables.iterkeys()))

        for insn in self.instructions:
            var_name = insn.get_assignee_var_name()

            if var_name not in admissible_vars:
                raise RuntimeError("variable '%s' not declared or not allowed for writing" % var_name)
            var_names = [var_name]

            for var_name in var_names:
                result.setdefault(var_name, set()).add(insn.id)

        return result

    @property
    @memoize_method
    def iname_to_dim(self):
        return self.domain.get_space().get_var_dict()

    @memoize_method
    def get_written_variables(self):
        return set(
            insn.get_assignee_var_name()
            for insn in self.instructions)

    @memoize_method
    def all_variable_names(self):
        return (
                set(self.temporary_variables.iterkeys())
                | set(self.substitutions.iterkeys())
                | set(arg.name for arg in self.args)
                | set(self.iname_to_dim.keys()))

    def make_unique_var_name(self, based_on="var", extra_used_vars=set()):
        used_vars = self.all_variable_names() | extra_used_vars

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
    def get_iname_bounds(self, iname):
        dom_intersect_assumptions = (
                isl.align_spaces(self.assumptions, self.domain)
                & self.domain)
        lower_bound_pw_aff = (
                self.cache_manager.dim_min(
                    dom_intersect_assumptions,
                    self.iname_to_dim[iname][1])
                .coalesce())
        upper_bound_pw_aff = (
                self.cache_manager.dim_max(
                    dom_intersect_assumptions,
                    self.iname_to_dim[iname][1])
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
            all_inames_by_insns |= self.insn_inames(insn)

        if not all_inames_by_insns <= self.all_inames():
            raise RuntimeError("inames collected from instructions (%s) "
                    "that are not present in domain (%s)"
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

        def to_dim_tuple(size_dict, which, forced_sizes={}):
            forced_sizes = forced_sizes.copy()

            size_list = []
            sorted_axes = sorted(size_dict.iterkeys())

            zero_aff = isl.Aff.zero_on_domain(self.space.params())

            while sorted_axes or forced_sizes:
                if sorted_axes:
                    cur_axis = sorted_axes.pop(0)
                else:
                    cur_axis = None

                if len(size_list) in forced_sizes:
                    size_list.append(
                            isl.PwAff.from_aff(
                                zero_aff + forced_sizes.pop(len(size_list))))
                    continue

                assert cur_axis is not None

                while cur_axis > len(size_list):
                    raise RuntimeError("%s axis %d unused" % (
                        which, len(size_list)))
                    size_list.append(zero_aff + 1)

                size_list.append(size_dict[cur_axis])

            if len(size_list) > max_dims:
                raise ValueError("more %s dimensions assigned than supported "
                        "by hardware (%d > %d)" % (which, len(size_list), max_dims))

            return tuple(size_list)

        return (to_dim_tuple(global_sizes, "global"),
                to_dim_tuple(local_sizes, "local", forced_sizes=self.local_sizes))

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

    @memoize_method
    def loop_nest_map(self):
        """Returns a dictionary mapping inames to other inames that are
        always nested around them.
        """
        result = {}
        iname_to_insns = self.iname_to_insns()

        for inner_iname in self.all_inames():
            result[inner_iname] = set()
            for outer_iname in self.all_inames():
                if iname_to_insns[inner_iname] < iname_to_insns[outer_iname]:
                    result[inner_iname].add(outer_iname)

        return result

    def map_expressions(self, func, exclude_instructions=False):
        if exclude_instructions:
            new_insns = self.instructions
        else:
            new_insns = [insn.copy(expression=func(insn.expression))
                    for insn in self.instructions]

        return self.copy(
                instructions=new_insns,
                substitutions=dict(
                    (subst.name, subst.copy(expression=func(subst.expression)))
                    for subst in self.substitutions.itervalues()))

    def __str__(self):
        lines = []

        sep = 75*"-"
        lines.append(sep)
        lines.append("INAME-TO-TAG MAP:")
        for iname in sorted(self.all_inames()):
            lines.append("%s: %s" % (iname, self.iname_to_tag.get(iname)))

        lines.append(sep)
        lines.append("DOMAIN:")
        lines.append(str(self.domain))

        if self.substitutions:
            lines.append(sep)
            lines.append("SUBSTIUTION RULES:")
            for rule in self.substitutions.itervalues():
                lines.append(str(rule))

        lines.append(sep)
        lines.append("INSTRUCTIONS:")
        loop_list_width = 35
        for insn in self.instructions:
            loop_list = ",".join(sorted(self.insn_inames(insn)))
            if len(loop_list) > loop_list_width:
                lines.append("[%s]" % loop_list)
                lines.append("%s%s <- %s   # %s" % (
                    (loop_list_width+2)*" ", insn.assignee, insn.expression, insn.id))
            else:
                lines.append("[%s]%s%s <- %s   # %s" % (
                    loop_list, " "*(loop_list_width-len(loop_list)),
                    insn.assignee, insn.expression, insn.id))

        lines.append(sep)
        lines.append("DEPENDENCIES:")
        for insn in self.instructions:
            if insn.insn_deps:
                lines.append("%s : %s" % (insn.id, ",".join(insn.insn_deps)))
        lines.append(sep)

        return "\n".join(lines)

# }}}




def find_var_base_indices_and_shape_from_inames(
        domain, inames, cache_manager, context=None):
    base_indices = []
    shape = []

    iname_to_dim = domain.get_space().get_var_dict()
    for iname in inames:
        lower_bound_pw_aff = cache_manager.dim_min(domain, iname_to_dim[iname][1])
        upper_bound_pw_aff = cache_manager.dim_max(domain, iname_to_dim[iname][1])

        from loopy.isl_helpers import static_max_of_pw_aff, static_value_of_pw_aff
        from loopy.symbolic import pw_aff_to_expr

        shape.append(pw_aff_to_expr(static_max_of_pw_aff(
                upper_bound_pw_aff - lower_bound_pw_aff + 1, constants_only=True,
                context=context)))
        base_indices.append(pw_aff_to_expr(
            static_value_of_pw_aff(lower_bound_pw_aff, constants_only=False,
                context=context)))

    return base_indices, shape




def get_dot_dependency_graph(kernel, iname_cluster=False, iname_edge=True):
    lines = []
    for insn in kernel.instructions:
        lines.append("%s [shape=\"box\"];" % insn.id)
        for dep in insn.insn_deps:
            lines.append("%s -> %s;" % (dep, insn.id))

        if iname_edge:
            for iname in kernel.insn_inames(insn):
                lines.append("%s -> %s [style=\"dotted\"];" % (iname, insn.id))

    if iname_cluster:
        for iname in kernel.all_inames():
            lines.append("subgraph cluster_%s { label=\"%s\" %s }" % (iname, iname,
                " ".join(insn.id for insn in kernel.instructions
                    if iname in kernel.insn_inames(insn))))

    return "digraph loopy_deps {\n%s\n}" % "\n".join(lines)




class SetOperationCacheManager:
    def __init__(self):
        # mapping: set hash -> [(set, op, args, result)]
        self.cache = {}

    def op(self, set, op, args):
        hashval = hash(set)
        bucket = self.cache.setdefault(hashval, [])

        for bkt_set, bkt_op, bkt_args, result  in bucket:
            if set.plain_is_equal(bkt_set) and op == bkt_op and args == bkt_args:
                return result

        #print op, set.get_dim_name(dim_type.set, args[0])
        result = getattr(set, op)(*args)
        bucket.append((set, op, args, result))
        return result

    def dim_min(self, set, *args):
        return self.op(set, "dim_min", args)

    def dim_max(self, set, *args):
        return self.op(set, "dim_max", args)




# vim: foldmethod=marker

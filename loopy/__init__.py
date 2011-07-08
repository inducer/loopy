from __future__ import division

import numpy as np
from pytools import Record, memoize_method
from pymbolic.mapper.dependency import DependencyMapper
from pymbolic.mapper.c_code import CCodeMapper
from pymbolic.mapper.stringifier import PREC_NONE
from pymbolic.mapper import CombineMapper, RecursiveMapper

import pyopencl as cl




# TODO: Multi-D array access
# TODO: Non-multiple loop splits
#       FIXME: Splitting an uneven-split loop?
# TODO: nD Texture access
# TODO: Functions
# TODO: Common subexpressions
# TODO: Try different kernels
# TODO:   - Tricky: Convolution, FD
# TODO: Try, fix indirect addressing

# TODO: Play with multi-d data layout (optionally?)
# TODO: Custom reductions per red. axis
# TODO: Switch to sympy, serve multiple accesses with one prefetch
# TODO: Vectorize
# TODO: Unroll
# TODO: Parallelize reduction




NEG_RELATION = {
        "==" : "!=",
        "!=" : "==",
        "<" : ">+",
        "<=" : ">",
        ">" : "<=",
        ">=" : "<",
        }



def generate_condition_code(ccm, condition, negate=False, expr_map=None):
    a, rel, b = condition

    if negate:
        rel = NEG_RELATION[rel]

    if expr_map is not None:
        a = expr_map(a)
        b = expr_map(b)

    return "%s %s %s" % (ccm(a, PREC_NONE), rel, ccm(b, PREC_NONE),)




# {{{ index tags

class IndexTag(object):
    def __init__(self, axis=None):
        self.axis = axis

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.axis == other.axis)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.axis)


class GROUP_IDX_TAG(IndexTag):
    def __repr__(self):
        if self.axis is None:
            return "GROUP_IDX"
        else:
            return "GROUP_IDX(%d)" % self.axis


class WORK_ITEM_IDX_TAG(IndexTag):
    def __repr__(self):
        if self.axis is None:
            return "WORK_ITEM_IDX"
        else:
            return "WORK_ITEM_IDX(%d)" % self.axis

def parse_tag(tag):
    if tag is None:
        return tag

    if isinstance(tag, IndexTag):
        return tag

    if not isinstance(tag, str):
        raise ValueError("cannot parse tag: %s" % tag)

    if tag.startswith("g."):
        return GROUP_IDX_TAG(int(tag[2:]))
    if tag.startswith("l."):
        return WORK_ITEM_IDX_TAG(int(tag[2:]))
    else:
        raise ValueError("cannot parse tag: %s" % tag)

# }}}

# {{{ loop dim, loop domain, kernel

class LoopDimension(Record):
    __slots__ = ["name", "length", "last_cond",  "tag", 
            "end_cond", "end_cond_if_last_of"]

    def __init__(self, name, length=None, last_cond=None, end_cond=None, tag=None, 
            end_cond_if_last_of=set()):
        """
        One of two end conditions governs a loop:

        :arg length:
        :arg last_cond: If not None, generate separate code for the 'last iteration'
            of this loop, as indicated by last cond.

        :arg end_cond: A condition indicating whether the loop has ended.
            This is not used for loop termination, but to check in nested
            blocks whether actions relating to this loop should be performed.

        Any 'condition' above is a (value, comparison_op, other_value) triple.

        All arguments except name are keyword-only.
        """

        # FIXME: Not sure what combinations of end conditions make sense

        Record.__init__(self, name=name, length=length, last_cond=last_cond,
                end_cond=end_cond, tag=tag, end_cond_if_last_of=end_cond_if_last_of)

        if tag is not None:
            assert isinstance(tag, IndexTag)

    def __hash__(self):
        return hash(self.name)





class LoopDomain(Record):
    __slots__ = ["dims"]

    def name_to_idx(self, name):
        for i, dim in enumerate(self.dims):
            if dim.name == name:
                return i
        else:
            raise KeyError("invalid dimension name: %s" % name)

    def name_to_dim(self, name):
        for dim in self.dims:
            if dim.name == name:
                return dim
        else:
            raise KeyError("invalid dimension name: %s" % name)

    def tag_to_idx(self, tag):
        for i, dim in enumerate(self.dims):
            if dim.tag == tag:
                return i
        raise KeyError("invalid tag: %s" % tag)

    def tag_to_dim(self, tag):
        return self.dims[self.tag_to_idx(tag)]

    def indices_by_tag_type(self, tag_type):
        return [i for i, dim in enumerate(self.dims)
                if isinstance(dim.tag, tag_type)]

    def dims_by_tag_type(self, tag_type):
        return [dim for dim in self.dims
                if isinstance(dim.tag, tag_type)]

    def ordered_dims_by_tag_type(self, tag_type):
        result = []
        from itertools import count
        for i in count():
            try:
                dim = self.tag_to_dim(tag_type(i))
            except KeyError:
                return result
            else:
                result.append(dim)

    def dims_by_tag(self, tag):
        return [dim for dim in self.dims if dim.tag == tag]

    def set_dim(self, idx, new_dim):
        return self.copy(dims=
                self.dims[:idx]
                + [new_dim]
                + self.dims[(idx+1):])




# {{{ arguments

class ArrayArg:
    def __init__(self, name, dtype, strides=None, shape=None, order="C"):
        """
        All of the following are optional. Specify either strides or shape.

        :arg strides: like numpy strides, but in multiples of
            data type size
        :arg shape:
        :arg order:
        """
        self.name = name
        self.dtype = np.dtype(dtype)

        if strides is not None and shape is not None:
            raise ValueError("can only specify one of shape and strides")

        if strides is not None:
            strides = tuple(strides)

        if shape is not None:
            from pyopencl.compyte.array import (
                    f_contiguous_strides,
                    c_contiguous_strides)

            if order == "F":
                strides = _f_contiguous_strides(
                        dtype.itemsize, shape)
            elif order == "C":
                strides = _c_contiguous_strides(
                        dtype.itemsize, shape)
            else:
                raise ValueError("invalid order: %s" % order)

        self.strides = strides

    def __repr__(self):
        return "<ArrayArg '%s' of type %s>" % (self.name, self.dtype)



class ScalarArg:
    def __init__(self, name, dtype, approximately):
        self.name = name
        self.dtype = np.dtype(dtype)
        self.approximately = approximately

    def __repr__(self):
        return "<ScalarArg '%s' of type %s>" % (self.name, self.dtype)

# }}}




class LoopKernel(LoopDomain):
    # possible attributes:
    # - device, a PyOpenCL target device
    # - dims from LoopDomain
    # - instructions
    # - args
    # - prefetch
    # - schedule
    # - register_prefetch
    # - name
    # - preamble

    def __init__(self, device, dims, instructions, args=None, prefetch={}, schedule=None,
            register_prefetch=None, default_vector_type=None, name="loopy_kernel",
            preamble=None):
        from pymbolic import parse

        def parse_if_necessary(v):
            if isinstance(v, str):
                return parse(v)
            else:
                return v

        insns = [
                (parse_if_necessary(lvalue),
                    parse_if_necessary(expr))
                for lvalue, expr in instructions]

        LoopDomain.__init__(self,
                device=device, args=args, dims=dims, instructions=insns,
                prefetch=prefetch, schedule=schedule,
                register_prefetch=register_prefetch, name=name,
                preamble=preamble)

        if args is None:
            self.args = [
                    ArrayArg(name, default_vector_type)
                    for name in
                    sorted(self.input_vectors()) + sorted(self.output_vectors())]

    @property
    @memoize_method
    def arg_dict(self):
        return dict((arg.name, arg) for arg in self.args)

    @memoize_method
    def scalar_args(self):
        if self.args is None:
            return set()
        else:
            return set(arg.name for arg in self.args if isinstance(arg, ScalarArg))

    @memoize_method
    def all_indices(self):
        return set(dim.name for dim in self.dims) - self.scalar_args()

    @memoize_method
    def output_indices(self):
        dm = DependencyMapper(include_subscripts=False)

        output_indices = set()
        for lvalue, expr in self.instructions:
            output_indices.update(
                    set(v.name for v in dm(lvalue))
                    & self.all_indices())

        return output_indices - set(arg.name for arg in self.args)

    @memoize_method
    def output_dimensions(self):
        return [dim for dim in self.dims if dim.name in self.output_indices()]

    @memoize_method
    def reduction_dimensions(self):
        return [dim for dim in self.dims if dim.name not in self.output_indices()]

    def group_dims(self):
        dims = self.ordered_dims_by_tag_type(GROUP_IDX_TAG)
        return tuple(dim.length for dim in dims)

    def local_dims(self):
        dims = self.ordered_dims_by_tag_type(WORK_ITEM_IDX_TAG)
        return tuple(dim.length for dim in dims)

    def group_size(self):
        from pytools import product
        return product(self.local_dims())

    def group_count(self):
        from pytools import product
        return product(self.group_dims())

    def parse_sloppy_dim_to_dim_idx(self, dim):
        if isinstance(dim, str):
            try:
                tag = parse_tag(dim)
            except ValueError:
                pass
            else:
                return self.tag_to_idx(tag)

            return self.name_to_idx(dim)

        if isinstance(dim, LoopDimension):
            return self.dims.index(dim)

        if isinstance(dim, int):
            return dim

    def parse_sloppy_dim(self, dim):
        return self.dims[self.parse_sloppy_dim_to_dim_idx(dim)]

    def local_mem_use(self):
        return sum(pf.size() for pf in self.prefetch.itervalues())

    @memoize_method
    def input_vectors(self):
        dm = DependencyMapper(include_subscripts=False)

        input_vectors = set()
        for lvalue, expr in self.instructions:
            input_vectors.update(
                    set(v.name for v in dm(expr)))

        return input_vectors - self.all_indices() - self.scalar_args()

    @memoize_method
    def output_vectors(self):
        dm = DependencyMapper(include_subscripts=False)

        output_vectors = set()
        for lvalue, expr in self.instructions:
            output_vectors.update(
                    set(v.name for v in dm(lvalue)))

        return output_vectors - self.all_indices() - self.scalar_args()

    def _subst_insns(self, old_var, new_expr):
        from pymbolic.mapper.substitutor import substitute

        subst_map = {old_var: new_expr}

        return [(substitute(lvalue, subst_map),
            substitute(expr, subst_map))
            for lvalue, expr in self.instructions]

    def is_prefetch_variable(self, varname):
        if self.prefetch:
            for pf in self.prefetch.itervalues():
                for pfdim in pf.dims:
                    if pfdim.name == varname:
                        return True

        return False

    def _subst_prefetch(self, old_var, new_expr):
        from pymbolic.mapper.substitutor import substitute
        subst_map = {old_var: new_expr}

        result = {}
        for pf in self.prefetch.itervalues():
            for pfdim in pf.dims:
                if pfdim.name == old_var:
                    raise RuntimeError("can't substitute prefetch dimension %s"
                            % old_var)

            new_pf = pf.copy(index_expr=substitute(pf.index_expr, subst_map))
            result[pf.input_vector, new_pf.index_expr] = new_pf

        return result

    def substitute(self, old_var, new_expr):
        copy = self.copy(instructions=self._subst_insns(old_var, new_expr))
        if self.prefetch:
            copy.prefetch = self._subst_prefetch(old_var, new_expr)

        if self.schedule is not None:
            for sched_item in self.schedule:
                if (isinstance(sched_item, LoopDimension)
                        and sched_item.name == old_var):
                    raise RuntimeError("can't substitute already-scheduled variable: %s"
                            % old_var)

        return copy

    def split_dimension(self, idx, inner_length, outer_name=None, inner_name=None,
            outer_tag=None, inner_tag=None, is_even_split=None):
        if isinstance(idx, str):
            idx = self.name_to_idx(idx)

        outer_tag = parse_tag(outer_tag)
        inner_tag = parse_tag(inner_tag)

        new_tags = set(tag for tag in [outer_tag, inner_tag] if tag is not None)

        for d in self.dims:
            if d.tag in new_tags:
                raise RuntimeError("repeated tag: %s" % d.tag)

        dim = self.dims[idx]
        if dim.end_cond is not None or dim.last_cond is not None:
            raise NotImplementedError("don't yet know how to split "
                    "last_cond or end_cond loops")

        if dim.tag:
            raise ValueError("cannot split already-tagged dimension")

        if new_tags and dim.name not in self.output_indices():
            raise NotImplementedError("cannot yet tag a non-output dimension")

        if is_even_split != False and dim.length % inner_length == 0:
            is_even_split = True

        if outer_name is None:
            outer_name = dim.name+"_outer"
        if inner_name is None:
            inner_name = dim.name+"_inner"
        from pymbolic import var
        outer = var(outer_name)
        inner = var(inner_name)

        new_loop_index = inner + outer*inner_length

        if is_even_split:
            new_dims = [
                    LoopDimension(
                        name=outer_name,
                        length=dim.length//inner_length,
                        tag=outer_tag),
                    LoopDimension(
                        name=inner_name,
                        length=inner_length,
                        tag=inner_tag),
                    ]
        else:
            from pytools import div_ceil
            new_dims = [
                    LoopDimension(
                        name=outer_name,
                        length=div_ceil(dim.length, inner_length),
                        last_cond=((outer+1)*inner_length, ">=", dim.length),
                        tag=outer_tag),
                    LoopDimension(
                        name=inner_name,
                        length=inner_length,
                        end_cond=(new_loop_index, ">=", dim.length),
                        tag=inner_tag,
                        end_cond_if_last_of=dim.end_cond_if_last_of | set([outer_name])),
                    ]

        return (self
                .substitute(dim.name, new_loop_index)
                .copy(dims=self.dims[:idx] + new_dims + self.dims[(idx+1):]), 
                new_loop_index)

    def get_invalid_reason(self):
        gdims = self.group_dims()
        ldims = self.local_dims()
        if (max(len(gdims), len(ldims))
                > self.device.max_work_item_dimensions):
            return "too many work item dimensions"

        for i in range(len(ldims)):
            if ldims[i] > self.device.max_work_item_sizes[i]:
                return "group axis %d too big"

        if self.group_size() > self.device.max_work_group_size:
            return "work group too big"

        from pyopencl.characterize import usable_local_mem_size
        if self.local_mem_use() > usable_local_mem_size(self.device):
            return "using too much local memory"

        return None

# }}}

# {{{ local-mem prefetch-related

class PrefetchDescriptor(Record):
    """
    Attributes:
    :ivar kernel:
    :ivar input_vector: A string indicating the input vector variable name.
    :ivar index_expr: An expression identifying the access which this prefetch
      serves.
    :ivar dims: A sequence of loop dimensions identifying which part of the
      input vector, given the index_expr, should be prefetched.
    :ivar loc_fetch_axes: dictionary from integers 0..len(dims) to lists of
      local index axes which should be used to realize that dimension of the
      prefetch. The last dimension in this list is used as the fastest-changing
      one.
    :ivar name: the variable name used for the prefetch
    :ivar dim_storage_lengths: a sequence of integers indicating the size of
        the storage for each dimension. It may may differ from the size of the
        actual loop dimensions to mitigate bank conflicts.

    The latter two values are only assigned during code generation.
    """

    def size(self):
        from pytools import product
        return (self.kernel.arg_dict[self.input_vector].dtype.itemsize
                * product(dim.length for dim in self.dims))

    @memoize_method
    def free_variables(self):
        return set(var.name
                for var in DependencyMapper()(self.index_expr)
                ) - set(dim.name for dim in self.dims) - self.kernel.scalar_args()

    def hash(self):
        return (hash(type(self)) ^ hash(self.input_vector)
                ^ hash(self.index_expr))

    def __eq__(self, other):
        # in particular, dim_storage_lengths should not factor into equality
        return (type(self) == type(other)
                and self.input_vector == other.input_vector
                and self.index_expr == other.index_expr)






class VariableIndexExpressionCollector(CombineMapper):
    def __init__(self, tgt_vector_name):
        self.tgt_vector_name = tgt_vector_name

    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, expr):
        return set()

    def map_algebraic_leaf(self, expr):
        return set()

    def map_subscript(self, expr):
        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        if expr.aggregate.name == self.tgt_vector_name:
            return set([expr.index])
        else:
            return CombineMapper.map_subscript(self, expr)




class StrideCollector(RecursiveMapper):
    def map_sum(self, expr):
        stride_dicts = [self.rec(ch) for ch in expr.children]

        result = {}
        for stride_dict in stride_dicts:
            for var, stride in stride_dict.iteritems():
                if var in result:
                    result[var] += stride
                else:
                    result[var] = stride

        return result

    def map_product(self, expr):
        result = {}
        for i, ch in enumerate(expr.children):
            strides = self.rec(ch)
            from pymbolic import flattened_product
            prod_other_children = flattened_product(
                    expr.children[:i] + expr.children[(i+1):])

            for var, stride in strides.iteritems():
                if var in result:
                    raise NotImplementedError(
                            "nonlinear index expression")
                else:
                    result[var] = prod_other_children*stride

        return result

    def map_divide(self, expr):
        num_strides = self.rec(expr.numerator)
        denom_strides = self.rec(expr.denominator)

        if denom_strides:
            raise NotImplementedError

        return dict(
                (var, stride/expr.denominator)
                    for var, stride in num_strides.iteritems())

    def map_constant(self, expr):
        return {}

    def map_variable(self, expr):
        return {expr.name: 1}

    def map_subscript(self, expr):
        raise RuntimeError("cannot gather strides--indirect addressing in use")

# }}}

# {{{ loop scheduling

def generate_loop_schedules(kernel):
    prev_schedule = kernel.schedule
    if prev_schedule is None:
        prev_schedule = (
            kernel.dims_by_tag_type(GROUP_IDX_TAG)
            + kernel.dims_by_tag_type(WORK_ITEM_IDX_TAG))

    already_scheduled = set(sch_item
            for sch_item in prev_schedule
            if isinstance(sch_item, LoopDimension))

    # have a schedulable prefetch? load, schedule it
    scheduled_names = set(dim.name for dim in already_scheduled)

    had_usable_prefetch = False
    scheduled_work_item_dim_names = set(
            dim.name for dim in already_scheduled
            if isinstance(dim.tag, WORK_ITEM_IDX_TAG))

    for pf in kernel.prefetch.itervalues():
        # already scheduled? never mind then.
        if pf in prev_schedule:
            continue

        # a free variable not known yet? then we're not ready
        if not pf.free_variables() <= scheduled_names:
            continue

        # a prefetch variable already scheduled, but not borrowable?
        # (only work item index variables are borrowable)
        pf_loop_names = set(dim.name for dim in pf.dims)

        if pf_loop_names & (already_scheduled - scheduled_work_item_dim_names):
            # dead end: we won't be able to schedule this prefetch
            # in this branch. at least one of its loop dimensions
            # was already scheduled, and that dimension is not
            # borrowable.
            print "UNSCHEDULABLE:"
            print_kernel_info(kernel)
            raw_input()
            return

        new_kernel = kernel.copy(schedule=prev_schedule+[pf])
        for knl in generate_loop_schedules(new_kernel):
            had_usable_prefetch = True
            yield knl

    if had_usable_prefetch:
        return

    # Build set of potentially schedulable variables
    schedulable = set(kernel.dims)

    # Don't re-schedule already scheduled variables
    schedulable -= already_scheduled

    # Don't schedule reduction variables until all output
    # variables are taken care of. Once they are, schedule
    # output writing.
    serial_output_dims = set(od for od in kernel.output_dimensions()
            if od.tag is None)

    if not serial_output_dims <= already_scheduled:
        schedulable -= set(kernel.reduction_dimensions())
    else:
        if not any(isinstance(sch_item, WriteOutput)
                for sch_item in prev_schedule):
            kernel = kernel.copy(
                    schedule=prev_schedule + [WriteOutput()])
            prev_schedule = kernel.schedule

    # Don't schedule variables that are prefetch axes
    # for not-yet-scheduled prefetches.
    unsched_prefetch_axes = set(dim
            for pf in kernel.prefetch.itervalues()
            if pf not in prev_schedule
            for dim in pf.dims)
    schedulable -= unsched_prefetch_axes

    if schedulable:
        # have a schedulable variable? schedule a loop for it, recurse
        for dim in schedulable:
            new_kernel = kernel.copy(schedule=prev_schedule+[dim])
            for knl in generate_loop_schedules(new_kernel):
                yield knl
    else:
        # all loop dimensions and prefetches scheduled?
        # great! yield the finished product if it is complete

        all_dims_scheduled = len(already_scheduled) == len(kernel.dims)
        all_pf_scheduled =  len(set(sch_item for sch_item in prev_schedule
            if isinstance(sch_item, PrefetchDescriptor))) == len(kernel.prefetch)
        output_scheduled = len(set(sch_item for sch_item in prev_schedule
            if isinstance(sch_item, WriteOutput))) == 1

        if all_dims_scheduled and all_pf_scheduled and output_scheduled:
            yield kernel

# }}}

# {{{ register prefetches

class AllIndexExpressionCollector(CombineMapper):
    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, expr):
        return set()

    def map_algebraic_leaf(self, expr):
        return set()

    def map_subscript(self, expr):
        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        return set([expr])




def insert_register_prefetches(kernel):
    reg_pf = {}

    total_loop_count = len(kernel.all_indices())
    known_vars = set()

    unused_index_exprs = set()
    for tgt, expr in kernel.instructions:
        unused_index_exprs |= AllIndexExpressionCollector()(expr)
    unused_index_exprs = [
            (iexpr, set(v.name for v in DependencyMapper()(iexpr.index)))
            for iexpr in unused_index_exprs]

    schedule = kernel.schedule[:]

    sched_index = 0
    loop_count = 0
    while sched_index < len(schedule):
        sched_item = schedule[sched_index]
        if isinstance(sched_item, LoopDimension):
            known_vars.add(sched_item.name)
            loop_count += 1
        sched_index += 1

        if loop_count < total_loop_count:
            i = 0
            while i < len(unused_index_exprs):
                iexpr, index_deps = unused_index_exprs[i]
                if (index_deps <= known_vars
                        and (iexpr.aggregate.name, iexpr.index)
                            not in kernel.prefetch):
                    unused_index_exprs.pop(i)
                    new_name = "reg_prefetch_"+iexpr.aggregate.name+str(sched_index)
                    reg_pf[iexpr] = new_name
                    schedule.insert(sched_index,
                            RegisterPrefetch(
                                index_expr=iexpr, new_name=new_name))
                    sched_index += 1
                else:
                    i += 1

    return kernel.copy(schedule=schedule, register_prefetch=reg_pf)

# }}}

# {{{ code generation

class LoopyCCodeMapper(CCodeMapper):
    def __init__(self, kernel):
        def constant_mapper(c):
            if isinstance(c, float):
                # FIXME: type-variable
                return "%sf" % repr(c)
            else:
                return repr(c)

        CCodeMapper.__init__(self, constant_mapper=constant_mapper)
        self.kernel = kernel

    def map_subscript(self, expr, enclosing_prec):
        from pymbolic.primitives import Variable
        if (isinstance(expr.aggregate, Variable)
                and expr.aggregate.name in self.kernel.input_vectors()):
            try:
                pf = self.kernel.prefetch[expr.aggregate.name, expr.index]
            except KeyError:
                pass
            else:
                return pf.name+"".join(
                        "[%s]" % dim.name for dim in pf.dims)

        return CCodeMapper.map_subscript(self, expr, enclosing_prec)





class WriteOutput(Record):
    pass

class RegisterPrefetch(Record):
    __slots__ = ["subscript_expr", "new_name"]




def generate_prefetch_code(ccm, kernel, sched_index, last_of):
    from pymbolic import var

    from cgen import (Block,
            Assign, Statement as S,
            For, If, Line, Comment)

    # find surrounding schedule items
    if sched_index-1 >= 0:
        next_outer_sched_item = kernel.schedule[sched_index-1]
    else:
        next_outer_sched_item = None

    if sched_index+1 < len(kernel.schedule):
        next_inner_sched_item = kernel.schedule[sched_index+1]
    else:
        next_inner_sched_item = None

    scheduled_pf = kernel.schedule[sched_index]
    pf = kernel.prefetch[
            scheduled_pf.input_vector, scheduled_pf.index_expr]

    # figure out dimension types
    from pytools import partition2
    work_item_pf_dims, non_work_item_pf_dims = partition2(
            (isinstance(dim.tag, WORK_ITEM_IDX_TAG), dim)
            for dim in pf.dims)

    # Prefetch has a good amount of flexibility over what axes it
    # uses to accomplish the prefetch. In particular, it can (and should!)
    # use all work group dimensions.

    # {{{ determine which dims are used to realize the fetch

    # realization_dims is a list of lists of dims, to represent when two dims jointly
    # make up one fetch axis

    realization_dims = [None] * len(pf.dims)

    # {{{ first, fix the user-specified fetch dims

    knl_work_item_dims = kernel.ordered_dims_by_tag_type(WORK_ITEM_IDX_TAG)

    for realization_dim_idx, loc_fetch_axis_list in \
            getattr(pf, "loc_fetch_axes", {}).iteritems():
        realization_dims[realization_dim_idx] = [knl_work_item_dims.pop(axis)
            for axis in loc_fetch_axis_list]

    # }}}

    # {{{ next use the work group dimensions, least-stride dim first

    strides = StrideCollector()(pf.index_expr)

    approximate_arg_values = dict(
            (arg.name, arg.approximately)
            for arg in kernel.args
            if isinstance(arg, ScalarArg))

    def stride_key(a):
        idx, a_stride = a

        from pymbolic import evaluate
        key = evaluate(a_stride, approximate_arg_values)
        assert isinstance(key, int)
        return key

    pf_dim_strides = sorted(((dim_idx, strides[dim.name])
        for dim_idx, dim in enumerate(pf.dims)
        if realization_dims[dim_idx] is None),
        key=stride_key)

    while knl_work_item_dims and pf_dim_strides:
        # grab least-stride prefetch dim
        least_stride_pf_dim_idx, _ = pf_dim_strides.pop(0)

        # FIXME: It might be good to join multiple things together here
        # for size reasons
        realization_dims[least_stride_pf_dim_idx] = [knl_work_item_dims.pop(0)]

    if knl_work_item_dims:
        # FIXME
        from warnings import warn
        warn("There were leftover work group dimensions in prefetch "
                "assignment. For now, this won't lead to wrong code, "
                "but it will lead to unnecessary memory bandwidth use.")

    # }}}

    # }}}

    # {{{ generate fetch code

    def make_fetch_loop_nest(pf_dim_idx, pf_dim_exprs=[], pf_idx_subst_map={}):
        # may mutate kernel for prefetch dim enlargement

        from pymbolic.mapper.substitutor import substitute
        if pf_dim_idx >= len(pf.dims):
            # done, return
            return Assign(
                    pf.name + "".join("[%s]" % ccm(dexpr, PREC_NONE)
                        for dexpr in pf_dim_exprs),
                    "%s[%s]"
                    % (pf.input_vector,
                        substitute(pf.index_expr, pf_idx_subst_map))
                    )

        pf_dim = pf.dims[pf_dim_idx]
        realiz_dim_list = realization_dims[pf_dim_idx]

        if realiz_dim_list is not None:
            # {{{ parallel fetch

            from pytools import product
            total_realiz_size = product(rd.length for rd in realiz_dim_list)

            start_index = 0
            result = None

            while start_index < pf_dim.length:
                pf_dim_expr = 0
                for realiz_dim in realiz_dim_list:
                    assert isinstance(realiz_dim.tag, WORK_ITEM_IDX_TAG)

                    pf_dim_expr = (pf_dim_expr*realiz_dim.length
                            + var("get_local_id(%d)" % realiz_dim.tag.axis))

                pf_dim_expr += start_index

                pf_idx_subst_map = pf_idx_subst_map.copy()
                pf_idx_subst_map[pf_dim.name] = pf_dim_expr
                inner = make_fetch_loop_nest(pf_dim_idx+1,
                        pf_dim_exprs+[pf_dim_expr], pf_idx_subst_map)

                if start_index+total_realiz_size > pf_dim.length:
                    inner = If(
                            "%s < %s" % (ccm(pf_dim_expr, PREC_NONE), pf_dim.length),
                            inner)

                if (pf_dim.end_cond is not None
                        and pf_dim.end_cond_if_last_of <= last_of):
                    inner = If(
                            generate_condition_code(ccm,
                                pf_dim.end_cond, negate=True,
                                expr_map=lambda expr: substitute(expr, pf_idx_subst_map)),
                            inner)

                if result is None:
                    result = inner
                elif isinstance(result, Block):
                    result.append(inner)
                else:
                    result = Block([result, inner])

                start_index += total_realiz_size


            return result

            # }}}
        else:
            # {{{ sequential fetch

            pf_dim_var = "prefetch_dim_idx_%d" % pf_dim_idx
            pf_dim_expr = var(pf_dim_var)

            pf_idx_subst_map = pf_idx_subst_map.copy()
            pf_idx_subst_map[pf_dim.name] = pf_dim_expr
            inner = make_fetch_loop_nest(pf_dim_idx+1,
                    pf_dim_exprs+[pf_dim_expr], pf_idx_subst_map)

            return For(
                    "int %s = 0" % pf_dim_var,
                    "%s < %s" % (pf_dim_var, ccm(dim.length, PREC_NONE)),
                    "++%s" % pf_dim_var,
                    fetch_block)

            # }}}


    fetch_block = make_fetch_loop_nest(0)

    # }}}

    new_block = Block([
            Comment(("prefetch %s dim: " % pf.input_vector) + ", ".join(
                "%s[%d]" % (pfdim.name, pfdim.length) for pfdim in pf.dims)),
            Line(),
            ])

    # omit head sync primitive if we just came out of a prefetch
    if not isinstance(next_outer_sched_item, PrefetchDescriptor):
        new_block.append(S("barrier(CLK_LOCAL_MEM_FENCE)"))
    else:
        new_block.append(Comment("next outer schedule item is a prefetch: "
            "no sync needed"))

    new_block.extend([
        fetch_block,
        ])

    # omit tail sync primitive if we're headed into another prefetch
    if not isinstance(next_inner_sched_item, PrefetchDescriptor):
        new_block.append(S("barrier(CLK_LOCAL_MEM_FENCE)"))
    else:
        new_block.append(Comment("next inner schedule item is a prefetch: "
            "no sync needed"))

    new_block.extend([Line(), build_loop_nest(ccm, kernel, sched_index+1, last_of)])

    return new_block





def generate_loop_dim_code(ccm, kernel, sched_index, last_of):
    from cgen import (POD, Block, Initializer,
            For, If, Line, Comment, add_comment)

    dim = kernel.schedule[sched_index]

    if dim.tag is None:
        # regular loop
        if dim.last_cond is not None:
            return Block([
                    Initializer(POD(np.uint32, dim.name), 0),
                    For(
                        "",
                        generate_condition_code(ccm, dim.last_cond, negate=True),
                        "++%s" % dim.name, 
                        build_loop_nest(ccm, kernel, sched_index+1, last_of)),
                    Line(),
                    Comment("last iteration of %s loop, with added bounds checks" % dim.name),
                    build_loop_nest(ccm, kernel, sched_index+1, 
                        last_of=last_of | set([dim.name]))
                    ])

        elif dim.length is not None:
            if dim.end_cond is not None and dim.end_cond_if_last_of <= last_of:
                return For(
                        "int %s = 0" % dim.name,
                        generate_condition_code(ccm, dim.end_cond, negate=True),
                        "++%s" % dim.name, 
                        build_loop_nest(ccm, kernel, sched_index+1, last_of))
            else:
                return For(
                        "int %s = 0" % dim.name,
                        "%s < %s" % (dim.name, ccm(dim.length, PREC_NONE)),
                        "++%s" % dim.name, 
                        build_loop_nest(ccm, kernel, sched_index+1, last_of))
        else:
            raise RuntimeError("unsupported loop ending condition")
    else:
        if dim.last_cond is not None:
            return If(generate_condition_code(ccm, dim.last_cond, negate=True),
                add_comment(
                    "not the last entry along the '%s' work group axis" % dim.name,
                    build_loop_nest(ccm, kernel, sched_index+1, last_of)),
                add_comment(
                    "last entry along the '%s' work group axis" % dim.name,
                    build_loop_nest(ccm, kernel, sched_index+1, 
                        last_of=last_of | set([dim.name]))))
        else:
            return build_loop_nest(ccm, kernel, sched_index+1, last_of)




def get_parallel_dim_bounds_checks(ccm, kernel, last_of, stmt):
    from cgen import If

    for dim in (
            kernel.dims_by_tag_type(GROUP_IDX_TAG)
            + kernel.dims_by_tag_type(WORK_ITEM_IDX_TAG)):
        if (dim.end_cond is not None
                and dim.end_cond_if_last_of <= last_of):
            stmt = If(
                    generate_condition_code(ccm, dim.end_cond, negate=True),
                    stmt)

    return stmt




def build_loop_nest(ccm, kernel, sched_index, last_of=set()):
    from cgen import (POD, Block, Initializer, Assign, Statement as S,
            block_if_necessary)

    if sched_index >= len(kernel.schedule):
        # write innermost loop body

        from pymbolic.primitives import Subscript

        insns = []
        for lvalue, expr in kernel.instructions:
            assert isinstance(lvalue, Subscript)
            name = lvalue.aggregate.name
            insns.append(S("tmp_%s += %s"
                % (name, ccm(expr, PREC_NONE))))

        return get_parallel_dim_bounds_checks(ccm, kernel, last_of, 
                block_if_necessary(insns))

        # }}}

    sched_item = kernel.schedule[sched_index]

    if isinstance(sched_item, LoopDimension):
        return generate_loop_dim_code(ccm, kernel, sched_index, last_of)

    elif isinstance(sched_item, WriteOutput):
        return Block(
                [Initializer(POD(kernel.arg_dict[lvalue.aggregate.name].dtype,
                    "tmp_"+lvalue.aggregate.name), 0)
                    for lvalue, expr in kernel.instructions]
                +[build_loop_nest(ccm, kernel, sched_index+1, last_of)]+
                [get_parallel_dim_bounds_checks(ccm, kernel, last_of,
                    block_if_necessary([
                        Assign(
                            ccm(lvalue, PREC_NONE),
                            "tmp_"+lvalue.aggregate.name)
                        for lvalue, expr in kernel.instructions]))])

    elif isinstance(sched_item, PrefetchDescriptor):
        return generate_prefetch_code(ccm, kernel, sched_index, last_of)

    elif isinstance(sched_item, RegisterPrefetch):
        agg_name = sched_item.subscript_expr.aggregate.name
        return Block([
            get_parallel_dim_bounds_checks(ccm, kernel, last_of,
                Initializer(POD(kernel.arg_dict[agg_name].dtype,
                    sched_item.new_name),
                    "%s[%s]"
                    % (agg_name,
                        ccm(sched_item.subscript_expr.index, PREC_NONE)))),

            build_loop_nest(ccm, kernel, sched_index+1, last_of)])

    else:
        raise ValueError("invalid schedule item encountered")





def generate_code(kernel):
    from cgen import (FunctionBody, FunctionDeclaration, \
            POD, Value, RestrictPointer, ArrayOf, Module, Block,
            Define, Line, Const, LiteralLines)

    from cgen.opencl import CLKernel, CLGlobal, CLRequiredWorkGroupSize, CLLocal


    # {{{ assign names, dim storage lengths to prefetches

    all_pf_list = kernel.prefetch.values()
    all_pf_sizes = [opf.size() for opf in all_pf_list]

    new_prefetch = {}
    for i_pf, pf in enumerate(kernel.prefetch.itervalues()):
        dim_storage_lengths = [pfdim.length for pfdim in pf.dims]

        other_pf_sizes = sum(all_pf_sizes[:i_pf]+all_pf_sizes[i_pf+1:])

        from pytools import product
        other_dim_sizes = (
                kernel.arg_dict[pf.input_vector].dtype.itemsize
                * product(odim.length for odim in pf.dims[:-1]))

        from pyopencl.characterize import usable_local_mem_size
        if (pf.dims[-1].length % 2 == 0
                and other_pf_sizes+other_dim_sizes*(pf.dims[-1].length+1)
                < usable_local_mem_size(kernel.device)):
            dim_storage_lengths[-1] += 1

        new_prefetch[pf.input_vector, pf.index_expr] = \
                pf.copy(dims=pf.dims,
                        dim_storage_lengths=dim_storage_lengths,
                        name="prefetch_%s_%d" % (pf.input_vector, i_pf))

    kernel = kernel.copy(prefetch=new_prefetch)

    # }}}

    ccm = LoopyCCodeMapper(kernel)


    # {{{ build top-level

    mod = Module()

    group_size = kernel.group_size()

    # {{{ examine arg list

    has_double = False

    args = []
    for arg in kernel.args:
        if isinstance(arg, ArrayArg):
            arg_decl = RestrictPointer(POD(arg.dtype, arg.name))
            if arg_decl.name in kernel.input_vectors():
                arg_decl = Const(arg_decl)
            arg_decl = CLGlobal(arg_decl)
        else:
            arg_decl = POD(arg.dtype, arg.name)

        if arg.dtype in [np.float64, np.complex128]:
            has_double = True

        args.append(arg_decl)

    if has_double:
        mod.extend([
            Line("#pragma OPENCL EXTENSION cl_khr_fp64: enable"),
            Line()])

    # }}}

    if kernel.preamble is not None:
        mod.extend([LiteralLines(kernel.preamble), Line()])

    # {{{ symbolic names for group and local indices

    mod.extend([Define(dim.name, "get_group_id(%d) /* 0..(%s) */"
                % (dim.tag.axis, ccm(dim.length-1, PREC_NONE)))
                for dim in kernel.ordered_dims_by_tag_type(GROUP_IDX_TAG)]
            + [Define(dim.name, "get_local_id(%d) /* 0..(%s) */"
                % (dim.tag.axis, ccm(dim.length-1, PREC_NONE)))
                for dim in kernel.ordered_dims_by_tag_type(WORK_ITEM_IDX_TAG)]
            + [Line()])

    # }}}

    body = Block()

    # {{{ build lmem array declarators for prefetches

    for pf in kernel.prefetch.itervalues():
        smem_pf_array = POD(kernel.arg_dict[pf.input_vector].dtype, pf.name)
        for l in pf.dim_storage_lengths:
            smem_pf_array = ArrayOf(smem_pf_array, l)
        body.append(CLLocal(smem_pf_array))

    # }}}

    body.extend([
        Line(),
        build_loop_nest(ccm, kernel, 0)])

    mod.append(
        FunctionBody(
            CLRequiredWorkGroupSize(
                tuple(dim.length 
                    for dim in kernel.ordered_dims_by_tag_type(WORK_ITEM_IDX_TAG)),
                CLKernel(FunctionDeclaration(
                    Value("void", kernel.name), args))),
            body))

    # }}}

    return str(mod)

# }}}

# {{{ debugging

def print_kernel_info(knl):
    if hasattr(knl, "prefetch"):
        print "PREFETCH", knl.local_mem_use()
        for pf in knl.prefetch.itervalues():
            print "   %s[%s]: %s" % (pf.input_vector, pf.index_expr, pf.dims)
        print

    if hasattr(knl, "schedule"):
        print "Scheduling: ---------------------"
        for sched_item in knl.schedule:
            print sched_item
        print

    for ld in knl.dims:
        print ld
    print
    for t, e in knl.instructions:
        print "%s <- %s" % (t, e)

# }}}

# {{{ high-level modifiers

def split_dimension(knl, *args, **kwargs):
    knl, _ = knl.split_dimension(*args, **kwargs)
    return knl

def get_input_access_descriptors(kernel):
    """Return a dictionary mapping input vectors to
    a list of input access descriptor. An input access
    descriptor is a tuple (input_vec, index_expr).
    """
    from pytools import flatten
    result = {}
    for ivec in kernel.input_vectors():
        result[ivec] = [
                (ivec, iexpr)
                for iexpr in flatten(
                    VariableIndexExpressionCollector(ivec)(expression)
                    for lvalue, expression in kernel.instructions
                    )]

    return result

def add_prefetch_dims(kernel, input_access_descr, dims, loc_fetch_axes={}):
    """
    :arg input_access_descr: see :func:`get_input_access_descriptors`.
        May also be the name of the variable if there is only one
        reference to that variable.
    :arg dims: loop dimensions that are used to carry out the prefetch
    """

    if isinstance(input_access_descr, str):
        var_name = input_access_descr
        var_iads = get_input_access_descriptors(kernel)[var_name]

        if len(var_iads) != 1:
            raise ValueError("input access descriptor for variable %s is "
                    "not unique" % var_name)

        input_access_descr, = var_iads

    dims = [kernel.parse_sloppy_dim(dim) for dim in dims]
    ivec, iexpr = input_access_descr

    new_prefetch = getattr(kernel, "prefetch", {}).copy()
    if input_access_descr in new_prefetch:
        raise ValueError("a prefetch descriptor for the input access %s[%s] "
                "already exists" % (ivec, iexpr))

    new_prefetch[input_access_descr] = PrefetchDescriptor(
            kernel=kernel,
            input_vector=ivec,
            index_expr=iexpr,
            dims=dims,
            loc_fetch_axes={})

    return kernel.copy(prefetch=new_prefetch)

# }}}




class CompiledKernel:
    def __init__(self, context, kernel, size_args=None):
        self.kernel = kernel
        self.code = generate_code(kernel)
        self.cl_kernel = getattr(
                cl.Program(context, self.code).build(),
                kernel.name)

        arg_types = []
        for arg in kernel.args:
            if isinstance(arg, ScalarArg):
                arg_types.append(arg.dtype)
            else:
                arg_types.append(None)

        self.cl_kernel.set_scalar_arg_dtypes(arg_types)

        from pymbolic import compile
        if size_args is None:
            self.size_args = [arg.name for arg in kernel.args if isinstance(arg, ScalarArg)]
        else:
            self.size_args = size_args

        self.global_size_func = compile(self.kernel.group_dims(), self.size_args)
        self.local_size_func = compile(self.kernel.local_dims(), self.size_args)




# {{{ speed measurement


# }}}




# driver ----------------------------------------------------------------------
def drive_timing_run(kernel_generator, queue, launch, flop_count=None):

    def time_run(compiled_knl, warmup_rounds=2, timing_rounds=5):
        check = True
        for i in range(warmup_rounds):
            launch(compiled_knl.cl_kernel,
                    compiled.global_size_func, compiled.local_size_func,
                    check=check)
            check = False

        events = []
        for i in range(timing_rounds):
            events.append(
                    launch(compiled_knl.cl_kernel,
                        compiled.global_size_func, compiled.local_size_func,
                        check=check))
        for evt in events:
            evt.wait()

        return sum(1e-9*evt.profile.END-1e-9*evt.profile.START for evt in events)/timing_rounds

    soln_count = 0
    for kernel in kernel_generator:

        compiled = CompiledKernel(queue.context, kernel)

        print "-----------------------------------------------"
        print "SOLUTION #%d" % soln_count
        print "-----------------------------------------------"
        print compiled.code
        print "-----------------------------------------------"

        elapsed = time_run(compiled)

        print "time: %f" % elapsed
        if flop_count is not None:
            print "gflops/s: %f (#%d)" % (
                    flop_count/elapsed/1e9, soln_count)
        print "-----------------------------------------------"

        soln_count += 1

    print "%d solutions" % soln_count




# vim: foldmethod=marker

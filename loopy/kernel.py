"""Elements of loopy's user-facing language."""

from __future__ import division

import numpy as np
from pytools import Record, memoize_method
from pymbolic.mapper.dependency import DependencyMapper
import pyopencl as cl




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
            strides = tuple(strides)

        if shape is not None:
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
    def __init__(self, name, dtype, approximately):
        self.name = name
        self.dtype = np.dtype(dtype)
        self.approximately = approximately

    def __repr__(self):
        return "<ScalarArg '%s' of type %s>" % (self.name, self.dtype)

# }}}

# {{{ index tags

class IndexTag(Record):
    __slots__ = []

    def __hash__(self):
        raise RuntimeError("use .key to hash index tags")




class ParallelTag(IndexTag):
    pass

class UniqueTag(IndexTag):
    @property
    def key(self):
        return type(self)

class ParallelTagWithAxis(ParallelTag, UniqueTag):
    __slots__ = ["axis", "forced_length"]

    def __init__(self, axis, forced_length=None):
        Record.__init__(self,
                axis=axis, forced_length=forced_length)

    @property
    def key(self):
        return (type(self), self.axis)

    def __repr__(self):
        if self.forced_length:
            return "%s(%d, flen=%d)" % (
                    self.print_name, self.axis,
                    self.forced_length)
        else:
            return "%s(%d)" % (
                    self.print_name, self.axis)

class TAG_GROUP_IDX(ParallelTagWithAxis):
    print_name = "GROUP_IDX"

class TAG_WORK_ITEM_IDX(ParallelTagWithAxis):
    print_name = "WORK_ITEM_IDX"

class TAG_ILP(ParallelTag):
    def __repr__(self):
        return "TAG_ILP"

class BaseUnrollTag(IndexTag):
    pass

class TAG_UNROLL_STATIC(BaseUnrollTag):
    def __repr__(self):
        return "TAG_UNROLL_STATIC"

class TAG_UNROLL_INCR(BaseUnrollTag):
    def __repr__(self):
        return "TAG_UNROLL_INCR"

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

# {{{ loop kernel object

class LoopKernel(Record):
    """
    :ivar device: :class:`pyopencl.Device`
    :ivar domain: :class:`islpy.BasicSet`
    :ivar iname_to_tag:
    :ivar instructions:
    :ivar args:
    :ivar prefetch:
    :ivar schedule:
    :ivar register_prefetch:
    :ivar name:
    :ivar preamble:
    """

    def __init__(self, device, domain, instructions, args=None, prefetch={}, schedule=None,
            register_prefetch=None, name="loopy_kernel",
            iname_to_tag={}, is_divisible=lambda dividend, divisor: False,
            preamble=None):
        """
        :arg domain: a :class:`islpy.BasicSet`, or a string parseable to a basic set by the isl.
            Example: "{[i,j]: 0<=i < 10 and 0<= j < 9}"
        :arg iname_to_tag: a map from loop domain variables to subclasses of :class:`IndexTag`
        """
        from pymbolic import parse

        def parse_if_necessary(v):
            if isinstance(v, str):
                return parse(v)
            else:
                return v

        if isinstance(domain, str):
            import islpy as isl
            ctx = isl.Context()
            domain = isl.Set.read_from_str(ctx, domain, nparam=-1)

        insns = [
                (parse_if_necessary(lvalue),
                    parse_if_necessary(expr))
                for lvalue, expr in instructions]

        Record.__init__(self,
                device=device, args=args, domain=domain, instructions=insns,
                prefetch=prefetch, schedule=schedule,
                register_prefetch=register_prefetch, name=name,
                iname_to_tag=iname_to_tag,
                is_divisible=is_divisible,
                preamble=preamble)

        # FIXME: assert empyt intersection of loop vars and args
        # FIXME: assert that isl knows about loop parameters

    @property
    @memoize_method
    def space(self):
        return self.domain.get_dim()

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

    @memoize_method
    def scalar_args(self):
        if self.args is None:
            return set()
        else:
            return set(arg.name for arg in self.args if isinstance(arg, ScalarArg))

    @memoize_method
    def all_inames(self):
        from islpy import dim_type
        return set(self.space.get_var_dict(dim_type.set).iterkeys())

    @memoize_method
    def output_inames(self):
        dm = DependencyMapper(include_subscripts=False)

        output_indices = set()
        for lvalue, expr in self.instructions:
            output_indices.update(
                    set(v.name for v in dm(lvalue))
                    & self.all_inames())

        return output_indices - set(arg.name for arg in self.args)

    @memoize_method
    def reduction_inames(self):
        return self.all_inames() - self.output_inames()

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
        return sum(pf.nbytes for pf in self.prefetch.itervalues())

    @memoize_method
    def input_vectors(self):
        dm = DependencyMapper(include_subscripts=False)

        input_vectors = set()
        for lvalue, expr in self.instructions:
            input_vectors.update(
                    set(v.name for v in dm(expr)))

        return input_vectors - self.all_inames() - self.scalar_args()

    @memoize_method
    def output_vectors(self):
        dm = DependencyMapper(include_subscripts=False)

        output_vectors = set()
        for lvalue, expr in self.instructions:
            output_vectors.update(
                    set(v.name for v in dm(lvalue)))

        return output_vectors - self.all_inames() - self.scalar_args()

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
        # FIXME delete me
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
            raise RuntimeError("cannot substitute-prefetches already generated")
            #copy.prefetch = self._subst_prefetch(old_var, new_expr)

        if self.schedule is not None:
            raise RuntimeError("cannot substitute-schedule already generated")

        return copy

    def split_dimension(self, name, inner_length, padded_length=None,
            outer_name=None, inner_name=None,
            outer_tag=None, inner_tag=None):

        outer_tag = parse_tag(outer_tag)
        inner_tag = parse_tag(inner_tag)

        if self.iname_to_tag.get(name) is not None:
            raise RuntimeError("cannot split tagged dimension '%s'" % name)

        # {{{ check for repeated unique tag keys

        new_tag_keys = set(tag.key
                for tag in [outer_tag, inner_tag]
                if tag is not None
                if isinstance(tag, UniqueTag))

        repeated_tag_keys = new_tag_keys & set(
                tag.key for tag in self.iname_to_tag.itervalues()
                if isinstance(tag, UniqueTag))

        if repeated_tag_keys:
            raise RuntimeError("repeated tag(s): %s" % repeated_tag_keys)

        # }}}

        if padded_length is not None:
            inner_tag = inner_tag.copy(forced_length=padded_length)

        if outer_name is None:
            outer_name = name+"_outer"
        if inner_name is None:
            inner_name = name+"_inner"

        new_iname_to_tag = self.iname_to_tag.copy()
        if inner_tag is not None:
            new_iname_to_tag[inner_name] = inner_tag
        if outer_tag is not None:
            new_iname_to_tag[outer_name] = outer_tag

        from islpy import dim_type
        outer_var_nr = self.space.size(dim_type.set)
        inner_var_nr = self.space.size(dim_type.set)+1
        new_domain = self.domain.add_dims(dim_type.set, 2)
        new_domain.set_dim_name(dim_type.set, outer_var_nr, outer_name)
        new_domain.set_dim_name(dim_type.set, inner_var_nr, inner_name)

        import islpy as isl
        from loopy.isl import make_slab

        space = new_domain.get_dim()
        inner_constraint_set = (
                make_slab(space, inner_name, 0, inner_length)
                # name = inner + length*outer
                .add_constraint(isl.Constraint.eq_from_names(
                    space, {name:1, inner_name: -1, outer_name:-inner_length})))

        name_dim_type, name_idx = space.get_var_dict()[name]
        new_domain = (new_domain
                .intersect(inner_constraint_set)
                .project_out(name_dim_type, name_idx, 1))

        from pymbolic import var
        inner = var(inner_name)
        outer = var(outer_name)
        new_loop_index = inner + outer*inner_length

        return (self
                .substitute(name, new_loop_index)
                .copy(domain=new_domain, iname_to_tag=new_iname_to_tag))

    def get_problems(self, parameters, emit_warnings=True):
        """
        :return: *(max_severity, list of (severity, msg))*, where *severity* ranges from 1-5.
            '5' means 'will certainly not run'.
        """
        msgs = []

        def msg(severity, s):
            if emit_warnings:
                from warnings import warn
                from loopy import LoopyAdvisory
                warn(s, LoopyAdvisory)

            msgs.append((severity, s))

        glens = self.tag_type_lengths(TAG_GROUP_IDX, allow_parameters=True)
        llens = self.tag_type_lengths(TAG_WORK_ITEM_IDX, allow_parameters=False)

        from pymbolic import evaluate
        glens = evaluate(glens, parameters)
        llens = evaluate(llens, parameters)

        if (max(len(glens), len(llens))
                > self.device.max_work_item_dimensions):
            msg(5, "too many work item dimensions")

        for i in range(len(llens)):
            if llens[i] > self.device.max_work_item_sizes[i]:
                msg(5, "group axis %d too big")

        from pytools import product
        if product(llens) > self.device.max_work_group_size:
            msg(5, "work group too big")

        from pyopencl.characterize import usable_local_mem_size
        if self.local_mem_use() > usable_local_mem_size(self.device):
            if self.device.local_mem_type == cl.device_local_mem_type.LOCAL:
                msg(5, "using too much local memory")
            else:
                msg(4, "using more local memory than available--"
                        "possibly OK due to cache nature")

        const_arg_count = sum(
                1 for arg in self.args
                if isinstance(arg, ArrayArg) and arg.constant_mem)

        if const_arg_count > self.device.max_constant_args:
            msg(5, "too many constant arguments")

        max_severity = 0
        for sev, msg in msgs:
            max_severity = max(sev, max_severity)
        return max_severity, msgs

# }}}

# vim: foldmethod=marker

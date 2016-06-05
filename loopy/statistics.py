from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2015 James Stevens"

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

import six

import loopy as lp
import warnings
from islpy import dim_type
import islpy as isl
from pytools import memoize_in
from pymbolic.mapper import CombineMapper
from functools import reduce
from loopy.kernel.data import MultiAssignmentBase
from loopy.diagnostic import warn, LoopyError


__doc__ = """

.. currentmodule:: loopy

.. autofunction:: get_op_poly

.. autofunction:: get_gmem_access_poly

.. autofunction:: sum_mem_access_to_bytes

.. autofunction:: get_synchronization_poly

.. autofunction:: gather_access_footprints
.. autofunction:: gather_access_footprint_bytes

"""


# {{{ ToCountMap

class ToCountMap:
    """Maps any type of key to an arithmetic type."""

    def __init__(self, init_dict=None):
        if init_dict is None:
            init_dict = {}
        self.dict = init_dict

    def __add__(self, other):
        result = self.dict.copy()
        for k, v in six.iteritems(other.dict):
            result[k] = self.dict.get(k, 0) + v
        return ToCountMap(result)

    def __radd__(self, other):
        if other != 0:
            raise ValueError("ToCountMap: Attempted to add ToCountMap "
                                "to {0} {1}. ToCountMap may only be added to "
                                "0 and other ToCountMap objects."
                                .format(type(other), other))
        return self

    def __mul__(self, other):
        if isinstance(other, isl.PwQPolynomial):
            return ToCountMap(dict(
                (index, self.dict[index]*other)
                for index in self.dict.keys()))
        else:
            raise ValueError("ToCountMap: Attempted to multiply "
                                "ToCountMap by {0} {1}."
                                .format(type(other), other))

    __rmul__ = __mul__

    def __getitem__(self, index):
        try:
            return self.dict[index]
        except KeyError:
            return isl.PwQPolynomial('{ 0 }')

    def __repr__(self):
        return repr(self.dict)

# }}}


def stringify_stats_mapping(m):
    result = ""
    for key in sorted(m.keys(), key=lambda k: str(k)):
        result += ("%s : %s\n" % (key, m[key]))
    return result


# {{{ ExpressionOpCounter

class ExpressionOpCounter(CombineMapper):

    def __init__(self, knl):
        self.knl = knl
        from loopy.expression import TypeInferenceMapper
        self.type_inf = TypeInferenceMapper(knl)

    def combine(self, values):
        return sum(values)

    def map_constant(self, expr):
        return ToCountMap()

    map_tagged_variable = map_constant
    map_variable = map_constant

    #def map_wildcard(self, expr):
    #    return 0,0

    #def map_function_symbol(self, expr):
    #    return 0,0

    def map_call(self, expr):
        return ToCountMap(
                    {(self.type_inf(expr), 'func:'+str(expr.function)): 1}
                    ) + self.rec(expr.parameters)

    # def map_call_with_kwargs(self, expr):  # implemented in CombineMapper

    def map_subscript(self, expr):  # implemented in CombineMapper
        return self.rec(expr.index)

    # def map_lookup(self, expr):  # implemented in CombineMapper

    def map_sum(self, expr):
        assert expr.children
        return ToCountMap(
                    {(self.type_inf(expr), 'add'): len(expr.children)-1}
                    ) + sum(self.rec(child) for child in expr.children)

    def map_product(self, expr):
        from pymbolic.primitives import is_zero
        assert expr.children
        return sum(ToCountMap({(self.type_inf(expr), 'mul'): 1})
                   + self.rec(child)
                   for child in expr.children
                   if not is_zero(child + 1)) + \
                   ToCountMap({(self.type_inf(expr), 'mul'): -1})

    def map_quotient(self, expr, *args):
        return ToCountMap({(self.type_inf(expr), 'div'): 1}) \
                                + self.rec(expr.numerator) \
                                + self.rec(expr.denominator)

    map_floor_div = map_quotient
    map_remainder = map_quotient

    def map_power(self, expr):
        return ToCountMap({(self.type_inf(expr), 'pow'): 1}) \
                                + self.rec(expr.base) \
                                + self.rec(expr.exponent)

    def map_left_shift(self, expr):
        return ToCountMap({(self.type_inf(expr), 'shift'): 1}) \
                                + self.rec(expr.shiftee) \
                                + self.rec(expr.shift)

    map_right_shift = map_left_shift

    def map_bitwise_not(self, expr):
        return ToCountMap({(self.type_inf(expr), 'bw'): 1}) \
                                + self.rec(expr.child)

    def map_bitwise_or(self, expr):
        return ToCountMap(
                        {(self.type_inf(expr), 'bw'): len(expr.children)-1}
                        ) + sum(self.rec(child) for child in expr.children)

    map_bitwise_xor = map_bitwise_or
    map_bitwise_and = map_bitwise_or

    def map_comparison(self, expr):
        return self.rec(expr.left)+self.rec(expr.right)

    def map_logical_not(self, expr):
        return self.rec(expr.child)

    def map_logical_or(self, expr):
        return sum(self.rec(child) for child in expr.children)

    map_logical_and = map_logical_or

    def map_if(self, expr):
        warnings.warn("ExpressionOpCounter counting ops as "
                      "sum of if-statement branches.")
        return self.rec(expr.condition) + self.rec(expr.then) + self.rec(expr.else_)

    def map_if_positive(self, expr):
        warnings.warn("ExpressionOpCounter counting ops as "
                      "sum of if_pos-statement branches.")
        return self.rec(expr.criterion) + self.rec(expr.then) + self.rec(expr.else_)

    def map_min(self, expr):
        return ToCountMap(
                        {(self.type_inf(expr), 'maxmin'): len(expr.children)-1}
                        ) + sum(self.rec(child) for child in expr.children)

    map_max = map_min

    def map_common_subexpression(self, expr):
        raise NotImplementedError("ExpressionOpCounter encountered "
                                  "common_subexpression, "
                                  "map_common_subexpression not implemented.")

    def map_substitution(self, expr):
        raise NotImplementedError("ExpressionOpCounter encountered substitution, "
                                  "map_substitution not implemented.")

    def map_derivative(self, expr):
        raise NotImplementedError("ExpressionOpCounter encountered derivative, "
                                  "map_derivative not implemented.")

    def map_slice(self, expr):
        raise NotImplementedError("ExpressionOpCounter encountered slice, "
                                  "map_slice not implemented.")

# }}}


# {{{ GlobalSubscriptCounter

class GlobalSubscriptCounter(CombineMapper):

    def __init__(self, knl):
        self.knl = knl
        from loopy.expression import TypeInferenceMapper
        self.type_inf = TypeInferenceMapper(knl)

    def combine(self, values):
        return sum(values)

    def map_constant(self, expr):
        return ToCountMap()

    map_tagged_variable = map_constant
    map_variable = map_constant

    def map_call(self, expr):
        return self.rec(expr.parameters)

    def map_subscript(self, expr):
        name = expr.aggregate.name  # name of array

        if name in self.knl.arg_dict:
            array = self.knl.arg_dict[name]
        else:
            # this is a temporary variable
            return self.rec(expr.index)

        if not isinstance(array, lp.GlobalArg):
            # this array is not in global memory
            return self.rec(expr.index)

        index = expr.index  # could be tuple or scalar index
        if not isinstance(index, tuple):
            index = (index,)

        from loopy.symbolic import get_dependencies
        from loopy.kernel.data import LocalIndexTag
        my_inames = get_dependencies(index) & self.knl.all_inames()
        local_id0 = None
        local_id_found = False
        for iname in my_inames:
            # find local id0
            tag = self.knl.iname_to_tag.get(iname)
            if isinstance(tag, LocalIndexTag):
                local_id_found = True
                if tag.axis == 0:
                    local_id0 = iname
                    break  # there will be only one local_id0

        if not local_id_found:
            # count as uniform access
            return ToCountMap(
                    {(self.type_inf(expr), 'uniform'): 1}
                    ) + self.rec(expr.index)

        if local_id0 is None:
            # only non-zero local id(s) found, assume non-consecutive access
            return ToCountMap(
                    {(self.type_inf(expr), 'nonconsecutive'): 1}
                    ) + self.rec(expr.index)

        # check coefficient of local_id0 for each axis
        from loopy.symbolic import CoefficientCollector
        from pymbolic.primitives import Variable
        for idx, axis_tag in zip(index, array.dim_tags):

            coeffs = CoefficientCollector()(idx)
            # check if he contains the lid 0 guy
            try:
                coeff_id0 = coeffs[Variable(local_id0)]
            except KeyError:
                # does not contain local_id0
                continue

            if coeff_id0 != 1:
                # non-consecutive access
                return ToCountMap(
                        {(self.type_inf(expr), 'nonconsecutive'): 1}
                        ) + self.rec(expr.index)

            # coefficient is 1, now determine if stride is 1
            from loopy.kernel.array import FixedStrideArrayDimTag
            if isinstance(axis_tag, FixedStrideArrayDimTag):
                stride = axis_tag.stride
            else:
                continue

            if stride != 1:
                # non-consecutive
                return ToCountMap(
                        {(self.type_inf(expr), 'nonconsecutive'): 1}
                        ) + self.rec(expr.index)

            # else, stride == 1, continue since another idx could contain id0

        # loop finished without returning, stride==1 for every instance of local_id0
        return ToCountMap(
                {(self.type_inf(expr), 'consecutive'): 1}
                ) + self.rec(expr.index)

    def map_sum(self, expr):
        if expr.children:
            return sum(self.rec(child) for child in expr.children)
        else:
            return ToCountMap()

    map_product = map_sum

    def map_quotient(self, expr, *args):
        return self.rec(expr.numerator) + self.rec(expr.denominator)

    map_floor_div = map_quotient
    map_remainder = map_quotient

    def map_power(self, expr):
        return self.rec(expr.base) + self.rec(expr.exponent)

    def map_left_shift(self, expr):
        return self.rec(expr.shiftee)+self.rec(expr.shift)

    map_right_shift = map_left_shift

    def map_bitwise_not(self, expr):
        return self.rec(expr.child)

    def map_bitwise_or(self, expr):
        return sum(self.rec(child) for child in expr.children)

    map_bitwise_xor = map_bitwise_or
    map_bitwise_and = map_bitwise_or

    def map_comparison(self, expr):
        return self.rec(expr.left)+self.rec(expr.right)

    map_logical_not = map_bitwise_not
    map_logical_or = map_bitwise_or
    map_logical_and = map_logical_or

    def map_if(self, expr):
        warnings.warn("GlobalSubscriptCounter counting GMEM accesses as "
                      "sum of if-statement branches.")
        return self.rec(expr.condition) + self.rec(expr.then) + self.rec(expr.else_)

    def map_if_positive(self, expr):
        warnings.warn("GlobalSubscriptCounter counting GMEM accesses as "
                      "sum of if_pos-statement branches.")
        return self.rec(expr.criterion) + self.rec(expr.then) + self.rec(expr.else_)

    map_min = map_bitwise_or
    map_max = map_min

    def map_common_subexpression(self, expr):
        raise NotImplementedError("GlobalSubscriptCounter encountered "
                                  "common_subexpression, "
                                  "map_common_subexpression not implemented.")

    def map_substitution(self, expr):
        raise NotImplementedError("GlobalSubscriptCounter encountered "
                                  "substitution, "
                                  "map_substitution not implemented.")

    def map_derivative(self, expr):
        raise NotImplementedError("GlobalSubscriptCounter encountered "
                                  "derivative, "
                                  "map_derivative not implemented.")

    def map_slice(self, expr):
        raise NotImplementedError("GlobalSubscriptCounter encountered slice, "
                                  "map_slice not implemented.")

# }}}


# {{{ AccessFootprintGatherer

class AccessFootprintGatherer(CombineMapper):
    def __init__(self, kernel, domain, ignore_uncountable=False):
        self.kernel = kernel
        self.domain = domain
        self.ignore_uncountable = ignore_uncountable

    @staticmethod
    def combine(values):
        assert values

        def merge_dicts(a, b):
            result = a.copy()

            for var_name, footprint in six.iteritems(b):
                if var_name in result:
                    result[var_name] = result[var_name] | footprint
                else:
                    result[var_name] = footprint

            return result

        from functools import reduce
        return reduce(merge_dicts, values)

    def map_constant(self, expr):
        return {}

    def map_variable(self, expr):
        return {}

    def map_subscript(self, expr):
        subscript = expr.index

        if not isinstance(subscript, tuple):
            subscript = (subscript,)

        from loopy.symbolic import get_access_range

        try:
            access_range = get_access_range(self.domain, subscript,
                    self.kernel.assumptions)
        except isl.Error:
            # Likely: index was non-linear, nothing we can do.
            if self.ignore_uncountable:
                return {}
            else:
                raise LoopyError("failed to gather footprint: %s" % expr)

        except TypeError:
            # Likely: index was non-linear, nothing we can do.
            if self.ignore_uncountable:
                return {}
            else:
                raise LoopyError("failed to gather footprint: %s" % expr)

        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        return self.combine([
            self.rec(expr.index),
            {expr.aggregate.name: access_range}])

# }}}


# {{{ count

def count(kernel, set):
    try:
        return set.card()
    except AttributeError:
        pass

    count = isl.PwQPolynomial.zero(
            set.space
            .drop_dims(dim_type.set, 0, set.dim(dim_type.set))
            .add_dims(dim_type.set, 1))

    set = set.make_disjoint()

    from loopy.isl_helpers import get_simple_strides

    for bset in set.get_basic_sets():
        bset_count = None
        bset_rebuilt = bset.universe(bset.space)

        bset_strides = get_simple_strides(bset, key_by="index")

        for i in range(bset.dim(isl.dim_type.set)):
            dmax = bset.dim_max(i)
            dmin = bset.dim_min(i)

            stride = bset_strides.get((dim_type.set, i))
            if stride is None:
                stride = 1

            length = isl.PwQPolynomial.from_pw_aff(dmax - dmin + stride)
            length = length.scale_down_val(stride)

            if bset_count is None:
                bset_count = length
            else:
                bset_count = bset_count * length

            # {{{ rebuild check domain

            zero = isl.Aff.zero_on_domain(isl.LocalSpace.from_space(bset.space))
            iname = isl.PwAff.from_aff(
                    zero.set_coefficient_val(isl.dim_type.in_, i, 1))
            dmin_matched = dmin.insert_dims(
                    dim_type.in_, 0, bset.dim(isl.dim_type.set))
            dmax_matched = dmax.insert_dims(
                    dim_type.in_, 0, bset.dim(isl.dim_type.set))
            for idx in range(bset.dim(isl.dim_type.set)):
                if bset.has_dim_id(isl.dim_type.set, idx):
                    dim_id = bset.get_dim_id(isl.dim_type.set, idx)
                    dmin_matched = dmin_matched.set_dim_id(
                            isl.dim_type.in_, idx, dim_id)
                    dmax_matched = dmax_matched.set_dim_id(
                            isl.dim_type.in_, idx, dim_id)

            bset_rebuilt = (
                    bset_rebuilt
                    & iname.le_set(dmax_matched)
                    & iname.ge_set(dmin_matched)
                    & (iname-dmin_matched).mod_val(stride).eq_set(zero))

            # }}}

        if bset_count is not None:
            count += bset_count

        is_subset = bset <= bset_rebuilt
        is_superset = bset >= bset_rebuilt

        if not (is_subset and is_superset):
            if is_subset:
                from loopy.diagnostic import warn
                warn(kernel, "count_overestimate",
                        "Barvinok wrappers are not installed. "
                        "Counting routines have overestimated the "
                        "number of integer points in your loop "
                        "domain.")
            elif is_superset:
                from loopy.diagnostic import warn
                warn(kernel, "count_underestimate",
                        "Barvinok wrappers are not installed. "
                        "Counting routines have underestimated the "
                        "number of integer points in your loop "
                        "domain.")
            else:
                from loopy.diagnostic import warn
                warn(kernel, "count_misestimate",
                        "Barvinok wrappers are not installed. "
                        "Counting routines have misestimated the "
                        "number of integer points in your loop "
                        "domain.")

    return count

# }}}


# {{{ get_op_poly

def get_op_poly(knl, numpy_types=True):

    """Count the number of operations in a loopy kernel.

    :parameter knl: A :class:`loopy.LoopKernel` whose operations are to be counted.

    :return: A mapping of **{(** *type* **,** :class:`string` **)**
             **:** :class:`islpy.PwQPolynomial` **}**.

             - The *type* specifies the type of the data being
               accessed. This can be a :class:`numpy.dtype` if
               *numpy_types* is True, otherwise the internal
               loopy type.

             - The string specifies the operation type as
               *add*, *sub*, *mul*, *div*, *pow*, *shift*, *bw* (bitwise), etc.

             - The :class:`islpy.PwQPolynomial` holds the number of operations of
               the kind specified in the key (in terms of the
               :class:`loopy.LoopKernel` *parameter inames*).

    Example usage::

        # (first create loopy kernel and specify array data types)

        poly = get_op_poly(knl)
        params = {'n': 512, 'm': 256, 'l': 128}
        f32add = poly[(np.dtype(np.float32), 'add')].eval_with_dict(params)
        f32mul = poly[(np.dtype(np.float32), 'mul')].eval_with_dict(params)

        # (now use these counts to predict performance)

    """

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    knl = infer_unknown_types(knl, expect_completion=True)
    knl = preprocess_kernel(knl)

    op_poly = ToCountMap()
    op_counter = ExpressionOpCounter(knl)
    for insn in knl.instructions:
        # how many times is this instruction executed?
        # check domain size:
        insn_inames = knl.insn_inames(insn)
        inames_domain = knl.get_inames_domain(insn_inames)
        domain = (inames_domain.project_out_except(insn_inames, [dim_type.set]))
        ops = op_counter(insn.assignee) + op_counter(insn.expression)
        op_poly = op_poly + ops*count(knl, domain)
    result = op_poly.dict

    if numpy_types:
        result = dict(
                ((dtype.numpy_dtype, kind), count)
                for (dtype, kind), count in six.iteritems(result))

    return result
# }}}


def sum_ops_to_dtypes(op_poly_dict):
    result = {}
    for (dtype, kind), v in op_poly_dict.items():
        new_key = dtype
        if new_key in result:
            result[new_key] += v
        else:
            result[new_key] = v

    return result


# {{{ get_gmem_access_poly
def get_gmem_access_poly(knl, numpy_types=True):  # for now just counting subscripts

    """Count the number of global memory accesses in a loopy kernel.

    :parameter knl: A :class:`loopy.LoopKernel` whose DRAM accesses are to be
                    counted.

    :return: A mapping of **{(** *type* **,** :class:`string` **,**
             :class:`string` **)** **:** :class:`islpy.PwQPolynomial` **}**.

             - The *type* specifies the type of the data being
               accessed. This can be a :class:`numpy.dtype` if
               *numpy_types* is True, otherwise the internal
               loopy type.

             - The first string in the map key specifies the global memory
               access type as
               *consecutive*, *nonconsecutive*, or *uniform*.

             - The second string in the map key specifies the global memory
               access type as a
               *load*, or a *store*.

             - The :class:`islpy.PwQPolynomial` holds the number of DRAM accesses
               with the characteristics specified in the key (in terms of the
               :class:`loopy.LoopKernel` *inames*).

    Example usage::

        # (first create loopy kernel and specify array data types)

        subscript_map = get_gmem_access_poly(knl)
        params = {'n': 512, 'm': 256, 'l': 128}

        f32_uncoalesced_load = subscript_map.dict[
                            (np.dtype(np.float32), 'nonconsecutive', 'load')
                            ].eval_with_dict(params)
        f32_coalesced_load = subscript_map.dict[
                            (np.dtype(np.float32), 'consecutive', 'load')
                            ].eval_with_dict(params)
        f32_coalesced_store = subscript_map.dict[
                            (np.dtype(np.float32), 'consecutive', 'store')
                            ].eval_with_dict(params)

        # (now use these counts to predict performance)

    """

    from loopy.preprocess import preprocess_kernel, infer_unknown_types

    class CacheHolder(object):
        pass

    cache_holder = CacheHolder()

    @memoize_in(cache_holder, "insn_count")
    def get_insn_count(knl, insn_inames, uniform=False):
        if uniform:
            from loopy.kernel.data import LocalIndexTag
            insn_inames = [iname for iname in insn_inames if not
                           isinstance(knl.iname_to_tag.get(iname), LocalIndexTag)]
        inames_domain = knl.get_inames_domain(insn_inames)
        domain = (inames_domain.project_out_except(
                                insn_inames, [dim_type.set]))
        return count(knl, domain)

    knl = infer_unknown_types(knl, expect_completion=True)
    knl = preprocess_kernel(knl)

    subs_poly = ToCountMap()
    subscript_counter = GlobalSubscriptCounter(knl)
    for insn in knl.instructions:
        # count subscripts, distinguishing loads and stores
        subs_expr = subscript_counter(insn.expression)
        subs_expr = ToCountMap(dict(
            (key + ("load",), val)
            for key, val in six.iteritems(subs_expr.dict)))
        subs_assignee = subscript_counter(insn.assignee)
        subs_assignee = ToCountMap(dict(
            (key + ("store",), val)
            for key, val in six.iteritems(subs_assignee.dict)))

        insn_inames = knl.insn_inames(insn)

        # use count excluding local index tags for uniform accesses
        for key in subs_expr.dict:
            poly = ToCountMap({key: subs_expr.dict[key]})
            if key[1] == "uniform":
                subs_poly = subs_poly + poly*get_insn_count(knl, insn_inames, True)
            else:
                subs_poly = subs_poly + poly*get_insn_count(knl, insn_inames)
        for key in subs_assignee.dict:
            poly = ToCountMap({key: subs_assignee.dict[key]})
            if key[1] == "uniform":
                subs_poly = subs_poly + poly*get_insn_count(knl, insn_inames, True)
            else:
                subs_poly = subs_poly + poly*get_insn_count(knl, insn_inames)

    result = subs_poly.dict

    if numpy_types:
        result = dict(
                ((dtype.numpy_dtype, kind, direction), count)
                for (dtype, kind, direction), count in six.iteritems(result))

    return result


def get_DRAM_access_poly(knl):
    from warnings import warn
    warn("get_DRAM_access_poly is deprecated. Use get_gmem_access_poly instead",
            DeprecationWarning, stacklevel=2)
    return get_gmem_access_poly(knl)

# }}}


# {{{ sum_mem_access_to_bytes

def sum_mem_access_to_bytes(m):
    """Sum the mapping returned by :func:`get_gmem_access_poly` to a mapping

    **{(** :class:`string` **,** :class:`string` **)**
    **:** :class:`islpy.PwQPolynomial` **}**

    i.e., aggregate the transfer numbers for all types into a single byte count.
    """

    result = {}
    for (dtype, kind, direction), v in m.items():
        new_key = (kind, direction)
        bytes_transferred = int(dtype.itemsize) * v
        if new_key in result:
            result[new_key] += bytes_transferred
        else:
            result[new_key] = bytes_transferred

    return result

# }}}


# {{{ get_synchronization_poly

def get_synchronization_poly(knl):

    """Count the number of synchronization events each thread encounters in a
    loopy kernel.

    :parameter knl: A :class:`loopy.LoopKernel` whose barriers are to be counted.

    :return: A dictionary mapping each type of synchronization event to a
            :class:`islpy.PwQPolynomial` holding the number of such events
            per thread.

            Possible keys include ``barrier_local``, ``barrier_global``
            (if supported by the target) and ``kernel_launch``.

    Example usage::

        # (first create loopy kernel and specify array data types)

        barrier_poly = get_barrier_poly(knl)
        params = {'n': 512, 'm': 256, 'l': 128}
        barrier_count = barrier_poly.eval_with_dict(params)

        # (now use this count to predict performance)

    """

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    from loopy.schedule import (EnterLoop, LeaveLoop, Barrier,
            CallKernel, ReturnFromKernel, RunInstruction)
    from operator import mul
    knl = infer_unknown_types(knl, expect_completion=True)
    knl = preprocess_kernel(knl)
    knl = lp.get_one_scheduled_kernel(knl)
    iname_list = []

    result = ToCountMap()

    one = isl.PwQPolynomial('{ 1 }')

    def get_count_poly(iname_list):
        if iname_list:  # (if iname_list is not empty)
            ct = (count(knl, (
                            knl.get_inames_domain(iname_list).
                            project_out_except(iname_list, [dim_type.set])
                            )), )
            return reduce(mul, ct)
        else:
            return one

    for sched_item in knl.schedule:
        if isinstance(sched_item, EnterLoop):
            if sched_item.iname:  # (if not empty)
                iname_list.append(sched_item.iname)
        elif isinstance(sched_item, LeaveLoop):
            if sched_item.iname:  # (if not empty)
                iname_list.pop()

        elif isinstance(sched_item, Barrier):
            result = result + ToCountMap(
                    {"barrier_%s" % sched_item.kind: get_count_poly(iname_list)})

        elif isinstance(sched_item, CallKernel):
            result = result + ToCountMap(
                    {"kernel_launch": get_count_poly(iname_list)})

        elif isinstance(sched_item, (ReturnFromKernel, RunInstruction)):
            pass

        else:
            raise LoopyError("unexpected schedule item: %s"
                    % type(sched_item).__name__)

    return result.dict

# }}}


# {{{ gather_access_footprints

def gather_access_footprints(kernel, ignore_uncountable=False):
    """Return a dictionary mapping ``(var_name, direction)``
    to :class:`islpy.Set` instances capturing which indices
    of each the array *var_name* are read/written (where
    *direction* is either ``read`` or ``write``.

    :arg ignore_uncountable: If *True*, an error will be raised for
        accesses on which the footprint cannot be determined (e.g.
        data-dependent or nonlinear indices)
    """

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    kernel = infer_unknown_types(kernel, expect_completion=True)
    kernel = preprocess_kernel(kernel)

    write_footprints = []
    read_footprints = []

    for insn in kernel.instructions:
        if not isinstance(insn, MultiAssignmentBase):
            warn(kernel, "count_non_assignment",
                    "Non-assignment instruction encountered in "
                    "gather_access_footprints, not counted")
            continue

        insn_inames = kernel.insn_inames(insn)
        inames_domain = kernel.get_inames_domain(insn_inames)
        domain = (inames_domain.project_out_except(insn_inames, [dim_type.set]))

        afg = AccessFootprintGatherer(kernel, domain,
                ignore_uncountable=ignore_uncountable)

        for assignee in insn.assignees:
            write_footprints.append(afg(insn.assignees))
        read_footprints.append(afg(insn.expression))

    write_footprints = AccessFootprintGatherer.combine(write_footprints)
    read_footprints = AccessFootprintGatherer.combine(read_footprints)

    result = {}

    for vname, footprint in six.iteritems(write_footprints):
        result[(vname, "write")] = footprint

    for vname, footprint in six.iteritems(read_footprints):
        result[(vname, "read")] = footprint

    return result


def gather_access_footprint_bytes(kernel, ignore_uncountable=False):
    """Return a dictionary mapping ``(var_name, direction)`` to
    :class:`islpy.PwQPolynomial` instances capturing the number of bytes  are
    read/written (where *direction* is either ``read`` or ``write`` on array
    *var_name*

    :arg ignore_uncountable: If *True*, an error will be raised for
        accesses on which the footprint cannot be determined (e.g.
        data-dependent or nonlinear indices)
    """

    result = {}
    fp = gather_access_footprints(kernel, ignore_uncountable=ignore_uncountable)

    for key, var_fp in fp.items():
        vname, direction = key

        var_descr = kernel.get_var_descriptor(vname)
        bytes_transferred = (
                int(var_descr.dtype.numpy_dtype.itemsize)
                * count(kernel, var_fp))
        if key in result:
            result[key] += bytes_transferred
        else:
            result[key] = bytes_transferred

    return result

# }}}

# vim: foldmethod=marker

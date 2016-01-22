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
from pymbolic.mapper import CombineMapper
from functools import reduce
from loopy.kernel.data import Assignment
from loopy.diagnostic import warn


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
                                "to {} {}. ToCountMap may only be added to "
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
                                "ToCountMap by {} {}."
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
        return self.rec(expr.parameters)

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
    def __init__(self, kernel, domain):
        self.kernel = kernel
        self.domain = domain

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
            return
        except TypeError:
            # Likely: index was non-linear, nothing we can do.
            return

        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        return self.combine([
            self.rec(expr.index),
            {expr.aggregate.name: access_range}])

# }}}


# {{{ count

def count(kernel, bset):
    try:
        return bset.card()
    except AttributeError:
        pass

    if not bset.is_box():
        from loopy.diagnostic import warn
        warn(kernel, "count_overestimate",
                "Barvinok wrappers are not installed. "
                "Counting routines may overestimate the "
                "number of integer points in your loop "
                "domain.")

    result = None

    for i in range(bset.dim(isl.dim_type.set)):
        dmax = bset.dim_max(i)
        dmin = bset.dim_min(i)

        length = isl.PwQPolynomial.from_pw_aff(dmax - dmin + 1)

        if result is None:
            result = length
        else:
            result = result * length

    return result

# }}}


# {{{ get_op_poly

def get_op_poly(knl):

    """Count the number of operations in a loopy kernel.

    :parameter knl: A :class:`loopy.LoopKernel` whose operations are to be counted.

    :return: A mapping of **{(** :class:`numpy.dtype` **,** :class:`string` **)**
             **:** :class:`islpy.PwQPolynomial` **}**.

             - The :class:`numpy.dtype` specifies the type of the data being
               operated on.

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
    return op_poly.dict

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
def get_gmem_access_poly(knl):  # for now just counting subscripts

    """Count the number of global memory accesses in a loopy kernel.

    :parameter knl: A :class:`loopy.LoopKernel` whose DRAM accesses are to be
                    counted.

    :return: A mapping of **{(** :class:`numpy.dtype` **,** :class:`string` **,**
             :class:`string` **)** **:** :class:`islpy.PwQPolynomial` **}**.

             - The :class:`numpy.dtype` specifies the type of the data being
               accessed.

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
    knl = infer_unknown_types(knl, expect_completion=True)
    knl = preprocess_kernel(knl)

    subs_poly = ToCountMap()
    subscript_counter = GlobalSubscriptCounter(knl)
    for insn in knl.instructions:
        insn_inames = knl.insn_inames(insn)
        inames_domain = knl.get_inames_domain(insn_inames)
        domain = (inames_domain.project_out_except(insn_inames, [dim_type.set]))
        subs_expr = subscript_counter(insn.expression)
        subs_expr = ToCountMap(dict(
            (key + ("load",), val)
            for key, val in six.iteritems(subs_expr.dict)))

        subs_assignee = subscript_counter(insn.assignee)
        subs_assignee = ToCountMap(dict(
            (key + ("store",), val)
            for key, val in six.iteritems(subs_assignee.dict)))

        subs_poly = subs_poly + (subs_expr + subs_assignee)*count(knl, domain)
    return subs_poly.dict


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


# {{{ get_barrier_poly

def get_barrier_poly(knl):

    """Count the number of barriers each thread encounters in a loopy kernel.

    :parameter knl: A :class:`loopy.LoopKernel` whose barriers are to be counted.

    :return: An :class:`islpy.PwQPolynomial` holding the number of barrier calls
             made (in terms of the :class:`loopy.LoopKernel` *inames*).

    Example usage::

        # (first create loopy kernel and specify array data types)

        barrier_poly = get_barrier_poly(knl)
        params = {'n': 512, 'm': 256, 'l': 128}
        barrier_count = barrier_poly.eval_with_dict(params)

        # (now use this count to predict performance)

    """

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    from loopy.schedule import EnterLoop, LeaveLoop, Barrier
    from operator import mul
    knl = infer_unknown_types(knl, expect_completion=True)
    knl = preprocess_kernel(knl)
    knl = lp.get_one_scheduled_kernel(knl)
    iname_list = []
    barrier_poly = isl.PwQPolynomial('{ 0 }')

    for sched_item in knl.schedule:
        if isinstance(sched_item, EnterLoop):
            if sched_item.iname:  # (if not empty)
                iname_list.append(sched_item.iname)
        elif isinstance(sched_item, LeaveLoop):
            if sched_item.iname:  # (if not empty)
                iname_list.pop()
        elif isinstance(sched_item, Barrier):
            if iname_list:  # (if iname_list is not empty)
                ct = (count(knl, (
                                knl.get_inames_domain(iname_list).
                                project_out_except(iname_list, [dim_type.set])
                                )), )
                barrier_poly += reduce(mul, ct)
            else:
                barrier_poly += isl.PwQPolynomial('{ 1 }')

    return barrier_poly

# }}}


# {{{ gather_access_footprints

def gather_access_footprints(kernel):
    # TODO: Docs

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    kernel = infer_unknown_types(kernel, expect_completion=True)
    kernel = preprocess_kernel(kernel)

    write_footprints = []
    read_footprints = []

    for insn in kernel.instructions:
        if not isinstance(insn, Assignment):
            warn(kernel, "count_non_assignment",
                    "Non-assignment instruction encountered in "
                    "gather_access_footprints, not counted")
            continue

        insn_inames = kernel.insn_inames(insn)
        inames_domain = kernel.get_inames_domain(insn_inames)
        domain = (inames_domain.project_out_except(insn_inames, [dim_type.set]))

        afg = AccessFootprintGatherer(kernel, domain)

        write_footprints.append(afg(insn.assignee))
        read_footprints.append(afg(insn.expression))

    write_footprints = AccessFootprintGatherer.combine(write_footprints)
    read_footprints = AccessFootprintGatherer.combine(read_footprints)

    result = {}

    for vname, footprint in six.iteritems(write_footprints):
        result[(vname, "write")] = footprint

    for vname, footprint in six.iteritems(read_footprints):
        result[(vname, "read")] = footprint

    return result

# }}}

# vim: foldmethod=marker

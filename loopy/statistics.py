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
from islpy import dim_type
import islpy as isl
from pymbolic.mapper import CombineMapper
from functools import reduce
from loopy.kernel.data import (
        MultiAssignmentBase, TemporaryVariable, AddressSpace)
from loopy.diagnostic import warn_with_kernel, LoopyError
from loopy.symbolic import CoefficientCollector
from pytools import Record, memoize_method


__doc__ = """

.. currentmodule:: loopy

.. autoclass:: ToCountMap
.. autoclass:: CountGranularity
.. autoclass:: Op
.. autoclass:: MemAccess

.. autofunction:: get_op_map
.. autofunction:: get_mem_access_map
.. autofunction:: get_synchronization_map

.. autofunction:: gather_access_footprints
.. autofunction:: gather_access_footprint_bytes

.. currentmodule:: loopy.statistics

.. autoclass:: GuardedPwQPolynomial

.. currentmodule:: loopy
"""


# {{{ GuardedPwQPolynomial

class GuardedPwQPolynomial(object):
    def __init__(self, pwqpolynomial, valid_domain):
        self.pwqpolynomial = pwqpolynomial
        self.valid_domain = valid_domain

    def __add__(self, other):
        if isinstance(other, GuardedPwQPolynomial):
            return GuardedPwQPolynomial(
                    self.pwqpolynomial + other.pwqpolynomial,
                    self.valid_domain & other.valid_domain)
        else:
            return GuardedPwQPolynomial(
                    self.pwqpolynomial + other,
                    self.valid_domain)

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, GuardedPwQPolynomial):
            return GuardedPwQPolynomial(
                    self.pwqpolynomial * other.pwqpolynomial,
                    self.valid_domain & other.valid_domain)
        else:
            return GuardedPwQPolynomial(
                    self.pwqpolynomial * other,
                    self.valid_domain)

    __rmul__ = __mul__

    def eval_with_dict(self, value_dict):
        space = self.pwqpolynomial.space
        pt = isl.Point.zero(space.params())

        for i in range(space.dim(dim_type.param)):
            par_name = space.get_dim_name(dim_type.param, i)
            pt = pt.set_coordinate_val(
                dim_type.param, i, value_dict[par_name])

        if not (isl.Set.from_point(pt) <= self.valid_domain):
            raise ValueError("evaluation point outside of domain of "
                    "definition of piecewise quasipolynomial")

        return self.pwqpolynomial.eval(pt).to_python()

    @staticmethod
    def zero():
        p = isl.PwQPolynomial('{ 0 }')
        return GuardedPwQPolynomial(p, isl.Set.universe(p.domain().space))

    def __str__(self):
        return str(self.pwqpolynomial)

    def __repr__(self):
        return repr(self.pwqpolynomial)

# }}}


# {{{ ToCountMap

class ToCountMap(object):
    """Maps any type of key to an arithmetic type.

    .. automethod:: filter_by
    .. automethod:: filter_by_func
    .. automethod:: group_by
    .. automethod:: to_bytes
    .. automethod:: sum
    .. automethod:: eval_and_sum

    """

    def __init__(self, init_dict=None, val_type=GuardedPwQPolynomial):
        if init_dict is None:
            init_dict = {}
        self.count_map = init_dict
        self.val_type = val_type

    def __add__(self, other):
        result = self.count_map.copy()
        for k, v in six.iteritems(other.count_map):
            result[k] = self.count_map.get(k, 0) + v
        return ToCountMap(result, self.val_type)

    def __radd__(self, other):
        if other != 0:
            raise ValueError("ToCountMap: Attempted to add ToCountMap "
                                "to {0} {1}. ToCountMap may only be added to "
                                "0 and other ToCountMap objects."
                                .format(type(other), other))
        return self

    def __mul__(self, other):
        if isinstance(other, GuardedPwQPolynomial):
            return ToCountMap(dict(
                (index, self.count_map[index]*other)
                for index in self.keys()))
        else:
            raise ValueError("ToCountMap: Attempted to multiply "
                                "ToCountMap by {0} {1}."
                                .format(type(other), other))

    __rmul__ = __mul__

    def __getitem__(self, index):
        try:
            return self.count_map[index]
        except KeyError:
            #TODO what is the best way to handle this?
            if self.val_type is GuardedPwQPolynomial:
                return GuardedPwQPolynomial.zero()
            else:
                return 0

    def __setitem__(self, index, value):
        self.count_map[index] = value

    def __repr__(self):
        return repr(self.count_map)

    def __len__(self):
        return len(self.count_map)

    def get(self, key, default=None):
        return self.count_map.get(key, default)

    def items(self):
        return self.count_map.items()

    def keys(self):
        return self.count_map.keys()

    def pop(self, item):
        return self.count_map.pop(item)

    def copy(self):
        return ToCountMap(dict(self.count_map), self.val_type)

    def with_set_attributes(self, **kwargs):
        return ToCountMap(dict(
            (key.copy(**kwargs), val)
            for key, val in six.iteritems(self.count_map)),
            self.val_type)

    def filter_by(self, **kwargs):
        """Remove items without specified key fields.

        :arg kwargs: Keyword arguments matching fields in the keys of the
            :class:`ToCountMap`, each given a list of allowable values for that
            key field.

        :return: A :class:`ToCountMap` containing the subset of the items in
            the original :class:`ToCountMap` that match the field values
            passed.

        Example usage::

            # (first create loopy kernel and specify array data types)

            params = {'n': 512, 'm': 256, 'l': 128}
            mem_map = lp.get_mem_access_map(knl)
            filtered_map = mem_map.filter_by(direction=['load'],
                                             variable=['a','g'])
            tot_loads_a_g = filtered_map.eval_and_sum(params)

            # (now use these counts to, e.g., predict performance)

        """

        result_map = ToCountMap(val_type=self.val_type)

        from loopy.types import to_loopy_type
        if 'dtype' in kwargs.keys():
            kwargs['dtype'] = [to_loopy_type(d) for d in kwargs['dtype']]

        # for each item in self.count_map
        for self_key, self_val in self.items():
            try:
                # check to see if key attribute values match all filters
                for arg_field, allowable_vals in kwargs.items():
                    attr_val = getattr(self_key, arg_field)
                    # see if the value is in the filter list
                    if attr_val not in allowable_vals:
                        break
                else:  # loop terminated without break or error
                    result_map[self_key] = self_val
            except(AttributeError):
                # the field passed is not a field of this key
                continue

        return result_map

    def filter_by_func(self, func):
        """Keep items that pass a test.

        :arg func: A function that takes a map key a parameter and returns a
            :class:`bool`.

        :arg: A :class:`ToCountMap` containing the subset of the items in the
            original :class:`ToCountMap` for which func(key) is true.

        Example usage::

            # (first create loopy kernel and specify array data types)

            params = {'n': 512, 'm': 256, 'l': 128}
            mem_map = lp.get_mem_access_map(knl)
            def filter_func(key):
                return key.lid_strides[0] > 1 and key.lid_strides[0] <= 4:

            filtered_map = mem_map.filter_by_func(filter_func)
            tot = filtered_map.eval_and_sum(params)

            # (now use these counts to, e.g., predict performance)

        """

        result_map = ToCountMap(val_type=self.val_type)

        # for each item in self.count_map, call func on the key
        for self_key, self_val in self.items():
            if func(self_key):
                result_map[self_key] = self_val

        return result_map

    def group_by(self, *args):
        """Group map items together, distinguishing by only the key fields
        passed in args.

        :arg args: Zero or more :class:`str` fields of map keys.

        :return: A :class:`ToCountMap` containing the same total counts grouped
            together by new keys that only contain the fields specified in the
            arguments passed.

        Example usage::

            # (first create loopy kernel and specify array data types)

            params = {'n': 512, 'm': 256, 'l': 128}
            mem_map = get_mem_access_map(knl)
            grouped_map = mem_map.group_by('mtype', 'dtype', 'direction')

            f32_global_ld = grouped_map[MemAccess(mtype='global',
                                                  dtype=np.float32,
                                                  direction='load')
                                       ].eval_with_dict(params)
            f32_global_st = grouped_map[MemAccess(mtype='global',
                                                  dtype=np.float32,
                                                  direction='store')
                                       ].eval_with_dict(params)
            f32_local_ld = grouped_map[MemAccess(mtype='local',
                                                 dtype=np.float32,
                                                 direction='load')
                                      ].eval_with_dict(params)
            f32_local_st = grouped_map[MemAccess(mtype='local',
                                                 dtype=np.float32,
                                                 direction='store')
                                      ].eval_with_dict(params)

            op_map = get_op_map(knl)
            ops_dtype = op_map.group_by('dtype')

            f32ops = ops_dtype[Op(dtype=np.float32)].eval_with_dict(params)
            f64ops = ops_dtype[Op(dtype=np.float64)].eval_with_dict(params)
            i32ops = ops_dtype[Op(dtype=np.int32)].eval_with_dict(params)

            # (now use these counts to, e.g., predict performance)

        """

        result_map = ToCountMap(val_type=self.val_type)

        # make sure all item keys have same type
        if self.count_map:
            key_type = type(list(self.keys())[0])
            if not all(isinstance(x, key_type) for x in self.keys()):
                raise ValueError("ToCountMap: group_by() function may only "
                                 "be used on ToCountMaps with uniform keys")
        else:
            return result_map

        # for each item in self.count_map
        for self_key, self_val in self.items():
            new_key = key_type()

            # set all specified fields
            for field in args:
                setattr(new_key, field, getattr(self_key, field))

            if new_key in result_map.keys():
                result_map[new_key] += self_val
            else:
                result_map[new_key] = self_val

        return result_map

    def to_bytes(self):
        """Convert counts to bytes using data type in map key.

        :return: A :class:`ToCountMap` mapping each original key to an
            :class:`islpy.PwQPolynomial` with counts in bytes rather than
            instances.

        Example usage::

            # (first create loopy kernel and specify array data types)

            bytes_map = get_mem_access_map(knl).to_bytes()
            params = {'n': 512, 'm': 256, 'l': 128}

            s1_g_ld_byt = bytes_map.filter_by(
                                mtype=['global'], lid_strides={0: 1},
                                direction=['load']).eval_and_sum(params)
            s2_g_ld_byt = bytes_map.filter_by(
                                mtype=['global'], lid_strides={0: 2},
                                direction=['load']).eval_and_sum(params)
            s1_g_st_byt = bytes_map.filter_by(
                                mtype=['global'], lid_strides={0: 1},
                                direction=['store']).eval_and_sum(params)
            s2_g_st_byt = bytes_map.filter_by(
                                mtype=['global'], lid_strides={0: 2},
                                direction=['store']).eval_and_sum(params)

            # (now use these counts to, e.g., predict performance)

        """

        result = self.copy()

        for key, val in self.items():
            bytes_processed = int(key.dtype.itemsize) * val
            result[key] = bytes_processed

        #TODO again, is this okay?
        result.val_type = int

        return result

    def sum(self):
        """Add all counts in ToCountMap.

        :return: An :class:`islpy.PwQPolynomial` or :class:`int` containing the
            sum of counts.

        """

        if self.val_type is GuardedPwQPolynomial:
            total = GuardedPwQPolynomial.zero()
        else:
            total = 0

        for k, v in self.items():
            total += v
        return total

    #TODO test and document
    def eval(self, params):
        result = self.copy()
        for key, val in self.items():
            result[key] = val.eval_with_dict(params)
        result.val_type = int
        return result

    def eval_and_sum(self, params):
        """Add all counts in :class:`ToCountMap` and evaluate with provided
        parameter dict.

        :return: An :class:`int` containing the sum of all counts in the
            :class:`ToCountMap` evaluated with the parameters provided.

        Example usage::

            # (first create loopy kernel and specify array data types)

            params = {'n': 512, 'm': 256, 'l': 128}
            mem_map = lp.get_mem_access_map(knl)
            filtered_map = mem_map.filter_by(direction=['load'],
                                             variable=['a', 'g'])
            tot_loads_a_g = filtered_map.eval_and_sum(params)

            # (now use these counts to, e.g., predict performance)

        """
        return self.sum().eval_with_dict(params)

# }}}


def stringify_stats_mapping(m):
    result = ""
    for key in sorted(m.keys(), key=lambda k: str(k)):
        result += ("%s : %s\n" % (key, m[key]))
    return result


class CountGranularity:
    """Strings specifying whether an operation should be counted once per
    *work-item*, *sub-group*, or *work-group*.

    .. attribute:: WORKITEM

       A :class:`str` that specifies that an operation should be counted
       once per *work-item*.

    .. attribute:: SUBGROUP

       A :class:`str` that specifies that an operation should be counted
       once per *sub-group*.

    .. attribute:: WORKGROUP

       A :class:`str` that specifies that an operation should be counted
       once per *work-group*.

    """

    WORKITEM = "workitem"
    SUBGROUP = "subgroup"
    WORKGROUP = "workgroup"
    ALL = [WORKITEM, SUBGROUP, WORKGROUP]


# {{{ Op descriptor

class Op(Record):
    """A descriptor for a type of arithmetic operation.

    .. attribute:: dtype

       A :class:`loopy.types.LoopyType` or :class:`numpy.dtype` that specifies the
       data type operated on.

    .. attribute:: name

       A :class:`str` that specifies the kind of arithmetic operation as
       *add*, *mul*, *div*, *pow*, *shift*, *bw* (bitwise), etc.

    .. attribute:: count_granularity

       A :class:`str` that specifies whether this operation should be counted
       once per *work-item*, *sub-group*, or *work-group*. The granularities
       allowed can be found in :class:`CountGranularity`, and may be accessed,
       e.g., as ``CountGranularity.WORKITEM``. A work-item is a single instance
       of computation executing on a single processor (think 'thread'), a
       collection of which may be grouped together into a work-group. Each
       work-group executes on a single compute unit with all work-items within
       the work-group sharing local memory. A sub-group is an
       implementation-dependent grouping of work-items within a work-group,
       analagous to an NVIDIA CUDA warp.

    """

    def __init__(self, dtype=None, name=None, count_granularity=None):
        if count_granularity not in CountGranularity.ALL+[None]:
            raise ValueError("Op.__init__: count_granularity '%s' is "
                    "not allowed. count_granularity options: %s"
                    % (count_granularity, CountGranularity.ALL+[None]))
        if dtype is None:
            Record.__init__(self, dtype=dtype, name=name,
                            count_granularity=count_granularity)
        else:
            from loopy.types import to_loopy_type
            Record.__init__(self, dtype=to_loopy_type(dtype), name=name,
                            count_granularity=count_granularity)

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        # Record.__repr__ overridden for consistent ordering and conciseness
        return "Op(%s, %s, %s)" % (self.dtype, self.name, self.count_granularity)

# }}}


# {{{ MemAccess descriptor

class MemAccess(Record):
    """A descriptor for a type of memory access.

    .. attribute:: mtype

       A :class:`str` that specifies the memory type accessed as **global**
       or **local**

    .. attribute:: dtype

       A :class:`loopy.types.LoopyType` or :class:`numpy.dtype` that specifies the
       data type accessed.

    .. attribute:: lid_strides

       A :class:`dict` of **{** :class:`int` **:**
       :class:`pymbolic.primitives.Expression` or :class:`int` **}** that
       specifies local strides for each local id in the memory access index.
       Local ids not found will not be present in ``lid_strides.keys()``.
       Uniform access (i.e. work-items within a sub-group access the same
       item) is indicated by setting ``lid_strides[0]=0``, but may also occur
       when no local id 0 is found, in which case the 0 key will not be
       present in lid_strides.

    .. attribute:: gid_strides

       A :class:`dict` of **{** :class:`int` **:**
       :class:`pymbolic.primitives.Expression` or :class:`int` **}** that
       specifies global strides for each global id in the memory access index.
       global ids not found will not be present in ``gid_strides.keys()``.

    .. attribute:: direction

       A :class:`str` that specifies the direction of memory access as
       **load** or **store**.

    .. attribute:: variable

       A :class:`str` that specifies the variable name of the data
       accessed.

    .. attribute:: variable_tag

       A :class:`str` that specifies the variable tag of a
       :class:`loopy.symbolic.TaggedVariable`.

    .. attribute:: count_granularity

       A :class:`str` that specifies whether this operation should be counted
       once per *work-item*, *sub-group*, or *work-group*. The granularities
       allowed can be found in :class:`CountGranularity`, and may be accessed,
       e.g., as ``CountGranularity.WORKITEM``. A work-item is a single instance
       of computation executing on a single processor (think 'thread'), a
       collection of which may be grouped together into a work-group. Each
       work-group executes on a single compute unit with all work-items within
       the work-group sharing local memory. A sub-group is an
       implementation-dependent grouping of work-items within a work-group,
       analagous to an NVIDIA CUDA warp.

    """

    def __init__(self, mtype=None, dtype=None, lid_strides=None, gid_strides=None,
                 direction=None, variable=None, variable_tag=None,
                 count_granularity=None):

        if count_granularity not in CountGranularity.ALL+[None]:
            raise ValueError("Op.__init__: count_granularity '%s' is "
                    "not allowed. count_granularity options: %s"
                    % (count_granularity, CountGranularity.ALL+[None]))

        if dtype is None:
            Record.__init__(self, mtype=mtype, dtype=dtype, lid_strides=lid_strides,
                            gid_strides=gid_strides, direction=direction,
                            variable=variable, variable_tag=variable_tag,
                            count_granularity=count_granularity)
        else:
            from loopy.types import to_loopy_type
            Record.__init__(self, mtype=mtype, dtype=to_loopy_type(dtype),
                            lid_strides=lid_strides, gid_strides=gid_strides,
                            direction=direction, variable=variable,
                            variable_tag=variable_tag,
                            count_granularity=count_granularity)

    def __hash__(self):
        # Note that this means lid_strides and gid_strides must be sorted
        # in self.__repr__()
        return hash(repr(self))

    def __repr__(self):
        # Record.__repr__ overridden for consistent ordering and conciseness
        return "MemAccess(%s, %s, %s, %s, %s, %s, %s, %s)" % (
            self.mtype,
            self.dtype,
            None if self.lid_strides is None else dict(
                sorted(six.iteritems(self.lid_strides))),
            None if self.gid_strides is None else dict(
                sorted(six.iteritems(self.gid_strides))),
            self.direction,
            self.variable,
            self.variable_tag,
            self.count_granularity)

# }}}


# {{{ counter base

class CounterBase(CombineMapper):
    def __init__(self, knl):
        self.knl = knl
        from loopy.type_inference import TypeInferenceMapper
        self.type_inf = TypeInferenceMapper(knl)

    def combine(self, values):
        return sum(values)

    def map_constant(self, expr):
        return ToCountMap()

    def map_call(self, expr):
        return self.rec(expr.parameters)

    def map_sum(self, expr):
        if expr.children:
            return sum(self.rec(child) for child in expr.children)
        else:
            return ToCountMap()

    map_product = map_sum

    def map_comparison(self, expr):
        return self.rec(expr.left)+self.rec(expr.right)

    def map_if(self, expr):
        warn_with_kernel(self.knl, "summing_if_branches",
                         "%s counting sum of if-expression branches."
                         % type(self).__name__)
        return self.rec(expr.condition) + self.rec(expr.then) \
               + self.rec(expr.else_)

    def map_if_positive(self, expr):
        warn_with_kernel(self.knl, "summing_if_branches",
                         "%s counting sum of if-expression branches."
                         % type(self).__name__)
        return self.rec(expr.criterion) + self.rec(expr.then) \
               + self.rec(expr.else_)

    def map_common_subexpression(self, expr):
        raise RuntimeError("%s encountered %s--not supposed to happen"
                % (type(self).__name__, type(expr).__name__))

    map_substitution = map_common_subexpression
    map_derivative = map_common_subexpression
    map_slice = map_common_subexpression

    # preprocessing should have removed these
    def map_reduction(self, expr):
        raise RuntimeError("%s encountered %s--not supposed to happen"
                % (type(self).__name__, type(expr).__name__))

# }}}


# {{{ ExpressionOpCounter

class ExpressionOpCounter(CounterBase):
    def __init__(self, knl, count_within_subscripts=True):
        self.knl = knl
        self.count_within_subscripts = count_within_subscripts
        from loopy.type_inference import TypeInferenceMapper
        self.type_inf = TypeInferenceMapper(knl)

    def combine(self, values):
        return sum(values)

    def map_constant(self, expr):
        return ToCountMap()

    map_tagged_variable = map_constant
    map_variable = map_constant

    def map_call(self, expr):
        return ToCountMap(
                    {Op(dtype=self.type_inf(expr),
                        name='func:'+str(expr.function),
                        count_granularity=CountGranularity.SUBGROUP): 1}
                    ) + self.rec(expr.parameters)

    def map_subscript(self, expr):
        if self.count_within_subscripts:
            return self.rec(expr.index)
        else:
            return ToCountMap()

    def map_sum(self, expr):
        assert expr.children
        return ToCountMap(
                    {Op(dtype=self.type_inf(expr),
                        name='add',
                        count_granularity=CountGranularity.SUBGROUP):
                     len(expr.children)-1}
                    ) + sum(self.rec(child) for child in expr.children)

    def map_product(self, expr):
        from pymbolic.primitives import is_zero
        assert expr.children
        return sum(ToCountMap({Op(dtype=self.type_inf(expr),
                                  name='mul',
                                  count_granularity=CountGranularity.SUBGROUP): 1})
                   + self.rec(child)
                   for child in expr.children
                   if not is_zero(child + 1)) + \
                   ToCountMap({Op(dtype=self.type_inf(expr),
                                  name='mul',
                                  count_granularity=CountGranularity.SUBGROUP): -1})

    def map_quotient(self, expr, *args):
        return ToCountMap({Op(dtype=self.type_inf(expr),
                              name='div',
                              count_granularity=CountGranularity.SUBGROUP): 1}) \
                                + self.rec(expr.numerator) \
                                + self.rec(expr.denominator)

    map_floor_div = map_quotient
    map_remainder = map_quotient

    def map_power(self, expr):
        return ToCountMap({Op(dtype=self.type_inf(expr),
                              name='pow',
                              count_granularity=CountGranularity.SUBGROUP): 1}) \
                                + self.rec(expr.base) \
                                + self.rec(expr.exponent)

    def map_left_shift(self, expr):
        return ToCountMap({Op(dtype=self.type_inf(expr),
                              name='shift',
                              count_granularity=CountGranularity.SUBGROUP): 1}) \
                                + self.rec(expr.shiftee) \
                                + self.rec(expr.shift)

    map_right_shift = map_left_shift

    def map_bitwise_not(self, expr):
        return ToCountMap({Op(dtype=self.type_inf(expr),
                              name='bw',
                              count_granularity=CountGranularity.SUBGROUP): 1}) \
                                + self.rec(expr.child)

    def map_bitwise_or(self, expr):
        return ToCountMap({Op(dtype=self.type_inf(expr),
                              name='bw',
                              count_granularity=CountGranularity.SUBGROUP):
                           len(expr.children)-1}) \
                                + sum(self.rec(child) for child in expr.children)

    map_bitwise_xor = map_bitwise_or
    map_bitwise_and = map_bitwise_or

    def map_if(self, expr):
        warn_with_kernel(self.knl, "summing_if_branches_ops",
                         "ExpressionOpCounter counting ops as sum of "
                         "if-statement branches.")
        return self.rec(expr.condition) + self.rec(expr.then) \
               + self.rec(expr.else_)

    def map_if_positive(self, expr):
        warn_with_kernel(self.knl, "summing_ifpos_branches_ops",
                         "ExpressionOpCounter counting ops as sum of "
                         "if_pos-statement branches.")
        return self.rec(expr.criterion) + self.rec(expr.then) \
               + self.rec(expr.else_)

    def map_min(self, expr):
        return ToCountMap({Op(dtype=self.type_inf(expr),
                              name='maxmin',
                              count_granularity=CountGranularity.SUBGROUP):
                           len(expr.children)-1}) \
               + sum(self.rec(child) for child in expr.children)

    map_max = map_min

    def map_common_subexpression(self, expr):
        raise NotImplementedError("ExpressionOpCounter encountered "
                                  "common_subexpression, "
                                  "map_common_subexpression not implemented.")

    def map_substitution(self, expr):
        raise NotImplementedError("ExpressionOpCounter encountered "
                                  "substitution, "
                                  "map_substitution not implemented.")

    def map_derivative(self, expr):
        raise NotImplementedError("ExpressionOpCounter encountered "
                                  "derivative, "
                                  "map_derivative not implemented.")

    def map_slice(self, expr):
        raise NotImplementedError("ExpressionOpCounter encountered slice, "
                                  "map_slice not implemented.")

# }}}


# {{{ modified coefficient collector that ignores denominator of floor div

class _IndexStrideCoefficientCollector(CoefficientCollector):

    def map_floor_div(self, expr):
        from warnings import warn
        warn("_IndexStrideCoefficientCollector encountered FloorDiv, ignoring "
             "denominator in expression %s" % (expr))
        return self.rec(expr.numerator)

# }}}


def _get_lid_and_gid_strides(knl, array, index):
    # find all local and global index tags and corresponding inames
    from loopy.symbolic import get_dependencies
    my_inames = get_dependencies(index) & knl.all_inames()

    from loopy.kernel.data import (LocalIndexTag, GroupIndexTag,
                                   filter_iname_tags_by_type)
    lid_to_iname = {}
    gid_to_iname = {}
    for iname in my_inames:
        tags = knl.iname_tags_of_type(iname, (GroupIndexTag, LocalIndexTag))
        if tags:
            tag, = filter_iname_tags_by_type(
                tags, (GroupIndexTag, LocalIndexTag), 1)
            if isinstance(tag, LocalIndexTag):
                lid_to_iname[tag.axis] = iname
            else:
                gid_to_iname[tag.axis] = iname

    # create lid_strides and gid_strides dicts

    # strides are coefficents in flattened index, i.e., we want
    # lid_strides = {0:l0, 1:l1, 2:l2, ...} and
    # gid_strides = {0:g0, 1:g1, 2:g2, ...},
    # where l0, l1, l2, g0, g1, and g2 come from flattened index
    # [... + g2*gid2 + g1*gid1 + g0*gid0 + ... + l2*lid2 + l1*lid1 + l0*lid0]

    from loopy.kernel.array import FixedStrideArrayDimTag
    from pymbolic.primitives import Variable
    from loopy.symbolic import simplify_using_aff
    from loopy.diagnostic import ExpressionNotAffineError

    def get_iname_strides(tag_to_iname_dict):
        tag_to_stride_dict = {}

        if array.dim_tags is None:
            assert len(index) <= 1
            dim_tags = (None,) * len(index)
        else:
            dim_tags = array.dim_tags

        for tag, iname in six.iteritems(tag_to_iname_dict):
            total_iname_stride = 0
            # find total stride of this iname for each axis
            for idx, axis_tag in zip(index, dim_tags):
                # collect index coefficients
                try:
                    coeffs = _IndexStrideCoefficientCollector(
                            [tag_to_iname_dict[tag]])(
                                    simplify_using_aff(knl, idx))
                except ExpressionNotAffineError:
                    total_iname_stride = None
                    break

                # check if idx contains this iname
                try:
                    coeff = coeffs[Variable(tag_to_iname_dict[tag])]
                except KeyError:
                    # idx does not contain this iname
                    continue

                # found coefficient of this iname
                # now determine stride
                if isinstance(axis_tag, FixedStrideArrayDimTag):
                    axis_tag_stride = axis_tag.stride

                    if axis_tag_stride is lp.auto:
                        total_iname_stride = None
                        break

                elif axis_tag is None:
                    axis_tag_stride = 1

                else:
                    continue

                total_iname_stride += axis_tag_stride*coeff

            tag_to_stride_dict[tag] = total_iname_stride

        return tag_to_stride_dict

    return get_iname_strides(lid_to_iname), get_iname_strides(gid_to_iname)


class MemAccessCounter(CounterBase):
    pass


# {{{ LocalMemAccessCounter

class LocalMemAccessCounter(MemAccessCounter):
    def count_var_access(self, dtype, name, index):
        sub_map = ToCountMap()
        if name in self.knl.temporary_variables:
            array = self.knl.temporary_variables[name]
            if isinstance(array, TemporaryVariable) and (
                    array.address_space == AddressSpace.LOCAL):
                if index is None:
                    # no subscript
                    sub_map[MemAccess(
                                mtype='local',
                                dtype=dtype,
                                count_granularity=CountGranularity.SUBGROUP)
                            ] = 1
                    return sub_map

                array = self.knl.temporary_variables[name]

                # could be tuple or scalar index
                index_tuple = index
                if not isinstance(index_tuple, tuple):
                    index_tuple = (index_tuple,)

                lid_strides, gid_strides = _get_lid_and_gid_strides(
                                                self.knl, array, index_tuple)

                sub_map[MemAccess(
                        mtype='local',
                        dtype=dtype,
                        lid_strides=dict(sorted(six.iteritems(lid_strides))),
                        gid_strides=dict(sorted(six.iteritems(gid_strides))),
                        variable=name,
                        count_granularity=CountGranularity.SUBGROUP)] = 1

        return sub_map

    def map_variable(self, expr):
        return self.count_var_access(
                    self.type_inf(expr), expr.name, None)

    map_tagged_variable = map_variable

    def map_subscript(self, expr):
        return (self.count_var_access(self.type_inf(expr),
                                      expr.aggregate.name,
                                      expr.index)
                + self.rec(expr.index))

# }}}


# {{{ GlobalMemAccessCounter

class GlobalMemAccessCounter(MemAccessCounter):
    def map_variable(self, expr):
        name = expr.name

        if name in self.knl.arg_dict:
            array = self.knl.arg_dict[name]
        else:
            # this is a temporary variable
            return ToCountMap()

        if not isinstance(array, lp.ArrayArg):
            # this array is not in global memory
            return ToCountMap()

        return ToCountMap({MemAccess(mtype='global',
                                     dtype=self.type_inf(expr), lid_strides={},
                                     gid_strides={}, variable=name,
                                     count_granularity=CountGranularity.WORKITEM): 1}
                          ) + self.rec(expr.index)

    def map_subscript(self, expr):
        name = expr.aggregate.name
        try:
            var_tag = expr.aggregate.tag
        except AttributeError:
            var_tag = None

        if name in self.knl.arg_dict:
            array = self.knl.arg_dict[name]
        else:
            # this is a temporary variable
            return self.rec(expr.index)

        if not isinstance(array, lp.ArrayArg):
            # this array is not in global memory
            return self.rec(expr.index)

        index_tuple = expr.index  # could be tuple or scalar index
        if not isinstance(index_tuple, tuple):
            index_tuple = (index_tuple,)

        lid_strides, gid_strides = _get_lid_and_gid_strides(
                                        self.knl, array, index_tuple)

        count_granularity = CountGranularity.WORKITEM if (
                                0 in lid_strides and lid_strides[0] != 0
                                ) else CountGranularity.SUBGROUP

        return ToCountMap({MemAccess(
                            mtype='global',
                            dtype=self.type_inf(expr),
                            lid_strides=dict(sorted(six.iteritems(lid_strides))),
                            gid_strides=dict(sorted(six.iteritems(gid_strides))),
                            variable=name,
                            variable_tag=var_tag,
                            count_granularity=count_granularity
                            ): 1}
                          ) + self.rec(expr.index_tuple)

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

def add_assumptions_guard(kernel, pwqpolynomial):
    return GuardedPwQPolynomial(pwqpolynomial, kernel.assumptions)


def count(kernel, set, space=None):
    try:
        if space is not None:
            set = set.align_params(space)

        return add_assumptions_guard(kernel, set.card())
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

            length_pwaff = dmax - dmin + stride
            if space is not None:
                length_pwaff = length_pwaff.align_params(space)

            length = isl.PwQPolynomial.from_pw_aff(length_pwaff)
            length = length.scale_down_val(stride)

            if bset_count is None:
                bset_count = length
            else:
                bset_count = bset_count * length

            # {{{ rebuild check domain

            zero = isl.Aff.zero_on_domain(
                        isl.LocalSpace.from_space(bset.space))
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
                warn_with_kernel(kernel, "count_overestimate",
                        "Barvinok wrappers are not installed. "
                        "Counting routines have overestimated the "
                        "number of integer points in your loop "
                        "domain.")
            elif is_superset:
                warn_with_kernel(kernel, "count_underestimate",
                        "Barvinok wrappers are not installed. "
                        "Counting routines have underestimated the "
                        "number of integer points in your loop "
                        "domain.")
            else:
                warn_with_kernel(kernel, "count_misestimate",
                        "Barvinok wrappers are not installed. "
                        "Counting routines have misestimated the "
                        "number of integer points in your loop "
                        "domain.")

    return add_assumptions_guard(kernel, count)


def get_unused_hw_axes_factor(knl, insn, disregard_local_axes, space=None):
    # FIXME: Multi-kernel support
    gsize, lsize = knl.get_grid_size_upper_bounds()

    g_used = set()
    l_used = set()

    from loopy.kernel.data import LocalIndexTag, GroupIndexTag
    for iname in knl.insn_inames(insn):
        tags = knl.iname_tags_of_type(iname,
                              (LocalIndexTag, GroupIndexTag), max_num=1)
        if tags:
            tag, = tags
            if isinstance(tag, LocalIndexTag):
                l_used.add(tag.axis)
            elif isinstance(tag, GroupIndexTag):
                g_used.add(tag.axis)

    def mult_grid_factor(used_axes, size):
        result = 1
        for iaxis, size in enumerate(size):
            if iaxis not in used_axes:
                if not isinstance(size, int):
                    if space is not None:
                        size = size.align_params(space)

                    size = isl.PwQPolynomial.from_pw_aff(size)

                result = result * size

        return result

    if disregard_local_axes:
        result = mult_grid_factor(g_used, gsize)
    else:
        result = mult_grid_factor(g_used, gsize) * mult_grid_factor(l_used, lsize)

    return add_assumptions_guard(knl, result)


def count_insn_runs(knl, insn, count_redundant_work, disregard_local_axes=False):

    insn_inames = knl.insn_inames(insn)

    if disregard_local_axes:
        from loopy.kernel.data import LocalIndexTag
        insn_inames = [iname
                for iname in insn_inames
                if not knl.iname_tags_of_type(iname, LocalIndexTag)]

    inames_domain = knl.get_inames_domain(insn_inames)
    domain = (inames_domain.project_out_except(
                            insn_inames, [dim_type.set]))

    space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT,
            set=[], params=knl.outer_params())

    c = count(knl, domain, space=space)

    if count_redundant_work:
        unused_fac = get_unused_hw_axes_factor(knl, insn,
                        disregard_local_axes=disregard_local_axes,
                        space=space)
        return c * unused_fac
    else:
        return c


@memoize_method
def _get_insn_count(knl, insn_id, subgroup_size, count_redundant_work,
                    count_granularity=CountGranularity.WORKITEM):
    insn = knl.id_to_insn[insn_id]

    if count_granularity is None:
        warn_with_kernel(knl, "get_insn_count_assumes_granularity",
                         "get_insn_count: No count granularity passed, "
                         "assuming %s granularity."
                         % (CountGranularity.WORKITEM))
        count_granularity == CountGranularity.WORKITEM

    if count_granularity == CountGranularity.WORKITEM:
        return count_insn_runs(
            knl, insn, count_redundant_work=count_redundant_work,
            disregard_local_axes=False)

    ct_disregard_local = count_insn_runs(
            knl, insn, disregard_local_axes=True,
            count_redundant_work=count_redundant_work)

    if count_granularity == CountGranularity.WORKGROUP:
        return ct_disregard_local
    elif count_granularity == CountGranularity.SUBGROUP:
        # get the group size
        from loopy.symbolic import aff_to_expr
        _, local_size = knl.get_grid_size_upper_bounds()
        workgroup_size = 1
        if local_size:
            for size in local_size:
                s = aff_to_expr(size)
                if not isinstance(s, int):
                    raise LoopyError("Cannot count insn with %s granularity, "
                                     "work-group size is not integer: %s"
                                     % (CountGranularity.SUBGROUP, local_size))
                workgroup_size *= s

        warn_with_kernel(knl, "insn_count_subgroups_upper_bound",
                "get_insn_count: when counting instruction %s with "
                "count_granularity=%s, using upper bound for work-group size "
                "(%d work-items) to compute sub-groups per work-group. When "
                "multiple device programs present, actual sub-group count may be"
                "lower." % (insn_id, CountGranularity.SUBGROUP, workgroup_size))

        from pytools import div_ceil
        return ct_disregard_local*div_ceil(workgroup_size, subgroup_size)
    else:
        # this should not happen since this is enforced in Op/MemAccess
        raise ValueError("get_insn_count: count_granularity '%s' is"
                "not allowed. count_granularity options: %s"
                % (count_granularity, CountGranularity.ALL+[None]))

# }}}


# {{{ get_op_map

def get_op_map(knl, numpy_types=True, count_redundant_work=False,
               count_within_subscripts=True, subgroup_size=None):

    """Count the number of operations in a loopy kernel.

    :arg knl: A :class:`loopy.LoopKernel` whose operations are to be counted.

    :arg numpy_types: A :class:`bool` specifying whether the types in the
        returned mapping should be numpy types instead of
        :class:`loopy.types.LoopyType`.

    :arg count_redundant_work: Based on usage of hardware axes or other
        specifics, a kernel may perform work redundantly. This :class:`bool`
        flag indicates whether this work should be included in the count.
        (Likely desirable for performance modeling, but undesirable for code
        optimization.)

    :arg count_within_subscripts: A :class:`bool` specifying whether to
        count operations inside array indices.

    :arg subgroup_size: (currently unused) An :class:`int`, :class:`str`
        ``'guess'``, or *None* that specifies the sub-group size. An OpenCL
        sub-group is an implementation-dependent grouping of work-items within
        a work-group, analagous to an NVIDIA CUDA warp. subgroup_size is used,
        e.g., when counting a :class:`MemAccess` whose count_granularity
        specifies that it should only be counted once per sub-group. If set to
        *None* an attempt to find the sub-group size using the device will be
        made, if this fails an error will be raised. If a :class:`str`
        ``'guess'`` is passed as the subgroup_size, get_mem_access_map will
        attempt to find the sub-group size using the device and, if
        unsuccessful, will make a wild guess.

    :return: A :class:`ToCountMap` of **{** :class:`Op` **:**
        :class:`islpy.PwQPolynomial` **}**.

        - The :class:`Op` specifies the characteristics of the arithmetic
          operation.

        - The :class:`islpy.PwQPolynomial` holds the number of operations of
          the kind specified in the key (in terms of the
          :class:`loopy.LoopKernel` parameter *inames*).

    Example usage::

        # (first create loopy kernel and specify array data types)

        op_map = get_op_map(knl)
        params = {'n': 512, 'm': 256, 'l': 128}
        f32add = op_map[Op(np.float32,
                           'add',
                           count_granularity=CountGranularity.WORKITEM)
                       ].eval_with_dict(params)
        f32mul = op_map[Op(np.float32,
                           'mul',
                           count_granularity=CountGranularity.WORKITEM)
                       ].eval_with_dict(params)

        # (now use these counts to, e.g., predict performance)

    """

    subgroup_size = _process_subgroup_size(knl, subgroup_size)

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    knl = infer_unknown_types(knl, expect_completion=True)
    knl = preprocess_kernel(knl)

    op_map = ToCountMap()
    op_counter = ExpressionOpCounter(knl, count_within_subscripts)

    from loopy.kernel.instruction import (
            CallInstruction, CInstruction, Assignment,
            NoOpInstruction, BarrierInstruction)

    for insn in knl.instructions:
        if isinstance(insn, (CallInstruction, CInstruction, Assignment)):
            ops = op_counter(insn.assignee) + op_counter(insn.expression)
            for key, val in six.iteritems(ops.count_map):
                op_map = (
                        op_map
                        + ToCountMap({key: val})
                        * _get_insn_count(knl, insn.id, subgroup_size,
                                         count_redundant_work,
                                         key.count_granularity))

        elif isinstance(insn, (NoOpInstruction, BarrierInstruction)):
            pass
        else:
            raise NotImplementedError("unexpected instruction item type: '%s'"
                    % type(insn).__name__)

    if numpy_types:
        return ToCountMap(
                    init_dict=dict(
                        (Op(
                            dtype=op.dtype.numpy_dtype,
                            name=op.name,
                            count_granularity=op.count_granularity),
                        ct)
                        for op, ct in six.iteritems(op_map.count_map)),
                    val_type=op_map.val_type
                    )
    else:
        return op_map

# }}}


def _find_subgroup_size_for_knl(knl):
    from loopy.target.pyopencl import PyOpenCLTarget
    if isinstance(knl.target, PyOpenCLTarget) and knl.target.device is not None:
        from pyopencl.characterize import get_simd_group_size
        subgroup_size_guess = get_simd_group_size(knl.target.device, None)
        warn_with_kernel(knl, "getting_subgroup_size_from_device",
                         "Device: %s. Using sub-group size given by "
                         "pyopencl.characterize.get_simd_group_size(): %d"
                         % (knl.target.device, subgroup_size_guess))
        return subgroup_size_guess
    else:
        return None


@memoize_method
def _process_subgroup_size(knl, subgroup_size_requested):

    if isinstance(subgroup_size_requested, int):
        return subgroup_size_requested
    else:
        # try to find subgroup_size
        subgroup_size_guess = _find_subgroup_size_for_knl(knl)

        if subgroup_size_requested is None:
            if subgroup_size_guess is None:
                # 'guess' was not passed and either no target device found
                # or get_simd_group_size returned None
                raise ValueError("No sub-group size passed, no target device found. "
                                 "Either (1) pass integer value for subgroup_size, "
                                 "(2) ensure that kernel.target is PyOpenClTarget "
                                 "and kernel.target.device is set, or (3) pass "
                                 "subgroup_size='guess' and hope for the best.")
            else:
                return subgroup_size_guess

        elif subgroup_size_requested == 'guess':
            if subgroup_size_guess is None:
                # unable to get subgroup_size from device, so guess
                subgroup_size_guess = 32
                warn_with_kernel(knl, "get_x_map_guessing_subgroup_size",
                                 "'guess' sub-group size passed, no target device "
                                 "found, wildly guessing that sub-group size is %d."
                                 % (subgroup_size_guess))
                return subgroup_size_guess
            else:
                return subgroup_size_guess
        else:
            raise ValueError("Invalid value for subgroup_size: %s. subgroup_size "
                             "must be integer, 'guess', or, if you're feeling "
                             "lucky, None." % (subgroup_size_requested))


# {{{ get_mem_access_map

def get_mem_access_map(knl, numpy_types=True, count_redundant_work=False,
                       subgroup_size=None):
    """Count the number of memory accesses in a loopy kernel.

    :arg knl: A :class:`loopy.LoopKernel` whose memory accesses are to be
        counted.

    :arg numpy_types: A :class:`bool` specifying whether the types in the
        returned mapping should be numpy types instead of
        :class:`loopy.types.LoopyType`.

    :arg count_redundant_work: Based on usage of hardware axes or other
        specifics, a kernel may perform work redundantly. This :class:`bool`
        flag indicates whether this work should be included in the count.
        (Likely desirable for performance modeling, but undesirable for
        code optimization.)

    :arg subgroup_size: An :class:`int`, :class:`str` ``'guess'``, or
        *None* that specifies the sub-group size. An OpenCL sub-group is an
        implementation-dependent grouping of work-items within a work-group,
        analagous to an NVIDIA CUDA warp. subgroup_size is used, e.g., when
        counting a :class:`MemAccess` whose count_granularity specifies that it
        should only be counted once per sub-group. If set to *None* an attempt
        to find the sub-group size using the device will be made, if this fails
        an error will be raised. If a :class:`str` ``'guess'`` is passed as
        the subgroup_size, get_mem_access_map will attempt to find the
        sub-group size using the device and, if unsuccessful, will make a wild
        guess.

    :return: A :class:`ToCountMap` of **{** :class:`MemAccess` **:**
        :class:`islpy.PwQPolynomial` **}**.

        - The :class:`MemAccess` specifies the characteristics of the memory
          access.

        - The :class:`islpy.PwQPolynomial` holds the number of memory accesses
          with the characteristics specified in the key (in terms of the
          :class:`loopy.LoopKernel` *inames*).

    Example usage::

        # (first create loopy kernel and specify array data types)

        params = {'n': 512, 'm': 256, 'l': 128}
        mem_map = get_mem_access_map(knl)

        f32_s1_g_ld_a = mem_map[MemAccess(
                                    mtype='global',
                                    dtype=np.float32,
                                    lid_strides={0: 1},
                                    gid_strides={0: 256},
                                    direction='load',
                                    variable='a',
                                    count_granularity=CountGranularity.WORKITEM)
                               ].eval_with_dict(params)
        f32_s1_g_st_a = mem_map[MemAccess(
                                    mtype='global',
                                    dtype=np.float32,
                                    lid_strides={0: 1},
                                    gid_strides={0: 256},
                                    direction='store',
                                    variable='a',
                                    count_granularity=CountGranularity.WORKITEM)
                               ].eval_with_dict(params)
        f32_s1_l_ld_x = mem_map[MemAccess(
                                    mtype='local',
                                    dtype=np.float32,
                                    lid_strides={0: 1},
                                    gid_strides={0: 256},
                                    direction='load',
                                    variable='x',
                                    count_granularity=CountGranularity.WORKITEM)
                               ].eval_with_dict(params)
        f32_s1_l_st_x = mem_map[MemAccess(
                                    mtype='local',
                                    dtype=np.float32,
                                    lid_strides={0: 1},
                                    gid_strides={0: 256},
                                    direction='store',
                                    variable='x',
                                    count_granularity=CountGranularity.WORKITEM)
                               ].eval_with_dict(params)

        # (now use these counts to, e.g., predict performance)

    """

    subgroup_size = _process_subgroup_size(knl, subgroup_size)

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    knl = infer_unknown_types(knl, expect_completion=True)
    knl = preprocess_kernel(knl)

    access_map = ToCountMap()
    access_counter_g = GlobalMemAccessCounter(knl)
    access_counter_l = LocalMemAccessCounter(knl)

    from loopy.kernel.instruction import (
            CallInstruction, CInstruction, Assignment,
            NoOpInstruction, BarrierInstruction)

    for insn in knl.instructions:
        if isinstance(insn, (CallInstruction, CInstruction, Assignment)):
            access_expr = (
                    access_counter_g(insn.expression)
                    + access_counter_l(insn.expression)
                    ).with_set_attributes(direction="load")

            access_assignee = (
                    access_counter_g(insn.assignee)
                    + access_counter_l(insn.assignee)
                    ).with_set_attributes(direction="store")

            for key, val in six.iteritems(access_expr.count_map):

                access_map = (
                        access_map
                        + ToCountMap({key: val})
                        * _get_insn_count(knl, insn.id, subgroup_size,
                                          count_redundant_work,
                                          key.count_granularity))

            for key, val in six.iteritems(access_assignee.count_map):

                access_map = (
                        access_map
                        + ToCountMap({key: val})
                        * _get_insn_count(knl, insn.id, subgroup_size,
                                          count_redundant_work,
                                          key.count_granularity))

        elif isinstance(insn, (NoOpInstruction, BarrierInstruction)):
            pass
        else:
            raise NotImplementedError("unexpected instruction item type: '%s'"
                    % type(insn).__name__)

    if numpy_types:
        return ToCountMap(
                    init_dict=dict(
                        (MemAccess(
                            mtype=mem_access.mtype,
                            dtype=mem_access.dtype.numpy_dtype,
                            lid_strides=mem_access.lid_strides,
                            gid_strides=mem_access.gid_strides,
                            direction=mem_access.direction,
                            variable=mem_access.variable,
                            variable_tag=mem_access.variable_tag,
                            count_granularity=mem_access.count_granularity),
                        ct)
                        for mem_access, ct in six.iteritems(access_map.count_map)),
                    val_type=access_map.val_type
                    )
    else:
        return access_map

# }}}


# {{{ get_synchronization_map

def get_synchronization_map(knl, subgroup_size=None):

    """Count the number of synchronization events each work-item encounters in
    a loopy kernel.

    :arg knl: A :class:`loopy.LoopKernel` whose barriers are to be counted.

    :arg subgroup_size: (currently unused) An :class:`int`, :class:`str`
        ``'guess'``, or *None* that specifies the sub-group size. An OpenCL
        sub-group is an implementation-dependent grouping of work-items within
        a work-group, analagous to an NVIDIA CUDA warp. subgroup_size is used,
        e.g., when counting a :class:`MemAccess` whose count_granularity
        specifies that it should only be counted once per sub-group. If set to
        *None* an attempt to find the sub-group size using the device will be
        made, if this fails an error will be raised. If a :class:`str`
        ``'guess'`` is passed as the subgroup_size, get_mem_access_map will
        attempt to find the sub-group size using the device and, if
        unsuccessful, will make a wild guess.

    :return: A dictionary mapping each type of synchronization event to an
        :class:`islpy.PwQPolynomial` holding the number of events per
        work-item.

        Possible keys include ``barrier_local``, ``barrier_global``
        (if supported by the target) and ``kernel_launch``.

    Example usage::

        # (first create loopy kernel and specify array data types)

        sync_map = get_synchronization_map(knl)
        params = {'n': 512, 'm': 256, 'l': 128}
        barrier_ct = sync_map['barrier_local'].eval_with_dict(params)

        # (now use this count to, e.g., predict performance)

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
            result = result + ToCountMap({"barrier_%s" %
                                          sched_item.synchronization_kind:
                                          get_count_poly(iname_list)})

        elif isinstance(sched_item, CallKernel):
            result = result + ToCountMap(
                    {"kernel_launch": get_count_poly(iname_list)})

        elif isinstance(sched_item, (ReturnFromKernel, RunInstruction)):
            pass

        else:
            raise LoopyError("unexpected schedule item: %s"
                    % type(sched_item).__name__)

    return result

# }}}


# {{{ gather_access_footprints

def gather_access_footprints(kernel, ignore_uncountable=False):
    """Return a dictionary mapping ``(var_name, direction)`` to
    :class:`islpy.Set` instances capturing which indices of each the array
    *var_name* are read/written (where *direction* is either ``read`` or
    ``write``.

    :arg ignore_uncountable: If *False*, an error will be raised for accesses
        on which the footprint cannot be determined (e.g. data-dependent or
        nonlinear indices)
    """

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    kernel = infer_unknown_types(kernel, expect_completion=True)

    from loopy.kernel import KernelState
    if kernel.state < KernelState.PREPROCESSED:
        kernel = preprocess_kernel(kernel)

    write_footprints = []
    read_footprints = []

    for insn in kernel.instructions:
        if not isinstance(insn, MultiAssignmentBase):
            warn_with_kernel(kernel, "count_non_assignment",
                    "Non-assignment instruction encountered in "
                    "gather_access_footprints, not counted")
            continue

        insn_inames = kernel.insn_inames(insn)
        inames_domain = kernel.get_inames_domain(insn_inames)
        domain = (inames_domain.project_out_except(insn_inames,
                                                   [dim_type.set]))

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

    :arg ignore_uncountable: If *True*, an error will be raised for accesses on
        which the footprint cannot be determined (e.g. data-dependent or
        nonlinear indices)
    """

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    kernel = infer_unknown_types(kernel, expect_completion=True)

    from loopy.kernel import KernelState
    if kernel.state < KernelState.PREPROCESSED:
        kernel = preprocess_kernel(kernel)

    result = {}
    fp = gather_access_footprints(kernel,
                                  ignore_uncountable=ignore_uncountable)

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

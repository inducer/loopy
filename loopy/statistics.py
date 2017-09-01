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
from pytools import memoize_in
from pymbolic.mapper import CombineMapper
from functools import reduce
from loopy.kernel.data import MultiAssignmentBase
from loopy.diagnostic import warn_with_kernel, LoopyError


__doc__ = """

.. currentmodule:: loopy

.. autoclass:: ToCountMap
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

        :arg kwargs: Keyword arguments matching fields in the keys of
                 the :class:`ToCountMap`, each given a list of
                 allowable values for that key field.

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

            # (now use these counts to predict performance)

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

        :arg func: A function that takes a map key a parameter and
             returns a :class:`bool`.

        :arg: A :class:`ToCountMap` containing the subset of the items in
                 the original :class:`ToCountMap` for which func(key) is true.

        Example usage::

            # (first create loopy kernel and specify array data types)

            params = {'n': 512, 'm': 256, 'l': 128}
            mem_map = lp.get_mem_access_map(knl)
            def filter_func(key):
                return key.stride > 1 and key.stride <= 4:

            filtered_map = mem_map.filter_by_func(filter_func)
            tot = filtered_map.eval_and_sum(params)

            # (now use these counts to predict performance)

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

        :return: A :class:`ToCountMap` containing the same total counts
                 grouped together by new keys that only contain the fields
                 specified in the arguments passed.

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

            # (now use these counts to predict performance)

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

        :return: A :class:`ToCountMap` mapping each original key to a
                 :class:`islpy.PwQPolynomial` with counts in bytes rather than
                 instances.

        Example usage::

            # (first create loopy kernel and specify array data types)

            bytes_map = get_mem_access_map(knl).to_bytes()
            params = {'n': 512, 'm': 256, 'l': 128}

            s1_g_ld_byt = bytes_map.filter_by(
                                mtype=['global'], stride=[1],
                                direction=['load']).eval_and_sum(params)
            s2_g_ld_byt = bytes_map.filter_by(
                                mtype=['global'], stride=[2],
                                direction=['load']).eval_and_sum(params)
            s1_g_st_byt = bytes_map.filter_by(
                                mtype=['global'], stride=[1],
                                direction=['store']).eval_and_sum(params)
            s2_g_st_byt = bytes_map.filter_by(
                                mtype=['global'], stride=[2],
                                direction=['store']).eval_and_sum(params)

            # (now use these counts to predict performance)

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

        :return: A :class:`islpy.PwQPolynomial` or :class:`int` containing the sum of
                 counts.

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
                                             variable=['a','g'])
            tot_loads_a_g = filtered_map.eval_and_sum(params)

            # (now use these counts to predict performance)

        """
        return self.sum().eval_with_dict(params)

# }}}


def stringify_stats_mapping(m):
    result = ""
    for key in sorted(m.keys(), key=lambda k: str(k)):
        result += ("%s : %s\n" % (key, m[key]))
    return result


# {{{ Op descriptor

class Op(object):
    """A descriptor for a type of arithmetic operation.

    .. attribute:: dtype

       A :class:`loopy.LoopyType` or :class:`numpy.dtype` that specifies the
       data type operated on.

    .. attribute:: name

       A :class:`str` that specifies the kind of arithmetic operation as
       *add*, *sub*, *mul*, *div*, *pow*, *shift*, *bw* (bitwise), etc.

    """

    # FIXME: This could be done much more briefly by inheriting from Record.

    def __init__(self, dtype=None, name=None):
        self.name = name
        if dtype is None:
            self.dtype = dtype
        else:
            from loopy.types import to_loopy_type
            self.dtype = to_loopy_type(dtype)

    def __eq__(self, other):
        return isinstance(other, Op) and (
                (self.dtype is None or other.dtype is None or
                 self.dtype == other.dtype) and
                (self.name is None or other.name is None or
                 self.name == other.name))

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return "Op(%s, %s)" % (self.dtype, self.name)

# }}}


# {{{ MemAccess descriptor

class MemAccess(object):
    """A descriptor for a type of memory access.

    .. attribute:: mtype

       A :class:`str` that specifies the memory type accessed as **global**
       or **local**

    .. attribute:: dtype

       A :class:`loopy.LoopyType` or :class:`numpy.dtype` that specifies the
       data type accessed.

    .. attribute:: stride

       An :class:`int` that specifies stride of the memory access. A stride of 0
       indicates a uniform access (i.e. all threads access the same item).

    .. attribute:: direction

       A :class:`str` that specifies the direction of memory access as
       **load** or **store**.

    .. attribute:: variable

       A :class:`str` that specifies the variable name of the data
       accessed.

    """

    # FIXME: This could be done much more briefly by inheriting from Record.

    def __init__(self, mtype=None, dtype=None, stride=None, direction=None,
                 variable=None):
        self.mtype = mtype
        self.stride = stride
        self.direction = direction
        self.variable = variable
        if dtype is None:
            self.dtype = dtype
        else:
            from loopy.types import to_loopy_type
            self.dtype = to_loopy_type(dtype)

        #TODO currently giving all lmem access stride=None
        if (mtype == 'local') and (stride is not None):
            raise NotImplementedError("MemAccess: stride must be None when "
                                      "mtype is 'local'")

        #TODO currently giving all lmem access variable=None
        if (mtype == 'local') and (variable is not None):
            raise NotImplementedError("MemAccess: variable must be None when "
                                      "mtype is 'local'")

    def copy(self, mtype=None, dtype=None, stride=None, direction=None,
            variable=None):
        return MemAccess(
                mtype=mtype if mtype is not None else self.mtype,
                dtype=dtype if dtype is not None else self.dtype,
                stride=stride if stride is not None else self.stride,
                direction=direction if direction is not None else self.direction,
                variable=variable if variable is not None else self.variable,
                )

    def __eq__(self, other):
        return isinstance(other, MemAccess) and (
                (self.mtype is None or other.mtype is None or
                 self.mtype == other.mtype) and
                (self.dtype is None or other.dtype is None or
                 self.dtype == other.dtype) and
                (self.stride is None or other.stride is None or
                 self.stride == other.stride) and
                (self.direction is None or other.direction is None or
                 self.direction == other.direction) and
                (self.variable is None or other.variable is None or
                 self.variable == other.variable))

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        if self.mtype is None:
            mtype = 'None'
        else:
            mtype = self.mtype
        if self.dtype is None:
            dtype = 'None'
        else:
            dtype = str(self.dtype)
        if self.stride is None:
            stride = 'None'
        else:
            stride = str(self.stride)
        if self.direction is None:
            direction = 'None'
        else:
            direction = self.direction
        if self.variable is None:
            variable = 'None'
        else:
            variable = self.variable
        return "MemAccess(" + mtype + ", " + dtype + ", " + stride + ", " \
               + direction + ", " + variable + ")"

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
    def __init__(self, knl):
        self.knl = knl
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
                        name='func:'+str(expr.function)): 1}
                    ) + self.rec(expr.parameters)

    def map_subscript(self, expr):
        return self.rec(expr.index)

    def map_sum(self, expr):
        assert expr.children
        return ToCountMap(
                    {Op(dtype=self.type_inf(expr),
                        name='add'): len(expr.children)-1}
                    ) + sum(self.rec(child) for child in expr.children)

    def map_product(self, expr):
        from pymbolic.primitives import is_zero
        assert expr.children
        return sum(ToCountMap({Op(dtype=self.type_inf(expr), name='mul'): 1})
                   + self.rec(child)
                   for child in expr.children
                   if not is_zero(child + 1)) + \
                   ToCountMap({Op(dtype=self.type_inf(expr), name='mul'): -1})

    def map_quotient(self, expr, *args):
        return ToCountMap({Op(dtype=self.type_inf(expr), name='div'): 1}) \
                                + self.rec(expr.numerator) \
                                + self.rec(expr.denominator)

    map_floor_div = map_quotient
    map_remainder = map_quotient

    def map_power(self, expr):
        return ToCountMap({Op(dtype=self.type_inf(expr), name='pow'): 1}) \
                                + self.rec(expr.base) \
                                + self.rec(expr.exponent)

    def map_left_shift(self, expr):
        return ToCountMap({Op(dtype=self.type_inf(expr), name='shift'): 1}) \
                                + self.rec(expr.shiftee) \
                                + self.rec(expr.shift)

    map_right_shift = map_left_shift

    def map_bitwise_not(self, expr):
        return ToCountMap({Op(dtype=self.type_inf(expr), name='bw'): 1}) \
                                + self.rec(expr.child)

    def map_bitwise_or(self, expr):
        return ToCountMap({Op(dtype=self.type_inf(expr), name='bw'):
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
        return ToCountMap({Op(dtype=self.type_inf(expr), name='maxmin'):
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


class MemAccessCounter(CounterBase):
    pass


# {{{ LocalMemAccessCounter

class LocalMemAccessCounter(MemAccessCounter):
    def count_var_access(self, dtype, name, subscript):
        sub_map = ToCountMap()
        if name in self.knl.temporary_variables:
            array = self.knl.temporary_variables[name]
            if array.is_local:
                sub_map[MemAccess(mtype='local', dtype=dtype)] = 1
        return sub_map

    def map_variable(self, expr):
        return self.count_var_access(
                    self.type_inf(expr), expr.name, None)

    map_tagged_variable = map_variable

    def map_subscript(self, expr):
        return (
                self.count_var_access(
                    self.type_inf(expr), expr.aggregate.name, expr.index)
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

        if not isinstance(array, lp.GlobalArg):
            # this array is not in global memory
            return ToCountMap()

        return ToCountMap({MemAccess(mtype='global',
                                     dtype=self.type_inf(expr), stride=0,
                                     variable=name): 1}
                          ) + self.rec(expr.index)

    def map_subscript(self, expr):
        name = expr.aggregate.name

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

        # find min tag axis
        import sys
        min_tag_axis = sys.maxsize
        local_id_found = False
        for iname in my_inames:
            tag = self.knl.iname_to_tag.get(iname)
            if isinstance(tag, LocalIndexTag):
                local_id_found = True
                if tag.axis < min_tag_axis:
                    min_tag_axis = tag.axis

        if not local_id_found:
            # count as uniform access
            return ToCountMap({MemAccess(mtype='global',
                                         dtype=self.type_inf(expr), stride=0,
                                         variable=name): 1}
                              ) + self.rec(expr.index)

        if min_tag_axis != 0:
            warn_with_kernel(self.knl, "unknown_gmem_stride",
                             "GlobalSubscriptCounter: Memory access minimum "
                             "tag axis %d != 0, stride unknown, using "
                             "sys.maxsize." % (min_tag_axis))
            return ToCountMap({MemAccess(mtype='global',
                                         dtype=self.type_inf(expr),
                                         stride=sys.maxsize, variable=name): 1}
                              ) + self.rec(expr.index)

        # get local_id associated with minimum tag axis
        min_lid = None
        for iname in my_inames:
            tag = self.knl.iname_to_tag.get(iname)
            if isinstance(tag, LocalIndexTag):
                if tag.axis == min_tag_axis:
                    min_lid = iname
                    break  # there will be only one min local_id

        # found local_id associated with minimum tag axis

        total_stride = 0
        # check coefficient of min_lid for each axis
        from loopy.symbolic import CoefficientCollector
        from loopy.kernel.array import FixedStrideArrayDimTag
        from pymbolic.primitives import Variable
        for idx, axis_tag in zip(index, array.dim_tags):

            from loopy.symbolic import simplify_using_aff
            coeffs = CoefficientCollector()(simplify_using_aff(self.knl, idx))
            # check if he contains the lid 0 guy
            try:
                coeff_min_lid = coeffs[Variable(min_lid)]
            except KeyError:
                # does not contain min_lid
                continue
            # found coefficient of min_lid
            # now determine stride
            if isinstance(axis_tag, FixedStrideArrayDimTag):
                stride = axis_tag.stride
            else:
                continue

            total_stride += stride*coeff_min_lid

        return ToCountMap({MemAccess(mtype='global', dtype=self.type_inf(expr),
                                     stride=total_stride, variable=name): 1}
                          ) + self.rec(expr.index)

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
        tag = knl.iname_to_tag.get(iname)

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
        insn_inames = [iname for iname in insn_inames if not
                       isinstance(knl.iname_to_tag.get(iname), LocalIndexTag)]

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

# }}}


# {{{ get_op_map

def get_op_map(knl, numpy_types=True, count_redundant_work=False):

    """Count the number of operations in a loopy kernel.

    :arg knl: A :class:`loopy.LoopKernel` whose operations are to be counted.

    :arg numpy_types: A :class:`bool` specifying whether the types
         in the returned mapping should be numpy types
         instead of :class:`loopy.LoopyType`.

    :arg count_redundant_work: Based on usage of hardware axes or other
        specifics, a kernel may perform work redundantly. This :class:`bool`
        flag indicates whether this work should be included in the count.
        (Likely desirable for performance modeling, but undesirable for
        code optimization.)

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
        f32add = op_map[Op(np.float32, 'add')].eval_with_dict(params)
        f32mul = op_map[Op(np.float32, 'mul')].eval_with_dict(params)

        # (now use these counts to predict performance)

    """

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    knl = infer_unknown_types(knl, expect_completion=True)
    knl = preprocess_kernel(knl)

    op_map = ToCountMap()
    op_counter = ExpressionOpCounter(knl)
    for insn in knl.instructions:
        ops = op_counter(insn.assignee) + op_counter(insn.expression)
        op_map = op_map + ops*count_insn_runs(
                knl, insn,
                count_redundant_work=count_redundant_work)

    if numpy_types:
        op_map.count_map = dict((Op(dtype=op.dtype.numpy_dtype, name=op.name),
                                 count)
                for op, count in six.iteritems(op_map.count_map))

    return op_map

# }}}


# {{{ get_mem_access_map

def get_mem_access_map(knl, numpy_types=True, count_redundant_work=False):
    """Count the number of memory accesses in a loopy kernel.

    :arg knl: A :class:`loopy.LoopKernel` whose memory accesses are to be
        counted.

    :arg numpy_types: A :class:`bool` specifying whether the types
        in the returned mapping should be numpy types
        instead of :class:`loopy.LoopyType`.

    :arg count_redundant_work: Based on usage of hardware axes or other
        specifics, a kernel may perform work redundantly. This :class:`bool`
        flag indicates whether this work should be included in the count.
        (Likely desirable for performance modeling, but undesirable for
        code optimization.)

    :return: A :class:`ToCountMap` of **{** :class:`MemAccess` **:**
        :class:`islpy.PwQPolynomial` **}**.

        - The :class:`MemAccess` specifies the characteristics of the
          memory access.

        - The :class:`islpy.PwQPolynomial` holds the number of memory
          accesses with the characteristics specified in the key (in terms
          of the :class:`loopy.LoopKernel` *inames*).

    Example usage::

        # (first create loopy kernel and specify array data types)

        params = {'n': 512, 'm': 256, 'l': 128}
        mem_map = get_mem_access_map(knl)

        f32_s1_g_ld_a = mem_map[MemAccess(mtype='global',
                                          dtype=np.float32,
                                          stride=1,
                                          direction='load',
                                          variable='a')
                               ].eval_with_dict(params)
        f32_s1_g_st_a = mem_map[MemAccess(mtype='global',
                                          dtype=np.float32,
                                          stride=1,
                                          direction='store',
                                          variable='a')
                               ].eval_with_dict(params)
        f32_s1_l_ld_x = mem_map[MemAccess(mtype='local',
                                          dtype=np.float32,
                                          stride=1,
                                          direction='load',
                                          variable='x')
                               ].eval_with_dict(params)
        f32_s1_l_st_x = mem_map[MemAccess(mtype='local',
                                          dtype=np.float32,
                                          stride=1,
                                          direction='store',
                                          variable='x')
                               ].eval_with_dict(params)

        # (now use these counts to predict performance)

    """
    from loopy.preprocess import preprocess_kernel, infer_unknown_types

    class CacheHolder(object):
        pass

    cache_holder = CacheHolder()

    @memoize_in(cache_holder, "insn_count")
    def get_insn_count(knl, insn_id, uniform=False):
        insn = knl.id_to_insn[insn_id]
        return count_insn_runs(
                knl, insn, disregard_local_axes=uniform,
                count_redundant_work=count_redundant_work)

    knl = infer_unknown_types(knl, expect_completion=True)
    knl = preprocess_kernel(knl)

    access_map = ToCountMap()
    access_counter_g = GlobalMemAccessCounter(knl)
    access_counter_l = LocalMemAccessCounter(knl)

    for insn in knl.instructions:
        access_expr = (
                access_counter_g(insn.expression)
                + access_counter_l(insn.expression)
                ).with_set_attributes(direction="load")

        access_assignee_g = access_counter_g(insn.assignee).with_set_attributes(
                direction="store")

        # FIXME: (!!!!) for now, don't count writes to local mem

        # use count excluding local index tags for uniform accesses
        for key, val in six.iteritems(access_expr.count_map):
            is_uniform = (key.mtype == 'global' and
                    isinstance(key.stride, int) and
                    key.stride == 0)
            access_map = (
                    access_map
                    + ToCountMap({key: val})
                    * get_insn_count(knl, insn.id, is_uniform))
            #currently not counting stride of local mem access

        for key, val in six.iteritems(access_assignee_g.count_map):
            is_uniform = (key.mtype == 'global' and
                    isinstance(key.stride, int) and
                    key.stride == 0)
            access_map = (
                    access_map
                    + ToCountMap({key: val})
                    * get_insn_count(knl, insn.id, is_uniform))
            # for now, don't count writes to local mem

    if numpy_types:
        # FIXME: Don't modify in-place
        access_map.count_map = dict((MemAccess(mtype=mem_access.mtype,
                                             dtype=mem_access.dtype.numpy_dtype,
                                             stride=mem_access.stride,
                                             direction=mem_access.direction,
                                             variable=mem_access.variable),
                                  count)
                      for mem_access, count in six.iteritems(access_map.count_map))

    return access_map

# }}}


# {{{ get_synchronization_map

def get_synchronization_map(knl):

    """Count the number of synchronization events each thread encounters in a
    loopy kernel.

    :arg knl: A :class:`loopy.LoopKernel` whose barriers are to be counted.

    :return: A dictionary mapping each type of synchronization event to a
            :class:`islpy.PwQPolynomial` holding the number of events per
            thread.

            Possible keys include ``barrier_local``, ``barrier_global``
            (if supported by the target) and ``kernel_launch``.

    Example usage::

        # (first create loopy kernel and specify array data types)

        sync_map = get_synchronization_map(knl)
        params = {'n': 512, 'm': 256, 'l': 128}
        barrier_ct = sync_map['barrier_local'].eval_with_dict(params)

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
            result = result + ToCountMap({"barrier_%s" % sched_item.kind:
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
    """Return a dictionary mapping ``(var_name, direction)``
    to :class:`islpy.Set` instances capturing which indices
    of each the array *var_name* are read/written (where
    *direction* is either ``read`` or ``write``.

    :arg ignore_uncountable: If *False*, an error will be raised for
        accesses on which the footprint cannot be determined (e.g.
        data-dependent or nonlinear indices)
    """

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    kernel = infer_unknown_types(kernel, expect_completion=True)

    from loopy.kernel import kernel_state
    if kernel.state < kernel_state.PREPROCESSED:
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

    :arg ignore_uncountable: If *True*, an error will be raised for
        accesses on which the footprint cannot be determined (e.g.
        data-dependent or nonlinear indices)
    """

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    kernel = infer_unknown_types(kernel, expect_completion=True)

    from loopy.kernel import kernel_state
    if kernel.state < kernel_state.PREPROCESSED:
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


# {{{ compat goop

def get_lmem_access_poly(knl):
    """Count the number of local memory accesses in a loopy kernel.

    get_lmem_access_poly is deprecated. Use get_mem_access_map and filter the
    result with the mtype=['local'] option.

    """
    warn_with_kernel(knl, "deprecated_get_lmem_access_poly",
                     "get_lmem_access_poly is deprecated. Use "
                     "get_mem_access_map and filter the result with the "
                     "mtype=['local'] option.")
    return get_mem_access_map(knl).filter_by(mtype=['local'])


def get_DRAM_access_poly(knl):
    """Count the number of global memory accesses in a loopy kernel.

    get_DRAM_access_poly is deprecated. Use get_mem_access_map and filter the
    result with the mtype=['global'] option.

    """
    warn_with_kernel(knl, "deprecated_get_DRAM_access_poly",
                     "get_DRAM_access_poly is deprecated. Use "
                     "get_mem_access_map and filter the result with the "
                     "mtype=['global'] option.")
    return get_mem_access_map(knl).filter_by(mtype=['global'])


def get_gmem_access_poly(knl):
    """Count the number of global memory accesses in a loopy kernel.

    get_DRAM_access_poly is deprecated. Use get_mem_access_map and filter the
    result with the mtype=['global'] option.

    """
    warn_with_kernel(knl, "deprecated_get_gmem_access_poly",
                     "get_DRAM_access_poly is deprecated. Use "
                     "get_mem_access_map and filter the result with the "
                     "mtype=['global'] option.")
    return get_mem_access_map(knl).filter_by(mtype=['global'])


def get_synchronization_poly(knl):
    """Count the number of synchronization events each thread encounters in a
    loopy kernel.

    get_synchronization_poly is deprecated. Use get_synchronization_map instead.

    """
    warn_with_kernel(knl, "deprecated_get_synchronization_poly",
                     "get_synchronization_poly is deprecated. Use "
                     "get_synchronization_map instead.")
    return get_synchronization_map(knl)


def get_op_poly(knl, numpy_types=True):
    """Count the number of operations in a loopy kernel.

    get_op_poly is deprecated. Use get_op_map instead.

    """
    warn_with_kernel(knl, "deprecated_get_op_poly",
                     "get_op_poly is deprecated. Use get_op_map instead.")
    return get_op_map(knl, numpy_types)

# }}}

# vim: foldmethod=marker

__copyright__ = """
Copyright (C) 2015 James Stevens
Copyright (C) 2018 Kaushik Kulkarni
Copyright (C) 2019 Andreas Kloeckner
"""


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

from functools import partial, cached_property

from islpy import dim_type
import islpy as isl
from pymbolic.mapper import CombineMapper

import loopy as lp
from loopy.kernel.data import (
        MultiAssignmentBase, TemporaryVariable, AddressSpace)
from loopy.diagnostic import warn_with_kernel, LoopyError
from loopy.symbolic import CoefficientCollector
from pytools import ImmutableRecord, memoize_method
from loopy.kernel.function_interface import CallableKernel
from loopy.translation_unit import TranslationUnit


__doc__ = """

.. currentmodule:: loopy

.. autoclass:: ToCountMap
.. autoclass:: ToCountPolynomialMap
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


# FIXME:
# - The SUBGROUP granularity is completely broken if the root kernel
#   contains the grid and the operations get counted in the callee.
#   To test, most of those are set to WORKITEM instead below (marked
#   with FIXMEs). This leads to value mismatches and key errors in
#   the tests.
# - Currently, nothing prevents summation across different
#   granularities, which is guaranteed to yield bogus results.
# - AccessFootprintGatherer needs to be redone to match get_op_map and
#   get_mem_access_map style
# - Test for the subkernel functionality need to be written


def get_kernel_parameter_space(kernel):
    return isl.Space.create_from_names(kernel.isl_context,
            set=[], params=sorted(kernel.outer_params())).params()


def get_kernel_zero_pwqpolynomial(kernel):
    space = get_kernel_parameter_space(kernel)
    space = space.insert_dims(dim_type.out, 0, 1)
    return isl.PwQPolynomial.zero(space)


# {{{ GuardedPwQPolynomial

def _get_param_tuple(obj):
    return tuple(
            obj.get_dim_name(dim_type.param, i)
            for i in range(obj.dim(dim_type.param)))


class GuardedPwQPolynomial:
    def __init__(self, pwqpolynomial, valid_domain):
        assert isinstance(pwqpolynomial, isl.PwQPolynomial)
        self.pwqpolynomial = pwqpolynomial
        self.valid_domain = valid_domain

        assert (_get_param_tuple(pwqpolynomial.space)
                == _get_param_tuple(valid_domain.space))

    @property
    def space(self):
        return self.valid_domain.space

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
        p = isl.PwQPolynomial("{ 0 }")
        return GuardedPwQPolynomial(p, isl.Set.universe(p.domain().space))

    def __str__(self):
        return str(self.pwqpolynomial)

    def __repr__(self):
        return "Guarded" + repr(self.pwqpolynomial)

# }}}


# {{{ ToCountMap

class ToCountMap:
    """A map from work descriptors like :class:`Op` and :class:`MemAccess`
    to any arithmetic type.

    .. automethod:: __getitem__
    .. automethod:: __str__
    .. automethod:: __repr__
    .. automethod:: __len__
    .. automethod:: get
    .. automethod:: items
    .. automethod:: keys
    .. automethod:: values

    .. automethod:: copy
    .. automethod:: with_set_attributes

    .. automethod:: filter_by
    .. automethod:: filter_by_func
    .. automethod:: group_by
    .. automethod:: to_bytes
    .. automethod:: sum

    """

    def __init__(self, count_map=None):
        if count_map is None:
            count_map = {}

        self.count_map = count_map

    def _zero(self):
        return 0

    def __add__(self, other):
        result = self.count_map.copy()
        for k, v in other.count_map.items():
            result[k] = self.count_map.get(k, 0) + v
        return self.copy(count_map=result)

    def __radd__(self, other):
        if other != 0:
            raise ValueError("ToCountMap: Attempted to add ToCountMap "
                                "to {} {}. ToCountMap may only be added to "
                                "0 and other ToCountMap objects."
                                .format(type(other), other))

        return self

    def __mul__(self, other):
        if isinstance(other, GuardedPwQPolynomial):
            return self.copy({
                index: other*value
                for index, value in self.count_map.items()})
        else:
            raise ValueError("ToCountMap: Attempted to multiply "
                                "ToCountMap by {} {}."
                                .format(type(other), other))

    __rmul__ = __mul__

    def __getitem__(self, index):
        return self.count_map[index]

    def __repr__(self):
        return repr(self.count_map)

    def __str__(self):
        return "\n".join(
                f"{k}: {v}"
                for k, v in sorted(self.count_map.items(),
                    key=lambda k: str(k)))

    def __len__(self):
        return len(self.count_map)

    def get(self, key, default=None):
        return self.count_map.get(key, default)

    def items(self):
        return self.count_map.items()

    def keys(self):
        return self.count_map.keys()

    def values(self):
        return self.count_map.values()

    def copy(self, count_map=None):
        if count_map is None:
            count_map = self.count_map

        return type(self)(count_map=count_map)

    def with_set_attributes(self, **kwargs):
        return self.copy(count_map={
            key.copy(**kwargs): val
            for key, val in self.count_map.items()})

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

            params = {"n": 512, "m": 256, "l": 128}
            mem_map = lp.get_mem_access_map(knl)
            filtered_map = mem_map.filter_by(direction=["load"],
                                             variable=["a","g"])
            tot_loads_a_g = filtered_map.eval_and_sum(params)

            # (now use these counts to, e.g., predict performance)

        """

        new_count_map = {}

        class _Sentinel:
            pass

        new_kwargs = {}
        for arg_field, allowable_vals in kwargs.items():
            if arg_field == "dtype":
                from loopy.types import to_loopy_type
                allowable_vals = [to_loopy_type(dtype) for dtype in allowable_vals]

            new_kwargs[arg_field] = allowable_vals

        for key, val in self.count_map.items():
            if all(getattr(key, arg_field, _Sentinel) in allowable_vals
                    for arg_field, allowable_vals in new_kwargs.items()):
                new_count_map[key] = val

        return self.copy(count_map=new_count_map)

    def filter_by_func(self, func):
        """Keep items that pass a test.

        :arg func: A function that takes a map key a parameter and returns a
            :class:`bool`.

        :arg: A :class:`ToCountMap` containing the subset of the items in the
            original :class:`ToCountMap` for which func(key) is true.

        Example usage::

            # (first create loopy kernel and specify array data types)

            params = {"n": 512, "m": 256, "l": 128}
            mem_map = lp.get_mem_access_map(knl)
            def filter_func(key):
                return key.lid_strides[0] > 1 and key.lid_strides[0] <= 4:

            filtered_map = mem_map.filter_by_func(filter_func)
            tot = filtered_map.eval_and_sum(params)

            # (now use these counts to, e.g., predict performance)

        """

        new_count_map = {}

        for self_key, self_val in self.count_map.items():
            if func(self_key):
                new_count_map[self_key] = self_val

        return self.copy(count_map=new_count_map)

    def group_by(self, *args):
        """Group map items together, distinguishing by only the key fields
        passed in args.

        :arg args: Zero or more :class:`str` fields of map keys.

        :return: A :class:`ToCountMap` containing the same total counts grouped
            together by new keys that only contain the fields specified in the
            arguments passed.

        Example usage::

            # (first create loopy kernel and specify array data types)

            params = {"n": 512, "m": 256, "l": 128}
            mem_map = get_mem_access_map(knl)
            grouped_map = mem_map.group_by("mtype", "dtype", "direction")

            f32_global_ld = grouped_map[MemAccess(mtype="global",
                                                  dtype=np.float32,
                                                  direction="load")
                                       ].eval_with_dict(params)
            f32_global_st = grouped_map[MemAccess(mtype="global",
                                                  dtype=np.float32,
                                                  direction="store")
                                       ].eval_with_dict(params)
            f32_local_ld = grouped_map[MemAccess(mtype="local",
                                                 dtype=np.float32,
                                                 direction="load")
                                      ].eval_with_dict(params)
            f32_local_st = grouped_map[MemAccess(mtype="local",
                                                 dtype=np.float32,
                                                 direction="store")
                                      ].eval_with_dict(params)

            op_map = get_op_map(knl)
            ops_dtype = op_map.group_by("dtype")

            f32ops = ops_dtype[Op(dtype=np.float32)].eval_with_dict(params)
            f64ops = ops_dtype[Op(dtype=np.float64)].eval_with_dict(params)
            i32ops = ops_dtype[Op(dtype=np.int32)].eval_with_dict(params)

            # (now use these counts to, e.g., predict performance)

        """

        new_count_map = {}

        # make sure all item keys have same type
        if self.count_map:
            key_type = type(list(self.keys())[0])
            if not all(isinstance(x, key_type) for x in self.keys()):
                raise ValueError("ToCountMap: group_by() function may only "
                                 "be used on ToCountMaps with uniform keys")
        else:
            return self

        for self_key, self_val in self.count_map.items():
            new_key = key_type(
                    **{
                        field: getattr(self_key, field)
                        for field in args})

            new_count_map[new_key] = new_count_map.get(new_key, 0) + self_val

        return self.copy(count_map=new_count_map)

    def to_bytes(self):
        """Convert counts to bytes using data type in map key.

        :return: A :class:`ToCountMap` mapping each original key to an
            :class:`islpy.PwQPolynomial` with counts in bytes rather than
            instances.

        Example usage::

            # (first create loopy kernel and specify array data types)

            bytes_map = get_mem_access_map(knl).to_bytes()
            params = {"n": 512, "m": 256, "l": 128}

            s1_g_ld_byt = bytes_map.filter_by(
                                mtype=["global"], lid_strides={0: 1},
                                direction=["load"]).eval_and_sum(params)
            s2_g_ld_byt = bytes_map.filter_by(
                                mtype=["global"], lid_strides={0: 2},
                                direction=["load"]).eval_and_sum(params)
            s1_g_st_byt = bytes_map.filter_by(
                                mtype=["global"], lid_strides={0: 1},
                                direction=["store"]).eval_and_sum(params)
            s2_g_st_byt = bytes_map.filter_by(
                                mtype=["global"], lid_strides={0: 2},
                                direction=["store"]).eval_and_sum(params)

            # (now use these counts to, e.g., predict performance)

        """

        new_count_map = {}

        for key, val in self.count_map.items():
            new_count_map[key] = int(key.dtype.itemsize) * val

        return self.copy(new_count_map)

    def sum(self):
        """:return: A sum of the values of the dictionary."""

        total = self._zero()

        for v in self.count_map.values():
            total = v + total

        return total

# }}}


# {{{ ToCountPolynomialMap

class ToCountPolynomialMap(ToCountMap):
    """Maps any type of key to a :class:`islpy.PwQPolynomial` or a
    :class:`~loopy.statistics.GuardedPwQPolynomial`.

    .. automethod:: eval_and_sum
    """

    def __init__(self, space, count_map=None):
        if not isinstance(space, isl.Space):
            raise TypeError(
                    "first argument to ToCountPolynomialMap must be "
                    "of type islpy.Space")

        assert space.is_params()
        self.space = space

        space_param_tuple = _get_param_tuple(space)

        for val in count_map.values():
            if isinstance(val, isl.PwQPolynomial):
                assert val.dim(dim_type.out) == 1
            elif isinstance(val, GuardedPwQPolynomial):
                assert val.pwqpolynomial.dim(dim_type.out) == 1
            else:
                raise TypeError("unexpected value type")

            assert _get_param_tuple(val.space) == space_param_tuple

        super().__init__(count_map)

    def _zero(self):
        space = self.space.insert_dims(dim_type.out, 0, 1)
        return isl.PwQPolynomial.zero(space)

    def copy(self, count_map=None, space=None):
        if count_map is None:
            count_map = self.count_map

        if space is None:
            space = self.space

        return type(self)(space, count_map)

    def eval_and_sum(self, params=None):
        """Add all counts and evaluate with provided parameter dict *params*

        :return: An :class:`int` containing the sum of all counts
            evaluated with the parameters provided.

        Example usage::

            # (first create loopy kernel and specify array data types)

            params = {"n": 512, "m": 256, "l": 128}
            mem_map = lp.get_mem_access_map(knl)
            filtered_map = mem_map.filter_by(direction=["load"],
                                             variable=["a", "g"])
            tot_loads_a_g = filtered_map.eval_and_sum(params)

            # (now use these counts to, e.g., predict performance)

        """
        if params is None:
            params = {}

        return self.sum().eval_with_dict(params)

# }}}


# {{{ subst_into_to_count_map

def subst_into_guarded_pwqpolynomial(new_space, guarded_poly, subst_dict):
    from loopy.isl_helpers import subst_into_pwqpolynomial, get_param_subst_domain

    poly = subst_into_pwqpolynomial(
            new_space, guarded_poly.pwqpolynomial, subst_dict)

    valid_domain = guarded_poly.valid_domain
    i_begin_subst_space = valid_domain.dim(dim_type.param)

    valid_domain, subst_domain, _ = get_param_subst_domain(
            new_space, guarded_poly.valid_domain, subst_dict)

    valid_domain = valid_domain & subst_domain
    valid_domain = valid_domain.project_out(dim_type.param, 0, i_begin_subst_space)
    return GuardedPwQPolynomial(poly, valid_domain)


def subst_into_to_count_map(space, tcm, subst_dict):
    from loopy.isl_helpers import subst_into_pwqpolynomial
    new_count_map = {}
    for key, value in tcm.count_map.items():
        if isinstance(value, GuardedPwQPolynomial):
            new_count_map[key] = subst_into_guarded_pwqpolynomial(
                    space, value, subst_dict)

        elif isinstance(value, isl.PwQPolynomial):
            new_count_map[key] = subst_into_pwqpolynomial(space, value, subst_dict)

        elif isinstance(value, int):
            new_count_map[key] = value

        else:
            raise ValueError("unexpected value type")

    return tcm.copy(space=space, count_map=new_count_map)

# }}}


# {{{ CountGranularity

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

# }}}


# {{{ Op descriptor

class Op(ImmutableRecord):
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
       of computation executing on a single processor (think "thread"), a
       collection of which may be grouped together into a work-group. Each
       work-group executes on a single compute unit with all work-items within
       the work-group sharing local memory. A sub-group is an
       implementation-dependent grouping of work-items within a work-group,
       analagous to an NVIDIA CUDA warp.

    .. attribute:: kernel_name

        A :class:`str` representing the kernel name where the operation occurred.

    """

    def __init__(self, dtype=None, name=None, count_granularity=None,
            kernel_name=None):
        if count_granularity not in CountGranularity.ALL+[None]:
            raise ValueError("Op.__init__: count_granularity '%s' is "
                    "not allowed. count_granularity options: %s"
                    % (count_granularity, CountGranularity.ALL+[None]))

        if dtype is not None:
            from loopy.types import to_loopy_type
            dtype = to_loopy_type(dtype)

        super().__init__(dtype=dtype, name=name,
                        count_granularity=count_granularity,
                        kernel_name=kernel_name)

    def __repr__(self):
        # Record.__repr__ overridden for consistent ordering and conciseness
        if self.kernel_name is not None:
            return (f"Op({self.dtype}, {self.name}, {self.count_granularity},"
                    f' "{self.kernel_name}")')
        else:
            return f"Op({self.dtype}, {self.name}, {self.count_granularity})"

# }}}


# {{{ MemAccess descriptor

class MemAccess(ImmutableRecord):
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

    .. attribute:: variable_tags

       A :class:`frozenset` of subclasses of :class:`~pytools.tag.Tag`
       that reflects :attr:`~loopy.symbolic.TaggedVariable.tags` of
       an accessed variable.

    .. attribute:: count_granularity

       A :class:`str` that specifies whether this operation should be counted
       once per *work-item*, *sub-group*, or *work-group*. The granularities
       allowed can be found in :class:`CountGranularity`, and may be accessed,
       e.g., as ``CountGranularity.WORKITEM``. A work-item is a single instance
       of computation executing on a single processor (think "thread"), a
       collection of which may be grouped together into a work-group. Each
       work-group executes on a single compute unit with all work-items within
       the work-group sharing local memory. A sub-group is an
       implementation-dependent grouping of work-items within a work-group,
       analagous to an NVIDIA CUDA warp.

    .. attribute:: kernel_name

        A :class:`str` representing the kernel name where the operation occurred.
    """

    def __init__(self, mtype=None, dtype=None, lid_strides=None, gid_strides=None,
                 direction=None, variable=None,
                 *, variable_tags=None,
                 count_granularity=None, kernel_name=None):

        if count_granularity not in CountGranularity.ALL+[None]:
            raise ValueError("Op.__init__: count_granularity '%s' is "
                    "not allowed. count_granularity options: %s"
                    % (count_granularity, CountGranularity.ALL+[None]))

        if variable_tags is None:
            variable_tags = frozenset()

        if dtype is not None:
            from loopy.types import to_loopy_type
            dtype = to_loopy_type(dtype)

        ImmutableRecord.__init__(self, mtype=mtype, dtype=dtype,
                            lid_strides=lid_strides,
                            gid_strides=gid_strides, direction=direction,
                            variable=variable, variable_tags=variable_tags,
                            count_granularity=count_granularity,
                            kernel_name=kernel_name)

    def __hash__(self):
        # dicts in gid_strides and lid_strides aren't natively hashable
        return hash(repr(self))

    def __repr__(self):
        # Record.__repr__ overridden for consistent ordering and conciseness
        return "MemAccess({}, {}, {}, {}, {}, {}, {}, {}, {})".format(
            self.mtype,
            self.dtype,
            None if self.lid_strides is None else dict(
                sorted(self.lid_strides.items())),
            None if self.gid_strides is None else dict(
                sorted(self.gid_strides.items())),
            self.direction,
            self.variable,
            "None" if not self.variable_tags else str(self.variable_tags),
            self.count_granularity,
            repr(self.kernel_name))

# }}}


# {{{ Sync descriptor

class Sync(ImmutableRecord):
    """A descriptor for a type of synchronization.

    .. attribute:: kind

       A string describing the synchronization kind, e.g. ``"barrier_global"`` or
       ``"barrier_local"`` or ``"kernel_launch"``.

    .. attribute:: kernel_name

        A :class:`str` representing the kernel name where the operation occurred.
    """

    def __init__(self, kind=None, kernel_name=None):
        super().__init__(kind=kind, kernel_name=kernel_name)

    def __repr__(self):
        # Record.__repr__ overridden for consistent ordering and conciseness
        return f"Sync({self.kind}, {self.kernel_name})"

# }}}


# {{{ CounterBase

class CounterBase(CombineMapper):
    def __init__(self, knl, callables_table, kernel_rec):
        self.knl = knl
        self.callables_table = callables_table
        self.kernel_rec = kernel_rec

        from loopy.type_inference import TypeReader
        self.type_inf = TypeReader(knl, callables_table)
        self.zero = get_kernel_zero_pwqpolynomial(self.knl)
        self.one = self.zero + 1

    @cached_property
    def param_space(self):
        return get_kernel_parameter_space(self.knl)

    def new_poly_map(self, count_map):
        return ToCountPolynomialMap(self.param_space, count_map)

    def new_zero_poly_map(self):
        return self.new_poly_map({})

    def combine(self, values):
        return sum(values)

    def map_constant(self, expr):
        return self.new_zero_poly_map()

    def map_call(self, expr):
        from loopy.symbolic import ResolvedFunction
        assert isinstance(expr.function, ResolvedFunction)
        clbl = self.callables_table[expr.function.name]

        from loopy.kernel.function_interface import (CallableKernel,
                get_kw_pos_association)
        from loopy.kernel.data import ValueArg
        if isinstance(clbl, CallableKernel):
            sub_result = self.kernel_rec(clbl.subkernel)
            _, pos_to_kw = get_kw_pos_association(clbl.subkernel)

            subst_dict = {
                    pos_to_kw[i]: param
                    for i, param in enumerate(expr.parameters)
                    if isinstance(clbl.subkernel.arg_dict[pos_to_kw[i]],
                                  ValueArg)}

            return subst_into_to_count_map(
                    self.param_space,
                    sub_result, subst_dict) \
                    + self.rec(expr.parameters)

        else:
            raise NotImplementedError()

    def map_call_with_kwargs(self, expr):
        # See https://github.com/inducer/loopy/pull/323
        raise NotImplementedError

    def map_sum(self, expr):
        if expr.children:
            return sum(self.rec(child) for child in expr.children)
        else:
            return self.new_zero_poly_map()

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

    def map_reduction(self, expr):
        # preprocessing should have removed these
        raise RuntimeError("%s encountered %s--not supposed to happen"
                % (type(self).__name__, type(expr).__name__))

# }}}


# {{{ ExpressionOpCounter

class ExpressionOpCounter(CounterBase):
    def __init__(self, knl, callables_table, kernel_rec,
            count_within_subscripts=True):
        super().__init__(
                knl, callables_table, kernel_rec)
        self.count_within_subscripts = count_within_subscripts

    arithmetic_count_granularity = CountGranularity.SUBGROUP

    def combine(self, values):
        return sum(values)

    def map_constant(self, expr):
        return self.new_zero_poly_map()

    map_tagged_variable = map_constant
    map_variable = map_constant
    map_nan = map_constant

    def map_call(self, expr):
        from loopy.symbolic import ResolvedFunction
        assert isinstance(expr.function, ResolvedFunction)
        clbl = self.callables_table[expr.function.name]

        from loopy.kernel.function_interface import CallableKernel
        if not isinstance(clbl, CallableKernel):
            return self.new_poly_map(
                        {Op(dtype=self.type_inf(expr),
                            name="func:"+clbl.name,
                            count_granularity=self.arithmetic_count_granularity,
                            kernel_name=self.knl.name): self.one}
                        ) + self.rec(expr.parameters)
        else:
            return super().map_call(expr)

    def map_subscript(self, expr):
        if self.count_within_subscripts:
            return self.rec(expr.index)
        else:
            return self.new_zero_poly_map()

    def map_sub_array_ref(self, expr):
        # generates an array view, considered free
        return self.new_zero_poly_map()

    def map_sum(self, expr):
        assert expr.children
        return self.new_poly_map(
                    {Op(dtype=self.type_inf(expr),
                        name="add",
                        count_granularity=self.arithmetic_count_granularity,
                        kernel_name=self.knl.name):
                     self.zero + (len(expr.children)-1)}
                    ) + sum(self.rec(child) for child in expr.children)

    def map_product(self, expr):
        from pymbolic.primitives import is_zero
        assert expr.children
        return sum(self.new_poly_map({Op(dtype=self.type_inf(expr),
                                  name="mul",
                                  count_granularity=(
                                      self.arithmetic_count_granularity),
                                  kernel_name=self.knl.name): self.one})
                   + self.rec(child)
                   for child in expr.children
                   if not is_zero(child + 1)) + \
                   self.new_poly_map({Op(dtype=self.type_inf(expr),
                                  name="mul",
                                  count_granularity=(
                                      self.arithmetic_count_granularity),
                                  kernel_name=self.knl.name): -self.one})

    def map_quotient(self, expr, *args):
        return self.new_poly_map({Op(dtype=self.type_inf(expr),
                              name="div",
                              count_granularity=self.arithmetic_count_granularity,
                              kernel_name=self.knl.name): self.one}) \
                                + self.rec(expr.numerator) \
                                + self.rec(expr.denominator)

    map_floor_div = map_quotient
    map_remainder = map_quotient

    def map_power(self, expr):
        return self.new_poly_map({Op(dtype=self.type_inf(expr),
                              name="pow",
                              count_granularity=self.arithmetic_count_granularity,
                              kernel_name=self.knl.name): self.one}) \
                                + self.rec(expr.base) \
                                + self.rec(expr.exponent)

    def map_left_shift(self, expr):
        return self.new_poly_map({Op(dtype=self.type_inf(expr),
                              name="shift",
                              count_granularity=self.arithmetic_count_granularity,
                              kernel_name=self.knl.name): self.one}) \
                                + self.rec(expr.shiftee) \
                                + self.rec(expr.shift)

    map_right_shift = map_left_shift

    def map_bitwise_not(self, expr):
        return self.new_poly_map({Op(dtype=self.type_inf(expr),
                              name="bw",
                              count_granularity=self.arithmetic_count_granularity,
                              kernel_name=self.knl.name): self.one}) \
                                + self.rec(expr.child)

    def map_bitwise_or(self, expr):
        return self.new_poly_map({Op(dtype=self.type_inf(expr),
                              name="bw",
                              count_granularity=self.arithmetic_count_granularity,
                              kernel_name=self.knl.name):
                           self.zero + (len(expr.children)-1)}) \
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
        return self.new_poly_map({Op(dtype=self.type_inf(expr),
                              name="maxmin",
                              count_granularity=self.arithmetic_count_granularity,
                              kernel_name=self.knl.name):
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


# {{{ _get_lid_and_gid_strides

def _get_lid_and_gid_strides(knl, array, index):
    # find all local and global index tags and corresponding inames
    from loopy.symbolic import get_dependencies
    my_inames = get_dependencies(index) & knl.all_inames()

    from loopy.kernel.data import (LocalInameTag, GroupInameTag,
                                   filter_iname_tags_by_type)
    lid_to_iname = {}
    gid_to_iname = {}
    for iname in my_inames:
        tags = knl.iname_tags_of_type(iname, (GroupInameTag, LocalInameTag))
        if tags:
            tag, = filter_iname_tags_by_type(
                tags, (GroupInameTag, LocalInameTag), 1)
            if isinstance(tag, LocalInameTag):
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

        for tag in tag_to_iname_dict:
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

# }}}


# {{{ MemAccessCounterBase

class MemAccessCounterBase(CounterBase):
    def map_sub_array_ref(self, expr):
        # generates an array view, considered free
        return self.new_zero_poly_map()

    def map_call(self, expr):
        from loopy.symbolic import ResolvedFunction
        assert isinstance(expr.function, ResolvedFunction)
        clbl = self.callables_table[expr.function.name]

        from loopy.kernel.function_interface import CallableKernel
        if not isinstance(clbl, CallableKernel):
            return self.rec(expr.parameters)
        else:
            return super().map_call(expr)

# }}}


# {{{ LocalMemAccessCounter

class LocalMemAccessCounter(MemAccessCounterBase):
    local_mem_count_granularity = CountGranularity.SUBGROUP

    def count_var_access(self, dtype, name, index):
        count_map = {}
        if name in self.knl.temporary_variables:
            array = self.knl.temporary_variables[name]
            if isinstance(array, TemporaryVariable) and (
                    array.address_space == AddressSpace.LOCAL):
                if index is None:
                    # no subscript
                    count_map[MemAccess(
                                mtype="local",
                                dtype=dtype,
                                count_granularity=self.local_mem_count_granularity,
                                kernel_name=self.knl.name)] = self.one
                    return self.new_poly_map(count_map)

                array = self.knl.temporary_variables[name]

                # could be tuple or scalar index
                index_tuple = index
                if not isinstance(index_tuple, tuple):
                    index_tuple = (index_tuple,)

                lid_strides, gid_strides = _get_lid_and_gid_strides(
                                                self.knl, array, index_tuple)

                count_map[MemAccess(
                        mtype="local",
                        dtype=dtype,
                        lid_strides=dict(sorted(lid_strides.items())),
                        gid_strides=dict(sorted(gid_strides.items())),
                        variable=name,
                        count_granularity=self.local_mem_count_granularity,
                        kernel_name=self.knl.name)] = self.one

        return self.new_poly_map(count_map)

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

class GlobalMemAccessCounter(MemAccessCounterBase):
    def map_variable(self, expr):
        name = expr.name

        if name in self.knl.arg_dict:
            array = self.knl.arg_dict[name]
        else:
            # this is a temporary variable
            # FIXME temporary variable could have global address space
            return self.new_zero_poly_map()

        if not isinstance(array, lp.ArrayArg):
            # this array is not in global memory
            return self.new_zero_poly_map()

        return self.new_poly_map({MemAccess(mtype="global",
                    dtype=self.type_inf(expr), lid_strides={},
                    gid_strides={}, variable=name,
                    count_granularity=CountGranularity.WORKITEM,
                    kernel_name=self.knl.name): self.one}
                    ) + self.rec(expr.index)

    def map_subscript(self, expr):
        name = expr.aggregate.name
        try:
            var_tags = expr.aggregate.tags
        except AttributeError:
            var_tags = frozenset()

        is_global_temp = False
        if name in self.knl.arg_dict:
            array = self.knl.arg_dict[name]
        elif name in self.knl.temporary_variables:
            # This a temporary, but might have global address space
            from loopy.kernel.data import AddressSpace
            array = self.knl.temporary_variables[name]
            if array.address_space != AddressSpace.GLOBAL:
                # This temporary does not have global address space
                return self.rec(expr.index)
            # This temporary has global address space
            is_global_temp = True
        else:
            # This temporary does not have global address space
            return self.rec(expr.index)

        if (not is_global_temp) and not isinstance(array, lp.ArrayArg):
            # This array is not in global memory
            return self.rec(expr.index)

        index_tuple = expr.index  # could be tuple or scalar index
        if not isinstance(index_tuple, tuple):
            index_tuple = (index_tuple,)

        lid_strides, gid_strides = _get_lid_and_gid_strides(
                                        self.knl, array, index_tuple)

        global_access_count_granularity = CountGranularity.SUBGROUP

        # Account for broadcasts once per subgroup
        count_granularity = CountGranularity.WORKITEM if (
                # if the stride in lid.0 is known
                0 in lid_strides
                and
                # it is nonzero
                lid_strides[0] != 0
                ) else global_access_count_granularity

        return self.new_poly_map({MemAccess(
                            mtype="global",
                            dtype=self.type_inf(expr),
                            lid_strides=dict(sorted(lid_strides.items())),
                            gid_strides=dict(sorted(gid_strides.items())),
                            variable=name,
                            variable_tags=var_tags,
                            count_granularity=count_granularity,
                            kernel_name=self.knl.name,
                            ): self.one}
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

            for var_name, footprint in b.items():
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

        from loopy.symbolic import get_access_map

        try:
            access_range = get_access_map(self.domain, subscript,
                    self.kernel.assumptions).range()
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
    return GuardedPwQPolynomial(
            pwqpolynomial,
            kernel.assumptions.align_params(pwqpolynomial.space))


def count(kernel, set, space=None):
    if isinstance(kernel, TranslationUnit):
        kernel_names = [i for i, clbl in kernel.callables_table.items()
                if isinstance(clbl, CallableKernel)]
        if len(kernel_names) > 1:
            raise LoopyError()
        return count(kernel[kernel_names[0]], set, space)

    try:
        if space is not None:
            set = set.align_params(space)

        return add_assumptions_guard(kernel, set.card())
    except AttributeError:
        pass

    total_count = isl.PwQPolynomial.zero(
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
            total_count += bset_count

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

    return add_assumptions_guard(kernel, total_count)


def get_unused_hw_axes_factor(knl, callables_table, insn, disregard_local_axes):
    # FIXME: Multi-kernel support
    gsize, lsize = knl.get_grid_size_upper_bounds(callables_table)

    g_used = set()
    l_used = set()

    from loopy.kernel.data import LocalInameTag, GroupInameTag
    for iname in insn.within_inames:
        tags = knl.iname_tags_of_type(iname,
                              (LocalInameTag, GroupInameTag), max_num=1)
        if tags:
            tag, = tags
            if isinstance(tag, LocalInameTag):
                l_used.add(tag.axis)
            elif isinstance(tag, GroupInameTag):
                g_used.add(tag.axis)

    def mult_grid_factor(used_axes, sizes):
        result = get_kernel_zero_pwqpolynomial(knl) + 1

        for iaxis, size in enumerate(sizes):
            if iaxis not in used_axes:
                if isinstance(size, int):
                    result = result * size
                else:
                    result = result * isl.PwQPolynomial.from_pw_aff(
                            size.align_params(result.space))

        return result

    if disregard_local_axes:
        result = mult_grid_factor(g_used, gsize)
    else:
        result = mult_grid_factor(g_used, gsize) * mult_grid_factor(l_used, lsize)

    return add_assumptions_guard(knl, result)


def count_inames_domain(knl, inames):
    space = get_kernel_parameter_space(knl)
    if not inames:
        return add_assumptions_guard(knl,
                get_kernel_zero_pwqpolynomial(knl) + 1)

    inames_domain = knl.get_inames_domain(inames)
    domain = inames_domain.project_out_except(inames, [dim_type.set])
    return count(knl, domain, space=space)


def count_insn_runs(knl, callables_table, insn, count_redundant_work,
        disregard_local_axes=False):

    insn_inames = insn.within_inames

    if disregard_local_axes:
        from loopy.kernel.data import LocalInameTag
        insn_inames = frozenset(
                [iname for iname in insn_inames
                    if not knl.iname_tags_of_type(iname, LocalInameTag)])

    c = count_inames_domain(knl, insn_inames)

    if count_redundant_work:
        unused_fac = get_unused_hw_axes_factor(knl, callables_table,
                insn, disregard_local_axes=disregard_local_axes)
        return c * unused_fac
    else:
        return c


def _get_insn_count(knl, callables_table, insn_id, subgroup_size,
        count_redundant_work, count_granularity=CountGranularity.WORKITEM):
    insn = knl.id_to_insn[insn_id]

    if count_granularity is None:
        warn_with_kernel(knl, "get_insn_count_assumes_granularity",
                         "get_insn_count: No count granularity passed, "
                         "assuming %s granularity."
                         % (CountGranularity.WORKITEM))
        count_granularity = CountGranularity.WORKITEM

    if count_granularity == CountGranularity.WORKITEM:
        return count_insn_runs(
            knl, callables_table, insn,
            count_redundant_work=count_redundant_work,
            disregard_local_axes=False)

    ct_disregard_local = count_insn_runs(
            knl, callables_table, insn, disregard_local_axes=True,
            count_redundant_work=count_redundant_work)

    if count_granularity == CountGranularity.WORKGROUP:
        return ct_disregard_local
    elif count_granularity == CountGranularity.SUBGROUP:
        # {{{ compute workgroup_size

        from loopy.symbolic import aff_to_expr
        _, local_size = knl.get_grid_size_upper_bounds(callables_table)
        workgroup_size = 1
        if local_size:
            for size in local_size:
                if size.n_piece() != 1:
                    raise LoopyError("Workgroup size found to be genuinely "
                        "piecewise defined, which is not allowed in stats gathering")

                (valid_set, aff), = size.get_pieces()

                assert ((valid_set.n_basic_set() == 1)
                        and (valid_set.get_basic_sets()[0].is_universe()))

                s = aff_to_expr(aff)
                if not isinstance(s, int):
                    raise LoopyError("Cannot count insn with %s granularity, "
                                     "work-group size is not integer: %s"
                                     % (CountGranularity.SUBGROUP, local_size))
                workgroup_size *= s

        # }}}

        warn_with_kernel(knl, "insn_count_subgroups_upper_bound",
                "get_insn_count: when counting instruction %s with "
                "count_granularity=%s, using upper bound for work-group size "
                "(%d work-items) to compute sub-groups per work-group. When "
                "multiple device programs present, actual sub-group count may be "
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

def _get_op_map_for_single_kernel(knl, callables_table,
        count_redundant_work,
        count_within_subscripts, subgroup_size, within):

    subgroup_size = _process_subgroup_size(knl, subgroup_size)

    kernel_rec = partial(_get_op_map_for_single_kernel,
            callables_table=callables_table,
            count_redundant_work=count_redundant_work,
            count_within_subscripts=count_within_subscripts,
            subgroup_size=subgroup_size, within=within)

    op_counter = ExpressionOpCounter(knl, callables_table, kernel_rec,
            count_within_subscripts)
    op_map = op_counter.new_zero_poly_map()

    from loopy.kernel.instruction import (
            CallInstruction, CInstruction, Assignment,
            NoOpInstruction, BarrierInstruction)

    for insn in knl.instructions:
        if within(knl, insn):
            if isinstance(insn, (CallInstruction, Assignment)):
                ops = op_counter(insn.assignees) + op_counter(insn.expression)
                for key, val in ops.count_map.items():
                    count = _get_insn_count(knl, callables_table, insn.id,
                                subgroup_size, count_redundant_work,
                                key.count_granularity)
                    op_map = op_map + ToCountMap({key: val}) * count

            elif isinstance(
                    insn, (CInstruction, NoOpInstruction, BarrierInstruction)):
                pass
            else:
                raise NotImplementedError("unexpected instruction item type: '%s'"
                        % type(insn).__name__)

    return op_map


def get_op_map(program, count_redundant_work=False,
               count_within_subscripts=True, subgroup_size=None,
               entrypoint=None, within=None):

    """Count the number of operations in a loopy kernel.

    :arg knl: A :class:`loopy.LoopKernel` whose operations are to be counted.

    :arg count_redundant_work: Based on usage of hardware axes or other
        specifics, a kernel may perform work redundantly. This :class:`bool`
        flag indicates whether this work should be included in the count.
        (Likely desirable for performance modeling, but undesirable for code
        optimization.)

    :arg count_within_subscripts: A :class:`bool` specifying whether to
        count operations inside array indices.

    :arg subgroup_size: (currently unused) An :class:`int`, :class:`str`
        ``"guess"``, or *None* that specifies the sub-group size. An OpenCL
        sub-group is an implementation-dependent grouping of work-items within
        a work-group, analagous to an NVIDIA CUDA warp. subgroup_size is used,
        e.g., when counting a :class:`MemAccess` whose count_granularity
        specifies that it should only be counted once per sub-group. If set to
        *None* an attempt to find the sub-group size using the device will be
        made, if this fails an error will be raised. If a :class:`str`
        ``"guess"`` is passed as the subgroup_size, get_mem_access_map will
        attempt to find the sub-group size using the device and, if
        unsuccessful, will make a wild guess.

    :arg within: If not None, limit the result to matching contexts.
        See :func:`loopy.match.parse_match` for syntax.

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
        params = {"n": 512, "m": 256, "l": 128}
        f32add = op_map[Op(np.float32,
                           "add",
                           count_granularity=CountGranularity.WORKITEM)
                       ].eval_with_dict(params)
        f32mul = op_map[Op(np.float32,
                           "mul",
                           count_granularity=CountGranularity.WORKITEM)
                       ].eval_with_dict(params)

        # (now use these counts to, e.g., predict performance)

    """

    if entrypoint is None:
        if len(program.entrypoints) > 1:
            raise LoopyError("Must provide entrypoint")

        entrypoint = list(program.entrypoints)[0]

    assert entrypoint in program.entrypoints

    from loopy.preprocess import preprocess_program, infer_unknown_types
    program = preprocess_program(program)

    from loopy.match import parse_match
    within = parse_match(within)

    # Ordering restriction: preprocess might insert arguments to
    # make strides valid. Those also need to go through type inference.
    program = infer_unknown_types(program, expect_completion=True)

    return _get_op_map_for_single_kernel(
            program[entrypoint], program.callables_table,
            count_redundant_work=count_redundant_work,
            count_within_subscripts=count_within_subscripts,
            subgroup_size=subgroup_size,
            within=within)

# }}}


# {{{ subgoup size finding

def _find_subgroup_size_for_knl(knl):
    from loopy.target.pyopencl import PyOpenCLTarget
    if isinstance(knl.target, PyOpenCLTarget) and knl.target.device is not None:
        from pyopencl.characterize import get_simd_group_size
        subgroup_size_guess = get_simd_group_size(knl.target.device, None)
        warn_with_kernel(knl, "getting_subgroup_size_from_device",
                         "Device: %s. Using sub-group size given by "
                         "pyopencl.characterize.get_simd_group_size(): %s"
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
                # "guess" was not passed and either no target device found
                # or get_simd_group_size returned None
                raise ValueError("No sub-group size passed, no target device found. "
                                 "Either (1) pass integer value for subgroup_size, "
                                 "(2) ensure that kernel.target is PyOpenClTarget "
                                 "and kernel.target.device is set, or (3) pass "
                                 "subgroup_size='guess' and hope for the best.")
            else:
                return subgroup_size_guess

        elif subgroup_size_requested == "guess":
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

# }}}


# {{{ get_mem_access_map

def _get_mem_access_map_for_single_kernel(knl, callables_table,
        count_redundant_work, subgroup_size, within):

    subgroup_size = _process_subgroup_size(knl, subgroup_size)

    kernel_rec = partial(_get_mem_access_map_for_single_kernel,
            callables_table=callables_table,
            count_redundant_work=count_redundant_work,
            subgroup_size=subgroup_size)

    access_counter_g = GlobalMemAccessCounter(
            knl, callables_table, kernel_rec)
    access_counter_l = LocalMemAccessCounter(
            knl, callables_table, kernel_rec)
    access_map = access_counter_g.new_zero_poly_map()

    from loopy.kernel.instruction import (
            CallInstruction, CInstruction, Assignment,
            NoOpInstruction, BarrierInstruction)

    for insn in knl.instructions:
        if within(knl, insn):
            if isinstance(insn, (CallInstruction, Assignment)):
                insn_access_map = (
                            access_counter_g(insn.expression)
                            + access_counter_l(insn.expression)
                            ).with_set_attributes(direction="load")
                for assignee in insn.assignees:
                    insn_access_map = insn_access_map + (
                            access_counter_g(assignee)
                            + access_counter_l(assignee)
                            ).with_set_attributes(direction="store")

                for key, val in insn_access_map.count_map.items():
                    count = _get_insn_count(knl, callables_table, insn.id,
                                subgroup_size, count_redundant_work,
                                key.count_granularity)
                    access_map = access_map + ToCountMap({key: val}) * count

            elif isinstance(
                    insn, (CInstruction, NoOpInstruction, BarrierInstruction)):
                pass

            else:
                raise NotImplementedError("unexpected instruction item type: '%s'"
                        % type(insn).__name__)

    return access_map


def get_mem_access_map(program, count_redundant_work=False,
                       subgroup_size=None, entrypoint=None,
                       within=None):
    """Count the number of memory accesses in a loopy kernel.

    :arg knl: A :class:`loopy.LoopKernel` whose memory accesses are to be
        counted.

    :arg count_redundant_work: Based on usage of hardware axes or other
        specifics, a kernel may perform work redundantly. This :class:`bool`
        flag indicates whether this work should be included in the count.
        (Likely desirable for performance modeling, but undesirable for
        code optimization.)

    :arg subgroup_size: An :class:`int`, :class:`str` ``"guess"``, or
        *None* that specifies the sub-group size. An OpenCL sub-group is an
        implementation-dependent grouping of work-items within a work-group,
        analagous to an NVIDIA CUDA warp. subgroup_size is used, e.g., when
        counting a :class:`MemAccess` whose count_granularity specifies that it
        should only be counted once per sub-group. If set to *None* an attempt
        to find the sub-group size using the device will be made, if this fails
        an error will be raised. If a :class:`str` ``"guess"`` is passed as
        the subgroup_size, get_mem_access_map will attempt to find the
        sub-group size using the device and, if unsuccessful, will make a wild
        guess.

    :arg within: If not None, limit the result to matching contexts.
        See :func:`loopy.match.parse_match` for syntax.

    :return: A :class:`ToCountMap` of **{** :class:`MemAccess` **:**
        :class:`islpy.PwQPolynomial` **}**.

        - The :class:`MemAccess` specifies the characteristics of the memory
          access.

        - The :class:`islpy.PwQPolynomial` holds the number of memory accesses
          with the characteristics specified in the key (in terms of the
          :class:`loopy.LoopKernel` *inames*).

    Example usage::

        # (first create loopy kernel and specify array data types)

        params = {"n": 512, "m": 256, "l": 128}
        mem_map = get_mem_access_map(knl)

        f32_s1_g_ld_a = mem_map[MemAccess(
                                    mtype="global",
                                    dtype=np.float32,
                                    lid_strides={0: 1},
                                    gid_strides={0: 256},
                                    direction="load",
                                    variable="a",
                                    count_granularity=CountGranularity.WORKITEM)
                               ].eval_with_dict(params)
        f32_s1_g_st_a = mem_map[MemAccess(
                                    mtype="global",
                                    dtype=np.float32,
                                    lid_strides={0: 1},
                                    gid_strides={0: 256},
                                    direction="store",
                                    variable="a",
                                    count_granularity=CountGranularity.WORKITEM)
                               ].eval_with_dict(params)
        f32_s1_l_ld_x = mem_map[MemAccess(
                                    mtype="local",
                                    dtype=np.float32,
                                    lid_strides={0: 1},
                                    gid_strides={0: 256},
                                    direction="load",
                                    variable="x",
                                    count_granularity=CountGranularity.WORKITEM)
                               ].eval_with_dict(params)
        f32_s1_l_st_x = mem_map[MemAccess(
                                    mtype="local",
                                    dtype=np.float32,
                                    lid_strides={0: 1},
                                    gid_strides={0: 256},
                                    direction="store",
                                    variable="x",
                                    count_granularity=CountGranularity.WORKITEM)
                               ].eval_with_dict(params)

        # (now use these counts to, e.g., predict performance)

    """

    if entrypoint is None:
        if len(program.entrypoints) > 1:
            raise LoopyError("Must provide entrypoint")

        entrypoint = list(program.entrypoints)[0]

    assert entrypoint in program.entrypoints

    from loopy.preprocess import preprocess_program, infer_unknown_types

    program = preprocess_program(program)

    from loopy.match import parse_match
    within = parse_match(within)

    # Ordering restriction: preprocess might insert arguments to
    # make strides valid. Those also need to go through type inference.
    program = infer_unknown_types(program, expect_completion=True)

    return _get_mem_access_map_for_single_kernel(
            program[entrypoint], program.callables_table,
            count_redundant_work=count_redundant_work,
            subgroup_size=subgroup_size,
            within=within)

# }}}


# {{{ get_synchronization_map

def _get_synchronization_map_for_single_kernel(knl, callables_table,
        subgroup_size=None):

    knl = lp.get_one_linearized_kernel(knl, callables_table)

    from loopy.schedule import (EnterLoop, LeaveLoop, Barrier,
            CallKernel, ReturnFromKernel, RunInstruction)

    kernel_rec = partial(_get_synchronization_map_for_single_kernel,
            callables_table=callables_table,
            subgroup_size=subgroup_size)

    sync_counter = CounterBase(knl, callables_table, kernel_rec)
    sync_map = sync_counter.new_zero_poly_map()

    iname_list = []

    for sched_item in knl.linearization:
        if isinstance(sched_item, EnterLoop):
            if sched_item.iname:  # (if not empty)
                iname_list.append(sched_item.iname)
        elif isinstance(sched_item, LeaveLoop):
            if sched_item.iname:  # (if not empty)
                iname_list.pop()

        elif isinstance(sched_item, Barrier):
            sync_map = sync_map + ToCountMap(
                    {Sync(
                        "barrier_%s" % sched_item.synchronization_kind,
                        knl.name): count_inames_domain(knl, frozenset(iname_list))})

        elif isinstance(sched_item, RunInstruction):
            pass

        elif isinstance(sched_item, CallKernel):
            sync_map = sync_map + ToCountMap(
                    {Sync("kernel_launch", knl.name):
                        count_inames_domain(knl, frozenset(iname_list))})

        elif isinstance(sched_item, ReturnFromKernel):
            pass

        else:
            raise LoopyError("unexpected schedule item: %s"
                    % type(sched_item).__name__)

    return sync_map


def get_synchronization_map(program, subgroup_size=None, entrypoint=None):
    """Count the number of synchronization events each work-item encounters in
    a loopy kernel.

    :arg knl: A :class:`loopy.LoopKernel` whose barriers are to be counted.

    :arg subgroup_size: (currently unused) An :class:`int`, :class:`str`
        ``"guess"``, or *None* that specifies the sub-group size. An OpenCL
        sub-group is an implementation-dependent grouping of work-items within
        a work-group, analagous to an NVIDIA CUDA warp. subgroup_size is used,
        e.g., when counting a :class:`MemAccess` whose count_granularity
        specifies that it should only be counted once per sub-group. If set to
        *None* an attempt to find the sub-group size using the device will be
        made, if this fails an error will be raised. If a :class:`str`
        ``"guess"`` is passed as the subgroup_size, get_mem_access_map will
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
        params = {"n": 512, "m": 256, "l": 128}
        barrier_ct = sync_map["barrier_local"].eval_with_dict(params)

        # (now use this count to, e.g., predict performance)

    """
    if entrypoint is None:
        if len(program.entrypoints) > 1:
            raise LoopyError("Must provide entrypoint")

        entrypoint = list(program.entrypoints)[0]

    assert entrypoint in program.entrypoints
    from loopy.preprocess import preprocess_program, infer_unknown_types

    program = preprocess_program(program)
    # Ordering restriction: preprocess might insert arguments to
    # make strides valid. Those also need to go through type inference.
    program = infer_unknown_types(program, expect_completion=True)

    return _get_synchronization_map_for_single_kernel(
            program[entrypoint], program.callables_table,
            subgroup_size=subgroup_size)

# }}}


# {{{ gather_access_footprints

def _gather_access_footprints_for_single_kernel(kernel, ignore_uncountable):
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

        write_footprints.append(afg(insn.assignees))
        read_footprints.append(afg(insn.expression))

    return write_footprints, read_footprints


def gather_access_footprints(program, ignore_uncountable=False, entrypoint=None):
    """Return a dictionary mapping ``(var_name, direction)`` to
    :class:`islpy.Set` instances capturing which indices of each the array
    *var_name* are read/written (where *direction* is either ``read`` or
    ``write``.

    :arg ignore_uncountable: If *False*, an error will be raised for accesses
        on which the footprint cannot be determined (e.g. data-dependent or
        nonlinear indices)
    """

    if entrypoint is None:
        if len(program.entrypoints) > 1:
            raise LoopyError("Must provide entrypoint")

        entrypoint = list(program.entrypoints)[0]

    assert entrypoint in program.entrypoints

    # FIMXE: works only for one callable kernel till now.
    if len([in_knl_callable for in_knl_callable in
        program.callables_table.values() if isinstance(in_knl_callable,
            CallableKernel)]) != 1:
        raise NotImplementedError("Currently only supported for program with "
            "only one CallableKernel.")

    from loopy.preprocess import preprocess_program, infer_unknown_types

    program = preprocess_program(program)
    # Ordering restriction: preprocess might insert arguments to
    # make strides valid. Those also need to go through type inference.
    program = infer_unknown_types(program, expect_completion=True)

    write_footprints = []
    read_footprints = []

    write_footprints, read_footprints = _gather_access_footprints_for_single_kernel(
            program[entrypoint], ignore_uncountable)

    write_footprints = AccessFootprintGatherer.combine(write_footprints)
    read_footprints = AccessFootprintGatherer.combine(read_footprints)

    result = {}

    for vname, footprint in write_footprints.items():
        result[(vname, "write")] = footprint

    for vname, footprint in read_footprints.items():
        result[(vname, "read")] = footprint

    return result


def gather_access_footprint_bytes(program, ignore_uncountable=False):
    """Return a dictionary mapping ``(var_name, direction)`` to
    :class:`islpy.PwQPolynomial` instances capturing the number of bytes  are
    read/written (where *direction* is either ``read`` or ``write`` on array
    *var_name*

    :arg ignore_uncountable: If *True*, an error will be raised for accesses on
        which the footprint cannot be determined (e.g. data-dependent or
        nonlinear indices)
    """

    from loopy.preprocess import preprocess_program, infer_unknown_types
    kernel = infer_unknown_types(program, expect_completion=True)

    from loopy.kernel import KernelState
    if kernel.state < KernelState.PREPROCESSED:
        kernel = preprocess_program(program)

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

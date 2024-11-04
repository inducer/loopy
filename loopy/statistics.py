from __future__ import annotations

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

from functools import cached_property, partial

import islpy as isl
from islpy import dim_type
from pymbolic.mapper import CombineMapper
from pytools import ImmutableRecord, memoize_method

import loopy as lp
from loopy.diagnostic import LoopyError, warn_with_kernel
from loopy.kernel.data import AddressSpace, MultiAssignmentBase, TemporaryVariable
from loopy.kernel.function_interface import CallableKernel
from loopy.symbolic import CoefficientCollector, flatten
from loopy.translation_unit import TranslationUnit
from loopy.typing import Expression
from loopy.types import LoopyType


__doc__ = """

.. currentmodule:: loopy

.. autoclass:: ToCountMap
.. autoclass:: ToCountPolynomialMap
.. autoclass:: CountGranularity
.. autoclass:: OpType
.. autoclass:: Op
.. autoclass:: AccessDirection
.. autoclass:: MemAccess
.. autoclass:: SynchronizationKind
.. autoclass:: Sync

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


def get_kernel_parameter_space(kernel: LoopKernel) -> isl.Space:
    return isl.Space.create_from_names(kernel.isl_context,
            set=[], params=sorted(kernel.outer_params())).params()


def get_kernel_zero_pwqpolynomial(kernel: LoopKernel) -> PwQPolynomial:
    space = get_kernel_parameter_space(kernel)
    space = space.insert_dims(dim_type.out, 0, 1)
    return PwQPolynomial.zero(space)


# {{{ GuardedPwQPolynomial

def _get_param_tuple(obj) -> Tuple[str, ...]:
    return tuple(
            obj.get_dim_name(dim_type.param, i)
            for i in range(obj.dim(dim_type.param)))


class GuardedPwQPolynomial:
    def __init__(self,
                 pwqpolynomial: PwQPolynomial, valid_domain: isl.Set) -> None:
        assert isinstance(pwqpolynomial, PwQPolynomial)
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

Countable = Union["Op", "MemAccess", "Sync"]
CountT = TypeVar("CountT")


class ToCountMap(Generic[CountT]):
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

    count_map: Dict[Countable, CountT]

    def __init__(self, count_map: Optional[Dict[Countable, CountT]] = None) -> None:
        if count_map is None:
            count_map = {}

        self.count_map = count_map

    def _zero(self):
        return 0

    def __add__(self, other: ToCountMap[CountT]) -> ToCountMap[CountT]:
        result = self.count_map.copy()
        for k, v in other.count_map.items():
            result[k] = self.count_map.get(k, 0) + v
        return self.copy(count_map=result)

    def __radd__(self, other: Union[int, ToCountMap[CountT]]) -> ToCountMap[CountT]:
        if other != 0:
            raise ValueError("ToCountMap: Attempted to add ToCountMap "
                                "to {} {}. ToCountMap may only be added to "
                                "0 and other ToCountMap objects."
                                .format(type(other), other))

        return self

    def __mul__(self, other: GuardedPwQPolynomial) -> ToCountMap[CountT]:
        if isinstance(other, GuardedPwQPolynomial):
            return self.copy({
                index: other*value
                for index, value in self.count_map.items()})
        else:
            raise ValueError("ToCountMap: Attempted to multiply "
                                "ToCountMap by {} {}."
                                .format(type(other), other))

    __rmul__ = __mul__

    def __getitem__(self, index: Countable) -> CountT:
        return self.count_map[index]

    def __repr__(self) -> str:
        return repr(self.count_map)

    def __str__(self) -> str:
        return "\n".join(
                f"{k}: {v}"
                for k, v in sorted(self.count_map.items(),
                    key=lambda k: str(k)))

    def __len__(self) -> int:
        return len(self.count_map)

    def get(self,
            key: Countable, default: Optional[CountT] = None) -> Optional[CountT]:
        return self.count_map.get(key, default)

    def items(self):
        return self.count_map.items()

    def keys(self):
        return self.count_map.keys()

    def values(self):
        return self.count_map.values()

    def copy(
            self, count_map: Optional[Dict[Countable, CountT]] = None
             ) -> ToCountMap[CountT]:
        if count_map is None:
            count_map = self.count_map

        return type(self)(count_map=count_map)

    def with_set_attributes(self, **kwargs) -> ToCountMap:
        return self.copy(count_map={
            replace(key, **kwargs): val
            for key, val in self.count_map.items()})

    def filter_by(self, **kwargs) -> ToCountMap:
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

    def filter_by_func(
            self, func: Callable[[Countable], bool]) -> ToCountMap[CountT]:
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

    def group_by(self, *args) -> ToCountMap[CountT]:
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

        new_count_map: Dict[Countable, CountT] = {}

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

    def to_bytes(self) -> ToCountMap[CountT]:
        """Convert counts to bytes using data type in map key.

        :return: A :class:`ToCountMap` mapping each original key to an
            :class:`islpy.PwQPolynomial` with counts in bytes rather than
            instances.

        Example usage::

            # (first create loopy kernel and specify array data types)

            bytes_map = get_mem_access_map(knl).to_bytes()
            params = {"n": 512, "m": 256, "l": 128}

            s1_g_ld_bytes = bytes_map.filter_by(
                                mtype=["global"], lid_strides={0: 1},
                                direction=["load"]).eval_and_sum(params)
            s2_g_ld_bytes = bytes_map.filter_by(
                                mtype=["global"], lid_strides={0: 2},
                                direction=["load"]).eval_and_sum(params)
            s1_g_st_bytes = bytes_map.filter_by(
                                mtype=["global"], lid_strides={0: 1},
                                direction=["store"]).eval_and_sum(params)
            s2_g_st_bytes = bytes_map.filter_by(
                                mtype=["global"], lid_strides={0: 2},
                                direction=["store"]).eval_and_sum(params)

            # (now use these counts to, e.g., predict performance)

        """

        new_count_map = {}

        for key, val in self.count_map.items():
            new_count_map[key] = int(key.dtype.itemsize) * val  # type: ignore[union-attr]  # noqa: E501

        return self.copy(new_count_map)

    def sum(self) -> CountT:
        """:return: A sum of the values of the dictionary."""

        total = self._zero()

        for v in self.count_map.values():
            total = v + total

        return total

# }}}


# {{{ ToCountPolynomialMap

class ToCountPolynomialMap(ToCountMap[GuardedPwQPolynomial]):
    """Maps any type of key to a :class:`islpy.PwQPolynomial` or a
    :class:`~loopy.statistics.GuardedPwQPolynomial`.

    .. automethod:: eval_and_sum
    """

    def __init__(
            self,
            space: isl.Space,
            count_map: Dict[Countable, GuardedPwQPolynomial]
            ) -> None:
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

    def _zero(self) -> isl.PwQPolynomial:
        space = self.space.insert_dims(dim_type.out, 0, 1)
        return isl.PwQPolynomial.zero(space)

    def copy(self, count_map=None, space=None):
        if count_map is None:
            count_map = self.count_map

        if space is None:
            space = self.space

        return type(self)(space, count_map)

    def eval_and_sum(self, params: Optional[Mapping[str, int]] = None) -> int:
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
    from loopy.isl_helpers import get_param_subst_domain, subst_into_pwqpolynomial

    poly = subst_into_pwqpolynomial(
            new_space, guarded_poly.pwqpolynomial, subst_dict)

    valid_domain = guarded_poly.valid_domain
    i_begin_subst_space = valid_domain.dim(dim_type.param)

    valid_domain, subst_domain, _ = get_param_subst_domain(
            new_space, guarded_poly.valid_domain, subst_dict)

    valid_domain = valid_domain & subst_domain
    valid_domain = valid_domain.project_out(dim_type.param, 0, i_begin_subst_space)
    return GuardedPwQPolynomial(poly, valid_domain)


def subst_into_to_count_map(
        space: isl.Space,
        tcm: ToCountPolynomialMap,
        subst_dict: Mapping[str, PwQPolynomial]) -> ToCountPolynomialMap:
    from loopy.isl_helpers import subst_into_pwqpolynomial
    new_count_map: Dict[Countable, GuardedPwQPolynomial] = {}
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

class CountGranularity(Enum):
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

    WORKITEM = 0
    SUBGROUP = 1
    WORKGROUP = 2

# }}}


# {{{ Op descriptor

class OpType(Enum):
    """
    .. attribute:: ADD
    .. attribute:: MUL
    .. attribute:: DIV
    .. attribute:: POW
    .. attribute:: SHIFT
    .. attribute:: BITWISE
    """
    ADD = enum_auto()
    MUL = enum_auto()
    DIV = enum_auto()
    POW = enum_auto()
    SHIFT = enum_auto()
    BITWISE = enum_auto()
    MAXMIN = enum_auto()
    SPECIAL_FUNC = enum_auto()


@dataclass(frozen=True, eq=True)
class Op:
    """A descriptor for a type of arithmetic operation.

    .. attribute:: dtype

       A :class:`loopy.types.LoopyType` or :class:`numpy.dtype` that specifies the
       data type operated on.

    .. attribute:: op_type

       A :class:`OpType`.

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
       analogous to an NVIDIA CUDA warp.

    .. attribute:: kernel_name

        A :class:`str` representing the kernel name where the operation occurred.

    .. attribute:: tags

        A :class:`frozenset` of tags to the operation.

    """
    dtype: Optional[LoopyType] = None
    op_type: Optional[OpType] = None
    count_granularity: Optional[CountGranularity] = None
    kernel_name: Optional[str] = None
    tags: FrozenSet[Tag] = frozenset()

    def __repr__(self):
        if self.kernel_name is not None:
            return (f"Op({self.dtype}, {self.name}, {self.count_granularity},"
                    f' "{self.kernel_name}", {self.tags})')
        else:
            return f"Op({self.dtype}, {self.name}, " + \
                        f"{self.count_granularity}, {self.tags})"

# }}}


# {{{ MemAccess descriptor

class AccessDirection(Enum):
    """
    .. attribute:: READ
    .. attribute:: WRITE
    """
    READ = 0
    WRITE = 1


@dataclass(frozen=True, eq=True)
class MemAccess:
    """A descriptor for a type of memory access.

    .. attribute:: address_space

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

    .. attribute:: read_write

       A :class:`AccessDirection` or *None*.

    .. attribute:: variable

       A :class:`str` that specifies the variable name of the data
       accessed.

    .. attribute:: variable_tags

       A :class:`frozenset` of subclasses of :class:`~pytools.tag.Tag`
       that reflects :attr:`~loopy.TaggedVariable.tags` of
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
       analogous to an NVIDIA CUDA warp.

    .. attribute:: kernel_name

        A :class:`str` representing the kernel name where the operation occurred.

    .. attribute:: tags

        A :class:`frozenset` of tags to the operation.
    """

    address_space: Optional[AddressSpace] = None
    lid_strides: Optional[Mapping[int, Expression]] = None
    gid_strides: Optional[Mapping[int, Expression]] = None
    dtype: Optional[LoopyType] = None
    read_write: Optional[AccessDirection] = None
    variable: Optional[str] = None

    variable_tags: FrozenSet[Tag] = frozenset()
    count_granularity: Optional[CountGranularity] = None
    kernel_name: Optional[str] = None
    tags: FrozenSet[Tag] = frozenset()

    @property
    def mtype(self) -> str:
        from warnings import warn
        warn("MemAccess.mtype is deprecated and will stop working in 2024. "
             "Use MemAccess.address_space instead.",
             DeprecationWarning, stacklevel=2)

        if self.address_space == AddressSpace.GLOBAL:
            return "global"
        elif self.address_space == AddressSpace.LOCAL:
            return "local"
        else:
            raise ValueError(f"unexpected address_space: '{self.address_space}'")

    @property
    def direction(self) -> str:
        from warnings import warn
        warn("MemAccess.access_direction is deprecated "
             "and will stop working in 2024. "
             "Use MemAccess.read_write instead.",
             DeprecationWarning, stacklevel=2)

        if self.read_write == AccessDirection.READ:
            return "read"
        elif self.address_space == AccessDirection.WRITE:
            return "write"
        else:
            raise ValueError(f"unexpected read_write: '{self.read_write}'")

    def __repr__(self):
        # Record.__repr__ overridden for consistent ordering and conciseness
        return "MemAccess({}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
            self.address_space,
            self.dtype,
            None if self.lid_strides is None else dict(
                sorted(self.lid_strides.items())),
            None if self.gid_strides is None else dict(
                sorted(self.gid_strides.items())),
            self.read_write,
            self.variable,
            "None" if not self.variable_tags else str(self.variable_tags),
            self.count_granularity,
            repr(self.kernel_name),
            self.tags)

# }}}


# {{{ Sync descriptor

class SynchronizationKind(Enum):
    BARRIER_GLOBAL = 0
    BARRIER_LOCAL = 1
    KERNEL_LAUNCH = 2


@dataclass(frozen=True, eq=True)
class Sync:
    """A descriptor for a type of synchronization.

    .. attribute:: kind

       A :class:`SynchronizationKind` or *None*.

    .. attribute:: kernel_name

        A :class:`str` representing the kernel name where the operation occurred.

    .. attribute:: tags

        A :class:`frozenset` of tags attached to the synchronization.
    """
    sync_kind: Optional[SynchronizationKind] = None
    kernel_name: Optional[str] = None
    tags: FrozenSet[Tag] = frozenset()

    def __repr__(self):
        # Record.__repr__ overridden for consistent ordering and conciseness
        return f"Sync({self.sync_kind}, {self.kernel_name}, {self.tags})"

# }}}


# {{{ CounterBase

class CounterBase(CombineMapper):
    def __init__(self, knl: LoopKernel, callables_table, kernel_rec) -> None:
        self.knl = knl
        self.callables_table = callables_table
        self.kernel_rec = kernel_rec

        from loopy.type_inference import TypeReader
        self.type_inf = TypeReader(knl, callables_table)
        self.zero = get_kernel_zero_pwqpolynomial(self.knl)
        self.one = self.zero + 1

    @cached_property
    def param_space(self) -> isl.Space:
        return get_kernel_parameter_space(self.knl)

    def new_poly_map(self, count_map) -> ToCountPolynomialMap:
        return ToCountPolynomialMap(self.param_space, count_map)

    def _new_zero_map(self) -> ToCountPolynomialMap:
        return self.new_poly_map({})

    def combine(self, values: Iterable[ToCountMap]) -> ToCountPolynomialMap:
        return sum(values, self._new_zero_map())

    def map_tagged_expression(
            self, expr: TaggedExpression, tags: FrozenSet[Tag]
            ) -> ToCountPolynomialMap:
        return self.rec(expr.expr, expr.tags)

    def map_constant(
            self, expr: Expression, tags: FrozenSet[Tag]
            ) -> ToCountPolynomialMap:
        return self._new_zero_map()

    def map_call(self, expr: p.Call, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        from loopy.symbolic import ResolvedFunction
        assert isinstance(expr.function, ResolvedFunction)
        clbl = self.callables_table[expr.function.name]

        from loopy.kernel.data import ValueArg
        from loopy.kernel.function_interface import (
            CallableKernel,
            get_kw_pos_association,
        )
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
                    + self.rec(expr.parameters, tags)

        else:
            raise NotImplementedError()

    def map_call_with_kwargs(
            self, expr: p.CallWithKwargs, tags: FrozenSet[Tag]
            ) -> ToCountPolynomialMap:
        # See https://github.com/inducer/loopy/pull/323
        raise NotImplementedError

    def map_comparison(
            self, expr: p.Comparison, tags: FrozenSet[Tag]
            ) -> ToCountPolynomialMap:
        return self.rec(expr.left, tags) + self.rec(expr.right, tags)

    def map_if(
            self, expr: p.If, tags: FrozenSet[Tag]
            ) -> ToCountPolynomialMap:
        warn_with_kernel(self.knl, "summing_if_branches",
                         "%s counting sum of if-expression branches."
                         % type(self).__name__)
        return self.rec(expr.condition, tags) + self.rec(expr.then, tags) \
               + self.rec(expr.else_, tags)

    def map_if_positive(
            self, expr: p.IfPositive, tags: FrozenSet[Tag]) -> ToCountMap:
        warn_with_kernel(self.knl, "summing_if_branches",
                         "%s counting sum of if-expression branches."
                         % type(self).__name__)
        return self.rec(expr.criterion, tags) + self.rec(expr.then, tags) \
               + self.rec(expr.else_, tags)

    def map_common_subexpression(
            self, expr: p.CommonSubexpression, tags: FrozenSet[Tag]
            ) -> ToCountPolynomialMap:
        raise RuntimeError("%s encountered %s--not supposed to happen"
                % (type(self).__name__, type(expr).__name__))

    map_substitution = map_common_subexpression
    map_derivative = map_common_subexpression
    map_slice = map_common_subexpression

    def map_reduction(
            self, expr: Reduction, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        # preprocessing should have removed these
        raise RuntimeError("%s encountered %s--not supposed to happen"
                % (type(self).__name__, type(expr).__name__))

    def __call__(
            self, expr, tags: Optional[FrozenSet[Tag]] = None
            ) -> ToCountPolynomialMap:
        if tags is None:
            tags = frozenset()
        return self.rec(expr, tags=tags)

# }}}


# {{{ ExpressionOpCounter

class ExpressionOpCounter(CounterBase):
    def __init__(self, knl: LoopKernel, callables_table, kernel_rec,
                 count_within_subscripts: bool = True):
        super().__init__(knl, callables_table, kernel_rec)
        self.count_within_subscripts = count_within_subscripts

    arithmetic_count_granularity = CountGranularity.SUBGROUP

    def map_constant(self, expr: Any, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        return self._new_zero_map()

    map_tagged_variable = map_constant
    map_variable = map_constant
    map_nan = map_constant

    def map_call(self, expr: p.Call, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        from loopy.symbolic import ResolvedFunction
        assert isinstance(expr.function, ResolvedFunction)
        clbl = self.callables_table[expr.function.name]

        from loopy.kernel.function_interface import CallableKernel
        if not isinstance(clbl, CallableKernel):
            return self.new_poly_map(
                        {Op(dtype=self.type_inf(expr),
                            op_type=OpType.SPECIAL_FUNC,
                            tags=tags,
                            count_granularity=self.arithmetic_count_granularity,
                            kernel_name=self.knl.name): self.one}
                        ) + self.rec(expr.parameters, tags)
        else:
            return super().map_call(expr, tags)

    def map_subscript(
            self, expr: p.Subscript, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        if self.count_within_subscripts:
            return self.rec(expr.index, tags)
        else:
            return self._new_zero_map()

    def map_sub_array_ref(
            self, expr: SubArrayRef, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        # generates an array view, considered free
        return self._new_zero_map()

    def map_sum(self, expr: p.Sum, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        assert expr.children
        return self.new_poly_map(
                    {Op(dtype=self.type_inf(expr),
                        op_type=OpType.ADD,
                        tags=tags,
                        count_granularity=self.arithmetic_count_granularity,
                        kernel_name=self.knl.name):
                     self.zero + (len(expr.children)-1)}
                    ) + sum(self.rec(child, tags) for child in expr.children)

    def map_product(
            self, expr: p.Product, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        from pymbolic.primitives import is_zero
        assert expr.children
        return sum(self.new_poly_map({Op(dtype=self.type_inf(expr),
                                  op_type=OpType.MUL,
                                  tags=tags,
                                  count_granularity=(
                                      self.arithmetic_count_granularity),
                                  kernel_name=self.knl.name): self.one})
                   + self.rec(child, tags)
                   for child in expr.children
                   if not is_zero(child + 1)) + \
                   self.new_poly_map({Op(dtype=self.type_inf(expr),
                                  op_type=OpType.MUL,
                                  tags=tags,
                                  count_granularity=(
                                      self.arithmetic_count_granularity),
                                  kernel_name=self.knl.name): -self.one})

    def map_quotient(
            self, expr: p.QuotientBase, tags: FrozenSet[Tag]
            ) -> ToCountPolynomialMap:
        return self.new_poly_map({Op(dtype=self.type_inf(expr),
                              op_type=OpType.DIV,
                              tags=tags,
                              count_granularity=self.arithmetic_count_granularity,
                              kernel_name=self.knl.name): self.one}) \
                                + self.rec(expr.numerator, tags) \
                                + self.rec(expr.denominator, tags)

    map_floor_div = map_quotient
    map_remainder = map_quotient

    def map_power(self, expr: p.Power, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        return self.new_poly_map({Op(dtype=self.type_inf(expr),
                              op_type=OpType.POW,
                              tags=tags,
                              count_granularity=self.arithmetic_count_granularity,
                              kernel_name=self.knl.name): self.one}) \
                                + self.rec(expr.base, tags) \
                                + self.rec(expr.exponent, tags)

    def map_left_shift(
            self, expr: Union[p.LeftShift, p.RightShift], tags: FrozenSet[Tag]
            ) -> ToCountPolynomialMap:
        return self.new_poly_map({Op(dtype=self.type_inf(expr),
                              op_type=OpType.SHIFT,
                              tags=tags,
                              count_granularity=self.arithmetic_count_granularity,
                              kernel_name=self.knl.name): self.one}) \
                                + self.rec(expr.shiftee, tags) \
                                + self.rec(expr.shift, tags)

    map_right_shift = map_left_shift

    def map_bitwise_not(
            self, expr: p.BitwiseNot, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        return self.new_poly_map({Op(dtype=self.type_inf(expr),
                              op_type=OpType.BITWISE,
                              tags=tags,
                              count_granularity=self.arithmetic_count_granularity,
                              kernel_name=self.knl.name): self.one}) \
                                + self.rec(expr.child, tags)

    def map_bitwise_or(
            self, expr: Union[p.BitwiseOr, p.BitwiseAnd, p.BitwiseXor],
            tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        return self.new_poly_map({Op(dtype=self.type_inf(expr),
                              op_type=OpType.BITWISE,
                              tags=tags,
                              count_granularity=self.arithmetic_count_granularity,
                              kernel_name=self.knl.name):
                           self.zero + (len(expr.children)-1)}) \
                              + sum(self.rec(child, tags) for child in expr.children)

    map_bitwise_xor = map_bitwise_or
    map_bitwise_and = map_bitwise_or

    def map_if(self, expr: p.If, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        warn_with_kernel(self.knl, "summing_if_branches_ops",
                         "ExpressionOpCounter counting ops as sum of "
                         "if-statement branches.")
        return self.rec(expr.condition, tags) + self.rec(expr.then, tags) \
               + self.rec(expr.else_, tags)

    def map_if_positive(
            self, expr: p.IfPositive, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        warn_with_kernel(self.knl, "summing_ifpos_branches_ops",
                         "ExpressionOpCounter counting ops as sum of "
                         "if_pos-statement branches.")
        return self.rec(expr.criterion, tags) + self.rec(expr.then, tags) \
               + self.rec(expr.else_, tags)

    def map_min(
            self, expr: Union[p. Min, p.Max], tags: FrozenSet[Tag]
            ) -> ToCountPolynomialMap:
        return self.new_poly_map({Op(dtype=self.type_inf(expr),
                              op_type=OpType.MAXMIN,
                              tags=tags,
                              count_granularity=self.arithmetic_count_granularity,
                              kernel_name=self.knl.name):
                           len(expr.children)-1}) \
               + sum(self.rec(child, tags) for child in expr.children)

    map_max = map_min

    def map_common_subexpression(self, expr, tags):
        raise NotImplementedError("ExpressionOpCounter encountered "
                                  "common_subexpression, "
                                  "map_common_subexpression not implemented.")

    def map_substitution(self, expr, tags):
        raise NotImplementedError("ExpressionOpCounter encountered "
                                  "substitution, "
                                  "map_substitution not implemented.")

    def map_derivative(self, expr, tags):
        raise NotImplementedError("ExpressionOpCounter encountered "
                                  "derivative, "
                                  "map_derivative not implemented.")

    def map_slice(self, expr, tags):
        raise NotImplementedError("ExpressionOpCounter encountered slice, "
                                  "map_slice not implemented.")

# }}}


# {{{ modified coefficient collector that ignores denominator of floor div

class _IndexStrideCoefficientCollector(CoefficientCollector):

    def map_floor_div(self, expr):
        from warnings import warn
        warn("_IndexStrideCoefficientCollector encountered FloorDiv, ignoring "
             "denominator in expression %s" % (expr), stacklevel=1)
        return self.rec(expr.numerator)

# }}}


# {{{ _get_lid_and_gid_strides

def _get_lid_and_gid_strides(
        knl: LoopKernel, array: ArrayBase, index: Tuple[Expression, ...]
        ) -> Tuple[Mapping[int, Expression], Mapping[int, Expression]]:
    # find all local and global index tags and corresponding inames
    from loopy.symbolic import get_dependencies
    my_inames = get_dependencies(index) & knl.all_inames()

    from loopy.kernel.data import (
        GroupInameTag,
        LocalInameTag,
        filter_iname_tags_by_type,
    )
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

    # strides are coefficients in flattened index, i.e., we want
    # lid_strides = {0:l0, 1:l1, 2:l2, ...} and
    # gid_strides = {0:g0, 1:g1, 2:g2, ...},
    # where l0, l1, l2, g0, g1, and g2 come from flattened index
    # [... + g2*gid2 + g1*gid1 + g0*gid0 + ... + l2*lid2 + l1*lid1 + l0*lid0]

    from pymbolic.primitives import Variable

    from loopy.diagnostic import ExpressionNotAffineError
    from loopy.kernel.array import FixedStrideArrayDimTag
    from loopy.symbolic import simplify_using_aff

    def get_iname_strides(
            tag_to_iname_dict: Mapping[InameImplementationTag, str]
            ) -> Mapping[InameImplementationTag, Expression]:
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

            tag_to_stride_dict[tag] = flatten(total_iname_stride)

        return tag_to_stride_dict

    return get_iname_strides(lid_to_iname), get_iname_strides(gid_to_iname)

# }}}


# {{{ MemAccessCounterBase

class MemAccessCounter(CounterBase):
    def map_sub_array_ref(
            self, expr: SubArrayRef, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        # generates an array view, considered free
        return self._new_zero_map()

    def map_call(self, expr: p.Call, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        from loopy.symbolic import ResolvedFunction
        assert isinstance(expr.function, ResolvedFunction)
        clbl = self.callables_table[expr.function.name]

        from loopy.kernel.function_interface import CallableKernel
        if not isinstance(clbl, CallableKernel):
            return self.rec(expr.parameters, tags)
        else:
            return super().map_call(expr, tags)

    # local_mem_count_granularity = CountGranularity.SUBGROUP

    def count_var_access(self,
                         dtype: LoopyType,
                         name: str,
                         index: Optional[Tuple[Expression, ...]],
                         tags: FrozenSet[Tag]
                         ) -> ToCountPolynomialMap:
        count_map = {}

        array = self.knl.get_var_descriptor(name)

        if index is None:
            # no subscript
            count_map[MemAccess(
                        address_space=AddressSpace.LOCAL,
                        tags=tags,
                        dtype=dtype,
                        count_granularity=self.local_mem_count_granularity,
                        kernel_name=self.knl.name)] = self.one
            return self.new_poly_map(count_map)

        # could be tuple or scalar index
        index_tuple = index
        if not isinstance(index_tuple, tuple):
            index_tuple = (index_tuple,)

        lid_strides, gid_strides = _get_lid_and_gid_strides(
                                        self.knl, array, index_tuple)

        count_map[MemAccess(
            address_space=array.address_space,
            dtype=dtype,
            tags=tags,
            lid_strides=lid_strides,
            gid_strides=gid_strides,
            variable=name,
            count_granularity=self.local_mem_count_granularity,
            kernel_name=self.knl.name)] = self.one

        return self.new_poly_map(count_map)

    def map_variable(
            self, expr: p.Variable, tags: FrozenSet[Tag]
            ) -> ToCountPolynomialMap:
        return self.count_var_access(
                    self.type_inf(expr), expr.name, None, tags)

    map_tagged_variable = map_variable

    def map_subscript(
            self, expr: p.Subscript, tags: FrozenSet[Tag]) -> ToCountPolynomialMap:
        return (self.count_var_access(self.type_inf(expr),
                                      expr.aggregate.name,
                                      expr.index, tags)
                + self.rec(expr.index, tags))

# }}}


# {{{ AccessFootprintGatherer

FootprintsT = Dict[str, isl.Set]


class AccessFootprintGatherer(CombineMapper):
    def __init__(self,
                 kernel: LoopKernel,
                 domain: isl.BasicSet,
                 ignore_uncountable: bool = False) -> None:
        self.kernel = kernel
        self.domain = domain
        self.ignore_uncountable = ignore_uncountable

    @staticmethod
    def combine(values: Iterable[FootprintsT]) -> FootprintsT:
        assert values

        def merge_dicts(a: FootprintsT, b: FootprintsT) -> FootprintsT:
            result = a.copy()

            for var_name, footprint in b.items():
                if var_name in result:
                    result[var_name] = result[var_name] | footprint
                else:
                    result[var_name] = footprint

            return result

        from functools import reduce
        return reduce(merge_dicts, values)

    def map_constant(self, expr: p.Any) -> FootprintsT:
        return {}

    def map_variable(self, expr: p.Variable) -> FootprintsT:
        return {}

    def map_subscript(self, expr: p.Subscript) -> FootprintsT:
        subscript = expr.index

        if not isinstance(subscript, tuple):
            subscript = (subscript,)

        from loopy.symbolic import get_access_map

        try:
            access_range = get_access_map(self.domain, subscript,
                    self.kernel.assumptions).range()
        except isl.Error as err:
            # Likely: index was non-linear, nothing we can do.
            if self.ignore_uncountable:
                return {}
            else:
                raise LoopyError("failed to gather footprint: %s" % expr) from err

        except TypeError as err:
            # Likely: index was non-linear, nothing we can do.
            if self.ignore_uncountable:
                return {}
            else:
                raise LoopyError("failed to gather footprint: %s" % expr) from err

        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        return self.combine([
            self.rec(expr.index),
            {expr.aggregate.name: access_range}])

# }}}


# {{{ count

def add_assumptions_guard(
        kernel: LoopKernel, pwqpolynomial: isl.PwQPolynomial
        ) -> GuardedPwQPolynomial:
    return GuardedPwQPolynomial(
            pwqpolynomial,
            kernel.assumptions.align_params(pwqpolynomial.space))


def count(kernel, set: isl.Set, space=None) -> GuardedPwQPolynomial:
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


def get_unused_hw_axes_factor(
        knl: LoopKernel, callables_table, insn: InstructionBase,
        disregard_local_axes: bool) -> GuardedPwQPolynomial:
    # FIXME: Multi-kernel support
    gsize, lsize = knl.get_grid_size_upper_bounds(callables_table)

    g_used = set()
    l_used = set()

    from loopy.kernel.data import GroupInameTag, LocalInameTag
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


def count_inames_domain(
        knl: LoopKernel, inames: FrozenSet[str]) -> GuardedPwQPolynomial:
    space = get_kernel_parameter_space(knl)
    if not inames:
        return add_assumptions_guard(knl,
                get_kernel_zero_pwqpolynomial(knl) + 1)

    inames_domain = knl.get_inames_domain(inames)
    domain = inames_domain.project_out_except(inames, [dim_type.set])
    return count(knl, domain, space=space)


def count_insn_runs(
        knl: LoopKernel,
        callables_table: Mapping[str, InKernelCallable],
        insn: InstructionBase,
        count_redundant_work: bool,
        disregard_local_axes: bool = False) -> GuardedPwQPolynomial:

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


def _get_insn_count(
        knl: LoopKernel,
        callables_table: Mapping[str, InKernelCallable],
        insn_id: str,
        subgroup_size: Optional[int],
        count_redundant_work: bool,
        count_granularity: CountGranularity = CountGranularity.WORKITEM
        ) -> GuardedPwQPolynomial:
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
        raise ValueError("get_insn_count: count_granularity "
                         f"'{count_granularity}' is not allowed.")

# }}}


# {{{ get_op_map

def _get_op_map_for_single_kernel(
        knl: LoopKernel,
        callables_table: Mapping[str, InKernelCallable],
        count_redundant_work: bool,
        count_within_subscripts: bool,
        subgroup_size: int, within) -> ToCountPolynomialMap:

    subgroup_size = _process_subgroup_size(knl, subgroup_size)

    kernel_rec = partial(_get_op_map_for_single_kernel,
            callables_table=callables_table,
            count_redundant_work=count_redundant_work,
            count_within_subscripts=count_within_subscripts,
            subgroup_size=subgroup_size, within=within)

    op_counter = ExpressionOpCounter(knl, callables_table, kernel_rec,
            count_within_subscripts)
    op_map = op_counter._new_zero_map()

    from loopy.kernel.instruction import (
        Assignment,
        BarrierInstruction,
        CallInstruction,
        CInstruction,
        NoOpInstruction,
    )

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


def get_op_map(
        t_unit: TranslationUnit, *, count_redundant_work: bool = False,
        count_within_subscripts: bool = True,
        subgroup_size: Optional[int] = None,
        entrypoint: Optional[str] = None,
        within: Any = None):

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
        a work-group, analogous to an NVIDIA CUDA warp. subgroup_size is used,
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
        if len(t_unit.entrypoints) > 1:
            raise LoopyError("Must provide entrypoint")

        entrypoint = list(t_unit.entrypoints)[0]

    assert entrypoint in t_unit.entrypoints

    from loopy.preprocess import infer_unknown_types, preprocess_program
    program = preprocess_program(program)

    from loopy.match import parse_match
    within = parse_match(within)

    # Ordering restriction: preprocess might insert arguments to
    # make strides valid. Those also need to go through type inference.
    t_unit = infer_unknown_types(t_unit, expect_completion=True)

    kernel = t_unit[entrypoint]
    assert isinstance(kernel, LoopKernel)

    return _get_op_map_for_single_kernel(
            kernel, t_unit.callables_table,
            count_redundant_work=count_redundant_work,
            count_within_subscripts=count_within_subscripts,
            subgroup_size=subgroup_size,
            within=within)

# }}}


# {{{ subgroup size finding

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

def _get_mem_access_map_for_single_kernel(
        knl: LoopKernel,
        callables_table: Mapping[str, InKernelCallable],
        count_redundant_work: bool, subgroup_size: Optional[int],
        within: Any) -> ToCountPolynomialMap:

    subgroup_size = _process_subgroup_size(knl, subgroup_size)

    kernel_rec = partial(_get_mem_access_map_for_single_kernel,
            callables_table=callables_table,
            count_redundant_work=count_redundant_work,
            subgroup_size=subgroup_size)

    access_counter = MemAccessCounter(knl, callables_table, kernel_rec)
    access_map = access_counter._new_zero_map()

    from loopy.kernel.instruction import (
        Assignment,
        BarrierInstruction,
        CallInstruction,
        CInstruction,
        NoOpInstruction,
    )

    for insn in knl.instructions:
        if within(knl, insn):
            if isinstance(insn, (CallInstruction, Assignment)):
                insn_access_map = (
                            access_counter(insn.expression)
                            ).with_set_attributes(read_write=AccessDirection.READ)
                for assignee in insn.assignees:
                    insn_access_map = insn_access_map + (
                            access_counter(assignee)
                            ).with_set_attributes(read_write=AccessDirection.WRITE)

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


def get_mem_access_map(
        t_unit: TranslationUnit, *, count_redundant_work: bool = False,
        subgroup_size: Optional[int] = None,
        entrypoint: Optional[str] = None,
        within: Any = None) -> ToCountPolynomialMap:
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
        analogous to an NVIDIA CUDA warp. subgroup_size is used, e.g., when
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
        if len(t_unit.entrypoints) > 1:
            raise LoopyError("Must provide entrypoint")

        entrypoint = list(t_unit.entrypoints)[0]

    assert entrypoint in t_unit.entrypoints

    from loopy.preprocess import infer_unknown_types, preprocess_program

    t_unit = preprocess_program(t_unit)

    from loopy.match import parse_match
    within = parse_match(within)

    # Ordering restriction: preprocess might insert arguments to
    # make strides valid. Those also need to go through type inference.
    t_unit = infer_unknown_types(t_unit, expect_completion=True)

    return _get_mem_access_map_for_single_kernel(
            t_unit[entrypoint], t_unit.callables_table,
            count_redundant_work=count_redundant_work,
            subgroup_size=subgroup_size,
            within=within)

# }}}


# {{{ get_synchronization_map

def _get_synchronization_map_for_single_kernel(
        knl: LoopKernel,
        callables_table: Mapping[str, InKernelCallable],
        subgroup_size: Optional[int] = None):

    knl = lp.get_one_linearized_kernel(knl, callables_table)

    from loopy.schedule import (
        Barrier,
        CallKernel,
        EnterLoop,
        LeaveLoop,
        ReturnFromKernel,
        RunInstruction,
    )

    kernel_rec = partial(_get_synchronization_map_for_single_kernel,
            callables_table=callables_table,
            subgroup_size=subgroup_size)

    sync_counter = CounterBase(knl, callables_table, kernel_rec)
    sync_map = sync_counter._new_zero_map()

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
                    {Sync(SynchronizationKind.KERNEL_LAUNCH, knl.name):
                        count_inames_domain(knl, frozenset(iname_list))})

        elif isinstance(sched_item, ReturnFromKernel):
            pass

        else:
            raise LoopyError("unexpected schedule item: %s"
                    % type(sched_item).__name__)

    return sync_map


def get_synchronization_map(
        t_unit: TranslationUnit, *,
        subgroup_size: Optional[int] = None,
        entrypoint: Optional[str] = None) -> ToCountPolynomialMap:
    """Count the number of synchronization events each work-item encounters in
    a loopy kernel.

    :arg knl: A :class:`loopy.LoopKernel` whose barriers are to be counted.

    :arg subgroup_size: (currently unused) An :class:`int`, :class:`str`
        ``"guess"``, or *None* that specifies the sub-group size. An OpenCL
        sub-group is an implementation-dependent grouping of work-items within
        a work-group, analogous to an NVIDIA CUDA warp. subgroup_size is used,
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
        if len(t_unit.entrypoints) > 1:
            raise LoopyError("Must provide entrypoint")

        entrypoint = list(t_unit.entrypoints)[0]

    assert entrypoint in program.entrypoints
    from loopy.preprocess import infer_unknown_types, preprocess_program

    t_unit = preprocess_program(t_unit)
    # Ordering restriction: preprocess might insert arguments to
    # make strides valid. Those also need to go through type inference.
    t_unit = infer_unknown_types(t_unit, expect_completion=True)

    return _get_synchronization_map_for_single_kernel(
            t_unit[entrypoint], t_unit.callables_table,
            subgroup_size=subgroup_size)

# }}}


# {{{ gather_access_footprints

def _gather_access_footprints_for_single_kernel(
        kernel: LoopKernel, ignore_uncountable: bool
        ) -> Tuple[FootprintsT, FootprintsT]:
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

    return (
            AccessFootprintGatherer.combine(write_footprints),
            AccessFootprintGatherer.combine(read_footprints))


def gather_access_footprints(
        t_unit: TranslationUnit, *, ignore_uncountable: bool = False,
        entrypoint: Optional[str] = None) -> Mapping[MemAccess, isl.Set]:
    """Return a dictionary mapping ``(var_name, direction)`` to
    :class:`islpy.Set` instances capturing which indices of each the array
    *var_name* are read/written (where *direction* is either ``read`` or
    ``write``.

    :arg ignore_uncountable: If *False*, an error will be raised for accesses
        on which the footprint cannot be determined (e.g. data-dependent or
        nonlinear indices)
    """

    if entrypoint is None:
        if len(t_unit.entrypoints) > 1:
            raise LoopyError("Must provide entrypoint")

        entrypoint = list(t_unit.entrypoints)[0]

    assert entrypoint in t_unit.entrypoints

    # FIXME: works only for one callable kernel till now.
    if len([in_knl_callable for in_knl_callable in
        t_unit.callables_table.values() if isinstance(in_knl_callable,
            CallableKernel)]) != 1:
        raise NotImplementedError("Currently only supported for "
                                  "translation unit with only one CallableKernel.")

    from loopy.preprocess import infer_unknown_types, preprocess_program

    t_unit = preprocess_program(t_unit)
    # Ordering restriction: preprocess might insert arguments to
    # make strides valid. Those also need to go through type inference.
    t_unit = infer_unknown_types(t_unit, expect_completion=True)

    kernel = t_unit[entrypoint]
    assert isinstance(kernel, LoopKernel)
    write_footprints, read_footprints = _gather_access_footprints_for_single_kernel(
            kernel, ignore_uncountable)

    result = {}

    for vname, footprint in write_footprints.items():
        result[MemAccess(variable=vname, read_write=AccessDirection.WRITE)] \
                = footprint

    for vname, footprint in read_footprints.items():
        result[MemAccess(variable=vname, read_write=AccessDirection.READ)] \
                = footprint

    return result


def gather_access_footprint_bytes(
        t_unit: TranslationUnit, *, ignore_uncountable: bool = False
        ) -> ToCountPolynomialMap:
    """Return a dictionary mapping ``(var_name, direction)`` to
    :class:`islpy.PwQPolynomial` instances capturing the number of bytes  are
    read/written (where *direction* is either ``read`` or ``write`` on array
    *var_name*

    :arg ignore_uncountable: If *True*, an error will be raised for accesses on
        which the footprint cannot be determined (e.g. data-dependent or
        nonlinear indices)
    """

    from loopy.preprocess import infer_unknown_types, preprocess_program
    kernel = infer_unknown_types(program, expect_completion=True)

    fp = gather_access_footprints(t_unit, ignore_uncountable=ignore_uncountable)

    # FIXME: Only supporting a single kernel for now
    kernel = t_unit.default_entrypoint

    result = {}
    for ma, var_fp in fp.items():
        assert ma.variable
        var_descr = kernel.get_var_descriptor(ma.variable)
        assert var_descr.dtype
        bytes_transferred = (
                int(var_descr.dtype.numpy_dtype.itemsize)
                * count(kernel, var_fp))
        result[ma] = add_assumptions_guard(kernel, bytes_transferred)

    return ToCountPolynomialMap(get_kernel_parameter_space(kernel), result)

# }}}

# vim: foldmethod=marker

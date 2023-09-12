"""Implementation tagging of array axes."""

from __future__ import annotations

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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

import sys
from typing import (cast, Optional, Tuple, Union, FrozenSet, Type, Sequence,
        List, Callable, ClassVar, TypeVar, TYPE_CHECKING)
from dataclasses import dataclass
import re
from warnings import warn

from pytools import ImmutableRecord
from pytools.tag import Taggable, Tag

import numpy as np  # noqa

from loopy.diagnostic import LoopyError
from loopy.tools import is_integer
from loopy.typing import ExpressionT, ShapeType
from loopy.types import LoopyType

if TYPE_CHECKING:
    from loopy.target import TargetBase
    from loopy.kernel import LoopKernel
    from loopy.kernel.data import auto, TemporaryVariable, ArrayArg
    from loopy.codegen import VectorizationInfo

if getattr(sys, "_BUILDING_SPHINX_DOCS", False):
    from loopy.target import TargetBase  # noqa: F811


T = TypeVar("T")


__doc__ = """
.. currentmodule:: loopy.kernel.array

.. autoclass:: ArrayDimImplementationTag

.. autoclass:: _StrideArrayDimTagBase

.. autoclass:: FixedStrideArrayDimTag

.. autoclass:: ComputedStrideArrayDimTag

.. autoclass:: SeparateArrayArrayDimTag

.. autoclass:: VectorArrayDimTag

.. autofunction:: parse_array_dim_tags
"""


# {{{ array dimension tags

class ArrayDimImplementationTag(ImmutableRecord):
    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        key_builder.rec(key_hash, self.stringify(True).encode("utf8"))


class _StrideArrayDimTagBase(ArrayDimImplementationTag):
    """
    .. attribute :: target_axis

        For objects (such as images) with more than one axis, *target_axis*
        sets which of these indices is being targeted by this dimension.
        Note that there may be multiple dim_tags with the same *target_axis*,
        their contributions are combined additively.

        Note that "normal" arrays only have one *target_axis*.

    .. attribute:: layout_nesting_level

        For determining the stride of :class:`ComputedStrideArrayDimTag`,
        this determines the layout nesting level of this axis.
        This must be a contiguous sequence of unique
        integers starting at 0 in a single :attr:`ArrayBase.dim_tags`.
        The lowest nesting level varies fastest when viewed
        in linear memory.

        May be None on :class:`FixedStrideArrayDimTag`, in which case no
        :class:`ComputedStrideArrayDimTag` instances may occur.
    """

    def depends_on(self):
        raise NotImplementedError()


class FixedStrideArrayDimTag(_StrideArrayDimTagBase):
    """An arg dimension implementation tag for a fixed (potentially
    symbolic) stride.

    .. attribute :: stride

        May be one of the following:

        - A :class:`pymbolic.primitives.Expression`, including an
          integer, indicating the stride in units of the underlying
          array's :attr:`ArrayBase.dtype`.

        - :class:`loopy.auto`, indicating that a new kernel argument
          for this stride should automatically be created.

    The stride is given in units of :attr:`ArrayBase.dtype`.
    """

    def __init__(self, stride, target_axis=0, layout_nesting_level=None):
        if not (
                layout_nesting_level is None
                or
                isinstance(layout_nesting_level, int)):
            raise TypeError("layout_nesting_level must be an int or None")

        _StrideArrayDimTagBase.__init__(self,
                stride=stride, target_axis=target_axis,
                layout_nesting_level=layout_nesting_level)

    def stringify(self, include_target_axis):
        result = ""
        if self.layout_nesting_level is not None:
            result += "N%d:" % self.layout_nesting_level

        import loopy as lp
        if self.stride is lp.auto:
            result += "stride:auto"
        else:
            result += "stride:"+str(self.stride)

        if include_target_axis:
            result += "->%d" % self.target_axis
        return result

    def __str__(self):
        return self.stringify(True)

    def map_expr(self, mapper):
        from loopy.kernel.data import auto

        if self.stride is auto:
            # lp.auto not an expr => do not map
            return self

        return self.copy(stride=mapper(self.stride))

    def depends_on(self):
        from loopy.kernel.data import auto
        from loopy.symbolic import DependencyMapper
        if self.stride is auto:
            return frozenset()

        return DependencyMapper(composite_leaves=auto)(self.stride)


class ComputedStrideArrayDimTag(_StrideArrayDimTagBase):
    """
    .. attribute:: pad_to

        :attr:`ArrayBase.dtype` granularity to which to pad this dimension

    This type of stride arg dim gets converted to
    :class:`FixedStrideArrayDimTag` on input to :class:`ArrayBase` subclasses.
    """

    def __init__(self, layout_nesting_level, pad_to=None, target_axis=0, ):
        if not isinstance(layout_nesting_level, int):
            raise TypeError("layout_nesting_level must be an int")

        _StrideArrayDimTagBase.__init__(self, pad_to=pad_to,
                target_axis=target_axis, layout_nesting_level=layout_nesting_level)

    def stringify(self, include_target_axis):
        result = "N"+str(self.layout_nesting_level)
        if self.pad_to is not None:
            result += "(pad=%s)" % self.pad_to

        if include_target_axis:
            result += "->%d" % self.target_axis

        return result

    def __str__(self):
        return self.stringify(True)

    def map_expr(self, mapper):
        return self

    def depends_on(self):
        return frozenset()


class SeparateArrayArrayDimTag(ArrayDimImplementationTag):
    def stringify(self, include_target_axis):
        return "sep"

    def __str__(self):
        return self.stringify(True)

    def map_expr(self, mapper):
        return self

    def depends_on(self):
        return frozenset()


class VectorArrayDimTag(ArrayDimImplementationTag):
    def stringify(self, include_target_axis):
        return "vec"

    def __str__(self):
        return self.stringify(True)

    def map_expr(self, mapper):
        return self

    def depends_on(self):
        return frozenset()


NESTING_LEVEL_RE = re.compile(r"^N([-0-9]+)(?::(.*)|)$")
PADDED_STRIDE_TAG_RE = re.compile(r"^([a-zA-Z]*)\(pad=(.*)\)$")
TARGET_AXIS_RE = re.compile(r"->([0-9])$")


def _parse_array_dim_tag(tag, default_target_axis, nesting_levels):
    if isinstance(tag, ArrayDimImplementationTag):
        return False, False, tag

    if not isinstance(tag, str):
        raise TypeError("arg dimension implementation tag must be "
                "string or tag object")

    tag = tag.strip()
    is_optional = False
    if tag.endswith("?"):
        tag = tag[:-1]
        is_optional = True

    orig_tag = tag

    if tag == "sep":
        return False, is_optional, SeparateArrayArrayDimTag()
    elif tag == "vec":
        return False, is_optional, VectorArrayDimTag()

    nesting_level_match = NESTING_LEVEL_RE.match(tag)

    if nesting_level_match is not None:
        nesting_level = int(nesting_level_match.group(1))
        tag = nesting_level_match.group(2)
        if tag is None:
            tag = ""
    else:
        nesting_level = None

    has_explicit_nesting_level = nesting_level is not None

    target_axis_match = TARGET_AXIS_RE.search(tag)

    if target_axis_match is not None:
        target_axis = int(target_axis_match.group(1))
        tag = tag[:target_axis_match.start()]
    else:
        target_axis = default_target_axis

    ta_nesting_levels = nesting_levels.get(target_axis, [])

    if tag.startswith("stride:"):
        fixed_stride_descr = tag[7:]
        if fixed_stride_descr.strip() == "auto":
            import loopy as lp
            return (
                    has_explicit_nesting_level, is_optional,
                    FixedStrideArrayDimTag(
                        lp.auto, target_axis,
                        layout_nesting_level=nesting_level))
        else:
            from loopy.symbolic import parse
            return (
                has_explicit_nesting_level, is_optional,
                FixedStrideArrayDimTag(
                    parse(fixed_stride_descr), target_axis,
                    layout_nesting_level=nesting_level))

    else:
        padded_stride_match = PADDED_STRIDE_TAG_RE.match(tag)
        if padded_stride_match is not None:
            tag = padded_stride_match.group(1)

            from loopy.symbolic import parse
            pad_to = parse(padded_stride_match.group(2))
        else:
            pad_to = None

        if tag in ["c", "C"]:
            if nesting_level is not None:
                raise LoopyError("may not specify 'C' array order with explicit "
                        "layout nesting level")

            if ta_nesting_levels:
                nesting_level = min(ta_nesting_levels)-1
            else:
                nesting_level = 0

        elif tag in ["f", "F"]:
            if nesting_level is not None:
                raise LoopyError("may not specify 'C' array order with explicit "
                        "layout nesting level")

            if ta_nesting_levels:
                nesting_level = max(ta_nesting_levels)+1
            else:
                nesting_level = 0

        elif tag == "":
            if nesting_level is None:
                raise LoopyError("invalid dim tag: '%s'" % orig_tag)

        else:
            raise LoopyError("invalid dim tag: '%s'" % orig_tag)

        return (
                has_explicit_nesting_level, is_optional,
                ComputedStrideArrayDimTag(
                    nesting_level, pad_to=pad_to, target_axis=target_axis))


def parse_array_dim_tags(dim_tags, n_axes=None, use_increasing_target_axes=False,
        dim_names=None):
    if isinstance(dim_tags, str):
        dim_tags = dim_tags.split(",")
    if isinstance(dim_tags, dict):
        dim_tags_dict = dim_tags

        if dim_names is None:
            raise LoopyError("dim_tags may only be given as a dictionary if "
                    "dim_names is available")

        assert n_axes == len(dim_names)

        dim_tags = [None]*n_axes
        for dim_name, val in dim_tags_dict.items():
            try:
                dim_idx = dim_names.index(dim_name)
            except ValueError:
                raise LoopyError("'%s' does not name an array axis" % dim_name)

            dim_tags[dim_idx] = val

        for idim, dim_tag in enumerate(dim_tags):
            if dim_tag is None:
                raise LoopyError("array axis tag for axis %d (1-based) was not "
                        "set by passed dictionary" % (idim + 1))

    default_target_axis = 0

    result = []

    # a mapping from target axes to used nesting levels
    nesting_levels = {}

    target_axis_to_has_explicit_nesting_level = {}

    for iaxis, dim_tag in enumerate(dim_tags):
        has_explicit_nesting_level, is_optional, parsed_dim_tag = (
                _parse_array_dim_tag(
                    dim_tag, default_target_axis, nesting_levels))

        if (is_optional
                and n_axes is not None
                and len(result) + (len(dim_tags) - iaxis) > n_axes):
            continue

        if isinstance(parsed_dim_tag, _StrideArrayDimTagBase):
            # {{{ check for C/F mixed with explicit layout nesting level specs

            if (parsed_dim_tag.target_axis
                    in target_axis_to_has_explicit_nesting_level):
                if (has_explicit_nesting_level
                        != target_axis_to_has_explicit_nesting_level[
                            parsed_dim_tag.target_axis]):
                    raise LoopyError("may not mix C/F dim_tag specifications with "
                            "explicit specification of layout nesting levels")
            else:
                target_axis_to_has_explicit_nesting_level[
                        parsed_dim_tag.target_axis] = has_explicit_nesting_level

            # }}}

            lnl = parsed_dim_tag.layout_nesting_level
            target_axis = parsed_dim_tag.target_axis
            if lnl is not None:
                if lnl in nesting_levels.get(target_axis, []):
                    raise LoopyError("layout nesting level %d is not unique"
                            " in target axis %d"
                            % (lnl, target_axis))

                nesting_levels.setdefault(target_axis, []) \
                        .append(parsed_dim_tag.layout_nesting_level)

        result.append(parsed_dim_tag)

        if use_increasing_target_axes:
            default_target_axis += 1

    # {{{ check contiguity of nesting levels

    for target_axis, ta_nesting_levels in nesting_levels.items():
        if sorted(ta_nesting_levels) != list(
                range(
                    min(ta_nesting_levels),
                    min(ta_nesting_levels) + len(ta_nesting_levels))):
            raise LoopyError("layout nesting levels '%s' "
                    "for target axis %d not contiguous"
                    % (
                        ",".join(
                            str(nl)
                            for nl in ta_nesting_levels),
                        target_axis))

        ta_nesting_level_increment = -min(ta_nesting_levels)
        for i in range(len(result)):
            if (isinstance(result[i], _StrideArrayDimTagBase)
                    and result[i].target_axis == target_axis
                    and result[i].layout_nesting_level is not None):
                result[i] = result[i].copy(
                        layout_nesting_level=result[i].layout_nesting_level
                        + ta_nesting_level_increment)

    # }}}

    return result


def convert_computed_to_fixed_dim_tags(name, num_user_axes, num_target_axes,
        shape, dim_tags):

    # Just to clarify:
    #
    # - user axes are user-facing--what the user actually uses for indexing.
    #
    # - target axes are implementation facing. Normal in-memory arrays have one.
    #   3D images have three.

    import loopy as lp

    # {{{ pick apart arg dim tags into computed, fixed and vec

    vector_dim = None

    # a mapping from target axes to {layout_nesting_level: dim_tag_index}
    target_axis_to_nesting_level_map = {}

    for i, dim_tag in enumerate(dim_tags):
        if isinstance(dim_tag, VectorArrayDimTag):
            if vector_dim is not None:
                raise LoopyError("arg '%s' may only have one vector-tagged "
                        "argument dimension" % name)

            vector_dim = i

        elif isinstance(dim_tag, _StrideArrayDimTagBase):
            if dim_tag.layout_nesting_level is None:
                continue

            nl_map = target_axis_to_nesting_level_map \
                    .setdefault(dim_tag.target_axis, {})
            assert dim_tag.layout_nesting_level not in nl_map
            nl_map[dim_tag.layout_nesting_level] = i

        elif isinstance(dim_tag, SeparateArrayArrayDimTag):
            pass

        else:
            raise LoopyError("invalid array dim tag")

    # }}}

    # {{{ convert computed to fixed stride dim tags

    new_dim_tags = dim_tags[:]

    for target_axis in range(num_target_axes):
        if vector_dim is None:
            stride_so_far = 1
        else:
            if shape is None or shape is lp.auto:
                # unable to normalize without known shape
                return None

            if not is_integer(shape[vector_dim]):
                raise TypeError("shape along vector axis %d of array '%s' "
                        "must be an integer, not an expression ('%s')"
                        % (vector_dim, name, shape[vector_dim]))

            stride_so_far = shape[vector_dim]
            # FIXME: OpenCL-specific
            if stride_so_far == 3:
                stride_so_far = 4

        nesting_level_map = target_axis_to_nesting_level_map.get(target_axis, {})
        nl_keys = sorted(nesting_level_map.keys())

        if not nl_keys:
            continue

        for key in nl_keys:
            dim_tag_index = nesting_level_map[key]
            dim_tag = dim_tags[dim_tag_index]

            if isinstance(dim_tag, ComputedStrideArrayDimTag):
                if stride_so_far is None:
                    raise LoopyError("unable to determine fixed stride "
                            "for axis %d because it is nested outside of "
                            "an 'auto' stride axis"
                            % dim_tag_index)

                new_dim_tags[dim_tag_index] = FixedStrideArrayDimTag(stride_so_far,
                        target_axis=dim_tag.target_axis,
                        layout_nesting_level=dim_tag.layout_nesting_level)

                if shape is None or shape is lp.auto:
                    # unable to normalize without known shape
                    return None

                shape_axis = shape[dim_tag_index]
                if shape_axis is None:
                    stride_so_far = None
                else:
                    stride_so_far *= shape_axis

                if dim_tag.pad_to is not None:
                    from pytools import div_ceil
                    stride_so_far = (
                            div_ceil(stride_so_far, dim_tag.pad_to)
                            * stride_so_far)

            elif isinstance(dim_tag, FixedStrideArrayDimTag):
                stride_so_far = dim_tag.stride

                if stride_so_far is lp.auto:
                    stride_so_far = None

            else:
                raise TypeError("internal error in dim_tag conversion")

    # }}}

    return new_dim_tags

# }}}


# {{{ array base class (for arguments and temporary arrays)

def _pymbolic_parse_if_necessary(x):
    if isinstance(x, str):
        from pymbolic import parse
        return parse(x)
    else:
        return x


def _parse_shape_or_strides(x):
    import loopy as lp
    if x == "auto":
        warn("use of 'auto' as a shape or stride won't work "
                "any more--use loopy.auto instead",
                stacklevel=3)
    x = _pymbolic_parse_if_necessary(x)
    if isinstance(x, lp.auto):
        return x
    assert not isinstance(x, list)
    if not isinstance(x, tuple):
        assert x is not lp.auto
        x = (x,)

    return tuple(_pymbolic_parse_if_necessary(xi) for xi in x)


class ArrayBase(ImmutableRecord, Taggable):
    """
    .. attribute :: name

    .. attribute :: dtype

        The :class:`loopy.types.LoopyType` of the array. If this is *None*,
        :mod:`loopy` will try to continue without knowing the type of this
        array, where the idea is that precise knowledge of the type will become
        available at invocation time.  Calling the kernel
        (via :meth:`loopy.LoopKernel.__call__`)
        automatically adds this type information based on invocation arguments.

        Note that some transformations, such as :func:`loopy.add_padding`
        cannot be performed without knowledge of the exact *dtype*.

    .. attribute :: shape

        May be one of the following:

        * *None*. In this case, no shape is intended to be specified,
          only the strides will be used to access the array. Bounds checking
          will not be performed.

        * :class:`loopy.auto`. The shape will be determined by finding the
          access footprint.

        * a tuple like like :attr:`numpy.ndarray.shape`.

          Each entry of the tuple is also allowed to be a :mod:`pymbolic`
          expression involving kernel parameters, or a (potentially-comma
          separated) or a string that can be parsed to such an expression.

          Any element of the shape tuple not used to compute strides
          may be *None*.

    .. attribute:: dim_tags

        See :ref:`data-dim-tags`.

    .. attribute:: offset

        Offset from the beginning of the buffer to the point from
        which the strides are counted, in units of the :attr:`dtype`.
        May be one of

            * 0 or None
            * a string (that is interpreted as an argument name).
            * a pymbolic expression
            * :class:`loopy.auto`, in which case an offset argument
              is added automatically, immediately following this argument.

    .. attribute:: dim_names

        A tuple of strings providing names for the array axes, or *None*.
        If given, must have the same number of entries as :attr:`dim_tags`
        and :attr:`dim_tags`. These do not live in any particular namespace
        (i.e. collide with no other names) and serve a purely
        informational/documentational purpose. On occasion, they are used
        to generate more informative names than could be achieved by
        axis numbers.

    .. attribute:: alignment

        Memory alignment of the array in bytes. For temporary arrays,
        this ensures they are allocated with this alignment. For arguments,
        this entails a promise that the incoming array obeys this alignment
        restriction.

        Defaults to *None*.

        If an integer N is given, the array would be declared
        with ``__attribute__((aligned(N)))`` in code generation for
        :class:`loopy.CFamilyTarget`.

        .. versionadded:: 2018.1

    .. attribute:: tags

        A (possibly empty) frozenset of instances of
        :class:`pytools.tag.Tag` intended for
        consumption by an application.

        .. versionadded:: 2020.2.2

    .. automethod:: __init__
    .. automethod:: __eq__
    .. automethod:: num_user_axes
    .. automethod:: num_target_axes
    .. automethod:: vector_size

    (supports persistent hashing)
    """
    name: str
    dtype: Optional[LoopyType]
    shape: Union[ShapeType, Type["auto"], None]
    dim_tags: Optional[Sequence[ArrayDimImplementationTag]]
    offset: Union[ExpressionT, str, None]
    dim_names: Optional[Tuple[str, ...]]
    alignment: Optional[int]
    tags: FrozenSet[Tag]

    # Note that order may also wind up in attributes, if the
    # number of dimensions has not yet been determined.

    allowed_extra_kwargs: ClassVar[Tuple[str, ...]] = ()

    def __init__(self, name, dtype=None, shape=None, dim_tags=None, offset=0,
            dim_names=None, strides=None, order=None, for_atomic=False,
            alignment=None, tags=None, **kwargs):
        """
        All of the following (except *name*) are optional.
        Specify either strides or shape.

        :arg name: When passed to :class:`loopy.make_kernel`, this may contain
            multiple names separated by commas, in which case multiple arguments,
            each with identical properties, are created for each name.

        :arg shape: May be any of the things specified under :attr:`shape`,
            or a string which can be parsed into the previous form.

        :arg dim_tags: A comma-separated list of tags as understood by
            :func:`loopy.kernel.array.parse_array_dim_tags`.

        :arg strides: May be one of the following:

            * None

            * :class:`loopy.auto`. The strides will be determined by *order*
              and the access footprint.

            * a tuple like like :attr:`numpy.ndarray.shape`.

              Each entry of the tuple is also allowed to be a :mod:`pymbolic`
              expression involving kernel parameters, or a (potentially-comma
              separated) or a string that can be parsed to such an expression.

            * A string which can be parsed into the previous form.

        :arg order: "F" or "C" for C (row major) or Fortran
            (column major). Defaults to the *default_order* argument
            passed to :func:`loopy.make_kernel`.
        :arg for_atomic:
            Whether the array is declared for atomic access, and, if necessary,
            using atomic-capable data types.
        :arg offset: (See :attr:`offset`)
        :arg alignment: memory alignment in bytes
        :arg tags: An instance of or an Iterable of instances of
            :class:`pytools.tag.Tag`.
        """

        for kwarg_name in kwargs:
            if kwarg_name not in self.allowed_extra_kwargs:
                raise TypeError("invalid kwarg: %s" % kwarg_name)

        import loopy as lp

        from loopy.types import to_loopy_type
        dtype = to_loopy_type(dtype, allow_auto=True, allow_none=True,
                for_atomic=for_atomic)

        if dtype is lp.auto:
            raise ValueError("dtype may not be lp.auto")

        strides_known = strides is not None and strides is not lp.auto
        shape_known = shape is not None and shape is not lp.auto

        if strides_known:
            strides = _parse_shape_or_strides(strides)

        if shape_known:
            shape = _parse_shape_or_strides(shape)

        # {{{ check dim_names

        if dim_names is not None:
            if len(dim_names) != len(set(dim_names)):
                raise LoopyError("dim_names are not unique")

            for n in dim_names:
                if not isinstance(n, str):
                    raise LoopyError("found non-string '%s' in dim_names"
                            % type(n).__name__)

        # }}}

        # {{{ convert strides to dim_tags (Note: strides override order)

        if dim_tags is not None and strides_known:
            raise TypeError("may not specify both strides and dim_tags")

        if dim_tags is None and strides_known:
            dim_tags = [FixedStrideArrayDimTag(s) for s in strides]
            strides = None

        # }}}

        if dim_tags is not None:
            dim_tags = parse_array_dim_tags(dim_tags,
                    n_axes=(len(shape) if shape_known else None),
                    use_increasing_target_axes=self.max_target_axes > 1,
                    dim_names=dim_names)

        # {{{ determine number of user axes

        num_user_axes = None
        if shape_known:
            num_user_axes = len(shape)
        for dim_iterable in [dim_tags, dim_names]:
            if dim_iterable is not None:
                new_num_user_axes = len(dim_iterable)

                if num_user_axes is None:
                    num_user_axes = new_num_user_axes
                else:
                    if new_num_user_axes != num_user_axes:
                        raise LoopyError(
                            "contradictory values for number of dimensions of "
                            f"array '{name}' from shape, strides, dim_tags, or "
                            f"dim_names: got {new_num_user_axes} but expected "
                            f"{num_user_axes}")

                del new_num_user_axes

        # }}}

        # {{{ convert order to dim_tags

        if order is None and self.max_target_axes > 1:
            # FIXME: Hackety hack. ImageArgs need to generate dim_tags even
            # if no order is specified. Plus they don't care that much.
            order = "C"

        if dim_tags is None and num_user_axes is not None and order is not None:
            dim_tags = parse_array_dim_tags(num_user_axes*[order],
                    n_axes=num_user_axes,
                    use_increasing_target_axes=self.max_target_axes > 1,
                    dim_names=dim_names)

        if dim_tags is not None:
            order = None

        # }}}

        if dim_tags is not None:
            # {{{ find number of target axes

            target_axes = set()
            for dim_tag in dim_tags:
                if isinstance(dim_tag, _StrideArrayDimTagBase):
                    target_axes.add(dim_tag.target_axis)

            if target_axes != set(range(len(target_axes))):
                raise LoopyError("target axes for variable '%s' are non-"
                        "contiguous" % self.name)

            num_target_axes = len(target_axes)
            del target_axes

            # }}}

            if not (self.min_target_axes <= num_target_axes <= self.max_target_axes):
                raise LoopyError("%s only supports between %d and %d target axes "
                        "('%s' has %d)" % (type(self).__name__, self.min_target_axes,
                            self.max_target_axes, self.name, num_target_axes))

            new_dim_tags = convert_computed_to_fixed_dim_tags(
                    name, num_user_axes, num_target_axes,
                    shape, dim_tags)

            if new_dim_tags is not None:
                # successfully normalized
                dim_tags = new_dim_tags
                del new_dim_tags

        if dim_tags is not None:
            # for hashability
            dim_tags = tuple(dim_tags)
            order = None

        if strides is not None:
            # Preserve strides if we weren't able to process them yet.
            # That only happens if they're set to loopy.auto (and 'guessed'
            # in loopy.kernel.creation).

            kwargs["strides"] = strides

        if dim_names is not None and not isinstance(dim_names, tuple):
            warn("dim_names is not a tuple when calling ArrayBase constructor",
                    DeprecationWarning, stacklevel=2)

        if tags is None:
            tags = frozenset()

        ImmutableRecord.__init__(self,
                name=name,
                dtype=dtype,
                shape=shape,
                dim_tags=dim_tags,
                offset=offset,
                dim_names=dim_names,
                order=order,
                alignment=alignment,
                for_atomic=for_atomic,
                tags=tags,
                **kwargs)

    # Without this __hash__ is set to None because this class overrides __eq__.
    # Source: https://docs.python.org/3/reference/datamodel.html#object.__hash__
    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        from loopy.symbolic import (
                is_tuple_of_expressions_equal as istoee,
                is_expression_equal as isee)
        return (
                type(self) is type(other)
                and self.name == other.name
                and self.dtype == other.dtype
                and istoee(self.shape, other.shape)
                and self.dim_tags == other.dim_tags
                and isee(self.offset, other.offset)
                and self.dim_names == other.dim_names
                and self.order == other.order
                and self.alignment == other.alignment
                and self.for_atomic == other.for_atomic
                and self.tags == other.tags
                )

    def __ne__(self, other):
        return not self.__eq__(other)

    def _with_new_tags(self, tags):
        return self.copy(tags=tags)

    def stringify(self, include_typename):
        import loopy as lp

        info_entries = []
        if include_typename:
            info_entries.append(type(self).__name__)

        assert self.dtype is not lp.auto

        if self.dtype is None:
            type_str = "<auto/runtime>"
        else:
            type_str = str(self.dtype)

        info_entries.append("type: %s" % type_str)

        if self.shape is None:
            info_entries.append("shape: unknown")
        elif self.shape is lp.auto:
            info_entries.append("shape: auto")
        else:
            # shape is iterable
            if self.dim_names is not None:
                info_entries.append("shape: (%s)"
                        % ", ".join(
                            f"{n}:{i}"
                            for n, i in zip(self.dim_names, self.shape)))
            else:
                info_entries.append("shape: (%s)"
                        % ", ".join(str(i) for i in self.shape))

        if self.dim_tags is not None and self.dim_tags != ():
            info_entries.append("dim_tags: (%s)"
                    % ", ".join(i.stringify(self.max_target_axes > 1)
                        for i in self.dim_tags))

        if self.offset:
            info_entries.append(f"offset: {self.offset}")

        if self.tags:
            info_entries.append(
                    "tags: {%s}" % (", ".join(str(tag) for tag in self.tags)))

        return "{}: {}".format(self.name, ", ".join(info_entries))

    def __str__(self):
        return self.stringify(include_typename=True)

    def __repr__(self):
        return "<%s>" % self.__str__()

    def update_persistent_hash_for_shape(self, key_hash, key_builder, shape):
        if isinstance(shape, tuple):
            for shape_i in shape:
                if shape_i is None:
                    key_builder.rec(key_hash, shape_i)
                else:
                    key_builder.update_for_pymbolic_expression(key_hash, shape_i)
        else:
            key_builder.rec(key_hash, shape)

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        key_builder.rec(key_hash, type(self).__name__)
        key_builder.rec(key_hash, self.name)
        key_builder.rec(key_hash, self.dtype)
        self.update_persistent_hash_for_shape(key_hash, key_builder, self.shape)
        key_builder.rec(key_hash, self.dim_tags)
        key_builder.rec(key_hash, self.offset)
        key_builder.rec(key_hash, self.dim_names)
        key_builder.rec(key_hash, self.order)
        key_builder.rec(key_hash, self.alignment)
        key_builder.rec(key_hash, self.tags)

    def num_target_axes(self):
        target_axes = set()
        for dim_tag in self.dim_tags:
            if isinstance(dim_tag, _StrideArrayDimTagBase):
                target_axes.add(dim_tag.target_axis)

        return len(target_axes)

    def num_user_axes(self, require_answer=True):
        from loopy import auto
        if self.shape not in (None, auto):
            return len(self.shape)
        if self.dim_tags is not None:
            return len(self.dim_tags)
        if require_answer:
            raise LoopyError("number of user axes of array '%s' cannot be found"
                    % self.name)
        else:
            return None

    def map_exprs(self, mapper):
        """Return a copy of self with all expressions replaced with what *mapper*
        transformed them into.
        """
        changed = False
        kwargs = {}
        import loopy as lp

        if self.shape is not None and self.shape is not lp.auto:
            def none_pass_mapper(s):
                if s is None:
                    return s
                else:
                    return mapper(s)

            new_shape = tuple(none_pass_mapper(s) for s in self.shape)
            kwargs["shape"] = new_shape
            if new_shape != self.shape:
                changed = True

        if self.dim_tags is not None:
            new_dim_tags = [dim_tag.map_expr(mapper)
                    for dim_tag in self.dim_tags]
            kwargs["dim_tags"] = new_dim_tags
            if new_dim_tags != self.dim_tags:
                changed = True

        # offset is not an expression, do not map.
        if changed:
            return self.copy(**kwargs)
        else:
            return self

    def vector_size(self, target: TargetBase) -> int:
        """Return the size of the vector type used for the array
        divided by the basic data type.

        Note: For 3-vectors, this will be 4.
        """

        if self.dim_tags is None or self.shape is None:
            return 1

        assert isinstance(self.shape, tuple)
        assert isinstance(self.dtype, LoopyType)

        saw_vec_tag = False

        for i, dim_tag in enumerate(self.dim_tags):
            if isinstance(dim_tag, VectorArrayDimTag):
                if saw_vec_tag:
                    raise LoopyError("more than one axis of '{self.name}' "
                            "is tagged 'vec'")
                saw_vec_tag = True

                shape_i = self.shape[i]
                if not is_integer(shape_i):
                    raise LoopyError("shape of '%s' has non-constant-integer "
                            "length for vector axis %d (0-based)" % (
                                self.name, i))

                vec_dtype = target.vector_dtype(self.dtype, shape_i)

                return int(vec_dtype.itemsize) // int(self.dtype.itemsize)

        return 1


# }}}

def drop_vec_dims(
        dim_tags: Tuple[ArrayDimImplementationTag, ...],
        t: Tuple[T, ...]) -> Tuple[T, ...]:
    assert len(dim_tags) == len(t)
    return tuple(t_i for dim_tag, t_i in zip(dim_tags, t)
            if not isinstance(dim_tag, VectorArrayDimTag))


def get_strides(array: ArrayBase) -> Tuple[ExpressionT, ...]:
    from pymbolic import var
    result: List[ExpressionT] = []

    if array.dim_tags is None:
        return ()

    for dim_tag in array.dim_tags:
        if isinstance(dim_tag, VectorArrayDimTag):
            result.append(1)

        elif isinstance(dim_tag, FixedStrideArrayDimTag):
            if isinstance(dim_tag.stride, str):
                result.append(var(dim_tag.stride))
            else:
                result.append(dim_tag.stride)

        else:
            raise ValueError("unexpected dim tag type during stride finding: "
                    f"'{type(dim_tag)}'")

    return tuple(result)


# {{{ access code generation

@dataclass(frozen=True)
class AccessInfo(ImmutableRecord):
    array_name: str
    vector_index: Optional[int]
    subscripts: Tuple[ExpressionT, ...]


def _apply_offset(sub: ExpressionT, ary: ArrayBase) -> ExpressionT:
    """
    Helper for :func:`get_access_info`.
    Augments *ary*'s subscript index expression (*sub*) with its offset info.

    :arg ary: An instance of :class:`ArrayBase`.
    :arg array_name: Name to reference *ary* by.
    """
    import loopy as lp
    from pymbolic import var

    if ary.offset:
        from loopy.kernel.data import TemporaryVariable
        if isinstance(ary, TemporaryVariable):
            # offsets for base_storage are added when the temporary
            # is declared.
            return sub

        if ary.offset is lp.auto:
            raise AssertionError(
                    f"Offset for '{ary.name}' should have been replaced "
                    "with an actual argument by "
                    "make_temporaries_for_offsets_and_strides "
                    "during preprocessing.")
        elif isinstance(ary.offset, str):
            return var(ary.offset) + sub
        else:
            # assume it's an expression
            # FIXME: mypy can't figure out that ExpressionT + ExpressionT works
            return ary.offset + sub  # type: ignore[call-overload, arg-type, operator]  # noqa: E501
    else:
        return sub


def get_access_info(kernel: "LoopKernel",
        ary: Union["ArrayArg", "TemporaryVariable"],
        index: Union[ExpressionT, Tuple[ExpressionT, ...]],
        eval_expr: Callable[[ExpressionT], int],
        vectorization_info: "VectorizationInfo") -> AccessInfo:
    """
    :arg ary: an object of type :class:`ArrayBase`
    :arg index: a tuple of indices representing a subscript into ary
    :arg vectorization_info: an instance of :class:`loopy.codegen.VectorizationInfo`,
        or *None*.
    """

    import loopy as lp

    def eval_expr_assert_integer_constant(i, expr):
        from pymbolic.mapper.evaluator import UnknownVariableError
        try:
            result = eval_expr(expr)
        except UnknownVariableError as e:
            raise LoopyError("When trying to index the array '%s' along axis "
                    "%d (tagged '%s'), the index was not a compile-time "
                    "constant (but it has to be in order for code to be "
                    "generated). You likely want to unroll the iname(s) '%s'."
                    % (ary.name, i, ary.dim_tags[i], str(e)))

        if not is_integer(result):
            raise LoopyError("subscript '%s[%s]' has non-constant "
                    "index for separate-array axis %d (0-based)" % (
                        ary.name, index, i))

        return result

    if not isinstance(index, tuple):
        index = (index,)

    if ary.dim_tags is None:
        if len(index) != 1:
            raise LoopyError("Array '%s' has no known axis implementation "
                    "tags and therefore only supports one-dimensional "
                    "indexing. (Did you mean 'shape=loopy.auto' instead of "
                    "'shape=None'?)"
                    % ary.name)

        return AccessInfo(
                array_name=ary.name,
                subscripts=(_apply_offset(index[0], ary),),
                vector_index=None)

    if len(ary.dim_tags) != len(index):
        raise LoopyError("subscript to '%s[%s]' has the wrong "
                "number of indices (got: %d, expected: %d)" % (
                    ary.name, index, len(index), len(ary.dim_tags)))

    num_target_axes = ary.num_target_axes()

    vector_index = None
    subscripts: List[ExpressionT] = [0] * num_target_axes

    vector_size = ary.vector_size(kernel.target)

    # {{{ process separate-array dim tags first, to find array name

    from loopy.kernel.data import ArrayArg
    if isinstance(ary, ArrayArg) and ary._separation_info:
        sep_index = []
        remaining_index = []
        for iaxis, (index_i, dim_tag) in enumerate(zip(index, ary.dim_tags)):
            if iaxis in ary._separation_info.sep_axis_indices_set:
                sep_index.append(eval_expr_assert_integer_constant(iaxis, index_i))
                assert isinstance(dim_tag, SeparateArrayArrayDimTag)
            else:
                remaining_index.append(index_i)

        index = tuple(remaining_index)
        # only arguments (not temporaries) may be sep-tagged
        ary = cast(ArrayArg,
            kernel.arg_dict[ary._separation_info.subarray_names[tuple(sep_index)]])

    # }}}

    # {{{ process remaining dim tags

    assert ary.dim_tags is not None
    for i, (idx, dim_tag) in enumerate(zip(index, ary.dim_tags)):
        if isinstance(dim_tag, FixedStrideArrayDimTag):
            stride = dim_tag.stride

            if is_integer(stride):
                if not dim_tag.stride % vector_size == 0:
                    raise LoopyError("array '%s' has axis %d stride of "
                            "%d, which is not divisible by the size of the "
                            "vector (%d)"
                            % (ary.name, i, dim_tag.stride, vector_size))

            elif stride is lp.auto:
                raise AssertionError(
                        f"Stride for axis {i+1} (1-based) of "
                        "'{array_name}' should have been replaced "
                        "with an actual argument by "
                        "make_temporaries_for_offsets_and_strides "
                        "during preprocessing.")

            subscripts[dim_tag.target_axis] += (stride // vector_size)*idx

        elif isinstance(dim_tag, SeparateArrayArrayDimTag):
            raise AssertionError()

        elif isinstance(dim_tag, VectorArrayDimTag):
            from pymbolic.primitives import Variable
            index_i = index[i]
            if (vectorization_info is not None
                    and isinstance(index_i, Variable)
                    and index_i.name == vectorization_info.iname):
                # We'll do absolutely nothing here, which will result
                # in the vector being returned.
                pass

            else:
                idx = eval_expr_assert_integer_constant(i, idx)

                assert vector_index is None
                vector_index = idx

        else:
            raise LoopyError("unsupported array dim implementation tag '%s' "
                    "in array '%s'" % (dim_tag, ary.name))

    # }}}

    import loopy as lp
    if ary.offset:
        if num_target_axes > 1:
            raise NotImplementedError("offsets for multiple image axes")

        subscripts[0] = _apply_offset(subscripts[0], ary)

    return AccessInfo(
            array_name=ary.name,
            vector_index=vector_index,
            subscripts=tuple(subscripts))

# }}}

# vim: fdm=marker

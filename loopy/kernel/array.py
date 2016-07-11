"""Implementation tagging of array axes."""

from __future__ import division, absolute_import

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

import re

import six
from six.moves import range, zip
from six import iteritems

from pytools import Record, memoize_method

import numpy as np  # noqa

from loopy.diagnostic import LoopyError
from loopy.tools import is_integer


# {{{ array dimension tags

class ArrayDimImplementationTag(Record):
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

        May be None on :class:`FixedStrideArrayDimTag`, in which
        case no :class:`ComputedStrideArrayDimTag` instances may
        occur.
    """


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
        return self.copy(stride=mapper(self.stride))


class ComputedStrideArrayDimTag(_StrideArrayDimTagBase):
    """
    .. attribute:: pad_to

        :attr:`ArrayBase.dtype` granularity to which to pad this dimension

    This type of stride arg dim gets converted to :class:`FixedStrideArrayDimTag`
    on input to :class:`ArrayBase` subclasses.
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


class SeparateArrayArrayDimTag(ArrayDimImplementationTag):
    def stringify(self, include_target_axis):
        return "sep"

    def __str__(self):
        return self.stringify(True)

    def map_expr(self, mapper):
        return self


class VectorArrayDimTag(ArrayDimImplementationTag):
    def stringify(self, include_target_axis):
        return "vec"

    def __str__(self):
        return self.stringify(True)

    def map_expr(self, mapper):
        return self


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
        for dim_name, val in six.iteritems(dim_tags_dict):
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

    for target_axis, ta_nesting_levels in iteritems(nesting_levels):
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
        from warnings import warn
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


class ArrayBase(Record):
    """
    .. attribute :: name

    .. attribute :: dtype

    .. attribute :: shape

    .. attribute:: dim_tags

        See :ref:`data-dim-tags`.

    .. attribute:: offset

    .. attribute:: dim_names

        A tuple of strings providing names for the array axes, or *None*.
        If given, must have the same number of entries as :attr:`dim_tags`
        and :attr:`dim_tags`. These do not live in any particular namespace
        (i.e. collide with no other names) and serve a purely
        informational/documentational purpose. On occasion, they are used
        to generate more informative names than could be achieved by
        axis numbers.
    """

    # Note that order may also wind up in attributes, if the
    # number of dimensions has not yet been determined.

    allowed_extra_kwargs = []

    def __init__(self, name, dtype=None, shape=None, dim_tags=None, offset=0,
            dim_names=None, strides=None, order=None, for_atomic=False,
            **kwargs):
        """
        All of the following are optional. Specify either strides or shape.

        :arg name: May contain multiple names separated by
            commas, in which case multiple arguments,
            each with identical properties, are created
            for each name.
        :arg dtype: the :class:`numpy.dtype` of the array.
            If this is *None*, :mod:`loopy` will try to continue without
            knowing the type of this array, where the idea is that precise
            knowledge of the type will become available at invocation time.
            :class:`loopy.CompiledKernel` (and thereby
            :meth:`loopy.LoopKernel.__call__`) automatically add this type
            information based on invocation arguments.

            Note that some transformations, such as :func:`loopy.add_padding`
            cannot be performed without knowledge of the exact *dtype*.

        :arg shape: May be one of the following:

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

            * A string which can be parsed into the previous form.

        :arg dim_tags: A comma-separated list of tags as understood by
            :func:`parse_array_dim_tag`.

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
        :arg offset: Offset from the beginning of the buffer to the point from
            which the strides are counted. May be one of

            * 0
            * a string (that is interpreted as an argument name).
            * :class:`loopy.auto`, in which case an offset argument
              is added automatically, immediately following this argument.
              :class:`loopy.CompiledKernel` is even smarter in its treatment of
              this case and will compile custom versions of the kernel based on
              whether the passed arrays have offsets or not.
        """

        for kwarg_name in kwargs:
            if kwarg_name not in self.allowed_extra_kwargs:
                raise TypeError("invalid kwarg: %s" % kwarg_name)

        import loopy as lp

        from loopy.types import to_loopy_type
        dtype = to_loopy_type(dtype, allow_auto=True, allow_none=True,
                for_atomic=for_atomic)

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
                        raise LoopyError("contradictory values for number of "
                                "dimensions of array '%s' from shape, strides, "
                                "dim_tags, or dim_names"
                                % name)

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
            pu.db
            from warnings import warn
            warn("dim_names is not a tuple when calling ArrayBase constructor",
                    DeprecationWarning, stacklevel=2)

        Record.__init__(self,
                name=name,
                dtype=dtype,
                shape=shape,
                dim_tags=dim_tags,
                offset=offset,
                dim_names=dim_names,
                order=order,
                **kwargs)

    def __eq__(self, other):
        from loopy.symbolic import (
                is_tuple_of_expressions_equal as istoee,
                is_expression_equal as isee)
        return (
                type(self) == type(other)
                and self.name == other.name
                and self.dtype == other.dtype
                and istoee(self.shape, other.shape)
                and self.dim_tags == other.dim_tags
                and isee(self.offset, other.offset)
                and self.dim_names == other.dim_names
                and self.order == other.order
                )

    def __ne__(self, other):
        return not self.__eq__(other)

    def stringify(self, include_typename):
        import loopy as lp

        info_entries = []
        if include_typename:
            info_entries.append(type(self).__name__)

        if self.dtype is lp.auto:
            type_str = "<auto>"
        elif self.dtype is None:
            type_str = "<runtime>"
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
                            "%s:%s" % (n, i)
                            for n, i in zip(self.dim_names, self.shape)))
            else:
                info_entries.append("shape: (%s)"
                        % ", ".join(str(i) for i in self.shape))

        if self.dim_tags is not None and self.dim_tags != ():
            info_entries.append("dim_tags: (%s)"
                    % ", ".join(i.stringify(self.max_target_axes > 1)
                        for i in self.dim_tags))

        if self.offset:
            info_entries.append("offset: %s" % self.offset)

        return "%s: %s" % (self.name, ", ".join(info_entries))

    def __str__(self):
        return self.stringify(include_typename=True)

    def __repr__(self):
        return "<%s>" % self.__str__()

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        key_builder.rec(key_hash, self.name)
        key_builder.rec(key_hash, self.dtype)
        key_builder.update_for_pymbolic_expression(key_hash, self.shape)
        key_builder.rec(key_hash, self.dim_tags)
        key_builder.rec(key_hash, self.offset)
        key_builder.rec(key_hash, self.dim_names)

    @property
    @memoize_method
    def numpy_strides(self):
        return tuple(self.dtype.itemsize*s for s in self.strides)

    def num_target_axes(self):
        target_axes = set()
        for dim_tag in self.dim_tags:
            if isinstance(dim_tag, _StrideArrayDimTagBase):
                target_axes.add(dim_tag.target_axis)

        return len(target_axes)

    def num_user_axes(self, require_answer=True):
        if self.shape is not None:
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
        kwargs = {}
        import loopy as lp

        if self.shape is not None and self.shape is not lp.auto:
            def none_pass_mapper(s):
                if s is None:
                    return s
                else:
                    return mapper(s)

            kwargs["shape"] = tuple(none_pass_mapper(s) for s in self.shape)

        if self.dim_tags is not None:
            kwargs["dim_tags"] = [dim_tag.map_expr(mapper)
                    for dim_tag in self.dim_tags]

        # offset is not an expression, do not map.

        return self.copy(**kwargs)

    def vector_size(self, target):
        """Return the size of the vector type used for the array
        divided by the basic data type.

        Note: For 3-vectors, this will be 4.
        """

        if self.dim_tags is None:
            return 1

        for i, dim_tag in enumerate(self.dim_tags):
            if isinstance(dim_tag, VectorArrayDimTag):
                shape_i = self.shape[i]
                if not is_integer(shape_i):
                    raise LoopyError("shape of '%s' has non-constant-integer "
                            "length for vector axis %d (0-based)" % (
                                self.name, i))

                vec_dtype = target.vector_dtype(self.dtype, shape_i)

                return int(vec_dtype.itemsize) // int(self.dtype.itemsize)

        return 1

    def decl_info(self, target, is_written, index_dtype, shape_override=None):
        """Return a list of :class:`loopy.codegen.ImplementedDataInfo`
        instances corresponding to the array.
        """

        array_shape = self.shape
        if shape_override is not None:
            array_shape = shape_override

        from loopy.codegen import ImplementedDataInfo
        from loopy.kernel.data import ValueArg

        def gen_decls(name_suffix,
                shape, strides,
                unvec_shape, unvec_strides,
                stride_arg_axes,
                dtype, user_index):
            """
            :arg unvec_shape: shape tuple
                that accounts for :class:`loopy.kernel.array.VectorArrayDimTag`
                in a scalar manner
            :arg unvec_strides: strides tuple
                that accounts for :class:`loopy.kernel.array.VectorArrayDimTag`
                in a scalar manner
            :arg stride_arg_axes: a tuple *(user_axis, impl_axis, unvec_impl_axis)*
            :arg user_index: A tuple representing a (user-facing)
                multi-dimensional subscript. This is filled in with
                concrete integers when known (such as for separate-array
                dim tags), and with *None* where the index won't be
                known until run time.
            """

            if dtype is None:
                dtype = self.dtype

            user_axis = len(user_index)

            num_user_axes = self.num_user_axes(require_answer=False)

            if num_user_axes is None or user_axis >= num_user_axes:
                # {{{ recursion base case

                full_name = self.name + name_suffix

                stride_args = []
                strides = list(strides)
                unvec_strides = list(unvec_strides)

                # generate stride arguments, yielded later to keep array first
                for stride_user_axis, stride_impl_axis, stride_unvec_impl_axis \
                        in stride_arg_axes:
                    stride_name = full_name+"_stride%d" % stride_user_axis

                    from pymbolic import var
                    strides[stride_impl_axis] = \
                            unvec_strides[stride_unvec_impl_axis] = \
                            var(stride_name)

                    stride_args.append(
                            ImplementedDataInfo(
                                target=target,
                                name=stride_name,
                                dtype=index_dtype,
                                arg_class=ValueArg,
                                stride_for_name_and_axis=(
                                    full_name, stride_impl_axis),
                                is_written=False))

                yield ImplementedDataInfo(
                            target=target,
                            name=full_name,
                            base_name=self.name,

                            arg_class=type(self),
                            dtype=dtype,
                            shape=shape,
                            strides=tuple(strides),
                            unvec_shape=unvec_shape,
                            unvec_strides=tuple(unvec_strides),
                            allows_offset=bool(self.offset),

                            is_written=is_written)

                if self.offset:
                    offset_name = full_name+"_offset"
                    yield ImplementedDataInfo(
                                target=target,
                                name=offset_name,
                                dtype=index_dtype,
                                arg_class=ValueArg,
                                offset_for_name=full_name,
                                is_written=False)

                for sa in stride_args:
                    yield sa

                # }}}

                return

            dim_tag = self.dim_tags[user_axis]

            if isinstance(dim_tag, FixedStrideArrayDimTag):
                if array_shape is None:
                    new_shape_axis = None
                else:
                    new_shape_axis = array_shape[user_axis]

                import loopy as lp
                if dim_tag.stride is lp.auto:
                    new_stride_arg_axes = stride_arg_axes \
                            + ((user_axis, len(strides), len(unvec_strides)),)

                    # repaired above when final array name is known
                    # (and stride argument is created)
                    new_stride_axis = None
                else:
                    new_stride_arg_axes = stride_arg_axes
                    new_stride_axis = dim_tag.stride

                for res in gen_decls(name_suffix,
                        shape + (new_shape_axis,), strides + (new_stride_axis,),
                        unvec_shape + (new_shape_axis,),
                        unvec_strides + (new_stride_axis,),
                        new_stride_arg_axes,
                        dtype, user_index + (None,)):
                    yield res

            elif isinstance(dim_tag, SeparateArrayArrayDimTag):
                shape_i = array_shape[user_axis]
                if not is_integer(shape_i):
                    raise LoopyError("shape of '%s' has non-constant "
                            "integer axis %d (0-based)" % (
                                self.name, user_axis))

                for i in range(shape_i):
                    for res in gen_decls(name_suffix + "_s%d" % i,
                            shape, strides, unvec_shape, unvec_strides,
                            stride_arg_axes, dtype,
                            user_index + (i,)):
                        yield res

            elif isinstance(dim_tag, VectorArrayDimTag):
                shape_i = array_shape[user_axis]
                if not is_integer(shape_i):
                    raise LoopyError("shape of '%s' has non-constant "
                            "integer axis %d (0-based)" % (
                                self.name, user_axis))

                for res in gen_decls(name_suffix,
                        shape, strides,
                        unvec_shape + (shape_i,),
                        # vectors always have stride 1
                        unvec_strides + (1,),
                        stride_arg_axes,
                        target.vector_dtype(dtype, shape_i),
                        user_index + (None,)):
                    yield res

            else:
                raise LoopyError("unsupported array dim implementation tag '%s' "
                        "in array '%s'" % (dim_tag, self.name))

        for res in gen_decls(name_suffix="",
                shape=(), strides=(),
                unvec_shape=(), unvec_strides=(),
                stride_arg_axes=(),
                dtype=self.dtype, user_index=()):
            yield res

    @memoize_method
    def sep_shape(self):
        sep_shape = []
        for shape_i, dim_tag in zip(self.shape, self.dim_tags):
            if isinstance(dim_tag, SeparateArrayArrayDimTag):
                if not is_integer(shape_i):
                    raise TypeError("array '%s' has non-fixed-size "
                            "separate-array axis" % self.name)

                sep_shape.append(shape_i)

        return tuple(sep_shape)

    @memoize_method
    def subscripts_and_names(self):
        sep_shape = self.sep_shape()

        if not sep_shape:
            return None

        def unwrap_1d_indices(idx):
            # This allows these indices to work on Python sequences, too, not
            # just numpy arrays.

            if len(idx) == 1:
                return idx[0]
            else:
                return idx

        from pytools import indices_in_shape
        return [
                (unwrap_1d_indices(i),
                    self.name + "".join("_s%d" % sub_i for sub_i in i))
                for i in indices_in_shape(sep_shape)]

# }}}


# {{{ access code generation

class AccessInfo(Record):
    """
    .. attribute:: array_name
    .. attribute:: vector_index
    .. attribute:: subscripts
        List of expressions, one for each target axis
    """


def get_access_info(target, ary, index, eval_expr, vectorization_info):
    """
    :arg ary: an object of type :class:`ArrayBase`
    :arg index: a tuple of indices representing a subscript into ary
    :arg vectorization_info: an instance of :class:`loopy.codegen.VectorizationInfo`,
        or *None*.
    """

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

    array_name = ary.name

    if ary.dim_tags is None:
        if len(index) != 1:
            raise LoopyError("Array '%s' has no known axis implementation "
                    "tags and therefore only supports one-dimensional "
                    "indexing. (Did you mean 'shape=loopy.auto' instead of "
                    "'shape=None'?)"
                    % ary.name)

        return AccessInfo(array_name=array_name, subscripts=index, vector_index=None)

    if len(ary.dim_tags) != len(index):
        raise LoopyError("subscript to '%s[%s]' has the wrong "
                "number of indices (got: %d, expected: %d)" % (
                    ary.name, index, len(index), len(ary.dim_tags)))

    num_target_axes = ary.num_target_axes()

    vector_index = None
    subscripts = [0] * num_target_axes

    vector_size = ary.vector_size(target)

    # {{{ process separate-array dim tags first, to find array name

    for i, (idx, dim_tag) in enumerate(zip(index, ary.dim_tags)):
        if isinstance(dim_tag, SeparateArrayArrayDimTag):
            idx = eval_expr_assert_integer_constant(i, idx)
            array_name += "_s%d" % idx

    # }}}

    # {{{ process remaining dim tags

    for i, (idx, dim_tag) in enumerate(zip(index, ary.dim_tags)):
        if isinstance(dim_tag, FixedStrideArrayDimTag):
            import loopy as lp

            stride = dim_tag.stride

            if is_integer(stride):
                if not dim_tag.stride % vector_size == 0:
                    raise LoopyError("array '%s' has axis %d stride of "
                            "%d, which is not divisible by the size of the "
                            "vector (%d)"
                            % (ary.name, i, dim_tag.stride, vector_size))

            elif stride is lp.auto:
                from pymbolic import var
                stride = var(array_name + "_stride%d" % i)

            subscripts[dim_tag.target_axis] += (stride // vector_size)*idx

        elif isinstance(dim_tag, SeparateArrayArrayDimTag):
            pass

        elif isinstance(dim_tag, VectorArrayDimTag):
            from pymbolic.primitives import Variable
            if (vectorization_info is not None
                    and isinstance(index[i], Variable)
                    and index[i].name == vectorization_info.iname):
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

    from pymbolic import var
    import loopy as lp
    if ary.offset:
        if num_target_axes > 1:
            raise NotImplementedError("offsets for multiple image axes")

        offset_name = ary.offset
        if offset_name is lp.auto:
            offset_name = array_name+"_offset"

        subscripts[0] = var(offset_name) + subscripts[0]

    return AccessInfo(
            array_name=array_name,
            vector_index=vector_index,
            subscripts=subscripts)

# }}}

# vim: fdm=marker

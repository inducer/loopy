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


from pytools import MovedFunctionDeprecationWrapper
from loopy.symbolic import RuleAwareIdentityMapper, SubstitutionRuleMappingContext

from loopy.translation_unit import (for_each_kernel,
                                    TranslationUnit)
from loopy.kernel import LoopKernel
from loopy.kernel.function_interface import CallableKernel
from loopy.diagnostic import LoopyError


class SubscriptRewriter(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, arg_names, handler):
        super().__init__(rule_mapping_context)
        self.arg_names = arg_names
        self.handler = handler

    def map_subscript(self, expr, expn_state):
        if expr.aggregate.name in self.arg_names:
            return self.handler(expr)
        else:
            return super().map_subscript(expr, expn_state)

    def map_kernel(self, kernel, within=lambda *args: True):
        new_insns = [
            # While subst rules are not allowed in assignees, the mapper
            # may perform tasks entirely unrelated to subst rules, so
            # we must map assignees, too.
            insn if not kernel.substitutions and not within(kernel, insn, ()) \
            and not any(name in self.arg_names for name in \
                insn.dependency_names()) else
            self.map_instruction(kernel,
                insn.with_transformed_expressions(
                    lambda expr: self(expr, kernel, insn)))  # noqa: B023
            for insn in kernel.instructions]

        return kernel.copy(instructions=new_insns)


# {{{ split_array_dim (deprecated since June 2016)

@for_each_kernel
def split_array_dim(kernel, arrays_and_axes, count,
        auto_split_inames=True,
        split_kwargs=None):
    """
    :arg arrays_and_axes: a list of tuples *(array, axis_nr)* indicating
        that the index in *axis_nr* should be split. The tuples may
        also be *(array, axis_nr, "F")*, indicating that the index will
        be split as it would be according to Fortran order.

        *array* may name a temporary variable or an argument.

        If *arrays_and_axes* is a :class:`tuple`, it is automatically
        wrapped in a list, to make single splits easier.

    :arg count: The group size to use in the split.
    :arg auto_split_inames: Whether to automatically split inames
        encountered in the specified indices.
    :arg split_kwargs: arguments to pass to :func:`loopy.split_inames`

    Note that splits on the corresponding inames are carried out implicitly.
    The inames may *not* be split beforehand. (There's no *really* good reason
    for this--this routine is just not smart enough to deal with this.)
    """

    if count == 1:
        return kernel

    if split_kwargs is None:
        split_kwargs = {}

    # {{{ process input into array_to_rest

    # where "rest" is the non-argument-name part of the input tuples
    # in args_and_axes
    def normalize_rest(rest):
        if len(rest) == 1:
            return (rest[0], "C")
        elif len(rest) == 2:
            return rest
        else:
            raise RuntimeError("split instruction '%s' not understood" % rest)

    if isinstance(arrays_and_axes, tuple):
        arrays_and_axes = [arrays_and_axes]

    array_to_rest = {
            tup[0]: normalize_rest(tup[1:]) for tup in arrays_and_axes}

    if len(arrays_and_axes) != len(array_to_rest):
        raise RuntimeError("cannot split multiple axes of the same variable")

    del arrays_and_axes

    # }}}

    # {{{ adjust arrays

    from loopy.kernel.tools import ArrayChanger

    for array_name, (axis, order) in array_to_rest.items():
        achng = ArrayChanger(kernel, array_name)
        ary = achng.get()

        from pytools import div_ceil

        # {{{ adjust shape

        new_shape = ary.shape
        if new_shape is not None:
            new_shape = list(new_shape)
            axis_len = new_shape[axis]
            new_shape[axis] = count
            outer_len = div_ceil(axis_len, count)

            if order == "F":
                new_shape.insert(axis+1, outer_len)
            elif order == "C":
                new_shape.insert(axis, outer_len)
            else:
                raise RuntimeError("order '%s' not understood" % order)
            new_shape = tuple(new_shape)

        # }}}

        # {{{ adjust dim tags

        if ary.dim_tags is None:
            raise RuntimeError("dim_tags of '%s' are not known" % array_name)
        new_dim_tags = list(ary.dim_tags)

        old_dim_tag = ary.dim_tags[axis]

        from loopy.kernel.array import FixedStrideArrayDimTag
        if not isinstance(old_dim_tag, FixedStrideArrayDimTag):
            raise RuntimeError("axis %d of '%s' is not tagged fixed-stride"
                    % (axis, array_name))

        old_stride = old_dim_tag.stride
        outer_stride = count*old_stride

        if order == "F":
            new_dim_tags.insert(axis+1, FixedStrideArrayDimTag(outer_stride))
        elif order == "C":
            new_dim_tags.insert(axis, FixedStrideArrayDimTag(outer_stride))
        else:
            raise RuntimeError("order '%s' not understood" % order)

        new_dim_tags = tuple(new_dim_tags)

        # }}}

        # {{{ adjust dim_names

        new_dim_names = ary.dim_names
        if new_dim_names is not None:
            new_dim_names = list(new_dim_names)
            existing_name = new_dim_names[axis]
            new_dim_names[axis] = existing_name + "_inner"
            outer_name = existing_name + "_outer"

            if order == "F":
                new_dim_names.insert(axis+1, outer_name)
            elif order == "C":
                new_dim_names.insert(axis, outer_name)
            else:
                raise RuntimeError("order '%s' not understood" % order)
            new_dim_names = tuple(new_dim_names)

        # }}}

        kernel = achng.with_changed_array(ary.copy(
            shape=new_shape, dim_tags=new_dim_tags, dim_names=new_dim_names))

    # }}}

    split_vars = {}

    var_name_gen = kernel.get_var_name_generator()

    def split_access_axis(expr):
        axis_nr, order = array_to_rest[expr.aggregate.name]

        idx = expr.index
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = list(idx)

        axis_idx = idx[axis_nr]

        if auto_split_inames:
            from pymbolic.primitives import Variable
            if not isinstance(axis_idx, Variable):
                raise RuntimeError("found access '%s' in which axis %d is not a "
                        "single variable--cannot split "
                        "(Have you tried to do the split yourself, manually, "
                        "beforehand? If so, you shouldn't.)"
                        % (expr, axis_nr))

            split_iname = idx[axis_nr].name
            assert split_iname in kernel.all_inames()

            try:
                outer_iname, inner_iname = split_vars[split_iname]
            except KeyError:
                outer_iname = var_name_gen(split_iname+"_outer")
                inner_iname = var_name_gen(split_iname+"_inner")
                split_vars[split_iname] = outer_iname, inner_iname

            inner_index = Variable(inner_iname)
            outer_index = Variable(outer_iname)

        else:
            from loopy.symbolic import simplify_using_aff
            inner_index = simplify_using_aff(kernel, axis_idx % count)
            outer_index = simplify_using_aff(kernel, axis_idx // count)

        idx[axis_nr] = inner_index

        if order == "F":
            idx.insert(axis+1, outer_index)
        elif order == "C":
            idx.insert(axis, outer_index)
        else:
            raise RuntimeError("order '%s' not understood" % order)

        return expr.aggregate.index(tuple(idx))

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, var_name_gen)
    aash = SubscriptRewriter(rule_mapping_context,
            set(array_to_rest.keys()), split_access_axis)
    kernel = rule_mapping_context.finish_kernel(aash.map_kernel(kernel))

    if auto_split_inames:
        from loopy import split_iname
        for iname, (outer_iname, inner_iname) in split_vars.items():
            kernel = split_iname(kernel, iname, count,
                    outer_iname=outer_iname, inner_iname=inner_iname,
                    **split_kwargs)

    return kernel


split_arg_axis = (MovedFunctionDeprecationWrapper(split_array_dim))

# }}}


# {{{ split_array_axis

def _split_array_axis_inner(kernel, array_name, axis_nr, count, order="C"):
    if count == 1:
        return kernel

    # {{{ adjust arrays

    from loopy.kernel.tools import ArrayChanger

    achng = ArrayChanger(kernel, array_name)
    ary = achng.get()

    from pytools import div_ceil

    # {{{ adjust shape

    new_shape = ary.shape
    if new_shape is not None:
        new_shape = list(new_shape)
        axis_len = new_shape[axis_nr]
        new_shape[axis_nr] = count
        outer_len = div_ceil(axis_len, count)

        if order == "F":
            new_shape.insert(axis_nr+1, outer_len)
        elif order == "C":
            new_shape.insert(axis_nr, outer_len)
        else:
            raise RuntimeError("order '%s' not understood" % order)
        new_shape = tuple(new_shape)

    # }}}

    # {{{ adjust dim tags

    if ary.dim_tags is None:
        raise RuntimeError("dim_tags of '%s' are not known" % array_name)
    new_dim_tags = list(ary.dim_tags)

    old_dim_tag = ary.dim_tags[axis_nr]

    from loopy.kernel.array import FixedStrideArrayDimTag
    if not isinstance(old_dim_tag, FixedStrideArrayDimTag):
        raise RuntimeError("axis %d of '%s' is not tagged fixed-stride"
                % (axis_nr, array_name))

    old_stride = old_dim_tag.stride
    outer_stride = count*old_stride

    if order == "F":
        new_dim_tags.insert(axis_nr+1, FixedStrideArrayDimTag(outer_stride))
    elif order == "C":
        new_dim_tags.insert(axis_nr, FixedStrideArrayDimTag(outer_stride))
    else:
        raise RuntimeError("order '%s' not understood" % order)

    new_dim_tags = tuple(new_dim_tags)

    # }}}

    # {{{ adjust dim_names

    new_dim_names = ary.dim_names
    if new_dim_names is not None:
        new_dim_names = list(new_dim_names)
        existing_name = new_dim_names[axis_nr]
        new_dim_names[axis_nr] = existing_name + "_inner"
        outer_name = existing_name + "_outer"

        if order == "F":
            new_dim_names.insert(axis_nr+1, outer_name)
        elif order == "C":
            new_dim_names.insert(axis_nr, outer_name)
        else:
            raise RuntimeError("order '%s' not understood" % order)
        new_dim_names = tuple(new_dim_names)

    # }}}

    kernel = achng.with_changed_array(ary.copy(
        shape=new_shape, dim_tags=new_dim_tags, dim_names=new_dim_names))

    # }}}

    var_name_gen = kernel.get_var_name_generator()

    def split_access_axis(expr):
        idx = expr.index
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = list(idx)

        axis_idx = idx[axis_nr]

        from loopy.symbolic import simplify_using_aff
        inner_index = simplify_using_aff(kernel, axis_idx % count)
        outer_index = simplify_using_aff(kernel, axis_idx // count)

        idx[axis_nr] = inner_index

        if order == "F":
            idx.insert(axis_nr+1, outer_index)
        elif order == "C":
            idx.insert(axis_nr, outer_index)
        else:
            raise RuntimeError("order '%s' not understood" % order)

        return expr.aggregate.index(tuple(idx))

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, var_name_gen)
    aash = SubscriptRewriter(rule_mapping_context,
            {array_name}, split_access_axis)
    kernel = rule_mapping_context.finish_kernel(aash.map_kernel(kernel))

    return kernel


@for_each_kernel
def split_array_axis(kernel, array_names, axis_nr, count,
        order="C"):
    """
    :arg array: a list of names of temporary variables or arguments. May
        also be a comma-separated string of these.

    :arg axis_nr: the (zero-based) index of the axis that should be split.

    :arg count: The group size to use in the split.

    :arg order: The way the split array axis should be linearized.
        May be "C" or "F" to indicate C/Fortran (row/column)-major order.

    .. versionchanged:: 2016.2

        There was a more complicated, dumber function called
        ``loopy.split_array_dim`` that had the role of this function in
        versions prior to 2016.2.
    """
    assert isinstance(kernel, LoopKernel)

    if isinstance(array_names, str):
        array_names = [i.strip() for i in array_names.split(",") if i.strip()]

    for array_name in array_names:
        kernel = _split_array_axis_inner(kernel, array_name, axis_nr, count, order)

    return kernel

# }}}


# {{{ find_padding_multiple

def find_padding_multiple(kernel, variable, axis, align_bytes, allowed_waste=0.1):
    if isinstance(kernel, TranslationUnit):
        kernel_names = [i for i, clbl in kernel.callables_table.items()
                if isinstance(clbl, CallableKernel)]
        if len(kernel_names) > 1:
            raise LoopyError()
        return find_padding_multiple(kernel[kernel_names[0]], variable, axis,
                align_bytes, allowed_waste)
    assert isinstance(kernel, LoopKernel)

    arg = kernel.arg_dict[variable]

    if arg.dim_tags is None:
        raise RuntimeError("cannot find padding multiple--dim_tags of '%s' "
                "are not known" % variable)

    dim_tag = arg.dim_tags[axis]

    from loopy.kernel.array import FixedStrideArrayDimTag
    if not isinstance(dim_tag, FixedStrideArrayDimTag):
        raise RuntimeError("cannot find padding multiple--"
                "axis %d of '%s' is not tagged fixed-stride"
                % (axis, variable))

    stride = dim_tag.stride

    if not isinstance(stride, int):
        raise RuntimeError("cannot find padding multiple--stride is not a "
                "known integer")

    from pytools import div_ceil

    multiple = 1
    while True:
        true_size = multiple * stride
        padded_size = div_ceil(true_size, align_bytes) * align_bytes

        if (padded_size - true_size) / true_size <= allowed_waste:
            return multiple

        multiple += 1

# }}}


# {{{ add_padding

@for_each_kernel
def add_padding(kernel, variable, axis, align_bytes):
    arg_to_idx = {arg.name: i for i, arg in enumerate(kernel.args)}
    arg_idx = arg_to_idx[variable]

    new_args = kernel.args[:]
    arg = new_args[arg_idx]

    if arg.dim_tags is None:
        raise RuntimeError("cannot add padding--dim_tags of '%s' "
                "are not known" % variable)

    new_dim_tags = list(arg.dim_tags)
    dim_tag = new_dim_tags[axis]

    from loopy.kernel.array import FixedStrideArrayDimTag
    if not isinstance(dim_tag, FixedStrideArrayDimTag):
        raise RuntimeError("cannot find padding multiple--"
                "axis %d of '%s' is not tagged fixed-stride"
                % (axis, variable))

    stride = dim_tag.stride
    if not isinstance(stride, int):
        raise RuntimeError("cannot find split granularity--stride is not a "
                "known integer")

    from pytools import div_ceil
    new_dim_tags[axis] = FixedStrideArrayDimTag(
            div_ceil(stride, align_bytes) * align_bytes)

    new_args[arg_idx] = arg.copy(dim_tags=tuple(new_dim_tags))

    return kernel.copy(args=new_args)

# }}}


# vim: foldmethod=marker

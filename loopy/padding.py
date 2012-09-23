from __future__ import division
from loopy.symbolic import IdentityMapper




class ArgAxisSplitHelper(IdentityMapper):
    def __init__(self, arg_names, handler):
        self.arg_names = arg_names
        self.handler = handler

    def map_subscript(self, expr):
        if expr.aggregate.name in self.arg_names:
            return self.handler(expr)
        else:
            return IdentityMapper.map_subscript(self, expr)





def split_arg_axis(kernel, args_and_axes, count):
    """
    :arg args_and_axes: a list of tuples *(arg, axis_nr)* indicating
        that the index in *axis_nr* should be split. The tuples may
        also be *(arg, axis_nr, "F")*, indicating that the index will
        be split as it would according to Fortran order.

        If *args_and_axes* is a :class:`tuple`, it is automatically
        wrapped in a list, to make single splits easier.

    Note that splits on the corresponding inames are carried out implicitly.
    The inames may *not* be split beforehand.
    """

    if count == 1:
        return kernel

    def normalize_rest(rest):
        if len(rest) == 1:
            return (rest[0], "C")
        elif len(rest) == 2:
            return rest
        else:
            raise RuntimeError("split instruction '%s' not understood" % rest)

    if isinstance(args_and_axes, tuple):
        args_and_axes = [args_and_axes]

    arg_to_rest = dict((tup[0], normalize_rest(tup[1:])) for tup in args_and_axes)

    if len(args_and_axes) != len(arg_to_rest):
        raise RuntimeError("cannot split multiple axes of the same variable")

    from loopy.kernel import GlobalArg
    for arg_name in arg_to_rest:
        if not isinstance(kernel.arg_dict[arg_name], GlobalArg):
            raise RuntimeError("only GlobalArg axes may be split")

    arg_to_idx = dict((arg.name, i) for i, arg in enumerate(kernel.args))

    # {{{ adjust args

    new_args = kernel.args[:]
    for arg_name, (axis, order) in arg_to_rest.iteritems():
        arg_idx = arg_to_idx[arg_name]

        arg = new_args[arg_idx]

        from pytools import div_ceil

        # {{{ adjust shape

        new_shape = arg.shape
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

        # {{{ adjust strides

        new_strides = list(arg.strides)
        old_stride = new_strides[axis]
        outer_stride = count*old_stride

        if order == "F":
            new_strides.insert(axis+1, outer_stride)
        elif order == "C":
            new_strides.insert(axis, outer_stride)
        else:
            raise RuntimeError("order '%s' not understood" % order)

        new_strides = tuple(new_strides)

        # }}}

        new_args[arg_idx] = arg.copy(shape=new_shape, strides=new_strides)

    # }}}

    split_vars = {}

    def split_access_axis(expr):
        axis_nr, order = arg_to_rest[expr.aggregate.name]

        idx = expr.index
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = list(idx)

        axis_idx = idx[axis_nr]
        from pymbolic.primitives import Variable
        if not isinstance(axis_idx, Variable):
            raise RuntimeError("found access '%s' in which axis %d is not a "
                    "single variable--cannot split (Have you tried to do the split "
                    "yourself, manually, beforehand? If so, you shouldn't.)"
                    % (expr, axis_nr))

        split_iname = expr.index[axis_nr].name
        assert split_iname in kernel.all_inames()

        try:
            outer_iname, inner_iname = split_vars[split_iname]
        except KeyError:
            outer_iname = kernel.make_unique_var_name(
                    split_iname+"_outer")
            inner_iname = kernel.make_unique_var_name(
                    split_iname+"_inner")
            split_vars[split_iname] = outer_iname, inner_iname

        idx[axis_nr] = Variable(inner_iname)

        if order == "F":
            idx.insert(axis+1, Variable(outer_iname))
        elif order == "C":
            idx.insert(axis, Variable(outer_iname))
        else:
            raise RuntimeError("order '%s' not understood" % order)

        return expr.aggregate[tuple(idx)]

    aash = ArgAxisSplitHelper(set(arg_to_rest.iterkeys()), split_access_axis)

    result = (kernel
            .map_expressions(aash)
            .copy(args=new_args))

    from loopy import split_iname

    for iname, (outer_iname, inner_iname) in split_vars.iteritems():
        result = split_iname(result, iname, count,
                outer_iname=outer_iname, inner_iname=inner_iname)

    return result




def find_padding_multiple(kernel, variable, axis, align_bytes, allowed_waste=0.1):
    arg = kernel.arg_dict[variable]

    stride = arg.strides[axis]
    if not isinstance(stride, int):
        raise RuntimeError("cannot find padding multi--stride is not a "
                "known integer")

    from pytools import div_ceil

    multiple = 1
    while True:
        true_size = multiple * stride
        padded_size = div_ceil(true_size, align_bytes) * align_bytes

        if (padded_size - true_size) / true_size <= allowed_waste:
            return multiple

        multiple += 1




def add_padding(kernel, variable, axis, align_bytes):
    arg_to_idx = dict((arg.name, i) for i, arg in enumerate(kernel.args))
    arg_idx = arg_to_idx[variable]

    new_args = kernel.args[:]
    arg = new_args[arg_idx]

    new_strides = list(arg.strides)
    stride = new_strides[axis]
    if not isinstance(stride, int):
        raise RuntimeError("cannot find split granularity--stride is not a "
                "known integer")
    from pytools import div_ceil
    new_strides[axis] = div_ceil(stride, align_bytes) * align_bytes

    new_args[arg_idx] = arg.copy(strides=tuple(new_strides))

    return kernel.copy(args=new_args)






# vim: foldmethod=marker

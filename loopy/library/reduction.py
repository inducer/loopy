from __future__ import division

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


from pymbolic import var
import numpy as np

from loopy.symbolic import FunctionIdentifier
from loopy.diagnostic import LoopyError


class ReductionOperation(object):
    """Subclasses of this type have to be hashable, picklable, and
    equality-comparable.
    """

    def result_dtype(self, target, arg_dtype, inames):
        raise NotImplementedError

    def neutral_element(self, dtype, inames):
        raise NotImplementedError

    def __hash__(self):
        # Force subclasses to override
        raise NotImplementedError

    def __eq__(self, other):
        # Force subclasses to override
        raise NotImplementedError

    def __call__(self, dtype, operand1, operand2, inames):
        raise NotImplementedError

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def parse_result_type(target, op_type):
        try:
            return np.dtype(op_type)
        except TypeError:
            pass

        if op_type.startswith("vec_"):
            try:
                return target.get_or_register_dtype(op_type[4:])
            except AttributeError:
                pass

        raise LoopyError("unable to parse reduction type: '%s'"
                % op_type)


class ScalarReductionOperation(ReductionOperation):
    def __init__(self, forced_result_type=None):
        """
        :arg forced_result_type: Force the reduction result to be of this type.
            May be a string identifying the type for the backend under
            consideration.
        """
        self.forced_result_type = forced_result_type

    def result_dtype(self, target, arg_dtype, inames):
        if self.forced_result_type is not None:
            return self.parse_result_type(target, self.forced_result_type)

        return arg_dtype

    def __hash__(self):
        return hash((type(self), self.forced_result_type))

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.forced_result_type == other.forced_result_type)

    def __str__(self):
        result = type(self).__name__.replace("ReductionOperation", "").lower()

        if self.forced_result_type is not None:
            result = "%s<%s>" % (result, str(self.forced_result_type))

        return result


class SumReductionOperation(ScalarReductionOperation):
    def neutral_element(self, dtype, inames):
        return 0

    def __call__(self, dtype, operand1, operand2, inames):
        return operand1 + operand2


class ProductReductionOperation(ScalarReductionOperation):
    def neutral_element(self, dtype, inames):
        return 1

    def __call__(self, dtype, operand1, operand2, inames):
        return operand1 * operand2


def get_le_neutral(dtype):
    """Return a number y that satisfies (x <= y) for all y."""

    if dtype.kind == "f":
        # OpenCL 1.1, section 6.11.2
        return var("INFINITY")
    else:
        raise NotImplementedError("less")


class MaxReductionOperation(ScalarReductionOperation):
    def neutral_element(self, dtype, inames):
        return -get_le_neutral(dtype)

    def __call__(self, dtype, operand1, operand2, inames):
        return var("max")(operand1, operand2)


class MinReductionOperation(ScalarReductionOperation):
    @property
    def neutral_element(self, dtype, inames):
        return get_le_neutral(dtype)

    def __call__(self, dtype, operand1, operand2, inames):
        return var("min")(operand1, operand2)


# {{{ argmin/argmax

ARGEXT_STRUCT_DTYPES = {}


class _ArgExtremumReductionOperation(ReductionOperation):
    def prefix(self, dtype):
        return "loopy_arg%s_%s" % (self.which, dtype.type.__name__)

    def result_dtype(self, target, dtype, inames):
        try:
            return ARGEXT_STRUCT_DTYPES[dtype]
        except KeyError:
            struct_dtype = np.dtype([("value", dtype), ("index", np.int32)])
            ARGEXT_STRUCT_DTYPES[dtype] = struct_dtype

            target.get_or_register_dtype(self.prefix(dtype)+"_result", struct_dtype)
            return struct_dtype

    def neutral_element(self, dtype, inames):
        return ArgExtFunction(self, dtype, "init", inames)()

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) == type(other)

    def __call__(self, dtype, operand1, operand2, inames):
        iname, = inames

        return ArgExtFunction(self, dtype, "update", inames)(
                operand1, operand2, var(iname))


class ArgMaxReductionOperation(_ArgExtremumReductionOperation):
    which = "max"
    update_comparison = ">="
    neutral_sign = -1


class ArgMinReductionOperation(_ArgExtremumReductionOperation):
    which = "min"
    update_comparison = "<="
    neutral_sign = +1


class ArgExtFunction(FunctionIdentifier):
    init_arg_names = ("reduction_op", "scalar_dtype", "name", "inames")

    def __init__(self, reduction_op, scalar_dtype, name, inames):
        self.reduction_op = reduction_op
        self.scalar_dtype = scalar_dtype
        self.name = name
        self.inames = inames

    def __getinitargs__(self):
        return (self.reduction_op, self.scalar_dtype, self.name, self.inames)


def get_argext_preamble(target, func_id):
    op = func_id.reduction_op
    prefix = op.prefix(func_id.scalar_dtype)

    from pymbolic.mapper.c_code import CCodeMapper

    c_code_mapper = CCodeMapper()

    return (prefix, """
    typedef struct {
        %(scalar_type)s value;
        int index;
    } %(type_name)s;

    inline %(type_name)s %(prefix)s_init()
    {
        %(type_name)s result;
        result.value = %(neutral)s;
        result.index = INT_MIN;
        return result;
    }

    inline %(type_name)s %(prefix)s_update(
        %(type_name)s state, %(scalar_type)s op2, int index)
    {
        %(type_name)s result;
        if (op2 %(comp)s state.value)
        {
            result.value = op2;
            result.index = index;
            return result;
        }
        else return state;
    }
    """ % dict(
            type_name=prefix+"_result",
            scalar_type=target.dtype_to_typename(func_id.scalar_dtype),
            prefix=prefix,
            neutral=c_code_mapper(
                op.neutral_sign*get_le_neutral(func_id.scalar_dtype)),
            comp=op.update_comparison,
            ))

# }}}


# {{{ reduction op registry

_REDUCTION_OPS = {
        "sum": SumReductionOperation,
        "product": ProductReductionOperation,
        "max": MaxReductionOperation,
        "min": MinReductionOperation,
        "argmax": ArgMaxReductionOperation,
        "argmin": ArgMinReductionOperation,
        }

_REDUCTION_OP_PARSERS = [
        ]


def register_reduction_parser(parser):
    """Register a new :class:`ReductionOperation`.

    :arg parser: A function that receives a string and returns
        a subclass of ReductionOperation.
    """
    _REDUCTION_OP_PARSERS.append(parser)


def parse_reduction_op(name):
    import re

    red_op_match = re.match(r"^([a-z]+)_([a-z0-9_]+)$", name)
    if red_op_match:
        op_name = red_op_match.group(1)
        op_type = red_op_match.group(2)

        if op_name in _REDUCTION_OPS:
            return _REDUCTION_OPS[op_name](op_type)

    if name in _REDUCTION_OPS:
        return _REDUCTION_OPS[name]()

    for parser in _REDUCTION_OP_PARSERS:
        result = parser(name)
        if result is not None:
            return result

    return None

# }}}


def reduction_function_mangler(kernel, func_id, arg_dtypes):
    if isinstance(func_id, ArgExtFunction):
        from loopy.target.opencl import OpenCLTarget
        if not isinstance(kernel.target, OpenCLTarget):
            raise LoopyError("only OpenCL supported for now")

        op = func_id.reduction_op
        return (op.result_dtype(kernel.target, func_id.scalar_dtype, func_id.inames),
                "%s_%s" % (op.prefix(func_id.scalar_dtype), func_id.name))

    return None


def reduction_preamble_generator(kernel, seen_dtypes, seen_functions):
    from loopy.target.opencl import OpenCLTarget

    for func in seen_functions:
        if isinstance(func.name, ArgExtFunction):
            if not isinstance(kernel.target, OpenCLTarget):
                raise LoopyError("only OpenCL supported for now")

            yield get_argext_preamble(kernel.target, func.name)

# vim: fdm=marker

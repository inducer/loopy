from pymbolic import var
import numpy as np

from loopy.symbolic import FunctionIdentifier




class ReductionOperation(object):
    def result_dtype(self, arg_dtype, inames):
        raise NotImplementedError

    def neutral_element(self, dtype, inames):
        raise NotImplementedError

    def __call__(self, dtype, operand1, operand2, inames):
        raise NotImplementedError

class ScalarReductionOperation(ReductionOperation):
    def __init__(self, forced_result_dtype=None):
        """
        :arg forced_result_dtype: Force the reduction result to be of this type.
        """
        self.forced_result_dtype = forced_result_dtype

    def result_dtype(self, arg_dtype, inames):
        if self.forced_result_dtype is not None:
            return self.forced_result_dtype

        return arg_dtype

    def __str__(self):
        result = type(self).__name__.replace("ReductionOperation", "").lower()

        if self.forced_result_dtype is not None:
            result = "%s<%s>" % (result, str(self.dtype))

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
        return get_le_neutral(dtype)

    def __call__(self, dtype, operand1, operand2, inames):
        return var("max")(operand1, operand2)

class MinReductionOperation(ScalarReductionOperation):
    @property
    def neutral_element(self, dtype, inames):
        return -get_le_neutral(dtype)

    def __call__(self, dtype, operand1, operand2, inames):
        return var("min")(operand1, operand2)




# {{{ argmin/argmax

ARGEXT_STRUCT_DTYPES = {}

class _ArgExtremumReductionOperation(ReductionOperation):
    def prefix(self, dtype):
        return "loopy_arg%s_%s" % (self.which, dtype.type.__name__)

    def result_dtype(self, dtype, inames):
        try:
            return ARGEXT_STRUCT_DTYPES[dtype]
        except KeyError:
            struct_dtype = np.dtype([("value", dtype), ("index", np.int32)])
            ARGEXT_STRUCT_DTYPES[dtype] = struct_dtype

            from pyopencl.tools import get_or_register_dtype
            get_or_register_dtype(self.prefix(dtype)+"_result", struct_dtype)
            return struct_dtype

    def neutral_element(self, dtype, inames):
        return ArgExtFunction(self, dtype, "init", inames)()

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
    def __init__(self, reduction_op, scalar_dtype, name, inames):
        self.reduction_op = reduction_op
        self.scalar_dtype = scalar_dtype
        self.name = name
        self.inames = inames

def get_argext_preamble(func_id):
    op = func_id.reduction_op
    prefix = op.prefix(func_id.scalar_dtype)

    from pyopencl.tools import dtype_to_ctype
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
            scalar_type=dtype_to_ctype(func_id.scalar_dtype),
            prefix=prefix,
            neutral=c_code_mapper(
                op.neutral_sign*get_le_neutral(func_id.scalar_dtype)),
            comp=op.update_comparison,
            ))

# }}




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

        try:
            op_dtype = np.dtype(op_type)
        except TypeError:
            op_dtype = None

        if op_dtype is None and op_type.startswith("vec_"):
            import pyopencl.array as cl_array
            try:
                op_dtype = getattr(cl_array.vec, op_type[4:])
            except AttributeError:
                op_dtype = None

        if op_name in _REDUCTION_OPS and op_dtype is not None:
            return _REDUCTION_OPS[op_name](op_dtype)

    if name in _REDUCTION_OPS:
        return _REDUCTION_OPS[name]()

    for parser in _REDUCTION_OP_PARSERS:
        result = parser(name)
        if result is not None:
            return result

    return None

# }}}




def reduction_function_mangler(func_id, arg_dtypes):
    if isinstance(func_id, ArgExtFunction):
        op = func_id.reduction_op
        return (op.result_dtype(func_id.scalar_dtype, func_id.inames),
                "%s_%s" % (op.prefix(func_id.scalar_dtype), func_id.name))

    return None

def reduction_preamble_generator(seen_dtypes, seen_functions):
    for func_id, c_name, arg_dtypes in seen_functions:
        if isinstance(func_id, ArgExtFunction):
            yield get_argext_preamble(func_id)

# vim: fdm=marker

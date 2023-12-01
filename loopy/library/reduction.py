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


from typing import ClassVar, Tuple

from pymbolic import var
import numpy as np

from loopy.symbolic import ResolvedFunction
from loopy.kernel.function_interface import ScalarCallable
from loopy.symbolic import FunctionIdentifier
from loopy.diagnostic import LoopyError
from loopy.types import NumpyType
from loopy.tools import update_persistent_hash

__doc__ = """
.. currentmodule:: loopy.library.reduction

.. autoclass:: ReductionOperation

.. autoclass:: ScalarReductionOperation

.. autoclass:: SumReductionOperation

.. autoclass:: ProductReductionOperation

.. autoclass:: MaxReductionOperation

.. autoclass:: MinReductionOperation

.. autoclass:: ReductionOpFunction
"""


class ReductionOperation:
    """Subclasses of this type have to be hashable, picklable, and
    equality-comparable.
    """

    def result_dtypes(self, *arg_dtypes):
        """
        :arg arg_dtypes: may be None if not known
        :returns: None if not known, otherwise the returned type
        """

        raise NotImplementedError

    @property
    def arg_count(self):
        raise NotImplementedError

    def neutral_element(self, dtypes, callables_table, target):
        raise NotImplementedError

    def __hash__(self):
        # Force subclasses to override
        raise NotImplementedError

    def __eq__(self, other):
        # Force subclasses to override
        raise NotImplementedError

    def __call__(self, dtype, operand1, operand2):
        raise NotImplementedError

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return type(self).__name__

    @staticmethod
    def parse_result_type(target, op_type):
        try:
            return NumpyType(np.dtype(op_type))
        except TypeError:
            pass

        if op_type.startswith("vec_"):
            try:
                return NumpyType(target.get_or_register_dtype(op_type[4:]))
            except AttributeError:
                pass

        raise LoopyError("unable to parse reduction type: '%s'"
                % op_type)


class ScalarReductionOperation(ReductionOperation):
    @property
    def arg_count(self):
        return 1

    def result_dtypes(self, arg_dtype):
        if arg_dtype is None:
            return (None,)

        return (arg_dtype,)

    def __hash__(self):
        return hash((type(self),))

    def __eq__(self, other):
        return type(self) is type(other)

    def __str__(self):
        result = type(self).__name__.replace("ReductionOperation", "").lower()

        return result


class SumReductionOperation(ScalarReductionOperation):
    def neutral_element(self, dtype, callables_table, target):
        # FIXME: Document that we always use an int here.
        from loopy import auto
        if dtype not in [None, auto] and dtype.numpy_dtype.kind == "f":
            return 0.0, callables_table

        return 0, callables_table

    def __call__(self, dtype, operand1, operand2, callables_table, target):
        return operand1 + operand2, callables_table


class ProductReductionOperation(ScalarReductionOperation):
    def neutral_element(self, dtype, callables_table, target):
        # FIXME: Document that we always use an int here.
        from loopy import auto
        if dtype not in [None, auto] and dtype.numpy_dtype.kind == "f":
            return 1.0, callables_table

        return 1, callables_table

    def __call__(self, dtype, operand1, operand2, callables_table, target):
        return operand1 * operand2, callables_table


class AnyReductionOperation(ScalarReductionOperation):
    def neutral_element(self, dtype, callables_table, target):
        return False, callables_table

    def __call__(self, dtype, operand1, operand2, callables_table, target):
        from pymbolic.primitives import LogicalOr
        return LogicalOr((operand1, operand2)), callables_table


class AllReductionOperation(ScalarReductionOperation):
    def neutral_element(self, dtype, callables_table, target):
        return True, callables_table

    def __call__(self, dtype, operand1, operand2, callables_table, target):
        from pymbolic.primitives import LogicalAnd
        return LogicalAnd((operand1, operand2)), callables_table


def get_le_neutral(dtype):
    """Return a number y that satisfies (x <= y) for all y."""

    if dtype.numpy_dtype.kind == "f":
        # OpenCL 1.2, section 6.12.2
        if dtype.numpy_dtype.itemsize == 4:
            #float
            return var("INFINITY")
        elif dtype.numpy_dtype.itemsize == 8:
            #double
            return var("HUGE_VAL")

    elif dtype.numpy_dtype.kind == "i":
        # OpenCL 1.1, section 6.11.3
        if dtype.numpy_dtype.itemsize == 4:
            # 32 bit integer
            return var("INT_MAX")
        elif dtype.numpy_dtype.itemsize == 8:
            # 64 bit integer
            return var("LONG_MAX")
    elif dtype.numpy_dtype.kind == "u":
        if dtype.numpy_dtype.itemsize == 4:
            # 32 bit integer
            return var("UINT_MAX")
        elif dtype.numpy_dtype.itemsize == 8:
            # 64 bit integer
            return var("ULONG_MAX")

    raise NotImplementedError(f"neutral element for <= and {dtype}")


def get_ge_neutral(dtype):
    """Return a number y that satisfies (x >= y) for all y."""

    if dtype.numpy_dtype.kind == "f":
        # OpenCL 1.2, section 6.12.2
        if dtype.numpy_dtype.itemsize == 4:
            #float
            return -var("INFINITY")
        elif dtype.numpy_dtype.itemsize == 8:
            #double
            return -var("HUGE_VAL")
    elif dtype.numpy_dtype.kind == "i":
        # OpenCL 1.1, section 6.11.3
        if dtype.numpy_dtype.itemsize == 4:
            # 32 bit integer
            return var("INT_MIN")
        elif dtype.numpy_dtype.itemsize == 8:
            # 64 bit integer
            return var("LONG_MIN")
    elif dtype.numpy_dtype.kind == "u":
        return 0

    raise NotImplementedError(f"neutral element for >= and {dtype}")


class MaxReductionOperation(ScalarReductionOperation):
    def neutral_element(self, dtype, callables_table, target):
        return get_ge_neutral(dtype), callables_table

    def __call__(self, dtype, operand1, operand2, callables_table, target):
        dtype, = dtype
        from loopy.translation_unit import add_callable_to_table

        # getting the callable 'max' from target
        max_scalar_callable = target.get_device_ast_builder().known_callables["max"]

        # type specialize the callable
        max_scalar_callable, callables_table = max_scalar_callable.with_types(
                {0: dtype, 1: dtype}, callables_table)

        # populate callables_table
        func_id, callables_table = add_callable_to_table(callables_table, "max",
                max_scalar_callable)

        return ResolvedFunction(func_id)(operand1, operand2), callables_table


class MinReductionOperation(ScalarReductionOperation):
    def neutral_element(self, dtype, callables_table, target):
        return get_le_neutral(dtype), callables_table

    def __call__(self, dtype, operand1, operand2, callables_table, target):
        dtype, = dtype
        from loopy.translation_unit import add_callable_to_table

        # getting the callable 'min' from target
        min_scalar_callable = target.get_device_ast_builder().known_callables["min"]

        # type specialize the callable
        min_scalar_callable, callables_table = min_scalar_callable.with_types(
                {0: dtype, 1: dtype}, callables_table)

        # populate callables_table
        func_id, callables_table = add_callable_to_table(callables_table, "min",
                min_scalar_callable)

        return ResolvedFunction(func_id)(operand1, operand2), callables_table


# {{{ base class for symbolic reduction ops

class ReductionOpFunction(FunctionIdentifier):
    init_arg_names: ClassVar[Tuple[str, ...]] = ("reduction_op",)

    def __init__(self, reduction_op):
        self.reduction_op = reduction_op

    def __getinitargs__(self):
        return (self.reduction_op,)

    @property
    def name(self):
        return self.__class__.__name__

    def copy(self, reduction_op=None):
        if reduction_op is None:
            reduction_op = self.reduction_op

        return type(self)(reduction_op)

    hash_fields = (
            "reduction_op",)

    update_persistent_hash = update_persistent_hash

# }}}


# {{{ segmented reduction

class SegmentedOp(ReductionOpFunction):
    pass


class _SegmentedScalarReductionOperation(ReductionOperation):
    def __init__(self, **kwargs):
        self.inner_reduction = self.base_reduction_class(**kwargs)

    @property
    def base_reduction_class(self):
        raise NotImplementedError

    @property
    def which(self):
        raise NotImplementedError

    @property
    def arg_count(self):
        return 2

    def prefix(self, scalar_dtype, segment_flag_dtype):
        return "loopy_segmented_{}_{}_{}".format(self.which,
                scalar_dtype.numpy_dtype.type.__name__,
                segment_flag_dtype.numpy_dtype.type.__name__)

    def neutral_element(self, scalar_dtype, segment_flag_dtype,
            callables_table, target):
        from loopy.library.function import MakeTupleCallable
        from loopy.translation_unit import add_callable_to_table

        scalar_neutral_element, calables_table = (
                self.inner_reduction.neutral_element(
                    scalar_dtype, callables_table, target))

        make_tuple_callable = MakeTupleCallable(
                name="make_tuple")

        make_tuple_callable, callables_table = make_tuple_callable.with_types(
                dict(enumerate([scalar_dtype, segment_flag_dtype])),
                callables_table)

        func_id, callables_table = add_callable_to_table(
                callables_table, "make_tuple", make_tuple_callable)

        return ResolvedFunction(func_id)(scalar_neutral_element,
                segment_flag_dtype.numpy_dtype.type(0)), callables_table

    def result_dtypes(self, scalar_dtype, segment_flag_dtype):
        return (self.inner_reduction.result_dtypes(scalar_dtype)
                + (segment_flag_dtype,))

    def __str__(self):
        return "segmented(%s)" % self.which

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) is type(other) and (self.inner_reduction ==
                other.inner_reduction)

    def __call__(self, dtypes, operand1, operand2, callables_table, target):
        segmented_scalar_callable = SegmentOpCallable(SegmentedOp(self))

        # type specialize the callable
        segmented_scalar_callable, callables_table = (
                segmented_scalar_callable.with_types(
                    {0: dtypes[0], 1: dtypes[1], 2: dtypes[0], 3: dtypes[1]},
                    callables_table))

        # populate callables_table
        from loopy.translation_unit import add_callable_to_table
        func_id, callables_table = add_callable_to_table(
                callables_table, SegmentedOp(self), segmented_scalar_callable)

        return (ResolvedFunction(func_id)(*(operand1 + operand2)),
                callables_table)


class SegmentedSumReductionOperation(_SegmentedScalarReductionOperation):
    base_reduction_class = SumReductionOperation
    which = "sum"
    op = "((%s) + (%s))"

    hash_fields = (
            "which",
            "op",)

    update_persistent_hash = update_persistent_hash


class SegmentedProductReductionOperation(_SegmentedScalarReductionOperation):
    base_reduction_class = ProductReductionOperation
    op = "((%s) * (%s))"
    which = "product"

    hash_fields = (
            "which",
            "op",
            "base_reduction_class",)

    update_persistent_hash = update_persistent_hash

# }}}


# {{{ argmin/argmax

class ArgExtOp(ReductionOpFunction):
    pass


class _ArgExtremumReductionOperation(ReductionOperation):

    @property
    def which(self):
        raise NotImplementedError

    @property
    def neutral_sign(self):
        raise NotImplementedError

    def prefix(self, scalar_dtype, index_dtype):
        return "loopy_arg{}_{}_{}".format(self.which,
                scalar_dtype.numpy_dtype.type.__name__,
                index_dtype.numpy_dtype.type.__name__)

    def result_dtypes(self, scalar_dtype, index_dtype):
        return (scalar_dtype, index_dtype)

    def neutral_element(self, scalar_dtype, index_dtype, callables_table,
            target):
        scalar_neutral_func = (
                get_ge_neutral if self.neutral_sign < 0 else get_le_neutral)
        scalar_neutral_element = scalar_neutral_func(scalar_dtype)

        from loopy.library.function import MakeTupleCallable
        from loopy.translation_unit import add_callable_to_table
        make_tuple_callable = MakeTupleCallable(
                name="make_tuple")

        make_tuple_callable, callables_table = make_tuple_callable.with_types(
                dict(enumerate([scalar_dtype, index_dtype])),
                callables_table)

        # populate callables_table
        func_id, callables_table = add_callable_to_table(callables_table,
                                                         "make_tuple",
                                                         make_tuple_callable)

        return ResolvedFunction(func_id)(scalar_neutral_element,
                index_dtype.numpy_dtype.type(-1)), callables_table

    def __str__(self):
        return "arg" + self.which

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) is type(other)

    @property
    def arg_count(self):
        return 2

    def __call__(self, dtypes, operand1, operand2, callables_table, target):
        arg_ext_scalar_callable = ArgExtOpCallable(ArgExtOp(self))

        # type specialize the callable
        arg_ext_scalar_callable, callables_table = (
                arg_ext_scalar_callable.with_types(
                    {0: dtypes[0], 1: dtypes[1], 2: dtypes[0], 3: dtypes[1]},
                    callables_table))

        # populate callables_table
        from loopy.translation_unit import add_callable_to_table
        func_id, callables_table = add_callable_to_table(
                callables_table, ArgExtOp(self), arg_ext_scalar_callable)

        return (ResolvedFunction(func_id)(*(operand1 + operand2)),
                callables_table)


class ArgMaxReductionOperation(_ArgExtremumReductionOperation):
    which = "max"
    update_comparison = ">="
    neutral_sign = -1

    hash_fields = ("which",
            "update_comparison",
            "neutral_sign",)

    update_persistent_hash = update_persistent_hash


class ArgMinReductionOperation(_ArgExtremumReductionOperation):
    which = "min"
    update_comparison = "<="
    neutral_sign = +1

    hash_fields = ("which",
            "update_comparison",
            "neutral_sign",)

    update_persistent_hash = update_persistent_hash

# }}}


# {{{ reduction op registry

_REDUCTION_OPS = {
        "sum": SumReductionOperation,
        "product": ProductReductionOperation,
        "max": MaxReductionOperation,
        "min": MinReductionOperation,
        "any": AnyReductionOperation,
        "all": AllReductionOperation,
        "argmax": ArgMaxReductionOperation,
        "argmin": ArgMinReductionOperation,
        "segmented(sum)": SegmentedSumReductionOperation,
        "segmented(product)": SegmentedProductReductionOperation,
        }

_REDUCTION_OP_PARSERS = [
        ]


def register_reduction_parser(parser):
    """Register a new :class:`loopy.library.reduction.ReductionOperation`.

    :arg parser: A function that receives a string and returns
        a subclass of ReductionOperation.
    """
    _REDUCTION_OP_PARSERS.append(parser)


def parse_reduction_op(name):
    import re

    red_op_match = re.match(r"^([a-z]+)_([a-z0-9_]+)$", name)
    if red_op_match:
        op_name = red_op_match.group(1)

        if op_name in _REDUCTION_OPS:
            from warnings import warn
            warn("Reductions with forced result types are no longer supported. "
                    f"Encountered '{name}', which might be one.",
                    DeprecationWarning)
            return None

    if name in _REDUCTION_OPS:
        return _REDUCTION_OPS[name]()

    for parser in _REDUCTION_OP_PARSERS:
        result = parser(name)
        if result is not None:
            return result

    return None

# }}}


# {{{ reduction specific callables

class ReductionCallable(ScalarCallable):
    def with_types(self, arg_id_to_dtype, callables_table):
        scalar_dtype = arg_id_to_dtype[0]
        index_dtype = arg_id_to_dtype[1]
        result_dtypes = self.name.reduction_op.result_dtypes(scalar_dtype,
                index_dtype)
        new_arg_id_to_dtype = arg_id_to_dtype.copy()
        new_arg_id_to_dtype[-1] = result_dtypes[0]
        new_arg_id_to_dtype[-2] = result_dtypes[1]
        name_in_target = self.name.reduction_op.prefix(scalar_dtype,
                index_dtype) + "_op"

        return self.copy(arg_id_to_dtype=new_arg_id_to_dtype,
                name_in_target=name_in_target), callables_table

    def with_descrs(self, arg_id_to_descr, callables_table):
        from loopy.kernel.function_interface import ValueArgDescriptor
        new_arg_id_to_descr = arg_id_to_descr.copy()
        new_arg_id_to_descr[-1] = ValueArgDescriptor()
        return (
                self.copy(arg_id_to_descr=arg_id_to_descr),
                callables_table)


class ArgExtOpCallable(ReductionCallable):

    def generate_preambles(self, target):
        op = self.name.reduction_op
        scalar_dtype = self.arg_id_to_dtype[-1]
        index_dtype = self.arg_id_to_dtype[-2]

        prefix = op.prefix(scalar_dtype, index_dtype)

        yield (prefix, """
        inline {scalar_t} {prefix}_op(
            {scalar_t} op1, {index_t} index1,
            {scalar_t} op2, {index_t} index2,
            {index_t} *index_out)
        {{
            if (op2 {comp} op1)
            {{
                *index_out = index2;
                return op2;
            }}
            else
            {{
                *index_out = index1;
                return op1;
            }}
        }}
        """.format(
                scalar_t=target.dtype_to_typename(scalar_dtype),
                prefix=prefix,
                index_t=target.dtype_to_typename(index_dtype),
                comp=op.update_comparison,
                ))

        return


class SegmentOpCallable(ReductionCallable):

    def generate_preambles(self, target):
        op = self.name.reduction_op
        scalar_dtype = self.arg_id_to_dtype[-1]
        segment_flag_dtype = self.arg_id_to_dtype[-2]
        prefix = op.prefix(scalar_dtype, segment_flag_dtype)

        yield (prefix, """
        inline {scalar_t} {prefix}_op(
            {scalar_t} op1, {segment_flag_t} segment_flag1,
            {scalar_t} op2, {segment_flag_t} segment_flag2,
            {segment_flag_t} *segment_flag_out)
        {{
            *segment_flag_out = segment_flag1 | segment_flag2;
            return segment_flag2 ? op2 : {combined};
        }}
        """.format(
                scalar_t=target.dtype_to_typename(scalar_dtype),
                prefix=prefix,
                segment_flag_t=target.dtype_to_typename(segment_flag_dtype),
                combined=op.op % ("op1", "op2"),
                ))

        return

# }}}

# vim: fdm=marker

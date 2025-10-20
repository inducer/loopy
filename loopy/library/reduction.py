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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
from constantdict import constantdict
from typing_extensions import override

from pymbolic import ArithmeticExpression, Expression, var
from pymbolic.primitives import expr_dataclass, is_arithmetic_expression

from loopy.diagnostic import LoopyError
from loopy.kernel.function_interface import ScalarCallable
from loopy.symbolic import FunctionIdentifier, ResolvedFunction
from loopy.tools import update_persistent_hash
from loopy.types import LoopyType, NumpyType


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike

    from pytools import Hash
    from pytools.persistent_dict import KeyBuilder

    from loopy.target import TargetBase
    from loopy.translation_unit import CallablesTable


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


class ReductionOperation(ABC):
    """Subclasses of this type have to be hashable, picklable, and
    equality-comparable.
    """

    @abstractmethod
    def result_dtypes(self,
              *dtypes: LoopyType | None
          ) -> tuple[LoopyType | None, ...]:
        """
        :arg arg_dtypes: may be None if not known
        :returns: None if not known, otherwise the returned type
        """

        raise NotImplementedError

    @property
    def arg_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def neutral_element(self,
                 *dtypes: LoopyType | None,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        ...

    @override
    def __hash__(self) -> int:
        # Force subclasses to override
        raise NotImplementedError

    @override
    def __eq__(self, other: object) -> bool:
        # Force subclasses to override
        raise NotImplementedError

    @abstractmethod
    def __call__(self,
                 dtypes: Sequence[LoopyType | None],
                 operand1: Expression,
                 operand2: Expression,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        raise NotImplementedError

    @override
    def __ne__(self, other: object):
        return not self.__eq__(other)

    @override
    def __repr__(self) -> str:
        return type(self).__name__

    @staticmethod
    def parse_result_type(target: TargetBase, op_type: str | DTypeLike) -> NumpyType:
        try:
            return NumpyType(np.dtype(op_type))
        except TypeError:
            pass

        assert isinstance(op_type, str)
        if op_type.startswith("vec_"):
            try:
                return NumpyType(
                        target.get_dtype_registry().get_or_register_dtype(op_type[4:]))
            except AttributeError:
                pass

        raise LoopyError("unable to parse reduction type: '%s'"
                % op_type)


class ScalarReductionOperation(ReductionOperation, ABC):
    @property
    @override
    def arg_count(self) -> int:
        return 1

    @override
    def result_dtypes(self,
              *dtypes: LoopyType | None,
          ) -> tuple[LoopyType | None, ...]:
        arg_dtype, = dtypes
        if arg_dtype is None:
            return (None,)

        return (arg_dtype,)

    @override
    def __hash__(self):
        return hash((type(self),))

    @override
    def __eq__(self, other: object):
        return type(self) is type(other)

    @override
    def __str__(self) -> str:
        result = type(self).__name__.replace("ReductionOperation", "").lower()

        return result

    def update_persistent_hash(self, key_hash: Hash, key_builder: KeyBuilder) -> None:
        # They're all stateless.
        key_builder.rec(key_hash, type(self))


class SumReductionOperation(ScalarReductionOperation):
    @override
    def neutral_element(self,
                 *dtypes: LoopyType | None,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        dtype, = dtypes

        if dtype is not None and cast("NumpyType", dtype).numpy_dtype.kind == "f":
            return 0.0, callables_table

        return 0, callables_table

    @override
    def __call__(self,
                 dtypes: Sequence[LoopyType | None],
                 operand1: Expression,
                 operand2: Expression,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        assert not isinstance(operand1, tuple)
        assert not isinstance(operand2, tuple)
        if not is_arithmetic_expression(operand1):
            raise ValueError("operand 1 must be arithmetic")
        if not is_arithmetic_expression(operand2):
            raise ValueError("operand 2 must be arithmetic")
        return operand1 + operand2, callables_table


class ProductReductionOperation(ScalarReductionOperation):
    @override
    def neutral_element(self,
                 *dtypes: LoopyType | None,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        dtype, = dtypes

        if dtype is not None and cast("NumpyType", dtype).numpy_dtype.kind == "f":
            return 1.0, callables_table

        return 1, callables_table

    @override
    def __call__(self,
                 dtypes: Sequence[LoopyType | None],
                 operand1: Expression,
                 operand2: Expression,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        assert not isinstance(operand1, tuple)
        assert not isinstance(operand2, tuple)
        if not is_arithmetic_expression(operand1):
            raise ValueError("operand 1 must be arithmetic")
        if not is_arithmetic_expression(operand2):
            raise ValueError("operand 2 must be arithmetic")
        return operand1 * operand2, callables_table


class AnyReductionOperation(ScalarReductionOperation):
    @override
    def neutral_element(self,
                 *dtypes: LoopyType | None,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        return False, callables_table

    @override
    def __call__(self,
                 dtypes: Sequence[LoopyType | None],
                 operand1: Expression,
                 operand2: Expression,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        from pymbolic.primitives import LogicalOr
        return LogicalOr((operand1, operand2)), callables_table


class AllReductionOperation(ScalarReductionOperation):
    @override
    def neutral_element(self,
                 *dtypes: LoopyType | None,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        return True, callables_table

    @override
    def __call__(self,
                 dtypes: Sequence[LoopyType | None],
                 operand1: Expression,
                 operand2: Expression,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        from pymbolic.primitives import LogicalAnd
        return LogicalAnd((operand1, operand2)), callables_table


def get_le_neutral(dtype: LoopyType) -> ArithmeticExpression:
    """Return a number y that satisfies (x <= y) for all y."""

    assert isinstance(dtype, NumpyType)

    if dtype.numpy_dtype.kind == "f":
        # OpenCL 1.2, section 6.12.2
        if dtype.numpy_dtype.itemsize == 4:
            # float
            return var("INFINITY")
        elif dtype.numpy_dtype.itemsize == 8:
            # double
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


def get_ge_neutral(dtype: LoopyType) -> ArithmeticExpression:
    """Return a number y that satisfies (x >= y) for all y."""

    assert isinstance(dtype, NumpyType)

    if dtype.numpy_dtype.kind == "f":
        # OpenCL 1.2, section 6.12.2
        if dtype.numpy_dtype.itemsize == 4:
            # float
            return -var("INFINITY")
        elif dtype.numpy_dtype.itemsize == 8:
            # double
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
    @override
    def neutral_element(self,
                 *dtypes: LoopyType | None,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        dtype, = dtypes
        assert dtype is not None
        return get_ge_neutral(dtype), callables_table

    @override
    def __call__(self,
                 dtypes: Sequence[LoopyType | None],
                 operand1: Expression,
                 operand2: Expression,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        dtype, = dtypes
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
    @override
    def neutral_element(self,
                 *dtypes: LoopyType | None,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        dtype, = dtypes
        assert dtype is not None
        return get_le_neutral(dtype), callables_table

    @override
    def __call__(self,
                 dtypes: Sequence[LoopyType | None],
                 operand1: Expression,
                 operand2: Expression,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        dtype, = dtypes
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

@expr_dataclass()
class ReductionOpFunction(FunctionIdentifier):
    reduction_op: ReductionOperation

    @property
    def name(self):
        return self.__class__.__name__

    def copy(self, reduction_op=None):
        if reduction_op is None:
            reduction_op = self.reduction_op

        return type(self)(reduction_op)

# }}}


# {{{ segmented reduction

class SegmentedOp(ReductionOpFunction):
    pass


class _SegmentedScalarReductionOperation(ReductionOperation):
    inner_reduction: ReductionOperation
    base_reduction_class: ClassVar[type[ReductionOperation]]

    def __init__(self, **kwargs):
        self.inner_reduction = self.base_reduction_class(**kwargs)

    @property
    def which(self) -> str:
        raise NotImplementedError

    @property
    def arg_count(self):
        return 2

    def prefix(self, scalar_dtype, segment_flag_dtype):
        stype = scalar_dtype.numpy_dtype.type.__name__
        ftype = segment_flag_dtype.numpy_dtype.type.__name__
        return f"loopy_segmented_{self.which}_{stype}_{ftype}"

    @override
    def neutral_element(self,
                 *dtypes: LoopyType | None,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        scalar_dtype, segment_flag_dtype = dtypes

        from loopy.library.function import MakeTupleCallable
        from loopy.translation_unit import add_callable_to_table

        scalar_neutral_element, _calables_table = (
                self.inner_reduction.neutral_element(
                    scalar_dtype, callables_table=callables_table, target=target))

        make_tuple_callable = MakeTupleCallable(
                name="make_tuple")

        make_tuple_callable, callables_table = make_tuple_callable.with_types(
                dict(enumerate([scalar_dtype, segment_flag_dtype])),
                callables_table)

        func_id, callables_table = add_callable_to_table(
                callables_table, "make_tuple", make_tuple_callable)

        return ResolvedFunction(func_id)(
                scalar_neutral_element,
                cast("NumpyType", segment_flag_dtype).numpy_dtype.type(0)
            ), callables_table

    @override
    def result_dtypes(self,
              *dtypes: LoopyType | None
          ) -> tuple[LoopyType | None, ...]:
        scalar_dtype, segment_flag_dtype = dtypes

        return ((*self.inner_reduction.result_dtypes(scalar_dtype), segment_flag_dtype))

    @override
    def __str__(self):
        return "segmented(%s)" % self.which

    @override
    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other: object):
        return type(self) is type(other) and (self.inner_reduction ==
                cast("_SegmentedScalarReductionOperation", other).inner_reduction)

    @override
    def __call__(self,
                 dtypes: Sequence[LoopyType | None],
                 operand1: Expression,
                 operand2: Expression,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        assert isinstance(operand1, tuple)
        assert isinstance(operand2, tuple)

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

        return (ResolvedFunction(func_id)(*operand1, *operand2),
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

@expr_dataclass()
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
        stype = scalar_dtype.numpy_dtype.type.__name__
        itype = index_dtype.numpy_dtype.type.__name__
        return f"loopy_arg{self.which}_{stype}_{itype}"

    @override
    def result_dtypes(self,
              *dtypes: LoopyType | None
          ) -> tuple[LoopyType | None, ...]:
        scalar_dtype, index_dtype = dtypes
        return (scalar_dtype, index_dtype)

    @override
    def neutral_element(self,
                 *dtypes: LoopyType | None,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        scalar_dtype, index_dtype = dtypes
        assert scalar_dtype is not None
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

        # FIXME: This doesn't handle None
        return ResolvedFunction(func_id)(scalar_neutral_element,
                cast("NumpyType", index_dtype).numpy_dtype.type(-1)), callables_table

    @override
    def __str__(self):
        return "arg" + self.which

    @override
    def __hash__(self):
        return hash(type(self))

    @override
    def __eq__(self, other):
        return type(self) is type(other)

    @property
    @override
    def arg_count(self):
        return 2

    @override
    def __call__(self,
                 dtypes: Sequence[LoopyType | None],
                 operand1: Expression,
                 operand2: Expression,
                 callables_table: CallablesTable,
                 target: TargetBase,
             ) -> tuple[Expression, CallablesTable]:
        assert isinstance(operand1, tuple)
        assert isinstance(operand2, tuple)

        scalar_dtype, index_dtype = dtypes

        arg_ext_scalar_callable = ArgExtOpCallable(ArgExtOp(self))

        # type specialize the callable
        arg_ext_scalar_callable, callables_table = (
                arg_ext_scalar_callable.with_types(
                    {0: scalar_dtype, 1: index_dtype, 2: scalar_dtype, 3: index_dtype},
                    callables_table))

        # populate callables_table
        from loopy.translation_unit import add_callable_to_table
        func_id, callables_table = add_callable_to_table(
                callables_table, ArgExtOp(self), arg_ext_scalar_callable)

        return (ResolvedFunction(func_id)(*operand1, *operand2), callables_table)


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


def parse_reduction_op(name: str) -> ReductionOperation | None:
    import re

    red_op_match = re.match(r"^([a-z]+)_([a-z0-9_]+)$", name)
    if red_op_match:
        op_name = red_op_match.group(1)

        if op_name in _REDUCTION_OPS:
            from warnings import warn
            warn("Reductions with forced result types are no longer supported. "
                    f"Encountered '{name}', which might be one.",
                    DeprecationWarning, stacklevel=1)
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
        result_dtypes = self.name.reduction_op.result_dtypes(scalar_dtype,  # pylint: disable=no-member
                index_dtype)

        new_arg_id_to_dtype = constantdict(arg_id_to_dtype).mutate()
        new_arg_id_to_dtype[-1] = result_dtypes[0]
        new_arg_id_to_dtype[-2] = result_dtypes[1]
        name_in_target = self.name.reduction_op.prefix(scalar_dtype,  # pylint: disable=no-member
                index_dtype) + "_op"

        return (self.copy(arg_id_to_dtype=new_arg_id_to_dtype.finish(),
                          name_in_target=name_in_target),
                callables_table)

    def with_descrs(self, arg_id_to_descr, callables_table):
        from loopy.kernel.function_interface import ValueArgDescriptor

        new_arg_id_to_descr = constantdict(arg_id_to_descr).mutate()
        new_arg_id_to_descr[-1] = ValueArgDescriptor()

        return (
                self.copy(arg_id_to_descr=new_arg_id_to_descr.finish()),
                callables_table)


class ArgExtOpCallable(ReductionCallable):

    def generate_preambles(self, target):
        op = self.name.reduction_op  # pylint: disable=no-member
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
        op = self.name.reduction_op  # pylint: disable=no-member
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

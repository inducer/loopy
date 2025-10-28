"""Pymbolic mappers for loopy."""

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

import contextlib
import re
from collections import defaultdict
from collections.abc import Set
from dataclasses import dataclass, replace
from functools import cached_property, reduce
from sys import intern
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Concatenate,
    Literal as TypingLiteral,
    TypeAlias,
    TypeVar,
    cast,
)
from warnings import warn

import numpy as np
from constantdict import constantdict
from typing_extensions import Self, override

import islpy as isl
import pymbolic.primitives as p
import pytools.lex
from pymbolic.mapper import (
    CachedCombineMapper as CombineMapperBase,
    CachedIdentityMapper as IdentityMapperBase,
    CachedWalkMapper as WalkMapperBase,
    CallbackMapper as CallbackMapperBase,
    IdentityMapper as UncachedIdentityMapperBase,
    Mapper,
    P,
    ResultT,
    WalkMapper as UncachedWalkMapperBase,
)
from pymbolic.mapper.coefficient import (
    CoefficientCollector as CoefficientCollectorBase,
    CoeffsT,
)
from pymbolic.mapper.constant_folder import (
    ConstantFoldingMapper as ConstantFoldingMapperBase,
)
from pymbolic.mapper.dependency import (
    CachedDependencyMapper as DependencyMapperBase,
)
from pymbolic.mapper.evaluator import CachedEvaluationMapper as EvaluationMapperBase
from pymbolic.mapper.flattener import FlattenMapper as FlattenMapperBase
from pymbolic.mapper.stringifier import StringifyMapper as StringifyMapperBase
from pymbolic.mapper.substitutor import (
    CachedSubstitutionMapper as SubstitutionMapperBase,
)
from pymbolic.mapper.unifier import (
    UnidirectionalUnifier as UnidirectionalUnifierBase,
    UnificationRecord,
)
from pymbolic.parser import Parser as ParserBase
from pymbolic.typing import (
    ArithmeticExpression,
    ArithmeticOrExpressionT,
    Expression,
)
from pytools import memoize, memoize_method, memoize_on_first_arg
from pytools.tag import Tag, Taggable, ToTagSetConvertible

from loopy.diagnostic import (
    ExpressionToAffineConversionError,
    LoopyError,
    UnableToDetermineAccessRangeError,
)
from loopy.typing import InsnId, ShapeType, not_none


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

    from pymbolic.mapper.dependency import Dependencies

    from loopy.kernel import LoopKernel
    from loopy.kernel.data import KernelArgument, SubstitutionRule, TemporaryVariable
    from loopy.kernel.instruction import InstructionBase
    from loopy.library.reduction import ReductionOperation, ReductionOpFunction
    from loopy.match import ConcreteMatchable, RuleStack, StackMatch
    from loopy.types import LoopyType, NumpyType, ToLoopyTypeConvertible


__doc__ = """
Loopy-specific expression types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Literal
.. autoclass:: ArrayLiteral
.. autoclass:: FunctionIdentifier
.. autoclass:: TypedCSE

.. currentmodule:: loopy

.. autoclass:: TypeCast
.. autoclass:: TaggedVariable
.. autoclass:: Reduction
.. autoclass:: LinearSubscript

.. currentmodule:: loopy.symbolic
.. autoclass:: SubArrayRef


.. autoclass:: RuleArgument
.. autoclass:: ResolvedFunction

Rule-aware Mappers
^^^^^^^^^^^^^^^^^^

.. autoclass:: SubstitutionRuleMappingContext
.. autoclass:: ExpansionState
.. autoclass:: RuleAwareIdentityMapper


Expression Manipulation Helpers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: simplify_using_aff
"""


T = TypeVar("T")


# {{{ mappers with support for loopy-specific primitives

class IdentityMapperMixin(Mapper[Expression, P]):
    def map_literal(self, expr: Literal,
                    *args: P.args, **kwargs: P.kwargs) -> Expression:
        return expr

    def map_array_literal(self, expr: ArrayLiteral,
                          *args: P.args, **kwargs: P.kwargs) -> Expression:
        return type(expr)(tuple(self.rec(ch, *args, **kwargs)
                                for ch in expr.children))

    def map_group_hw_index(self, expr: GroupHardwareAxisIndex | RuleArgument,
                           *args: P.args, **kwargs: P.kwargs) -> Expression:
        return expr

    def map_local_hw_index(self, expr: LocalHardwareAxisIndex,
                           *args: P.args, **kwargs: P.kwargs) -> Expression:
        return expr

    def map_loopy_function_identifier(self, expr: FunctionIdentifier,
                                      *args: P.args, **kwargs: P.kwargs) -> Expression:
        return expr

    def map_reduction(self, expr: Reduction,
                      *args: P.args, **kwargs: P.kwargs) -> Expression:
        mapped_inames = [self.rec(p.Variable(iname), *args, **kwargs)
                         for iname in expr.inames]

        new_inames: list[str] = []
        for iname, new_sym_iname in zip(expr.inames, mapped_inames, strict=True):
            if not isinstance(new_sym_iname, p.Variable):
                from loopy.diagnostic import LoopyError
                raise LoopyError("%s did not map iname '%s' to a variable"
                        % (type(self).__name__, iname))

            new_inames.append(new_sym_iname.name)

        new_expr = self.rec(expr.expr, *args, **kwargs)
        if new_expr is expr.expr and new_inames == expr.inames:
            return expr

        return Reduction(
                expr.operation, tuple(new_inames),
                new_expr,
                allow_simultaneous=expr.allow_simultaneous)

    def map_tagged_variable(self, expr: TaggedVariable,
                            *args: P.args, **kwargs: P.kwargs) -> Expression:
        # leaf, doesn't change
        return expr

    def map_type_annotation(self, expr: TypeAnnotation,
                            *args: P.args, **kwargs: P.kwargs) -> Expression:
        new_child = self.rec(expr.child, *args, **kwargs)

        if new_child is expr.child:
            return expr

        return type(expr)(expr.type, new_child)

    def map_sub_array_ref(self, expr: SubArrayRef,
                          *args: P.args, **kwargs: P.kwargs) -> Expression:
        new_inames = self.rec(expr.swept_inames, *args, **kwargs)
        new_subscript = self.rec(expr.subscript, *args, **kwargs)

        assert isinstance(new_inames, tuple)
        assert isinstance(new_subscript, p.Subscript)
        if (all(new_iname is old_iname
                for new_iname, old_iname in
                zip(new_inames, expr.swept_inames, strict=True))
            and new_subscript is expr.subscript):
            return expr

        return SubArrayRef(cast("tuple[p.Variable, ...]", new_inames), new_subscript)

    def map_resolved_function(self, expr: ResolvedFunction,
                              *args: P.args, **kwargs: P.kwargs) -> Expression:
        # leaf, doesn't change
        return expr

    def map_type_cast(self, expr: TypeCast,
                      *args: P.args, **kwargs: P.kwargs) -> Expression:
        new_child = self.rec(expr.child, *args, **kwargs)

        if new_child is expr.child:
            return expr

        return type(expr)(expr.type, new_child)

    map_linear_subscript: Callable[Concatenate[Self, LinearSubscript, P],
                                   Expression] = IdentityMapperBase.map_subscript
    map_rule_argument: Callable[Concatenate[Self, RuleArgument, P],
                                Expression] = map_group_hw_index
    map_fortran_division: Callable[Concatenate[Self, FortranDivision, P],
                                   Expression] = IdentityMapperBase.map_quotient


class FlattenMapper(FlattenMapperBase, IdentityMapperMixin[[]]):
    # FIXME(pyright): Lies! This needs to be made precise.
    @override
    def is_expr_integer_valued(self, expr: Expression) -> bool:
        return True


def flatten(expr: ArithmeticOrExpressionT) -> ArithmeticOrExpressionT:
    return cast("ArithmeticOrExpressionT", FlattenMapper()(expr))


class IdentityMapper(IdentityMapperBase[P], IdentityMapperMixin[P]):
    pass


class UncachedIdentityMapper(UncachedIdentityMapperBase[P],
                             IdentityMapperMixin[P]):
    pass


class PartialEvaluationMapper(
        EvaluationMapperBase[Expression], IdentityMapperMixin[[]]):
    @override
    def map_variable(self, expr: p.Variable) -> Expression:
        return expr

    @override
    def map_common_subexpression_uncached(
                self, expr: p.CommonSubexpression) -> Expression:
        return type(expr)(self.rec(expr.child), expr.prefix, expr.scope)


class WalkMapperMixin(WalkMapperBase[P]):
    def map_literal(self, expr: Literal,
                    *args: P.args, **kwargs: P.kwargs) -> None:
        self.visit(expr, *args, **kwargs)

    def map_array_literal(self, expr: ArrayLiteral,
                          *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for ch in expr.children:
            self.rec(ch, *args, **kwargs)

    def map_group_hw_index(self, expr: GroupHardwareAxisIndex | RuleArgument,
                           *args: P.args, **kwargs: P.kwargs) -> None:
        self.visit(expr, *args, **kwargs)

    def map_local_hw_index(self, expr: LocalHardwareAxisIndex,
                           *args: P.args, **kwargs: P.kwargs) -> None:
        self.visit(expr, *args, **kwargs)

    def map_reduction(self, expr: Reduction,
                      *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.expr, *args, **kwargs)

    def map_type_cast(self, expr: TypeCast,
                      *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return
        self.rec(expr.child, *args, **kwargs)

    map_tagged_variable: Callable[Concatenate[Self, TaggedVariable, P],
                                  None] = WalkMapperBase.map_variable

    def map_loopy_function_identifier(self, expr: FunctionIdentifier,
                                      *args: P.args, **kwargs: P.kwargs) -> None:
        self.visit(expr, *args, **kwargs)

    def map_linear_subscript(self, expr: LinearSubscript,
                             *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.aggregate, *args, **kwargs)
        self.rec(expr.index, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    map_rule_argument: Callable[Concatenate[Self, RuleArgument, P],
                                None] = map_group_hw_index

    def map_sub_array_ref(self, expr: SubArrayRef,
                          *args: P.args, **kwargs: P.kwargs):
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.swept_inames, *args, **kwargs)
        self.rec(expr.subscript, *args, **kwargs)

    def map_resolved_function(self, expr: ResolvedFunction,
                              *args: P.args, **kwargs: P.kwargs):
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.function, *args, **kwargs)

    map_fortran_division: Callable[Concatenate[Self, FortranDivision, P],
                                   None] = WalkMapperBase.map_quotient


class WalkMapper(WalkMapperMixin[P], WalkMapperBase[P]):
    pass


class UncachedWalkMapper(WalkMapperMixin[P], UncachedWalkMapperBase[P]):
    pass


class CallbackMapper(IdentityMapperMixin[P],
                     CallbackMapperBase[Expression, P]):
    map_reduction: Callable[Concatenate[Self, Reduction, P],
                            Expression] = CallbackMapperBase.map_constant
    map_resolved_function: Callable[Concatenate[Self, ResolvedFunction, P],
                                    Expression] = CallbackMapperBase.map_constant


class CombineMapper(CombineMapperBase[ResultT, P]):
    def map_reduction(self, expr: Reduction,
                      *args: P.args, **kwargs: P.kwargs) -> ResultT:
        return self.rec(expr.expr, *args, **kwargs)

    def map_type_cast(self, expr: TypeCast, /,
                      *args: P.args, **kwargs: P.kwargs) -> ResultT:
        return self.rec(expr.child, *args, **kwargs)

    def map_sub_array_ref(self, expr: SubArrayRef,
                          *args: P.args, **kwargs: P.kwargs) -> ResultT:
        return self.combine((
            self.rec(expr.subscript, *args, **kwargs),
            self.combine(tuple(
                         self.rec(idx, *args, **kwargs)
                         for idx in expr.swept_inames))))

    def map_linear_subscript(self, expr: LinearSubscript,
                             *args: P.args, **kwargs: P.kwargs) -> ResultT:
        return self.combine(
                [self.rec(expr.aggregate, *args, **kwargs),
                    self.rec(expr.index, *args, **kwargs)])

    def map_fortran_division(self, expr: FortranDivision,
                             *args: P.args, **kwargs: P.kwargs) -> ResultT:
        return self.combine((
            self.rec(expr.numerator, *args, **kwargs),
            self.rec(expr.denominator, *args, **kwargs)))


class SubstitutionMapper(
        SubstitutionMapperBase, IdentityMapperMixin[[]]):
    def map_common_subexpression_uncached(
                self, expr: p.CommonSubexpression) -> Expression:
        return type(expr)(self.rec(expr.child), expr.prefix, expr.scope)


class ConstantFoldingMapper(ConstantFoldingMapperBase, IdentityMapperMixin[[]]):
    pass


class StringifyMapper(StringifyMapperBase[[]]):
    def map_literal(self, expr: Literal, enclosing_prec: int) -> str:
        return expr.s

    def map_array_literal(self, expr: ArrayLiteral, enclosing_prec: int) -> str:
        from pymbolic.mapper.stringifier import PREC_NONE
        return "{%s}" % ", ".join(self.rec(ch, PREC_NONE) for ch in expr.children)

    def map_group_hw_index(self, expr: GroupHardwareAxisIndex,
                           enclosing_prec: int) -> str:
        return "grp.%d" % expr.axis

    def map_local_hw_index(self, expr: LocalHardwareAxisIndex,
                           enclosing_prec: int) -> str:
        return "loc.%d" % expr.axis

    def map_reduction(self, expr: Reduction, enclosing_prec: int) -> str:
        from pymbolic.mapper.stringifier import PREC_NONE

        return "{}reduce({}, [{}], {})".format(
                "simul_" if expr.allow_simultaneous else "",
                expr.operation, ", ".join(expr.inames),
                self.rec(expr.expr, PREC_NONE))

    def map_tagged_variable(self, expr: TaggedVariable, enclosing_prec: int) -> str:
        return f"{expr.name}${{{', '.join(str(t) for t in expr.tags)}}}"

    def map_linear_subscript(self, expr: LinearSubscript, enclosing_prec: int) -> str:
        from pymbolic.mapper.stringifier import PREC_CALL, PREC_NONE
        return self.parenthesize_if_needed(
                self.format("%s[[%s]]",
                    self.rec(expr.aggregate, PREC_CALL),
                    self.rec(expr.index, PREC_NONE)),
                enclosing_prec, PREC_CALL)

    def map_loopy_function_identifier(
            self, expr: FunctionIdentifier, enclosing_prec: int) -> str:
        from dataclasses import fields
        initargs = tuple(getattr(self, fld.name) for fld in fields(expr))

        return "{}<{}>".format(type(expr).__name__, ", ".join(str(a) for a in initargs))

    def map_rule_argument(self, expr: RuleArgument, enclosing_prec: int) -> str:
        return "<arg%d>" % expr.index

    def map_type_cast(self, expr: TypeCast, enclosing_prec: int) -> str:
        from pymbolic.mapper.stringifier import PREC_NONE
        child = self.rec(expr.child, PREC_NONE)
        return f"cast({expr.type!r}, {child})"

    def map_resolved_function(self, expr: ResolvedFunction, enclosing_prec: int) -> str:
        # underlining a resolved call
        return "\u0332".join(str(expr.function))

    def map_sub_array_ref(self, expr: SubArrayRef, enclosing_prec: int) -> str:
        return "[{inames}]: {subscr}".format(
                inames=",".join(self.rec(iname, enclosing_prec) for iname in
                    expr.swept_inames),
                subscr=self.rec(expr.subscript, enclosing_prec))

    def map_fortran_division(self, expr: FortranDivision, enclosing_prec: int) -> str:
        from pymbolic.mapper.stringifier import PREC_NONE
        result = self.map_quotient(expr, PREC_NONE)  # pyright: ignore[reportArgumentType]
        return f"[FORTRANDIV]({result})"


class UnidirectionalUnifier(UnidirectionalUnifierBase):
    def map_reduction(
                self,
                expr: Reduction,
                other: Expression,
                urecs: Sequence[UnificationRecord]) -> Sequence[UnificationRecord]:
        if not isinstance(other, type(expr)):
            return self.treat_mismatch(expr, other, urecs)

        if (expr.inames != other.inames
                or type(expr.operation) is not type(other.operation)):
            return []

        return self.rec(expr.expr, other.expr, urecs)

    def map_tagged_variable(
                self,
                expr: TaggedVariable,
                other: Expression,
                urecs: Sequence[UnificationRecord]) -> Sequence[UnificationRecord]:
        new_uni_record = self.unification_record_from_equation(expr, other)
        if new_uni_record is None:
            # Check if the variables match literally--that's ok, too.
            if (isinstance(other, TaggedVariable)
                    and expr.name == other.name
                    and expr.tags == other.tags
                    and (
                        self.lhs_mapping_candidates is None
                        or expr.name not in self.lhs_mapping_candidates)):
                return urecs
            else:
                return []
        else:
            from pymbolic.mapper.unifier import unify_many
            return unify_many(urecs, new_uni_record)


class DependencyMapper(DependencyMapperBase[P]):
    def map_group_hw_index(self, expr: GroupHardwareAxisIndex,
                           *args: P.args, **kwargs: P.kwargs) -> Dependencies:
        return set()

    def map_local_hw_index(self, expr: LocalHardwareAxisIndex,
                           *args: P.args, **kwargs: P.kwargs) -> Dependencies:
        return set()

    @override
    def map_call(self, expr: p.Call,
                 *args: P.args, **kwargs: P.kwargs) -> Dependencies:
        # Loopy does not have first-class functions. Do not descend
        # into 'function' attribute of Call.
        return self.rec(expr.parameters, *args, **kwargs)

    def map_reduction(self, expr: Reduction,
                      *args: P.args, **kwargs: P.kwargs) -> Dependencies:
        deps = self.rec(expr.expr, *args, **kwargs)
        return deps - {p.Variable(iname) for iname in expr.inames}

    def map_tagged_variable(self, expr: TaggedVariable,
                            *args: P.args, **kwargs: P.kwargs) -> Dependencies:
        return {expr}

    def map_loopy_function_identifier(
                self, expr: FunctionIdentifier,
                *args: P.args, **kwargs: P.kwargs) -> Dependencies:
        return set()

    def map_sub_array_ref(self, expr: SubArrayRef,
                          *args: P.args, **kwargs: P.kwargs) -> Dependencies:
        deps = self.rec(expr.subscript, *args, **kwargs)
        return deps - set(expr.swept_inames)

    map_linear_subscript: Callable[Concatenate[Self, LinearSubscript, P],
                                   Dependencies] = DependencyMapperBase.map_subscript

    def map_type_cast(self, expr: TypeCast,
                      *args: P.args, **kwargs: P.kwargs) -> Dependencies:
        return self.rec(expr.child, *args, **kwargs)

    def map_resolved_function(self, expr: ResolvedFunction,
                              *args: P.args, **kwargs: P.kwargs) -> Dependencies:
        return self.rec(expr.function, *args, **kwargs)

    def map_literal(self, expr: Literal,
                    *args: P.args, **kwargs: P.kwargs) -> Dependencies:
        return set()

    @override
    def map_call_with_kwargs(self, expr: p.CallWithKwargs,
                             *args: P.args, **kwargs: P.kwargs) -> Dependencies:
        # See https://github.com/inducer/loopy/pull/323
        raise NotImplementedError

    map_fortran_division: Callable[Concatenate[Self, FortranDivision, P],
                                   Dependencies] = DependencyMapperBase.map_quotient


class SubstitutionRuleExpander(IdentityMapper[[]]):
    rules: Mapping[str, SubstitutionRule]

    def __init__(self, rules: Mapping[str, SubstitutionRule]) -> None:
        super().__init__()
        self.rules = rules

    @override
    def __call__(self, expr: Expression) -> Expression:
        if not self.rules:
            return expr
        return super().__call__(expr)

    @override
    def map_variable(self, expr: p.Variable) -> Expression:
        if expr.name in self.rules:
            return self.map_subst_rule(expr.name, self.rules[expr.name], ())
        else:
            return super().map_variable(expr)

    @override
    def map_call(self, expr: p.Call) -> Expression:
        assert isinstance(expr.function, p.Variable | ResolvedFunction)
        if expr.function.name in self.rules:
            assert isinstance(expr.function.name, str)
            return self.map_subst_rule(
                    expr.function.name,
                    self.rules[expr.function.name],
                    expr.parameters)
        else:
            return super().map_call(expr)

    def map_subst_rule(self,
                        name: str,
                        rule: SubstitutionRule,
                        arguments: Sequence[Expression],
                    ) -> Expression:
        if len(rule.arguments) != len(arguments):
            from loopy.diagnostic import LoopyError
            raise LoopyError("number of arguments to '%s' does not match "
                    "definition" % name)

        from pymbolic.mapper.substitutor import make_subst_func
        submap = SubstitutionMapper(
                make_subst_func(
                    dict(zip(rule.arguments, arguments, strict=True))))

        expr = submap(rule.expression)

        return self.rec(expr)

# }}}


# {{{ loopy-specific primitives

class LoopyExpressionBase(p.ExpressionNode):
    def stringifier(self):
        from loopy.diagnostic import LoopyError
        raise LoopyError("pymbolic < 2019.1 is in use. Please upgrade.")

    def make_stringifier(self, originating_stringifier=None):
        return StringifyMapper()


@p.expr_dataclass()
class Literal(LoopyExpressionBase):
    """A literal to be used during code generation.

    .. note::

        Only used in the output of
        :mod:`loopy.target.c.codegen.expression.ExpressionToCExpressionMapper` (and
        similar mappers). Not for use in Loopy source representation.

    .. autoattribute:: s
    """

    s: str


@p.expr_dataclass()
class ArrayLiteral(LoopyExpressionBase):
    """An array literal.

    .. note::

        Only used in the output of
        :mod:`loopy.target.c.codegen.expression.ExpressionToCExpressionMapper` (and
        similar mappers). Not for use in Loopy source representation.

    .. autoattribute:: children
    """

    children: tuple[Expression, ...]


@p.expr_dataclass()
class HardwareAxisIndex(LoopyExpressionBase):
    axis: int


@p.expr_dataclass()
class GroupHardwareAxisIndex(HardwareAxisIndex):
    """
    .. note::

        Only used in the output of
        :mod:`loopy.target.c.expression.ExpressionToCExpressionMapper` (and
        similar mappers). Not for use in Loopy source representation.

    .. autoattribute:: axis
    """

    mapper_method: ClassVar[str] = "map_group_hw_index"


@p.expr_dataclass()
class LocalHardwareAxisIndex(HardwareAxisIndex):
    """
    .. note::

        Only used in the output of
        :mod:`loopy.target.c.expression.ExpressionToCExpressionMapper` (and
        similar mappers). Not for use in Loopy source representation.

    .. autoattribute:: axis
    """

    mapper_method: ClassVar[str] = "map_local_hw_index"


@p.expr_dataclass()
class FunctionIdentifier(LoopyExpressionBase):
    """A base class for symbols representing functions."""

    mapper_method: ClassVar[str] = "map_loopy_function_identifier"


@p.expr_dataclass()
class TypedCSE(LoopyExpressionBase, p.CommonSubexpression):
    """A :class:`pymbolic.primitives.CommonSubexpression` annotated with
    a type.

    .. autoattribute:: dtype
    """

    dtype: LoopyType | None = None

    @override
    def get_extra_properties(self) -> dict[str, Any]:
        return {"dtype": self.dtype}


@p.expr_dataclass()
class TypeAnnotation(LoopyExpressionBase):
    """Undocumented for now. Currently only used internally around LHSs of
    assignments that create temporaries.

    .. autoattribute:: type
    .. autoattribute:: child
    """

    # FIXME(pyright): this is set to lp.Optional in `LoopyParser.parse_prefix`
    type: LoopyType
    child: Expression


@p.expr_dataclass(init=False)
class TypeCast(LoopyExpressionBase):
    """Only defined for numerical types with semantics matching
    :meth:`numpy.ndarray.astype`.

    .. autoattribute:: child
    .. autoattribute:: type
    """

    child: Expression
    """The expression to be cast."""

    # We're storing the type as a name for now to avoid
    # numpy pickling bug madness. (see loopy.types)
    _type_name: str

    def __init__(self, type: ToLoopyTypeConvertible, child: Expression) -> None:
        super().__init__()

        from loopy.types import NumpyType, to_loopy_type
        ltype = to_loopy_type(type)

        if (not isinstance(ltype, NumpyType)
                or not issubclass(ltype.dtype.type, np.number)):
            from loopy.diagnostic import LoopyError
            raise LoopyError(
                f"TypeCast only supports numerical numpy types: got '{ltype}'")

        object.__setattr__(self, "_type_name", ltype.dtype.name)
        object.__setattr__(self, "child", child)

    @property
    def type(self) -> NumpyType:
        from loopy.types import NumpyType
        return NumpyType(np.dtype(self._type_name))


@p.expr_dataclass(init=False)
class TaggedVariable(LoopyExpressionBase, p.Variable, Taggable):
    """This is an identifier with tags, such as ``matrix$one``, where
    'one' identifies this specific use of the identifier. This mechanism
    may then be used to address these uses--such as by prefetching only
    accesses tagged a certain way.

    .. autoattribute:: tags

    Inherits from :class:`pymbolic.primitives.Variable`
    and :class:`pytools.tag.Taggable`.
    """

    tags: frozenset[Tag]
    """A :class:`frozenset` of subclasses of :class:`pytools.tag.Tag` used to
    provide metadata on this object. Legacy string tags are converted to
    :class:`~loopy.LegacyStringInstructionTag` or, if they used to carry
    a functional meaning, the tag carrying that same functional meaning
    (e.g. :class:`~loopy.UseStreamingStoreTag`).
    """

    def __init__(self, name: str, tags: ToTagSetConvertible) -> None:
        p.Variable.__init__(self, name)
        if isinstance(tags, str):
            from loopy.kernel.creation import _normalize_string_tag
            tags = frozenset({_normalize_string_tag(tags)})

        assert isinstance(tags, frozenset)
        assert tags

        object.__setattr__(self, "tags", tags)

    def copy(self, *,
             name: str | None = None,
             tags: ToTagSetConvertible = None) -> TaggedVariable:
        name = self.name if name is None else name
        tags = self.tags if tags is None else tags
        return TaggedVariable(name, tags)


@p.expr_dataclass(init=False)
class Reduction(LoopyExpressionBase):
    """
    Represents a reduction operation on :attr:`expr` across :attr:`inames`.

    .. autoattribute:: operation
    .. autoattribute:: inames
    .. autoattribute:: expr
    .. autoattribute:: allow_simultaneous
    """

    operation: ReductionOperation

    inames: Sequence[str]
    """The inames across which reduction on :attr:`expr` is being carried out."""

    expr: Expression
    """An expression which may have tuple type. If the expression has tuple
    type, it must be one of the following:

    * a :class:`tuple` of :data:`pymbolic.typing.Expression`, or
    * a :class:`loopy.symbolic.Reduction`, or
    * a function call or substitution rule invocation.
    """

    allow_simultaneous: bool
    """If not *True*, an iname is allowed to be used in precisely one reduction,
    to avoid misnesting errors.
    """

    def __init__(self,
                 operation: ReductionOperation | str,
                 inames: tuple[str | p.Variable, ...] | p.Variable | str,
                 expr: Expression,
                 allow_simultaneous: bool = False
             ) -> None:
        if isinstance(inames, str):
            inames = tuple(iname.strip() for iname in inames.split(","))

        elif isinstance(inames, p.Variable):
            inames = (inames,)

        assert isinstance(inames, tuple)

        def strip_var(iname: str | p.Variable) -> str:
            if isinstance(iname, p.Variable):
                iname = iname.name

            assert isinstance(iname, str)
            return iname

        inames = tuple(strip_var(iname) for iname in inames)

        if isinstance(operation, str):
            from loopy.library.reduction import parse_reduction_op
            op = parse_reduction_op(operation)
        else:
            op = operation
        del operation

        from loopy.library.reduction import ReductionOperation
        assert isinstance(op, ReductionOperation)

        from loopy.diagnostic import LoopyError

        if op.arg_count > 1:
            from pymbolic.primitives import Call

            if not isinstance(expr, (tuple, Reduction, Call)):
                raise LoopyError("reduction argument must be one of "
                                 "a tuple, reduction, or call; "
                                 "got '%s'" % type(expr).__name__)
        else:
            # Sanity checks
            if isinstance(expr, tuple):
                raise LoopyError("got a tuple argument to a scalar reduction")
            elif isinstance(expr, Reduction) and expr.is_tuple_typed:
                raise LoopyError("got a tuple typed argument to a scalar reduction")

        object.__setattr__(self, "operation", op)
        object.__setattr__(self, "inames", inames)
        object.__setattr__(self, "expr", expr)
        object.__setattr__(self, "allow_simultaneous", allow_simultaneous)

    @property
    def is_tuple_typed(self) -> bool:
        return self.operation.arg_count > 1

    @cached_property
    def inames_set(self) -> set[str]:
        return set(self.inames)


@p.expr_dataclass()
class LinearSubscript(LoopyExpressionBase):
    """Represents a linear index into a multi-dimensional array, completely
    ignoring any multi-dimensional layout.
    """
    aggregate: Expression
    index: Expression

    def __post_init__(self) -> None:
        warn("LinearSubscript is deprecated "
             "and will be removed without replacement in 2026.",
             DeprecationWarning, stacklevel=3)


@p.expr_dataclass()
class RuleArgument(LoopyExpressionBase):
    """Represents a (numbered) argument of a :class:`loopy.SubstitutionRule`.
    Only used internally in the rule-aware mappers to match subst rules
    independently of argument names.
    """

    index: int


@p.expr_dataclass(init=False)
class ResolvedFunction(LoopyExpressionBase):
    """
    A function identifier whose definition is known in a :mod:`loopy` program.
    A function is said to be *known* in a :class:`~loopy.TranslationUnit` if its
    name maps to  an :class:`~loopy.kernel.function_interface.InKernelCallable`
    in :attr:`loopy.TranslationUnit.callables_table`. Refer to :ref:`func-interface`.

    .. autoattribute:: function
    .. autoattribute:: name
    """
    function: p.Variable | ReductionOpFunction

    def __init__(self, function: p.Variable | ReductionOpFunction) -> None:
        if isinstance(function, str):
            function = p.Variable(function)
        from loopy.library.reduction import ReductionOpFunction
        assert isinstance(function, (p.Variable, ReductionOpFunction))
        object.__setattr__(self, "function", function)

    @property
    def name(self) -> str | ReductionOpFunction:
        from loopy.library.reduction import ReductionOpFunction
        if isinstance(self.function, p.Variable):
            return self.function.name
        elif isinstance(self.function, ReductionOpFunction):
            return self.function
        else:
            raise LoopyError("Unexpected function type %s in ResolvedFunction." %
                    type(self.function))

# }}}


# {{{ more mappers

class EvaluatorWithDeficientContext(PartialEvaluationMapper):
    """Evaluation Mapper that does not need values of all the variables
    involved in the expression.

    Returns the expression with the values mapped from :attr:`context`.
    """

    @override
    def map_variable(self, expr: p.Variable) -> Expression:
        if expr.name in self.context:
            return self.context[expr.name]
        else:
            return expr


class VariableInAnExpression(CombineMapper[bool, []]):
    variables_to_search: Collection[p.Variable]

    def __init__(self, variables_to_search: Collection[p.Variable]) -> None:
        assert all(isinstance(variable, p.Variable) for variable in variables_to_search)

        super().__init__()
        self.variables_to_search = variables_to_search

    @override
    def combine(self, values: Iterable[bool]) -> bool:
        return any(values)

    @override
    def map_variable(self, expr: p.Variable) -> bool:
        return expr in self.variables_to_search

    @override
    def map_constant(self, expr: object) -> bool:
        return False


class SweptInameStrideCollector(CoefficientCollectorBase):
    """
    Mapper to compute the coefficient swept inames for :class:`SubArrayRef`.
    """

    @override
    def map_algebraic_leaf(self, expr: p.AlgebraicLeaf) -> CoeffsT:
        # subscripts that are not involved in :attr:`target_names` are treated
        # as constants.
        if isinstance(expr, p.Subscript) and (
                self.target_names is None
                or getattr(expr.aggregate, "name", None) not in self.target_names):
            return {1: expr}

        return super().map_algebraic_leaf(expr)


def get_start_subscript_from_sar(
            sar: SubArrayRef, kernel: LoopKernel,
        ) -> p.Variable | p.Subscript:
    """
    Returns an instance of :class:`pymbolic.primitives.Subscript`, the
    beginning subscript of the array swept by the *SubArrayRef*.

    **Example:** Consider ``[i, k]: a[i, j, k, l]``. The beginning
    subscript would be ``a[0, j, 0, l]``
    """

    def _get_lower_bound(iname: str) -> int:
        pwaff = kernel.get_iname_bounds(iname).lower_bound_pw_aff
        return int(pw_aff_to_expr(pwaff))

    swept_inames_to_zeros = {
            swept_iname.name: _get_lower_bound(swept_iname.name)
            for swept_iname in sar.swept_inames}

    result = EvaluatorWithDeficientContext(swept_inames_to_zeros)(sar.subscript)
    assert isinstance(result, (p.Variable, p.Subscript))

    return result


@p.expr_dataclass()
class SubArrayRef(LoopyExpressionBase):
    """
    An algebraic expression to map an affine memory layout pattern (known as
    sub-arary) as consecutive elements of the sweeping axes which are defined
    using :attr:`SubArrayRef.swept_inames`.

    .. attribute:: swept_inames

        An instance of :class:`tuple` denoting the axes to which the sub array
        is supposed to be mapped to.

    .. attribute:: subscript

        An instance of :class:`pymbolic.primitives.Subscript` denoting the
        array in the kernel.

    .. automethod:: is_equal
    """

    swept_inames: tuple[p.Variable, ...]
    subscript: p.Subscript

    def __post_init__(self) -> None:
        # {{{ sanity checks

        if not isinstance(self.swept_inames, tuple):
            warn(
                "swept_inames passed to SubArrayRef was not a tuple. "
                "This is deprecated and will stop working in 2025. "
                "Pass a tuple instead.", DeprecationWarning, stacklevel=3
            )
            object.__setattr__(self, "swept_inames", (self.swept_inames,))

        for iname in self.swept_inames:
            assert isinstance(iname, p.Variable)
        assert isinstance(self.subscript, p.Subscript)


@p.expr_dataclass()
class FortranDivision(p.QuotientBase, LoopyExpressionBase):
    """This exists for the benefit of the Fortran frontend, which specializes
    to floating point division for floating point inputs and round-to-zero
    division for integer inputs. Despite the name, this would also be usable
    for C semantics. (:mod:`loopy` division semantics match Python's.)

    .. note::

        This is not a documented expression node type. It may disappear
        at any moment.
    """

# }}}


class DependencyMapperWithReductionInames(DependencyMapper[P]):
    reduction_inames: set[str]

    def __init__(
        self,
        include_subscripts: bool = True,
        include_lookups: bool = True,
        include_calls: bool | TypingLiteral["descend_args"] = True,
        include_cses: bool = False,
        composite_leaves: bool | None = None,
    ) -> None:
        super().__init__(
            include_subscripts=include_subscripts,
            include_lookups=include_lookups,
            include_calls=include_calls,
            include_cses=include_cses,
            composite_leaves=composite_leaves,
        )
        self.reduction_inames = set()

    @override
    def map_reduction(
                self,
                expr: Reduction, *args: P.args, **kwargs: P.kwargs
            ) -> Dependencies:
        self.reduction_inames.update(expr.inames)
        return super().map_reduction(expr, *args, **kwargs)


@memoize
def _get_dependencies_and_reduction_inames(
            expr: Expression
        ) -> tuple[frozenset[str], frozenset[str]]:
    dep_mapper: DependencyMapperWithReductionInames[[]] = (
        DependencyMapperWithReductionInames(composite_leaves=False))

    deps = frozenset(cast("p.Variable", dep).name for dep in dep_mapper(expr))
    reduction_inames = dep_mapper.reduction_inames
    return frozenset(deps), frozenset(reduction_inames)


def get_dependencies(expr: Expression) -> frozenset[str]:
    return _get_dependencies_and_reduction_inames(expr)[0]


def get_reduction_inames(expr: Expression) -> frozenset[str]:
    return _get_dependencies_and_reduction_inames(expr)[1]


class SubArrayRefSweptInamesCollector(CombineMapper[Set[str], []]):
    @override
    def combine(self, values: Iterable[Set[str]]) -> Set[str]:
        result: Set[str] = set()
        for value in values:
            result |= value

        return frozenset(result)

    @override
    def map_sub_array_ref(self, expr: SubArrayRef) -> Set[str]:
        return frozenset({iname.name for iname in expr.swept_inames})

    @override
    def map_constant(
                self, expr: (
                    object
                    | p.Variable | p.FunctionSymbol | p.NaN
                    | TaggedVariable | TypeCast | ResolvedFunction)) -> Set[str]:
        return frozenset()

    map_variable: Callable[[Self, p.Variable], Set[str]] = map_constant
    map_function_symbol: Callable[[Self, p.FunctionSymbol], Set[str]] = map_constant
    map_nan: Callable[[Self, p.NaN], Set[str]] = map_constant
    map_tagged_variable: Callable[[Self, TaggedVariable], Set[str]] = map_constant
    map_type_cast: Callable[[Self, TypeCast], Set[str]] = map_constant
    map_resolved_function: Callable[[Self, ResolvedFunction], Set[str]] = map_constant


def get_sub_array_ref_swept_inames(expr: SubArrayRef) -> Set[str]:
    return SubArrayRefSweptInamesCollector()(expr)


# {{{ rule-aware mappers

def parse_tagged_name(expr: Expression) -> tuple[str, Set[Tag] | None]:
    from loopy.library.reduction import ArgExtOp, SegmentedOp

    if isinstance(expr, TaggedVariable):
        return expr.name, expr.tags
    elif isinstance(expr, ResolvedFunction):
        return parse_tagged_name(expr.function)
    elif isinstance(expr, (p.Variable, ArgExtOp, SegmentedOp)):
        return expr.name, None
    else:
        raise RuntimeError("subst rule name not understood: %s" % expr)


@dataclass(frozen=True)
class ExpansionState:
    """
    .. autoattribute:: kernel
    .. autoattribute:: instruction
    .. autoattribute:: stack
    .. autoattribute:: arg_context
    """

    kernel: LoopKernel
    instruction: InstructionBase
    stack: tuple[ConcreteMatchable, ...]
    """A tuple representing the current expansion stack, as a tuple of
    ``(name, tag)`` pairs.
    """
    arg_context: Mapping[str, Expression]
    """A mapping representing current argument values."""

    def __post_init__(self) -> None:
        hash(self.arg_context)

    def copy(self, **kwargs: Any) -> Self:
        return replace(self, **kwargs)

    @override
    def __hash__(self) -> int:
        # do not try to be precise about hash of loopy kernel
        # or the instruction as computing the hash of pymbolic
        # expressions could have exponential complexity
        return hash((id(self.kernel), id(self.instruction),
                     self.stack, self.arg_context))

    @property
    def insn_id(self) -> str:
        return self.instruction.id

    def apply_arg_context(self, expr: Expression) -> Expression:
        from pymbolic.mapper.substitutor import make_subst_func
        return SubstitutionMapper(
                make_subst_func(self.arg_context))(expr)


class SubstitutionRuleRenamer(IdentityMapper[[]]):
    renames: dict[str, str]

    def __init__(self, renames: dict[str, str]) -> None:
        self.renames = renames
        super().__init__()

    @override
    def map_call(self, expr: p.Call) -> Expression:
        if not isinstance(expr.function, p.Variable):
            return super().map_call(expr)

        name, tags = parse_tagged_name(expr.function)

        new_name = self.renames.get(name)
        if new_name is None:
            return super().map_call(expr)

        if tags:
            sym: p.Variable = TaggedVariable(new_name, tags)
        else:
            sym = p.Variable(new_name)

        return type(expr)(sym, tuple(self.rec(child) for child in expr.parameters))

    @override
    def map_variable(self, expr: p.Variable) -> Expression:
        name, tags = parse_tagged_name(expr)

        new_name = self.renames.get(name)
        if new_name is None:
            return super().map_variable(expr)

        if tags:
            return TaggedVariable(new_name, tags)
        else:
            return p.Variable(new_name)


def rename_subst_rules_in_instructions(
            insns: Sequence[InstructionBase], renames: dict[str, str]
        ) -> Sequence[InstructionBase]:
    subst_renamer = SubstitutionRuleRenamer(renames)

    return [
            insn.with_transformed_expressions(subst_renamer)
            for insn in insns]


class SubstitutionRuleMappingContext:
    old_subst_rules: dict[str, SubstitutionRule]
    make_unique_var_name: Callable[[str], str]

    subst_rule_registry: dict[Expression, tuple[str, Sequence[str], Expression]]
    subst_rule_old_names: dict[Expression, list[str]]

    def __init__(self,
                 old_subst_rules: dict[str, SubstitutionRule],
                 make_unique_var_name: Callable[[str], str]) -> None:
        self.old_subst_rules = old_subst_rules
        self.make_unique_var_name = make_unique_var_name

        # maps subst rule (args, bodies) to (names, original_name)
        self.subst_rule_registry = {
                self._get_subst_rule_key(rule.arguments, rule.expression):
                    (name, rule.arguments, rule.expression)
                for name, rule in old_subst_rules.items()}

        # maps subst rule (args, bodies) to a list of old names,
        # which doubles as (a) a histogram of uses and (b) a way
        # to deterministically find the 'lexicographically earliest'
        # name
        self.subst_rule_old_names = {}

    def _get_subst_rule_key(self,
                            args: Sequence[str],
                            body: Expression) -> Expression:
        subst_dict = {
                arg: RuleArgument(i)
                for i, arg in enumerate(args)}

        from pymbolic.mapper.substitutor import make_subst_func
        arg_subst_map = SubstitutionMapper(make_subst_func(subst_dict))

        return arg_subst_map(body)

    def register_subst_rule(self, original_name: str,
                            args: Sequence[str], body: Expression) -> str:
        """Returns a name (as a string) for a newly created substitution
        rule.
        """
        key = self._get_subst_rule_key(args, body)
        reg_value = self.subst_rule_registry.get(key)

        if reg_value is None:
            # These names are temporary and won't stick around.
            new_name = self.make_unique_var_name("_lpy_tmp_"+original_name)
            self.subst_rule_registry[key] = (new_name, args, body)
        else:
            new_name, _, _ = reg_value

        self.subst_rule_old_names.setdefault(key, []).append(original_name)
        return new_name

    def _get_new_substitutions_and_renames(
                self
            ) -> tuple[dict[str, SubstitutionRule], dict[str, str]]:
        """This makes a new dictionary of substitutions from the ones
        encountered in mapping all the encountered expressions.
        It tries hard to keep substitution names the same--i.e.
        if all derivative versions of a substitution rule ended
        up with the same mapped version, then this version should
        retain the name that the substitution rule had previously.
        Unfortunately, this can't be done in a single pass, and so
        the routine returns an additional dictionary *subst_renames*
        of renamings to be performed on the processed expressions.

        The returned substitutions already have the rename applied
        to them.

        :returns: (new_substitutions, subst_renames)
        """

        from loopy.kernel.data import SubstitutionRule

        result: dict[str, SubstitutionRule] = {}
        renames: dict[str, str] = {}
        used_names: set[str] = set()

        for key, (name, args, body) in self.subst_rule_registry.items():
            orig_names = self.subst_rule_old_names.get(key, [])

            # If no orig_names are found, then this particular
            # subst rule was never referenced, and so it's fine
            # to leave out.

            if not orig_names:
                continue

            new_name = min(orig_names)
            if new_name in used_names:
                new_name = self.make_unique_var_name(new_name)

            renames[name] = new_name
            used_names.add(new_name)

            result[new_name] = SubstitutionRule(
                    name=new_name,
                    arguments=args,
                    expression=body)

        # {{{ perform renames on new substitutions

        subst_renamer = SubstitutionRuleRenamer(renames)

        renamed_result = {}
        for name, rule in result.items():
            renamed_result[name] = rule.copy(
                    expression=subst_renamer(rule.expression))

        # }}}

        return renamed_result, renames

    def finish_kernel(self, kernel: LoopKernel) -> LoopKernel:
        new_substs, renames = self._get_new_substitutions_and_renames()
        if not renames:
            return kernel.copy(substitutions=new_substs)

        new_insns = rename_subst_rules_in_instructions(kernel.instructions, renames)

        return kernel.copy(
            substitutions=new_substs,
            instructions=new_insns)


class RuleAwareIdentityMapper(IdentityMapper[Concatenate[ExpansionState, P]]):
    """Note: the third argument dragged around by this mapper is the
    current :class:`ExpansionState`.

    Subclasses of this must be careful to not touch identifiers that
    are in :attr:`ExpansionState.arg_context`.
    """

    rule_mapping_context: SubstitutionRuleMappingContext

    def __init__(self, rule_mapping_context: SubstitutionRuleMappingContext) -> None:
        self.rule_mapping_context = rule_mapping_context
        super().__init__()

    @override
    def map_variable(
                self, expr: p.Variable, expn_state: ExpansionState,
                *args: P.args, **kwargs: P.kwargs
            ) -> Expression:
        name, tags = parse_tagged_name(expr)
        if name not in self.rule_mapping_context.old_subst_rules:
            return super().map_variable(expr, expn_state, *args, **kwargs)
        else:
            return self.map_subst_rule(name, tags, (), expn_state, *args, **kwargs)

    @override
    def map_call(
                self, expr: p.Call, expn_state: ExpansionState,
                *args: P.args, **kwargs: P.kwargs
            ) -> Expression:
        if not isinstance(expr.function, p.Variable):
            return super().map_call(expr, expn_state, *args, **kwargs)

        name, tags = parse_tagged_name(expr.function)

        if name not in self.rule_mapping_context.old_subst_rules:
            return super().map_call(expr, expn_state, *args, **kwargs)
        else:
            params = tuple(self.rec(param, expn_state, *args, **kwargs)
                           for param in expr.parameters)
            return self.map_subst_rule(name, tags, params, expn_state, *args, **kwargs)

    @staticmethod
    def make_new_arg_context(
            rule_name: str,
            arg_names: Sequence[str],
            arguments: Sequence[Expression],
            arg_context: Mapping[str, Expression]
            ) -> Mapping[str, Expression]:
        if len(arg_names) != len(arguments):
            raise RuntimeError("Rule '%s' invoked with %d arguments (needs %d)"
                    % (rule_name, len(arguments), len(arg_names), ))

        from pymbolic.mapper.substitutor import make_subst_func
        arg_subst_map = SubstitutionMapper(make_subst_func(arg_context))
        return constantdict({
            formal_arg_name: arg_subst_map(arg_value)
            for formal_arg_name, arg_value in zip(arg_names, arguments, strict=True)
        })

    def map_subst_rule(
                self,
                name: str,
                tags: Set[Tag] | None,
                arguments: Sequence[Expression],
                expn_state: ExpansionState,
                *args: P.args, **kwargs: P.kwargs
            ) -> Expression:
        if tags is None:
            tags = frozenset()
        tags = frozenset(tags)

        rule = self.rule_mapping_context.old_subst_rules[name]

        rec_arguments = self.rec(tuple(arguments), expn_state, *args, **kwargs)
        assert isinstance(rec_arguments, tuple)

        from loopy.match import ConcreteMatchable
        new_expn_state = expn_state.copy(
                stack=(*expn_state.stack, ConcreteMatchable(name, tags)),
                arg_context=self.make_new_arg_context(
                    name, rule.arguments, rec_arguments, expn_state.arg_context))

        result = self.rec(rule.expression, new_expn_state, *args, **kwargs)

        new_name = self.rule_mapping_context.register_subst_rule(
                name, rule.arguments, result)

        if tags:
            sym: p.Variable = TaggedVariable(new_name, tags)
        else:
            sym = p.Variable(new_name)

        if arguments:
            return sym(*rec_arguments)
        else:
            return sym

    @override
    def __call__(self,  # pyright: ignore[reportIncompatibleMethodOverride]
                 expr: Expression,
                 kernel: LoopKernel,
                 insn: InstructionBase,
                 *args: P.args, **kwargs: P.kwargs) -> Expression:
        """
        :arg insn: A :class:`~loopy.kernel.InstructionBase` of which *expr* is
            a part of, or *None* if *expr*'s source is not an instruction.
        """
        from loopy.kernel.instruction import InstructionBase
        assert insn is None or isinstance(insn, InstructionBase)

        expn_state = ExpansionState(
                kernel=kernel,
                instruction=insn,
                stack=(),
                arg_context=constantdict())

        return super().__call__(expr, expn_state, *args, **kwargs)

    def map_instruction(self,
                        kernel: LoopKernel,
                        insn: InstructionBase) -> InstructionBase:
        return insn

    def map_kernel(self,
                kernel: LoopKernel,
                within: StackMatch = lambda knl, insn, stack: True,
                map_args: bool = True,
                map_tvs: bool = True
            ) -> LoopKernel:
        new_insns = [
            # While subst rules are not allowed in assignees, the mapper
            # may perform tasks entirely unrelated to subst rules, so
            # we must map assignees, too.
            insn if not kernel.substitutions and not within(kernel, insn, ()) else
            self.map_instruction(kernel,
                insn.with_transformed_expressions(
                    lambda expr: self(expr, kernel, insn)))  # noqa: B023 # pyright: ignore[reportCallIssue]
            for insn in kernel.instructions]

        from functools import partial

        non_insn_self = partial(self, kernel=kernel, insn=None)

        from loopy.kernel.array import ArrayBase

        # {{{ args

        if map_args:
            new_args: Sequence[KernelArgument] = [
                arg.map_exprs(non_insn_self) if isinstance(arg, ArrayBase) else arg
                for arg in kernel.args]
        else:
            new_args = kernel.args[:]

        # }}}

        # {{{ tvs

        if map_tvs:
            new_tvs: Mapping[str, TemporaryVariable] = {
                tv_name: tv.map_exprs(non_insn_self)
                for tv_name, tv in kernel.temporary_variables.items()}
        else:
            new_tvs = kernel.temporary_variables

        # }}}

        # domains, var names: not exprs => do not map

        return kernel.copy(instructions=new_insns,
                           args=new_args,
                           temporary_variables=new_tvs)


class RuleAwareSubstitutionMapper(RuleAwareIdentityMapper[[]]):
    """
    Mapper to substitute expressions and record any divergence of substitution
    rule expressions of :class:`loopy.LoopKernel`.

    .. attribute:: rule_mapping_context

        An instance of :class:`SubstitutionRuleMappingContext` to record
        divergence of substitution rules.

    .. attribute:: within

        An instance of :class:`loopy.match.StackMatchComponent`.
        :class:`RuleAwareSubstitutionMapper` would perform
        substitutions in the expression if the stack match is ``True`` or
        if the expression does not arise from an :class:`~loopy.InstructionBase`.

    .. note::

        The mapped kernel should be passed through
        :meth:`SubstitutionRuleMappingContext.finish_kernel` to perform any
        renaming mandated by the rule expression divergences.
    """

    subst_func: Callable[[Expression], Expression | None]
    _within: StackMatch

    def __init__(self,
                 rule_mapping_context: SubstitutionRuleMappingContext,
                 subst_func: Callable[[Expression], Expression | None],
                 within: StackMatch) -> None:
        super().__init__(rule_mapping_context)

        self.subst_func = subst_func
        self._within = within

    def within(self,
                kernel: LoopKernel,
                instruction: InstructionBase | None,
                stack: RuleStack) -> bool:
        if instruction is None:
            # always perform substitutions on expressions not coming from
            # instructions.
            return True
        else:
            return self._within(kernel, instruction, stack)

    @override
    def map_variable(self, expr: p.Variable, expn_state: ExpansionState) -> Expression:
        if (expr.name in expn_state.arg_context
                or not self.within(
                    expn_state.kernel, expn_state.instruction, expn_state.stack)):
            # expr not in within => do nothing (call IdentityMapper)
            return super().map_variable(
                    expr, expn_state)

        result = self.subst_func(expr)
        if result is not None:
            return result
        else:
            return super().map_variable(expr, expn_state)


class RuleAwareSubstitutionRuleExpander(RuleAwareIdentityMapper[[]]):
    rules: dict[str, SubstitutionRule]
    within: StackMatch

    def __init__(self,
                 rule_mapping_context: SubstitutionRuleMappingContext,
                 rules: dict[str, SubstitutionRule],
                 within: StackMatch) -> None:
        super().__init__(rule_mapping_context)

        self.rules = rules
        self.within = within

    @override
    def map_subst_rule(
                self, name: str,
                tags: Set[Tag] | None,
                arguments: Sequence[Expression],
                expn_state: ExpansionState
            ) -> Expression:
        if tags is None:
            tags = frozenset()
        tags = frozenset(tags)

        from loopy.match import ConcreteMatchable
        new_stack = (*expn_state.stack, ConcreteMatchable(name, tags))

        if self.within(expn_state.kernel, expn_state.instruction, new_stack):
            # expand
            rule = self.rules[name]

            new_expn_state = expn_state.copy(
                    stack=new_stack,
                    arg_context=self.make_new_arg_context(
                        name, rule.arguments, arguments, expn_state.arg_context))

            result = self.rec(rule.expression, new_expn_state)

            # substitute in argument values
            from pymbolic.mapper.substitutor import make_subst_func
            subst_map = SubstitutionMapper(make_subst_func(
                new_expn_state.arg_context))

            return subst_map(result)

        else:
            # do not expand
            return super().map_subst_rule(name, tags, arguments, expn_state)

# }}}


# {{{ functions to primitives, parsing

class VarToTaggedVarMapper(IdentityMapper[[]]):
    @override
    def map_variable(self, expr: p.Variable) -> Expression:
        dollar_idx = expr.name.find("$")
        if dollar_idx == -1:
            return expr
        else:
            from loopy.kernel.instruction import LegacyStringInstructionTag
            return TaggedVariable(
                        expr.name[:dollar_idx],
                        frozenset({
                            LegacyStringInstructionTag(expr.name[dollar_idx+1:])
                        })
                    )


class FunctionToPrimitiveMapper(UncachedIdentityMapper[[]]):
    """Looks for invocations of a function called 'cse' or 'reduce' and
    turns those into the actual :mod:`pymbolic` primitives used for that.
    """

    def _parse_reduction(self,
                operation: ReductionOperation,
                inames: Expression,
                red_exprs: tuple[Expression, ...],
                allow_simultaneous: bool = False) -> Reduction:
        if isinstance(inames, p.Variable):
            inames = (inames,)

        if not isinstance(inames, (tuple, list)):
            raise TypeError("iname argument to reduce() must be a symbol "
                    "or a list/tuple of symbols")

        processed_inames: list[str] = []
        for iname in inames:
            if not isinstance(iname, p.Variable):
                raise TypeError("iname argument to reduce() must be a symbol "
                        "or a tuple or a tuple of symbols")

            processed_inames.append(iname.name)

        if len(red_exprs) == 1:
            expr_or_exprs: Expression | tuple[Expression, ...] = red_exprs[0]
        else:
            expr_or_exprs = red_exprs

        return Reduction(operation, tuple(processed_inames), expr_or_exprs,
                allow_simultaneous=allow_simultaneous)

    @override
    def map_call(self, expr: p.Call) -> Expression:
        from loopy.library.reduction import parse_reduction_op

        if not isinstance(expr.function, p.Variable):
            return super().map_call(expr)

        name = expr.function.name
        if name == "cse":
            if len(expr.parameters) in [1, 2]:
                if len(expr.parameters) == 2:
                    if not isinstance(expr.parameters[1], p.Variable):
                        raise TypeError("second argument to cse() must be a symbol")
                    tag = expr.parameters[1].name
                else:
                    tag = None

                return p.CommonSubexpression(
                        self.rec(expr.parameters[0]), tag,
                        scope=p.cse_scope.EVALUATION)
            else:
                raise TypeError("cse takes two arguments")

        elif name in ["reduce", "simul_reduce"]:

            if len(expr.parameters) >= 3:
                op_expr, inames = expr.parameters[:2]
                red_exprs = expr.parameters[2:]

                operation = parse_reduction_op(str(op_expr))
                return self._parse_reduction(not_none(operation), inames,
                        tuple(self.rec(red_expr) for red_expr in red_exprs),
                        allow_simultaneous=(name == "simul_reduce"))
            else:
                raise TypeError("invalid 'reduce' calling sequence")

        elif name == "if":
            if len(expr.parameters) == 3:
                from pymbolic.primitives import If
                return If(*tuple(self.rec(p) for p in expr.parameters))
            else:
                raise TypeError("if takes three arguments")

        elif name in ["minimum", "maximum"]:
            if len(expr.parameters) == 2:
                from pymbolic.primitives import Max, Min
                return {
                    "minimum": Min,
                    "maximum": Max
                }[name](tuple(self.rec(p) for p in expr.parameters))
            else:
                raise TypeError(f"{name} takes two arguments")

        else:
            # see if 'name' is an existing reduction op

            operation = parse_reduction_op(name)
            if operation:
                # arg_count counts arguments but not inames
                if len(expr.parameters) != 1 + operation.arg_count:
                    raise RuntimeError("invalid invocation of "
                            "reduction operation '%s': expected %d arguments, "
                            "got %d instead" % (expr.function.name,
                                                1 + operation.arg_count,
                                                len(expr.parameters)))

                inames = expr.parameters[0]
                red_exprs = tuple(self.rec(param) for param in expr.parameters[1:])
                return self._parse_reduction(operation, inames, red_exprs)

            else:
                return super().map_call(expr)


# {{{ customization to pymbolic parser

_open_dbl_bracket = intern("open_dbl_bracket")

TRAILING_FLOAT_TAG_RE = re.compile(r"^(.*?)([a-zA-Z]*)$")


class LoopyParser(ParserBase):
    lex_table: ClassVar[pytools.lex.LexTable] = [
            (_open_dbl_bracket, pytools.lex.RE(r"\[\[")),
            *ParserBase.lex_table
            ]

    @override
    def parse_float(self, s: str) -> np.inexact[Any] | float:
        match = TRAILING_FLOAT_TAG_RE.match(s)
        if match is None:
            raise RuntimeError(f"cannot parse string as float: '{s}'")

        val = match.group(1)
        tag = frozenset(match.group(2))
        if tag == frozenset("j"):
            return np.float64(val)*np.complex128(1j)
        elif tag == frozenset("jf"):
            return np.float32(val)*np.complex64(1j)
        elif tag == frozenset("f"):
            return np.float32(val)
        elif tag == frozenset("d"):
            return np.float64(val)
        else:
            return float(val)  # generic float

    @override
    def parse_prefix(self, pstate: pytools.lex.LexIterator) -> Expression:
        from pymbolic.parser import (
            _PREC_UNARY,
            _closebracket,
            _colon,
            _greater,
            _identifier,
            _less,
            _openbracket,
        )

        import loopy as lp

        if pstate.is_next(_less):
            pstate.advance()
            if pstate.is_next(_greater):
                typename = lp.Optional(None)
                pstate.advance()
            else:
                pstate.expect(_identifier)
                typename = lp.Optional(pstate.next_str())
                pstate.advance()
                pstate.expect(_greater)
                pstate.advance()

            return TypeAnnotation(
                    typename,
                    self.parse_expression(pstate, _PREC_UNARY))

        elif pstate.is_next(_openbracket):
            rollback_pstate = pstate.copy()
            pstate.advance()
            pstate.expect_not_end()
            if pstate.is_next(_closebracket):
                swept_inames = ()
            else:
                swept_inames = self.parse_expression(pstate)

            pstate.expect(_closebracket)
            pstate.advance()
            if pstate.is_next(_colon):
                # pstate.expect(_colon):
                pstate.advance()
                subscript = self.parse_expression(pstate, _PREC_UNARY)
                if isinstance(swept_inames, p.Variable):
                    swept_inames = (swept_inames,)
                assert isinstance(swept_inames, tuple)
                assert isinstance(subscript, p.Subscript)
                return SubArrayRef(
                    cast("tuple[p.Variable, ...]", swept_inames),
                    subscript)
            else:
                pstate = rollback_pstate
                return super().parse_prefix(rollback_pstate)
        else:
            return super().parse_prefix(pstate)

    @override
    def parse_postfix(self,
                pstate: pytools.lex.LexIterator,
                min_precedence: int,
                left_exp: Expression) -> tuple[Expression, bool]:
        from pymbolic.parser import _PREC_CALL, _closebracket
        if pstate.next_tag() is _open_dbl_bracket and min_precedence < _PREC_CALL:
            pstate.advance()
            pstate.expect_not_end()
            left_exp = LinearSubscript(
                        left_exp,
                        self.parse_expression(pstate))
            pstate.expect(_closebracket)
            pstate.advance()
            pstate.expect(_closebracket)
            pstate.advance()
            return left_exp, True

        return ParserBase.parse_postfix(self, pstate, min_precedence, left_exp)

# }}}


def parse(expr_str: str) -> Expression:
    return VarToTaggedVarMapper()(FunctionToPrimitiveMapper()(LoopyParser()(expr_str)))

# }}}


# {{{ coefficient collector

class CoefficientCollector(CoefficientCollectorBase):
    map_tagged_variable: Callable[[Self, TaggedVariable],
                                  CoeffsT] = CoefficientCollectorBase.map_variable

    @override
    def map_subscript(self, expr: p.Subscript) -> CoeffsT:
        from loopy.diagnostic import ExpressionNotAffineError
        raise ExpressionNotAffineError(
                "cannot gather coefficients -- indirect addressing in use")

# }}}


# {{{ variable index expression collector

class ArrayAccessFinder(CombineMapper[Set[p.Subscript], []]):
    tgt_vector_name: str | None

    def __init__(self, tgt_vector_name: str | None = None) -> None:
        self.tgt_vector_name = tgt_vector_name
        super().__init__()

    @override
    def combine(self, values: Iterable[Set[p.Subscript]]) -> Set[p.Subscript]:
        from pytools import flatten
        return set(flatten(values))

    @override
    def map_constant(self, expr: object) -> Set[p.Subscript]:
        return set()

    @override
    def map_algebraic_leaf(self, expr: p.AlgebraicLeaf) -> Set[p.Subscript]:
        return set()

    @override
    def map_subscript(self, expr: p.Subscript) -> Set[p.Subscript]:
        assert isinstance(expr.aggregate, p.Variable)

        if (
                self.tgt_vector_name is None
                or expr.aggregate.name == self.tgt_vector_name):
            return {expr} | self.rec(expr.index)
        else:
            return super().map_subscript(expr)

# }}}


# {{{ (pw)aff to expr conversion

def aff_to_expr(aff: isl.Aff) -> ArithmeticExpression:
    from pymbolic import var

    denom = aff.get_denominator_val().to_python()

    result = (aff.get_constant_val()*denom).to_python()
    for dt in [isl.dim_type.in_, isl.dim_type.param]:
        for i in range(aff.dim(dt)):
            coeff = (aff.get_coefficient_val(dt, i)*denom).to_python()
            if coeff:
                dim_name = not_none(aff.get_dim_name(dt, i))
                result += coeff*var(dim_name)

    for i in range(aff.dim(isl.dim_type.div)):
        coeff = (aff.get_coefficient_val(isl.dim_type.div, i)*denom).to_python()
        if coeff:
            result += coeff*aff_to_expr(aff.get_div(i))

    assert not isinstance(result, complex)
    return flatten(result // denom)


def pw_aff_to_expr(
            pw_aff: int | isl.PwAff | isl.Aff,
            int_ok: bool = False
        ) -> ArithmeticExpression:
    if isinstance(pw_aff, int):
        if not int_ok:
            warn(f"expected PwAff, got int: {pw_aff}", stacklevel=2)

        return pw_aff

    pieces = pw_aff.get_pieces()
    last_expr = aff_to_expr(pieces[-1][1])

    pairs = [(set_to_cond_expr(constr_set), aff_to_expr(aff))
             for constr_set, aff in pieces[:-1]]

    from pymbolic.primitives import If

    expr = last_expr
    for condition, then_expr in reversed(pairs):
        expr = If(condition, then_expr, expr)

    return expr


def pw_aff_to_pw_aff_implemented_by_expr(pw_aff: isl.PwAff) -> isl.PwAff:
    pieces = pw_aff.get_pieces()

    rest = isl.Set.universe(pw_aff.space.params())
    aff_set, aff = pieces[0]
    impl_pw_aff = isl.PwAff.alloc(aff_set, aff)
    rest = rest.intersect_params(aff_set.complement())

    for aff_set, aff in pieces[1:-1]:
        impl_pw_aff = impl_pw_aff.union_max(
            isl.PwAff.alloc(aff_set, aff))
        rest = rest.intersect_params(aff_set.complement())

    _, aff = pieces[-1]
    return impl_pw_aff.union_max(isl.PwAff.alloc(rest, aff)).coalesce()

# }}}


# {{{ (pw)aff_from_expr

class PwAffEvaluationMapper(EvaluationMapperBase[isl.PwAff]):
    zero: isl.Aff
    pw_zero: isl.PwAff

    def __init__(self,
                 space: isl.Space,
                 vars_to_zero: Collection[str] | None) -> None:
        if vars_to_zero is None:
            vars_to_zero = ()

        self.zero = isl.Aff.zero_on_domain(isl.LocalSpace.from_space(space))
        self.pw_zero = isl.PwAff.from_aff(self.zero)

        context: dict[str, isl.PwAff] = {}
        for name, (dt, pos) in space.get_var_dict().items():
            if dt == isl.dim_type.set:
                dt = isl.dim_type.in_

            context[name] = isl.PwAff.from_aff(
                    self.zero.set_coefficient_val(dt, pos, 1))

        for v in vars_to_zero:
            context[v] = self.pw_zero

        super().__init__(context)

    @override
    def map_constant(self, expr: object) -> isl.PwAff:
        iexpr = int(cast("int", expr))

        return self.pw_zero + iexpr

    @override
    def map_min(self, expr: p.Min) -> isl.PwAff:
        from functools import reduce
        return reduce(
                lambda a, b: a.min(b),
                (self.rec(ch) for ch in expr.children))

    @override
    def map_max(self, expr: p.Max) -> isl.PwAff:
        from functools import reduce
        return reduce(
                lambda a, b: a.max(b),
                (self.rec(ch) for ch in expr.children))

    @override
    def map_quotient(self, expr: p.Quotient) -> isl.PwAff:
        raise TypeError("true division in '%s' not supported "
                "for as-pwaff evaluation" % expr)

    @override
    def map_floor_div(self, expr: p.FloorDiv) -> isl.PwAff:
        num = self.rec(expr.numerator)
        denom = self.rec(expr.denominator)
        return num.div(denom).floor()

    @override
    def map_remainder(self, expr: p.Remainder) -> isl.PwAff:
        num = self.rec(expr.numerator)
        denom = self.rec(expr.denominator)
        if not denom.is_cst():
            raise TypeError("modulo non-constant in '%s' not supported "
                    "for as-pwaff evaluation" % expr)

        (_s, denom_aff), = denom.get_pieces()
        denom = denom_aff.get_constant_val()

        return num.mod_val(denom)

    def map_literal(self, expr: Literal) -> isl.PwAff:
        raise TypeError("literal '%s' not supported "
                        "for as-pwaff evaluation" % expr)

    def map_reduction(self, expr: Reduction) -> isl.PwAff:
        raise TypeError("reduction in '%s' not supported "
                "for as-pwaff evaluation" % expr)

    @override
    def map_call(self, expr: p.Call) -> isl.PwAff:
        # FIXME: There are some things here that we could handle, e.g. "abs".
        raise TypeError(f"call in '{expr}' not supported "
                "for as-pwaff evaluation")


def aff_from_expr(
            space: isl.Space,
            expr: Expression,
            vars_to_zero: Collection[str] | None = None,
        ) -> isl.Aff:
    if vars_to_zero is None:
        vars_to_zero = frozenset()

    pwaff = pwaff_from_expr(space, expr, vars_to_zero).coalesce()

    pieces = pwaff.get_pieces()
    if len(pieces) == 1:
        (_s, aff), = pieces
        return aff
    else:
        from loopy.diagnostic import ExpressionNotAffineError
        raise ExpressionNotAffineError("expression '%s' could not be converted to a "
                "non-piecewise quasi-affine expression" % expr)


def pwaff_from_expr(
            space: isl.Space,
            expr: Expression,
            vars_to_zero: Collection[str] | None = None,
        ) -> isl.PwAff:
    return PwAffEvaluationMapper(space, vars_to_zero)(expr)


def with_aff_conversion_guard(
            f: Callable[Concatenate[isl.Space, Expression, P], T],
            space: isl.Space,
            expr: Expression,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> T:
    import islpy as isl
    from pymbolic.mapper.evaluator import UnknownVariableError

    from loopy.diagnostic import ExpressionNotAffineError

    err = None

    try:
        return f(space, expr, *args, **kwargs)
    except TypeError as e:
        err = e
    except isl.Error as e:
        err = e
    except UnknownVariableError as e:
        err = e
    except ExpressionNotAffineError as e:
        err = e

    assert err is not None
    from loopy.diagnostic import ExpressionToAffineConversionError
    raise ExpressionToAffineConversionError(
            "could not convert expression '%s' to affine representation: "
            "%s: %s" % (expr, type(err).__name__, str(err)))


def guarded_aff_from_expr(
            space: isl.Space,
            expr: Expression,
            vars_to_zero: Collection[str] | None = None,
        ) -> isl.Aff:
    """Performs the same operation as :func:`aff_from_expr` but only raises
    :exc:`loopy.diagnostic.ExpressionToAffineConversionError`
    """
    return with_aff_conversion_guard(aff_from_expr, space, expr, vars_to_zero)


def guarded_pwaff_from_expr(
            space: isl.Space,
            expr: Expression,
            vars_to_zero: Collection[str] | None = None,
        ) -> isl.PwAff:
    """Performs the same operation as :func:`aff_from_expr` but only raises
    :exc:`loopy.diagnostic.ExpressionToAffineConversionError`
    """
    return with_aff_conversion_guard(pwaff_from_expr, space, expr, vars_to_zero)

# }}}


# {{{ (pw_)?qpoly_from_expr

class PwQPolyEvaluationMapper(EvaluationMapperBase[isl.PwQPolynomial]):
    pw_zero: isl.PwQPolynomial

    def __init__(self,
                 space: isl.Space,
                 vars_to_zero: Collection[str] | None = None) -> None:
        if vars_to_zero is None:
            vars_to_zero = ()

        context: dict[str, isl.PwQPolynomial] = {}
        for name, (dt, pos) in space.get_var_dict().items():
            if dt == isl.dim_type.set:
                dt = isl.dim_type.in_

            context[name] = isl.PwQPolynomial.from_qpolynomial(
                    isl.QPolynomial.var_on_domain(space, dt, pos))

        pw_zero = isl.PwQPolynomial.from_qpolynomial(
                    isl.QPolynomial.zero_on_domain(space))
        for v in vars_to_zero:
            context[v] = pw_zero

        self.pw_zero = pw_zero
        super().__init__(context)

    @override
    def map_constant(self, expr: object) -> isl.PwQPolynomial:
        iexpr = int(cast("int", expr))

        return self.pw_zero + iexpr

    @override
    def map_quotient(self, expr: p.Quotient) -> isl.PwQPolynomial:
        raise TypeError("true division in '%s' not supported "
                "for as-pwqpoly evaluation" % expr)

    @override
    def map_power(self, expr: p.Power) -> isl.PwQPolynomial:
        from numbers import Integral
        if not isinstance(expr.exponent, Integral):
            raise TypeError("Only integral powers allowed in pwqpolynomials.")

        # do not "rec" exponent as it will be cast to a pwqpoly and
        # pwqpoly ** pwqpoly isn't allowed
        iexponent = int(expr.exponent)

        return self.rec(expr.base) ** iexponent


def pw_qpolynomial_from_expr(
            space: isl.Space,
            expr: Expression,
            vars_to_zero: Collection[str] | None = None) -> isl.PwQPolynomial:
    return PwQPolyEvaluationMapper(space, vars_to_zero)(expr)


def qpolynomial_from_expr(space: isl.Space, expr: Expression) -> isl.QPolynomial:
    pw_qpoly = pw_qpolynomial_from_expr(space, expr).coalesce()

    pieces = pw_qpoly.get_pieces()
    if len(pieces) == 1:
        (_s, qpoly), = pieces
        return qpoly
    else:
        raise RuntimeError("expression '%s' could not be converted to a "
                "non-piecewise quasi-polynomial expression" % expr)

# }}}


# {{{ simplify using aff

def simplify_via_aff(expr: Expression) -> Expression:
    from loopy.diagnostic import ExpressionToAffineConversionError

    deps = sorted(get_dependencies(expr))
    try:
        return aff_to_expr(guarded_aff_from_expr(
            isl.Space.create_from_names(isl.DEFAULT_CONTEXT, list(deps)),
            expr))
    except ExpressionToAffineConversionError:
        return expr


@memoize_on_first_arg
def simplify_using_aff(
            kernel: LoopKernel,
            expr: ArithmeticExpression
        ) -> ArithmeticExpression:
    """
    Simplifies *expr* on *kernel*'s domain.

    :arg expr: An instance of :data:`pymbolic.typing.Expression`.
    """
    deps = get_dependencies(expr)

    inames = deps & kernel.all_inames()

    # FIXME: Ideally, we should find out what inames are usable and allow
    # the simplification to use all of those. For now, fall back to making
    # sure that the simplification only uses inames that were already there.
    domain = (
            kernel
            .get_inames_domain(inames)
            .project_out_except(inames, [isl.dim_type.set]))

    non_inames = deps - set(domain.get_var_dict().keys())
    non_inames = {name for name in set(non_inames) if name.isidentifier()}
    if non_inames:
        cur_dim = domain.dim(isl.dim_type.set)
        domain = domain.insert_dims(isl.dim_type.set, cur_dim, len(non_inames))
        for non_iname in sorted(non_inames):
            domain = domain.set_dim_name(isl.dim_type.set, cur_dim, non_iname)
            cur_dim += 1

    try:
        aff = guarded_aff_from_expr(domain.space, expr)
    except ExpressionToAffineConversionError:
        # Accomplish at least *some* simplification
        return flatten(expr)

    # FIXME: Deal with assumptions, too.
    aff = aff.gist(domain)

    return aff_to_expr(aff)

# }}}


# {{{ qpolynomial_to_expr

def _get_monomial_coeff_from_term(
            space: isl.Space, term: isl.Term
        ) -> tuple[ArithmeticExpression, isl.Val]:
    result = 1

    for dt in isl._CHECK_DIM_TYPES:
        for i in range(term.dim(dt)):
            exp = term.get_exp(dt, i)
            if exp:
                name = space.get_dim_name(dt, i)
                assert name is not None

                result = result*p.Variable(name)**exp

    for i in range(term.dim(isl.dim_type.div)):
        exp = term.get_exp(isl.dim_type.div, i)
        result *= (aff_to_expr(term.get_div(i))**exp)

    return result, term.get_coefficient_val()


def _take_common_denominator(
            coeffs: Sequence[isl.Val],
        ) -> tuple[tuple[int, ...], int]:
    denominators = [coeff.get_den_val() for coeff in coeffs]
    numerators = [coeff * den for coeff, den in zip(coeffs, denominators, strict=True)]

    common_denominator = isl.Val.one(coeffs[0].get_ctx())
    for den in denominators:
        # LCM(a, b) = a * b / GCD(a, b)
        common_denominator = ((common_denominator * den)
                              .div(den.gcd(common_denominator)))

    numerators_scaled = [numerator * (common_denominator.div(denominator))
                         for numerator, denominator in
                         zip(numerators, denominators, strict=True)]

    return (tuple(num.to_python() for num in numerators_scaled),
            common_denominator.to_python())


def qpolynomial_to_expr(qpoly: isl.QPolynomial) -> Expression:
    from pymbolic.primitives import FloorDiv

    space = qpoly.space
    monomials, coeffs = zip(*[_get_monomial_coeff_from_term(space, t)
                              for t in qpoly.get_terms()], strict=True)

    numerators, common_denominator = _take_common_denominator(coeffs)

    assert len(numerators) == len(monomials)
    assert all(isinstance(num, int) for num in numerators)
    assert isinstance(common_denominator, int)

    # FIXME: Delete if in favor of the general case once we depend on pymbolic 2024.1.
    if common_denominator == 1:
        res = sum(num * monomial
                  for num, monomial in zip(numerators, monomials, strict=True))
    else:
        res = FloorDiv(
            sum(num * monomial
                for num, monomial in zip(numerators, monomials, strict=True)),
            common_denominator)

    return flatten(res)

# }}}


# {{{ expression/set <-> constraint conversion

def constraint_to_cond_expr(cns: isl.Constraint) -> ArithmeticExpression:
    # Looks like this is ok after all--get_aff() performs some magic.
    # Not entirely sure though... FIXME
    #
    # ls = cns.get_local_space()
    # if ls.dim(dim_type.div):
    #     raise RuntimeError("constraint has an existentially quantified variable")

    expr = aff_to_expr(cns.get_aff())

    from pymbolic.primitives import Comparison
    if cns.is_equality():
        return Comparison(expr, "==", 0)
    else:
        return Comparison(expr, ">=", 0)

# }}}


# {{{ isl_set_from_expr

class ConditionExpressionToBooleanOpsExpression(IdentityMapper[[]]):
    """
    Mapper to convert expressions into composition of boolean operation nodes
    according to C-semantics.

    For ex.:
        - ``i`` becomes ``i != 0``
        - ``i>10 and j`` becomes ``i>10 and j!=0``
    """

    @override
    def map_comparison(self, expr: p.Comparison) -> Expression:
        return expr

    def _get_expr_neq_0(self, expr: (
                p.Variable | p.Subscript | p.Power
                | p.Sum | p.Product | p.Call)) -> Expression:
        return p.Comparison(expr, "!=", 0)

    @override
    def map_constant(self, expr: object) -> Expression:
        return p.Comparison(expr, "!=", 0)

    @override
    def map_variable(self, expr: p.Variable) -> Expression:
        return self._get_expr_neq_0(expr)

    @override
    def map_subscript(self, expr: p.Subscript) -> Expression:
        return self._get_expr_neq_0(expr)

    @override
    def map_sum(self, expr: p.Sum) -> Expression:
        return self._get_expr_neq_0(expr)

    @override
    def map_product(self, expr: p.Product) -> Expression:
        return self._get_expr_neq_0(expr)

    @override
    def map_call(self, expr: p.Call) -> Expression:
        return self._get_expr_neq_0(expr)

    @override
    def map_power(self, expr: p.Power) -> Expression:
        return self._get_expr_neq_0(expr)

    @override
    def map_reduction(self, expr: Reduction) -> Expression:
        raise ExpressionToAffineConversionError("cannot (yet) convert reduction "
                "to affine")


@dataclass
class AffineConditionToISLSetMapper(Mapper[isl.Set, []]):
    """
    Mapper to convert a condition :class:`~pymbolic.primitives.Expression` to a
    :class:`~islpy.Set`.
    """

    space: isl.Space

    @override
    def map_comparison(self, expr: p.Comparison) -> isl.Set:
        if expr.operator == "!=":
            return self.rec(p.LogicalNot(p.Comparison(expr.left, "==", expr.right)))

        left_aff = guarded_aff_from_expr(self.space, expr.left)
        right_aff = guarded_aff_from_expr(self.space, expr.right)

        if expr.operator == "==":
            cnst = isl.Constraint.equality_from_aff(left_aff-right_aff)
        elif expr.operator == ">=":
            cnst = isl.Constraint.inequality_from_aff(left_aff-right_aff)
        elif expr.operator == ">":
            cnst = isl.Constraint.inequality_from_aff(left_aff-right_aff-1)
        elif expr.operator == "<=":
            cnst = isl.Constraint.inequality_from_aff(right_aff-left_aff)
        elif expr.operator == "<":
            cnst = isl.Constraint.inequality_from_aff(right_aff-left_aff-1)
        else:
            raise AssertionError()

        return isl.Set.universe(self.space).add_constraint(cnst)

    def _map_logical_reduce(self,
                            expr: p.LogicalOr | p.LogicalAnd,
                            f: Callable[[isl.Set, isl.Set], isl.Set]) -> isl.Set:
        """
        :arg f: Reduction callable.
        """
        sets = [self.rec(child) for child in expr.children]
        return reduce(f, sets)

    @override
    def map_logical_or(self, expr: p.LogicalOr) -> isl.Set:
        import operator
        return self._map_logical_reduce(expr, operator.or_)

    @override
    def map_logical_and(self, expr: p.LogicalAnd) -> isl.Set:
        import operator
        return self._map_logical_reduce(expr, operator.and_)

    @override
    def map_logical_not(self, expr: p.LogicalNot) -> isl.Set:
        set_ = self.rec(expr.child)
        return set_.complement()


def isl_set_from_expr(space: isl.Space, expr: Expression) -> isl.Set:
    """
    :arg expr: An instance of :class:`pymbolic.primitives.Expression` whose
        boolean value is evaluated according to C-semantics.
    """
    mapper = AffineConditionToISLSetMapper(space)
    expr = ConditionExpressionToBooleanOpsExpression()(expr)
    set_ = mapper(expr)
    assert isinstance(set_, isl.Set)

    return set_


def condition_to_set(space: isl.Space, expr: Expression) -> isl.Set | None:
    """
    Returns an instance of :class:`islpy.Set` if *expr* can be expressed as an
    :class:`isl.Set` on *space*, if not then returns *None*.
    """
    from loopy.symbolic import get_dependencies
    if get_dependencies(expr) <= frozenset(
            space.get_var_dict()):
        try:
            from loopy.symbolic import isl_set_from_expr
            return isl_set_from_expr(space, expr)
        except ExpressionToAffineConversionError:
            # non-affine condition: can't do much
            return None
    else:
        # data-dependent condition: can't do much
        return None

# }}}


# {{{ set_to_cond_expr

def basic_set_to_cond_expr(isl_basicset: isl.BasicSet) -> Expression:
    constrs: list[Expression] = []
    for constr in isl_basicset.get_constraints():
        constrs.append(constraint_to_cond_expr(constr))

    if len(constrs) == 0:
        raise ValueError("may not be called on universe")
    elif len(constrs) == 1:
        constr, = constrs
        return constr
    else:
        return p.LogicalAnd(tuple(constrs))


def set_to_cond_expr(isl_set: isl.Set) -> Expression:
    conjs: list[Expression] = []
    for isl_basicset in isl_set.get_basic_sets():
        conjs.append(basic_set_to_cond_expr(isl_basicset))

    if len(conjs) == 0:
        raise ValueError("may not be called on universe")
    elif len(conjs) == 1:
        conj, = conjs
        return conj
    else:
        return p.LogicalOr(tuple(conjs))


# }}}


# {{{ Reduction callback mapper

CallbackType: TypeAlias = """Callable[
    Concatenate[Expression,
                Callable[Concatenate[Expression, P], Expression],
                P], Expression | None]"""


class ReductionCallbackMapper(UncachedIdentityMapper[P]):
    callback: CallbackType[P]

    def __init__(self, callback: CallbackType[P]) -> None:
        self.callback = callback
        super().__init__()

    @override
    def map_reduction(self, expr: Reduction,
                      *args: P.args, **kwargs: P.kwargs) -> Expression:
        result = self.callback(expr, self.rec, *args, **kwargs)
        if result is None:
            return super().map_reduction(expr, *args, **kwargs)
        return result

# }}}


# {{{ index dependency finding

class IndexVariableFinder(CombineMapper[Set[str], []]):
    include_reduction_inames: bool

    def __init__(self, include_reduction_inames: bool) -> None:
        super().__init__()
        self.include_reduction_inames = include_reduction_inames

    @override
    def combine(self, values: Iterable[Set[str]]) -> Set[str]:
        result = set()
        for value in values:
            result |= value

        return result

    @override
    def map_constant(self, expr: object) -> Set[str]:
        return set()

    @override
    def map_algebraic_leaf(self, expr: p.AlgebraicLeaf) -> Set[str]:
        return set()

    @override
    def map_subscript(self, expr: p.Subscript) -> Set[str]:
        idx_vars = DependencyMapper()(expr.index)

        result: set[str] = set()
        for idx_var in idx_vars:
            if isinstance(idx_var, p.Variable):
                result.add(idx_var.name)
            else:
                raise RuntimeError("index variable not understood: %s" % idx_var)

        return result

    @override
    def map_reduction(self, expr: Reduction) -> Set[str]:
        result = self.rec(expr.expr)

        if not (expr.inames_set & result):
            raise RuntimeError("reduction '%s' does not depend on "
                    "reduction inames (%s)" % (expr, ",".join(expr.inames)))

        if self.include_reduction_inames:
            return result
        else:
            return result - expr.inames_set

# }}}


# {{{ wildcard -> unique variable mapper

class WildcardToUniqueVariableMapper(IdentityMapper[[]]):
    unique_var_name_factory: Callable[[], str]

    def __init__(self, unique_var_name_factory: Callable[[], str]) -> None:
        self.unique_var_name_factory = unique_var_name_factory
        super().__init__()

    @override
    def map_wildcard(self, expr: p.Wildcard) -> Expression:
        from pymbolic import var
        return var(self.unique_var_name_factory())

# }}}


# {{{ prime ("'") adder

class PrimeAdder(IdentityMapper[[]]):
    which_vars: Collection[str]

    def __init__(self, which_vars: Collection[str]) -> None:
        self.which_vars = which_vars
        super().__init__()

    @override
    def map_variable(self, expr: p.Variable) -> Expression:
        from pymbolic import var
        if expr.name in self.which_vars:
            return var(f"{expr.name}'")
        else:
            return expr

    @override
    def map_tagged_variable(self, expr: TaggedVariable) -> Expression:
        if expr.name in self.which_vars:
            return TaggedVariable(f"{expr.name}'", expr.tags)
        else:
            return expr

# }}}


# {{{ get access range

def get_access_map(
            domain: isl.Set,
            subscript: tuple[Expression, ...],
            assumptions: isl.BasicSet | None = None,
            shape: ShapeType | None = None,
            allowed_constant_names: Collection[str] | None = None
        ) -> isl.Map:
    """
    Returns an instance of :class:`isl.Map` accessed by *subscript*.

    :arg subscript: An instance of :class:`tuple` of index expressions.
    :arg assumptions: An instance of :class:`islpy.BasicSet` or *None*. *None*
        is equivalent to the universal set over *domain*'s space.
    :arg shape: if not *None*, indicates that it is desired to return an
        overestimate of the access range based on the shape if a precise range
        cannot be determined.
    :arg allowed_constant_names: An iterable of names of constants that are to be
        permitted in the access range expressions. Names that are already
        parameters of *domain* may be repeated without ill effects.

    ::
        >>> import islpy as isl
        >>> from loopy.symbolic import get_access_map
        >>> from pymbolic import var
        >>> get_access_map(isl.BasicSet("[n]->{[i, j]: 0<=i<n and i<=j<n}"),
                           (var("i") + 1, var("j")))
        >>> Map("[n]->{[i, j]->[i+1, j]: 0<=i<n and i<=j<n}")
    """

    if assumptions is not None:
        domain, assumptions = isl.align_two(domain,
                assumptions)
        domain = domain & assumptions
        del assumptions

    dims = len(subscript)

    # we build access_map as a set because (idiocy!) Affs
    # cannot live on maps.

    # dims: [domain](dn)[storage]
    access_map = domain

    if isinstance(access_map, isl.BasicSet):
        access_map = isl.Set.from_basic_set(access_map)

    if allowed_constant_names is not None:
        allowed_constant_names = set(allowed_constant_names) - {
                access_map.get_dim_name(isl.dim_type.param, i)
                for i in range(access_map.dim(isl.dim_type.param))}

        par_base = access_map.dim(isl.dim_type.param)
        access_map = access_map.insert_dims(isl.dim_type.param, par_base,
                len(allowed_constant_names))
        for i, const_name in enumerate(allowed_constant_names):
            access_map = access_map.set_dim_name(
                    isl.dim_type.param, par_base+i, const_name)

    dn = access_map.dim(isl.dim_type.set)
    access_map = access_map.insert_dims(isl.dim_type.set, dn, dims)

    from loopy.diagnostic import ExpressionToAffineConversionError

    for idim in range(dims):
        idx_aff = None

        try:
            idx_aff = guarded_aff_from_expr(access_map.space, subscript[idim])
        except ExpressionToAffineConversionError as err:
            shape_aff = None

            if shape is not None and shape[idim] is not None:
                with contextlib.suppress(ExpressionToAffineConversionError):
                    shape_aff = guarded_aff_from_expr(access_map.space, shape[idim])

            if shape_aff is None:
                # failed to convert shape[idim] to aff
                raise UnableToDetermineAccessRangeError(
                        "unable to determine access range of subscript: [%s] "
                        "(encountered %s: %s)"
                        % (", ".join(str(si) for si in subscript),
                            # intentionally using 'outer' err
                            type(err).__name__, str(err))) from err

            # successfully converted shape[idim] to aff, but not subscript[idim]

            upper_bound_cns = isl.Constraint.inequality_from_aff(
                    shape_aff.set_coefficient_val(
                        isl.dim_type.in_, dn+idim, -1) - 1)
            lower_bound_cns = isl.Constraint.inequality_from_aff(
                    isl.Aff.zero_on_domain(access_map.space).set_coefficient_val(
                        isl.dim_type.in_, dn+idim, 1))

            access_map = access_map.add_constraint(upper_bound_cns)
            access_map = access_map.add_constraint(lower_bound_cns)

        else:
            # successfully converted subscript[idim] -> idx_aff

            idx_aff = idx_aff.set_coefficient_val(
                    isl.dim_type.in_, dn+idim, -1)

            access_map = access_map.add_constraint(
                    isl.Constraint.equality_from_aff(idx_aff))

    access_map_as_map = isl.Map.universe(access_map.get_space())
    access_map_as_map = access_map_as_map.intersect_range(access_map)
    access_map = access_map_as_map.move_dims(
            isl.dim_type.in_, 0,
            isl.dim_type.out, 0, dn)

    return access_map

# }}}


# {{{ access range mapper

class BatchedAccessMapMapper(WalkMapper[[Set[str]]]):
    kernel: LoopKernel
    access_maps: defaultdict[str, defaultdict[Set[str], isl.Map | None]]
    bad_subscripts: defaultdict[str, list[p.Subscript | LinearSubscript]]
    _overestimate: bool
    _var_names: Collection[str]

    def __init__(self,
                kernel: LoopKernel,
                var_names: Collection[str],
                overestimate: bool = False
            ):
        self.kernel = kernel
        self.access_maps = defaultdict(lambda: defaultdict(lambda: None))
        self.bad_subscripts = defaultdict(list)
        self._overestimate = overestimate
        self._var_names = set(var_names)
        super().__init__()

    def get_access_range(self, var_name: str) -> isl.Set | None:
        loops_to_amaps = self.access_maps[var_name]
        if not loops_to_amaps:
            return None

        import operator
        from functools import reduce
        return reduce(operator.or_,
                      (not_none(val).range() for val in loops_to_amaps.values()))

    @override
    def map_subscript(self, expr: p.Subscript, inames: Set[str]) -> None:
        domain = self.kernel.get_inames_domain(inames)
        super().map_subscript(expr, inames)

        assert isinstance(expr.aggregate, p.Variable)

        if expr.aggregate.name not in self._var_names:
            return

        if expr.aggregate.name in self.bad_subscripts:
            return

        arg_name = expr.aggregate.name
        subscript = expr.index_tuple

        descriptor = self.kernel.get_var_descriptor(arg_name)

        try:
            access_map = get_access_map(
                    domain.to_set(), subscript, self.kernel.assumptions,
                    shape=cast("ShapeType | None", descriptor.shape)
                        if self._overestimate else None,
                    allowed_constant_names=self.kernel.get_unwritten_value_args())
        except UnableToDetermineAccessRangeError:
            self.bad_subscripts[arg_name].append(expr)
            return

        # {{{ check that the access' dimensionality matches previously seen accesses

        if self.access_maps[arg_name]:
            other_access_map = next(iter(self.access_maps[arg_name].values()))
            assert other_access_map is not None

            if (other_access_map.dim(isl.dim_type.set)
                    != access_map.dim(isl.dim_type.set)):
                raise RuntimeError(
                        "error while determining shape of argument '%s': "
                        "varying number of indices encountered"
                        % arg_name)

        # }}}

        if self.access_maps[arg_name][inames] is None:
            self.access_maps[arg_name][inames] = access_map
        else:
            self.access_maps[arg_name][inames] = (
                not_none(self.access_maps[arg_name][inames]) | access_map)

    @override
    def map_linear_subscript(
                self,
                expr: LinearSubscript, inames: Set[str]
            ) -> None:
        self.rec(expr.index, inames)

        assert isinstance(expr.aggregate, p.Variable)
        if expr.aggregate.name in self._var_names:
            self.bad_subscripts[expr.aggregate.name].append(expr)

    @override
    def map_reduction(self, expr: Reduction, inames: Set[str]) -> None:
        return WalkMapper.map_reduction(self, expr, inames | set(expr.inames))

    @override
    def map_sub_array_ref(self, expr: SubArrayRef, inames: Set[str]) -> None:
        assert isinstance(expr.subscript.aggregate, p.Variable)
        arg_name = expr.subscript.aggregate.name
        if arg_name not in self._var_names:
            return

        if arg_name in self.bad_subscripts:
            return

        total_inames = inames | {iname.name for iname in expr.swept_inames}
        assert total_inames not in self.access_maps[arg_name]

        self.rec(expr.subscript, total_inames)

        # {{{ project out swept_inames as within inames they are swept locally

        amap = self.access_maps[arg_name].pop(total_inames)
        for iname in expr.swept_inames:
            assert amap is not None
            dt, pos = amap.get_var_dict()[iname.name]
            amap = amap.project_out(dt, pos, 1)

        # }}}

        if self.access_maps[arg_name][inames] is None:
            self.access_maps[arg_name][inames] = amap
        else:
            self.access_maps[arg_name][inames] |= amap


class AccessRangeMapper:
    """**IMPORTANT**

    Using this class *will likely* lead to performance bottlenecks.

    To avoid performance issues, rewrite your code to use
    BatchedAccessMapMapper if at all possible.

    For *n* variables and *m* expressions, calling this class to compute the
    access ranges will take *O(mn)* time for traversing the expressions.

    BatchedAccessMapMapper does the same traversal in *O(m + n)* time.
    """

    var_name: str
    inner_mapper: BatchedAccessMapMapper

    def __init__(self,
                 kernel: LoopKernel,
                 var_name: str,
                 overestimate: bool = False) -> None:
        self.var_name = var_name
        self.inner_mapper = BatchedAccessMapMapper(kernel, [var_name], overestimate)

    def __call__(self, expr: Expression, inames: Set[str]) -> None:
        return self.inner_mapper(expr, inames)

    @property
    def access_range(self) -> isl.Set | None:
        return self.inner_mapper.get_access_range(self.var_name)

    @property
    def bad_subscripts(self) -> Sequence[p.Subscript | LinearSubscript]:
        return self.inner_mapper.bad_subscripts[self.var_name]

# }}}


# {{{ check if access ranges overlap

@dataclass
class AccessRangeOverlapChecker:
    """Used for checking for overlap between access ranges of instructions."""

    kernel: LoopKernel

    @cached_property
    def vars(self) -> Set[str]:
        return (self.kernel.get_written_variables()
                | self.kernel.get_read_variables())

    @memoize_method
    def _get_access_ranges(self,
                insn_id: InsnId,
                access_dir: TypingLiteral["w"] | TypingLiteral["any"]
            ):
        insn = self.kernel.id_to_insn[insn_id]

        exprs: list[Expression] = list(insn.assignees)
        if access_dir == "any":
            exprs.append(insn.expression)
            exprs.extend(insn.predicates)

        from collections import defaultdict
        ranges: defaultdict[str, bool | isl.Set | None] = defaultdict(lambda: False)

        arm = BatchedAccessMapMapper(self.kernel, self.vars, overestimate=True)

        for expr in exprs:
            arm(expr, insn.within_inames)

        for name in arm.access_maps:
            if arm.bad_subscripts[name]:
                ranges[name] = True
                continue
            ranges[name] = arm.get_access_range(name)

        return ranges

    def _get_access_range_for_var(self,
                insn_id: str,
                access_dir: TypingLiteral["w"] | TypingLiteral["any"],
                var_name: str
            ):
        assert access_dir in ["w", "any"]

        insn = self.kernel.id_to_insn[insn_id]
        # Access range checks only apply to assignment-style instructions. For
        # non-assignments, we rely on read/write dependency information.
        from loopy.kernel.instruction import MultiAssignmentBase
        if not isinstance(insn, MultiAssignmentBase):
            if access_dir == "any":
                return var_name in insn.dependency_names()
            else:
                return var_name in insn.assignee_var_names()

        return self._get_access_ranges(insn_id, access_dir)[var_name]

    def do_access_ranges_overlap_conservative(self,
                insn1: InsnId,
                insn1_dir: TypingLiteral["w"] | TypingLiteral["any"],
                insn2: InsnId,
                insn2_dir: TypingLiteral["w"] | TypingLiteral["any"],
                var_name: str,
            ):
        """Determine whether the access ranges to *var_name* in the two
        given instructions overlap. This determination is made 'conservatively',
        i.e. if precise information is unavailable, it is concluded that the
        ranges overlap.

        :arg insn1_dir: either ``"w"`` or ``"any"``, to indicate which
            type of access is desired--writing or any
        :arg insn2_dir: either ``"w"`` or ``"any"``
        :returns: a :class:`bool`
        """

        insn1_arange = self._get_access_range_for_var(insn1, insn1_dir, var_name)
        insn2_arange = self._get_access_range_for_var(insn2, insn2_dir, var_name)

        if insn1_arange is False or insn2_arange is False:
            return False
        if insn1_arange is True or insn2_arange is True:
            return True

        assert insn1_arange is not None
        assert insn2_arange is not None

        return not (insn1_arange & insn2_arange).is_empty()

# }}}


# {{{ is_expression_equal

def is_expression_equal(a: Expression | None, b: Expression | None) -> bool:
    if a == b:
        return True

    if isinstance(a, p.ExpressionNode) or isinstance(b, p.ExpressionNode):
        if a is None or b is None:
            return False

        assert p.is_arithmetic_expression(a)
        assert p.is_arithmetic_expression(b)
        maybe_zero = a - b
        from pymbolic import distribute

        d_result = distribute(maybe_zero)
        return d_result == 0

    else:
        return False


def is_tuple_of_expressions_equal(
            a: Expression | None,
            b: Expression | None,
        ) -> bool:
    if a is None or b is None:
        return bool(a is None and b is None)

    if not isinstance(a, tuple):
        a = (a,)

    if not isinstance(b, tuple):
        b = (b,)

    if len(a) != len(b):
        return False

    return all(
        is_expression_equal(ai, bi)
        for ai, bi in zip(a, b, strict=True))

# }}}

# vim: foldmethod=marker

from __future__ import annotations


__copyright__ = "Copyright (C) 2018 Andreas Kloeckner, Kaushik Kulkarni"

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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from warnings import warn

from immutabledict import immutabledict
from typing_extensions import Self

from loopy.diagnostic import LoopyError
from loopy.kernel.array import ArrayBase, ArrayDimImplementationTag
from loopy.kernel.data import AddressSpace, ArrayArg, ValueArg
from loopy.symbolic import DependencyMapper, WalkMapper


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import Self

    from loopy.kernel import LoopKernel
    from loopy.translation_unit import CallablesTable, FunctionIdT
    from loopy.types import LoopyType
    from loopy.typing import ShapeType

__doc__ = """
.. currentmodule:: loopy.kernel.function_interface

.. autoclass:: ArgDescriptor
.. autoclass:: ValueArgDescriptor
.. autoclass:: ArrayArgDescriptor

.. currentmodule:: loopy

.. autoclass:: InKernelCallable
.. autoclass:: CallableKernel
.. autoclass:: ScalarCallable
"""


# {{{ argument descriptors

ArgDescriptorT = TypeVar("ArgDescriptorT", bound="ArgDescriptor")


class ArgDescriptor(ABC):
    @abstractmethod
    def map_expr(
                self,
                subst_mapper: Callable[[ArgDescriptorT], ArgDescriptorT]
            ) -> Self:
        ...

    @abstractmethod
    def depends_on(self) -> frozenset[str]:
        ...

    @abstractmethod
    def copy(self, **kwargs: Any) -> Self:
        ...


@dataclass(frozen=True)
class ValueArgDescriptor(ArgDescriptor):

    def map_expr(self, subst_mapper):
        return self

    def depends_on(self):
        return frozenset()

    def copy(self, **kwargs: Any) -> Self:
        return replace(self, **kwargs)


@dataclass(frozen=True)
class ArrayArgDescriptor(ArgDescriptor):
    """
    Records information about an array argument to an in-kernel callable. To be
    passed to and returned from
    :meth:`~loopy.InKernelCallable.with_descrs`, used for
    matching shape and address space of caller and callee kernels.

    .. autoattribute:: shape
    .. autoattribute:: address_space
    .. autoattribute:: dim_tags

    .. automethod:: map_expr
    .. automethod:: depends_on
    """

    shape: ShapeType | None
    address_space: AddressSpace
    dim_tags: Sequence[ArrayDimImplementationTag] | None
    """See :ref:`data-dim-tags`.
    """

    if __debug__:
        def __post_init__(self):
            # {{{ sanity checks

            from loopy.kernel.array import ArrayDimImplementationTag
            from loopy.kernel.data import auto

            assert isinstance(self.shape, tuple) or self.shape in [None, auto]
            assert isinstance(self.dim_tags, tuple) or self.dim_tags is None

            if self.dim_tags:
                # FIXME at least vector dim tags should be supported
                assert all(isinstance(dim_tag, ArrayDimImplementationTag) for dim_tag in
                        self.dim_tags)

            # }}}

    def copy(self, **kwargs: Any) -> Self:
        return replace(self, **kwargs)

    def map_expr(self, f):
        """
        Returns an instance of :class:`ArrayArgDescriptor` with its shapes, strides,
        mapped by *f*.
        """
        if self.shape is not None:
            new_shape = tuple(f(axis_len) for axis_len in self.shape)
        else:
            new_shape = None

        if self.dim_tags is not None:
            new_dim_tags = tuple(dim_tag.map_expr(f) for dim_tag in self.dim_tags)
        else:
            new_dim_tags = None

        return self.copy(shape=new_shape, dim_tags=new_dim_tags)

    def depends_on(self):
        """
        Returns :class:`frozenset` of all the variable names the
        :class:`ArrayArgDescriptor` depends on.
        """
        from loopy.kernel.data import auto
        result = set()

        if self.shape:
            dep_mapper = DependencyMapper(composite_leaves=False)
            for axis_len in self.shape:
                if axis_len not in [None, auto]:
                    result |= dep_mapper(axis_len)

        if self.dim_tags:
            for dim_tag in self.dim_tags:
                result |= dim_tag.depends_on()

        return frozenset(var.name for var in result)


class ExpressionIsScalarChecker(WalkMapper):
    def __init__(self, kernel):
        self.kernel = kernel
        super().__init__()

    def map_sub_array_ref(self, expr):
        raise LoopyError("Sub-array refs can only be used as call's parameters"
                f" or assignees. '{expr}' violates this.")

    def map_call(self, expr):
        self.rec(expr.parameters)

    def map_subscript(self, expr):
        for child in expr.index_tuple:
            self.rec(child)

    def map_variable(self, expr):
        from loopy.kernel.data import ArrayArg, TemporaryVariable, auto
        if expr.name in self.kernel.all_inames():
            # inames are scalar
            return

        var = self.kernel.arg_dict.get(expr.name, None) or (
                self.kernel.temporary_variables.get(expr.name, None))

        if var is not None:
            if isinstance(var, (ArrayArg, TemporaryVariable)) and (
                    var.shape != () and var.shape is not auto):
                raise LoopyError("Array regions can only passed as sub-array refs.")

    def map_slice(self, expr):
        raise LoopyError("Array regions can only passed as sub-array refs.")

    def map_call_with_kwargs(self, expr):
        # See https://github.com/inducer/loopy/pull/323
        raise NotImplementedError


def get_arg_descriptor_for_expression(kernel, expr):
    """
    :returns: a :class:`ArrayArgDescriptor` or a :class:`ValueArgDescriptor`
        describing the argument expression *expr* which occurs
        in a call in the code of *kernel*.
    """
    from loopy.kernel.data import ArrayArg, TemporaryVariable
    from loopy.symbolic import SubArrayRef, SweptInameStrideCollector, pw_aff_to_expr

    if isinstance(expr, SubArrayRef):
        name = expr.subscript.aggregate.name
        arg = kernel.get_var_descriptor(name)

        if not isinstance(arg, (TemporaryVariable, ArrayArg)):
            raise LoopyError("unsupported argument type "
                    "'%s' of '%s' in call statement"
                    % (type(arg).__name__, expr.name))

        aspace = arg.address_space

        from loopy.kernel.array import FixedStrideArrayDimTag as DimTag
        sub_dim_tags = []
        sub_shape = []

        # This helps in identifying identities like
        # "2*(i//2) + i%2" := "i"
        # See the kernel in
        # test_callables.py::test_shape_translation_through_sub_array_refs

        from loopy.symbolic import simplify_using_aff
        linearized_index = simplify_using_aff(
                kernel,
                sum(dim_tag.stride*iname for dim_tag, iname in
                    zip(arg.dim_tags, expr.subscript.index_tuple)))

        strides_as_dict = SweptInameStrideCollector(
                tuple(iname.name for iname in expr.swept_inames)
                )(linearized_index)
        sub_dim_tags = tuple(
                # Not all swept inames necessarily occur in the expression.
                DimTag(strides_as_dict.get(iname, 0))
                for iname in expr.swept_inames)
        sub_shape = tuple(
                pw_aff_to_expr(
                    kernel.get_iname_bounds(iname.name).upper_bound_pw_aff
                    - kernel.get_iname_bounds(iname.name).lower_bound_pw_aff)+1
                for iname in expr.swept_inames)

        return ArrayArgDescriptor(
                address_space=aspace,
                dim_tags=sub_dim_tags,
                shape=sub_shape)
    else:
        ExpressionIsScalarChecker(kernel)(expr)
        return ValueArgDescriptor()

# }}}


# {{{ helper function for in-kernel callables

def get_kw_pos_association(kernel):
    """
    Returns a tuple of ``(kw_to_pos, pos_to_kw)`` for the arguments in
    *kernel*.
    """
    kw_to_pos = {}
    pos_to_kw = {}

    read_count = 0
    write_count = -1

    for arg in kernel.args:
        if arg.is_output:
            kw_to_pos[arg.name] = write_count
            pos_to_kw[write_count] = arg.name
            write_count -= 1
        if arg.is_input:
            # if an argument is both input and output then kw_to_pos is
            # overwritten with its expected position in the parameters
            kw_to_pos[arg.name] = read_count
            pos_to_kw[read_count] = arg.name
            read_count += 1

    return kw_to_pos, pos_to_kw

# }}}


# {{{ template class

@dataclass(frozen=True, init=False)
class InKernelCallable(ABC):
    """
    An abstract interface to define a callable encountered in a kernel.

    .. autoattribute:: name
    .. autoattribute:: arg_id_to_dtype
    .. autoattribute:: arg_id_to_descr

    .. automethod:: __init__
    .. automethod:: with_types
    .. automethod:: with_descrs
    .. automethod:: generate_preambles
    .. automethod:: emit_call
    .. automethod:: emit_call_insn
    .. automethod:: is_ready_for_codegen
    .. automethod:: get_hw_axes_sizes
    .. automethod:: get_used_hw_axes
    .. automethod:: get_called_callables
    .. automethod:: with_name
    .. automethod:: is_type_specialized

    .. note::

        * "``arg_id`` can either be an instance of :class:`int` integer
          corresponding to the position of the argument or an instance of
          :class:`str` corresponding to the name of keyword argument accepted
          by the function.

        * Negative "arg_id" values ``-i`` in the mapping attributes indicate
          return value with (0-based) index *i*.

    """
    arg_id_to_dtype: Mapping[int | str, LoopyType] | None
    arg_id_to_descr: Mapping[int | str, ArgDescriptor] | None

    def __init__(self,
                 arg_id_to_dtype: Mapping[int | str, LoopyType] | None = None,
                 arg_id_to_descr: Mapping[int | str, ArgDescriptor] | None = None,
             ) -> None:
        try:
            hash(arg_id_to_dtype)
        except TypeError:
            arg_id_to_dtype = immutabledict(arg_id_to_dtype)
            warn("arg_id_to_dtype passed to InKernelCallable was not hashable. "
                 "This usage is deprecated and will stop working in 2026.",
                 DeprecationWarning, stacklevel=3)

        try:
            hash(arg_id_to_descr)
        except TypeError:
            arg_id_to_descr = immutabledict(arg_id_to_descr)
            warn("arg_id_to_descr passed to InKernelCallable was not hashable. "
                 "This usage is deprecated and will stop working in 2026.",
                 DeprecationWarning, stacklevel=3)

        object.__setattr__(self, "arg_id_to_dtype", arg_id_to_dtype)
        object.__setattr__(self, "arg_id_to_descr", arg_id_to_descr)

    if TYPE_CHECKING:
        @property
        def name(self) -> str:
            raise NotImplementedError()

    def copy(self, **kwargs: Any) -> Self:
        return replace(self, **kwargs)

    def with_types(self, arg_id_to_dtype, clbl_inf_ctx):
        """
        :arg arg_id_to_type: a mapping from argument identifiers (integers for
            positional arguments) to :class:`loopy.types.LoopyType` instances.
            Unspecified/unknown types are not represented in *arg_id_to_type*.

            Return values are denoted by negative integers, with the first
            returned value identified as *-1*.

        :arg clbl_inf_ctx: An instance of
            :class:`loopy.translation_unit.CallablesInferenceContext`. *clbl_inf_ctx*
            provides the namespace of other callables contained within *self*.

        :returns: a tuple ``(new_self, new_clbl_inf_ctx)``, where *new_self* is a
            new :class:`InKernelCallable` specialized for the given types.
            *new_clbl_inf_ctx* is *clbl_inf_ctx*'s updated state if the
            type-specialization of *self* updated other calls contained within
            it.

        .. note::

            If the :class:`InKernelCallable` does not contain any
            other callables within it, then *clbl_inf_ctx* is returned as is.
        """

        raise NotImplementedError()

    @abstractmethod
    def with_descrs(self, arg_id_to_descr, clbl_inf_ctx):
        """
        :arg arg_id_to_descr: a mapping from argument identifiers (integers for
            positional arguments) to instances of
            :class:`~loopy.kernel.function_interface.ArrayArgDescriptor`
            or :class:`~loopy.kernel.function_interface.ValueArgDescriptor`.
            Unspecified/unknown descriptors are not represented in *arg_id_to_type*.

            Return values are denoted by negative integers, with the first
            returned value identified as *-1*.

        :arg clbl_inf_ctx: An instance of
            :class:`loopy.translation_unit.CallablesInferenceContext`. *clbl_inf_ctx*
            provides the namespace of other callables contained within *self*.

        :returns: a tuple ``(new_self, new_clbl_inf_ctx)``, where *new_self* is a
            new :class:`InKernelCallable` specialized for the given argument
            descriptors. *new_clbl_inf_ctx* is the *clbl_inf_ctx*'s updated state
            if descriptor-specialization of *self* updated other calls contained
            within it.

        .. note::

            If the :class:`InKernelCallable` does not contain any
            other callables within it, then *clbl_inf_ctx* is returned as is.
        """

    def is_ready_for_codegen(self) -> bool:
        return (self.arg_id_to_dtype is not None and
                self.arg_id_to_descr is not None)

    @abstractmethod
    def get_hw_axes_sizes(self, arg_id_to_arg, space, callables_table):
        """
        Returns ``gsizes, lsizes``, where *gsizes* and *lsizes* are mappings
        from axis indices to corresponding group or local hw axis sizes. The hw
        axes sizes are represented as instances of :class:`islpy.PwAff` on the
        given *space*.

        :arg arg_id_to_arg: A mapping from the passed argument *id* to the
            arguments at a call-site.
        :arg space: An instance of :class:`islpy.Space`.
        """
        ...

    @abstractmethod
    def get_used_hw_axes(self, callables_table):
        """
        Returns a tuple ``group_axes_used, local_axes_used``, where
        ``(group|local)_axes_used`` are :class:`frozenset` of hardware axes
        indices used by the callable.
        """

    @abstractmethod
    def generate_preambles(self, target):
        """
        Yields the target specific preamble.
        """
        raise NotImplementedError()

    @abstractmethod
    def emit_call(self, expression_to_code_mapper, expression, target):
        ...

    @abstractmethod
    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        """
        Returns a tuple of ``(call, assignee_is_returned)`` which is the target
        facing function call that would be seen in the generated code. ``call``
        is an instance of ``pymbolic.primitives.Call`` ``assignee_is_returned``
        is an instance of :class:`bool` to indicate if the assignee is returned
        by value of C-type targets.

        *Example:* If ``assignee_is_returned=True``, then ``a, b = f(c, d)`` is
            interpreted in the target as ``a = f(c, d, &b)``. If
            ``assignee_is_returned=False``, then ``a, b = f(c, d)`` is interpreted
            in the target as the statement ``f(c, d, &a, &b)``.
        """

    @abstractmethod
    def with_added_arg(self, arg_dtype, arg_descr):
        """
        Registers a new argument to the callable and returns the name of the
        argument in the callable's namespace.
        """

    @abstractmethod
    def get_called_callables(
                             self,
                             callables_table: CallablesTable,
                             recursive: bool = True
                         ) -> frozenset[FunctionIdT]:
        """
        Returns a :class:`frozenset` of callable ids called by *self* that are
        resolved via *callables_table*.

        :arg callables_table: Similar to
            :attr:`loopy.TranslationUnit.callables_table`.
        :arg recursive: If *True* recursively searches for all the called
            callables, else only returns the callables directly called by
            *self*.
        """

    @abstractmethod
    def with_name(self, name):
        """
        Returns a copy of *self* so that it could be referred by *name*
        in a :attr:`loopy.TranslationUnit.callables_table`'s namespace.
        """

    @abstractmethod
    def is_type_specialized(self):
        """
        Returns *True* iff *self*'s type signature is known, else returns
        *False*.
        """

# }}}


# {{{ scalar callable

@dataclass(frozen=True, init=False)
class ScalarCallable(InKernelCallable):
    """
    An abstract interface to a scalar callable encountered in a kernel.

    .. attribute:: name_in_target

        A :class:`str` to denote the name of the function in a
        :class:`loopy.target.TargetBase` for which the callable is specialized.
        *None* if the callable is not specialized enough to know its name
        in target.

    .. automethod:: with_types

    .. automethod:: with_descrs

    .. note::

        The :meth:`ScalarCallable.with_types` is intended to assist with type
        specialization of the function and sub-classes must define it.
    """
    name: str
    name_in_target: str | None

    def __init__(self,
                 name: str,
                 arg_id_to_dtype: Mapping[int | str, LoopyType] | None = None,
                 arg_id_to_descr: Mapping[int | str, ArgDescriptor] | None = None,
                 name_in_target: str | None = None) -> None:
        super().__init__(
            arg_id_to_dtype=arg_id_to_dtype,
            arg_id_to_descr=arg_id_to_descr,
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "name_in_target", name_in_target)

    def with_types(self, arg_id_to_dtype, callables_table):
        raise LoopyError("No type inference information present for "
                "the function %s." % (self.name))

    def with_descrs(self, arg_id_to_descr, clbl_inf_ctx):

        arg_id_to_descr[-1] = ValueArgDescriptor()
        return (self.copy(arg_id_to_descr=arg_id_to_descr),
                clbl_inf_ctx)

    def get_hw_axes_sizes(self, arg_id_to_arg, space, callables_table):
        return {}, {}

    def get_used_hw_axes(self, callables_table):
        return frozenset(), frozenset()

    def is_ready_for_codegen(self):

        return (self.arg_id_to_dtype is not None and
                self.arg_id_to_descr is not None)

    # {{{ code generation

    def emit_call(self, expression_to_code_mapper, expression, target):

        assert self.is_ready_for_codegen()

        # must have single assignee
        assert len(expression.parameters) == len(self.arg_id_to_dtype) - 1
        arg_dtypes = tuple(self.arg_id_to_dtype[id] for id in
                range(len(self.arg_id_to_dtype)-1))

        par_dtypes = tuple(expression_to_code_mapper.infer_type(par) for par in
                expression.parameters)

        from loopy.expression import dtype_to_type_context
        # processing the parameters with the required dtypes
        processed_parameters = tuple(
                expression_to_code_mapper.rec(par,
                    dtype_to_type_context(target, tgt_dtype),
                    tgt_dtype)
                for par, par_dtype, tgt_dtype in zip(
                    expression.parameters, par_dtypes, arg_dtypes))

        from pymbolic import var
        return var(self.name_in_target)(*processed_parameters)

    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        """
        :arg insn: An instance of :class:`loopy.kernel.instructions.CallInstruction`.
        :arg target: An instance of :class:`loopy.target.TargetBase`.
        :arg expression_to_code_mapper: An instance of :class:`IdentityMapper`
            responsible for code mapping from :mod:`loopy` syntax to the
            **target syntax**.

        :returns: A tuple of the call to be generated and an instance of
            :class:`bool` whether the first assignee is a part of the LHS in
            the assignment instruction.

        .. note::

            The default implementation returns the first assignees and the
            references of the rest of the assignees are appended to the
            arguments of the call.

            *Example:* ``c, d = f(a, b)`` is returned as ``c = f(a, b, &d)``
        """
        from loopy.target.c import CFamilyTarget
        if not isinstance(target, CFamilyTarget):
            raise NotImplementedError()

        from pymbolic import var
        from pymbolic.mapper.stringifier import PREC_NONE

        from loopy.expression import dtype_to_type_context
        from loopy.kernel.instruction import CallInstruction

        assert isinstance(insn, CallInstruction)
        assert self.is_ready_for_codegen()

        ecm = expression_to_code_mapper
        parameters = insn.expression.parameters
        assignees = insn.assignees[1:]

        par_dtypes = tuple(expression_to_code_mapper.infer_type(par)
                           for par in parameters)
        arg_dtypes = tuple(self.arg_id_to_dtype[i]
                           for i, _ in enumerate(parameters))

        assignee_dtypes = tuple(self.arg_id_to_dtype[-i-2]
                                for i, _ in enumerate(assignees))

        tgt_parameters = [ecm(par, PREC_NONE,
                              dtype_to_type_context(target, tgt_dtype),
                              tgt_dtype).expr
                          for par, par_dtype, tgt_dtype in zip(parameters,
                                                               par_dtypes,
                                                               arg_dtypes)]

        for a, tgt_dtype in zip(assignees, assignee_dtypes):
            if tgt_dtype != expression_to_code_mapper.infer_type(a):
                raise LoopyError("Type Mismatch in function %s. Expected: %s"
                        "Got: %s" % (self.name, tgt_dtype,
                            expression_to_code_mapper.infer_type(a)))
            tgt_parameters.append(var("&")(ecm(a, PREC_NONE,
                                               dtype_to_type_context(target,
                                                                     tgt_dtype),
                                               tgt_dtype).expr))

        # assignee is returned whenever the size of assignees is non zero.
        first_assignee_is_returned = len(insn.assignees) > 0

        return var(self.name_in_target)(*tgt_parameters), first_assignee_is_returned

    def generate_preambles(self, target):
        return
        yield

    # }}}

    def with_added_arg(self, arg_dtype, arg_descr):
        raise LoopyError("Cannot add args to scalar callables.")

    def get_called_callables(self, callables_table, recursive=True):
        """
        Returns a :class:`frozenset` of callable ids called by *self*.
        """
        return frozenset()

    def with_name(self, name):
        return self

    def is_type_specialized(self):
        return (self.arg_id_to_dtype is not None
                and all(dtype is not None
                        for dtype in self.arg_id_to_dtype.values()))

# }}}


# {{{ callable kernel

@dataclass(frozen=True, init=False)
class CallableKernel(InKernelCallable):
    """
    Records information about a callee kernel. Also provides interface through
    member methods to make the callee kernel compatible to be called from a
    caller kernel.

    :meth:`CallableKernel.with_types` should be called in order to match
    the ``dtypes`` of the arguments that are shared between the caller and the
    callee kernel.

    :meth:`CallableKernel.with_descrs` should be called in order to match
    the arguments' shapes/strides across the caller and the callee kernel.

    .. autoattribute:: subkernel
    .. automethod:: with_descrs
    .. automethod:: with_types
    """

    subkernel: LoopKernel

    def __init__(self,
                 subkernel: LoopKernel,
                 arg_id_to_dtype: Mapping[int | str, LoopyType] | None = None,
                 arg_id_to_descr: Mapping[int | str, ArgDescriptor] | None = None,
             ) -> None:

        super().__init__(
                         arg_id_to_dtype=arg_id_to_dtype,
                         arg_id_to_descr=arg_id_to_descr)
        object.__setattr__(self, "subkernel", subkernel)

    @property
    def name(self) -> str:
        return self.subkernel.name

    def with_types(self, arg_id_to_dtype, callables_table):
        kw_to_pos, pos_to_kw = get_kw_pos_association(self.subkernel)

        new_args = []
        for arg in self.subkernel.args:
            kw = arg.name
            if kw in arg_id_to_dtype:
                # id exists as kw
                new_args.append(arg.copy(dtype=arg_id_to_dtype[kw]))
            elif kw_to_pos[kw] in arg_id_to_dtype:
                # id exists as positional argument
                new_args.append(arg.copy(
                    dtype=arg_id_to_dtype[kw_to_pos[kw]]))
            else:
                new_args.append(arg)

        from loopy.type_inference import infer_unknown_types_for_a_single_kernel
        pre_specialized_subkernel = self.subkernel.copy(
                args=new_args)

        # infer the types of the written variables based on the knowledge
        # of the types of the arguments supplied
        specialized_kernel, callables_table = (
                infer_unknown_types_for_a_single_kernel(
                    pre_specialized_subkernel,
                    callables_table))

        new_arg_id_to_dtype = {}
        for pos, kw in pos_to_kw.items():
            arg = specialized_kernel.arg_dict[kw]
            if arg.dtype:
                new_arg_id_to_dtype[kw] = arg.dtype
                new_arg_id_to_dtype[pos] = arg.dtype

        # Return the kernel call with specialized subkernel and the corresponding
        # new arg_id_to_dtype
        return self.copy(subkernel=specialized_kernel,
                arg_id_to_dtype=immutabledict(new_arg_id_to_dtype)), callables_table

    def with_descrs(self, arg_id_to_descr, clbl_inf_ctx):

        # arg_id_to_descr expressions provided are from the caller's namespace,
        # need to register

        kw_to_pos, pos_to_kw = get_kw_pos_association(self.subkernel)

        kw_to_callee_idx = {arg.name: i
                            for i, arg in enumerate(self.subkernel.args)}

        new_args = self.subkernel.args[:]

        for arg_id, descr in arg_id_to_descr.items():
            if isinstance(arg_id, int):
                arg_id = pos_to_kw[arg_id]

            callee_arg = new_args[kw_to_callee_idx[arg_id]]

            # {{{ checks

            if isinstance(callee_arg, ValueArg) and (
                    isinstance(descr, ArrayArgDescriptor)):
                raise LoopyError(f"In call to {self.subkernel.name}, '{arg_id}' "
                        "expected to be a scalar, got an array region.")

            if isinstance(callee_arg, ArrayArg) and (
                    isinstance(descr, ValueArgDescriptor)):
                raise LoopyError(f"In call to {self.subkernel.name}, '{arg_id}' "
                        "expected to be an array, got a scalar.")

            if (isinstance(descr, ArrayArgDescriptor)
                    and isinstance(callee_arg.shape, tuple)
                    and len(callee_arg.shape) != len(descr.shape)):
                raise LoopyError(f"In call to {self.subkernel.name}, '{arg_id}'"
                        " has a dimensionality mismatch, expected "
                        f"{len(callee_arg.shape)}, got {len(descr.shape)}")

            # }}}

            if isinstance(descr, ArrayArgDescriptor):
                callee_arg = callee_arg.copy(shape=descr.shape,
                                             dim_tags=descr.dim_tags,
                                             address_space=descr.address_space)
            else:
                # do nothing for a scalar arg.
                assert isinstance(descr, ValueArgDescriptor)

            new_args[kw_to_callee_idx[arg_id]] = callee_arg

        subkernel = self.subkernel.copy(args=new_args)

        from loopy.preprocess import traverse_to_infer_arg_descr
        subkernel, clbl_inf_ctx = traverse_to_infer_arg_descr(subkernel,
                                                              clbl_inf_ctx)

        # {{{ update the arg descriptors

        for arg in subkernel.args:
            kw = arg.name
            if isinstance(arg, ArrayBase):
                arg_id_to_descr[kw] = (
                        ArrayArgDescriptor(shape=arg.shape,
                                           dim_tags=arg.dim_tags,
                                           address_space=arg.address_space))
            else:
                assert isinstance(arg, ValueArg)
                arg_id_to_descr[kw] = ValueArgDescriptor()

            arg_id_to_descr[kw_to_pos[kw]] = arg_id_to_descr[kw]

        # }}}

        return (self.copy(subkernel=subkernel,
                          arg_id_to_descr=immutabledict(arg_id_to_descr)),
                clbl_inf_ctx)

    def with_added_arg(self, arg_dtype, arg_descr):
        var_name = self.subkernel.get_var_name_generator()(based_on="_lpy_arg")

        if isinstance(arg_descr, ValueArgDescriptor):
            subknl = self.subkernel.copy(
                    args=[
                        *self.subkernel.args,
                        ValueArg(var_name, arg_dtype, self.subkernel.target)])

            kw_to_pos, _pos_to_kw = get_kw_pos_association(subknl)

            if self.arg_id_to_dtype is None:
                arg_id_to_dtype = {}
            else:
                arg_id_to_dtype = dict(self.arg_id_to_dtype)
            if self.arg_id_to_descr is None:
                arg_id_to_descr = {}
            else:
                arg_id_to_descr = dict(self.arg_id_to_descr)

            arg_id_to_dtype[var_name] = arg_dtype
            arg_id_to_descr[var_name] = arg_descr
            arg_id_to_dtype[kw_to_pos[var_name]] = arg_dtype
            arg_id_to_descr[kw_to_pos[var_name]] = arg_descr

            return (self.copy(subkernel=subknl,
                              arg_id_to_dtype=arg_id_to_dtype,
                              arg_id_to_descr=arg_id_to_descr),
                    var_name)

        else:
            # don't think this should ever be needed
            raise NotImplementedError("with_added_arg not implemented for array"
                    " types arguments.")

    def with_packing_for_args(self):
        from loopy.kernel.data import AddressSpace
        _kw_to_pos, pos_to_kw = get_kw_pos_association(self.subkernel)

        arg_id_to_descr = {}

        for pos, kw in pos_to_kw.items():
            arg = self.subkernel.arg_dict[kw]
            arg_id_to_descr[pos] = ArrayArgDescriptor(
                    shape=arg.shape,
                    dim_tags=arg.dim_tags,
                    address_space=AddressSpace.GLOBAL)

        return self.copy(subkernel=self.subkernel,
                arg_id_to_descr=arg_id_to_descr)

    def get_used_hw_axes(self, callables_table):
        gsize, lsize = self.subkernel.get_grid_size_upper_bounds(callables_table,
                                                                 return_dict=True)

        return frozenset(gsize.keys()), frozenset(lsize.keys())

    def get_hw_axes_sizes(self, arg_id_to_arg, space, callables_table):
        from loopy.isl_helpers import subst_into_pwaff
        _, pos_to_kw = get_kw_pos_association(self.subkernel)
        gsize, lsize = self.subkernel.get_grid_size_upper_bounds(callables_table,
                                                                 return_dict=True)

        subst_dict = {pos_to_kw[i]: val
                      for i, val in arg_id_to_arg.items()
                      if isinstance(self.subkernel.arg_dict[pos_to_kw[i]],
                                    ValueArg)}

        gsize = {iaxis: subst_into_pwaff(space, size, subst_dict)
                 for iaxis, size in gsize.items()}
        lsize = {iaxis: subst_into_pwaff(space, size, subst_dict)
                 for iaxis, size in lsize.items()}

        return gsize, lsize

    def is_ready_for_codegen(self):
        return (self.arg_id_to_dtype is not None
                and self.arg_id_to_descr is not None)

    def generate_preambles(self, target):
        """ Yields the *target* specific preambles.
        """
        return
        yield

    def emit_call(self, expression_to_code_mapper, expression, target):
        raise LoopyError("Kernel '{self.name}' cannot be called "
                         "from within an expression, use a call statement")

    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        from loopy.target.c import CFamilyTarget
        if not isinstance(target, CFamilyTarget):
            raise NotImplementedError()

        from loopy.kernel.instruction import CallInstruction

        assert self.is_ready_for_codegen()
        assert isinstance(insn, CallInstruction)

        ecm = expression_to_code_mapper
        parameters = insn.expression.parameters
        assignees = insn.assignees

        parameters = list(parameters)
        par_dtypes = [self.arg_id_to_dtype[i] for i, _ in enumerate(parameters)]
        _kw_to_pos, _pos_to_kw = get_kw_pos_association(self.subkernel)

        # insert the assignees at the required positions
        assignee_write_count = -1
        for i, arg in enumerate(self.subkernel.args):
            if arg.is_output:
                if not arg.is_input:
                    assignee = assignees[-assignee_write_count-1]
                    parameters.insert(i, assignee)
                    par_dtypes.insert(i, self.arg_id_to_dtype[assignee_write_count])

                assignee_write_count -= 1

        # no type casting in array calls
        from pymbolic import var
        from pymbolic.mapper.stringifier import PREC_NONE

        from loopy.expression import dtype_to_type_context

        tgt_parameters = [ecm(par, PREC_NONE, dtype_to_type_context(target,
                                                                    par_dtype),
                              par_dtype).expr
                          for par, par_dtype in zip(parameters, par_dtypes)]

        return var(self.subkernel.name)(*tgt_parameters), False

    def get_called_callables(self, callables_table, recursive=True):
        from loopy.kernel.tools import get_resolved_callable_ids_called_by_knl
        return get_resolved_callable_ids_called_by_knl(self.subkernel,
                                                       callables_table,
                                                       recursive=recursive)

    def with_name(self, name):
        new_knl = self.subkernel.copy(name=name)
        return self.copy(subkernel=new_knl)

    def is_type_specialized(self):
        from loopy.kernel.data import auto
        return (self.arg_id_to_dtype is not None
                and all(arg.dtype not in [None, auto]
                        for arg in self.subkernel.args)
                and all(tv.dtype not in [None, auto]
                        for tv in self.subkernel.temporary_variables.values()))

# }}}


# vim: foldmethod=marker

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

from pytools import ImmutableRecord
from loopy.diagnostic import LoopyError

from loopy.tools import update_persistent_hash
from loopy.kernel import LoopKernel
from loopy.kernel.array import ArrayBase
from loopy.kernel.data import ValueArg, ArrayArg
from loopy.symbolic import DependencyMapper, WalkMapper

__doc__ = """

.. currentmodule:: loopy

.. autoclass:: ValueArgDescriptor
.. autoclass:: ArrayArgDescriptor
.. autoclass:: InKernelCallable
.. autoclass:: CallableKernel
.. autoclass:: ScalarCallable

"""


# {{{ argument descriptors

class ValueArgDescriptor(ImmutableRecord):
    hash_fields = ()

    def map_expr(self, subst_mapper):
        return self.copy()

    def depends_on(self):
        return frozenset()

    update_persistent_hash = update_persistent_hash


class ArrayArgDescriptor(ImmutableRecord):
    """
    Records information about an array argument to an in-kernel callable. To be
    passed to and returned from
    :meth:`loopy.kernel.function_interface.InKernelCallable.with_descrs`, used for
    matching shape and address space of caller and callee kernels.

    ..attribute:: shape

        Shape of the array.

    .. attribute:: address_space

        An attribute of :class:`loopy.kernel.data.AddressSpace`.

    .. attribute:: dim_tags

        A tuple of instances of
        :class:`loopy.kernel.array.ArrayDimImplementationTag`

    .. automethod:: map_expr
    .. automethod:: depends_on
    """

    fields = {"shape", "address_space", "dim_tags"}

    def __init__(self, shape, address_space, dim_tags):

        # {{{ sanity checks

        from loopy.kernel.array import ArrayDimImplementationTag
        from loopy.kernel.data import auto

        assert isinstance(shape, tuple) or shape in [None, auto]
        assert isinstance(dim_tags, tuple) or dim_tags is None

        if dim_tags:
            # FIXME at least vector dim tags should be supported
            assert all(isinstance(dim_tag, ArrayDimImplementationTag) for dim_tag in
                    dim_tags)

        # }}}

        super().__init__(
                shape=shape,
                address_space=address_space,
                dim_tags=dim_tags)

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
        Returns class:`frozenset` of all the variable names the
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

    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.update_for_pymbolic_expression(key_hash, self.shape)
        key_builder.rec(key_hash, self.address_space)
        key_builder.rec(key_hash, self.dim_tags)


class ExpressionIsScalarChecker(WalkMapper):
    def __init__(self, kernel):
        self.kernel = kernel

    def map_sub_array_ref(self, expr):
        raise LoopyError("Sub-array refs can only be used as call's parameters"
                f" or assignees. '{expr}'violates this.")

    def map_call(self, expr):
        for child in expr.parameters:
            self.rec(child)

    def map_call_with_kwargs(self, expr):
        for child in expr.parameters + tuple(expr.kw_parameters.values()):
            self.rec(child)

    def map_subscript(self, expr):
        for child in expr.index_tuple:
            self.rec(child)

    def map_variable(self, expr):
        from loopy.kernel.data import TemporaryVariable, ArrayArg, auto
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


def get_arg_descriptor_for_expression(kernel, expr):
    """
    :returns: a :class:`ArrayArgDescriptor` or a :class:`ValueArgDescriptor`
        describing the argument expression *expr* which occurs
        in a call in the code of *kernel*.
    """
    from loopy.symbolic import (SubArrayRef, pw_aff_to_expr,
            SweptInameStrideCollector)
    from loopy.kernel.data import TemporaryVariable, ArrayArg

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

        # FIXME This blindly assumes that dim_tag has a stride and
        # will not work for non-stride dim tags (e.g. vec or sep).

        # (AK) FIXME: This will almost always be nonlinear--when does this
        # actually help? Maybe remove this?
        # (KK) Reply: This helps in identifying identities like
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


class GridOverrideForCalleeKernel(ImmutableRecord):
    """
    Helper class to set the
    :attr:`loopy.kernel.LoopKernel.override_get_grid_size_for_insn_ids` of the
    callee kernels. Refer to
    :meth:`loopy.kernel.function_interface.GridOverrideForCalleeKernel.__call__`,
    :meth:`loopy.kernel.function_interface.CallbleKernel.with_hw_axes_sizes`.

    .. attribute:: global_size

        The global work group size that to be set in the callee kernel.

    .. attribute:: local_size

        The local work group size that has to be set in the callee kernel.

    .. note::

        This class acts as a pseudo-callable and its significance lies in
        solving picklability issues.
    """
    fields = {"local_size", "global_size"}

    def __init__(self, global_size, local_size):
        self.global_size = global_size
        self.local_size = local_size

    def __call__(self, insn_ids, callables_table, ignore_auto=True):
        return self.global_size, self.local_size

# }}}


# {{{ template class

class InKernelCallable(ImmutableRecord):
    """
    An abstract interface to define a callable encountered in a kernel.

    .. attribute:: name

        The name of the callable which can be encountered within expressions in
        a kernel.

    .. attribute:: arg_id_to_dtype

        A mapping which indicates the arguments types and result types of the
        callable.

    .. attribute:: arg_id_to_descr

        A mapping which gives indicates the argument shape and ``dim_tags`` it
        would be responsible for generating code.

    .. note::
        - "``arg_id`` can either be an instance of :class:`int` integer
          corresponding to the position of the argument or an instance of
          :class:`str` corresponding to the name of keyword argument accepted
          by the function.

        - Negative "arg_id" values ``-i`` in the mapping attributes indicate
        return value with (0-based) index *i*.

    .. automethod:: __init__
    .. automethod:: with_types
    .. automethod:: with_descrs
    .. automethod:: with_target
    .. automethod:: with_hw_axes_sizes
    .. automethod:: generate_preambles
    .. automethod:: emit_call
    .. automethod:: emit_call_insn
    .. automethod:: is_ready_for_codegen
    """

    fields = {"arg_id_to_dtype", "arg_id_to_descr"}
    init_arg_names = ("arg_id_to_dtype", "arg_id_to_descr")

    def __init__(self, arg_id_to_dtype=None, arg_id_to_descr=None):

        super().__init__(
                arg_id_to_dtype=arg_id_to_dtype,
                arg_id_to_descr=arg_id_to_descr)

    def __getinitargs__(self):
        return (self.arg_id_to_dtype, self.arg_id_to_descr)

    update_persistent_hash = update_persistent_hash

    def with_types(self, arg_id_to_dtype, callables_table):
        """
        :arg arg_id_to_type: a mapping from argument identifiers
            (integers for positional arguments, names for keyword
            arguments) to :class:`loopy.types.LoopyType` instances.
            Unspecified/unknown types are not represented in *arg_id_to_type*.

            Return values are denoted by negative integers, with the
            first returned value identified as *-1*.

        :returns: a tuple ``(new_self, arg_id_to_type)``, where *new_self* is a
            new :class:`InKernelCallable` specialized for the given types,
            and *arg_id_to_type* is a mapping of the same form as the
            argument above, however it may have more information present.
            Any argument information exists both by its positional and
            its keyword identifier.
        """

        raise NotImplementedError()

    def with_descrs(self, arg_id_to_descr, callables_table):
        """
        :arg arg_id_to_descr: a mapping from argument identifiers (integers for
            positional arguments, names for keyword arguments) to
            :class:`loopy.ArrayArgDescriptor` instances.  Unspecified/unknown
            descriptors are not represented in *arg_id_to_descr*.

            All the expressions in arg_id_to_descr must have variables that belong
            to the callable's namespace.

            Return values are denoted by negative integers, with the
            first returned value identified as *-1*.

        :returns: a copy of *self* which is a new instance of
            :class:`InKernelCallable` specialized for the given types, and
            *arg_id_to_descr* is a mapping of the same form as the argument above,
            however it may have more information present.  Any argument information
            exists both by its positional and its keyword identifier.
        """

        raise NotImplementedError()

    def with_target(self, target):
        """
        Returns a copy of *self* with all the ``dtypes`` in
        ``in_knl_callable.arg_id_to_dtype`` associated with the *target*. Refer
        :meth:`loopy.types.LoopyType.with_target`.

        :arg target: An instance of :class:`loopy.target.TargetBase`.
        """

        if target is None:
            raise LoopyError("target cannot be None for with_target")

        def with_target_if_not_None(dtype):
            """
            Returns a copy of :arg:`dtype` associated with the target. If
            ``dtype`` is *None* returns *None*.
            """
            if dtype:
                return dtype.with_target(target)
            else:
                return None

        new_arg_id_to_dtype = None
        if self.arg_id_to_dtype is not None:
            new_arg_id_to_dtype = {id: with_target_if_not_None(dtype)
                                   for id, dtype in self.arg_id_to_dtype.items()}

        return self.copy(arg_id_to_dtype=new_arg_id_to_dtype)

    def with_hw_axes_sizes(self, global_size, local_size):
        """
        Returns a copy of *self* with modifications to comply with the grid
        sizes ``(local_size, global_size)`` of the program in which it is
        supposed to be called.

        :arg local_size: An instance of :class:`islpy.PwAff`.
        :arg global_size: An instance of :class:`islpy.PwAff`.
        """
        raise NotImplementedError()

    def is_ready_for_codegen(self):

        return (self.arg_id_to_dtype is not None and
                self.arg_id_to_descr is not None)

    def generate_preambles(self, target):
        """
        Yields the target specific preamble.
        """
        raise NotImplementedError()

    def emit_call(self, expression_to_code_mapper, expression, target):

        raise NotImplementedError()

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

        raise NotImplementedError()

    def __hash__(self):

        return hash(tuple(self.fields))

    def with_added_arg(self, arg_dtype, arg_descr):
        """
        Registers a new argument to the callable and returns the name of the
        argument in the callable's namespace.
        """
        raise NotImplementedError()

# }}}


# {{{ scalar callable

class ScalarCallable(InKernelCallable):
    """
    An abstract interface the to a scalar callable encountered in a kernel.

    .. note::

        The :meth:`ScalarCallable.with_types` is intended to assist with type
        specialization of the function and sub-classes must define it.
    """

    fields = {"name", "arg_id_to_dtype", "arg_id_to_descr", "name_in_target"}
    init_arg_names = ("name", "arg_id_to_dtype", "arg_id_to_descr",
            "name_in_target")
    hash_fields = ("name", "arg_id_to_dtype", "arg_id_to_descr", "name_in_target")

    def __init__(self, name, arg_id_to_dtype=None,
            arg_id_to_descr=None, name_in_target=None):

        super().__init__(
                arg_id_to_dtype=arg_id_to_dtype,
                arg_id_to_descr=arg_id_to_descr)

        self.name = name
        self.name_in_target = name_in_target

    def __getinitargs__(self):
        return (self.arg_id_to_dtype, self.arg_id_to_descr,
                self.name_in_target)

    def with_types(self, arg_id_to_dtype, callables_table):
        raise LoopyError("No type inference information present for "
                "the function %s." % (self.name))

    def with_descrs(self, arg_id_to_descr, callables_table):

        arg_id_to_descr[-1] = ValueArgDescriptor()
        return (
                self.copy(arg_id_to_descr=arg_id_to_descr),
                callables_table)

    def with_hw_axes_sizes(self, global_size, local_size):
        return self.copy()

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

        # Currently this is formulated such that the first argument is returned
        # and rest all are passed by reference as arguments to the function.
        assert self.is_ready_for_codegen()

        from loopy.kernel.instruction import CallInstruction

        assert isinstance(insn, CallInstruction)

        parameters = insn.expression.parameters
        assignees = insn.assignees[1:]

        par_dtypes = tuple(expression_to_code_mapper.infer_type(par) for par in
                parameters)
        arg_dtypes = tuple(self.arg_id_to_dtype[i] for i, _ in
                enumerate(parameters))

        assignee_dtypes = tuple(self.arg_id_to_dtype[-i-2] for i, _ in
                enumerate(assignees))

        from loopy.expression import dtype_to_type_context
        from pymbolic.mapper.stringifier import PREC_NONE
        from pymbolic import var

        c_parameters = [
                expression_to_code_mapper(par, PREC_NONE,
                    dtype_to_type_context(target, tgt_dtype),
                    tgt_dtype).expr
                for par, par_dtype, tgt_dtype in zip(
                    parameters, par_dtypes, arg_dtypes)]

        for i, (a, tgt_dtype) in enumerate(zip(assignees, assignee_dtypes)):
            if tgt_dtype != expression_to_code_mapper.infer_type(a):
                raise LoopyError("Type Mismatch in function %s. Expected: %s"
                        "Got: %s" % (self.name, tgt_dtype,
                            expression_to_code_mapper.infer_type(a)))
            c_parameters.append(
                        var("&")(
                            expression_to_code_mapper(a, PREC_NONE,
                                dtype_to_type_context(target, tgt_dtype),
                                tgt_dtype).expr))

        # assignee is returned whenever the size of assignees is non zero.
        first_assignee_is_returned = len(insn.assignees) > 0

        return var(self.name_in_target)(*c_parameters), first_assignee_is_returned

    def generate_preambles(self, target):
        return
        yield

    # }}}

    def with_added_arg(self, arg_dtype, arg_descr):
        raise LoopyError("Cannot add args to scalar callables.")

# }}}


# {{{ callable kernel

class CallableKernel(InKernelCallable):
    """
    Records informations about a callee kernel. Also provides interface through
    member methods to make the callee kernel compatible to be called from a
    caller kernel. The :meth:`loopy.register_callable_kernel` should be called
    in order to initiate association between a function in caller kernel and
    the callee kernel.

    :meth:`CallableKernel.with_types` should be called in order to match
    the ``dtypes`` of the arguments that are shared between the caller and the
    callee kernel.

    :meth:`CallableKernel.with_descrs` should be called in order to match
    :attr:`ArrayArgDescriptor.dim_tags`, :attr:`ArrayArgDescriptor.shape`,
    :attr:`ArrayArgDescriptor.address_space`` of the arguments shared between the
    caller and the callee kernel.

    :meth:`CallableKernel.with_hw_axes` should be called to set the grid
    sizes for the :attr:`subkernel` of the callable.
    """

    fields = {"subkernel", "arg_id_to_dtype", "arg_id_to_descr"}
    init_arg_names = ("subkernel", "arg_id_to_dtype", "arg_id_to_descr")
    hash_fields = ("subkernel", "arg_id_to_dtype", "arg_id_to_descr")

    def __init__(self, subkernel, arg_id_to_dtype=None,
            arg_id_to_descr=None):
        assert isinstance(subkernel, LoopKernel)

        super().__init__(
                arg_id_to_dtype=arg_id_to_dtype,
                arg_id_to_descr=arg_id_to_descr)

        self.subkernel = subkernel

    def __getinitargs__(self):
        return (self.subkernel, self.arg_id_to_dtype,
                self.arg_id_to_descr)

    @property
    def name(self):
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

        from loopy.type_inference import (
                infer_unknown_types_for_a_single_kernel)
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
                arg_id_to_dtype=new_arg_id_to_dtype), callables_table

    def with_descrs(self, arg_id_to_descr, callables_table):

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
        subkernel, callables_table = (
                traverse_to_infer_arg_descr(subkernel,
                    callables_table))

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
                          arg_id_to_descr=arg_id_to_descr),
                callables_table)

    def with_added_arg(self, arg_dtype, arg_descr):
        var_name = self.subkernel.get_var_name_generator()(based_on="_lpy_arg")

        if isinstance(arg_descr, ValueArgDescriptor):
            subknl = self.subkernel.copy(
                    args=self.subkernel.args+[
                        ValueArg(var_name, arg_dtype, self.subkernel.target)])

            kw_to_pos, pos_to_kw = get_kw_pos_association(subknl)

            if self.arg_id_to_dtype is None:
                arg_id_to_dtype = {}
            else:
                arg_id_to_dtype = self.arg_id_to_dtype.copy()
            if self.arg_id_to_descr is None:
                arg_id_to_descr = {}
            else:
                arg_id_to_descr = self.arg_id_to_descr.copy()

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
        kw_to_pos, pos_to_kw = get_kw_pos_association(self.subkernel)

        arg_id_to_descr = {}

        for pos, kw in pos_to_kw.items():
            arg = self.subkernel.arg_dict[kw]
            arg_id_to_descr[pos] = ArrayArgDescriptor(
                    shape=arg.shape,
                    dim_tags=arg.dim_tags,
                    address_space=AddressSpace.GLOBAL)

        return self.copy(subkernel=self.subkernel,
                arg_id_to_descr=arg_id_to_descr)

    def with_hw_axes_sizes(self, gsize, lsize):
        return self.copy(
                subkernel=self.subkernel.copy(
                    overridden_get_grid_sizes_for_insn_ids=(
                        GridOverrideForCalleeKernel(gsize, lsize))))

    def is_ready_for_codegen(self):
        return (self.arg_id_to_dtype is not None and
                self.arg_id_to_descr is not None)

    def generate_preambles(self, target):
        """ Yields the *target* specific preambles.
        """
        # FIXME Check that this is correct.

        return
        yield

    def emit_call_insn(self, insn, target, expression_to_code_mapper):

        assert self.is_ready_for_codegen()

        from loopy.kernel.instruction import CallInstruction
        from pymbolic.primitives import CallWithKwargs

        assert isinstance(insn, CallInstruction)

        parameters = insn.expression.parameters
        kw_parameters = {}
        if isinstance(insn.expression, CallWithKwargs):
            kw_parameters = insn.expression.kw_parameters

        assignees = insn.assignees

        parameters = list(parameters)
        par_dtypes = [self.arg_id_to_dtype[i] for i, _ in enumerate(parameters)]
        kw_to_pos, pos_to_kw = get_kw_pos_association(self.subkernel)
        for i in range(len(parameters), len(parameters)+len(kw_parameters)):
            parameters.append(kw_parameters[pos_to_kw[i]])
            par_dtypes.append(self.arg_id_to_dtype[pos_to_kw[i]])

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
        from loopy.expression import dtype_to_type_context
        from pymbolic.mapper.stringifier import PREC_NONE
        from loopy.symbolic import SubArrayRef
        from pymbolic import var

        c_parameters = [
                expression_to_code_mapper(par, PREC_NONE,
                    dtype_to_type_context(target, par_dtype),
                    par_dtype).expr if isinstance(par, SubArrayRef) else
                expression_to_code_mapper(par, PREC_NONE,
                    dtype_to_type_context(target, par_dtype),
                    par_dtype).expr
                for par, par_dtype in zip(
                    parameters, par_dtypes)]

        return var(self.subkernel.name)(*c_parameters), False

# }}}


# vim: foldmethod=marker

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

import islpy as isl
from pytools import ImmutableRecord
from loopy.diagnostic import LoopyError

from loopy.tools import update_persistent_hash
from loopy.kernel import LoopKernel
from loopy.kernel.data import ValueArg, ArrayArg, ConstantArg
from loopy.symbolic import (SubstitutionMapper, DependencyMapper)
from pymbolic.primitives import Variable

__doc__ = """

.. currentmodule:: loopy

.. autoclass:: ValueArgDescriptor
.. autoclass:: ArrayArgDescriptor
.. autoclass:: InKernelCallable
.. autoclass:: CallableKernel
.. autoclass:: ScalarCallable
.. autoclass:: ManglerCallable

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
    Records information about an array argument to an in-kernel callable, to be
    passed to and returned from
    :meth:`loopy.kernel.function_interface.InKernelCallable.with_descrs`, used
    for matching shape and scope of caller and callee kernels.

    ..attribute:: shape

        Shape of the array.

    .. attribute:: address_space

        An attribute of :class:`loopy.kernel.data.AddressSpace`.

    .. attribute:: dim_tags

        A tuple of instances of
        :class:`loopy.kernel.array.ArrayDimImplementationTag`
    """

    fields = {"shape", "address_space", "dim_tags"}

    def __init__(self, shape, address_space, dim_tags):

        # {{{ sanity checks

        from loopy.kernel.array import ArrayDimImplementationTag

        assert isinstance(shape, tuple)
        assert isinstance(dim_tags, tuple)

        # FIXME at least vector dim tags should be supported
        assert all(isinstance(dim_tag, ArrayDimImplementationTag) for dim_tag in
                dim_tags)

        # }}}

        super().__init__(
                shape=shape,
                address_space=address_space,
                dim_tags=dim_tags)

    hash_fields = (
            "shape",
            "address_space",
            "dim_tags")

    def map_expr(self, subst_mapper):
        new_shape = tuple(subst_mapper(axis_len) for axis_len in self.shape)
        new_dim_tags = tuple(dim_tag.map_expr(subst_mapper) for dim_tag in
                self.dim_tags)
        return self.copy(shape=new_shape, dim_tags=new_dim_tags)

    def depends_on(self):
        from loopy.kernel.data import auto
        result = DependencyMapper(composite_leaves=False)([lngth for lngth in
            self.shape if lngth not in [None, auto]]) | (
                frozenset().union(*(dim_tag.depends_on() for dim_tag in
                    self.dim_tags)))
        return frozenset(var.name for var in result)

    # FIXME ArrayArgDescriptor should never need to be persisted, remove
    # this method when that is so.
    def update_persistent_hash(self, key_hash, key_builder):
        for shape_i in self.shape:
            if shape_i is None:
                key_builder.rec(key_hash, shape_i)
            else:
                key_builder.update_for_pymbolic_expression(key_hash, shape_i)
        key_builder.rec(key_hash, self.address_space)
        key_builder.rec(key_hash, self.dim_tags)


def get_arg_descriptor_for_expression(kernel, expr):
    """
    :returns: a :class:`ArrayArgDescriptor` or a :class:`ValueArgDescriptor`
        describing the argument expression *expr* which occurs
        in a call in the code of *kernel*.
    """
    from pymbolic.primitives import Variable
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
        # actually help? Maybe the
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
                # Also, some may have been simplified away by simplify_using_aff.
                DimTag(strides_as_dict.get(iname, 0))

                for iname in expr.swept_inames)
        sub_shape = tuple(
                pw_aff_to_expr(
                    kernel.get_iname_bounds(iname.name).upper_bound_pw_aff
                    - kernel.get_iname_bounds(iname.name).lower_bound_pw_aff)+1
                for iname in expr.swept_inames)
        if expr.swept_inames == ():
            sub_shape = (1, )
            sub_dim_tags = (DimTag(1),)

        return ArrayArgDescriptor(
                address_space=aspace,
                dim_tags=sub_dim_tags,
                shape=sub_shape)

    elif isinstance(expr, Variable):
        arg = kernel.get_var_descriptor(expr.name)
        from loopy.kernel.array import ArrayBase

        if isinstance(arg, ValueArg) or (isinstance(arg, ArrayBase)
                and arg.shape == ()):
            return ValueArgDescriptor()
        elif isinstance(arg, (ArrayArg, TemporaryVariable)):
            raise LoopyError("may not pass entire array "
                    "'%s' in call statement in kernel '%s'"
                    % (expr.name, kernel.name))
        else:
            raise LoopyError("unsupported argument type "
                    "'%s' of '%s' in call statement"
                    % (type(arg).__name__, expr.name))

    else:
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
    :func:`loopy.kernel.function_interface.GridOverrideForCalleeKernel.__call__`,
    :func:`loopy.kernel.function_interface.CallbleKernel.with_hw_axes_sizes`.

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

    def with_types(self, arg_id_to_dtype, caller_kernel, callables_table):
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

    def with_descrs(self, arg_id_to_descr, caller_kernel, callables_table, expr):
        """
        :arg arg_id_to_descr: a mapping from argument identifiers
            (integers for positional arguments, names for keyword
            arguments) to :class:`loopy.ArrayArgDescriptor` instances.
            Unspecified/unknown types are not represented in *arg_id_to_descr*.

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

# }}}


# {{{ scalar callable

class ScalarCallable(InKernelCallable):
    """
    An abstract interface the to a scalar callable encountered in a kernel.

    .. note::

        The :meth:`ScalarCallable.with_types` is intended to assist with type
        specialization of the function and is expected to be supplemented in the
        derived subclasses.
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

    def with_types(self, arg_id_to_dtype, caller_kernel, callables_table):
        raise LoopyError("No type inference information present for "
                "the function %s." % (self.name))

    def with_descrs(self, arg_id_to_descr, caller_kernel, callables_table, expr):

        arg_id_to_descr[-1] = ValueArgDescriptor()
        return (
                self.copy(arg_id_to_descr=arg_id_to_descr),
                callables_table, ())

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

        # TODO: Maybe this interface a bit confusing. Should we allow this
        # method to directly return a cgen.Assign or cgen.ExpressionStatement?

        return var(self.name_in_target)(*c_parameters), first_assignee_is_returned

    def generate_preambles(self, target):
        return
        yield

    # }}}

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

        self.subkernel = subkernel.copy(
                args=[arg.copy(dtype=arg.dtype.with_target(subkernel.target))
                    if arg.dtype is not None else arg for arg in subkernel.args])

    def __getinitargs__(self):
        return (self.subkernel, self.arg_id_to_dtype,
                self.arg_id_to_descr)

    @property
    def name(self):
        return self.subkernel.name

    def with_types(self, arg_id_to_dtype, caller_kernel,
            callables_table):
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
            new_arg_id_to_dtype[kw] = specialized_kernel.arg_dict[kw].dtype
            new_arg_id_to_dtype[pos] = specialized_kernel.arg_dict[kw].dtype

        # Return the kernel call with specialized subkernel and the corresponding
        # new arg_id_to_dtype
        return self.copy(subkernel=specialized_kernel,
                arg_id_to_dtype=new_arg_id_to_dtype), callables_table

    def with_descrs(self, arg_id_to_descr, caller_kernel, callables_table,
            expr=None):
        # tune the subkernel so that we have the matching shapes and
        # dim_tags

        # {{{ map the arg_descrs so that all the variables are from the callees
        # perspective

        domain_dependent_vars = frozenset().union(
                *(frozenset(dom.get_var_names(isl.dim_type.param)) for dom in
                    self.subkernel.domains))

        # FIXME: This is ill-formed, because par can be an expression, e.g.
        # 2*i+2 or 2*(i+1). A key feature of expression is that structural
        # equality and semantic equality are not the same, so even if the
        # SubstitutionMapper allowed non-variables, it would have to solve the
        # (considerable) problem of expression equivalence.

        import numbers
        substs = {}
        assumptions = {}

        if expr:
            for arg, par in zip(self.subkernel.args, expr.parameters):
                if isinstance(arg, ValueArg) and arg.name in domain_dependent_vars:
                    if isinstance(par, Variable):
                        if par in substs:
                            assumptions[arg.name] = substs[par].name
                        else:
                            substs[par] = Variable(arg.name)
                    elif isinstance(par, numbers.Number):
                        assumptions[arg.name] = par

            def subst_func(expr):
                if expr in substs:
                    return substs[expr]
                else:
                    return expr

            subst_mapper = SubstitutionMapper(subst_func)

            arg_id_to_descr = {arg_id: descr.map_expr(subst_mapper)
                               for arg_id, descr in arg_id_to_descr.items()}

        # }}}

        dependents = frozenset().union(*(descr.depends_on() for descr in
            arg_id_to_descr.values()))
        unknown_deps = dependents - self.subkernel.all_variable_names()

        if expr is None:
            assert unknown_deps == frozenset()
        # FIXME: Need to make sure that we make the name of the variables
        # unique, and then run a subst_mapper

        new_args = self.subkernel.args[:]
        kw_to_pos, pos_to_kw = get_kw_pos_association(self.subkernel)

        for arg_id, descr in arg_id_to_descr.items():
            if isinstance(arg_id, int):
                arg_id = pos_to_kw[arg_id]
            assert isinstance(arg_id, str)

            if isinstance(descr, ArrayArgDescriptor):
                if not isinstance(self.subkernel.arg_dict[arg_id], (ArrayArg,
                        ConstantArg)):
                    raise LoopyError("Array passed to scalar argument "
                            "'%s' of the function '%s' (in '%s')." % (
                                arg_id, self.subkernel.name,
                                caller_kernel.name))
                if self.subkernel.arg_dict[arg_id].shape and (
                        len(self.subkernel.arg_dict[arg_id].shape) !=
                        len(descr.shape)):
                    raise LoopyError("Dimension mismatch for argument "
                            " '%s' of the function '%s' (in '%s')." % (
                                arg_id, self.subkernel.name,
                                caller_kernel.name))

                new_arg = self.subkernel.arg_dict[arg_id].copy(
                        shape=descr.shape,
                        dim_tags=descr.dim_tags,
                        address_space=descr.address_space)
                # replacing the new arg with the arg of the same name
                new_args = [new_arg if arg.name == arg_id else arg for arg in
                        new_args]
            elif isinstance(descr, ValueArgDescriptor):
                if not isinstance(self.subkernel.arg_dict[arg_id], ValueArg):
                    raise LoopyError("Scalar passed to array argument "
                            "'%s' of the callable '%s' (in '%s')" % (
                                arg_id, self.subkernel.name,
                                caller_kernel.name))
            else:
                raise LoopyError("Descriptor must be either an instance of "
                        "ArrayArgDescriptor or ValueArgDescriptor -- got %s" %
                        type(descr))

        descriptor_specialized_knl = self.subkernel.copy(args=new_args)
        # add the variables on which the strides/shapes depend but not provided
        # as arguments
        args_added_knl = descriptor_specialized_knl.copy(
                args=descriptor_specialized_knl.args
                + [ValueArg(dep) for dep in unknown_deps])
        from loopy.preprocess import traverse_to_infer_arg_descr
        from loopy.transform.parameter import assume
        args_added_knl, callables_table = (
                traverse_to_infer_arg_descr(args_added_knl,
                    callables_table))

        if assumptions:
            assumption_str = " and ".join([f"{key}={val}"
                                           for key, val in assumptions.items()])
            args_added_knl = assume(args_added_knl, assumption_str)

        return (
                self.copy(
                    subkernel=args_added_knl,
                    arg_id_to_descr=arg_id_to_descr),
                callables_table, tuple(Variable(dep) for dep in unknown_deps))

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


# {{{ mangler callable

class ManglerCallable(ScalarCallable):
    """
    A callable whose characteristic is defined by a function mangler.

    .. attribute:: function_mangler

        A function of signature ``(kernel, name , arg_dtypes)`` and returns an
        instance of ``loopy.CallMangleInfo``.
    """
    fields = {"name", "function_mangler", "arg_id_to_dtype", "arg_id_to_descr",
        "name_in_target"}
    init_arg_names = ("name", "function_mangler", "arg_id_to_dtype",
            "arg_id_to_descr", "name_in_target")
    hash_fields = ("name", "arg_id_to_dtype", "arg_id_to_descr",
        "name_in_target")

    def __init__(self, name, function_mangler, arg_id_to_dtype=None,
            arg_id_to_descr=None, name_in_target=None):

        self.function_mangler = function_mangler

        super().__init__(
                name=name,
                arg_id_to_dtype=arg_id_to_dtype,
                arg_id_to_descr=arg_id_to_descr,
                name_in_target=name_in_target)

    def __getinitargs__(self):
        return (self.name, self.function_mangler, self.arg_id_to_dtype,
                self.arg_id_to_descr, self.name_in_target)

    def with_types(self, arg_id_to_dtype, kernel, callables_table):
        if self.arg_id_to_dtype is not None:
            # specializing an already specialized function.
            for arg_id, dtype in arg_id_to_dtype.items():
                # only checking for the ones which have been provided
                # if does not match, returns an error.
                if self.arg_id_to_dtype[arg_id] != arg_id_to_dtype[arg_id]:
                    raise LoopyError("Overwriting a specialized"
                            " function is illegal--maybe start with new instance of"
                            " ManglerCallable?")

        sorted_keys = sorted(arg_id_to_dtype.keys())
        arg_dtypes = tuple(arg_id_to_dtype[key] for key in sorted_keys if
                key >= 0)

        mangle_result = self.function_mangler(kernel, self.name,
                arg_dtypes)
        if mangle_result:
            new_arg_id_to_dtype = dict(enumerate(mangle_result.arg_dtypes))
            new_arg_id_to_dtype.update({-i-1: dtype for i, dtype in
                enumerate(mangle_result.result_dtypes)})
            return (
                    self.copy(name_in_target=mangle_result.target_name,
                        arg_id_to_dtype=new_arg_id_to_dtype),
                    callables_table)
        else:
            # The function mangler does not agree with the arg id to dtypes
            # provided. Indicating that is illegal.
            raise LoopyError("Function %s not coherent with the provided types." % (
                self.name))

    def mangle_result(self, kernel):
        """
        Returns an instance of :class:`loopy.kernel.data.CallMangleInfo` for
        the given pair :attr:`function_mangler` and :attr:`arg_id_to_dtype`.
        """
        sorted_keys = sorted(self.arg_id_to_dtype.keys())
        arg_dtypes = tuple(self.arg_id_to_dtype[key] for key in sorted_keys if
                key >= 0)

        return self.function_mangler(kernel, self.name, arg_dtypes)

# }}}

# vim: foldmethod=marker

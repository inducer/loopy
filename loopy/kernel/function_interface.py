from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2018 Andreas Klöckner, Kaushik Kulkarni"

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


import re
import six

from six.moves import zip

from pytools import ImmutableRecord
from loopy.diagnostic import LoopyError

from pymbolic.primitives import Variable
from loopy.symbolic import parse_tagged_name

from loopy.symbolic import (ScopedFunction, SubstitutionRuleMappingContext,
        RuleAwareIdentityMapper, SubstitutionRuleExpander)


# {{{ argument descriptors

class ValueArgDescriptor(ImmutableRecord):
    pass


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

        A tuple of instances of :class:`loopy.kernel.array._StrideArrayDimTagBase`
    """

    fields = set(['shape', 'address_space', 'dim_tags'])

    def __init__(self, shape, address_space, dim_tags):

        # {{{ sanity checks

        from loopy.kernel.array import FixedStrideArrayDimTag

        assert isinstance(shape, tuple)
        assert isinstance(dim_tags, tuple)

        # FIXME at least vector dim tags should be supported
        assert all(isinstance(dim_tag, FixedStrideArrayDimTag) for dim_tag in
                dim_tags)

        # }}}

        super(ArrayArgDescriptor, self).__init__(
                shape=shape,
                address_space=address_space,
                dim_tags=dim_tags)

# }}}


# {{{ helper function for in-kernel callables

def get_kw_pos_association(kernel):
    """
    Returns a tuple of ``(kw_to_pos, pos_to_kw)`` for the arguments in
    *kernel*.
    """
    from loopy.kernel.tools import infer_arg_is_output_only
    kernel = infer_arg_is_output_only(kernel)
    kw_to_pos = {}
    pos_to_kw = {}

    read_count = 0
    write_count = -1

    for arg in kernel.args:
        if not arg.is_output_only:
            kw_to_pos[arg.name] = read_count
            pos_to_kw[read_count] = arg.name
            read_count += 1
        else:
            kw_to_pos[arg.name] = write_count
            pos_to_kw[write_count] = arg.name
            write_count -= 1

    return kw_to_pos, pos_to_kw


class GridOverrideForCalleeKernel(ImmutableRecord):
    """
    Helper class to set the
    :attr:`loopy.kernel.LoopKernel.override_get_grid_size_for_insn_ids` of the
    callee kernels. Refer
    :func:`loopy.kernel.function_interface.GridOverrideForCalleeKernel.__call__`,
    :func:`loopy.kernel.function_interface.CallbleKernel.with_hw_axes_sizes`.

    .. attribute:: local_size

        The local work group size that has to be set in the callee kernel.

    .. attribute:: global_size

        The global work group size that to be set in the callee kernel.

    .. note::

        This class acts as a pseduo-callable and its significance lies in
        solving picklability issues.
    """
    fields = set(["local_size", "global_size"])

    def __init__(self, local_size, global_size):
        self.local_size = local_size
        self.global_size = global_size

    def __call__(self, insn_ids, ignore_auto=True):
        return self.local_size, self.global_size

# }}}


# {{{ template class

class InKernelCallable(ImmutableRecord):
    """
    An abstract interface to define a callable encountered in a kernel.

    .. attribute:: name

        The name of the callable which can be encountered within a kernel.

    .. attribute:: arg_id_to_dtype

        A mapping which indicates the arguments types and result types it would
        be handling. This would be set once the callable is type specialized.

    .. attribute:: arg_id_to_descr

        A mapping which gives indicates the argument shape and ``dim_tags`` it
        would be responsible for generating code. These parameters would be set,
        once it is shape and stride(``dim_tags``) specialized.

    .. note::

        Negative "id" values ``-i`` in the mapping attributes indicate
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

    fields = set(["arg_id_to_dtype", "arg_id_to_descr"])
    init_arg_names = ("arg_id_to_dtype", "arg_id_to_descr")

    def __init__(self, arg_id_to_dtype=None, arg_id_to_descr=None):

        super(InKernelCallable, self).__init__(
                arg_id_to_dtype=arg_id_to_dtype,
                arg_id_to_descr=arg_id_to_descr)

    def __getinitargs__(self):
        return (self.arg_id_to_dtype, self.arg_id_to_descr)

    def with_types(self, arg_id_to_dtype, kernel):
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

    def with_descrs(self, arg_id_to_descr):
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
        if self.arg_id_to_dtype:
            new_arg_id_to_dtype = dict((id, with_target_if_not_None(dtype)) for id,
                    dtype in self.arg_id_to_dtype.items())

        return self.copy(arg_id_to_dtype=new_arg_id_to_dtype)

    def with_hw_axes_sizes(self, local_size, global_size):
        """
        Returns a copy of *self* with modifications to comply with the grid
        sizes ``(local_size, global_size)`` of the kernel in which it is
        supposed to be called.

        :arg local_size: An instance of :class:`islpy.PwAff`.
        :arg global_size: An instance of :class:`islpy.PwAff`.
        """
        raise NotImplementedError()

    def is_ready_for_codegen(self):

        return (self.arg_id_to_dtype is not None and
                self.arg_id_to_descr is not None)

    def generate_preambles(self, target):
        """ Yields the target specific preamble.
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

        :Example: If ``assignee_is_returned=True``, then ``a, b = f(c, d)`` is
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
    An abstranct interface the to a scalar callable encountered in a kernel.

    .. note::

        The :meth:`ScalarCallable.with_types` is intended to assist with type
        specialization of the funciton and is expected to be supplemented in the
        derived subclasses.
    """

    fields = set(["name", "arg_id_to_dtype", "arg_id_to_descr", "name_in_target"])
    init_arg_names = ("name", "arg_id_to_dtype", "arg_id_to_descr",
            "name_in_target")

    def __init__(self, name, arg_id_to_dtype=None,
            arg_id_to_descr=None, name_in_target=None):

        super(ScalarCallable, self).__init__(
                arg_id_to_dtype=arg_id_to_dtype,
                arg_id_to_descr=arg_id_to_descr)

        self.name = name
        self.name_in_target = name_in_target

    def __getinitargs__(self):
        return (self.arg_id_to_dtype, self.arg_id_to_descr,
                self.name_in_target)

    def with_types(self, arg_id_to_dtype, kernel):
        raise LoopyError("No type inference information present for "
                "the function %s." % (self.name))

    def with_descrs(self, arg_id_to_descr):

        arg_id_to_descr[-1] = ValueArgDescriptor()
        return self.copy(arg_id_to_descr=arg_id_to_descr)

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
        Returns a pymbolic call for C-based targets, when the instructions
        involve multiple return values along with the required type casting.
        The first assignee is returned, but the rest of them are appended to
        the parameters and passed by reference.

        :Example: ``c, d = f(a, b)`` is returned as ``c = f(a, b, &d)``

        :arg insn: An instance of :class:`loopy.kernel.instructions.CallInstruction`.
        :arg target: An instance of :class:`loopy.target.TargetBase`.
        :arg expression_to_code_mapper: An instance of :class:`IdentityMapper`
            responsible for code mapping from :mod:`loopy` syntax to the
            **target syntax**.
        """

        # FIXME: needs to get information about whether the callable has should
        # do pass by reference by all values or should return one value for
        # pass by value return.

        # For example: The code generation of `sincos` would be different for
        # C-Target and OpenCL-target.

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
        assignee_is_returned = len(assignees) > 0

        return var(self.name_in_target)(*c_parameters), assignee_is_returned

    def generate_preambles(self, target):
        return
        yield

    # }}}

# }}}


# {{{ mangler callable

class ManglerCallable(ScalarCallable):
    """
    A callable whose characateristic is defined by a function mangler.

    .. attribute:: function_mangler

        A function of signature ``(kernel, name , arg_dtypes)`` and returns an
        instance of ``loopy.CallMangleInfo``.
    """
    fields = set(["name", "function_mangler", "arg_id_to_dtype", "arg_id_to_descr",
        "name_in_target"])
    init_arg_names = ("name", "function_mangler", "arg_id_to_dtype",
            "arg_id_to_descr", "name_in_target")

    def __init__(self, name, function_mangler, arg_id_to_dtype=None,
            arg_id_to_descr=None, name_in_target=None):

        self.function_mangler = function_mangler

        super(ManglerCallable, self).__init__(
                name=name,
                arg_id_to_dtype=arg_id_to_dtype,
                arg_id_to_descr=arg_id_to_descr,
                name_in_target=name_in_target)

    def __getinitargs__(self):
        return (self.name, self.function_mangler, self.arg_id_to_dtype,
                self.arg_id_to_descr, self.name_in_target)

    def with_types(self, arg_id_to_dtype, kernel):
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
            new_arg_id_to_dtype.update(dict((-i-1, dtype) for i, dtype in
                enumerate(mangle_result.result_dtypes)))
            return self.copy(name_in_target=mangle_result.target_name,
                    arg_id_to_dtype=new_arg_id_to_dtype)
        else:
            # The function mangler does not agree with the arg id to dtypes
            # provided. Indicating that is illegal.
            raise LoopyError("Function %s not coherent with the provided types." % (
                self.name, kernel.target))

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


# {{{ new pymbolic calls to scoped functions

# FIXME Are these identifiers guaranteed to be available? Is there a var name
# generator somewhere ensuring that that's the case?
def next_indexed_variable(function):
    """
    Returns an instance of :class:`str` with the next indexed-name in the
    sequence for the name of *function*.

    **Example:** ``Variable('sin_0')`` will return ``'sin_1'``.

    :arg function: Either an instance of :class:`pymbolic.primitives.Variable`
        or :class:`loopy.reduction.ArgExtOp` or
        :class:`loopy.reduction.SegmentedOp`.
    """
    from loopy.library.reduction import ArgExtOp, SegmentedOp
    if isinstance(function, (ArgExtOp, SegmentedOp)):
        return function.copy()
    func_name = re.compile(r"^(?P<alpha>\S+?)_(?P<num>\d+?)$")

    match = func_name.match(function.name)

    if match is None:
        if function.name[-1] == '_':
            return "{old_name}0".format(old_name=function.name)
        else:
            return "{old_name}_0".format(old_name=function.name)

    return "{alpha}_{num}".format(alpha=match.group('alpha'),
            num=int(match.group('num'))+1)


class ScopedFunctionNameChanger(RuleAwareIdentityMapper):
    """
    Changes the names of scoped functions in calls of expressions according to
    the mapping ``expr_to_new_names``
    """

    def __init__(self, rule_mapping_context, expr_to_new_names, subst_expander):
        super(ScopedFunctionNameChanger, self).__init__(rule_mapping_context)
        self.expr_to_new_names = expr_to_new_names
        self.subst_expander = subst_expander

    def map_call(self, expr, expn_state):
        name, tag = parse_tagged_name(expr.function)

        if name not in self.rule_mapping_context.old_subst_rules:
            expanded_expr = self.subst_expander(expr)
            if expr in self.expr_to_new_names:
                return type(expr)(
                        ScopedFunction(self.expr_to_new_names[expr]),
                        tuple(self.rec(child, expn_state)
                            for child in expr.parameters))
            elif expanded_expr in self.expr_to_new_names:
                return type(expr)(
                        ScopedFunction(self.expr_to_new_names[expanded_expr]),
                        tuple(self.rec(child, expn_state)
                            for child in expr.parameters))
            else:
                return super(ScopedFunctionNameChanger, self).map_call(
                        expr, expn_state)
        else:
            return self.map_substitution(name, tag, expr.parameters, expn_state)

    def map_call_with_kwargs(self, expr, expn_state):
        name, tag = parse_tagged_name(expr.function)

        if name not in self.rule_mapping_context.old_subst_rules:
            expanded_expr = self.subst_expander(expr)
            if expr in self.expr_to_new_names:
                return type(expr)(
                    ScopedFunction(self.expr_to_new_names[expr]),
                    tuple(self.rec(child, expn_state)
                        for child in expr.parameters),
                    dict(
                        (key, self.rec(val, expn_state))
                        for key, val in six.iteritems(expr.kw_parameters))
                        )
            elif expanded_expr in self.expr_to_new_names:
                return type(expr)(
                    ScopedFunction(self.expr_to_new_names[expanded_expr]),
                    tuple(self.rec(child, expn_state)
                        for child in expr.parameters),
                    dict(
                        (key, self.rec(val, expn_state))
                        for key, val in six.iteritems(expr.kw_parameters))
                        )
            else:
                return super(ScopedFunctionNameChanger, self).map_call_with_kwargs(
                        expr, expn_state)
        else:
            return self.map_substitution(name, tag, expr.parameters, expn_state)


def register_pymbolic_calls_to_knl_callables(kernel,
        pymbolic_exprs_to_knl_callables):
    # FIXME This could use an example. I have no idea what this does.
    # Surely I can't associate arbitrary pymbolic expresions (3+a?)
    # with callables?
    """
    Returns a copy of :arg:`kernel` which includes an association with the given
    pymbolic expressions to  the instances of :class:`InKernelCallable` for the
    mapping given by :arg:`pymbolic_exprs_to_knl_calllables`.

    :arg kernel: An instance of :class:`loopy.kernel.LoopKernel`.

    :arg pymbolic_exprs_to_knl_callables: A mapping from :mod:`pymbolic` expressions
        to the instances of
        :class:`loopy.kernel.function_interface.InKernelCallable`.
    """

    scoped_names_to_functions = kernel.scoped_functions.copy()

    # A dict containing the new scoped functions to the names which have been
    # assigned to them
    scoped_functions_to_names = {}

    # A dict containing the new name that need to be assigned to the
    # corresponding pymbolic call
    pymbolic_calls_to_new_names = {}

    for pymbolic_call, in_knl_callable in pymbolic_exprs_to_knl_callables.items():
        # checking if such a in-kernel callable already exists.
        if in_knl_callable not in scoped_functions_to_names:
            # No matching in_knl_callable found => make a new one with a new
            # name.
            if isinstance(pymbolic_call.function, Variable):
                pymbolic_call_function = pymbolic_call.function
            elif isinstance(pymbolic_call.function, ScopedFunction):
                pymbolic_call_function = pymbolic_call.function.function
            else:
                raise NotImplementedError("Unknown type %s for pymbolic call "
                        "function" % type(pymbolic_call).__name__)

            unique_var = next_indexed_variable(pymbolic_call_function)
            from loopy.library.reduction import ArgExtOp, SegmentedOp
            while unique_var in scoped_names_to_functions and not isinstance(
                    unique_var, (ArgExtOp, SegmentedOp)):
                # keep on finding new names till one a unique one is found.
                unique_var = next_indexed_variable(Variable(unique_var))

            scoped_names_to_functions[unique_var] = in_knl_callable
            scoped_functions_to_names[in_knl_callable] = unique_var

        pymbolic_calls_to_new_names[pymbolic_call] = (
                scoped_functions_to_names[in_knl_callable])

    # Use the data populated in pymbolic_calls_to_new_names to change the
    # names of the scoped functions of all the calls in the kernel.
    rule_mapping_context = SubstitutionRuleMappingContext(
                kernel.substitutions, kernel.get_var_name_generator())
    subst_expander = SubstitutionRuleExpander(kernel.substitutions)
    scope_changer = ScopedFunctionNameChanger(rule_mapping_context,
            pymbolic_calls_to_new_names, subst_expander)
    scoped_kernel = scope_changer.map_kernel(kernel)

    return scoped_kernel.copy(scoped_functions=scoped_names_to_functions)

# }}}


# vim: foldmethod=marker

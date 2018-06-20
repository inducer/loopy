from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2018 Kaushik Kulkarni"

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

from loopy.kernel import LoopKernel
from loopy.kernel.function_interface import CallableKernel
from pytools import ImmutableRecord
from loopy.diagnostic import LoopyError
from loopy.kernel.instruction import (CallInstruction, MultiAssignmentBase,
        CInstruction, _DataObliviousInstruction)
from loopy.symbolic import IdentityMapper
from loopy.isl_helpers import simplify_via_aff
from pymbolic.primitives import CallWithKwargs
from loopy.kernel.function_interface import (get_kw_pos_association,
        register_pymbolic_calls_to_knl_callables)


__doc__ = """
.. currentmodule:: loopy

.. autofunction:: register_function_lookup

.. autofunction:: register_callable_kernel
"""


# {{{ register function lookup

def register_function_lookup(kernel, function_lookup):
    """
    Returns a copy of *kernel* with the *function_lookup* registered.

    :arg function_lookup: A function of signature ``(target, identifier)``
        returning a :class:`loopy.kernel.function_interface.InKernelCallable`.
    """

    # adding the function lookup to the set of function lookers in the kernel.
    if function_lookup not in kernel.function_scopers:
        from loopy.tools import unpickles_equally
        if not unpickles_equally(function_lookup):
            raise LoopyError("function '%s' does not "
                    "compare equally after being upickled "
                    "and would disrupt loopy's caches"
                    % function_lookup)
        new_function_scopers = kernel.function_scopers + [function_lookup]
    registered_kernel = kernel.copy(function_scopers=new_function_scopers)
    from loopy.kernel.creation import scope_functions

    # returning the scoped_version of the kernel, as new functions maybe
    # resolved.
    return scope_functions(registered_kernel)

# }}}


# {{{ register_callable_kernel

class _RegisterCalleeKernel(ImmutableRecord):
    """
    Helper class to make the function scoper from
    :func:`loopy.transform.register_callable_kernel` picklable. As python
    cannot pickle lexical closures.
    """
    fields = set(['function_name', 'callable_kernel'])

    def __init__(self, function_name, callable_kernel):
        self.function_name = function_name
        self.callable_kernel = callable_kernel

    def __call__(self, target, identifier):
        if identifier == self.function_name:
            return self.callable_kernel
        return None


def register_callable_kernel(caller_kernel, function_name, callee_kernel):
    """Returns a copy of *caller_kernel*, which would resolve *function_name* in an
    expression as a call to *callee_kernel*.

    :arg caller_kernel: An instance of :class:`loopy.kernel.LoopKernel`.
    :arg function_name: An instance of :class:`str`.
    :arg callee_kernel: An instance of :class:`loopy.kernel.LoopKernel`.
    """

    # {{{ sanity checks

    assert isinstance(caller_kernel, LoopKernel)
    assert isinstance(callee_kernel, LoopKernel)
    assert isinstance(function_name, str)

    # check to make sure that the variables with 'out' direction is equal to
    # the number of assigness in the callee kernel intructions.
    from loopy.kernel.tools import infer_arg_is_output_only
    callee_kernel = infer_arg_is_output_only(callee_kernel)
    expected_num_assignees = len([arg for arg in callee_kernel.args if
        arg.is_output_only])
    expected_num_parameters = len(callee_kernel.args) - expected_num_assignees
    for insn in caller_kernel.instructions:
        if isinstance(insn, CallInstruction) and (
                insn.expression.function.name == 'function_name'):
            if insn.assignees != expected_num_assignees:
                raise LoopyError("The number of arguments with 'out' direction "
                        "in callee kernel %s and the number of assignees in "
                        "instruction %s do not match." % (
                            callee_kernel.name, insn.id))
            if insn.expression.prameters != expected_num_parameters:
                raise LoopyError("The number of expected arguments "
                        "for the callee kernel %s and the number of parameters in "
                        "instruction %s do not match." % (
                            callee_kernel.name, insn.id))

        elif isinstance(insn, (MultiAssignmentBase, CInstruction,
                _DataObliviousInstruction)):
            pass
        else:
            raise NotImplementedError("unknown instruction %s" % type(insn))

    # }}}

    # making the target of the child kernel to be same as the target of parent
    # kernel.
    callable_kernel = CallableKernel(subkernel=callee_kernel.copy(
                        target=caller_kernel.target,
                        name=function_name,
                        is_called_from_host=False))

    # disabling global barriers for callee kernel
    from loopy import set_options
    callee_kernel = set_options(callee_kernel, "disable_global_barriers")

    return register_function_lookup(caller_kernel,
            _RegisterCalleeKernel(function_name, callable_kernel))

# }}}


# {{{ inline callable kernel

def inline_callable_kernel(kernel, function_name):
    """
    Returns a copy of *kernel* with the callable kernel addresed by
    *function_name* inlined.
    """
    from loopy.preprocess import infer_arg_descr
    kernel = infer_arg_descr(kernel)

    old_insns = kernel.instructions
    for insn in old_insns:
        if isinstance(insn, CallInstruction):
            if insn.expression.function.name in kernel.scoped_functions:
                in_knl_callable = kernel.scoped_functions[
                        insn.expression.function.name]
                from loopy.kernel.function_interface import CallableKernel
                if isinstance(in_knl_callable, CallableKernel) and (
                        in_knl_callable.subkernel.name == function_name):
                    kernel = in_knl_callable.inline_within_kernel(kernel, insn)
        elif isinstance(insn, (MultiAssignmentBase, CInstruction,
                _DataObliviousInstruction)):
            pass
        else:
            raise NotImplementedError("Unknown instruction %s." % type(insn))

    return kernel

# }}}


# {{{ matching caller to callee args if dimenstions dont match

class DimChanger(IdentityMapper):
    """
    Mapper to change the dimensions of an argument.

    .. attribute:: callee_arg_dict

        A mapping from the argument name (:class:`str`) to instances of
        :class:`loopy.kernel.array.ArrayBase`.

    .. attribute:: desried_shape

        A mapping from argument name (:class:`str`) to an instance of
        :class:`tuple`.
    """
    def __init__(self, callee_arg_dict, desired_shape):
        self.callee_arg_dict = callee_arg_dict
        self.desired_shape = desired_shape

    def map_subscript(self, expr):
        if expr.aggregate.name not in self.callee_arg_dict:
            return super(DimChanger, self).map_subscript(expr)
        callee_arg_dim_tags = self.callee_arg_dict[expr.aggregate.name].dim_tags
        flattened_index = sum(dim_tag.stride*idx for dim_tag, idx in
                zip(callee_arg_dim_tags, expr.index_tuple))
        new_indices = []

        from operator import mul
        from functools import reduce
        stride = reduce(mul, self.desired_shape[expr.aggregate.name], 1)

        for length in self.desired_shape[expr.aggregate.name]:
            stride /= length
            ind = flattened_index // int(stride)
            flattened_index -= (int(stride) * ind)
            new_indices.append(simplify_via_aff(ind))

        return expr.aggregate.index(tuple(new_indices))


def _match_caller_callee_argument_dimension(caller_knl, callee_function_name):
    """
    Returns a copy of *caller_knl* with the instance of
    :class:`loopy.kernel.function_interface.CallableKernel` addressed by
    *callee_function_name* in the *caller_knl* aligned with the argument
    dimesnsions required by *caller_knl*.
    """
    pymbolic_calls_to_new_callables = {}
    for insn in caller_knl.instructions:
        if not isinstance(insn, CallInstruction) or (
                insn.expression.function.name not in
                caller_knl.scoped_functions):
            # Call to a callable kernel can only occur through a
            # CallInstruction.
            continue

        in_knl_callable = caller_knl.scoped_functions[
                insn.expression.function.name]

        if in_knl_callable.subkernel.name != callee_function_name:
            # Not the callable we're looking for.
            continue

        # getting the caller->callee arg association

        parameters = insn.expression.parameters[:]
        kw_parameters = {}
        if isinstance(insn.expression, CallWithKwargs):
            kw_parameters = insn.expression.kw_parameters

        assignees = insn.assignees

        parameter_shapes = [par.get_array_arg_descriptor(caller_knl).shape
                for par in parameters]
        kw_to_pos, pos_to_kw = get_kw_pos_association(in_knl_callable.subkernel)
        for i in range(len(parameters), len(parameters)+len(kw_parameters)):
            parameter_shapes.append(kw_parameters[pos_to_kw[i]]
                    .get_array_arg_descriptor(caller_knl).shape)

        # inserting the assigness at the required positions.
        assignee_write_count = -1
        for i, arg in enumerate(in_knl_callable.subkernel.args):
            if arg.is_output_only:
                assignee = assignees[-assignee_write_count-1]
                parameter_shapes.insert(i, assignee
                        .get_array_arg_descriptor(caller_knl).shape)
                assignee_write_count -= 1

        callee_arg_to_desired_dim_tag = dict(zip([arg.name for arg in
            in_knl_callable.subkernel.args], parameter_shapes))
        dim_changer = DimChanger(in_knl_callable.subkernel.arg_dict,
                callee_arg_to_desired_dim_tag)
        new_callee_insns = []
        for callee_insn in in_knl_callable.subkernel.instructions:
            if isinstance(callee_insn, MultiAssignmentBase):
                new_callee_insns.append(callee_insn.copy(expression=dim_changer(
                    callee_insn.expression),
                    assignee=dim_changer(callee_insn.assignee)))
            elif isinstance(callee_insn, (CInstruction,
                    _DataObliviousInstruction)):
                pass
            else:
                raise NotImplementedError("Unknwon instruction %s." %
                        type(insn))

        # subkernel with instructions adjusted according to the new dimensions.
        new_subkernel = in_knl_callable.subkernel.copy(instructions=new_callee_insns)

        new_in_knl_callable = in_knl_callable.copy(subkernel=new_subkernel)

        pymbolic_calls_to_new_callables[insn.expression] = new_in_knl_callable

    if not pymbolic_calls_to_new_callables:
        # complain if no matching function found.
        raise LoopyError("No CallableKernel with the name %s found in %s." % (
            callee_function_name, caller_knl.name))

    return register_pymbolic_calls_to_knl_callables(caller_knl,
            pymbolic_calls_to_new_callables)

# }}}


# vim: foldmethod=marker

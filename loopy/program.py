from __future__ import division, absolute_import

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

import six
import re

from pytools import ImmutableRecord
from pymbolic.primitives import Variable

from loopy.symbolic import RuleAwareIdentityMapper
from loopy.kernel.function_interface import (
        CallableKernel, ScalarCallable)


class FunctionResolver(RuleAwareIdentityMapper):
    """
    Mapper to convert the  ``function`` attribute of a
    :class:`pymbolic.primitives.Call` known in the kernel as instances of
    :class:`loopy.symbolic.ResolvedFunction`. A function is known in the
    *kernel*, :func:`loopy.kernel.LoopKernel.find_scoped_function_identifier`
    returns an instance of
    :class:`loopy.kernel.function_interface.InKernelCallable`.

    **Example:** If given an expression of the form ``sin(x) + unknown_function(y) +
    log(z)``, then the mapper would return ``ResolvedFunction('sin')(x) +
    unknown_function(y) + ResolvedFunction('log')(z)``.

    :arg rule_mapping_context: An instance of
        :class:`loopy.symbolic.RuleMappingContext`.
    :arg function_ids: A container with instances of :class:`str` indicating
        the function identifiers to look for while scoping functions.
    """
    def __init__(self, rule_mapping_context, kernel, program_callables_info,
            function_resolvers):
        super(FunctionResolver, self).__init__(rule_mapping_context)
        self.kernel = kernel
        self.program_callables_info = program_callables_info
        # FIXME: function_resolvesrs looks like a very bad name change it
        self.function_resolvers = function_resolvers

    def find_resolved_function_from_identifier(self, identifier):
        """
        Returns an instance of
        :class:`loopy.kernel.function_interface.InKernelCallable` if the
        :arg:`identifier` is known to any kernel function scoper, otherwise returns
        *None*.
        """
        # FIXME change docs
        for scoper in self.function_resolvers:
            # fixme: do we really need to given target for the function
            in_knl_callable = scoper(self.kernel.target, identifier)
            if in_knl_callable is not None:
                return in_knl_callable

        return None

    def map_call(self, expr, expn_state):
        from pymbolic.primitives import Call, CallWithKwargs
        from loopy.symbolic import parse_tagged_name

        name, tag = parse_tagged_name(expr.function)
        if name not in self.rule_mapping_context.old_subst_rules:
            new_call_with_kwargs = self.rec(CallWithKwargs(
                function=expr.function, parameters=expr.parameters,
                kw_parameters={}), expn_state)
            return Call(new_call_with_kwargs.function,
                    new_call_with_kwargs.parameters)
        else:
            return self.map_substitution(name, tag, expr.parameters, expn_state)

    def map_call_with_kwargs(self, expr, expn_state):
        from loopy.symbolic import ResolvedFunction

        if not isinstance(expr.function, ResolvedFunction):

            # search the kernel for the function.
            in_knl_callable = self.find_scoped_function_identifier(
                    expr.function.name)

            if in_knl_callable:
                # associate the newly created ResolvedFunction with the
                # resolved in-kernel callable
                self.scoped_functions[expr.function.name] = in_knl_callable
                return type(expr)(
                        ResolvedFunction(expr.function.name),
                        tuple(self.rec(child, expn_state)
                            for child in expr.parameters),
                        dict(
                            (key, self.rec(val, expn_state))
                            for key, val in six.iteritems(expr.kw_parameters))
                            )

        # this is an unknown function as of yet, do not modify it
        return super(FunctionResolver, self).map_call_with_kwargs(expr,
                expn_state)

    def map_reduction(self, expr, expn_state):
        self.scoped_functions.update(
                expr.operation.get_scalar_callables(self.kernel))
        return super(FunctionResolver, self).map_reduction(expr, expn_state)


def resolve_callables(name, resolved_functions, function_resolvers):

    kernel = resolved_functions[name].subkernel

    from loopy.symbolic import SubstitutionRuleMappingContext
    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())

    function_scoper = FunctionResolver(rule_mapping_context, kernel)

    # scoping fucntions and collecting the scoped functions
    kernel_with_scoped_functions = rule_mapping_context.finish_kernel(
            function_scoper.map_kernel(kernel))

    # updating the functions collected during the scoped functions
    updated_scoped_functions = kernel.scoped_functions.copy()
    updated_scoped_functions.update(function_scoper.scoped_functions)

    return kernel_with_scoped_functions.copy(
            scoped_functions=updated_scoped_functions)

# {{{ program definition

class Program(ImmutableRecord):
    def __init__(self,
            root_kernel_name,
            program_callables_info,
            target=None,
            function_resolvers=None):

        # fixme: check if all sanity checks have been covered?
        assert root_kernel_name in program_callables_info

        if target is None:
            target = program_callables_info[root_kernel_name].subkernel.target

        if function_resolvers is None:
            # populate the function scopers from the target and the loopy
            # specific callable scopers

            assert len(program_callables_info.resolved_functons) == 1

            from loopy.library.function import loopy_specific_callable_scopers
            function_resolvers = [loopy_specific_callable_scopers] + (
                    target.get_device_ast_builder().function_scopers())

            # new function resolvers have arrived, implies we need to resolve
            # the callables identified by this set of resolvers
            program_callables_info = (
                    program_callables_info.with_edit_callables_mode())

            for name, in_knl_callable in program_callables_info.items():
                if isinstance(in_knl_callable, CallableKernel):
                    # resolve the callables in the subkernel
                    resolved_functions = resolve_callables(name,
                            program_callables_info, function_resolvers)

                elif isinstance(in_knl_callable, ScalarCallable):
                    pass
                else:
                    raise NotImplementedError("Unknown callable %s." %
                            type(in_knl_callable).__name__)

            program_callables_info, renames_needed = (
                    program_callables_info.with_exit_edit_mode())
            assert not renames_needed

        super(Program, self).__init__(
                root_kernel_name=root_kernel_name,
                resolved_functions=resolved_functions,
                target=target,
                function_resolvers=function_resolvers)

# }}}


def next_indexed_function_identifier(function):
    """
    Returns an instance of :class:`str` with the next indexed-name in the
    sequence for the name of *function*.

    *Example:* ``Variable('sin_0')`` will return ``'sin_1'``.

    :arg function: Either an instance of :class:`pymbolic.primitives.Variable`
        or :class:`loopy.reduction.ArgExtOp` or
        :class:`loopy.reduction.SegmentedOp`.
    """
    from loopy.library.reduction import ArgExtOp, SegmentedOp
    if isinstance(function, (ArgExtOp, SegmentedOp)):
        return function.copy()
    elif isinstance(function, str):
        function = Variable(function)

    assert isinstance(function, Variable)
    func_name = re.compile(r"^(?P<alpha>\S+?)_(?P<num>\d+?)$")

    match = func_name.match(function.name)

    if match is None:
        if function.name[-1] == '_':
            return "{old_name}0".format(old_name=function.name)
        else:
            return "{old_name}_0".format(old_name=function.name)

    return "{alpha}_{num}".format(alpha=match.group('alpha'),
            num=int(match.group('num'))+1)


class ProgramCallablesInfo(ImmutableRecord):
    def __init__(self, resolved_functions, num_times_callables_called=None,
            history_of_callable_names=None, is_being_edited=False,
            old_resolved_functions={}, num_times_hit_during_editing={},
            renames_needed_after_editing={}):

        if num_times_callables_called is None:
            num_times_callables_called = dict((func_id, 1) for func_id in
                    resolved_functions)
        if history_of_callable_names is None:
            history_of_callable_names = dict((func_id, [func_id]) for func_id in
                    resolved_functions)

        super(ProgramCallablesInfo, self).__init__(
                resolved_functions=resolved_functions,
                num_times_callables_called=num_times_callables_called,
                history_of_callables_callable_names=history_of_callable_names,
                old_resolved_functions=old_resolved_functions,
                is_being_edited=is_being_edited,
                num_times_hit_during_editing=num_times_hit_during_editing,
                renames_needed_after_editing=renames_needed_after_editing)

    def with_edit_callables_mode(self):
        return self.copy(is_being_edited=True,
                old_resolved_functions=self.resolved_functions.copy(),
                num_times_hit_during_editring=dict((func_id, 0) for func_id in
                    self.resolved_functions))

    def with_callable(self, function, in_kernel_callable):
        """
        :arg function: An instance of :class:`pymbolic.primitives.Variable` or
            :class:`loopy.library.reduction.ReductionOpFunction`.

        :arg in_kernel_callables: An instance of
            :class:`loopy.InKernelCallable`.
        """
        assert self.is_being_edited

        from loopy.library.reduction import ArgExtOp, SegmentedOp

        # {{{ sanity checks

        assert isinstance(function, (Variable, ArgExtOp, SegmentedOp))

        # }}}

        renames_needed_after_editing = self.renames_needed_after_editing.copy()
        num_times_hit_during_editing = self.num_times_hit_during_editing.copy()
        num_times_callable_being_called = self.num_times_being_called.copy()
        num_times_hit_during_editing[function.name] += 1

        if in_kernel_callable in self.resolved_functions.values():
            for func_id, in_knl_callable in self.scoped_functions.items():
                if in_knl_callable == in_kernel_callable:
                    num_times_callable_being_called[func_id] += 1
                    num_times_callable_being_called[function] -= 1
                    if num_times_callable_being_called[function] == 0:
                        renames_needed_after_editing[func_id] = function

                    return self, func_id
        else:

            # {{{ ingoring this for now

            if False and isinstance(function, (ArgExtOp, SegmentedOp)):
                # ignoring this casse for now
                # FIXME: If a kernel has two flavors of ArgExtOp then they are
                # overwritten and hence not supported.(for now).
                updated_scoped_functions = self.scoped_functions.copy()
                updated_scoped_functions[function] = in_kernel_callable

                return self.copy(updated_scoped_functions), function.copy()
            # }}}

            #fixme: deal with the history over here.
            unique_function_identifier = function.name
            if self.num_times[function.name] > 1:
                while unique_function_identifier in self.scoped_functions:
                    unique_function_identifier = (
                            next_indexed_function_identifier(
                                unique_function_identifier))

            num_times_callable_being_called[function] -= 1
            num_times_callable_being_called[unique_function_identifier] = 1

            updated_scoped_functions = self.scoped_functions.copy()
            updated_scoped_functions[unique_function_identifier] = in_kernel_callable

            return (self.copy(scoped_functions=updated_scoped_functions),
                    Variable(unique_function_identifier))

    def with_exit_edit_mode(self):
        assert self.is_being_edited

        num_times_callable_being_called = self.num_times_callable_being_called.copy()

        for func_id in self.old_resolved_functions:

            if self.num_times_hit_during_editing[func_id] > 0 and (
                    self.num_times_hit_during_editing[func_id] <
                    num_times_callable_being_called[func_id]):
                unique_function_identifier = func_id

                while unique_function_identifier in self.scoped_functions:
                    unique_function_identifier = (
                            next_indexed_function_identifier(
                                unique_function_identifier))

                (num_times_callable_being_called[func_id],
                    num_times_callable_being_called[unique_function_identifier]) = (
                            self.num_times_hit_while_editing[func_id],
                            num_times_callable_being_called[func_id] -
                            self.num_times_being_hit_while_editing[func_id])

            if self.num_times_hit_during_edition[func_id] > 0 and (
                    self.num_times_hit_during_editing[func_id] >
                    num_times_callable_being_called[func_id]):
                raise RuntimeError("Should not traverse more number of times than "
                        "it is called.")

        return (
                self.copy(
                    is_begin_edited=False,
                    num_times_callable_being_called=num_times_callable_being_called,
                    num_times_hit_during_editing={},
                    renames_needed_while_editing={}),
                self.renames_needed_while_editing)

    def __getitem__(self, item):
        return self.reoslved_functions[item]

    def __contains__(self, item):
        return item in self.resolved_functions

    def items(self):
        return self.resolved_functions.items()


def make_program_from_kernel(kernel):
    callable_knl = CallableKernel(subkernel=kernel)
    resolved_functions = {kernel.name: callable_knl}
    program_callables_info = ProgramCallablesInfo(resolved_functions)

    program = Program(
            root_kernel_name=kernel.name,
            program_callables_info=program_callables_info)

    return program


# vim: foldmethod=marker

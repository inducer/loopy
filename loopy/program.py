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

import re

from pytools import ImmutableRecord
from pymbolic.primitives import Variable
from functools import wraps

from loopy.symbolic import (RuleAwareIdentityMapper, ResolvedFunction,
        CombineMapper, SubstitutionRuleMappingContext)
from loopy.kernel.function_interface import (
        CallableKernel, ScalarCallable)
from loopy.kernel.instruction import (
        MultiAssignmentBase, CInstruction, _DataObliviousInstruction)
from loopy.diagnostic import LoopyError
from loopy.library.reduction import ReductionOpFunction

from loopy.kernel import LoopKernel
from loopy.tools import update_persistent_hash
from pymbolic.primitives import Call, CallWithKwargs
from functools import reduce

__doc__ = """

.. currentmodule:: loopy

.. autoclass:: Program

.. autofunction:: make_program
.. autofunction:: iterate_over_kernels_if_given_program

"""


def find_in_knl_callable_from_identifier(
        function_id_to_in_knl_callable_mappers, target, identifier):
    """
    Returns an instance of
    :class:`loopy.kernel.function_interface.InKernelCallable` if the
    :arg:`identifier` is known to any kernel function scoper, otherwise returns
    *None*.
    """
    for func_id_to_in_knl_callable_mapper in (
            function_id_to_in_knl_callable_mappers):
        # fixme: do we really need to given target for the function
        in_knl_callable = func_id_to_in_knl_callable_mapper(
                target, identifier)
        if in_knl_callable is not None:
            return in_knl_callable

    return None


class CallableResolver(RuleAwareIdentityMapper):
    #FIXME: Recheck this!
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
    def __init__(self, rule_mapping_context, known_callables):
        super().__init__(rule_mapping_context)
        self.resolved_functions = {}
        self.known_callables = known_callables

    def map_call(self, expr, expn_state):
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

        if not isinstance(expr.function, ResolvedFunction):
            # FIXME: Do we need to care about ReductionOpFunctions over here?
            in_knl_callable = self.known_callables.get(expr.function.name)

            if in_knl_callable:
                if expr.function.name in self.resolved_functions:
                    assert self.resolved_functions[expr.function.name] == (
                            in_knl_callable)
                self.resolved_functions[expr.function.name] = in_knl_callable
                return type(expr)(
                        ResolvedFunction(expr.function.name),
                        tuple(self.rec(child, expn_state)
                            for child in expr.parameters),
                        {
                            key: self.rec(val, expn_state)
                            for key, val in expr.kw_parameters.items()}
                            )
            else:
                # FIXME: Once function mangler is completely deprecated raise here.
                # Oh function mangler I loathe you so much!
                pass
        else:
            self.resolved_functions[expr.function.name] = (
                    self.known_callables[expr.function.name])

        return super().map_call_with_kwargs(expr,
                expn_state)


# {{{ program

class Program(ImmutableRecord):
    """
    Records the information about all the callables in a :mod:`loopy` program.

    .. attribute:: entrypoints

        A :class:`frozenset` of the names of the kernels which
        could be called from the host.

    .. attribute:: callables_table

        An instance of :class:`dict` mapping the function identifiers in a
        kernel to their associated instances of
        :class:`loopy.kernel.function_interface.InKernelCallable`.

    .. attribute:: target

        An instance of :class:`loopy.target.TargetBase`.

    .. attribute:: func_id_to_in_knl_callables_mappers

        A :class:`frozenset` of functions of the signature ``(target:
        TargetBase, function_indentifier: str)`` that would return an instance
        of :class:`loopy.kernel.function_interface.InKernelCallable` or *None*.

    .. note::

        - To create an instance of :class:`loopy.Program`, it is recommended to
            go through :method:`loopy.make_kernel`.
        - This data structure and its attributes should be considered
          immutable, any modifications should be done through :method:`copy`.

    .. automethod:: __init__
    .. automethod:: with_root_kernel
    .. method:: __getitem__(name)

        Look up the resolved callable with identifier *name*.
    """
    def __init__(self,
            entrypoints=None,
            callables_table={},
            target=None,
            func_id_to_in_knl_callable_mappers=[]):

        # {{{ sanity checks

        assert isinstance(callables_table, dict)

        # }}}

        super().__init__(
                entrypoints=entrypoints,
                callables_table=callables_table,
                target=target,
                func_id_to_in_knl_callable_mappers=(
                    func_id_to_in_knl_callable_mappers))

        self._program_executor_cache = {}

    hash_fields = (
            "entrypoints",
            "callables_table",
            "target",)

    update_persistent_hash = update_persistent_hash

    def copy(self, **kwargs):
        target = kwargs.pop("target", None)
        program = super().copy(**kwargs)
        if target:
            from loopy.kernel import KernelState
            if max(callable_knl.subkernel.state for callable_knl in
                    self.callables_table.values() if
                    isinstance(callable_knl, CallableKernel)) > (
                            KernelState.INITIAL):
                if not isinstance(target, type(self.target)):
                    raise LoopyError("One of the kenels in the program has been "
                            "preprocessed, cannot modify target now.")
            callables = {}
            for func_id, clbl in program.callables_table.items():
                if isinstance(clbl, CallableKernel):
                    knl = clbl.subkernel
                    knl = knl.copy(target=target)
                    clbl = clbl.copy(subkernel=knl)
                elif isinstance(clbl, ScalarCallable):
                    pass
                else:
                    raise NotImplementedError()
                callables[func_id] = clbl

            program = super().copy(
                callables_table=callables, target=target)

        return program

    def with_entrypoints(self, entrypoints):
        """
        :param entrypoints: Either a comma-separated :class:`str` or
        :class:`frozenset`.
        """
        if isinstance(entrypoints, str):
            entrypoints = frozenset([e.strip() for e in
                entrypoints.split(",")])

        assert isinstance(entrypoints, frozenset)

        return self.copy(entrypoints=entrypoints)

    @property
    def state(self):
        """ Returns an instance of :class:`loopy.kernel.KernelState`. """
        return min(callable_knl.subkernel.state for callable_knl in
                self.callables_table.values() if
                isinstance(callable_knl, CallableKernel))

    def with_kernel(self, kernel):
        # FIXME: Currently only replaces kernel. Should also work for adding.
        # FIXME: Document
        new_in_knl_callable = self.callables_table[kernel.name].copy(
                subkernel=kernel)
        new_callables = self.callables_table.copy()
        new_callables[kernel.name] = new_in_knl_callable
        return self.copy(callables_table=new_callables)

    def with_resolved_callables(self):
        from loopy.library.function import get_loopy_callables
        from loopy.kernel import KernelState

        if self.state >= KernelState.CALLS_RESOLVED:
            return self

        known_callables = self.callables_table
        known_callables.update(self.target.get_device_ast_builder().known_callables)
        known_callables.update(get_loopy_callables())
        # update the known callables from the target.
        callables_table = {e: self.callables_table[e] for e in
                self.entrypoints}

        # start a traversal to collect all the callables
        queue = list(self.entrypoints)

        while queue:
            top = queue[0]
            assert top in callables_table
            queue = queue[1:]

            knl = callables_table[top].subkernel
            rule_mapping_context = SubstitutionRuleMappingContext(
                    knl.substitutions, knl.get_var_name_generator())
            callables_collector = CallableResolver(
                    rule_mapping_context,
                    known_callables)
            knl = rule_mapping_context.finish_kernel(
                    callables_collector.map_kernel(knl))
            knl = knl.copy(state=KernelState.CALLS_RESOLVED)
            callables_table[top] = callables_table[top].copy(subkernel=knl)

            for func, clbl in callables_collector.resolved_functions.items():
                if func not in callables_table:
                    if isinstance(clbl, CallableKernel):
                        queue.append(func)
                    callables_table[func] = clbl
                else:
                    assert callables_table[func] == clbl

        return self.copy(callables_table=callables_table)

    def __getitem__(self, name):
        result = self.callables_table[name]
        if isinstance(result, CallableKernel):
            return result.subkernel
        else:
            return result

    def __getattr__(self, attr):
        if self.entrypoints:
            if attr in self.entrypoints:
                return lambda *args, **kwargs: self(*args, entrypoint=attr,
                        **kwargs)

        return super().__getattr__(attr)

    def __call__(self, *args, **kwargs):
        entrypoint = kwargs.get("entrypoint", None)

        if entrypoint is None:
            # did not receive an entrypoint for the program to execute
            if len(self.entrypoints) == 1:
                entrypoint, = list(self.entrypoints)
            else:
                raise TypeError("Program.__call__() missing 1 required"
                        " keyword argument: 'entrypoint'")

        if entrypoint not in self.entrypoints:
            raise LoopyError("'{}' not in list possible entrypoints supplied to"
                    " the program. Maybe you want to invoke 'with_entrypoints'"
                    " before calling the program.".format(entrypoint))

        kwargs["entrypoint"] = entrypoint

        key = self.target.get_kernel_executor_cache_key(*args, **kwargs)
        try:
            pex = self._program_executor_cache[key]
        except KeyError:
            pex = self.target.get_kernel_executor(self, *args, **kwargs)
            self._program_executor_cache[key] = pex

        return pex(*args, **kwargs)

    def __str__(self):
        # FIXME: do a topological sort by the call graph

        def strify_callable(clbl):
            if isinstance(clbl, CallableKernel):
                return str(clbl.subkernel)
            else:
                return str(clbl)

        return "\n".join(
                strify_callable(clbl)
                for name, clbl in self.callables_table).items()

    def __setstate__(self, state_obj):
        super().__setstate__(state_obj)

        self._program_executor_cache = {}

# }}}


def next_indexed_function_identifier(function_id):
    """
    Returns an instance of :class:`str` with the next indexed-name in the
    sequence for the name of *function*.

    *Example:* ``'sin_0'`` will return ``'sin_1'``.

    :arg function_id: Either an instance of :class:`str`.
    """

    # {{{ sanity checks

    assert isinstance(function_id, str)

    # }}}

    func_name = re.compile(r"^(?P<alpha>\S+?)_(?P<num>\d+?)$")

    match = func_name.match(function_id)

    if match is None:
        if function_id[-1] == "_":
            return f"{function_id}0"
        else:
            return f"{function_id}_0"

    return "{alpha}_{num}".format(alpha=match.group("alpha"),
            num=int(match.group("num"))+1)


class ResolvedFunctionRenamer(RuleAwareIdentityMapper):
    """
    Mapper to rename the resolved functions in an expression according to
    *renaming_dict*.
    """
    def __init__(self, rule_mapping_context, renaming_dict):
        super().__init__(
                rule_mapping_context)
        self.renaming_dict = renaming_dict

    def map_resolved_function(self, expr, expn_state):
        if expr.name in self.renaming_dict:
            return ResolvedFunction(self.renaming_dict[expr.name])
        else:
            return super().map_resolved_function(
                    expr, expn_state)


def rename_resolved_functions_in_a_single_kernel(kernel,
        renaming_dict):
    """
    Returns a copy of *kernel* with the instances of :class:`ResolvedFunction`
    renames according to *renaming_dict*.
    """
    from loopy.symbolic import SubstitutionRuleMappingContext
    rule_mapping_context = SubstitutionRuleMappingContext(
                kernel.substitutions, kernel.get_var_name_generator())
    resolved_function_renamer = ResolvedFunctionRenamer(rule_mapping_context,
            renaming_dict)
    return (
            rule_mapping_context.finish_kernel(
                resolved_function_renamer.map_kernel(kernel)))


class CallablesIDCollector(CombineMapper):
    """
    Returns an instance of :class:`frozenset` containing instances of
    :class:`loopy.kernel.function_interface.InKernelCallable` in the
    :attr:``kernel`.
    """
    def combine(self, values):
        import operator
        return reduce(operator.or_, values, frozenset())

    def map_resolved_function(self, expr):
        return frozenset([expr.name])

    def map_constant(self, expr):
        return frozenset()

    def map_kernel(self, kernel):
        callables_in_insn = frozenset()

        for insn in kernel.instructions:
            if isinstance(insn, MultiAssignmentBase):
                callables_in_insn = callables_in_insn | (
                        self(insn.expression))
            elif isinstance(insn, (CInstruction, _DataObliviousInstruction)):
                pass
            else:
                raise NotImplementedError(type(insn).__name__)

        for rule in kernel.substitutions.values():
            callables_in_insn = callables_in_insn | (
                    self(rule.expression))

        return callables_in_insn

    map_variable = map_constant
    map_function_symbol = map_constant
    map_tagged_variable = map_constant
    map_type_cast = map_constant


def _get_callable_ids_for_knl(knl, callables):
    clbl_id_collector = CallablesIDCollector()

    return frozenset().union(*(
        _get_callable_ids_for_knl(callables[clbl].subkernel, callables) |
        frozenset([clbl]) if isinstance(callables[clbl], CallableKernel) else
        frozenset([clbl])
        for clbl in clbl_id_collector.map_kernel(knl)))


def _get_callable_ids(callables, entrypoints):
    return frozenset().union(*(
        _get_callable_ids_for_knl(callables[e].subkernel, callables) for e in
        entrypoints))


def make_clbl_inf_ctx(callables, entrypoints):
    return CallablesInferenceContext(callables, _get_callable_ids(callables,
        entrypoints))


class CallablesInferenceContext(ImmutableRecord):
    def __init__(self, callables, old_callable_ids, history={}):
        assert isinstance(callables, dict)

        super().__init__(
                callables=callables,
                old_callable_ids=old_callable_ids,
                history=history)

    # {{{ interface to perform edits on callables

    def with_callable(self, function, in_kernel_callable):
        """
        Returns an instance of :class:`tuple` ``(new_self, new_function)``.

        :arg function: An instance of :class:`pymbolic.primitives.Variable` or
            :class:`loopy.library.reduction.ReductionOpFunction`.

        :arg in_kernel_callable: An instance of
            :class:`loopy.InKernelCallable`.
        """

        # {{{ sanity checks

        if isinstance(function, str):
            function = Variable(function)

        assert isinstance(function, (Variable, ReductionOpFunction))

        # }}}

        history = self.history.copy()

        if in_kernel_callable in self.callables.values():
            # the callable already exists, hence return the function
            # identifier corresponding to that callable.
            for func_id, in_knl_callable in self.callables.items():
                if in_knl_callable == in_kernel_callable:
                    history[func_id] = function.name
                    if isinstance(func_id, str):
                        return (
                                self.copy(
                                    history=history),
                                Variable(func_id))
                    else:
                        assert isinstance(func_id, ReductionOpFunction)
                        return (
                                self.copy(
                                    history=history),
                                func_id)

            assert False
        else:
            # {{{ handle ReductionOpFunction

            if isinstance(function, ReductionOpFunction):
                # FIXME: Check if we have 2 ArgMax functions
                # with different types in the same kernel the generated code
                # does not mess up the types.
                unique_function_identifier = function.copy()
                updated_callables = self.callables.copy()
                updated_callables[unique_function_identifier] = (
                        in_kernel_callable)

                return (
                        self.copy(
                            callables=updated_callables),
                        unique_function_identifier)

            # }}}

            unique_function_identifier = function.name

            while unique_function_identifier in self.callables:
                unique_function_identifier = (
                        next_indexed_function_identifier(
                            unique_function_identifier))

        updated_callables = self.callables.copy()
        updated_callables[unique_function_identifier] = (
                in_kernel_callable)

        history[unique_function_identifier] = function.name

        return (
                self.copy(
                    history=history,
                    callables=updated_callables),
                Variable(unique_function_identifier))

    def finish_program(self, program, renamed_entrypoints):
        """
        Returns a copy of *program* with renaming of the callables done whenever
        needed.

        *For example: * If all the ``sin`` got diverged as ``sin_0, sin_1``,
        then all the renaming is done such that one of flavors of the callable
        is renamed back to ``sin``.

        :param renamed_entrypoints: A :class:`frozenset` of the names of the
            renamed callable kernels which correspond to the entrypoints in
            *self.callables_table*.
        """
        assert len(renamed_entrypoints) == len(program.entrypoints)
        new_callable_ids = _get_callable_ids(self.callables, renamed_entrypoints)

        callees_with_entrypoint_names = (program.entrypoints &
                new_callable_ids) - renamed_entrypoints

        renames = {}
        new_callables = {}

        for c in callees_with_entrypoint_names:
            unique_function_identifier = c

            while unique_function_identifier in self.callables:
                unique_function_identifier = (
                        next_indexed_function_identifier(
                            unique_function_identifier))

            renames[c] = unique_function_identifier

        # we should perform a rewrite here.

        for e in renamed_entrypoints:
            renames[e] = self.history[e]
            assert renames[e] in program.entrypoints

        # {{{ calculate the renames needed

        for old_func_id in ((self.old_callable_ids-new_callable_ids) -
                program.entrypoints):
            # at this point we should not rename anything to the names of
            # entrypoints
            for new_func_id in (new_callable_ids-renames.keys()) & set(
                    self.history.keys()):
                if old_func_id == self.history[new_func_id]:
                    renames[new_func_id] = old_func_id
                    break
        # }}}

        for e in renamed_entrypoints:
            new_subkernel = self.callables[e].subkernel.copy(name=self.history[e])
            new_subkernel = rename_resolved_functions_in_a_single_kernel(
                    new_subkernel, renames)
            new_callables[self.history[e]] = self.callables[e].copy(
                    subkernel=new_subkernel)

        for func_id in new_callable_ids-renamed_entrypoints:
            in_knl_callable = self.callables[func_id]
            if isinstance(in_knl_callable, CallableKernel):
                # if callable kernel, perform renames inside its expressions.
                old_subkernel = in_knl_callable.subkernel
                new_subkernel = rename_resolved_functions_in_a_single_kernel(
                        old_subkernel, renames)
                in_knl_callable = (
                        in_knl_callable.copy(subkernel=new_subkernel))
            elif isinstance(in_knl_callable, ScalarCallable):
                pass
            else:
                raise NotImplementedError("Unknown callable type %s." %
                        type(in_knl_callable).__name__)

            if func_id in renames:
                new_func_id = renames[func_id]
                if isinstance(in_knl_callable, CallableKernel):
                    in_knl_callable = (in_knl_callable.copy(
                        subkernel=in_knl_callable.subkernel.copy(
                            name=new_func_id)))
                new_callables[new_func_id] = in_knl_callable
            else:
                if isinstance(in_knl_callable, CallableKernel):
                    in_knl_callable = in_knl_callable.copy(
                        subkernel=in_knl_callable.subkernel.copy(
                            name=func_id))
                new_callables[func_id] = in_knl_callable

        return program.copy(callables_table=new_callables)

    # }}}

    def __getitem__(self, name):
        result = self.callables[name]
        return result


# {{{ helper functions

def make_program(kernel):
    """
    Returns an instance of :class:`loopy.Program` with *kernel* as the only
    callable kernel.
    """

    # get the program from program callables info
    #FIXME:(For KK): do we need to register the current kernel in
    # func_id_to_in_knl_callable_mappers
    #FIXME(For inducer): Deriving the target of this program from the kernel's
    # target.
    program = Program(
            callables_table={
                kernel.name: CallableKernel(kernel)},
            target=kernel.target)

    return program


def iterate_over_kernels_if_given_program(transform_for_single_kernel):
    """
    Function wrapper for transformations of the type ``transform(kernel:
    LoopKernel, *args, **kwargs): LoopKernel``. Returns a function with the
    ``transform`` being implemented on all of the callable kernels in a
    :class:`loopy.Program`.
    """
    def _collective_transform(*args, **kwargs):
        if "program" in kwargs:
            program_or_kernel = kwargs.pop("program")
        elif "kernel" in kwargs:
            program_or_kernel = kwargs.pop("kernel")
        else:
            program_or_kernel = args[0]
            args = args[1:]

        if isinstance(program_or_kernel, Program):
            program = program_or_kernel
            new_callables = {}
            for func_id, in_knl_callable in program.callables_table.items():
                if isinstance(in_knl_callable, CallableKernel):
                    new_subkernel = transform_for_single_kernel(
                            in_knl_callable.subkernel, *args, **kwargs)
                    in_knl_callable = in_knl_callable.copy(
                            subkernel=new_subkernel)
                elif isinstance(in_knl_callable, ScalarCallable):
                    pass
                else:
                    raise NotImplementedError("Unknown type of callable %s." % (
                        type(in_knl_callable).__name__))

                new_callables[func_id] = in_knl_callable

            return program.copy(callables_table=new_callables)
        else:
            assert isinstance(program_or_kernel, LoopKernel)
            kernel = program_or_kernel
            return transform_for_single_kernel(kernel, *args, **kwargs)

    return wraps(transform_for_single_kernel)(_collective_transform)


def update_table(callables_table, clbl_id, clbl):
    from loopy.kernel.function_interface import InKernelCallable
    assert isinstance(clbl, InKernelCallable)

    for i, c in callables_table.items():
        if c == clbl:
            return i, callables_table

    while clbl_id in callables_table:
        clbl_id = next_indexed_function_identifier(clbl_id)

    callables_table[clbl_id] = clbl

    return clbl_id, callables_table

# }}}


# vim: foldmethod=marker

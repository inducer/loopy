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
import collections

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
from pymbolic.primitives import Call
from functools import reduce
from pyrsistent import pmap, PMap

__doc__ = """
.. currentmodule:: loopy.translation_unit

.. autoclass:: CallablesInferenceContext

.. autofunction:: make_program

.. autofunction:: for_each_kernel

"""


# {{{ CallableResolver

def _is_a_reduction_op(expr):
    if isinstance(expr, ResolvedFunction):
        return _is_a_reduction_op(expr.function)

    from loopy.library.reduction import ReductionOpFunction
    return isinstance(expr, ReductionOpFunction)


class CallableResolver(RuleAwareIdentityMapper):
    """
    Resolves callables in expressions and records the names of the calls
    resolved.

    .. attribute:: known_callables

        An instance of :class:`frozenset` of the call names to be resolved.

    .. attribute:: rule_mapping_context

        An instance of :class:`loopy.symbolic.RuleMappingContext`.

    .. attribute:: calls_resolved

        A :class:`set` of calls that were resolved. Updated during an
        expression traversal.
    """
    def __init__(self, rule_mapping_context, known_callables):
        assert isinstance(known_callables, frozenset)

        super().__init__(rule_mapping_context)

        self.known_callables = known_callables

        # a record of the call names that were resolved
        self.calls_resolved = set()

    def map_call(self, expr, expn_state):
        from loopy.symbolic import parse_tagged_name

        if not _is_a_reduction_op(expr.function):
            name, tag = parse_tagged_name(expr.function)
        else:
            if isinstance(expr.function, ResolvedFunction):
                name = expr.function.function
            else:
                name = expr.function

        if name in self.known_callables:
            params = tuple(self.rec(par, expn_state) for par in expr.parameters)

            # record that we resolved a call
            self.calls_resolved.add(name)

            function = expr.function

            if not isinstance(expr.function, ResolvedFunction):
                function = ResolvedFunction(expr.function)

            return Call(function, params)

        return super().map_call(expr, expn_state)

    def map_call_with_kwargs(self, expr):
        # See https://github.com/inducer/loopy/pull/323
        raise NotImplementedError

# }}}


# {{{ translation unit

class TranslationUnit(ImmutableRecord):
    """
    Records the information about all the callables in a :mod:`loopy` program.

    An instance of :class:`TranslationUnit` is the object that gets lowered
    for a :class:`loopy.target.TargetBase`.


    .. attribute:: entrypoints

        A :class:`frozenset` of the names of the kernels which
        could be called from the host.

    .. attribute:: default_entrypoint

        The :class:`~loopy.LoopKernel` representing the main entrypoint
        of the program, if defined. Currently, this attribute may only be
        accessed if there is exactly one entrypoint in the program.

    .. attribute:: callables_table

        An instance of :class:`pyrsistent.PMap` mapping the function
        identifiers in a kernel to their associated instances of
        :class:`~loopy.kernel.function_interface.InKernelCallable`.

    .. attribute:: target

        An instance of :class:`loopy.target.TargetBase`.

    .. attribute:: func_id_to_in_knl_callables_mappers

        A :class:`frozenset` of functions of the signature ``(target:
        TargetBase, function_indentifier: str)`` that would return an instance
        of :class:`loopy.kernel.function_interface.InKernelCallable` or *None*.

    .. automethod:: __call__
    .. automethod:: copy
    .. automethod:: __getitem__
    .. automethod:: with_kernel

    .. note::

        - To create an instance of :class:`loopy.TranslationUnit`, it is
          recommended to go through :func:`loopy.make_kernel`.
        - This data structure and its attributes should be considered
          immutable, any modifications should be done through
          :meth:`~TranslationUnit.copy`.

    """
    def __init__(self,
            entrypoints=frozenset(),
            callables_table=pmap(),
            target=None,
            func_id_to_in_knl_callable_mappers=[]):

        # {{{ sanity checks

        assert isinstance(callables_table, collections.abc.Mapping)
        assert isinstance(entrypoints, frozenset)

        if not isinstance(callables_table, PMap):
            callables_table = pmap(callables_table)

        # }}}

        super().__init__(
                entrypoints=entrypoints,
                callables_table=pmap(callables_table),
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
            if max(callable_knl.subkernel.state
                   for callable_knl in self.callables_table.values()
                   if isinstance(callable_knl, CallableKernel)) > (
                            KernelState.INITIAL):
                if not isinstance(target, type(self.target)):
                    raise LoopyError("One of the kernels in the program has been "
                            "preprocessed, cannot modify target now.")

            new_callables = {}
            for func_id, clbl in program.callables_table.items():
                if isinstance(clbl, CallableKernel):
                    knl = clbl.subkernel
                    knl = knl.copy(target=target)
                    clbl = clbl.copy(subkernel=knl)
                elif isinstance(clbl, ScalarCallable):
                    pass
                else:
                    raise NotImplementedError()
                new_callables[func_id] = clbl

            program = super().copy(
                callables_table=new_callables, target=target)

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
        return min(callable_knl.subkernel.state
                   for callable_knl in self.callables_table.values()
                   if isinstance(callable_knl, CallableKernel))

    def with_kernel(self, kernel):
        """
        If *self* contains a callable kernel with *kernel*'s name, replaces its
        subkernel and returns a copy of *self*. Else records a new callable
        kernel with *kernel* as its subkernel.

        :arg kernel: An instance of :class:`loopy.LoopKernel`.
        :returns: Copy of *self* with updated callable kernels.
        """
        if kernel.name in self.callables_table:
            # update the callable kernel
            new_in_knl_callable = self.callables_table[kernel.name].copy(
                    subkernel=kernel)
            new_callables = self.callables_table.remove(kernel.name).set(
                    kernel.name, new_in_knl_callable)
            return self.copy(callables_table=new_callables)
        else:
            # add a new callable kernel
            clbl = CallableKernel(kernel)
            new_callables = self.callables_table.set(kernel.name, clbl)
            return self.copy(callables_table=new_callables)

    def __getitem__(self, name):
        """
        For the callable named *name*, return a :class:`loopy.LoopKernel` if
        it's a :class:`~loopy.kernel.function_interface.CallableKernel`
        otherwise return the callable itself.
        """
        result = self.callables_table[name]
        if isinstance(result, CallableKernel):
            return result.subkernel
        else:
            return result

    @property
    def default_entrypoint(self):
        if len(self.entrypoints) == 1:
            entrypoint, = self.entrypoints
            return self[entrypoint]
        else:
            raise ValueError("TranslationUnit has multiple possible entrypoints."
                             " The default entry point kernel is not uniquely"
                             " determined.")

    def __call__(self, *args, **kwargs):
        """
        Builds and calls the *entrypoint* kernel, if
        :attr:`TranslationUnit.target` is an executable target.

        :arg entrypoint: The name of the entrypoint callable to be called.
            Defaults to *the* entrypoint if there is only one.
        """
        entrypoint = kwargs.get("entrypoint", None)

        if entrypoint is None:
            # did not receive an entrypoint for the program to execute
            if len(self.entrypoints) == 1:
                entrypoint, = self.entrypoints
            else:
                raise TypeError("TranslationUnit.__call__() missing 1 required"
                        " keyword argument: 'entrypoint'. "
                        "(Multiple possible entrypoints are present in the "
                        "program.)")

        if entrypoint not in self.entrypoints:
            raise LoopyError(f"'{entrypoint}' not in list of possible entrypoints "
                    "for the program. "
                    "Maybe you want to invoke 'with_entrypoints' before "
                    "calling the program?")

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
            return str(clbl.subkernel)

        return "\n".join(
                strify_callable(clbl)
                for name, clbl in self.callables_table.items()
                if isinstance(clbl, CallableKernel))

    def __setstate__(self, state_obj):
        super().__setstate__(state_obj)

        self._program_executor_cache = {}

    def __hash__(self):
        from loopy.tools import LoopyKeyBuilder
        from pytools.persistent_dict import new_hash
        key_hash = new_hash()
        self.update_persistent_hash(key_hash, LoopyKeyBuilder())
        return hash(key_hash.digest())


class Program(TranslationUnit):
    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("Program is deprecated, use TranslationUnit instead, "
             "will be removed in 2022", DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

# }}}


# {{{ next_indexed_function_id

def next_indexed_function_id(function_id):
    """
    Returns an instance of :class:`str` with the next indexed-name in the
    sequence for the name of *function_id*.

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

# }}}


# {{{ rename_resolved_functions_in_a_single_kernel

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

# }}}


# {{{ CallablesIDCollector

class CallablesIDCollector(CombineMapper):
    """
    Mapper to collect function identifiers of all resolved callables in an
    expression.
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
        _get_callable_ids_for_knl(callables[e].subkernel, callables)
        for e in entrypoints))

# }}}


# {{{ CallablesInferenceContext

def make_clbl_inf_ctx(callables, entrypoints):
    return CallablesInferenceContext(callables)


class CallablesInferenceContext(ImmutableRecord):
    """
    Helper class for housekeeping a :attr:`loopy.TranslationUnit.callables_table`
    while traversing through callables of :class:`loopy.TranslationUnit`.

    .. attribute:: callables

       A mapping from the callable names to instances of
       :class:`loopy.kernel.function_interface.InKernelCallable`.

    .. attribute:: renames

       A mapping from old function identifiers to a :class:`frozenset` of new
       function identifiers.

    .. attribute:: new_entrypoints

       A :class:`frozenset` of renamed entrypoint names.

    .. automethod:: with_callable

    .. automethod:: finish_program

    .. automethod:: __getitem__
    """
    def __init__(self, callables,
                 renames=collections.defaultdict(frozenset),
                 new_entrypoints=frozenset()):
        assert isinstance(callables, collections.abc.Mapping)
        callables = dict(callables)

        super().__init__(callables=callables,
                         renames=renames,
                         new_entrypoints=new_entrypoints)

    def with_callable(self, old_function_id, new_clbl,
                      is_entrypoint=False):
        """
        Updates the callable referred by *function_id*'s in *self*'s namespace
        to *new_clbl*.

        :arg old_function_id: An instance of :class:`pymbolic.primitives.Variable` or
            :class:`loopy.library.reduction.ReductionOpFunction`.

        :arg new_clbl: An instance of
            :class:`loopy.kernel.function_interface.InKernelCallable`.

        :returns: ``(new_self, new_function_id)`` is a copy of *self* with
            *new_clbl* in its namespace. *new_clbl* would be referred by
            *new_function_id* in *new_self*'s namespace.
        """

        assert isinstance(old_function_id, (str, Variable, ReductionOpFunction))

        if isinstance(old_function_id, Variable):
            old_function_id = old_function_id.name

        renames = self.renames.copy()

        # if the callable already exists => return the function
        # identifier corresponding to that callable.
        for func_id, clbl in self.callables.items():
            if clbl == new_clbl:
                renames[old_function_id] |= frozenset([func_id])
                if isinstance(func_id, str):
                    new_entrypoints = self.new_entrypoints
                    if is_entrypoint:
                        new_entrypoints |= frozenset([func_id])
                    return (self.copy(renames=renames,
                                      new_entrypoints=new_entrypoints),
                            Variable(func_id),)
                else:
                    assert not is_entrypoint
                    assert isinstance(func_id, ReductionOpFunction)
                    return (self.copy(renames=renames),
                            func_id)

        # {{{ handle ReductionOpFunction

        if isinstance(old_function_id, ReductionOpFunction):
            # FIXME: Check if we have 2 ArgMax functions
            # with different types in the same kernel the generated code
            # does not mess up the types.
            assert not is_entrypoint
            unique_function_id = old_function_id.copy()
            updated_callables = self.callables.copy()
            updated_callables[unique_function_id] = new_clbl
            renames[old_function_id] |= frozenset([unique_function_id])

            return (self.copy(callables=updated_callables,
                              renames=renames),
                    unique_function_id)

        # }}}

        # {{{ must allocate a new clbl in the namespace => find a unique id for it

        unique_function_id = old_function_id

        while unique_function_id in self.callables:
            unique_function_id = next_indexed_function_id(unique_function_id)

        # }}}

        updated_callables = self.callables.copy()
        updated_callables[unique_function_id] = new_clbl
        renames[old_function_id] |= frozenset([unique_function_id])

        new_entrypoints = self.new_entrypoints
        if is_entrypoint:
            new_entrypoints |= frozenset([unique_function_id])

        return (self.copy(renames=renames,
                          callables=updated_callables,
                          new_entrypoints=new_entrypoints),
                Variable(unique_function_id))

    def finish_program(self, program):
        """
        Returns a copy of *program* with rollback renaming of the callables
        done whenever possible.

        For example: If all the ``sin`` function ids diverged as
        ``sin_0``, ``sin_1``, then all the renaming is done such that one of
        the flavors of the callable is renamed back to ``sin``.
        """
        # FIXME: Generalize this if an inference happens over a proper subgraph
        # of the callgraph (the following assert should be removed)
        assert len(self.new_entrypoints) == len(program.entrypoints)

        # {{{ get all the callables reachable from the new entrypoints.

        # get the names of all callables reachable from the new entrypoints
        new_callable_ids = _get_callable_ids(self.callables, self.new_entrypoints)

        # get the history of function ids from the performed renames:
        history = {}
        for old_func_id, new_func_ids in self.renames.items():
            for new_func_id in new_func_ids:
                if new_func_id in (new_callable_ids | self.new_entrypoints):
                    history[new_func_id] = old_func_id

        # }}}

        # AIM: Preserve the entrypoints of *program*

        # If there are any callees having old entrypoint names => mark them for
        # renaming
        callees_with_old_entrypoint_names = ((program.entrypoints & new_callable_ids)
                                             - self.new_entrypoints)

        todo_renames = {}
        new_callables = {}

        for c in callees_with_old_entrypoint_names:
            unique_func_id = c

            while unique_func_id in self.callables:
                unique_func_id = next_indexed_function_id(unique_func_id)

            todo_renames[c] = unique_func_id

        for e in self.new_entrypoints:
            # note renames to "rollback" the renaming of entrypoints
            todo_renames[e] = history[e]
            assert todo_renames[e] in program.entrypoints

        # try to rollback the names as much as possible
        for new_id in new_callable_ids:
            old_func_id = history[new_id]
            if (isinstance(old_func_id, str)
                    and old_func_id not in set(todo_renames.values())):
                todo_renames[new_id] = old_func_id

        # {{{ perform the renames form todo_renames

        for func_id in (new_callable_ids | self.new_entrypoints):
            clbl = self.callables[func_id]
            if func_id in todo_renames:
                assert history[func_id] == todo_renames[func_id]
                func_id = todo_renames[func_id]
            if isinstance(clbl, CallableKernel):
                subknl = clbl.subkernel.copy(name=func_id)
                subknl = rename_resolved_functions_in_a_single_kernel(subknl,
                                                                      todo_renames)

                clbl = clbl.copy(subkernel=subknl)

            new_callables[func_id] = clbl

        # }}}

        return program.copy(callables_table=new_callables)

    def __getitem__(self, name):
        result = self.callables[name]
        return result

# }}}


# {{{ helper functions

def make_program(kernel):
    """
    Returns an instance of :class:`loopy.TranslationUnit` with *kernel* as the only
    callable kernel.
    """

    program = TranslationUnit(
            callables_table={
                kernel.name: CallableKernel(kernel)},
            target=kernel.target)

    return program


def for_each_kernel(transform):
    """
    Function wrapper for transformations of the type ``transform(kernel:
    LoopKernel, *args, **kwargs) -> LoopKernel``. Returns a function that would
    apply *transform* to all callable kernels in a :class:`loopy.TranslationUnit`.
    """
    def _collective_transform(*args, **kwargs):
        if "translation_unit" in kwargs:
            t_unit_or_kernel = kwargs.pop("translation_unit")
        elif "kernel" in kwargs:
            t_unit_or_kernel = kwargs.pop("kernel")
        else:
            t_unit_or_kernel = args[0]
            args = args[1:]

        if isinstance(t_unit_or_kernel, TranslationUnit):
            t_unit = t_unit_or_kernel
            new_callables = {}
            for func_id, clbl in t_unit.callables_table.items():
                if isinstance(clbl, CallableKernel):
                    new_subkernel = transform(clbl.subkernel, *args, **kwargs)
                    clbl = clbl.copy(subkernel=new_subkernel)
                elif isinstance(clbl, ScalarCallable):
                    pass
                else:
                    raise NotImplementedError(f"{type(clbl)}")

                new_callables[func_id] = clbl

            return t_unit.copy(callables_table=new_callables)
        else:
            assert isinstance(t_unit_or_kernel, LoopKernel)
            kernel = t_unit_or_kernel
            return transform(kernel, *args, **kwargs)

    return wraps(transform)(_collective_transform)


def update_table(callables_table, clbl_id, clbl):
    from loopy.kernel.function_interface import InKernelCallable
    assert isinstance(clbl, InKernelCallable)

    for i, c in callables_table.items():
        if c == clbl:
            return i, callables_table

    while clbl_id in callables_table:
        clbl_id = next_indexed_function_id(clbl_id)

    callables_table[clbl_id] = clbl

    return clbl_id, callables_table

# }}}


# {{{ resolve_callables

def resolve_callables(program):
    """
    Returns a :class:`TranslationUnit` with known :class:`pymbolic.primitives.Call`
    expression nodes converted to :class:`loopy.symbolic.ResolvedFunction`.
    """
    from loopy.library.function import get_loopy_callables
    from loopy.kernel import KernelState

    if program.state >= KernelState.CALLS_RESOLVED:
        # program's callables have been resolved
        return program

    # get registered callables
    known_callables = dict(program.callables_table)
    # get target specific callables
    known_callables.update(program.target.get_device_ast_builder().known_callables)
    # get loopy specific callables
    known_callables.update(get_loopy_callables())

    callables_table = {}

    # callables: name of the calls seen in the program
    callables = set(program.entrypoints)

    while callables:
        clbl_name = callables.pop()
        clbl = known_callables[clbl_name]

        if isinstance(clbl, CallableKernel):
            knl = clbl.subkernel

            rule_mapping_context = SubstitutionRuleMappingContext(
                    knl.substitutions, knl.get_var_name_generator())
            clbl_resolver = CallableResolver(rule_mapping_context,
                                             frozenset(known_callables))
            knl = rule_mapping_context.finish_kernel(clbl_resolver.map_kernel(knl))
            knl = knl.copy(state=KernelState.CALLS_RESOLVED)

            # add the updated callable kernel to the table
            callables_table[clbl_name] = clbl.copy(subkernel=knl)

            # note the resolved callable for traversal
            callables.update(clbl_resolver.calls_resolved - set(callables_table))
        elif isinstance(clbl, ScalarCallable):
            # nothing to resolve within a scalar callable
            callables_table[clbl_name] = clbl
        else:
            raise NotImplementedError(f"{type(clbl)}")

    return program.copy(callables_table=callables_table)

# }}}


# vim: foldmethod=marker

from __future__ import annotations

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

import collections
from collections.abc import Set as abc_Set
from dataclasses import field, dataclass, replace
from typing import FrozenSet, Optional, TYPE_CHECKING, Mapping, Callable, Union, Any
from warnings import warn

from pymbolic.primitives import Variable
from functools import wraps

from loopy.symbolic import (RuleAwareIdentityMapper, ResolvedFunction,
                            SubstitutionRuleMappingContext)
from loopy.kernel.function_interface import (
        CallableKernel, InKernelCallable, ScalarCallable)
from loopy.diagnostic import LoopyError, DirectCallUncachedWarning
from loopy.library.reduction import ReductionOpFunction

from loopy.kernel import LoopKernel
from loopy.target import TargetBase
from pymbolic.primitives import Call
from immutables import Map

if TYPE_CHECKING:
    from loopy.target.execution import ExecutorBase


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
            # FIXME: We should have never used parse_tagged_name here.
            name, tag = parse_tagged_name(expr.function)

            if tag:
                raise LoopyError(f"tagged name in call: {expr.function}")

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

FunctionIdT = Union[str, ReductionOpFunction]


@dataclass(frozen=True)
class TranslationUnit:
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
        accessed if there is exactly one entrypoint in the translation unit.

    .. attribute:: callables_table

        An instance of :class:`pyrsistent.PMap` mapping the function
        identifiers in a kernel to their associated instances of
        :class:`~loopy.kernel.function_interface.InKernelCallable`.

    .. attribute:: target

        An instance of :class:`loopy.target.TargetBase`.

    .. attribute:: func_id_to_in_knl_callables_mappers

        A :class:`frozenset` of functions of the signature ``(target:
        TargetBase, function_indentifier: str)`` that returns an instance
        of :class:`loopy.kernel.function_interface.InKernelCallable` or *None*.

    .. automethod:: executor
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

    callables_table: Map[FunctionIdT, CallableKernel]
    target: TargetBase
    entrypoints: FrozenSet[str]

    def __post_init__(self):

        assert isinstance(self.entrypoints, abc_Set)
        assert isinstance(self.callables_table, Map)

        object.__setattr__(self, "_program_executor_cache", {})

    def copy(self, **kwargs):
        target = kwargs.pop("target", None)
        program = replace(self, **kwargs)
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

            program = replace(
                    self, callables_table=Map(new_callables), target=target)

        return program

    def with_entrypoints(self, entrypoints):
        """
        :param entrypoints: Either a comma-separated :class:`str` or
        :class:`frozenset`.
        """
        if isinstance(entrypoints, str):
            entrypoints = frozenset([e.strip() for e in
                entrypoints.split(",")])

        assert isinstance(entrypoints, abc_Set)

        return self.copy(entrypoints=entrypoints)

    @property
    def state(self):
        """ Returns an instance of :class:`loopy.kernel.KernelState`. """
        from loopy.kernel import KernelState
        return min((callable_knl.subkernel.state
                    for callable_knl in self.callables_table.values()
                    if isinstance(callable_knl, CallableKernel)),
                   default=KernelState.INITIAL)

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
            new_callables = self.callables_table.delete(kernel.name).set(
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
                             " The default entrypoint kernel is not uniquely"
                             " determined.")

    def executor(self,
                 *args, entrypoint: Optional[str] = None, **kwargs) -> ExecutorBase:
        """Return an object that hosts caches of compiled code for execution (i.e.
        a subclass of :class:`ExecutorBase`, specific to an execution
        environment (e.g. an OpenCL context) and a given entrypoint.

        :arg entrypoint: The name of the entrypoint callable to be called.
            Defaults to :attr:`default_entrypoint`.
            An error will result if multiple entrypoints exist and no
            entrypoint is specified.

        The variable arguments to this are target-specific. The
        :class:`PyOpenCLTarget` takes a :class:`~pyopencl.Context` or a
        :class:`~pyopencl.CommandQueue`.
        """
        if entrypoint is None:
            nentrypoints = len(self.entrypoints)
            if nentrypoints == 1:
                entrypoint, = self.entrypoints
            elif nentrypoints > 1:
                raise ValueError("TranslationUnit has multiple possible entrypoints."
                                 " The default entrypoint kernel is not uniquely"
                                 " determined. You may explicitly specify an "
                                 " entrypoint using the 'entrypoint' kwarg.")
            elif nentrypoints == 0:
                raise ValueError("TranslationUnit has no entrypoints, but"
                                 f" {len(self.callables_table)} callables."
                                 " Use TranslationUnit.with_entrypoints to"
                                 " set an entrypoint.")
            else:
                raise AssertionError
        else:
            if entrypoint not in self.entrypoints:
                raise LoopyError(f"'{entrypoint}' not in list of possible "
                        "entrypoints for the translation unit. "
                        "Maybe you want to invoke 'with_entrypoints' before "
                        "calling the translation unit?")

        return self.target.get_kernel_executor(self, *args,
                                               entrypoint=entrypoint, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Builds and calls the *entrypoint* kernel, if
        :attr:`TranslationUnit.target` is an executable target.

        :arg entrypoint: The name of the entrypoint callable to be called.
            Defaults to :attr:`default_entrypoint`.

        .. warning::

            While this was the main execution interface for loopy for many
            years (and reasonably efficient), the caches that made this so
            kept lots of expensive 'stuff' (such as OpenCL contexts) alive
            for no good reason, leading to major inefficiencies.
            See :meth:`executor` for an efficient, cached way to
            invoke kernels.
        """

        # The rationale for this is that the executor cache held long-lived
        # references to OpenCL contexts, and translation units were kept alive
        # long-term by caches, leading to many stale contexts being kept alive.
        # While attempts were made to turn those into weak references, this was
        # ultimately cumbersome and ineffective.
        #
        # In addition, the executor interface speeds up kernel invocation
        # by removing one unnecessary layer of function call.
        warn("TranslationUnit.__call__ will become uncached in 2024, "
             "meaning it will incur possibly substantial compilation cost "
             "with every invocation. Use TranslationUnit.executor to obtain "
             "an object that holds longer-lived caches.",
             DirectCallUncachedWarning, stacklevel=2)

        entrypoint = kwargs.get("entrypoint", None)

        if entrypoint is None:
            nentrypoints = len(self.entrypoints)
            if nentrypoints == 1:
                entrypoint, = self.entrypoints
            elif nentrypoints > 1:
                raise ValueError("TranslationUnit has multiple possible entrypoints."
                                 " The default entrypoint kernel is not uniquely"
                                 " determined. You may explicitly specify an "
                                 " entrypoint using the 'entrypoint' kwarg.")
            elif nentrypoints == 0:
                raise ValueError("TranslationUnit has no entrypoints, but"
                                 f" {len(self.callables_table)} callables."
                                 " Use TranslationUnit.with_entrypoints to"
                                 " set an entrypoint.")
            else:
                raise AssertionError
        else:
            if entrypoint not in self.entrypoints:
                raise LoopyError(f"'{entrypoint}' not in list of possible "
                        "entrypoints for the translation unit. "
                        "Maybe you want to invoke 'with_entrypoints' before "
                        "calling the translation unit?")

        kwargs["entrypoint"] = entrypoint

        key = self.target.get_kernel_executor_cache_key(*args, **kwargs)
        try:
            pex = self._program_executor_cache[key]  # pylint: disable=no-member
        except KeyError:
            pex = self.target.get_kernel_executor(self, *args, **kwargs)
            self._program_executor_cache[key] = pex  # pylint: disable=no-member

        del kwargs["entrypoint"]

        return pex(*args, **kwargs)

    def __str__(self):
        # FIXME: do a topological sort by the call graph

        return "\n".join(
                str(clbl.subkernel)
                for name, clbl in self.callables_table.items()
                if isinstance(clbl, CallableKernel))

    # FIXME: Delete these when _program_executor_cache leaves the building
    def __getstate__(self):
        from dataclasses import asdict
        return asdict(self)

    def __setstate__(self, state_obj):
        for k, v in state_obj.items():
            object.__setattr__(self, k, v)

        object.__setattr__(self, "_program_executor_cache", {})

    # FIXME: This is here because Firedrake expects it, for some legacy reason.
    # Without that, it would be safe to delete.
    def update_persistent_hash(self, key_hash, key_builder):
        key_builder.update_for_dataclass(key_hash, self)

# }}}


# {{{ rename resolved functions

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


def get_reachable_resolved_callable_ids(callables, entrypoints):
    """
    Returns a :class:`frozenset` of callables ids that are resolved and
    reachable from *entrypoints*.
    """
    return frozenset().union(*(callables[e].get_called_callables(callables)
                               for e in entrypoints))


# {{{ CallablesInferenceContext

def get_all_subst_names(callables):
    """
    Returns a :class:`set` of all substitution rule names in the callable
    kernels of *callables*.

    :arg callables: A mapping from function identifiers to
        :class:`~loopy.kernel.function_interface.InKernelCallable`.
    """
    return set().union(*(set(clbl.subkernel.substitutions.keys())
                         for clbl in callables.values()
                         if isinstance(clbl, CallableKernel)))


def make_callable_name_generator(callables):
    from pytools import UniqueNameGenerator
    all_substs = get_all_subst_names(callables)
    return UniqueNameGenerator(set(callables.keys()) | all_substs)


def make_clbl_inf_ctx(callables, entrypoints):
    name_gen = make_callable_name_generator(callables)
    return CallablesInferenceContext(callables, name_gen)


@dataclass(frozen=True)
class CallablesInferenceContext:
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
    callables: Mapping[str, InKernelCallable]
    clbl_name_gen: Callable[[str], str]
    renames: Mapping[str, FrozenSet[str]] = field(
            default_factory=lambda: collections.defaultdict(frozenset))
    new_entrypoints: FrozenSet[str] = frozenset()

    def copy(self, **kwargs: Any) -> CallablesInferenceContext:
        return replace(self, **kwargs)

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

        renames = collections.defaultdict(frozenset, self.renames)

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

        # must allocate a new clbl in the namespace => find a unique id for it
        unique_function_id = self.clbl_name_gen(old_function_id)

        updated_callables = dict(self.callables)
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
        new_callable_ids = get_reachable_resolved_callable_ids(self.callables,
                                                               self.new_entrypoints)

        # get the history of function ids from the performed renames:
        history = {}
        for old_func_id, new_func_ids in self.renames.items():
            for new_func_id in new_func_ids:
                if new_func_id in (new_callable_ids | self.new_entrypoints):
                    history[new_func_id] = old_func_id

        # }}}

        # {{{ preserve the entrypoints of *program*

        # If there are any callees having old entrypoint names => mark them for
        # renaming
        callees_with_old_entrypoint_names = ((program.entrypoints & new_callable_ids)
                                             - self.new_entrypoints)

        todo_renames = {}
        new_callables = dict(program.callables_table)

        for c in callees_with_old_entrypoint_names:
            todo_renames[c] = self.clbl_name_gen(c)

        for e in self.new_entrypoints:
            # note renames to "rollback" the renaming of entrypoints
            todo_renames[e] = history[e]
            assert todo_renames[e] in program.entrypoints

        # }}}

        # try to roll back the names as much as possible
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

        return program.copy(callables_table=Map(new_callables))

    def __getitem__(self, name):
        result = self.callables[name]
        return result

# }}}


# {{{ helper functions

def make_program(kernel: LoopKernel) -> TranslationUnit:
    """
    Returns an instance of :class:`loopy.TranslationUnit` with *kernel* as the only
    callable kernel.
    """

    return TranslationUnit(
            callables_table=Map({
                kernel.name: CallableKernel(kernel)}),
            target=kernel.target,
            entrypoints=frozenset())


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

            return t_unit.copy(callables_table=Map(new_callables))
        else:
            assert isinstance(t_unit_or_kernel, LoopKernel)
            kernel = t_unit_or_kernel
            return transform(kernel, *args, **kwargs)

    return wraps(transform)(_collective_transform)


def add_callable_to_table(callables_table, clbl_id, clbl):
    """
    Returns a tuple ``new_clbl_id, new_callables_table`` where
    *new_callables_table* is a copy of *callables_table* with *clbl* in its
    namespace. *clbl* is referred to in *new_callables_table*'s namespace by
    *new_clbl_id*.

    :arg clbl_id: An instance of :class:`str` or
        :class:`~loopy.library.reduction.ReductionOpFunction` based on which
        the unique identifier, *new_clbl_id* , is to be chosen.
    """
    from loopy.kernel.function_interface import InKernelCallable
    assert isinstance(clbl, InKernelCallable)

    for i, c in callables_table.items():
        if c == clbl:
            return i, callables_table

    if isinstance(clbl_id, ReductionOpFunction):
        new_clbl_id = clbl_id
    else:
        assert isinstance(clbl_id, str)
        ung = make_callable_name_generator(callables_table)
        new_clbl_id = ung(clbl_id)

    new_callables_table = callables_table.copy()
    new_callables_table[new_clbl_id] = clbl.with_name(new_clbl_id)

    return new_clbl_id, new_callables_table

# }}}


# {{{ resolve_callables

def resolve_callables(program):
    """
    Returns a :class:`TranslationUnit` with known :class:`pymbolic.primitives.Call`
    expression nodes converted to :class:`loopy.symbolic.ResolvedFunction`.
    """
    from loopy.library.function import get_loopy_callables
    from loopy.check import validate_kernel_call_sites
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
    callables = {name for name, clbl in program.callables_table.items()
                 if isinstance(clbl, CallableKernel)}

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

    program = program.copy(callables_table=Map(callables_table))

    validate_kernel_call_sites(program)

    return program

# }}}


# vim: foldmethod=marker

.. currentmodule:: loopy


.. _func-interface:

Function Interface
==================


Resolving and specialization
----------------------------

In :mod:`loopy`, a :class:`loopy.TranslationUnit` is a collection of callables
and entrypoints. Callable are of type
:class`:loopy.kernel.function_interface.InKernelCallable`. Any expression node
which has a callable corresponding to it appears as
:class:`~loopy.symbolic.ResolvedFunction`. The process of realizing a function as
a :class:`~loopy.kernel.function_interface.InKernelCallable` is referred to as resolving.


During code-generation process for a :class:`~loopy.TranslationUnit`, a callable
is *specialized* depending on the types and shapes of the arguments passed at a
call site. For example, a call to ``sin(x)`` in :mod:`loopy` is type-generic to
begin with, but it later specialized to either ``sinf``, ``sin`` or ``sinl``
depending on the type of its argument ``x``. A callable's behavior during type
or shape specialization is encoded via
:meth:`~loopy.kernel.function_interface.InKernelCallable.with_types` and
:meth:`~loopy.kernel.function_interface.InKernelCallable.with_descrs`.


Registering callables
---------------------

A user can *register* callables within a  :class:`~loopy.TranslationUnit` to
allow loopy to resolve calls not pre-defined in :mod:`loopy`. In :mod:`loopy`,
we typically aim to expose all the standard math functions defined for
a :class:`~loopy.target.TargetBase`. Other foreign functions could be invoked by
*registering* them.

An example demonstrating registering a CBlasGemv as a loopy callable:

.. literalinclude:: ../examples/python/call-external.py


Call Instruction for a kernel call
----------------------------------

At a call-site involving a call to a :class:`loopy.LoopKernel`, the arguments to
the call must be ordered by the order of input arguments of the callee kernel.
Similarly, the assignees must be ordered by the order of callee kernel's output
arguments. Since a :class:`~loopy.kernel.data.KernelArgument` can be both an
input and an output, such arguments would be a part of the call instruction's
assignees as well as the call expression node's parameters.


Reference
---------

.. automodule:: loopy.kernel.function_interface

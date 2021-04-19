.. currentmodule:: loopy


.. _func-interface:

Function Interface
==================


Resolving and specialization
----------------------------

In :mod:`loopy`, a :class:`loopy.Program` is a collection of callables
and entrypoints. Callable are of type
:class`:loopy.kernel.function_interface.InKernelCallable`. Any expression node
which has a callable corresponding to it appears as
:class:`~loopy.symbolic.ResolvedFunction`. The process of realizing a function as
a :class:`~loopy.kernel.function_interface.InKernelCallable` is referred to as resolving.


During code-generation process for a :class:`~loopy.Program`, a callable
is *specialized* depending on the types and shapes of the arguments passed at a
call site. For example, a call to ``sin(x)`` in :mod:`loopy` is type-generic to
begin with, but it later specialized to either ``sinf``, ``sin`` or ``sinl``
depending on the type of its argument ``x``. A callable's behavior during type
or shape specialization is encoded via
:meth:`~loopy.kernel.function_interface.InKernelCallable.with_types` and
:meth:`~loopy.kernel.function_interface.InKernelCallable.with_descrs`.


Registering callables
---------------------

A user can *register* callables within a  :class:`~loopy.Program` to
allow loopy to resolve calls not pre-defined in :mod:`loopy`. In :mod:`loopy`,
we typically aim to expose all the standard math functions defined for
a :class:`~loopy.target.TargetBase`. Other foreign functions could be invoked by
*registering* them.

An example demonstrating registering a CBlasGemv as a loopy callable:

.. literalinclude:: ../examples/python/call-external.py

Reference
---------

.. automodule:: loopy.kernel.function_interface

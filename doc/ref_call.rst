Calling Loopy Kernels and External Functions
============================================

Goals of a function interface
-----------------------------

- Must be able to have complete information of the function just through the
  epxression node.
- Must adhere to :mod:`loopy` semantics of immutability.
- Must have a class instance linked with the expression node which would record
  the properties of the function.
- Must indicate in the expression if the function is known to the kernel. (This
  is intended to be done by making the function expression node an instance of
  ``ResolvedFunction`` as soon as the function definition is resolved by the
  kernel)
- Function overloading is not encouraged in :mod:`loopy` as it gives rise to
  contention while debugging with the help of the kernel intermediate
  representation and hence if the expression nodes point to different function
  instances they must differ in their representation. For example: ``float
  sin(float )`` and ``double sin(double )`` should diverge by having different
  identifiers as soon as data type of the argument is inferred.
- Must have an interface to register external functions.


Scoped Function and resolving
-----------------------------

``ResolvedFunctions`` are pymbolic nodes within expressions in a ``Loo.py``
kernel, whose name has been resolved by the kernel. The process of matching a
function idenitifier with the function definition is called "resolving".

A pymbolic ``Call`` node can be converted to a ``ResolvedFunction`` if it
is "resolved" by one of the ``function_scoper`` in a
:attr:`LoopKernel.scoped_functions`

-  Functions already registered by the target. Some examples include --
   ``sin()``, ``cos()``, ``exp()``, ``max()`` (for C-Targets.)
-  Functions that are defined in ``Loo.py`` and are realized into
   different set of instructions during code generation. Some examples
   include ``make_tuple``, ``ArgExtOp``, ``index_of``, ...
-  Functions registered as ``CallableKernels`` using
   ``lp.register_callable_kernel(...)``.
-  Functions that have been provided through
   ``lp.register_function_scoper(...)``
-  Functions that can be made known from the user through
   ``lp.register_function_mangler``. This is planned to be deprecated,
   as its functionality is superseded by
   ``lp.register_function_scoper(...)``.

Expressions after a function is scoped
--------------------------------------

Consider the following expression.

::

    sin(a[i]) + unknown_func(b[i]) + callable_knl_func(c[i])*mangler_call(d[i])

During the kernel creation phase, the kernel would know that ``sin`` is
a function known to the target and hence it should be scoped. And as
expected, after ``make_kernel`` has been called the above expression
would get converted to:

::

    ResolvedFunction(Variable('sin'))(a[i]) + unknown_func(b[i]) +
    callable_knl_func(c[i])*mangler_call(d[i])

This would also make an entry in the kernel's ``scoped_functions``
dictionary as:

::

    {Variable('sin'): ScalarCallable(name='sin', arg_id_to_dtype=None,
    arg_id_to_descr=None, name_in_target=None)}

It might be noteworthy that at this step, it only scopes functions
through their names without any information about the types of the
function.

Once, the user calls the transformation:
``lp.register_callable_kernel(knl, 'callable_knl_func', callee_knl)``,
the expression gets converted to:

::

    ResolvedFunction(Variable('sin'))(a[i]) + unknown_func(b[i]) +
    ResolvedFunction('callable_knl_func')(c[i])*mangler_call(d[i])

This also makes an entry in the ``scoped_functions`` dictionary as --

::

    {Variable('sin'): ScalarCallable(name='sin', arg_id_to_dtype=None,
    arg_id_to_descr=None, name_in_target=None),
    Variable('callable_knl_func'): CallableKernel(subkernel=LoopKernel(...),
    arg_id_to_dtype=None, arg_id_to_descr=None, name_in_target=None)}

Now, if the user calls
``register_function_mangler(knl, 'mangler_call')``, one might expect
that the mangler call function should get scoped, but that does **not**
happen, because the "old" ``function_manglers``, would return a match
only if all the parameters of the function match viz. name, argument
arity and argument types. Hence, the ``scoped_functions`` dictionary
would remain unchanged.

``ResolvedFunctions`` and specializations
---------------------------------------

Consider the same ``ResolvedFunction('sin')`` as above. This function
although scoped does not the know the types i.e. it does yet know that
for a ``C-Target``, whether it should emit ``sin`` or ``sinf`` or
``sinl``. Hence, right now the function can be called as a
"type-generic" function as further in the pipeline it can take any one
of the above definitions. The functions go through a "specialization"
processes at various points in the pipeline, where the attributes of the
callables are resolved.

-  During type inference, the functions go though type specialization
   where in the ``arg_id_to_dtype`` of the functions is realized.
-  During descriptor inference, the functions goes through a description
   specialization where the ``arg_id_to_descr`` is populated. The
   ``arg_id_to_descr`` contains important information regarding shape,
   strides and scope of the arguments which form an important part of
   ``CallableKernel`` as this information would be helpful to to
   generate the function signature and make changes to the data access
   pattern of the variables in the callee kernel.
-  Whenever a ``ResolvedFunction`` goes through a specialization, this is
   indicated by changing the name in the ``pymbolic`` node.

If during type inference, it is inferred that the type of ``a[i]`` is
``np.float32``. The new ``pymbolic`` node would be:

::

    ResolvedFunction('sin_0')(a[i]) + ...

This name change is done so that it indicates that the node points to a
different ``ScalarCallable`` in the dictionary. And hence a new entry is
added to the ``scoped_functions`` dictionary as:

::

    {'sin': ScalarCallable(name='sin', arg_id_to_dtype=None,
    arg_id_to_descr=None, name_in_target=None),
    Variable('callable_knl_func'): CallableKernel(subkernel=LoopKernel(...),
    arg_id_to_dtype=None, arg_id_to_descr=None, name_in_target=None),
    'sin_0': ScalarCallable(name='sin', arg_id_to_dtype={0:np.float32,
    -1: np.float32}, arg_id_to_descr=None, name_in_target='sinf')}

Description Inference
---------------------

Although this step has no significance for a ``ScalarCallable``, it
forms a very important part of ``CallableKernel``. In which the
``dim_tags``, ``shape`` and ``address_space`` of the arguments of the
callable kernel is altered.

-  The ``dim_tags`` attribute helps to ensure that the memory layout
   between the caller and the callee kernel is coherent.
-  The ``address_space`` attribute ensures that, while writing the device
   code we emit the appropriate scope qualifiers for the function
   declaration arguments.
-  The ``shape`` attribute helps in:

   -  Storage allocation.
   -  Memory layout.
   -  Out of bounds accesses to be caught in ``Loo.py``.

Hence, in the ``Loo.py`` pipeline, one might expect the following
developments of the ``sin`` pymbolic call expression node.

::

    sin -> (Kernel creation) -> ResolvedFunction(Variable('sin')) ->
    (Type Inference) -> ResolvedFunction(Variable('sin_0')) ->
    (Descriptor Inference) -> ResolvedFunction(Variable('sin_1'))

Changes on the target side to accommodate the new function interface
--------------------------------------------------------------------

The earlier "function\_mangler" as a member method of the class
``lp.ASTBuilderBase`` will be replaced by ``function_scopers``. The
function scopers would return a list of functions with the signature
``(target, identifier)->lp.InKernelCallable``.

An example: Calling BLAS
------------------------

.. literalinclude:: ../examples/python/external-call.py


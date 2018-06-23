ScopedFunctions
===============

``ScopedFunctions`` are pymbolic nodes within expressions in a
``Loo.py`` kernel, whose name has been resolved by the kernel.

A pymbolic ``Call`` node can be converted to a ``ScopedFunction`` if it
is resolved by one of the ``function_scoper`` in a :attr:`LoopKernel.scoped_functions`

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

Expressions after a function is scoped.
---------------------------------------

Consider the following expression.

::

    sin(a[i]) + unknown_func(b[i]) + callable_knl_func(c[i])*mangler_call(d[i])

During the kernel creation phase, the kernel would know that ``sin`` is
a function known to the target and hence it should be scoped. And as
expected, after ``make_kernel`` has been called the above expression
would get converted to:

::

    ScopedFunction(Variable('sin'))(a[i]) + unknown_func(b[i]) +
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

    ScopedFunction(Variable('sin'))(a[i]) + unknown_func(b[i]) +
    ScopedFunction('callable_knl_func')(c[i])*mangler_call(d[i])

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

``ScopedFunctions`` and specializations
---------------------------------------

Consider the same ``ScopedFunction('sin')`` as above. This function
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
-  Whenever a ``ScopedFunction`` goes through a specialization, this is
   indicated by changing the name in the ``pymbolic`` node.

If during type inference, it is inferred that the type of ``a[i]`` is
``np.float32``. The new ``pymbolic`` node would be:

::

    ScopedFunction('sin_0')(a[i]) + ...

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
``dim_tags``, ``shape`` and ``mem_scope`` of the arguments of the
callable kernel is altered.

-  The ``dim_tags`` attribute helps to ensure that the memory layout
   between the caller and the callee kernel is coherent.
-  The ``mem_scope`` attribute ensures that, while writing the device
   code we emit the appropriate scope qualifiers for the function
   declaration arguments.
-  The ``shape`` attribute helps in:

   -  Storage allocation.
   -  Memory layout.
   -  Out of bounds accesses to be caught in ``Loo.py``.

Hence, in the ``Loo.py`` pipeline, one might expect the following
developments of the ``sin`` pymbolic call expression node.

::

    sin -> (Kernel creation) -> ScopedFunction(Variable('sin')) ->
    (Type Inference) -> ScopedFunction(Variable('sin_0')) ->
    (Descriptor Inference) -> ScopedFunction(Variable('sin_1'))

Changes on the target side to accommodate the new function interface.
---------------------------------------------------------------------

The earlier "function\_mangler" as a member method of the class
``lp.ASTBuilderBase`` will be replaced by ``function_scopers``. The
function scopers would return a list of functions with the signature
``(target, identifier)->lp.InKernelCallable``.

An example of registering Vector callables is shown below.
----------------------------------------------------------

.. code:: python

    import loopy as lp
    import numpy as np
    from loopy.diagnostic import LoopyError
    from loopy.target.c import CTarget


    # {{{ blas callable

    class BLASCallable(lp.ScalarCallable):
        def with_types(self, arg_id_to_dtype, kernel):
            for i in range(0, 2):
                if i not in arg_id_to_dtype or arg_id_to_dtype[i] is None:
                    # the types provided aren't mature enough to specialize the
                    # callable
                    return self.copy(arg_id_to_dtype=arg_id_to_dtype)

            mat_dtype = arg_id_to_dtype[0].numpy_dtype
            vec_dtype = arg_id_to_dtype[1].numpy_dtype

            if mat_dtype != vec_dtype:
                raise LoopyError("DGEMV should have same dtype for matrix and "
                        "vector")

            if vec_dtype == np.float32:
                name_in_target = "cblas_sgemv"
            elif vec_dtype == np.float64:
                name_in_target = "cblas_dgemv"
            else:
                raise LoopyError("GEMV only supported for float32 and float64 "
                        "types")

            from loopy.types import NumpyType
            return self.copy(name_in_target=name_in_target,
                    arg_id_to_dtype={0: NumpyType(vec_dtype), 1: NumpyType(vec_dtype),
                        -1: NumpyType(vec_dtype)})

        def emit_call_insn(self, insn, target, expression_to_code_mapper):
            assert self.is_ready_for_codegen()

            from loopy.kernel.instruction import CallInstruction

            assert isinstance(insn, CallInstruction)

            parameters = insn.expression.parameters

            parameters = list(parameters)
            par_dtypes = [self.arg_id_to_dtype[i] for i, _ in enumerate(parameters)]

            parameters.append(insn.assignees[0])
            par_dtypes.append(self.arg_id_to_dtype[-1])

            # no type casting in array calls.
            from loopy.expression import dtype_to_type_context
            from pymbolic.mapper.stringifier import PREC_NONE
            from loopy.symbolic import SubArrayRef
            from pymbolic import var

            mat_descr = self.arg_id_to_descr[0]

            c_parameters = [
                    expression_to_code_mapper(par, PREC_NONE,
                        dtype_to_type_context(target, par_dtype),
                        par_dtype).expr if isinstance(par, SubArrayRef) else
                    expression_to_code_mapper(par, PREC_NONE,
                        dtype_to_type_context(target, par_dtype),
                        par_dtype).expr
                    for par, par_dtype in zip(
                        parameters, par_dtypes)]
            c_parameters.insert(0, var('CblasRowMajor'))
            c_parameters.insert(1, var('CblasNoTrans'))
            c_parameters.insert(2, mat_descr.shape[0])
            c_parameters.insert(3, mat_descr.shape[1])
            c_parameters.insert(4, 1)
            c_parameters.insert(6, 1)
            c_parameters.insert(8, 1)
            c_parameters.insert(10, 1)
            return var(self.name_in_target)(*c_parameters), False

        def generate_preambles(self, target):
            assert isinstance(target, CTarget)
            yield("99_cblas", "#include <cblas.h>")
            return


    def blas_fn_lookup(target, identifier):
        if identifier == 'gemv':
            return BLASCallable(name='gemv')
        return None

    # }}}


    n = 10

    knl = lp.make_kernel(
            "{[i]: 0<=i<10}",
            """
            y[:] = gemv(A[:, :], x[:])
            """, [
                lp.ArrayArg('A', dtype=np.float64, shape=(n, n)),
                lp.ArrayArg('x', dtype=np.float64, shape=(n, )),
                lp.ArrayArg('y', shape=(n, )), ...],
            target=CTarget())
    knl = lp.register_function_lookup(knl, blas_fn_lookup)


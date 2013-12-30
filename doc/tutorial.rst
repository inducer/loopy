.. _tutorial:

Tutorial
========

This guide provides a gentle introduction into what loopy is, how it works, and
what it can do. There's also the :ref:`reference` that clearly defines all
aspects of loopy.

Preparation
-----------

For now, :mod:`loopy` depends on :mod:`pyopencl`. We import a few modules
and initialize :mod:`pyopencl`

.. doctest::
    :options: +ELLIPSIS

    >>> import numpy as np
    >>> import pyopencl as cl
    >>> import pyopencl.array
    >>> import pyopencl.clrandom
    >>> import loopy as lp

    >>> ctx = cl.create_some_context(interactive=False)
    >>> queue = cl.CommandQueue(ctx)

We also create some data on the device that we'll use throughout our
examples:

.. doctest::

    >>> n = 1000
    >>> x_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
    >>> y_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
    >>> a_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
    >>> b_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)

And some data on the host:

.. doctest::

    >>> x_vec_host = np.random.randn(n).astype(np.float32)
    >>> y_vec_host = np.random.randn(n).astype(np.float32)

Getting started
---------------

We'll start by taking a closer look at a very simple kernel that reads in
one vector, doubles it, and writes it to another.

.. doctest::

    >>> knl = lp.make_kernel(ctx.devices[0],
    ...     "{ [i]: 0<=i<n }",
    ...     "out[i] = 2*a[i]")

The parts that you see here are the two main components of a loopy kernel:

* The **loop domain**: ``{ [i]: 0<=i<n }``. This tells loopy the values that
  you would like your loop variables to assume. It is written in
  :ref:`isl-syntax`. Loopy calls the loop variables **inames**.  These are
  the identifiers that occur in between the brackets at the beginning of
  the loop domain.

  Note that *n* is not an iname in the example. It is a parameter that is
  passed to the kernel by the user that, in this case, determines the
  length of the vector being multiplied.

* The **instructions** to be executed. These are generally scalar
  assignments between array elements, consisting of a left hand
  side and a right hand side. See :ref:`assignments` for the
  full syntax of an assignment.

  Reductions are allowed, too, and are given as, for example::

    sum(k, a[i,k]*b[k,j])

  See :ref:`expression-syntax` for a full list of allowed constructs in the
  left- and right-hand side expression of an assignment.

Loopy also needs to know which OpenCL device to target.  ``ctx.devices[0]``
specifies the first device in our OpenCL context.

As you create and transform kernels, it's useful to know that you can
always see loopy's view of a kernel by printing it.

.. doctest::

    >>> print knl
    ---------------------------------------------------------------------------
    KERNEL: loopy_kernel
    ---------------------------------------------------------------------------
    ARGUMENTS:
    a: GlobalArg, type: <runtime>, shape: (n), dim_tags: (stride:1)
    n: ValueArg, type: <runtime>
    out: GlobalArg, type: <runtime>, shape: (n), dim_tags: (stride:1)
    ---------------------------------------------------------------------------
    DOMAINS:
    [n] -> { [i] : i >= 0 and i <= -1 + n }
    ---------------------------------------------------------------------------
    INAME-TO-TAG MAP:
    i: None
    ---------------------------------------------------------------------------
    INSTRUCTIONS:
    [i]                                  out[i] <- 2*a[i]   # insn
    ---------------------------------------------------------------------------

You'll likely have noticed that there's quite a bit more information here
than there was in the input. Most of this comes from default values that
loopy assumes to cover common use cases. These defaults can all be
overridden.

We've seen the domain and the instructions above, and we'll discuss the
'iname-to-tag-map' in :ref:`implementing-inames`. The remaining big chunk
of added information is in the 'arguments' section, where we observe the
following:

* ``a`` and ``out`` have been classified as pass-by-reference (i.e.
  pointer) arguments in global device memory. Any referenced array variable
  will default to global unless otherwise specified.

* Loopy has also examined our access to ``a`` and ``out`` and determined
  the bounds of the array from the values we are accessing. This is shown
  after **shape:**. Like :mod:`numpy`, loopy works on multi-dimensional
  arrays. Loopy's idea of arrays is very similar to that of :mod:`numpy`,
  including the *shape* attribute.

  Sometimes, loopy will be unable to make this determination. It will tell
  you so--for example when the array indices consist of data read from
  memory.  Other times, arrays are larger than the accessed footprint. In
  either case, you will want to specify the kernel arguments explicitly.
  See :ref:`specifying-arguments`.

* Loopy has not determined the type of ``a`` and ``out``. The data type is
  given as ``<runtime>``, which means that these types will be determined
  by the data passed in when the kernel is invoked. Loopy generates (and
  caches!) a copy of the kernel for each combination of types passed in.

* In addition, each array axis has a 'dimension tag'. This is shown above
  as ``(stride:1)``. We will see more on this in
  :ref:`implementing-array-axes`.

Running a kernel
----------------

Running the kernel that we've just created is easy. Let's check the result
for good measure.

.. doctest::

    >>> evt, (out,) = knl(queue, a=x_vec_dev)
    >>> assert (out.get() == (2*x_vec_dev).get()).all()

We can have loopy print the OpenCL kernel it generated
by passing :attr:`loopy.Flags.write_cl`.

.. doctest::

    >>> evt, (out,) = knl(queue, a=x_vec_dev, flags="write_cl")
    <BLANKLINE>
    #define lid(N) ((int) get_local_id(N))
    #define gid(N) ((int) get_group_id(N))
    <BLANKLINE>
    __kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global float const *restrict a, int const n, __global float *restrict out)
    {
    <BLANKLINE>
      for (int i = 0; i <= (-1 + n); ++i)
        out[i] = 2.0f * a[i];
    }


As promised, loopy has used the type of *x_vec_dev* to specialize the
kernel. If a variable is written as part of the kernel code, loopy will
automatically return it in the second element of the result of a kernel
call (the first being the :class:`pyopencl.Event` associated with the
execution of the kernel). (If the ordering of the output tuple is not
clear, it can be specified or turned into a :class:`dict`. See the
*kernel_data* argument of :func:`loopy.make_kernel` and
:attr:`loopy.Flags.return_dict`.)

For convenience, loopy kernels also directly accept :mod:`numpy` arrays:

.. doctest::

    >>> evt, (out,) = knl(queue, a=x_vec_host)
    >>> assert (out == (2*x_vec_host)).all()

Notice how both *out* nor *a* are :mod:`numpy` arrays, but neither needed
to be transferred to or from the device.  Checking for numpy arrays and
transferring them if needed comes at a potential performance cost.  If you
would like to make sure that you avoid this cost, pass
:attr:`loopy.Flags.no_numpy`.

Further notice how *n*, while technically being an argument, did not need
to be passed, as loopy is able to find *n* from the shape of the input
argument *a*.

For efficiency, loopy generates Python code that handles kernel invocation.
If you are suspecting that this code is causing you an issue, you can
inspect that code, too, using :attr:`loopy.Flags.write_wrapper`:

.. doctest::
    :options: +ELLIPSIS

    >>> evt, (out,) = knl(queue, a=x_vec_host, flags="write_wrapper")
    from __future__ import division
    ...
    def invoke_loopy_kernel_loopy_kernel(cl_kernel, queue, allocator=None, wait_for=None, out_host=None, a=None, n=None, out=None):
        if allocator is None:
            allocator = _lpy_cl_tools.DeferredAllocator(queue.context)
    <BLANKLINE>
        # {{{ find integer arguments from shapes
    <BLANKLINE>
        if n is None:
            if a is not None:
                n = a.shape[0]
            elif out is not None:
                n = out.shape[0]
    <BLANKLINE>
        # }}}
    ...

Generating code
~~~~~~~~~~~~~~~

Instead of using loopy to run the code it generates, you can also just use
loopy as a code generator and take care of executing the generated kernels
yourself. In this case, make sure loopy knows about all types, and then
call :func:`loopy.generate_code`:

.. doctest::

    >>> typed_knl = lp.add_dtypes(knl, dict(a=np.float32))
    >>> code, _ = lp.generate_code(typed_knl)
    >>> print code
    <BLANKLINE>
    #define lid(N) ((int) get_local_id(N))
    #define gid(N) ((int) get_group_id(N))
    <BLANKLINE>
    __kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global float const *restrict a, int const n, __global float *restrict out)
    {
    <BLANKLINE>
      for (int i = 0; i <= (-1 + n); ++i)
        out[i] = 2.0f * a[i];
    }

Ordering
--------

Next, we'll change our kernel a bit. Our goal will be to transpose a matrix
and double its entries, and we will do this in two steps for the sake of
argument:

.. doctest::

    >>> # WARNING: Incorrect.
    >>> knl = lp.make_kernel(ctx.devices[0],
    ...     "{ [i,j]: 0<=i,j<n }",
    ...     """
    ...     out[j,i] = a[i,j]
    ...     out[i,j] = 2*out[i,j]
    ...     """)

loopy's programming model is completely *unordered* by default. This means
that:

* There is no guarantee about the order in which the loop domain is
  traversed. ``i==3`` could be reached before ``i==0`` but still after
  ``i==0``. Your program is only correct if it produces a valid result
  irrespective of this ordering.

* In addition, there is (by default) no ordering between instructions
  either. In other words, loopy is free to execute the instructions above
  in any order whatsoever.

Reading the above two rules, you'll notice that our transpose-and-multiply
kernel is incorrect, because it only computes the desired result if the
first instruction completes before the second one. To fix this, we declare
an explicit dependency:

.. doctest::

    >>> # WARNING: Incorrect.
    >>> knl = lp.make_kernel(ctx.devices[0],
    ...     "{ [i,j]: 0<=i,j<n }",
    ...     """
    ...     out[j,i] = a[i,j] {id=transpose}
    ...     out[i,j] = 2*out[i,j]  {dep=transpose}
    ...     """)

``{id=transpose}`` assigns the identifier *transpose* to the first
instruction, and ``{dep=transpose}`` declares a dependency of the second
instruction on the first. Looking at loopy's view of this kernel, we see
that these dependencies show up there, too:

.. doctest::
    :options: +ELLIPSIS

    >>> print knl
    ---------------------------------------------------------------------------
    KERNEL: loopy_kernel
    ---------------------------------------------------------------------------
    ...
    ---------------------------------------------------------------------------
    DEPENDENCIES: (use loopy.show_dependency_graph to visualize)
    insn : transpose
    ---------------------------------------------------------------------------

These dependencies are in a ``dependent : prerequisite`` format that should
be familiar if you have previously dealt with Makefiles. For larger
kernels, these dependency lists can become quite verbose, and there is an
increasing risk that required dependencies are missed. To help catch these,
loopy can also show an instruction dependency graph, using
:func:`loopy.show_dependency_graph`:

.. image:: images/dep-graph-incorrect.svg

Dependencies are shown as arrows from prerequisite to dependent in the
graph.  This functionality requires the open-source `graphviz
<http://graphviz.org>`_ graph drawing tools to be installed. The generated
graph will open in a browser window.

Since manually notating lots of dependencies is cumbersome, loopy has
a heuristic:

    If a variable is written by exactly one instruction, then all
    instructions reading that variable will automatically depend on the
    writing instruction.

The intention of this heuristic is to cover the common case of a
precomputed result being stored and used many times. Generally, these
dependencies are *in addition* to any manual dependencies added via
``{dep=...}``.  It is possible (but rare) that the heuristic adds undesired
dependencies.  In this case, ``{dep=*...}`` (i.e. a leading asterisk) to
prevent the heuristic from adding dependencies for this instruction.

Next, it is important to understand how loops and dependencies interact.
Let us take a look at the generated code for the above kernel:

.. doctest::

    >>> evt, (out,) = knl(queue, a=a_mat_dev, flags="write_cl")
    <BLANKLINE>
    #define lid(N) ((int) get_local_id(N))
    #define gid(N) ((int) get_group_id(N))
    <BLANKLINE>
    __kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global float const *restrict a, int const n, __global float *restrict out)
    {
    <BLANKLINE>
      for (int i = 0; i <= (-1 + n); ++i)
        for (int j = 0; j <= (-1 + n); ++j)
        {
          out[n * j + i] = a[n * i + j];
          out[n * i + j] = 2.0f * out[n * i + j];
        }
    }

While our requested instruction ordering has been obeyed, something is
still not right:

.. doctest::
    :options: +ELLIPSIS

    >>> assert (out.get() == a_mat_dev.get().T*2).all()
    Traceback (most recent call last):
    ...
    AssertionError

For the kernel to perform the desired computation, *all
instances* (loop iterations) of the first instruction need to be completed,
not just the one for the current values of *(i, j)*.

    Dependencies in loopy act *within* the largest common set of shared
    inames.

As a result, our example above realizes the dependency *within* the *i* and *j*
loops. To fix our example, we simply create a new pair of loops *ii* and *jj*
with identical bounds, for the use of the transpose:

.. doctest::

    >>> knl = lp.make_kernel(ctx.devices[0],
    ...     "{ [i,j,ii,jj]: 0<=i,j,ii,jj<n }",
    ...     """
    ...     out[j,i] = a[i,j] {id=transpose}
    ...     out[ii,jj] = 2*out[ii,jj]  {dep=transpose}
    ...     """)

:func:`loopy.duplicate_inames` can be used to achieve the same goal.
Now the intended code is generated and our test passes.

.. doctest::

    >>> evt, (out,) = knl(queue, a=a_mat_dev, flags="write_cl")
    <BLANKLINE>
    #define lid(N) ((int) get_local_id(N))
    #define gid(N) ((int) get_group_id(N))
    <BLANKLINE>
    __kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global float const *restrict a, int const n, __global float *restrict out)
    {
    <BLANKLINE>
      for (int i = 0; i <= (-1 + n); ++i)
        for (int j = 0; j <= (-1 + n); ++j)
          out[n * j + i] = a[n * i + j];
      for (int ii = 0; ii <= (-1 + n); ++ii)
        for (int jj = 0; jj <= (-1 + n); ++jj)
          out[n * ii + jj] = 2.0f * out[n * ii + jj];
    }
    >>> assert (out.get() == a_mat_dev.get().T*2).all()

Also notice how the changed loop structure is reflected in the dependency
graph:

.. image:: images/dep-graph-correct.svg

.. _implementing-inames:

Implementing Inames
-------------------

.. _specifying-arguments:

Specifying arguments
--------------------

.. _implementing-array-axes:

Implementing Array Axes
-----------------------

Common Problems
---------------

A static maximum was not found
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attempting to create this kernel results in an error:

.. doctest::
    :options: +ELLIPSIS

    >>> knl = lp.make_kernel(ctx.devices[0],
    ...      "{ [i]: 0<=i<n }",
    ...      """
    ...      out[i] = 5
    ...      out[0] = 6
    ...      """)
    ... # Loopy prints the following before this exception:
    ... # While trying to find shape axis 0 of argument 'out', the following exception occurred:
    Traceback (most recent call last):
    ...
    ValueError: a static maximum was not found for PwAff '[n] -> { [(1)] : n = 1; [(n)] : n >= 2; [(1)] : n <= 0 }'

The problem is that loopy cannot find a simple, universally valid expression
for the length of *out* in this case. The set notation at the end of the error
message summarizes its best attempt:

* If n=1, then out has size 1.
* If n>=2, then out has size n.
* If n<=0, then out has size 1.

Sure, some of these cases could be coalesced, but that's beside the point.
Loopy does not know that non-positive values of *n* make no sense. It needs to
be told in order for the error to disappear--note the *assumptions* argument:

.. doctest::
    :options: +ELLIPSIS

    >>> knl = lp.make_kernel(ctx.devices[0],
    ...      "{ [i]: 0<=i<n }",
    ...      """
    ...      out[i] = 5
    ...      out[0] = 6
    ...      """, assumptions="n>=1")

Other situations where this error message can occur include:

* Finding size of prefetch/precompute arrays
* Finding sizes of argument arrays
* Finding workgroup sizes

.. vim: tw=75:spell

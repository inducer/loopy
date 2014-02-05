.. _tutorial:

Tutorial
========

This guide provides a gentle introduction into what loopy is, how it works,
and what it can do. There's also the :ref:`reference` that aims to
unambiguously define all aspects of loopy.

Preparation
-----------

:mod:`loopy` currently requires on :mod:`pyopencl` to be installed. We
import a few modules and set up a :class:`pyopencl.Context` and a
:class:`pyopencl.CommandQueue`:

.. doctest::

    >>> import numpy as np
    >>> import pyopencl as cl
    >>> import pyopencl.array
    >>> import pyopencl.clrandom
    >>> import loopy as lp
    >>> from pytools import StderrToStdout as IncludeWarningsInDoctest

    >>> ctx = cl.create_some_context(interactive=False)
    >>> queue = cl.CommandQueue(ctx)

We also create some data on the device that we'll use throughout our
examples:

.. doctest::

    >>> n = 16*16
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
    INAME IMPLEMENTATION TAGS:
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
by passing :attr:`loopy.Options.write_cl`.

.. doctest::

    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=x_vec_dev)
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
:attr:`loopy.Options.return_dict`.)

For convenience, loopy kernels also directly accept :mod:`numpy` arrays:

.. doctest::

    >>> evt, (out,) = knl(queue, a=x_vec_host)
    >>> assert (out == (2*x_vec_host)).all()

Notice how both *out* nor *a* are :mod:`numpy` arrays, but neither needed
to be transferred to or from the device.  Checking for numpy arrays and
transferring them if needed comes at a potential performance cost.  If you
would like to make sure that you avoid this cost, pass
:attr:`loopy.Options.no_numpy`.

Further notice how *n*, while technically being an argument, did not need
to be passed, as loopy is able to find *n* from the shape of the input
argument *a*.

For efficiency, loopy generates Python code that handles kernel invocation.
If you are suspecting that this code is causing you an issue, you can
inspect that code, too, using :attr:`loopy.Options.write_wrapper`:

.. doctest::

    >>> knl = lp.set_options(knl, write_wrapper=True, write_cl=False)
    >>> evt, (out,) = knl(queue, a=x_vec_host)
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
  traversed. ``i==3`` could be reached before ``i==0`` but also before
  ``i==17``. Your program is only correct if it produces a valid result
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

Loops and dependencies
~~~~~~~~~~~~~~~~~~~~~~

Next, it is important to understand how loops and dependencies interact.
Let us take a look at the generated code for the above kernel:

.. doctest::

    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=a_mat_dev)
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

    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=a_mat_dev)
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

Loop nesting
~~~~~~~~~~~~

One last aspect of ordering over which we have thus far not exerted any
control is the nesting of loops. For example, should the *i* loop be nested
around the *j* loop, or the other way around, in the following simple
zero-fill kernel?

.. doctest::

    >>> knl = lp.make_kernel(ctx.devices[0],
    ...     "{ [i,j]: 0<=i,j<n }",
    ...     """
    ...     a[i,j] = 0
    ...     """)


    >>> knl = lp.set_options(knl, "write_cl")
    >>> with IncludeWarningsInDoctest():
    ...     evt, (out,) = knl(queue, a=a_mat_dev)
    <BLANKLINE>
    ...
      for (int i = 0; i <= (-1 + n); ++i)
        for (int j = 0; j <= (-1 + n); ++j)
          a[n * i + j] = 0.0f;
    ...

Loopy has chosen a loop nesting for us, but it did not like doing so, as it
also issued the following warning::

    LoopyWarning: kernel scheduling was ambiguous--more than one schedule found, ignoring

This is easily resolved:

.. doctest::

    >>> knl = lp.set_loop_priority(knl, "j,i")

:func:`loopy.set_loop_priority` indicates the textual order in which loops
should be entered in the kernel code.  Note that this priority has an
advisory role only. If the kernel logically requires a different nesting,
loop priority is ignored.  Priority is only considered if loop nesting is
ambiguous.

.. doctest::

    >>> with IncludeWarningsInDoctest():
    ...     evt, (out,) = knl(queue, a=a_mat_dev)
    <BLANKLINE>
    ...
      for (int j = 0; j <= (-1 + n); ++j)
        for (int i = 0; i <= (-1 + n); ++i)
          a[n * i + j] = 0.0f;
    ...

No more warnings! Loop nesting is also reflected in the dependency graph:

.. image:: images/dep-graph-nesting.svg

.. _intro-transformations:

Introduction to Kernel Transformations
--------------------------------------

What we have covered thus far puts you in a position to describe many kinds
of computations to loopy--in the sense that loopy will generate code that
carries out the correct operation. That's nice, but it's natural to also
want control over *how* a program is executed. Loopy's way of capturing
this information is by way of *transformations*. These have the following
general shape::

    new_kernel = lp.do_something(old_knl, arguments...)

These transformations always return a *copy* of the old kernel with the
requested change applied. Typically, the variable holding the old kernel
is overwritten with the new kernel::

    knl = lp.do_something(knl, arguments...)

We've already seen an example of a transformation above:
For instance, :func:`set_loop_priority` fit the pattern.

:func:`loopy.split_iname` is another fundamental (and useful) transformation. It
turns one existing iname (recall that this is loopy's word for a 'loop
variable', roughly) into two new ones, an 'inner' and an 'outer' one,
where the 'inner' loopy is of a fixed, specified length, and the 'outer'
loop runs over these fixed-length 'chunks'. The three inames have the
following relationship to one another::

    OLD = INNER + GROUP_SIZE * OUTER

Consider this example:

.. doctest::

    >>> knl = lp.make_kernel(ctx.devices[0],
    ...     "{ [i]: 0<=i<n }",
    ...     "a[i] = 0", assumptions="n>=0")
    >>> knl = lp.split_iname(knl, "i", 16)
    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=x_vec_dev)
    <BLANKLINE>
    ...
      for (int i_outer = 0; i_outer <= (-1 + ((15 + n) / 16)); ++i_outer)
        for (int i_inner = 0; i_inner <= 15; ++i_inner)
          if ((-1 + -1 * i_inner + -16 * i_outer + n) >= 0)
            a[i_inner + i_outer * 16] = 0.0f;
    ...

By default, the new, split inames are named *OLD_outer* and *OLD_inner*,
where *OLD* is the name of the previous iname. Upon exit from
:func:`loopy.split_iname`, *OLD* is removed from the kernel and replaced by
*OLD_inner* and *OLD_outer*.

Also take note of the *assumptions* argument. This makes it possible to
communicate assumptions about loop domain parameters. (but *not* about
data) In this case, assuming non-negativity helps loopy generate more
efficient code for division in the loop bound for *i_outer*. See below
on how to communicate divisibility assumptions.

Note that the words 'inner' and 'outer' here have no implied meaning in
relation to loop nesting. For example, it's perfectly possible to request
*i_inner* to be nested outside *i_outer*:

.. doctest::

    >>> knl = lp.set_loop_priority(knl, "i_inner,i_outer")
    >>> evt, (out,) = knl(queue, a=x_vec_dev)
    <BLANKLINE>
    ...
      for (int i_inner = 0; i_inner <= 15; ++i_inner)
        if ((-1 + n) >= 0)
          for (int i_outer = 0; i_outer <= (-1 + -1 * i_inner + ((15 + n + 15 * i_inner) / 16)); ++i_outer)
            a[i_inner + i_outer * 16] = 0.0f;
    ...

Notice how loopy has automatically generated guard conditionals to make
sure the bounds on the old iname are obeyed.

The combination of :func:`loopy.split_iname` and
:func:`loopy.set_loop_priority` is already good enough to implement what is
commonly called 'loop tiling':

.. doctest::

    >>> knl = lp.make_kernel(ctx.devices[0],
    ...     "{ [i,j]: 0<=i,j<n }",
    ...     "out[i,j] = a[j,i]",
    ...     assumptions="n mod 16 = 0 and n >= 1")
    >>> knl = lp.split_iname(knl, "i", 16)
    >>> knl = lp.split_iname(knl, "j", 16)
    >>> knl = lp.set_loop_priority(knl, "i_outer,j_outer,i_inner")
    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=a_mat_dev)
    <BLANKLINE>
    ...
      for (int i_outer = 0; i_outer <= (-1 + ((15 + n) / 16)); ++i_outer)
        for (int j_outer = 0; j_outer <= (-1 + ((15 + n) / 16)); ++j_outer)
          for (int i_inner = 0; i_inner <= 15; ++i_inner)
            for (int j_inner = 0; j_inner <= 15; ++j_inner)
              out[n * (i_inner + i_outer * 16) + j_inner + j_outer * 16] = a[n * (j_inner + j_outer * 16) + i_inner + i_outer * 16];
    ...



.. _implementing-inames:

Implementing Loop Axes ("Inames")
---------------------------------

So far, all the loops we have seen loopy implement were ``for`` loops. Each
iname in loopy carries a so-called 'implementation tag'.  :ref:`tags` shows
all possible choices for iname implementation tags. The important ones are
explained below.

Unrolling
~~~~~~~~~

Our first example of an 'implementation tag' is ``"unr"``, which performs
loop unrolling.  Let us split the main loop of a vector fill kernel into
chunks of 4 and unroll the fixed-length inner loop by setting the inner
loop's tag to ``"unr"``:

.. doctest::

    >>> knl = lp.make_kernel(ctx.devices[0],
    ...     "{ [i]: 0<=i<n }",
    ...     "a[i] = 0", assumptions="n>=0 and n mod 4 = 0")
    >>> orig_knl = knl
    >>> knl = lp.split_iname(knl, "i", 4)
    >>> knl = lp.tag_inames(knl, dict(i_inner="unr"))
    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=x_vec_dev)
    <BLANKLINE>
    ...
      for (int i_outer = 0; i_outer <= (-1 + ((3 + n) / 4)); ++i_outer)
      {
        a[0 + i_outer * 4] = 0.0f;
        a[1 + i_outer * 4] = 0.0f;
        a[2 + i_outer * 4] = 0.0f;
        a[3 + i_outer * 4] = 0.0f;
      }
    ...


:func:`loopy.tag_inames` is a new transformation that assigns
implementation tags to kernels.  ``"unr"`` is the first tag we've
explicitly learned about. Technically, though, it is the second--``"for"``
(or, equivalently, *None*), which is the default, instructs loopy to
implement an iname using a for loop.

Unrolling obviously only works for inames with a fixed maximum number of
values, since only a finite amount of code can be generated. Unrolling the
entire *i* loop in the kernel above would not work.

Split-and-tag
~~~~~~~~~~~~~

Since split-and-tag is such a common combination, :func:`loopy.split_iname`
provides a shortcut:

.. doctest::

    >>> knl = orig_knl
    >>> knl = lp.split_iname(knl, "i", 4, inner_tag="unr")

The *outer_tag* keyword argument exists, too, and works just like you would
expect.

Printing
~~~~~~~~

Iname implementation tags are also printed along with the entire kernel:

.. doctest::

    >>> print knl
    ---------------------------------------------------------------------------
    ...
    INAME IMPLEMENTATION TAGS:
    i_inner: unr
    i_outer: None
    ---------------------------------------------------------------------------
    ...

Parallelization
~~~~~~~~~~~~~~~

Loops are also parallelized in loopy by assigning them parallelizing
implementation tags. In OpenCL, this means that the loop variable
corresponds to either a local ID or a workgroup ID. The implementation tags
for local IDs are ``"l.0"``, ``"l.1"``, ``"l.2"``, and so on.  The
corresponding tags for group IDs are ``"g.0"``, ``"g.1"``, ``"g.2"``, and
so on.

Let's try this out on our vector fill kernel by creating workgroups of size
128:

.. doctest::

    >>> knl = lp.make_kernel(ctx.devices[0],
    ...     "{ [i]: 0<=i<n }",
    ...     "a[i] = 0", assumptions="n>=0")
    >>> knl = lp.split_iname(knl, "i", 128,
    ...         outer_tag="g.0", inner_tag="l.0")
    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=x_vec_dev)
    <BLANKLINE>
    ...
    __kernel void __attribute__ ((reqd_work_group_size(128, 1, 1))) loopy_kernel(__global float *restrict a, int const n)
    {
    <BLANKLINE>
      if ((-1 + -128 * gid(0) + -1 * lid(0) + n) >= 0)
        a[lid(0) + gid(0) * 128] = 0.0f;
    }

Loopy requires that workgroup sizes are fixed and constant at compile time.
By comparison, the overall execution ND-range size (i.e. the number of
workgroups) is allowed to be runtime-variable.

Note how there was no need to specify group or range sizes. Loopy computes
those for us:

.. doctest::

    >>> glob, loc = knl.get_grid_sizes()
    >>> print glob
    (Aff("[n] -> { [(floor((127 + n)/128))] }"),)
    >>> print loc
    (Aff("[n] -> { [(128)] }"),)

Note that this functionality returns internal objects and is not really
intended for end users.

Avoiding Conditionals
~~~~~~~~~~~~~~~~~~~~~

You may have observed above that we have used a divisibility assumption on
*n* in the kernels above. Without this assumption, loopy would generate
conditional code to make sure no out-of-bounds loop instances are executed.
This here is the original unrolling example without the divisibility
assumption:

.. doctest::

    >>> knl = lp.make_kernel(ctx.devices[0],
    ...     "{ [i]: 0<=i<n }",
    ...     "a[i] = 0", assumptions="n>=0")
    >>> orig_knl = knl
    >>> knl = lp.split_iname(knl, "i", 4)
    >>> knl = lp.tag_inames(knl, dict(i_inner="unr"))
    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=x_vec_dev)
    <BLANKLINE>
    ...
      for (int i_outer = 0; i_outer <= (-1 + ((3 + n) / 4)); ++i_outer)
      {
        a[0 + i_outer * 4] = 0.0f;
        if ((-2 + -4 * i_outer + n) >= 0)
          a[1 + i_outer * 4] = 0.0f;
        if ((-3 + -4 * i_outer + n) >= 0)
          a[2 + i_outer * 4] = 0.0f;
        if ((-4 + -4 * i_outer + n) >= 0)
          a[3 + i_outer * 4] = 0.0f;
      }
    ...

While these conditionals enable the generated code to deal with arbitrary
*n*, they come at a performance cost. Loopy allows generating separate code
for the last iteration of the *i_outer* loop, by using the *slabs* keyword
argument to :func:`split_iname`. Since this last iteration of *i_outer* is
the only iteration for which ``i_inner + 4*i_outer`` can become larger than
*n*, only the (now separate) code for that iteration contains conditionals,
enabling some cost savings:

.. doctest::

    >>> knl = orig_knl
    >>> knl = lp.split_iname(knl, "i", 4, slabs=(0, 1), inner_tag="unr")
    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=x_vec_dev)
    <BLANKLINE>
    ...
      /* bulk slab for 'i_outer' */
      for (int i_outer = 0; i_outer <= (-2 + ((3 + n) / 4)); ++i_outer)
      {
        a[0 + i_outer * 4] = 0.0f;
        a[1 + i_outer * 4] = 0.0f;
        a[2 + i_outer * 4] = 0.0f;
        a[3 + i_outer * 4] = 0.0f;
      }
      /* final slab for 'i_outer' */
      for (int i_outer = (-1 + n + -1 * (3 * n / 4)); i_outer <= (-1 + ((3 + n) / 4)); ++i_outer)
        if ((-1 + n) >= 0)
        {
          a[0 + i_outer * 4] = 0.0f;
          if ((-2 + -4 * i_outer + n) >= 0)
            a[1 + i_outer * 4] = 0.0f;
          if ((-3 + -4 * i_outer + n) >= 0)
            a[2 + i_outer * 4] = 0.0f;
          if ((4 + 4 * i_outer + -1 * n) == 0)
            a[3 + i_outer * 4] = 0.0f;
        }
    ...

.. _specifying-arguments:

Specifying arguments
--------------------

.. _implementing-array-axes:

Implementing Array Axes
-----------------------

.. _more-complicated-programs:

More complicated programs
-------------------------

SCOP

Data-dependent control flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conditionals
~~~~~~~~~~~~

Snippets of C
~~~~~~~~~~~~~


Common Problems
---------------

A static maximum was not found
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attempting to create this kernel results in an error:

.. doctest::

    >>> lp.make_kernel(ctx.devices[0],
    ...     "{ [i]: 0<=i<n }",
    ...     """
    ...     out[i] = 5
    ...     out[0] = 6
    ...     """)
    ... # Loopy prints the following before this exception:
    ... # While trying to find shape axis 0 of argument 'out', the following exception occurred:
    Traceback (most recent call last):
    ...
    ValueError: a static maximum was not found for PwAff '[n] -> { [(1)] : n = 1; [(n)] : n >= 2; [(1)] : n <= 0 }'

The problem is that loopy cannot find a simple, universally valid expression
for the length of *out* in this case. Notice how the kernel accesses both the
*i*th and the first element of out.  The set notation at the end of the error
message summarizes its best attempt:

* If n=1, then out has size 1.
* If n>=2, then out has size n.
* If n<=0, then out has size 1.

Sure, some of these cases could be coalesced, but that's beside the point.
Loopy does not know that non-positive values of *n* make no sense. It needs to
be told in order for the error to disappear--note the *assumptions* argument:

.. doctest::

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

Write races
~~~~~~~~~~~

This kernel performs a simple transposition of an input matrix:

.. doctest::

    >>> knl = lp.make_kernel(ctx.devices[0],
    ...       "{ [i,j]: 0<=i,j<n }",
    ...       """
    ...       out[j,i] = a[i,j]
    ...       """, assumptions="n>=1", name="transpose")

To get it ready for execution on a GPU, we split the *i* and *j* loops into
groups of 16.

.. doctest::

    >>> knl = lp.split_iname(knl,  "j", 16, inner_tag="l.1", outer_tag="g.0")
    >>> knl = lp.split_iname(knl,  "i", 16, inner_tag="l.0", outer_tag="g.1")

We'll also request a prefetch--but suppose we only do so across the
*i_inner* iname:

.. doctest::

    >>> knl = lp.add_prefetch(knl, "a", "i_inner")

When we try to run our code, we get the following warning from loopy as a first
sign that something is amiss:

.. doctest::

    >>> with IncludeWarningsInDoctest():
    ...     evt, (out,) = knl(queue, a=a_mat_dev)
    /...: WriteRaceConditionWarning: instruction 'a_fetch' looks invalid: it assigns to indices based on local IDs, but its temporary 'a_fetch_0' cannot be made local because a write race across the iname(s) 'j_inner' would emerge. (Do you need to add an extra iname to your prefetch?) (add 'write_race_local(a_fetch)' to silenced_warnings kernel argument to disable)
      warn(text, type)

When we ask to see the code, the issue becomes apparent:

.. doctest::

    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=a_mat_dev)
    <BLANKLINE>
    #define lid(N) ((int) get_local_id(N))
    #define gid(N) ((int) get_group_id(N))
    <BLANKLINE>
    __kernel void __attribute__ ((reqd_work_group_size(16, 16, 1))) transpose(__global float const *restrict a, int const n, __global float *restrict out)
    {
      float a_fetch_0[16];
    <BLANKLINE>
      ...
          a_fetch_0[lid(0)] = a[n * (lid(0) + 16 * gid(1)) + lid(1) + 16 * gid(0)];
      ...
          out[n * (lid(1) + gid(0) * 16) + lid(0) + gid(1) * 16] = a_fetch_0[lid(0)];
      ...
    }

Loopy has a 2D workgroup to use for prefetching of a 1D array. When it
considers making *a_fetch_0* ``local`` (in the OpenCL memory sense of the word)
to make use of parallelism in prefetching, it discovers that a write race
across the remaining axis of the workgroup would emerge.

TODO

.. vim: tw=75:spell

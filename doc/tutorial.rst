.. _tutorial:

Tutorial
========

This guide provides a gentle introduction into what loopy is, how it works,
and what it can do. There's also the :ref:`reference` that aims to
unambiguously define all aspects of loopy.

Preparation
-----------

.. {{{

:mod:`loopy` currently requires on :mod:`pyopencl` to be installed. We
import a few modules and set up a :class:`pyopencl.Context` and a
:class:`pyopencl.CommandQueue`:

.. doctest::

    >>> import numpy as np
    >>> import pyopencl as cl
    >>> import pyopencl.array
    >>> import pyopencl.clrandom

    >>> import loopy as lp
    >>> lp.set_caching_enabled(False)

    >>> from warnings import filterwarnings, catch_warnings
    >>> filterwarnings('error', category=lp.LoopyWarning)

    >>> ctx = cl.create_some_context(interactive=False)
    >>> queue = cl.CommandQueue(ctx)

We also create some data on the device that we'll use throughout our
examples:

.. doctest::

    >>> n = 16*16
    >>> x_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
    >>> y_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
    >>> z_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
    >>> a_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
    >>> b_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)

And some data on the host:

.. doctest::

    >>> x_vec_host = np.random.randn(n).astype(np.float32)
    >>> y_vec_host = np.random.randn(n).astype(np.float32)

.. }}}

Getting started
---------------

.. {{{

We'll start by taking a closer look at a very simple kernel that reads in
one vector, doubles it, and writes it to another.

.. doctest::

    >>> knl = lp.make_kernel(
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

As you create and transform kernels, it's useful to know that you can
always see loopy's view of a kernel by printing it.

.. doctest::

    >>> knl = lp.set_options(knl, allow_terminal_colors=False)
    >>> print(knl)
    ---------------------------------------------------------------------------
    KERNEL: loopy_kernel
    ---------------------------------------------------------------------------
    ARGUMENTS:
    a: GlobalArg, type: <runtime>, shape: (n), dim_tags: (N0:stride:1)
    n: ValueArg, type: <runtime>
    out: GlobalArg, type: <runtime>, shape: (n), dim_tags: (N0:stride:1)
    ---------------------------------------------------------------------------
    DOMAINS:
    [n] -> { [i] : 0 <= i < n }
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

.. }}}

Running a kernel
----------------

.. {{{

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
    #define lid(N) ((int) get_local_id(N))
    #define gid(N) ((int) get_group_id(N))
    <BLANKLINE>
    __kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global float const *restrict a, int const n, __global float *restrict out)
    {
      for (int i = 0; i <= -1 + n; ++i)
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

Notice how both *out* and *a* are :mod:`numpy` arrays, but neither needed
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
    def invoke_loopy_kernel_loopy_kernel(_lpy_cl_kernels, queue, allocator=None, wait_for=None, out_host=None, a=None, n=None, out=None):
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
    >>> print(code)
    #define lid(N) ((int) get_local_id(N))
    #define gid(N) ((int) get_group_id(N))
    <BLANKLINE>
    __kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global float const *restrict a, int const n, __global float *restrict out)
    {
      for (int i = 0; i <= -1 + n; ++i)
        out[i] = 2.0f * a[i];
    }

.. }}}

.. _ordering:

Ordering
--------

.. {{{

Next, we'll change our kernel a bit. Our goal will be to transpose a matrix
and double its entries, and we will do this in two steps for the sake of
argument:

.. doctest::

    >>> # WARNING: Incorrect.
    >>> knl = lp.make_kernel(
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
    >>> knl = lp.make_kernel(
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

    >>> print(knl.stringify(with_dependencies=True))
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

The intent of this heuristic is to cover the common case of a
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
    >>> knl = lp.set_loop_priority(knl, "i,j")
    >>> evt, (out,) = knl(queue, a=a_mat_dev)
    #define lid(N) ((int) get_local_id(N))
    #define gid(N) ((int) get_group_id(N))
    <BLANKLINE>
    __kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global float const *restrict a, int const n, __global float *restrict out)
    {
      for (int i = 0; i <= -1 + n; ++i)
        for (int j = 0; j <= -1 + n; ++j)
        {
          out[n * j + i] = a[n * i + j];
          out[n * i + j] = 2.0f * out[n * i + j];
        }
    }

While our requested instruction ordering has been obeyed, something is
still not right:

.. doctest::

    >>> print((out.get() == a_mat_dev.get().T*2).all())
    False

For the kernel to perform the desired computation, *all
instances* (loop iterations) of the first instruction need to be completed,
not just the one for the current values of *(i, j)*.

    Dependencies in loopy act *within* the largest common set of shared
    inames.

As a result, our example above realizes the dependency *within* the *i* and *j*
loops. To fix our example, we simply create a new pair of loops *ii* and *jj*
with identical bounds, for the use of the transpose:

.. doctest::

    >>> knl = lp.make_kernel(
    ...     "{ [i,j,ii,jj]: 0<=i,j,ii,jj<n }",
    ...     """
    ...     out[j,i] = a[i,j] {id=transpose}
    ...     out[ii,jj] = 2*out[ii,jj]  {dep=transpose}
    ...     """)
    >>> knl = lp.set_loop_priority(knl, "i,j,ii,jj")

:func:`loopy.duplicate_inames` can be used to achieve the same goal.
Now the intended code is generated and our test passes.

.. doctest::

    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=a_mat_dev)
    #define lid(N) ((int) get_local_id(N))
    #define gid(N) ((int) get_group_id(N))
    <BLANKLINE>
    __kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global float const *restrict a, int const n, __global float *restrict out)
    {
      for (int i = 0; i <= -1 + n; ++i)
        for (int j = 0; j <= -1 + n; ++j)
          out[n * j + i] = a[n * i + j];
      for (int ii = 0; ii <= -1 + n; ++ii)
        for (int jj = 0; jj <= -1 + n; ++jj)
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

It turns out that Loopy will typically choose a loop nesting for us, but it
does not like doing so. Loo.py will react to the following code

.. doctest::

    >>> knl = lp.make_kernel(
    ...     "{ [i,j]: 0<=i,j<n }",
    ...     """
    ...     a[i,j] = 0
    ...     """)

By saying::

    LoopyWarning: kernel scheduling was ambiguous--more than one schedule found, ignoring

And by picking one of the possible loop orderings at random.

The warning (and the nondeterminism it warns about) is easily resolved:

.. doctest::

    >>> knl = lp.set_loop_priority(knl, "j,i")

:func:`loopy.set_loop_priority` indicates the textual order in which loops
should be entered in the kernel code.  Note that this priority has an
advisory role only. If the kernel logically requires a different nesting,
loop priority is ignored.  Priority is only considered if loop nesting is
ambiguous.

.. doctest::

    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=a_mat_dev)
    #define lid(N) ((int) get_local_id(N))
    ...
      for (int j = 0; j <= -1 + n; ++j)
        for (int i = 0; i <= -1 + n; ++i)
          a[n * i + j] = 0.0f;
    ...

No more warnings! Loop nesting is also reflected in the dependency graph:

.. image:: images/dep-graph-nesting.svg

.. }}}

.. _intro-transformations:

Introduction to Kernel Transformations
--------------------------------------

.. {{{

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
where the 'inner' loop is of a fixed, specified length, and the 'outer'
loop runs over these fixed-length 'chunks'. The three inames have the
following relationship to one another::

    OLD = INNER + GROUP_SIZE * OUTER

Consider this example:

.. doctest::

    >>> knl = lp.make_kernel(
    ...     "{ [i]: 0<=i<n }",
    ...     "a[i] = 0", assumptions="n>=1")
    >>> knl = lp.split_iname(knl, "i", 16)
    >>> knl = lp.set_loop_priority(knl, "i_outer,i_inner")
    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=x_vec_dev)
    #define lid(N) ((int) get_local_id(N))
    ...
      for (int i_outer = 0; i_outer <= -1 + ((15 + n) / 16); ++i_outer)
        for (int i_inner = 0; i_inner <= 15; ++i_inner)
          if (-1 + -1 * i_inner + -16 * i_outer + n >= 0)
            a[16 * i_outer + i_inner] = 0.0f;
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
    #define lid(N) ((int) get_local_id(N))
    ...
      for (int i_inner = 0; i_inner <= 15; ++i_inner)
        if (-1 + -1 * i_inner + n >= 0)
          for (int i_outer = 0; i_outer <= -1 + -1 * i_inner + ((15 + n + 15 * i_inner) / 16); ++i_outer)
            a[16 * i_outer + i_inner] = 0.0f;
    ...

Notice how loopy has automatically generated guard conditionals to make
sure the bounds on the old iname are obeyed.

The combination of :func:`loopy.split_iname` and
:func:`loopy.set_loop_priority` is already good enough to implement what is
commonly called 'loop tiling':

.. doctest::

    >>> knl = lp.make_kernel(
    ...     "{ [i,j]: 0<=i,j<n }",
    ...     "out[i,j] = a[j,i]",
    ...     assumptions="n mod 16 = 0 and n >= 1")
    >>> knl = lp.split_iname(knl, "i", 16)
    >>> knl = lp.split_iname(knl, "j", 16)
    >>> knl = lp.set_loop_priority(knl, "i_outer,j_outer,i_inner")
    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=a_mat_dev)
    #define lid(N) ((int) get_local_id(N))
    ...
      for (int i_outer = 0; i_outer <= ((-16 + n) / 16); ++i_outer)
        for (int j_outer = 0; j_outer <= ((-16 + n) / 16); ++j_outer)
          for (int i_inner = 0; i_inner <= 15; ++i_inner)
            for (int j_inner = 0; j_inner <= 15; ++j_inner)
              out[n * (16 * i_outer + i_inner) + 16 * j_outer + j_inner] = a[n * (16 * j_outer + j_inner) + 16 * i_outer + i_inner];
    ...

.. }}}

.. _implementing-inames:

Implementing Loop Axes ("Inames")
---------------------------------

.. {{{

So far, all the loops we have seen loopy implement were ``for`` loops. Each
iname in loopy carries a so-called 'implementation tag'.  :ref:`iname-tags` shows
all possible choices for iname implementation tags. The important ones are
explained below.

Unrolling
~~~~~~~~~

Our first example of an 'implementation tag' is ``"unr"``, which performs
loop unrolling.  Let us split the main loop of a vector fill kernel into
chunks of 4 and unroll the fixed-length inner loop by setting the inner
loop's tag to ``"unr"``:

.. doctest::

    >>> knl = lp.make_kernel(
    ...     "{ [i]: 0<=i<n }",
    ...     "a[i] = 0", assumptions="n>=0 and n mod 4 = 0")
    >>> orig_knl = knl
    >>> knl = lp.split_iname(knl, "i", 4)
    >>> knl = lp.tag_inames(knl, dict(i_inner="unr"))
    >>> knl = lp.set_loop_priority(knl, "i_outer,i_inner")
    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=x_vec_dev)
    #define lid(N) ((int) get_local_id(N))
    #define gid(N) ((int) get_group_id(N))
    #define int_floor_div_pos_b(a,b) (                 ( (a) - ( ((a)<0) ? ((b)-1) : 0 )  ) / (b)                 )
    ...
      for (int i_outer = 0; i_outer <= int_floor_div_pos_b(-4 + n, 4); ++i_outer)
      {
        a[4 * i_outer + 0] = 0.0f;
        a[4 * i_outer + 1] = 0.0f;
        a[4 * i_outer + 2] = 0.0f;
        a[4 * i_outer + 3] = 0.0f;
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

    >>> print(knl)
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

    >>> knl = lp.make_kernel(
    ...     "{ [i]: 0<=i<n }",
    ...     "a[i] = 0", assumptions="n>=0")
    >>> knl = lp.split_iname(knl, "i", 128,
    ...         outer_tag="g.0", inner_tag="l.0")
    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=x_vec_dev)
    #define lid(N) ((int) get_local_id(N))
    ...
    __kernel void __attribute__ ((reqd_work_group_size(128, 1, 1))) loopy_kernel(__global float *restrict a, int const n)
    {
      if (-1 + -128 * gid(0) + -1 * lid(0) + n >= 0)
        a[128 * gid(0) + lid(0)] = 0.0f;
    }

Loopy requires that workgroup sizes are fixed and constant at compile time.
By comparison, the overall execution ND-range size (i.e. the number of
workgroups) is allowed to be runtime-variable.

Note how there was no need to specify group or range sizes. Loopy computes
those for us:

.. doctest::

    >>> glob, loc = knl.get_grid_size_upper_bounds()
    >>> print(glob)
    (Aff("[n] -> { [(floor((127 + n)/128))] }"),)
    >>> print(loc)
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

    >>> knl = lp.make_kernel(
    ...     "{ [i]: 0<=i<n }",
    ...     "a[i] = 0", assumptions="n>=0")
    >>> orig_knl = knl
    >>> knl = lp.split_iname(knl, "i", 4)
    >>> knl = lp.tag_inames(knl, dict(i_inner="unr"))
    >>> knl = lp.set_loop_priority(knl, "i_outer,i_inner")
    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=x_vec_dev)
    #define lid(N) ((int) get_local_id(N))
    ...
      for (int i_outer = 0; i_outer <= -1 + ((3 + n) / 4); ++i_outer)
      {
        a[4 * i_outer + 0] = 0.0f;
        if (-2 + -4 * i_outer + n >= 0)
          a[4 * i_outer + 1] = 0.0f;
        if (-3 + -4 * i_outer + n >= 0)
          a[4 * i_outer + 2] = 0.0f;
        if (-4 + -4 * i_outer + n >= 0)
          a[4 * i_outer + 3] = 0.0f;
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
    >>> knl = lp.set_loop_priority(knl, "i_outer,i_inner")
    >>> evt, (out,) = knl(queue, a=x_vec_dev)
    #define lid(N) ((int) get_local_id(N))
    ...
      /* bulk slab for 'i_outer' */
      for (int i_outer = 0; i_outer <= -2 + ((3 + n) / 4); ++i_outer)
      {
        a[4 * i_outer + 0] = 0.0f;
        a[4 * i_outer + 1] = 0.0f;
        a[4 * i_outer + 2] = 0.0f;
        a[4 * i_outer + 3] = 0.0f;
      }
      /* final slab for 'i_outer' */
      for (int i_outer = -1 + n + -1 * (3 * n / 4); i_outer <= -1 + ((3 + n) / 4); ++i_outer)
        if (-1 + n >= 0)
        {
          a[4 * i_outer + 0] = 0.0f;
          if (-2 + -4 * i_outer + n >= 0)
            a[4 * i_outer + 1] = 0.0f;
          if (-3 + -4 * i_outer + n >= 0)
            a[4 * i_outer + 2] = 0.0f;
          if (4 + 4 * i_outer + -1 * n == 0)
            a[4 * i_outer + 3] = 0.0f;
        }
    ...

.. }}}

.. _specifying-arguments:

Specifying arguments
--------------------

* Kinds: global, constant, value
* Types

.. _argument-shapes:

Argument shapes
~~~~~~~~~~~~~~~

Shapes (and automatic finding thereof)

.. _implementing-array-axes:

Implementing Array Axes
~~~~~~~~~~~~~~~~~~~~~~~


Precomputation, Storage, and Temporary Variables
------------------------------------------------

.. {{{

The loopy kernels we have seen thus far have consisted only of assignments
from one global-memory storage location to another. Sometimes, computation
results obviously get reused, so that recomputing them or even just
re-fetching them from global memory becomes uneconomical. Loopy has
a number of different ways of addressing this need.

Explicit private temporaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest of these ways is the creation of an explicit temporary
variable, as one might do in C or another programming language:

.. doctest::

    >>> knl = lp.make_kernel(
    ...     "{ [i]: 0<=i<n }",
    ...     """
    ...     <float32> a_temp = sin(a[i])
    ...     out1[i] = a_temp {id=out1}
    ...     out2[i] = sqrt(1-a_temp*a_temp) {dep=out1}
    ...     """)

The angle brackets ``<>`` denote the creation of a temporary. The name of
the temporary may not override inames, argument names, or other names in
the kernel. The name in between the angle brackets is a typename as
understood by the type registry :mod:`pyopencl.array`. To first order,
the conventional :mod:`numpy` scalar types (:class:`numpy.int16`,
:class:`numpy.complex128`) will work. (Yes, :mod:`loopy` supports and
generates correct code for complex arithmetic.)

(If you're wondering, the dependencies above were added to make the doctest
produce predictable output.)

The generated code places this variable into what OpenCL calls 'private'
memory, local to each work item.

.. doctest::

    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out1, out2) = knl(queue, a=x_vec_dev)
    #define lid(N) ((int) get_local_id(N))
    ...
    {
      float a_temp;
    <BLANKLINE>
      for (int i = 0; i <= -1 + n; ++i)
      {
        a_temp = sin(a[i]);
        out1[i] = a_temp;
        out2[i] = sqrt(1.0f + -1.0f * a_temp * a_temp);
      }
    }

Type inference for temporaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most :mod:`loopy` code can be written so as to be type-generic (with types
determined by parameters passed at run time). The same is true for
temporary variables--specifying a type for the variable is optional. As you
can see in the code below, angle brackets alone denote that a temporary
should be created, and the type of the variable will be deduced from the
expression being assigned.

.. doctest::

    >>> knl = lp.make_kernel(
    ...     "{ [i]: 0<=i<n }",
    ...     """
    ...     <> a_temp = sin(a[i])
    ...     out1[i] = a_temp
    ...     out2[i] = sqrt(1-a_temp*a_temp)
    ...     """)
    >>> evt, (out1, out2) = knl(queue, a=x_vec_dev)

Temporaries in local memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In most situations, :mod:`loopy` will  automatically deduce whether a given
temporary should be placed into local or private storage. If the variable
is ever written to in parallel and indexed by expressions containing local
IDs, then it is marked as residing in local memory. If this heuristic is
insufficient, :class:`loopy.TemporaryVariable` instances can be marked
local manually.

Consider the following example:

.. doctest::

    >>> knl = lp.make_kernel(
    ...     "{ [i_outer,i_inner, k]:  "
    ...          "0<= 16*i_outer + i_inner <n and 0<= i_inner,k <16}",
    ...     """
    ...     <> a_temp[i_inner] = a[16*i_outer + i_inner] {priority=10}
    ...     out[16*i_outer + i_inner] = sum(k, a_temp[k])
    ...     """)
    >>> knl = lp.tag_inames(knl, dict(i_outer="g.0", i_inner="l.0"))
    >>> knl = lp.set_options(knl, "write_cl")
    >>> evt, (out,) = knl(queue, a=x_vec_dev)
    #define lid(N) ((int) get_local_id(N))
    ...
    {
      __local float a_temp[16];
      float acc_k;
    <BLANKLINE>
      if (-1 + -16 * gid(0) + -1 * lid(0) + n >= 0)
      {
        a_temp[lid(0)] = a[16 * gid(0) + lid(0)];
        acc_k = 0.0f;
      }
      barrier(CLK_LOCAL_MEM_FENCE) /* for a_temp (insn_0_k_update depends on insn) */;
      if (-1 + -16 * gid(0) + -1 * lid(0) + n >= 0)
      {
        for (int k = 0; k <= 15; ++k)
          acc_k = acc_k + a_temp[k];
        out[16 * gid(0) + lid(0)] = acc_k;
      }
    }

Observe that *a_temp* was automatically placed in local memory, because
it is written in parallel across values of the group-local iname
*i_inner*. In addition, :mod:`loopy` has emitted a barrier instruction to
achieve the :ref:`ordering` specified by the instruction dependencies.

(The ``priority=10`` attribute was added to make the output of the test
deterministic.)

.. note::

    It is worth noting that it was not necessary to provide a size for the
    temporary ``a_temp``. :mod:`loopy` deduced the size to be allocated (16
    entries in this case) from the indices being accessed. This works just
    as well for 2D and 3D temporaries.

    The mechanism for finding accessed indices is the same as described
    in :ref:`argument-shapes`.

    If the size-finding heuristic fails or is impractical to use, the of
    the temporary can be specified by explicitly creating a
    :class:`loopy.TemporaryVariable`.

    Note that the size of local temporaries must, for now, be a constant at
    compile time.

Prefetching
~~~~~~~~~~~

The above code example may have struck you as 'un-loopy-ish' in the sense
that whether the contents of *a* is loaded into an temporary should be
considered an implementation detail that is taken care of by a
transformation rather than being baked into the code. Indeed, such a
transformation exists in :func:`loopy.add_prefetch`:

.. doctest::

    >>> knl = lp.make_kernel(
    ...     "{ [i_outer,i_inner, k]:  "
    ...          "0<= 16*i_outer + i_inner <n and 0<= i_inner,k <16}",
    ...     """
    ...     out[16*i_outer + i_inner] = sum(k, a[16*i_outer + i_inner])
    ...     """)
    >>> knl = lp.tag_inames(knl, dict(i_outer="g.0", i_inner="l.0"))
    >>> knl = lp.set_options(knl, "write_cl")
    >>> knl_pf = lp.add_prefetch(knl, "a")
    >>> evt, (out,) = knl_pf(queue, a=x_vec_dev)
    #define lid(N) ((int) get_local_id(N))
    ...
        acc_k = 0.0f;
        a_fetch = a[16 * gid(0) + lid(0)];
        for (int k = 0; k <= 15; ++k)
          acc_k = acc_k + a_fetch;
        out[16 * gid(0) + lid(0)] = acc_k;
    ...

This is not the same as our previous code and, in this scenario, a little
bit useless, because each entry of *a* is 'pre-fetched', used, and then
thrown away. (But realize that this could perhaps be useful in other
situations when the same entry of *a* is accessed multiple times.)

What's missing is that we need to tell :mod:`loopy` that we would like to
fetch the *access footprint* of an entire loop--in this case, of *i_inner*,
as the second argument of :func:`loopy.add_prefetch`. We thus arrive back
at the same code with a temporary in local memory that we had generated
earlier:

.. doctest::

    >>> knl_pf = lp.add_prefetch(knl, "a", ["i_inner"])
    >>> evt, (out,) = knl_pf(queue, a=x_vec_dev)
    #define lid(N) ((int) get_local_id(N))
    ...
      if (-1 + -16 * gid(0) + -1 * lid(0) + n >= 0)
        acc_k = 0.0f;
      if (-1 + -16 * gid(0) + -1 * lid(0) + n >= 0)
        a_fetch[lid(0)] = a[16 * gid(0) + lid(0)];
      barrier(CLK_LOCAL_MEM_FENCE) /* for a_fetch (insn_k_update depends on a_fetch_rule) */;
      if (-1 + -16 * gid(0) + -1 * lid(0) + n >= 0)
      {
        for (int k = 0; k <= 15; ++k)
          acc_k = acc_k + a_fetch[lid(0)];
        out[16 * gid(0) + lid(0)] = acc_k;
      }
    ...

Tagged prefetching
~~~~~~~~~~~~~~~~~~


Substitution rules
~~~~~~~~~~~~~~~~~~

Generic Precomputation
~~~~~~~~~~~~~~~~~~~~~~

.. }}}

.. _more-complicated-programs:

More complicated programs
-------------------------

.. {{{

SCOP

Data-dependent control flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conditionals
~~~~~~~~~~~~

Snippets of C
~~~~~~~~~~~~~

Atomic operations
~~~~~~~~~~~~~~~~~

Loopy supports atomic operations. To use them, both the data on which the
atomic operations work as well as the operations themselves must be
suitably tagged, as in the following example::


    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i%20] = out[i%20] + 2*a[i] {atomic}",
            [
                lp.GlobalArg("out", dtype, shape=lp.auto, for_atomic=True),
                lp.GlobalArg("a", dtype, shape=lp.auto),
                "..."
                ],
            assumptions="n>0")

.. }}}

Common Problems
---------------

.. {{{

A static maximum was not found
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attempting to create this kernel results in an error:

.. doctest::

    >>> lp.make_kernel(
    ...     "{ [i]: 0<=i<n }",
    ...     """
    ...     out[i] = 5
    ...     out[0] = 6
    ...     """)
    ... # Loopy prints the following before this exception:
    ... # While trying to find shape axis 0 of argument 'out', the following exception occurred:
    Traceback (most recent call last):
    ...
    StaticValueFindingError: a static maximum was not found for PwAff '[n] -> { [(1)] : n <= 1; [(n)] : n >= 2 }'

The problem is that loopy cannot find a simple, universally valid expression
for the length of *out* in this case. Notice how the kernel accesses both the
*i*-th and the first element of out.  The set notation at the end of the error
message summarizes its best attempt:

* If n=1, then out has size 1.
* If n>=2, then out has size n.
* If n<=0, then out has size 1.

Sure, some of these cases could be coalesced, but that's beside the point.
Loopy does not know that non-positive values of *n* make no sense. It needs to
be told in order for the error to disappear--note the *assumptions* argument:

.. doctest::

    >>> knl = lp.make_kernel(
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

    >>> knl = lp.make_kernel(
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

    >>> evt, (out,) = knl(queue, a=a_mat_dev)
    Traceback (most recent call last):
    ...
    WriteRaceConditionWarning: in kernel transpose: instruction 'a_fetch_rule' looks invalid: it assigns to indices based on local IDs, but its temporary 'a_fetch' cannot be made local because a write race across the iname(s) 'j_inner' would emerge. (Do you need to add an extra iname to your prefetch?) (add 'write_race_local(a_fetch_rule)' to silenced_warnings kernel argument to disable)

When we ask to see the code, the issue becomes apparent:

.. doctest::

    >>> knl = lp.set_options(knl, "write_cl")
    >>> from warnings import catch_warnings
    >>> with catch_warnings():
    ...     filterwarnings("always", category=lp.LoopyWarning)
    ...     evt, (out,) = knl(queue, a=a_mat_dev)
    #define lid(N) ((int) get_local_id(N))
    #define gid(N) ((int) get_group_id(N))
    <BLANKLINE>
    __kernel void __attribute__ ((reqd_work_group_size(16, 16, 1))) transpose(__global float const *restrict a, int const n, __global float *restrict out)
    {
      float a_fetch[16];
    <BLANKLINE>
      ...
          a_fetch[lid(0)] = a[n * (16 * gid(1) + lid(0)) + 16 * gid(0) + lid(1)];
      ...
          out[n * (16 * gid(0) + lid(1)) + 16 * gid(1) + lid(0)] = a_fetch[lid(0)];
      ...
    }

Loopy has a 2D workgroup to use for prefetching of a 1D array. When it
considers making *a_fetch* ``local`` (in the OpenCL memory sense of the word)
to make use of parallelism in prefetching, it discovers that a write race
across the remaining axis of the workgroup would emerge.

TODO

.. }}}

Obtaining Performance Statistics
--------------------------------

.. {{{

Operations, array access, and barriers can all be counted, which may facilitate
performance prediction and optimization of a :mod:`loopy` kernel.

.. note::

    The functions used in the following examples may produce warnings. If you have
    already made the filterwarnings and catch_warnings calls used in the examples
    above, you may need to reset these before continuing:

    .. doctest::

        >>> from warnings import resetwarnings
        >>> resetwarnings()

Counting operations
~~~~~~~~~~~~~~~~~~~

:func:`loopy.get_op_poly` provides information on the number and type of operations
being performed in a kernel. To demonstrate this, we'll create an example kernel
that performs several operations on arrays containing different types of data:

.. doctest::

    >>> knl = lp.make_kernel(
    ...     "[n,m,l] -> {[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<l}",
    ...     """
    ...     c[i, j, k] = a[i,j,k]*b[i,j,k]/3.0+a[i,j,k]
    ...     e[i, k] = g[i,k]*(2+h[i,k+1])
    ...     """)
    >>> knl = lp.add_and_infer_dtypes(knl,
    ...     dict(a=np.float32, b=np.float32, g=np.float64, h=np.float64))

Note that loopy will infer the data types for arrays c and e from the
information provided. Now we will count the operations:

.. doctest::

    >>> from loopy.statistics import get_op_poly
    >>> op_map = get_op_poly(knl)

:func:`loopy.get_op_poly` returns a mapping of **{(** :class:`numpy.dtype` **,** 
:class:`string` **)** **:** :class:`islpy.PwQPolynomial` **}**. The 
:class:`islpy.PwQPolynomial` holds the number of operations for the type specified 
in the key (in terms of the :class:`loopy.LoopKernel` *inames*). We'll print this 
map now:

.. doctest::

    >>> print(lp.stringify_stats_mapping(op_map))
    (dtype('float32'), 'add') : [n, m, l] -> { n * m * l : n > 0 and m > 0 and l > 0 }
    (dtype('float32'), 'div') : [n, m, l] -> { n * m * l : n > 0 and m > 0 and l > 0 }
    (dtype('float32'), 'mul') : [n, m, l] -> { n * m * l : n > 0 and m > 0 and l > 0 }
    (dtype('float64'), 'add') : [n, m, l] -> { n * m : n > 0 and m > 0 and l > 0 }
    (dtype('float64'), 'mul') : [n, m, l] -> { n * m : n > 0 and m > 0 and l > 0 }
    (dtype('int32'), 'add') : [n, m, l] -> { n * m : n > 0 and m > 0 and l > 0 }
    <BLANKLINE>

We can evaluate these polynomials using :func:`islpy.eval_with_dict`:

.. doctest::

    >>> param_dict = {'n': 256, 'm': 256, 'l': 8}
    >>> f32add = op_map[(np.dtype(np.float32), 'add')].eval_with_dict(param_dict)
    >>> f32div = op_map[(np.dtype(np.float32), 'div')].eval_with_dict(param_dict)
    >>> f32mul = op_map[(np.dtype(np.float32), 'mul')].eval_with_dict(param_dict)
    >>> f64add = op_map[(np.dtype(np.float64), 'add')].eval_with_dict(param_dict)
    >>> f64mul = op_map[(np.dtype(np.float64), 'mul')].eval_with_dict(param_dict)
    >>> i32add = op_map[(np.dtype(np.int32), 'add')].eval_with_dict(param_dict)
    >>> print("%i\n%i\n%i\n%i\n%i\n%i" % 
    ...     (f32add, f32div, f32mul, f64add, f64mul, i32add))
    524288
    524288
    524288
    65536
    65536
    65536

Counting array accesses
~~~~~~~~~~~~~~~~~~~~~~~

:func:`loopy.get_gmem_access_poly` provides information on the number and type of
array loads and stores being performed in a kernel. To demonstrate this, we'll
continue using the kernel from the previous example:

.. doctest::

    >>> from loopy.statistics import get_gmem_access_poly
    >>> load_store_map = get_gmem_access_poly(knl)
    >>> print(lp.stringify_stats_mapping(load_store_map))
    (dtype('float32'), 'uniform', 'load') : [n, m, l] -> { 3 * n * m * l : n > 0 and m > 0 and l > 0 }
    (dtype('float32'), 'uniform', 'store') : [n, m, l] -> { n * m * l : n > 0 and m > 0 and l > 0 }
    (dtype('float64'), 'uniform', 'load') : [n, m, l] -> { 2 * n * m : n > 0 and m > 0 and l > 0 }
    (dtype('float64'), 'uniform', 'store') : [n, m, l] -> { n * m : n > 0 and m > 0 and l > 0 }
    <BLANKLINE>

:func:`loopy.get_gmem_access_poly` returns a mapping of **{(**
:class:`numpy.dtype` **,** :class:`string` **,** :class:`string` **)**
**:** :class:`islpy.PwQPolynomial` **}**.

- The :class:`numpy.dtype` specifies the type of the data being accessed.

- The first string in the map key specifies the DRAM access type as *consecutive*,
  *nonconsecutive*, or *uniform*. *Consecutive* memory accesses occur when
  consecutive threads access consecutive array elements in memory, *nonconsecutive*
  accesses occur when consecutive threads access nonconsecutive array elements in
  memory, and *uniform* accesses occur when consecutive threads access the *same*
  element in memory.

- The second string in the map key specifies the DRAM access type as a *load*, or a
  *store*.

- The :class:`islpy.PwQPolynomial` holds the number of DRAM accesses with the
  characteristics specified in the key (in terms of the :class:`loopy.LoopKernel`
  *inames*).

We can evaluate these polynomials using :func:`islpy.eval_with_dict`:

.. doctest::

    >>> f64ld = load_store_map[(np.dtype(np.float64), "uniform", "load")
    ...     ].eval_with_dict(param_dict)
    >>> f64st = load_store_map[(np.dtype(np.float64), "uniform", "store")
    ...     ].eval_with_dict(param_dict)
    >>> f32ld = load_store_map[(np.dtype(np.float32), "uniform", "load")
    ...     ].eval_with_dict(param_dict)
    >>> f32st = load_store_map[(np.dtype(np.float32), "uniform", "store")
    ...     ].eval_with_dict(param_dict)
    >>> print("f32 load: %i\nf32 store: %i\nf64 load: %i\nf64 store: %i" %
    ...     (f32ld, f32st, f64ld, f64st))
    f32 load: 1572864
    f32 store: 524288
    f64 load: 131072
    f64 store: 65536

~~~~~~~~~~~

Since we have not tagged any of the inames or parallelized the kernel across threads
(which would have produced iname tags), :func:`loopy.get_gmem_access_poly` considers
the array accesses *uniform*. Now we'll parallelize the kernel and count the array
accesses again. The resulting :class:`islpy.PwQPolynomial` will be more complicated
this time, so we'll print the mapping manually to make it more legible:

.. doctest::

    >>> knl_consec = lp.split_iname(knl, "k", 128, outer_tag="l.1", inner_tag="l.0")
    >>> load_store_map = get_gmem_access_poly(knl_consec)
    >>> for key in sorted(load_store_map.keys(), key=lambda k: str(k)):
    ...     print("%s :\n%s\n" % (key, load_store_map[key]))
    (dtype('float32'), 'consecutive', 'load') :
    [n, m, l] -> { ... }
    <BLANKLINE>
    (dtype('float32'), 'consecutive', 'store') :
    [n, m, l] -> { ... }
    <BLANKLINE>
    (dtype('float64'), 'consecutive', 'load') :
    [n, m, l] -> { ... }
    <BLANKLINE>
    (dtype('float64'), 'consecutive', 'store') :
    [n, m, l] -> { ... }
    <BLANKLINE>


With this parallelization, consecutive threads will access consecutive array
elements in memory. The polynomials are a bit more complicated now due to the
parallelization, but when we evaluate them, we see that the total number of array
accesses has not changed:

.. doctest::

    >>> f64ld = load_store_map[(np.dtype(np.float64), "consecutive", "load")
    ...     ].eval_with_dict(param_dict)
    >>> f64st = load_store_map[(np.dtype(np.float64), "consecutive", "store")
    ...     ].eval_with_dict(param_dict)
    >>> f32ld = load_store_map[(np.dtype(np.float32), "consecutive", "load")
    ...     ].eval_with_dict(param_dict)
    >>> f32st = load_store_map[(np.dtype(np.float32), "consecutive", "store")
    ...     ].eval_with_dict(param_dict)
    >>> print("f32 load: %i\nf32 store: %i\nf64 load: %i\nf64 store: %i" %
    ...     (f32ld, f32st, f64ld, f64st))
    f32 load: 1572864
    f32 store: 524288
    f64 load: 131072
    f64 store: 65536

~~~~~~~~~~~

To produce *nonconsecutive* array accesses, we'll switch the inner and outer tags in
our parallelization of the kernel:

.. doctest::

    >>> knl_nonconsec = lp.split_iname(knl, "k", 128, outer_tag="l.0", inner_tag="l.1")
    >>> load_store_map = get_gmem_access_poly(knl_nonconsec)
    >>> for key in sorted(load_store_map.keys(), key=lambda k: str(k)):
    ...     print("%s :\n%s\n" % (key, load_store_map[key]))
    (dtype('float32'), 'nonconsecutive', 'load') :
    [n, m, l] -> { ... }
    <BLANKLINE>
    (dtype('float32'), 'nonconsecutive', 'store') :
    [n, m, l] -> { ... }
    <BLANKLINE>
    (dtype('float64'), 'nonconsecutive', 'load') :
    [n, m, l] -> { ... }
    <BLANKLINE>
    (dtype('float64'), 'nonconsecutive', 'store') :
    [n, m, l] -> { ... }
    <BLANKLINE>


With this parallelization, consecutive threads will access *nonconsecutive* array
elements in memory. The total number of array accesses has not changed:

.. doctest::

    >>> f64ld = load_store_map[
    ...     (np.dtype(np.float64), "nonconsecutive", "load")
    ...     ].eval_with_dict(param_dict)
    >>> f64st = load_store_map[
    ...     (np.dtype(np.float64), "nonconsecutive", "store")
    ...     ].eval_with_dict(param_dict)
    >>> f32ld = load_store_map[
    ...     (np.dtype(np.float32), "nonconsecutive", "load")
    ...     ].eval_with_dict(param_dict)
    >>> f32st = load_store_map[
    ...     (np.dtype(np.float32), "nonconsecutive", "store")
    ...     ].eval_with_dict(param_dict)
    >>> print("f32 load: %i\nf32 store: %i\nf64 load: %i\nf64 store: %i" %
    ...     (f32ld, f32st, f64ld, f64st))
    f32 load: 1572864
    f32 store: 524288
    f64 load: 131072
    f64 store: 65536

Counting synchronization events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`loopy.get_synchronization_poly` counts the number of synchronization
events per **thread** in a kernel. First, we'll call this function on the
kernel from the previous example:

.. doctest::

    >>> from loopy.statistics import get_synchronization_poly
    >>> barrier_poly = get_synchronization_poly(knl)
    >>> print(lp.stringify_stats_mapping(barrier_poly))
    kernel_launch : { 1 }
    <BLANKLINE>

We can evaluate this polynomial using :func:`islpy.eval_with_dict`:

.. doctest::

    >>> launch_count = barrier_poly["kernel_launch"].eval_with_dict(param_dict)
    >>> print("Kernel launch count: %s" % launch_count)
    Kernel launch count: 1

Now to make things more interesting, we'll create a kernel with barriers:

.. doctest::

    >>> knl = lp.make_kernel(
    ...     "[] -> {[i,k,j]: 0<=i<50 and 1<=k<98 and 0<=j<10}",
    ...     [
    ...     """
    ...     c[i,j,k] = 2*a[i,j,k]
    ...     e[i,j,k] = c[i,j,k+1]+c[i,j,k-1]
    ...     """
    ...     ], [
    ...     lp.TemporaryVariable("c", lp.auto, shape=(50, 10, 99)),
    ...     "..."
    ...     ])
    >>> knl = lp.add_and_infer_dtypes(knl, dict(a=np.int32))
    >>> knl = lp.split_iname(knl, "k", 128, outer_tag="g.0", inner_tag="l.0")
    >>> code, _ = lp.generate_code(lp.preprocess_kernel(knl))
    >>> print(code)
    #define lid(N) ((int) get_local_id(N))
    #define gid(N) ((int) get_group_id(N))
    <BLANKLINE>
    __kernel void __attribute__ ((reqd_work_group_size(97, 1, 1))) loopy_kernel(__global int const *restrict a, __global int *restrict e)
    {
      __local int c[50 * 10 * 99];
    <BLANKLINE>
      for (int j = 0; j <= 9; ++j)
        for (int i = 0; i <= 49; ++i)
        {
          barrier(CLK_LOCAL_MEM_FENCE) /* for c (insn rev-depends on insn_0) */;
          c[990 * i + 99 * j + lid(0) + 1] = 2 * a[980 * i + 98 * j + lid(0) + 1];
          barrier(CLK_LOCAL_MEM_FENCE) /* for c (insn_0 depends on insn) */;
          e[980 * i + 98 * j + lid(0) + 1] = c[990 * i + 99 * j + 1 + lid(0) + 1] + c[990 * i + 99 * j + -1 + lid(0) + 1];
        }
    }


In this kernel, when a thread performs the second instruction it uses data produced
by *different* threads during the first instruction. Because of this, barriers are
required for correct execution, so loopy inserts them. Now we'll count the barriers
using :func:`loopy.get_barrier_poly`:

.. doctest::

    >>> sync_map = lp.get_synchronization_poly(knl)
    >>> print(lp.stringify_stats_mapping(sync_map))
    barrier_local : { 1000 }
    kernel_launch : { 1 }
    <BLANKLINE>

Based on the kernel code printed above, we would expect each thread to encounter
50x10x2 barriers, which matches the result from :func:`loopy.get_barrier_poly`. In
this case, the number of barriers does not depend on any inames, so we can pass an
empty dictionary to :func:`islpy.eval_with_dict`.

.. }}}

.. vim: tw=75:spell:foldmethod=marker

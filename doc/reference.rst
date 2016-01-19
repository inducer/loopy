.. _reference:

Reference Guide
===============

.. module:: loopy
.. moduleauthor:: Andreas Kloeckner <inform@tiker.net>

This guide defines all functionality exposed by loopy. If you would like
a more gentle introduction, you may consider reading the example-based
:ref:`tutorial` instead.

.. _inames:

Domain Tree
-----------



Inames
^^^^^^

Loops are (by default) entered exactly once. This is necessary to preserve
dependency semantics--otherwise e.g. a fetch could happen inside one loop nest,
and then the instruction using that fetch could be inside a wholly different
loop nest.

Instructions
------------

Expressions
^^^^^^^^^^^

Loopy's expressions are a slight superset of the expressions supported by
:mod:`pymbolic`.

* `if`
* `reductions`
    * duplication of reduction inames
* complex-valued arithmetic
* tagging of array access and substitution rule use ("$")
* ``indexof``, ``indexof_vec``

.. _types:

Specifying Types
----------------

:mod:`loopy` uses the same type system as :mod:`numpy`. (See
:class:`numpy.dtype`) It also uses :mod:`pyopencl` for a registry of
user-defined types and their C equivalents. See :func:`pyopencl.tools.get_or_register_dtype`
and related functions.

For a string representation of types, all numpy types (e.g. ``float32`` etc.)
are accepted, in addition to what is registered in :mod:`pyopencl`.

.. _iname-tags:

Iname Implementation Tags
-------------------------

=============================== ====================================================
Tag                             Meaning
=============================== ====================================================
``None`` | ``"for"``            Sequential loop
``"l.N"``                       Local (intra-group) axis N ("local")
``"g.N"``                       Group-number axis N ("group")
``"unr"``                       Unroll
``"ilp"`` | ``"ilp.unr"``       Unroll using instruction-level parallelism
``"ilp.seq"``                   Realize parallel iname as innermost loop
``"like.INAME"``                Can be used when tagging inames to tag like another
``"unused.g"`` | ``"unused.l"`` Can be to tag as the next unused group/local axis
=============================== ====================================================

(Throughout this table, `N` must be replaced by an actual, zero-based number.)

"ILP" does three things:

* Restricts loops to be innermost
* Duplicates reduction storage for any reductions nested around ILP usage
* Causes a loop (unrolled or not) to be opened/generated for each
  involved instruction

.. _data-dim-tags:

Data Axis Tags
--------------

Data axis tags specify how a multi-dimensional array (which is loopy's
main way of storing data) is represented in (linear, 1D) computer
memory. This storage format is given as a number of "tags", as listed
in the table below. Each axis of an array has a tag corresponding to it.
In the user interface, array dim tags are specified as a tuple of these
tags or a comma-separated string containing them, such as the following::

    c,vec,sep,c

The interpretation of these tags is order-dependent, they are read
from left to right.

===================== ====================================================
Tag                   Meaning
===================== ====================================================
``c``                 Nest current axis around the ones that follow
``f``                 Nest current axis inside the ones that follow
``N0`` ... ``N9``     Specify an explicit nesting level for this axis
``stride:EXPR``       A fixed stride
``sep``               Implement this axis by mapping to separate arrays
``vec``               Implement this axis as entries in a vector
===================== ====================================================

``sep`` and ``vec`` obviously require the number of entries
in the array along their respective axis to be known at code
generation time.

When the above speaks about 'nesting levels', this means that axes
"nested inside" others are "faster-moving" when viewed from linear
memory.

In addition, each tag may be followed by a question mark (``?``),
which indicates that if there are more dimension tags specified
than array axes present, that this axis should be omitted. Axes
with question marks are omitted in a left-first manner until the correct
number of dimension tags is achieved.

Some examples follow, all of which use a three-dimensional array of shape
*(3, M, 4)*. For simplicity, we assume that array entries have size one.

*   ``c,c,c``: The axes will have strides *(M*4, 4, 1)*,
    leading to a C-like / row-major layout.

*   ``f,f,f``: The axes will have strides *(1, 3, 3*M)*,
    leading to a Fortran-like / row-major layout.

*   ``sep,c,c``: The array will be mapped to three arrays of
    shape *(M, 4)*, each with strides *(4, 1)*.

*   ``c,c,vec``: The array will be mapped to an array of
    ``float4`` vectors, with (``float4``-based) strides of
    *(M, 1)*.

*   ``N1,N0,N2``: The axes will have strides *(M, 1, 3*M)*.

.. _creating-kernels:

Creating Kernels
----------------

.. autoclass:: auto

.. _arguments:

Arguments
^^^^^^^^^

.. autoclass:: ValueArg
    :members:
    :undoc-members:

.. autoclass:: GlobalArg
    :members:
    :undoc-members:

.. autoclass:: ConstantArg
    :members:
    :undoc-members:

.. autoclass:: ImageArg
    :members:
    :undoc-members:

.. _temporaries:

Loop domains
^^^^^^^^^^^^

TODO: Explain the domain tree

.. _isl-syntax:

ISL syntax
~~~~~~~~~~

The general syntax of an ISL set is the following::

    {[VARIABLES]: CONDITIONS}

``VARIABLES`` is a simple list of identifiers representing loop indices,
or, as loopy calls them, inames. Example::

    {[i, j, k]: CONDITIONS}

The following constructs are supported for ``CONDITIONS``:

* Simple conditions: ``i <= 15``, ``i>0``

* Conjunctions: ``i > 0 and i <= 15``

* Two-sided conditions: ``0 < i <= 15`` (equivalent to the previous
  example)

* Identical conditions on multiple variables:
  ``0 < i,j <= 15``

* Equality constraints: ``i = j*3`` (**Note:** ``=``, not ``==``.)

* Modulo: ``i mod 3 = 0``

* Existential quantifiers: ``(exists l: i = 3*l)`` (equivalent to the
  previous example)

Examples of constructs that are **not** allowed:

* Multiplication by non-constants: ``j*k``

* Disjunction: ``(i=1) or (i=5)``
  (**Note:** This may be added in a future version of loopy.
  For now, loop domains have to be convex.)

Temporary Variables
^^^^^^^^^^^^^^^^^^^

Temporary variables model OpenCL's ``private`` and ``local`` address spaces. Both
have the lifetime of a kernel invocation.

.. autoclass:: TemporaryVariable
    :members:
    :undoc-members:

Instructions
^^^^^^^^^^^^

.. autoclass:: UniqueName

.. autoclass:: InstructionBase

.. _assignments:

Assignments
~~~~~~~~~~~

The general syntax of an instruction is a simple assignment::

    LHS[i,j,k] = EXPRESSION

Several extensions of this syntax are defined, as discussed below.  They
may be combined freely.

You can also use an instruction to declare a new temporary variable. (See
:ref:`temporaries`.) See :ref:`types` for what types are acceptable. If the
``LHS`` has a subscript, bounds on the indices are inferred (which must be
constants at the time of kernel creation) and the declared temporary is
created as an array. Instructions declaring temporaries have the following
form::

    <temp_var_type> LHS[i,j,k] = EXPRESSION

You can also create a temporary and ask loopy to determine its type
automatically. This uses the following syntax::

    <> LHS[i,j,k] = EXPRESSION

Lastly, each instruction may optionally have a number of attributes
specified, using the following format::

    LHS[i,j,k] = EXPRESSION {attr1,attr2=value1:value2}

These are usually key-value pairs. The following attributes are recognized:

* ``id=value`` sets the instruction's identifier to ``value``. ``value``
  must be unique within the kernel. This identifier is used to refer to the
  instruction after it has been created, such as from ``dep`` attributes
  (see below) or from :mod:`context matches <loopy.context_matching>`.

* ``id_prefix=value`` also sets the instruction's identifier, however
  uniqueness is ensured by loopy itself, by appending further components
  (often numbers) to the given ``id_prefix``.

* ``inames=i:j:k`` forces the instruction to reside within the loops over
  :ref:`inames` ``i``, ``j`` and ``k`` (and only those).

  .. note::

      The default for the inames that the instruction depends on is
      the inames used in the instruction itself plus the common
      subset of inames shared by writers of all variables read by the
      instruction.

      You can add a plus sign ("``+``") to the front of this option
      value to indicate that you would like the inames you specify here
      to be in addition to the ones found by the heuristic described above.

* ``dup=i:j->j_new:k->k_new`` makes a copy of the inames ``i``, ``j``, and
  ``k``, with all the same domain constraints as the original inames.
  A new name of the copy of ``i`` will be automatically chosen, whereas
  the new name of ``j`` will be ``j_new``, and the new name of ``k`` will
  be ``k_new``.

  This is a shortcut for calling :func:`loopy.duplicate_inames` later
  (once the kernel is created).

* ``dep=id1:id2`` creates a dependency of this instruction on the
  instructions with identifiers ``id1`` and ``id2``. The meaning of this
  dependency is that the code generated for this instruction is required to
  appear textually after all of these dependees' generated code.

  Identifiers here are allowed to be wildcards as defined by
  the Python module :mod:`fnmatchcase`. This is helpful in conjunction
  with ``id_prefix``.

  .. note::

      Since specifying all possible dependencies is cumbersome and
      error-prone, :mod:`loopy` employs a heuristic to automatically find
      dependencies. Specifically, :mod:`loopy` will automatically add
      a dependency to an instruction reading a variable if there is
      exactly one instruction writing that variable. ("Variable" here may
      mean either temporary variable or kernel argument.)

      If each variable in a kernel is only written once, then this
      heuristic should be able to compute all required dependencies.

      Conversely, if a variable is written by two different instructions,
      all ordering around that variable needs to be specified explicitly.
      It is recommended to use :func:`get_dot_dependency_graph` to
      visualize the dependency graph of possible orderings.

      You may use a leading asterisk ("``*``") to turn off the single-writer
      heuristic and indicate that the specified list of dependencies is
      exhaustive.

* ``priority=integer`` sets the instructions priority to the value
  ``integer``. Instructions with higher priority will be scheduled sooner,
  if possible. Note that the scheduler may still schedule a lower-priority
  instruction ahead of a higher-priority one if loop orders or dependencies
  require it.

* ``if=variable1:variable2`` Only execute this instruction if all condition
  variables (which must be scalar variables) evaluate to ``true`` (as
  defined by C).

* ``tags=tag1:tag2`` Apply tags to this instruction that can then be used
  for :ref:`context-matching`.

* ``groups=group1:group2`` Make this instruction part of the given
  instruction groups. See :class:`InstructionBase.groups`.

* ``conflicts_grp=group1:group2`` Make this instruction conflict with the
  given instruction groups. See
  :class:`InstructionBase.conflicts_with_groups`.

Assignment instructions are expressed as instances of the following class:

.. autoclass:: ExpressionInstruction

.. _expression-syntax:

Expression Syntax
~~~~~~~~~~~~~~~~~

TODO: Functions
TODO: Reductions

C Block Instructions
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CInstruction

.. _substitution-rule:

Substitution Rules
^^^^^^^^^^^^^^^^^^

Syntax of a substitution rule::

    rule_name(arg1, arg2) := EXPRESSION

Kernels
^^^^^^^

.. class:: LoopKernel

Do not create :class:`LoopKernel` objects directly. Instead, use the following
function, which is responsible for creating kernels:

.. autofunction:: make_kernel

.. autofunction:: parse_fortran

.. autofunction:: parse_transformed_fortran

.. autofunction:: make_copy_kernel

.. autofunction:: fuse_kernels

.. autofunction:: c_preprocess

Transforming Kernels
--------------------

.. _context-matching:

Matching contexts
^^^^^^^^^^^^^^^^^

TODO: Matching instruction tags

.. automodule:: loopy.context_matching

.. autofunction:: parse_match

.. autofunction:: parse_stack_match

.. currentmodule:: loopy

Wrangling inames
^^^^^^^^^^^^^^^^

.. autofunction:: split_iname

.. autofunction:: chunk_iname

.. autofunction:: join_inames

.. autofunction:: tag_inames

.. autofunction:: duplicate_inames

.. undocumented .. autofunction:: link_inames

.. autofunction:: rename_iname

.. autofunction:: remove_unused_inames

.. autofunction:: set_loop_priority

.. autofunction:: split_reduction_inward

.. autofunction:: split_reduction_outward

.. autofunction:: affine_map_inames

.. autofunction:: realize_ilp

.. autofunction:: find_unused_axis_tag

Dealing with Parameters
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: fix_parameters

.. autofunction:: assume

Dealing with Substitution Rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: extract_subst

.. autofunction:: assignment_to_subst

.. autofunction:: expand_subst

.. autofunction:: find_rules_matching

.. autofunction:: find_one_rule_matching

Caching, Precomputation and Prefetching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: precompute

.. autofunction:: add_prefetch

.. autofunction:: buffer_array

.. autofunction:: alias_temporaries

Influencing data access
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: change_arg_to_image

.. autofunction:: tag_data_axes

.. autofunction:: remove_unused_arguments

.. autofunction:: set_array_dim_names

Padding
^^^^^^^

.. autofunction:: split_array_dim

.. autofunction:: find_padding_multiple

.. autofunction:: add_padding

Manipulating Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: set_instruction_priority

.. autofunction:: add_dependency

.. autofunction:: remove_instructions

.. autofunction:: tag_instructions

Library interface
^^^^^^^^^^^^^^^^^

.. autofunction:: register_reduction_parser

.. autofunction:: register_preamble_generators

.. autofunction:: register_symbol_manglers

.. autofunction:: register_function_manglers

Arguments
^^^^^^^^^

.. autofunction:: set_argument_order

.. autofunction:: add_dtypes

.. autofunction:: infer_unknown_types

.. autofunction:: add_and_infer_dtypes

Batching
^^^^^^^^

.. autofunction:: to_batched

Finishing up
^^^^^^^^^^^^

.. autofunction:: generate_loop_schedules

.. autofunction:: get_one_scheduled_kernel

.. autofunction:: generate_code

Running
-------

.. autoclass:: CompiledKernel

Automatic Testing
-----------------

.. autofunction:: auto_test_vs_ref

Troubleshooting
---------------

Printing :class:`LoopKernel` objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're confused about things loopy is referring to in an error message or
about the current state of the :class:`LoopKernel` you are transforming, the
following always works::

    print kernel

(And it yields a human-readable--albeit terse--representation of *kernel*.)

.. autofunction:: preprocess_kernel

.. autofunction:: get_dot_dependency_graph

.. autofunction:: show_dependency_graph

Options
-------

.. autoclass:: Options

.. autofunction:: set_options

Controlling caching
-------------------

.. autofunction:: set_caching_enabled

.. autoclass:: CacheMode

Obtaining Kernel Statistics
---------------------------

.. autofunction:: get_op_poly

.. autofunction:: get_gmem_access_poly

.. autofunction:: sum_mem_access_to_bytes

.. autofunction:: get_barrier_poly

.. vim: tw=75:spell

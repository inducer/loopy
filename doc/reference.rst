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

.. _types:

Specifying Types
----------------

:mod:`loopy` uses the same type system as :mod:`numpy`. (See
:class:`numpy.dtype`) It also uses :mod:`pyopencl` for a registry of
user-defined types and their C equivalents. See :func:`pyopencl.tools.get_or_register_dtype`
and related functions.

For a string representation of types, all numpy types (e.g. ``float32`` etc.)
are accepted, in addition to what is registered in :mod:`pyopencl`.

.. _tags:

Iname Implementation Tags
-------------------------

===================== ====================================================
Tag                   Meaning
===================== ====================================================
`None` | `"for"`      Sequential loop
`"l.N"`               Local (intra-group) axis N
`"g.N"`               Group-number axis N
`"unr"`               Unroll
`"ilp"` | `"ilp.unr"` Unroll using instruction-level parallelism
`"ilp.seq"`           Realize parallel iname as innermost loop
===================== ====================================================

.. "l.auto" intentionally undocumented

(Throughout this table, `N` must be replaced by an actual number.)

"ILP" does three things:

* Restricts loops to be innermost
* Duplicates reduction storage for any reductions nested around ILP usage
* Causes a loop (unrolled or not) to be opened/generated for each
  involved instruction

.. _automatic-axes:

Automatic Axis Assignment
^^^^^^^^^^^^^^^^^^^^^^^^^

Automatic local axes are chosen as follows:

#. For each instruction containing `"l.auto"` inames:
    #. Find the lowest-numbered unused axis. If none exists,
        use sequential unrolling instead.
    #. Find the iname that has the smallest stride in any global
        array access occurring in the instruction.
    #. Assign the low-stride iname to the available axis, splitting
        the iname if it is too long for the available axis size.

If you need different behavior, use :func:`tag_inames` and
:func:`split_iname` to change the assignment of `"l.auto"` axes
manually.

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

.. autoclass:: ExpressionInstruction


.. _expression-syntax:

Expression Syntax
~~~~~~~~~~~~~~~~~

TODO: Functions
TODO: Reductions

C Block Instructions
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CInstruction

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

Transforming Kernels
--------------------

Matching contexts
^^^^^^^^^^^^^^^^^

.. automodule:: loopy.context_matching

.. autofunction:: parse_id_match

.. autofunction:: parse_stack_match

.. currentmodule:: loopy

Wrangling inames
^^^^^^^^^^^^^^^^

.. autofunction:: split_iname

.. autofunction:: join_inames

.. autofunction:: tag_inames

.. autofunction:: duplicate_inames

.. undocumented .. autofunction:: link_inames

.. autofunction:: rename_iname

.. autofunction:: remove_unused_inames

.. autofunction:: set_loop_priority

.. autofunction:: split_reduction_inward

.. autofunction:: split_reduction_outward

Dealing with Parameters
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: fix_parameters

.. autofunction:: assume

Dealing with Substitution Rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: extract_subst

.. autofunction:: expand_subst

Caching, Precomputation and Prefetching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: precompute

.. autofunction:: add_prefetch

Influencing data access
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: change_arg_to_image

.. autofunction:: tag_data_axes

Padding
^^^^^^^

.. autofunction:: split_arg_axis

.. autofunction:: find_padding_multiple

.. autofunction:: add_padding

Manipulating Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: set_instruction_priority

.. autofunction:: add_dependency

Library interface
^^^^^^^^^^^^^^^^^

.. autofunction:: register_reduction_parser

.. autofunction:: register_preamble_generators

.. autofunction:: register_symbol_manglers

.. autofunction:: register_function_manglers

Argument types
^^^^^^^^^^^^^^

.. autofunction:: add_dtypes

.. autofunction:: infer_unknown_types

.. autofunction:: add_and_infer_dtypes

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

.. vim: tw=75:spell

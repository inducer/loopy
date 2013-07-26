Reference Guide
===============

.. module:: loopy
.. moduleauthor:: Andreas Kloeckner <inform@tiker.net>

This guide defines all functionality exposed by loopy. If you would like
a more gentle introduction, you may consider reading the example-based
guide :ref:`guide` instead.

.. _inames:

Inames
------

Loops are (by default) entered exactly once. This is necessary to preserve
depdency semantics--otherwise e.g. a fetch could happen inside one loop nest,
and then the instruction using that fetch could be inside a wholly different
loop nest.

Integer Domain
--------------

Expressions
-----------

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

Tags
----

===================== ====================================================
Tag                   Meaning
===================== ====================================================
`None` | `"for"`      Sequential loop
`"l.N"`               Local (intra-group) axis N
`"l.auto"`            Automatically chosen local (intra-group) axis
`"g.N"`               Group-number axis N
`"unr"`               Plain unrolling
`"ilp"` | `"ilp.unr"` Unroll using instruction-level parallelism
`"ilp.seq"`           Realize parallel iname as innermost loop
===================== ====================================================

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

Temporary Variables
^^^^^^^^^^^^^^^^^^^

Temporary variables model OpenCL's ``private`` and ``local`` address spaces. Both
have the lifetime of a kernel invocation.

.. autoclass:: TemporaryVariable
    :members:
    :undoc-members:

Instructions
^^^^^^^^^^^^

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
  :ref:`inames` ``i``, ``j`` and ``k``.

* ``dep=id1:id2`` creates a dependency of this instruction on the
  instructions with identifiers ``id1`` and ``id2``. This requires that the
  code generated for this instruction appears textually after both of these
  instructions' generated code.

  .. note::

      If this is not specified, :mod:`loopy` will automatically add
      depdencies of reading instructions on writing instructions *if and
      only if* there is exactly one writing instruction for the written
      variable (temporary or argument).

* ``priority=integer`` sets the instructions priority to the value
  ``integer``. Instructions with higher priority will be scheduled sooner,
  if possible. Note that the scheduler may still schedule a lower-priority
  instruction ahead of a higher-priority one if loop orders or dependencies
  require it.

* ``if=variable1:variable2`` Only execute this instruction if all condition
  variables (which must be scalar variables) evaluate to ``true`` (as
  defined by C).

.. autoclass:: ExpressionInstruction

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

Dealing with Substitution Rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: extract_subst

.. autofunction:: expand_subst

Caching, Precomputation and Prefetching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: precompute

.. autofunction:: add_prefetch

    Uses :func:`extract_subst` and :func:`precompute`.

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

.. autofunction:: add_argument_dtypes

.. autofunction:: infer_unknown_types

.. autofunction:: add_and_infer_argument_dtypes

Finishing up
^^^^^^^^^^^^

.. autofunction:: generate_loop_schedules

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

Flags
-----

.. autoclass:: LoopyFlags

.. vim: tw=75

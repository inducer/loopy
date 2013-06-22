Reference Guide
===============

.. module:: loopy
.. moduleauthor:: Andreas Kloeckner <inform@tiker.net>

This guide defines all functionality exposed by loopy. If you would like
a more gentle introduction, you may consider reading the example-based
guide :ref:`guide` instead.

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

Assignments and Substitution Rules
----------------------------------

Syntax of an instruction::

    label: [i,j|k,l] <float32> lhs[i,j,k] = EXPRESSION : dep_label, dep_label_2

The above example illustrates all the parts that are allowed in loo.py's
instruction syntax. All of these except for `lhs` and `EXPRESSION` are
optional.

* `label` is a unique identifier for this instruction, enabling you to
  refer back to the instruction uniquely during further transformation
  as well as specifying ordering dependencies.

* `dep_label,dep_label_2` are dependencies of the current instruction.
  Loo.py will enforce that the instructions marked with these labels
  are scheduled before this instruction.

* `<float32>` declares `lhs` as a temporary variable, with shape given
  by the ranges of the `lhs` subscripts. (Note that in this case, the
  `lhs` subscripts must be pure inames, not expressions, for now.)
  Instead of a concrete type, an empty set of angle brackets `<>` may be
  given to indicate that type inference should figure out the type of the
  temporary.

* `[i,j|k,l]` specifies the inames within which this instruction is run.
  Independent copies of the inames `k` and `l` will be made for this
  instruction.

Syntax of an substitution rule::

    rule_name(arg1, arg2) := EXPRESSION

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

Temporary Variables
^^^^^^^^^^^^^^^^^^^

.. autoclass:: TemporaryVariable
    :members:
    :undoc-members:

Substitution rules
^^^^^^^^^^^^^^^^^^

.. autoclass:: SubstitutionRule

String sytnax: FIXME

Instructions
^^^^^^^^^^^^

.. autoclass:: ExpressionInstruction

.. autoclass:: CInstruction

String sytnax: FIXME

Kernels
^^^^^^^

.. class:: LoopKernel

Do not create :class:`LoopKernel` objects directly. Instead, use the following
function, which takes the same arguments, but does some extra post-processing.

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

.. autofunction:: link_inames

.. autofunction:: rename_iname

.. autofunction:: remove_unused_inames

.. autofunction:: set_loop_priority

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

.. autoclass:: InvocationFlags

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

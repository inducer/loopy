Reference Guide
===============

.. module:: loopy
.. moduleauthor:: Andreas Kloeckner <inform@tiker.net>

This guide defines all functionality exposed by loopy. If you would like
a more gentle introduction, you may consider reading the example-based
guide :ref:`guide` instead.

.. _tags:

Tags
----

===================== ====================================================
Tag                   Meaning
===================== ====================================================
`None` | `"for"`      Sequential loop
`"l.N"`               Local (intra-group) axis N
`"l.auto"`            Automatically chosen local (intra-group) axis
`"unr"`               Plain unrolling
`"ilp"`               Unroll using instruction-level parallelism
===================== ====================================================

(Throughout this table, `N` must be replaced by an actual number.)

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

If you need different behavior, use :func:`tag_dimensions` and
:func:`split_dimension` to change the assignment of `"l.auto"` axes
manually.

.. _creating-kernels:

Creating Kernels
----------------

.. _arguments:

Arguments
^^^^^^^^^

.. autoclass:: ScalarArg
    :members:
    :undoc-members:

.. autoclass:: ArrayArg
    :members:
    :undoc-members:

.. autoclass:: ConstantArrayArg
    :members:
    :undoc-members:

.. autoclass:: ImageArg
    :members:
    :undoc-members:

.. _syntax:

String Syntax
^^^^^^^^^^^^^

* Substitution rules

* Instructions

Kernels
^^^^^^^

.. autoclass:: LoopKernel

Do not create :class:`LoopKernel` objects directly. Instead, use the following
function, which takes the same arguments, but does some extra post-processing.

.. autofunction:: make_kernel

Wrangling dimensions
--------------------

.. autofunction:: split_dimension

.. autofunction:: join_dimensions

.. autofunction:: tag_dimensions

Dealing with Substitution Rules
-------------------------------

.. autofunction:: extract_subst

.. autofunction:: apply_subst

Precomputation and Prefetching
------------------------------

.. autofunction:: precompute

.. autofunction:: add_prefetch

    Uses :func:`extract_subst` and :func:`precompute`.

Finishing up
------------

.. autofunction:: generate_loop_schedules

.. autofunction:: check_kernels

.. autofunction:: generate_code

Automatic Testing
-----------------

.. autofunction:: auto_test_vs_seq

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

Investigating Scheduler Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

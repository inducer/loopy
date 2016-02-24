Reference: Other Functionality
==============================

Obtaining Kernel Performance Statistics
---------------------------------------

.. automodule:: loopy.statistics

Controlling caching
-------------------

.. autofunction:: set_caching_enabled

.. autoclass:: CacheMode

Running Kernels
---------------

In addition to simply calling kernels using :class:`LoopKernel.__call__`,
the following underlying functionality may be used:

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

    print(kernel)

(And it yields a human-readable--albeit terse--representation of *kernel*.)

.. autofunction:: get_dot_dependency_graph

.. autofunction:: show_dependency_graph


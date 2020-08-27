.. moduleauthor:: Andreas Kloeckner <inform@tiker.net>
.. module:: loopy

.. _creating-kernels:

Reference: Creating Kernels
===========================

From Loop Domains and Instructions
----------------------------------

.. autofunction:: make_kernel

From Fortran
------------

.. autofunction:: parse_fortran

.. autofunction:: parse_transformed_fortran

.. autofunction:: c_preprocess

From Other Kernels
------------------

.. autofunction:: fuse_kernels

To Copy between Data Formats
----------------------------

.. autofunction:: make_copy_kernel

.. automodule:: loopy.version

Checks
------
Before code generation phase starts a series of checks are performed.

.. automodule:: loopy.check

.. vim: tw=75:spell:fdm=marker

.. currentmodule:: loopy
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

Einstein summation convention kernels
-------------------------------------

.. autofunction:: make_einsum

.. automodule:: loopy.version

.. vim: tw=75:spell:fdm=marker

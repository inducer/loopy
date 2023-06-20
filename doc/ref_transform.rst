.. _reference-transform:

Reference: Transforming Kernels
===============================

Dealing with Parameters
-----------------------

.. automodule:: loopy.transform.parameter

Wrangling inames
----------------

.. automodule:: loopy.transform.iname

Dealing with Substitution Rules
-------------------------------

.. currentmodule:: loopy

.. autofunction:: extract_subst

.. autofunction:: assignment_to_subst

.. autofunction:: expand_subst

.. autofunction:: find_rules_matching

.. autofunction:: find_one_rule_matching

Caching, Precomputation and Prefetching
---------------------------------------

.. autofunction:: precompute

.. autofunction:: add_prefetch

.. autofunction:: buffer_array

.. autofunction:: alias_temporaries

Influencing data access
-----------------------

.. autofunction:: change_arg_to_image

.. autofunction:: tag_array_axes

.. autofunction:: remove_unused_arguments

.. autofunction:: set_array_axis_names

.. automodule:: loopy.transform.privatize

.. autofunction:: allocate_temporaries_for_base_storage

Padding Data
------------

.. autofunction:: split_array_axis

.. autofunction:: find_padding_multiple

.. autofunction:: add_padding

Manipulating Instructions
-------------------------

.. autofunction:: set_instruction_priority

.. autofunction:: add_dependency

.. autofunction:: remove_instructions

.. autofunction:: replace_instruction_ids

.. autofunction:: tag_instructions

.. autofunction:: add_nosync

.. autofunction:: add_barrier

Registering Library Routines
----------------------------

.. autofunction:: register_reduction_parser

.. autofunction:: register_preamble_generators

.. autofunction:: register_symbol_manglers

Modifying Arguments
-------------------

.. autofunction:: set_argument_order

.. autofunction:: add_dtypes

.. autofunction:: infer_unknown_types

.. autofunction:: add_and_infer_dtypes

.. autofunction:: rename_argument

.. autofunction:: set_temporary_address_space

Creating Batches of Operations
------------------------------

.. automodule:: loopy.transform.batch

Finishing up
------------

.. currentmodule:: loopy

.. autofunction:: preprocess_kernel

.. autofunction:: generate_loop_schedules

.. autofunction:: get_one_linearized_kernel

.. autofunction:: save_and_reload_temporaries

.. autoclass:: GeneratedProgram
.. autoclass:: CodeGenerationResult

.. autofunction:: generate_code_v2

.. autofunction:: generate_header

Setting options
---------------

.. autofunction:: set_options

.. _context-matching:

Matching contexts
-----------------

TODO: Matching instruction tags

.. automodule:: loopy.match

.. vim: tw=75:spell

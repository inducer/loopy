Welcome to loopy's documentation!
=================================

loopy is a code generator for array-based code in the OpenCL/CUDA execution
model. Here's a very simple example of how to double the entries of a vector
using loopy:

.. literalinclude:: ../examples/python/hello-loopy.py
   :end-before: ENDEXAMPLE

This example is included in the :mod:`loopy` distribution as
:download:`examples/python/hello-loopy.py <../examples/python/hello-loopy.py>`.

When you run this script, the following kernel is generated, compiled, and executed:

.. literalinclude:: ../examples/python/hello-loopy.cl
    :language: c

(See the full example for how to print the generated code.)

.. _static-binary:

Places on the web related to Loopy
----------------------------------

* `Python package index <https://pypi.org/project/loopy>`_ (download releases)
* `Github <https://github.com/inducer/loopy>`_ (get latest source code, file bugs)
* `Homepage <https://mathema.tician.de/software/loopy>`_

Table of Contents
-----------------

If you're only just learning about loopy, consider the following `paper
<https://arxiv.org/abs/1405.7470>`_ on loopy that may serve as a good
introduction.

Please check :ref:`installation` to get started.

.. toctree::
    :maxdepth: 2

    tutorial
    ref_creation
    ref_kernel
    ref_translation_unit
    ref_transform
    ref_call
    ref_other
    misc
    ref_internals
    🚀 Github <https://github.com/inducer/loopy>
    💾 Download Releases <https://pypi.org/project/loopy>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

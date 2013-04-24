Welcome to loopy's documentation!
=================================

.. note::
    Loo.py hasn't been released yet. What's documented here generally
    exists as code and has survived some light testing. So if you try
    it and it works for you, great. If not, please do make sure to shoot
    me a message.

loopy is a code generator for array-based code in the OpenCL/CUDA execution
model. Here's a very simple example of how to double the entries of a vector
using loopy:

.. literalinclude:: ../examples/hello-loopy.py

The following kernel is generated, compiled, and executed behind your back (and
also printed at the end):

.. literalinclude:: ../examples/hello-loopy.cl
    :language: c

This file is included in the :mod:`loopy` distribution as
:file:`examples/hello-loopy.py`.

.. toctree::
    :maxdepth: 2

    guide
    reference
    misc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


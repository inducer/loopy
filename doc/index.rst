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
   :end-before: ENDEXAMPLE

This example is included in the :mod:`loopy` distribution as
:download:`examples/hello-loopy.py <../examples/hello-loopy.py>`.

When you run this script, the following kernel is generated, compiled, and executed:

.. literalinclude:: ../examples/hello-loopy.cl
    :language: c

(See the full example for how to print the generated code.)

.. toctree::
    :maxdepth: 2

    tutorial
    reference
    misc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


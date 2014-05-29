Welcome to loopy's documentation!
=================================

.. note::

  As of May 28, 2014, :mod:`loopy` is release-ready as software, but this
  documentation is still somewhat of a work in progress. If you're OK with
  mildly incomplete documentation, please head over to :ref:`installation`.

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

Places on the web related to Loopy
----------------------------------

* `Python package index <http://pypi.python.org/pypi/loo.py>`_ (download releases) Note the extra '.' in the PyPI identifier!

* `Github <http://github.com/inducer/loopy>`_ (get latest source code, file bugs)
* `Wiki <http://wiki.tiker.net/Loopy>`_ (read installation tips, get examples, read FAQ)
* `Homepage <http://mathema.tician.de/software/loopy>`_

Table of Contents
-----------------

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

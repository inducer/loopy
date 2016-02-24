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

Want to try out loopy?
----------------------

There's no need to go through :ref:`installation` if you'd just like to get a
feel for what loopy is.  Instead, you may
`download a self-contained Linux binary <https://gitlab.tiker.net/inducer/loopy/builds/1989/artifacts/browse/build-helpers/>`_.
This is purposefully built on an ancient Linux distribution, so it should work
on most versions of Linux that are currently out there.

Once you have the binary, do the following::

    chmod +x ./loopy-centos6
    ./loopy-centos6 --target=opencl hello-loopy-lp.py
    ./loopy-centos6 --target=cuda hello-loopy-lp.py
    ./loopy-centos6 --target=ispc hello-loopy-lp.py

Grab the example here: :download:`examples/python/hello-loopy.py <../examples/python/hello-loopy-lp.py>`.

You may also donwload the most recent version by going to the `list of builds
<https://gitlab.tiker.net/inducer/loopy/builds>`_, clicking on the newest one
of type "CentOS binary", clicking on "Browse" under "Build Artifacts", then
navigating to "build-helpers", and downloading the binary from there.

Places on the web related to Loopy
----------------------------------

* `Python package index <http://pypi.python.org/pypi/loo.py>`_ (download releases) Note the extra '.' in the PyPI identifier!

* `Github <http://github.com/inducer/loopy>`_ (get latest source code, file bugs)
* `Wiki <http://wiki.tiker.net/Loopy>`_ (read installation tips, get examples, read FAQ)
* `Homepage <http://mathema.tician.de/software/loopy>`_

Table of Contents
-----------------

If you're only just learning about loopy, consider the following `paper
<http://arxiv.org/abs/1405.7470>`_ on loo.py that may serve as a good
introduction.

Please check :ref:`installation` to get started.

.. toctree::
    :maxdepth: 2

    tutorial
    ref_creation
    ref_kernel
    ref_transform
    ref_other
    misc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

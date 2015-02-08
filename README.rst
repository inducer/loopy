Loopy lets you easily generate the tedious, complicated code that is necessary
to get good performance out of GPUs and multi-core CPUs.

----

Places on the web related to Loopy:

* `Python package index <http://pypi.python.org/pypi/loo.py>`_ (download releases) Note the extra '.' in the PyPI identifier!

  .. image:: https://badge.fury.io/py/loo.py.png
      :target: http://pypi.python.org/pypi/loo.py

* `Documentation <http://documen.tician.de/loopy>`_ (read how things work)
* `Github <http://github.com/inducer/loopy>`_ (get latest source code, file bugs)
* `Wiki <http://wiki.tiker.net/Loopy>`_ (read installation tips, get examples, read FAQ)
* `Homepage <http://mathema.tician.de/software/loopy>`_

----

Loopy's core idea is that a computation should be described simply and then
*transformed* into a version that gets high performance. This transformation
takes place under user control, from within Python.

It can capture the following types of optimizations:

* Vector and multi-core parallelism in the OpenCL/CUDA model
* Data layout transformations (structure of arrays to array of structures)
* Loopy Unrolling
* Loop tiling with efficient handling of boundary cases
* Prefetching/copy optimizations
* Instruction level parallelism
* and many more

Loopy targets array-type computations, such as the following:

* dense linear algebra,
* convolutions,
* n-body interactions,
* PDE solvers, such as finite element, finite difference, and
  Fast-Multipole-type computations

It is not (and does not want to be) a general-purpose programming language.

Loopy is licensed under the liberal `MIT license
<http://en.wikipedia.org/wiki/MIT_License>`_ and free for commercial, academic,
and private use. All of Loopy's dependencies can be automatically installed from
the package index after using::

    pip install loo.py

In addition, Loopy is compatible with and enhances
`pyopencl <http://mathema.tician.de/software/pyopencl>`_.


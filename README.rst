Loopy: Transformation-Based Generation of High-Performance CPU/GPU Code
=======================================================================

.. image:: https://gitlab.tiker.net/inducer/loopy/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/loopy/commits/main
.. image:: https://github.com/inducer/loopy/workflows/CI/badge.svg?branch=main&event=push
    :alt: Github Build Status
    :target: https://github.com/inducer/loopy/actions?query=branch%3Amain+workflow%3ACI+event%3Apush
.. image:: https://badge.fury.io/py/loopy.png
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/loopy/

Loopy lets you easily generate the tedious, complicated code that is necessary
to get good performance out of GPUs and multi-core CPUs.
Loopy's core idea is that a computation should be described simply and then
*transformed* into a version that gets high performance. This transformation
takes place under user control, from within Python.

It can capture the following types of optimizations:

* Vector and multi-core parallelism in the OpenCL/CUDA model
* Data layout transformations (structure of arrays to array of structures)
* Loop unrolling
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
<https://en.wikipedia.org/wiki/MIT_License>`_ and free for commercial, academic,
and private use. All of Loopy's dependencies can be automatically installed from
the package index after using::

    pip install loopy

In addition, Loopy is compatible with and enhances
`pyopencl <https://mathema.tician.de/software/pyopencl>`_.

---

Places on the web related to Loopy:

* `Python package index <https://pypi.org/project/loopy>`_ (download releases)
* `Documentation <https://documen.tician.de/loopy>`_ (read how things work)
* `Github <https://github.com/inducer/loopy>`_ (get latest source code, file bugs)
* `Homepage <https://mathema.tician.de/software/loopy>`_
* `Benchmarks <https://documen.tician.de/loopy/benchmarks>`_

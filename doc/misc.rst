.. _installation:

Installation
============

Option 0: Static Binary
-----------------------

If you would just like to experiment with :mod:`loopy`'s code transformation
abilities, the easiest way to get loopy is to download a statically-linked
Linux binary.

See :ref:`static-binary` for details.

Option 1: From Source, no PyOpenCL integration
-----------------------------------------------

This command should install :mod:`loopy`::

    pip install loopy

You may need to run this with :command:`sudo`.
If you don't already have `pip <https://pypi.python.org/pypi/pip>`_,
run this beforehand::

    curl -O https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    python get-pip.py

For a more manual installation, `download the source
<https://pypi.org/project/loopy>`_, unpack it, and say::

    python setup.py install

You may also clone its git repository::

    git clone --recursive https://github.com/inducer/loopy.git

Option 2: From Conda Forge, with PyOpenCL integration
-----------------------------------------------------

This set of instructions is intended for 64-bit Linux and
MacOS support computers:

#.  Make sure your system has the basics to build software.

    On Debian derivatives (Ubuntu and many more),
    installing ``build-essential`` should do the trick.

    Everywhere else, just making sure you have the ``g++`` package should be
    enough.

#.  Install `miniforge <https://github.com/conda-forge/miniforge>`_.

#.  ``export CONDA=/WHERE/YOU/INSTALLED/miniforge3``

    If you accepted the default location, this should work:

    ``export CONDA=$HOME/miniforge3``

#.  ``$CONDA/bin/conda create -n dev``

#.  ``source $CONDA/bin/activate dev``

#.  ``conda install git pip pocl islpy pyopencl`` (Linux)

    or

    ``conda install osx-pocl-opencl git pip pocl islpy pyopencl`` (OS X)

#.  Type the following command::

        pip install git+https://github.com/inducer/loopy

Next time you want to use :mod:`loopy`, just run the following command::

    source /WHERE/YOU/INSTALLED/miniforge3/bin/activate dev

You may also like to add this to a startup file (like :file:`$HOME/.bashrc`) or create an alias for it.

See the `PyOpenCL installation instructions
<https://documen.tician.de/pyopencl/misc.html#installation>`_ for options
regarding OpenCL drivers.

User-visible Changes
====================

See also :ref:`language-versioning`.

Version 2018.1
--------------
.. note::

    This version is currently under development. You can get snapshots from
    loopy's `git repository <https://github.com/inducer/loopy>`_

Version 2016.1.1
----------------

* Add :func:`loopy.chunk_iname`.
* Add ``unused:l``, ``unused:g``, and ``like:INAME`` iname tag notation
* Release automatically built, self-contained Linux binary
* Many fixes and improvements
* Docs improvements

Version 2016.1
--------------

* Initial release.

.. _license:

Licensing
=========

Loopy is licensed to you under the MIT/X Consortium license:

Copyright (c) 2009-17 Andreas Klöckner and Contributors.

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

Frequently Asked Questions
==========================

Is Loopy specific to OpenCL?
----------------------------

No, absolutely not. You can switch to a different code generation target
(subclasses of :class:`loopy.TargetBase`) by using (say)::

    knl = knl.copy(target=loopy.CudaTarget())

Also see :ref:`targets`. (Py)OpenCL right now has the best support for
running kernels directly out of the box, but that could easily be expanded.
Open an issue to discuss what you need.

In the meantime, you can generate code simply by saying::

    cg_result = loopy.generate_code_v2(knl)
    print(cg_result.host_code())
    print(cg_result.device_code())

Additionally, for C-based languages, header defintions are available via::

    loopy.generate_header(knl)

For what types of codes does :mod:`loopy` work well?
----------------------------------------------------

Any array-based/number-crunching code whose control flow is not *too*
data dependent should be expressible. For example:

* Sparse matrix-vector multiplies, despite data-dependent control
  flow (varying row lengths, say), is easy and natural to express.

* Looping until convergence on the other hand is an example
  of something that can't be expressed easily. Such checks
  would have to be performed outside of :mod:`loopy` code.

Can I see some examples?
------------------------

Loopy has a ton of tests, and right now, those are probably the best
source of examples. Here are some links:

* `Tests directory <https://github.com/inducer/loopy/tree/master/test>`_
* `Applications tests <https://github.com/inducer/loopy/blob/master/test/test_apps.py>`_
* `Feature tests <https://github.com/inducer/loopy/blob/master/test/test_loopy.py>`_

Here's a more complicated example of a loopy code:

.. literalinclude:: ../examples/python/find-centers.py
    :language: python

This example is included in the :mod:`loopy` distribution as
:download:`examples/python/find-centers.py <../examples/python/find-centers.py>`.
What this does is find nearby "centers" satisfying some criteria
for an array of points ("targets").

Specifying dependencies for groups of instructions is cumbersome. Help?
-----------------------------------------------------------------------

You can now specify instruction ID prefixes and dependencies for groups
of instructions, like this::

    with {id_prefix=init_m}
        <> m[0] =   ...
        m[1] =   ...
        m[2] =   ...
    end

    with {id_prefix=update_m,dep=init_m*}
        m[0] = m[0] + ...
        m[1] = m[1] + ...
        m[2] = m[2] * ...
    end

    with {dep=update_m*}
        output[i, j, 0] =  0.25*m[0]
        output[i, j, 1] =  0.25*m[1]
        output[i, j, 2] =  0.25*m[2]
    end

.. versionadded:: 2016.2.1

    (There was a bug in prior versions that kept this from working.)

What types of transformations can I do?
---------------------------------------

This list is always growing, but here are a few pointers:

* Unroll

  Use :func:`loopy.tag_inames` with the ``"unr"`` tag.
  Unrolled loops must have a fixed size. (See either
  :func:`loopy.split_iname` or :func:`loopy.fix_parameters`.)

* Stride changes (Row/column/something major)

  Use :func:`loopy.tag_array_axes` with (e.g.) ``stride:17`` or
  ``N1,N2,N0`` to determine how each axis of an array is realized.

* Prefetch

  Use :func:`loopy.add_prefetch`.

* Reorder loops

  Use :func:`loopy.prioritize_loops`.

* Precompute subexpressions:

  Use a :ref:`substitution-rule` to assign a name to a subexpression,
  using may be :func:`loopy.assignment_to_subst` or :func:`loopy.extract_subst`.
  Then use :func:`loopy.precompute` to create an (array or scalar)
  temporary with precomputed values.

* Tile:

  Use :func:`loopy.split_iname` to produce enough loops, then use
  :func:`loopy.prioritize_loops` to set the ordering.

* Fix constants

  Use :func:`loopy.fix_parameters`.

* Parallelize (across cores)

  Use :func:`loopy.tag_inames` with the ``"g.0"``, ``"g.1"`` (and so on) tags.

* Parallelize (across vector lanes)

  Use :func:`loopy.tag_inames` with the ``"l.0"``, ``"l.1"`` (and so on) tags.

* Affinely map loop domains

  Use :func:`loopy.affine_map_inames`.

* Texture-based data access

  Use :func:`loopy.change_arg_to_image` to use texture memory
  for an argument.

* Kernel Fusion

  Use :func:`loopy.fuse_kernels`.

* Explicit-SIMD Vectorization

  Use :func:`loopy.tag_inames` with the ``"vec"`` iname tag.
  Note that the corresponding axis of an array must
  also be tagged using the ``"vec"`` array axis tag
  (using :func:`loopy.tag_array_axes`) in order for vector code to be
  generated.

  Vectorized loops (and array axes) must have a fixed size. (See either
  :func:`loopy.split_iname` or :func:`loopy.fix_parameters` along with
  :func:`loopy.split_array_axis`.)

* Reuse of Temporary Storage

  Use :func:`loopy.alias_temporaries` to reduce the size of intermediate
  storage.

* SoA $\leftrightarrow$ AoS

  Use :func:`loopy.tag_array_axes` with the ``"sep"`` array axis tag
  to generate separate arrays for each entry of a short, fixed-length
  array axis.

  Separated array axes must have a fixed size. (See either
  :func:`loopy.split_array_axis`.)

* Realization of Instruction-level parallelism

  Use :func:`loopy.tag_inames` with the ``"ilp"`` tag.
  ILP loops must have a fixed size. (See either
  :func:`loopy.split_iname` or :func:`loopy.fix_parameters`.)

* Type inference

  Use :func:`loopy.add_and_infer_dtypes`.

* Convey assumptions:

  Use :func:`loopy.assume` to say, e.g.
  ``loopy.assume(knl, "N mod 4 = 0")`` or
  ``loopy.assume(knl, "N > 0")``.

* Perform batch computations

  Use :func:`loopy.to_batched`.

* Interface with your own library functions

  See :ref:`func-interface` for details.

* Loop collapse

  Use :func:`loopy.join_inames`.

In what sense does Loopy suport vectorization?
----------------------------------------------

There are really two ways in which the OpenCL/CUDA model of computation exposes
vectorization:

* "SIMT": The user writes scalar program instances and either the compiler or
  the hardware joins the individual program instances into vectors of a
  hardware-given length for execution.

* "Short vectors": This type of vectorization is based on vector types,
  e.g. ``float4``, which support arithmetic with implicit vector semantics
  as well as a number of 'intrinsic' functions.

Loopy suports both. The first one, SIMT, is accessible by tagging inames with,
e.g., ``l.0```. Accessing the second one requires using both execution- and
data-reshaping capabilities in loopy. To start with, you need an array that
has an axis with the length of the desired vector. If that's not yet available,
you may use :func:`loopy.split_array_axis` to produce one. Similarly, you need
an iname whose bounds match those of the desired vector length. Again, if you
don't already have one, :func:`loopy.split_iname` will easily produce one.
Lastly, both the array axis an the iname need the implementation tag ``"vec"``.
Here is an example of this machinery in action:

.. literalinclude:: ../examples/python/vector-types.py
    :language: python

Note how the example slices off the last 'slab' of iterations to ensure that
the bulk of the iteration does not require conditionals which would prevent
successful vectorization. This generates the following code:

.. literalinclude:: ../examples/python/vector-types.cl
    :language: c

What is the story with language versioning?
-------------------------------------------

The idea is to keep supporting multiple versions at a time. There's a
tension in loopy between the need to build code that keeps working
unchanged for some number of years, and needing the language to
evolve--not just as a research vehicle, but also to enable to respond
to emerging needs in applications and hardware.

The idea is not to support all versions indefinitely, merely to allow
users to upgrade on their own schedule on the scale of a couple years.
Warnings about needing to upgrade would get noisier as a version nears
deprecation. In a way, it is intended to be a version of Python's
`__future__` flags, which IMO have the served the language tremendously
well.

One can also obtain the current language version programmatically:
:data:`loopy.MOST_RECENT_LANGUAGE_VERSION`.
But pinning your code to that would mean choosing to not use the
potentially valuable guarantee to keep existing code working unchanged
for a while. Instead, it might be wiser to just grab the version of the
language current at the time of writing the code.

Uh-oh. I got a scheduling error. Any hints?
-------------------------------------------

* Make sure that dependencies between instructions are as
  you intend.

  Use :func:`loopy.show_dependency_graph` to check.

  There's a heuristic that tries to help find dependencies. If there's
  only a single write to a variable, then it adds dependencies from all
  readers to the writer. In your case, that's actually counterproductive,
  because it creates a circular dependency, hence the scheduling issue.
  So you'll have to turn that off, like so::

      knl = lp.make_kernel(
          "{ [t]: 0 <= t < T}",
          """
          <> xt = x[t] {id=fetch,dep=*}
          x[t + 1] = xt * 0.1 {dep=fetch}
          """)

* Make sure that your loops are correctly nested.

  Print the kernel to make sure all instructions are within
  the set of inames you intend them to be in.

* One iname is one for loop.

  For sequential loops, one iname corresponds to exactly one
  ``for`` loop in generated code. Loopy will not generate multiple
  loops from one iname.

* Make sure that your loops are correctly nested.

  The scheduler will try to be as helpful as it can in telling
  you where it got stuck.

Citing Loopy
============

If you use loopy for your work and find its approach helpful, please
consider citing the following article.

    A. Klöckner. `Loo.py: transformation-based code generation for GPUs and
    CPUs <https://arxiv.org/abs/1405.7470>`_. Proceedings of ARRAY '14: ACM
    SIGPLAN Workshop on Libraries, Languages, and Compilers for Array
    Programming. Edinburgh, Scotland.

Here's a Bibtex entry for your convenience::

    @inproceedings{kloeckner_loopy_2014,
       author = {{Kl{\"o}ckner}, Andreas},
       title = "{Loo.py: transformation-based code~generation for GPUs and CPUs}",
       booktitle = "{Proceedings of ARRAY `14: ACM SIGPLAN Workshop
         on Libraries, Languages, and Compilers for Array Programming}",
       year = 2014,
       publisher = "{Association for Computing Machinery}",
       address = "{Edinburgh, Scotland.}",
       doi = "{10.1145/2627373.2627387}",
    }

Getting help
============

Email the friendly folks on the `loopy mailing list <https://lists.tiker.net/listinfo/loopy>`_.

Acknowledgments
===============

Work on loopy was supported in part by

- the Department of Energy, National Nuclear Security Administration, under Award Number DE-NA0003963,
- the US Navy ONR, under grant number N00014-14-1-0117, and
- the US National Science Foundation under grant numbers DMS-1418961, CCF-1524433, DMS-1654756, SHF-1911019, and OAC-1931577.

AK also gratefully acknowledges a hardware gift from Nvidia Corporation.

The views and opinions expressed herein do not necessarily reflect those of the funding agencies.

Cross-References to Other Documentation
=======================================

.. currentmodule:: numpy

.. class:: int16

    See :class:`numpy.generic`.

.. class:: complex128

    See :class:`numpy.generic`.

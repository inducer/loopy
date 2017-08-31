.. _installation:

Installation
============

This command should install :mod:`loopy`::

    pip install loo.py

(Note the extra "."!)

You may need to run this with :command:`sudo`.
If you don't already have `pip <https://pypi.python.org/pypi/pip>`_,
run this beforehand::

    curl -O https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    python get-pip.py

For a more manual installation, `download the source
<http://pypi.python.org/pypi/loo.py>`_, unpack it, and say::

    python setup.py install

You may also clone its git repository::

    git clone --recursive git://github.com/inducer/loopy
    git clone --recursive http://git.tiker.net/trees/loopy.git

User-visible Changes
====================

Version 2016.2
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

Copyright (c) 2009-13 Andreas Klöckner and Contributors.

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
    :language: c

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
  using may be :func:`loopy.assignment_to_subst` or :func:`extract_subst`.
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
  (using :func:`tag_array_axes`) in order for vector code to be
  generated.

  Vectorized loops (and array axes) must have a fixed size. (See either
  :func:`split_iname` or :func:`fix_parameters` along with
  :func:`split_array_axis`.)

* Reuse of Temporary Storage

  Use :func:`loopy.alias_temporaries` to reduce the size of intermediate
  storage.

* SoA $\leftrightarrow$ AoS

  Use :func:`tag_array_axes` with the ``"sep"`` array axis tag
  to generate separate arrays for each entry of a short, fixed-length
  array axis.

  Separated array axes must have a fixed size. (See either
  :func:`loopy.split_array_axis`.)

* Realization of Instruction-level parallelism

  Use :func:`loopy.tag_inames` with the ``"ilp"`` tag.
  ILP loops must have a fixed size. (See either
  :func:`split_iname` or :func:`fix_parameters`.)

* Type inference

  Use :func:`loopy.add_and_infer_dtypes`.

* Convey assumptions:

  Use :func:`loopy.assume` to say, e.g.
  ``loopy.assume(knl, "N mod 4 = 0")`` or
  ``loopy.assume(knl, "N > 0")``.

* Perform batch computations

  Use :func:`loopy.to_batched`.

* Interface with your own library functions

  Use :func:`loopy.register_function_manglers`.

* Loop collapse

  Use :func:`loopy.join_inames`.

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
    CPUs <http://arxiv.org/abs/1405.7470>`_. Proceedings of ARRAY '14: ACM
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

Andreas Klöckner's work on :mod:`loopy` was supported in part by

* US Navy ONR grant number N00014-14-1-0117
* the US National Science Foundation under grant numbers DMS-1418961 and CCF-1524433.

AK also gratefully acknowledges a hardware gift from Nvidia Corporation.  The
views and opinions expressed herein do not necessarily reflect those of the
funding agencies.

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
<http://pypi.python.org/pypi/islpy>`_, unpack it, and say::

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

The FAQ is maintained collaboratively on the
`Wiki FAQ page <http://wiki.tiker.net/Loopy/FrequentlyAskedQuestions>`_.

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




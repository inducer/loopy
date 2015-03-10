from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import six


def potential_loop_nest_map(kernel):
    """Returns a dictionary mapping inames to other inames that *could*
    be nested around them.

    :seealso: :func:`loopy.schedule.loop_nest_map`
    """

    result = {}

    all_inames = kernel.all_inames()
    iname_to_insns = kernel.iname_to_insns()

    # examine pairs of all inames--O(n**2), I know.
    from loopy.kernel.data import IlpBaseTag
    for inner_iname in all_inames:
        inner_result = set()
        for outer_iname in all_inames:
            if inner_iname == outer_iname:
                continue

            if iname_to_insns[inner_iname] <= iname_to_insns[outer_iname]:
                inner_result.add(outer_iname)

        if inner_result:
            result[inner_iname] = inner_result

    return result


def fuse_loop_domains(kernel):
    did_something = False
    while True:
        lnm = potential_loop_nest_map(kernel)

        for inner_iname, outer_inames in six.iteritems(lnm):
            for outer_iname in outer_inames:
                inner_do



        print kernel
        print lnm
        1/0

        if not did_something:
            break

    return kernel


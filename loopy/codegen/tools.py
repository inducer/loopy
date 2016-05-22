from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2016 Matt Wala"

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

from pytools import memoize

@memoize
def synthesize_idis_for_extra_args(kernel, schedule_index):
    """
    :returns: A list of :class:`loopy.codegen.ImplementedDataInfo`
    """
    sched_item = kernel.schedule[schedule_index]

    from loopy.schedule import CallKernel
    from loopy.codegen import ImplementedDataInfo
    from loopy.kernel.data import InameArg, temp_var_scope
    from loopy.types import NumpyType
    import numpy as np

    assert isinstance(sched_item, CallKernel)

    idis = []
    for iname in sched_item.extra_args:
        idis.append(ImplementedDataInfo(
            target=kernel.target,
            name=iname,
            dtype=NumpyType(np.int32, kernel.target),
            arg_class=InameArg,
            is_written=False))

    return idis

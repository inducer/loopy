__copyright__ = "Copyright (C) 2022 Isuru Fernando"

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

__doc__ = """
.. currentmodule:: loopy
.. autofunction:: merge_temporary_arrays
"""

from loopy.kernel.data import auto
from loopy.symbolic import SubstitutionRuleMappingContext
from loopy.kernel import LoopKernel
from loopy.translation_unit import for_each_kernel

import pymbolic.primitives as prim
from pytools import all_equal


@for_each_kernel
def merge_temporary_arrays(kernel, array_names, new_name=None, axis_nr=0):
    """Merges temporary arrays into one array along the axis given by *axis_nr*.

    :arg array_names: a list of names of temporary variables or arguments. May
        also be a comma-separated string of these.

    :arg axis_nr: the (zero-based) index of the axis of the arrays to be merged.

    :arg new_name: new name for the merged temporary. If not given, the name
        of the first array given by *array_names* is used.
    """
    assert isinstance(kernel, LoopKernel)

    if isinstance(array_names, str):
        array_names = [i.strip() for i in array_names.split(",") if i.strip()]

    new_name = new_name or array_names[0]
    new_aggregate = prim.Variable(new_name)

    tvs = []
    offsets = {}
    count = 0
    for array_name in array_names:
        tv = kernel.temporary_variables[array_name]
        if tv.shape in [None, auto]:
            raise ValueError(f"Shape of temporary variable '{array_name}' is "
                    "unknown. Cannot merge with unknown shapes")
        offsets[array_name] = count
        count += tv.shape[axis_nr]

        shape = list(tv.shape)
        shape[axis_nr] = 1
        tvs.append(tv.copy(shape=tuple(shape), name=new_name))

    if not all_equal(tvs) == 1:
        raise ValueError("Temporary variables need to have the same attribute "
                "(except shape) in order to merge.")

    new_tv = tvs[0]
    new_shape = list(new_tv.shape)
    new_shape[axis_nr] = count
    new_tv = new_tv.copy(shape=tuple(new_shape))

    # {{{ adjust arrays

    from loopy.transform.padding import ArrayAxisSplitHelper

    count = 0

    def modify_array_access(expr):
        idx = expr.index
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = list(idx)
        idx[axis_nr] += offsets[expr.aggregate.name]

        return new_aggregate.index(tuple(idx))

    var_name_gen = kernel.get_var_name_generator()
    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, var_name_gen)
    aash = ArrayAxisSplitHelper(rule_mapping_context,
            array_names, modify_array_access)
    kernel = rule_mapping_context.finish_kernel(aash.map_kernel(kernel))

    new_tvs = {name: tv for name, tv in kernel.temporary_variables.items()
            if name not in array_names}
    new_tvs[new_name] = new_tv

    return kernel.copy(temporary_variables=new_tvs)

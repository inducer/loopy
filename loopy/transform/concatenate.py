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
.. autofunction:: concatenate_arrays
"""

from typing import Sequence, Optional, List

from loopy.kernel.data import ArrayArg, KernelArgument, TemporaryVariable, auto
from loopy.symbolic import SubstitutionRuleMappingContext
from loopy.kernel import LoopKernel
from loopy.translation_unit import for_each_kernel

import pymbolic.primitives as prim
from pytools import all_equal


@for_each_kernel
def concatenate_arrays(
        kernel: LoopKernel,
        array_names: Sequence[str],
        new_name: Optional[str] = None,
        axis_nr: int = 0) -> LoopKernel:
    """Merges arrays (temporaries or arguments) into one array along the axis
    given by *axis_nr*.

    :arg array_names: a list of names of temporary variables.

    :arg axis_nr: the (zero-based) index of the axis of the arrays to be merged.

    :arg new_name: new name for the merged temporary. If not given, a new name
        is generated.
    """
    assert isinstance(kernel, LoopKernel)

    var_name_gen = kernel.get_var_name_generator()
    new_name = new_name or var_name_gen("concatenated_array")
    new_aggregate = prim.Variable(new_name)

    arrays = []
    for array_name in array_names:
        ary = kernel.get_var_descriptor(array_name)
        if ary.shape is None or ary.shape is auto:
            raise ValueError(f"Shape of temporary variable '{array_name}' is "
                    "unknown. Cannot merge with unknown shapes")

        assert isinstance(ary.shape, tuple)
        shape = list(ary.shape)
        # make the shape value at axis_nr a constant so that we can
        # check that the rest of the attributes (except name) are equal.
        shape[axis_nr] = 1
        arrays.append(ary.copy(shape=tuple(shape), name=new_name))

    if not all_equal(arrays):
        raise ValueError("Arrays must be identical except for shape "
                "(except for shape) in order to concatenate.")

    offsets = {}
    axis_length = 0
    for array_name in array_names:
        offsets[array_name] = axis_length
        ary = kernel.temporary_variables[array_name]
        assert isinstance(ary.shape, tuple)
        axis_length += ary.shape[axis_nr]

    new_ary = arrays[0]
    new_shape = list(new_ary.shape)
    new_shape[axis_nr] = axis_length
    new_ary = new_ary.copy(shape=tuple(new_shape))

    # {{{ rewrite subscripts

    from loopy.transform.padding import SubscriptRewriter

    def modify_array_access(expr):
        idx = expr.index
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = list(idx)
        idx[axis_nr] += offsets[expr.aggregate.name]

        return new_aggregate.index(tuple(idx))

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, var_name_gen)
    aash = SubscriptRewriter(rule_mapping_context,
            array_names, modify_array_access)
    kernel = rule_mapping_context.finish_kernel(aash.map_kernel(kernel))

    # }}}

    if isinstance(new_ary, TemporaryVariable):
        new_tvs = {name: tv for name, tv in kernel.temporary_variables.items()
                if name not in array_names}
        new_tvs[new_name] = new_ary
        return kernel.copy(temporary_variables=new_tvs)
    elif isinstance(new_ary, ArrayArg):
        new_args: List[KernelArgument] = []
        inserted = False
        for arg in kernel.args:
            if arg.name in array_names:
                if not inserted:
                    new_args.append(new_ary)
                    inserted = True
            else:
                new_args.append(arg)
        return kernel.copy(args=new_args)
    else:
        raise AssertionError()

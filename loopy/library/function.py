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

from loopy.kernel.function_interface import ScalarCallable
from loopy.diagnostic import LoopyError
from loopy.types import NumpyType
import numpy as np


class MakeTupleCallable(ScalarCallable):
    def with_types(self, arg_id_to_dtype, callables_table):
        new_arg_id_to_dtype = arg_id_to_dtype.copy()
        for i in range(len(arg_id_to_dtype)):
            if i in arg_id_to_dtype and arg_id_to_dtype[i] is not None:
                new_arg_id_to_dtype[-i-1] = new_arg_id_to_dtype[i]

        return (self.copy(arg_id_to_dtype=new_arg_id_to_dtype,
            name_in_target="loopy_make_tuple"), callables_table)

    def with_descrs(self, arg_id_to_descr, callables_table):
        from loopy.kernel.function_interface import ValueArgDescriptor
        new_arg_id_to_descr = {(id, ValueArgDescriptor()):
            (-id-1, ValueArgDescriptor()) for id in arg_id_to_descr.keys()}

        return (
                self.copy(arg_id_to_descr=new_arg_id_to_descr),
                callables_table)


class IndexOfCallable(ScalarCallable):
    def with_types(self, arg_id_to_dtype, callables_table):
        new_arg_id_to_dtype = {i: dtype
                               for i, dtype in arg_id_to_dtype.items()
                               if dtype is not None}
        new_arg_id_to_dtype[-1] = NumpyType(np.int32)

        return (self.copy(arg_id_to_dtype=new_arg_id_to_dtype),
                callables_table)

    def emit_call(self, expression_to_code_mapper, expression, target):
        from pymbolic.primitives import Subscript

        if len(expression.parameters) != 1:
            raise LoopyError("%s takes exactly one argument" % self.name)
        arg, = expression.parameters
        if not isinstance(arg, Subscript):
            raise LoopyError(
                    "argument to %s must be a subscript" % self.name)

        ary = expression_to_code_mapper.find_array(arg)

        from loopy.kernel.array import get_access_info
        from pymbolic import evaluate
        access_info = get_access_info(expression_to_code_mapper.kernel,
                ary, arg.index, lambda expr: evaluate(expr,
                    expression_to_code_mapper.codegen_state.var_subst_map),
                expression_to_code_mapper.codegen_state.vectorization_info)

        from loopy.kernel.data import ImageArg
        if isinstance(ary, ImageArg):
            raise LoopyError("%s does not support images" % self.name)

        if self.name == "indexof":
            return access_info.subscripts[0]
        elif self.name == "indexof_vec":
            from loopy.kernel.array import VectorArrayDimTag
            ivec = None
            for iaxis, dim_tag in enumerate(ary.dim_tags):
                if isinstance(dim_tag, VectorArrayDimTag):
                    ivec = iaxis

            if ivec is None:
                return access_info.subscripts[0]
            else:
                return (
                    access_info.subscripts[0]*ary.shape[ivec]
                    + access_info.vector_index)

        else:
            raise RuntimeError("should not get here")

    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        return self.emit_call(
                expression_to_code_mapper,
                insn.expression,
                target), True


def get_loopy_callables():
    """
    Returns a mapping from function ids to corresponding
    :class:`loopy.kernel.function_interface.InKernelCallable` for functions
    whose interface is provided by :mod:`loopy`. Callables that fall in this
    category are --

    - reductions leading to function calls like ``argmin``, ``argmax``.
    - callables that have a predefined meaning in :mod:`loo.py` like
      ``make_tuple``, ``index_of``, ``indexof_vec``.
    """
    known_callables = {
            "make_tuple": MakeTupleCallable(name="make_tuple"),
            "indexof": IndexOfCallable(name="indexof"),
            "indexof_vec": IndexOfCallable(name="indexof_vec"),
            }

    return known_callables


# vim: foldmethod=marker

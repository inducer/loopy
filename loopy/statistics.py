from __future__ import division
from __future__ import absolute_import
import six

__copyright__ = "Copyright (C) 2015 James Stevens"

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

import numpy as np
import loopy as lp
import warnings
from islpy import dim_type
import islpy._isl as isl
from pymbolic.mapper import CombineMapper


class TypedPolyDict:

    def __init__(self, i32=0, f32=0, f64=0):
        self.poly_dict = {
                            np.dtype(np.int32): i32,
                            np.dtype(np.float32): f32,
                            np.dtype(np.float64): f64}

    def __add__(self, TPD):
        return TypedPolyDict(
            self[np.dtype(np.int32)]+TPD[np.dtype(np.int32)],
            self[np.dtype(np.float32)]+TPD[np.dtype(np.float32)],
            self[np.dtype(np.float64)]+TPD[np.dtype(np.float64)])

    def __radd__(self, other):
        if (other != 0):
            print "ERROR TRYING TO ADD TPD TO NON-ZERO NON-TPD"  # TODO
            return
        return self

    def __mul__(self, other):
        if isinstance(other, isl.PwQPolynomial):
            return TypedPolyDict(
                self[np.dtype(np.int32)]*other,
                self[np.dtype(np.float32)]*other,
                self[np.dtype(np.float64)]*other)
        else:
            # TODO
            print "ERROR: Cannot multiply TypedPolyDict by type ", type(other)

    __rmul__ = __mul__

    def __getitem__(self, index):
        return self.poly_dict[index]

    def __setitem__(self, index, value):
        self.poly_dict[index] = value

    def __str__(self):
        return str(self.poly_dict)


class ExpressionOpCounter(CombineMapper):

    def __init__(self, knl):
        self.knl = knl
        from loopy.codegen.expression import TypeInferenceMapper
        self.type_inf = TypeInferenceMapper(knl)

    def combine(self, values):
        return sum(values)

    def map_constant(self, expr):
        return TypedPolyDict(0, 0, 0)

    def map_tagged_variable(self, expr):
        return TypedPolyDict(0, 0, 0)

    def map_variable(self, expr):   # implemented in FlopCounter
        return TypedPolyDict(0, 0, 0)

    #def map_wildcard(self, expr):
    #    return 0,0

    #def map_function_symbol(self, expr):
    #    return 0,0

    def map_call(self, expr):
        # implemented in CombineMapper (functions in opencl spec)
        return TypedPolyDict(0, 0, 0)

    # def map_call_with_kwargs(self, expr):  # implemented in CombineMapper

    def map_subscript(self, expr):  # implemented in CombineMapper
        return self.rec(expr.index)

    # def map_lookup(self, expr):  # implemented in CombineMapper

    def map_sum(self, expr):  # implemented in FlopCounter
        TPD = TypedPolyDict(0, 0, 0)
        TPD[self.type_inf(expr)] = len(expr.children)-1
        if expr.children:
            return TPD + sum(self.rec(child) for child in expr.children)
        else:
            return TypedPolyDict(0, 0, 0)

    map_product = map_sum

    def map_quotient(self, expr, *args):
        TPD = TypedPolyDict(0, 0, 0)
        TPD[self.type_inf(expr)] = 1
        return TPD + self.rec(expr.numerator) + self.rec(expr.denominator)

    map_floor_div = map_quotient

    def map_remainder(self, expr):  # implemented in CombineMapper
        TPD = TypedPolyDict(0, 0, 0)
        TPD[self.type_inf(expr)] = 1
        return TPD + self.rec(expr.numerator)+self.rec(expr.denominator)

    def map_power(self, expr):  # implemented in FlopCounter
        TPD = TypedPolyDict(0, 0, 0)
        TPD[self.type_inf(expr)] = 1
        return TPD + self.rec(expr.base)+self.rec(expr.exponent)

    def map_left_shift(self, expr):  # implemented in CombineMapper
        return self.rec(expr.shiftee)+self.rec(expr.shift)  # TODO test

    map_right_shift = map_left_shift  # TODO test

    def map_bitwise_not(self, expr):  # implemented in CombineMapper # TODO test
        return self.rec(expr.child)

    def map_bitwise_or(self, expr):
        # implemented in CombineMapper, maps to map_sum; # TODO test
        return sum(self.rec(child) for child in expr.children)

    map_bitwise_xor = map_bitwise_or
    # implemented in CombineMapper, maps to map_sum; # TODO test

    map_bitwise_and = map_bitwise_or
    # implemented in CombineMapper, maps to map_sum; # TODO test

    def map_comparison(self, expr):  # implemented in CombineMapper
        return self.rec(expr.left)+self.rec(expr.right)

    def map_logical_not(self, expr):
        # implemented in CombineMapper, maps to bitwise_not
        return self.rec(expr.child)

    def map_logical_or(self, expr):  # implemented in CombineMapper, maps to map_sum
        return sum(self.rec(child) for child in expr.children)

    map_logical_and = map_logical_or

    def map_if(self, expr):  # implemented in CombineMapper, recurses
        warnings.warn("Counting operations as sum of if-statement branches.")
        # return self.rec(expr.condition) + max(
        #                    self.rec(expr.then), self.rec(expr.else_))
        return self.rec(expr.condition) + self.rec(expr.then) + self.rec(expr.else_)

    def map_if_positive(self, expr):  # implemented in FlopCounter
        warnings.warn("Counting operations as sum of if_pos-statement branches.")
        # return self.rec(expr.criterion) + max(
        #                                    self.rec(expr.then),
        #                                    self.rec(expr.else_))
        return self.rec(expr.criterion) + self.rec(expr.then) + self.rec(expr.else_)

    def map_min(self, expr):
        # implemented in CombineMapper, maps to map_sum;  # TODO test
        return sum(self.rec(child) for child in expr.children)

    map_max = map_min  # implemented in CombineMapper, maps to map_sum;  # TODO test

    def map_common_subexpression(self, expr):
        raise NotImplementedError("OpCounter encountered common_subexpression, \
                                   map_common_subexpression not implemented.")
        return 0

    def map_substitution(self, expr):
        raise NotImplementedError("OpCounter encountered substitution, \
                                    map_substitution not implemented.")
        return 0

    def map_derivative(self, expr):
        raise NotImplementedError("OpCounter encountered derivative, \
                                    map_derivative not implemented.")
        return 0

    def map_slice(self, expr):
        raise NotImplementedError("OpCounter encountered slice, \
                                    map_slice not implemented.")
        return 0


class SubscriptCounter(CombineMapper):
    def __init__(self, kernel):
        self.kernel = kernel

    def combine(self, values):
        return sum(values)

    def map_subscript(self, expr):
        name = expr.aggregate.name
        arg = self.kernel.arg_dict.get(name)
        tv = self.kernel.temporary_variables.get(name)
        if arg is not None:
            if isinstance(arg, lp.GlobalArg):
                # It's global memory
                pass
        elif tv is not None:
            if tv.is_local:
                # It's shared memory
                pass

        return 1 + self.rec(expr.index)

    def map_constant(self, expr):
        return 0

    def map_variable(self, expr):
        return 0


# to evaluate poly: poly.eval_with_dict(dictionary)
def get_op_poly(knl):

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    knl = infer_unknown_types(knl, expect_completion=True)

    knl = preprocess_kernel(knl)
    #print knl

    op_poly = 0
    op_counter = ExpressionOpCounter(knl)
    for insn in knl.instructions:
        # how many times is this instruction executed?
        # check domain size:
        insn_inames = knl.insn_inames(insn)
        inames_domain = knl.get_inames_domain(insn_inames)
        domain = (inames_domain.project_out_except(insn_inames, [dim_type.set]))
        ops = op_counter(insn.expression)
        op_poly = op_poly + ops*domain.card()
    return op_poly


def get_DRAM_access_poly(knl):  # for now just counting subscripts
    poly = 0
    subscript_counter = SubscriptCounter(knl)
    for insn in knl.instructions:
        insn_inames = knl.insn_inames(insn)
        inames_domain = knl.get_inames_domain(insn_inames)
        domain = (inames_domain.project_out_except(insn_inames, [dim_type.set]))
        poly += subscript_counter(insn.expression) * domain.card()
    return poly


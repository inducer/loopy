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

import numpy as np  # noqa
import loopy as lp
import pyopencl as cl
import pyopencl.array
import warnings
from islpy import dim_type
from pymbolic.mapper.flop_counter import FlopCounter
from pymbolic.mapper import CombineMapper


class ExpressionOpCounter(FlopCounter):

    # ExpressionOpCounter extends FlopCounter extends CombineMapper extends RecursiveMapper
    
    def __init__(self, knl):
        self.knl = knl
        from loopy.codegen.expression import TypeInferenceMapper
        self.type_inf = TypeInferenceMapper(knl)

    def map_tagged_variable(self, expr):
        return 0

    #def map_variable(self, expr):   # implemented in FlopCounter
    #    return 0

    #def map_wildcard(self, expr):
    #    return 0,0

    #def map_function_symbol(self, expr):
    #    return 0,0

    def map_call(self, expr):  # implemented in CombineMapper (functions in opencl spec)
        return 0

    # def map_call_with_kwargs(self, expr):  # implemented in CombineMapper

    def map_subscript(self, expr):  # implemented in CombineMapper
        return self.rec(expr.index)

    # def map_lookup(self, expr):  # implemented in CombineMapper

    # need to worry about data type in these (and others):
    '''
    def map_sum(self, expr):  # implemented in FlopCounter
        return 0
    def map_product(self, expr):  # implemented in FlopCounter
        return 0
    def map_quotient(self, expr):  # implemented in FlopCounter
        return 0
    def map_floor_div(self, expr):  # implemented in FlopCounter
        return 0
    '''
    def map_remainder(self, expr):  # implemented in CombineMapper
        return 1+self.rec(expr.numerator)+self.rec(expr.denominator)

    def map_power(self, expr):  # implemented in FlopCounter
        return 1+self.rec(expr.base)+self.rec(expr.exponent)

    def map_left_shift(self, expr):  # implemented in CombineMapper
        return 0+self.rec(expr.shiftee)+self.rec(expr.shift)  #TODO test

    map_right_shift = map_left_shift  #TODO test

    def map_bitwise_not(self, expr):  # implemented in CombineMapper #TODO test
        return 0+self.rec(expr.child)  

    def map_bitwise_or(self, expr):  # implemented in CombineMapper, maps to map_sum; #TODO test
        return 0+sum(self.rec(child) for child in expr.children)

    map_bitwise_xor = map_bitwise_or  # implemented in CombineMapper, maps to map_sum; #TODO test
    map_bitwise_and = map_bitwise_or  # implemented in CombineMapper, maps to map_sum; #TODO test

    def map_comparison(self, expr):  # implemented in CombineMapper
        print expr
        my_type = self.type_inf(expr)
        print my_type
        return 0+self.rec(expr.left)+self.rec(expr.right)

    def map_logical_not(self, expr):  # implemented in CombineMapper, maps to bitwise_not
        return 0+self.rec(expr.child)

    def map_logical_or(self, expr):  # implemented in CombineMapper, maps to map_sum
        return 0+sum(self.rec(child) for child in expr.children) 

    map_logical_and = map_logical_or

    def map_if(self, expr):  # implemented in CombineMapper, recurses
        warnings.warn("Counting operations as max of if-statement branches.")
        return self.rec(expr.condition)+max(self.rec(expr.then), self.rec(expr.else_))

    # def map_if_positive(self, expr):  # implemented in FlopCounter

    def map_min(self, expr):  # implemented in CombineMapper, maps to map_sum;  #TODO test
        return 0+sum(self.rec(child) for child in expr.children)

    map_max = map_min  # implemented in CombineMapper, maps to map_sum;  #TODO test


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

    fpoly = 0
    dpoly = 0
    op_counter = ExpressionOpCounter(knl)
    for insn in knl.instructions:
        # how many times is this instruction executed?
        # check domain size:
        insn_inames = knl.insn_inames(insn) 
        inames_domain = knl.get_inames_domain(insn_inames)
        domain = (inames_domain.project_out_except(insn_inames, [dim_type.set]))
        #flops, dops = op_counter(insn.expression)
        flops = op_counter(insn.expression)
        fpoly += flops*domain.card()
        #dpoly += dops*domain.card()
    return fpoly

def get_DRAM_access_poly(knl): # for now just counting subscripts
    poly = 0
    subscript_counter = subscript_counter(knl)
    for insn in knl.instructions:
        insn_inames = knl.insn_inames(insn) 
        inames_domain = knl.get_inames_domain(insn_inames)
        domain = (inames_domain.project_out_except(insn_inames, [dim_type.set]))
        poly += subscript_counter(insn.expression) * domain.card()
    return poly


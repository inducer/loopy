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
from islpy import dim_type
import loopy as lp
import pyopencl as cl
import pyopencl.array
from pymbolic.mapper.flop_counter import FlopCounter

class ExpressionFlopCounter(FlopCounter):

	# ExpressionFlopCounter extends FlopCounter extends CombineMapper extends RecursiveMapper
	
	def map_reduction(self, expr, knl):
		inames_domain = knl.get_inames_domain(frozenset([expr.inames[0]]))
		domain = (inames_domain.project_out_except(frozenset([expr.inames[0]]), [dim_type.set]))
		if str(expr.operation) == 'sum' or str(expr.operation) == 'product' :
			return domain.card()*(1+self.rec(expr.expr))
		else:
			from warnings import warn
			warn("ExpressionFlopCounter counting reduction operation as 0 flops.", stacklevel=2)
			return domain.card()*(0+self.rec(expr.expr))

	# from pymbolic:

	def map_tagged_variable(self, expr):
		return 0

	# def map_variable(self, expr):   # implemented in FlopCounter

	def map_wildcard(self, expr):
		return 0

	def map_function_symbol(self, expr):
		return 0

	# def map_call(self, expr):  # implemented in CombineMapper, recurses
	# def map_call_with_kwargs(self, expr):  # implemented in CombineMapper, recurses

	def map_subscript(self, expr):  # implemented in CombineMapper
		return self.rec(expr.index)

	# def map_lookup(self, expr):  # implemented in CombineMapper, recurses
	# def map_sum(self, expr)  # implemented in FlopCounter
	# def map_product(self, expr):  # implemented in FlopCounter
	# def map_quotient(self, expr):  # implemented in FlopCounter
	# def map_floor_div(self, expr):  # implemented in FlopCounter

	def map_remainder(self, expr):  # implemented in CombineMapper
		return 0

	# def map_power(self, expr):  # implemented in FlopCounter, recurses; coming soon

	def map_left_shift(self, expr):  # implemented in CombineMapper, recurses; coming soon
		return 0

	def map_right_shift(self, expr):  # implemented in CombineMapper, maps to left_shift; coming soon
		return 0

	def map_bitwise_not(self, expr):  # implemented in CombineMapper, recurses; coming soon
		return 0

	def map_bitwise_or(self, expr):  # implemented in CombineMapper, maps to map_sum; coming soon
		return 0

	def map_bitwise_xor(self, expr):  # implemented in CombineMapper, maps to map_sum; coming soon
		return 0

	def map_bitwise_and(self, expr):  # implemented in CombineMapper, maps to map_sum; coming soon
		return 0

	def map_comparison(self, expr):  # implemented in CombineMapper, recurses; coming soon
		return 0

	def map_logical_not(self, expr):  # implemented in CombineMapper, maps to bitwise_not; coming soon
		return 0

	def map_logical_or(self, expr):  # implemented in CombineMapper, maps to map_sum; coming soon
		return 0

	def map_logical_and(self, expr):  # implemented in CombineMapper, maps to map_sum; coming soon
		return 0

	def map_if(self, expr):  # implemented in CombineMapper, recurses; coming soon
		return 0

	# def map_if_positive(self, expr):  # implemented in FlopCounter

	def map_min(self, expr):  # implemented in CombineMapper, maps to map_sum; coming soon
		return 0

	def map_max(self, expr):  # implemented in CombineMapper, maps to map_sum 
		return 0

	def map_common_subexpression(self, expr):
		print "TESTING-map_common_subexpression: ", expr
		return 0

	def map_substitution(self, expr):
		print "TESTING-map_substitution: ", expr
		return 0

	def map_derivative(self, expr):
		print "TESTING-map_derivative: ", expr
		return 0

	def map_slice(self, expr):
		print "TESTING-map_slice: ", expr
		return 0


# to evaluate poly: poly.eval_with_dict(dictionary)
def get_flop_poly(knl):
	poly = 0
	flopCounter = ExpressionFlopCounter()
	for insn in knl.instructions:
		# how many times is this instruction executed?
		# check domain size:
		insn_inames = knl.insn_inames(insn) 
		inames_domain = knl.get_inames_domain(insn_inames)
		domain = (inames_domain.project_out_except(insn_inames, [dim_type.set]))
		#flops = flopCounter(insn.expression())
		flops = flopCounter(insn.expression(),knl)
		poly += flops*domain.card()
	return poly



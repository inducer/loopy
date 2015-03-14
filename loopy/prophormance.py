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

	def map_reduction(self, expr):
		from warnings import warn
		warn("ExpressionFlopCounter counting reduction expression as 0 flops.", stacklevel=2)
		return 0

# from pymbolic:

	def map_tagged_variable(self, expr):
		return 0

	def map_variable(self, expr):
		return 0

	def map_wildcard(self, expr):
		return 0

	def map_function_symbol(self, expr):
		return 0

	def map_call(self, expr):
		return 0

	def map_call_with_kwargs(self, expr):
		return 0

	def map_subscript(self, expr):
		return self.rec(expr.index)

	def map_lookup(self, expr):
		return 0

	def map_sum(self, expr):
		return 0

	def map_product(self, expr):
		return 0

	def map_quotient(self, expr):
		return 0

	def map_floor_div(self, expr):
		return 0

	def map_remainder(self, expr):
		return 0

	def map_power(self, expr):
		return 0

	def map_left_shift(self, expr):
		return 0

	def map_right_shift(self, expr):
		return 0

	def map_bitwise_not(self, expr):
		return 0

	def map_bitwise_or(self, expr):
		return 0

	def map_bitwise_xor(self, expr):
		return 0

	def map_bitwise_and(self, expr):
		return 0

	def map_comparison(self, expr):
		return 0

	def map_logical_not(self, expr):
		return 0

	def map_logical_or(self, expr):
		return 0

	def map_logical_and(self, expr):
		return 0

	def map_if(self, expr):
		return 0

	def map_if_positive(self, expr):
		return 0

	def map_min(self, expr):
		return 0

	def map_max(self, expr):
		return 0

	def map_common_subexpression(self, expr):
		return 0

	def map_substitution(self, expr):
		return 0

	def map_derivative(self, expr):
		return 0

	def map_slice(self, expr):
		return 0


# to evaluate poly: poly.eval_with_dict(dictionary)
def get_flop_poly(knl):
	poly = 0
	flopCounter = ExpressionFlopCounter()
	for insn in knl.instructions:
		# how many times is this instruction executed?
		# check domain size:
		insn_inames = knl.insn_inames(insn)
		inames_domain = knl.get_inames_domain(knl.insn_inames(insn))
		domain = (inames_domain.project_out_except(insn_inames, [dim_type.set]))
		flops = flopCounter(insn.expression())
		poly += flops*domain.card()
	return poly


'''
class PerformanceForecaster:

	# to evaluate poly: poly.eval_with_dict(dictionary)
	def get_flop_poly(self, knl):
		poly = 0
		flopCounter = ExpressionFlopCounter()
		for insn in knl.instructions:
			# how many times is this instruction executed?
			# check domain size:
			insn_inames = knl.insn_inames(insn)
			inames_domain = knl.get_inames_domain(knl.insn_inames(insn))
			domain = (inames_domain.project_out_except(insn_inames, [dim_type.set]))
			flops = flopCounter(insn.expression())
			poly += flops*domain.card()
		return poly
'''

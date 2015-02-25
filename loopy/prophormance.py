from __future__ import division 
from __future__ import absolute_import 
import six 

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

	def map_subscript(self, expr):
		return 0

	def map_tagged_variable(self, expr):
		return 0


class PerformanceForecaster:

	# count the number of flops in the kernel
	# param_vals is a dictionary mapping parameters to values
	def count_kernel_flops(self, knl, param_vals):
		flopCounter = ExpressionFlopCounter()
		totalFlops = 0
		for insn in knl.instructions:
			# count flops for this instruction
			flops = flopCounter(insn.expression)

			# how many times is this instruction executed?
			# check domain size:
			insn_inames = knl.insn_inames(insn)
			inames_domain = knl.get_inames_domain(insn_inames)
			domain = (inames_domain.project_out_except(insn_inames, [dim_type.set]))

			print "(count_kernel_flops debug msg) domain: ", domain.card(), "; flops: ", flops
			print "(count_kernel_flops debug msg) param_vals: ", param_vals
			domain_size = 1  #TODO: 
			totalFlops += domain_size*flops

	
		return totalFlops



"""Target for Intel ISPC."""

from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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
from loopy.target.c import CTarget
from loopy.diagnostic import LoopyError

from pymbolic import var


class ISPCTarget(CTarget):
    def get_global_axis_expr(self, axis):
        return var("taskIndex%d" % axis)

    def get_local_axis_expr(self, axis):
        if axis == 0:
            return var("programIndex")
        else:
            raise LoopyError("ISPC only supports one local axis")

    def emit_barrier(self, kind, comment):
        from loopy.codegen import GeneratedInstruction
        from cgen import Comment, Statement

        assert comment

        if kind == "local":
            return GeneratedInstruction(
                    ast=Comment("local barrier: %s" % comment),
                    implemented_domain=None)

        elif kind == "global":
            return GeneratedInstruction(
                    ast=Statement("sync; /* %s */" % comment),
                    implemented_domain=None)

        else:
            raise LoopyError("unknown barrier kind")

    def get_global_arg_decl(self, name, shape, dtype, is_written):
        from loopy.codegen import POD  # uses the correct complex type
        from cgen import Const
        from cgen.ispc import ISPCUniformPointer

        arg_decl = ISPCUniformPointer(POD(self, dtype, name))

        if not is_written:
            arg_decl = Const(arg_decl)

        return arg_decl

    # }}}

# TODO: Fix argument wrapping (value,
# TODO: Fix local variable wrapping
# TODO: Fix local variable alloc
# TODO: Top-level foreach
# TODO: Generate launch code
# TODO: Vector types

# vim: foldmethod=marker

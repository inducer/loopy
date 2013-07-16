from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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


from pytools import Record


class LoopyFlags(Record):
    """
    .. rubric:: Code-generation flags

    .. attribute:: annotate_inames
    .. attribute:: trace_assignments
    .. attribute:: trace_assignment_values

    .. rubric:: Invocation-related flags

    .. attribute:: skip_arg_checks
    .. attribute:: no_numpy
    .. attribute:: return_dict
    .. attribute:: print_wrapper
    .. attribute:: print_hl_wrapper
    .. attribute:: print_cl
    .. attribute:: print_hl_cl
    .. attribute:: edit_cl
    """

    def __init__(
            # All of these should default to False for the string-based
            # interface of make_flags (below) to make sense.

            self,

            annotate_inames=False,
            trace_assignments=False,
            trace_assignment_values=False,

            skip_arg_checks=False, no_numpy=False, return_dict=False,
            print_wrapper=False, print_hl_wrapper=False,
            print_cl=False, print_hl_cl=False,
            edit_cl=False
            ):
        Record.__init__(
                self,

                annotate_inames=annotate_inames,
                trace_assignments=trace_assignments,
                trace_assignment_values=trace_assignment_values,

                skip_arg_checks=skip_arg_checks, no_numpy=no_numpy,
                return_dict=return_dict,
                print_wrapper=print_wrapper, print_hl_wrapper=print_hl_wrapper,
                print_cl=print_cl, print_hl_cl=print_hl_cl,
                edit_cl=edit_cl,
                )

    def update(self, other):
        for f in self.__class__.fields:
            setattr(self, f,
                    getattr(self, f, False)
                    or getattr(other, f, False))


def make_flags(flags_arg):
    if flags_arg is None:
        return LoopyFlags()
    elif isinstance(flags_arg, str):
        iflags_args = {}
        for name in flags_arg.split(","):
            iflags_args[name] = True
        return LoopyFlags(**iflags_args)
    elif not isinstance(flags_arg, LoopyFlags):
        return LoopyFlags(**flags_arg)

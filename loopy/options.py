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
import re


class _ColoramaStub(object):
    def __getattribute__(self, name):
        return ""


class Options(Record):
    """
    Unless otherwise specified, these options are Boolean-valued
    (i.e. on/off).

    .. rubric:: Code-generation options

    .. attribute:: annotate_inames

        When generating code for inames, annotate them with
        comments if it is not immediately apparent which
        iname is being referred to (such as for inames mapped
        to constants or OpenCL group/local IDs).

    .. attribute:: trace_assignments

        Generate code that uses *printf* in kernels to trace the
        execution of assignment instructions.

    .. attribute:: trace_assignment_values

        Like :attr:`trace_assignments`, but also trace the
        assigned values.

    .. rubric:: Invocation-related options

    .. attribute:: skip_arg_checks

        Do not do any checking (data type, data layout, shape,
        etc.) on arguments for a minor performance gain.

    .. attribute:: no_numpy

        Do not check for or accept :mod:`numpy` arrays as
        arguments.

    .. attribute:: return_dict

        Have kernels return a :class:`dict` instead of a tuple as
        output. Specifically, the result of a kernel invocation
        with this flag is a tuple ``(evt, out_dict)``, where
        *out_dict* is a dictionary mapping argument names to their
        output values. This is helpful if arguments are inferred
        and argument ordering is thus implementation-defined.

        See :meth:`CompiledKernel.__call__`.

    .. attribute:: write_wrapper

        Print the generated Python invocation wrapper.
        Accepts a file name as a value. Writes to
        ``sys.stdout`` if none is given.

    .. attribute:: highlight_wrapper

        Use syntax highlighting in :attr:`write_wrapper`.

    .. attribute:: write_cl

        Print the generated OpenCL kernel.
        Accepts a file name as a value. Writes to
        ``sys.stdout`` if none is given.

    .. attribute:: highlight_cl

        Use syntax highlighting in :attr:`write_cl`.

    .. attribute:: edit_cl

        Invoke an editor (given by the environment variable
        :envvar:`EDITOR`) on the generated kernel code,
        allowing for tweaks before the code is passed on to
        the OpenCL implementation for compilation.

    .. attribute:: cl_build_options

        Options to pass to the OpenCL compiler when building the kernel.
        A list of strings.

    .. attribute:: allow_terminal_colors

        A :class:`bool`. Whether to allow colors in terminal output

    .. rubric:: Features

    .. attribute:: disable_global_barriers
    """

    def __init__(
            # All Boolean flags in here should default to False for the
            # string-based interface of make_options (below) to make sense.

            # All defaults are further required to be False when cast to bool
            # for the update() functionality to work.

            self,

            annotate_inames=False,
            trace_assignments=False,
            trace_assignment_values=False,

            skip_arg_checks=False, no_numpy=False, return_dict=False,
            write_wrapper=False, highlight_wrapper=False,
            write_cl=False, highlight_cl=False,
            edit_cl=False, cl_build_options=[],
            allow_terminal_colors=None,
            disable_global_barriers=False,
            ):

        if allow_terminal_colors is None:
            try:
                import colorama  # noqa
            except ImportError:
                allow_terminal_colors = False
            else:
                allow_terminal_colors = True

        Record.__init__(
                self,

                annotate_inames=annotate_inames,
                trace_assignments=trace_assignments,
                trace_assignment_values=trace_assignment_values,

                skip_arg_checks=skip_arg_checks, no_numpy=no_numpy,
                return_dict=return_dict,
                write_wrapper=write_wrapper, highlight_wrapper=highlight_wrapper,
                write_cl=write_cl, highlight_cl=highlight_cl,
                edit_cl=edit_cl, cl_build_options=cl_build_options,
                allow_terminal_colors=allow_terminal_colors,
                disable_global_barriers=disable_global_barriers,
                )

    def update(self, other):
        for f in self.__class__.fields:
            setattr(self, f, getattr(self, f) or getattr(other, f))

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """
        for field_name in sorted(self.__class__.fields):
            key_builder.rec(key_hash, getattr(self, field_name))

    @property
    def _fore(self):
        if self.allow_terminal_colors:
            import colorama
            return colorama.Fore
        else:
            return _ColoramaStub()

    @property
    def _back(self):
        if self.allow_terminal_colors:
            import colorama
            return colorama.Back
        else:
            return _ColoramaStub()

    @property
    def _style(self):
        if self.allow_terminal_colors:
            import colorama
            return colorama.Style
        else:
            return _ColoramaStub()


KEY_VAL_RE = re.compile("^([a-zA-Z0-9]+)=(.*)$")


def make_options(options_arg):
    if options_arg is None:
        return Options()
    elif isinstance(options_arg, str):
        ioptions_args = {}
        for key_val in options_arg.split(","):
            kv_match = KEY_VAL_RE.match(key_val)
            if kv_match is not None:
                key = kv_match.group(1)
                val = kv_match.group(2)
                try:
                    val = int(val)
                except ValueError:
                    pass

                ioptions_args[key] = val
            else:
                ioptions_args[key_val] = True

        return Options(**ioptions_args)
    elif not isinstance(options_arg, Options):
        return Options(**options_arg)

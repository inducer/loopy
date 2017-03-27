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


import six
from pytools import ImmutableRecord
import re


ALLOW_TERMINAL_COLORS = True


class _ColoramaStub(object):
    def __getattribute__(self, name):
        return ""


def _apply_legacy_map(lmap, kwargs):
    result = {}

    for name, val in six.iteritems(kwargs):
        try:
            lmap_value = lmap[name]
        except KeyError:
            new_name = name
        else:
            if lmap_value is None:
                # ignore this
                from warnings import warn
                warn("option '%s' is deprecated and was ignored" % name,
                        DeprecationWarning)
                continue

            new_name, translator = lmap_value
            if name in result:
                raise TypeError("may not pass a value for both '%s' and '%s'"
                        % (name, new_name))

            if translator is not None:
                val = translator(val)

        result[new_name] = val

    return result


class Options(ImmutableRecord):
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

    .. attribute:: ignore_boostable_into

        Ignore the boostable_into field of the kernel, when
        determining whether an iname duplication is necessary
        for the kernel to be schedulable.

    .. attribute:: check_dep_resolution

        Whether loopy should issue an error if a dependency
        expression does not match any instructions in the kernel.

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

    .. attribute:: write_code

        Print the generated code.  Accepts a file name or a boolean as a value.
        Writes to ``sys.stdout`` if set to *True*.

    .. attribute:: edit_code

        Invoke an editor (given by the environment variable
        :envvar:`EDITOR`) on the generated kernel code,
        allowing for tweaks before the code is passed on to
        the target for compilation.

    .. attribute:: build_options

        Options to pass to the target compiler when building the kernel.
        A list of strings.

    .. attribute:: allow_terminal_colors

        A :class:`bool`. Whether to allow colors in terminal output

    .. rubric:: Features

    .. attribute:: disable_global_barriers
    """

    _legacy_options_map = {
            "cl_build_options": ("build_options", None),
            "write_cl": ("write_code", None),
            "highlight_cl": None,
            "highlight_wrapper": None,
            "disable_wrapper_highlight": None,
            "disable_code_highlight": None,
            "edit_cl": ("edit_code", None),
            }

    def __init__(
            # All Boolean flags in here should default to False for the
            # string-based interface of make_options (below) to make sense.

            # All defaults are further required to be False when cast to bool
            # for the update() functionality to work.

            self, **kwargs):

        kwargs = _apply_legacy_map(self._legacy_options_map, kwargs)

        try:
            import colorama  # noqa
        except ImportError:
            allow_terminal_colors_def = False
        else:
            allow_terminal_colors_def = True

        allow_terminal_colors_def = (
                ALLOW_TERMINAL_COLORS and allow_terminal_colors_def)

        ImmutableRecord.__init__(
                self,

                annotate_inames=kwargs.get("annotate_inames", False),
                trace_assignments=kwargs.get("trace_assignments", False),
                trace_assignment_values=kwargs.get("trace_assignment_values", False),
                ignore_boostable_into=kwargs.get("ignore_boostable_into", False),

                skip_arg_checks=kwargs.get("skip_arg_checks", False),
                no_numpy=kwargs.get("no_numpy", False),
                return_dict=kwargs.get("return_dict", False),
                write_wrapper=kwargs.get("write_wrapper", False),
                write_code=kwargs.get("write_code", False),
                edit_code=kwargs.get("edit_code", False),
                build_options=kwargs.get("build_options", []),
                allow_terminal_colors=kwargs.get("allow_terminal_colors",
                    allow_terminal_colors_def),
                disable_global_barriers=kwargs.get("disable_global_barriers",
                    False),
                check_dep_resolution=kwargs.get("check_dep_resolution", True),
                )

    # {{{ legacy compatibility

    @property
    def edit_cl(self):
        return self.edit_code

    @property
    def cl_build_options(self):
        return self.build_options

    @property
    def highlight_cl(self):
        return self.allow_terminal_colors

    @property
    def highlight_wrapper(self):
        return self.allow_terminal_colors

    @property
    def write_cl(self):
        return self.write_code

    # }}}

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
    elif isinstance(options_arg, Options):
        return options_arg
    else:
        raise TypeError("invalid argument to make_options")

from __future__ import division

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


from pytools import MovedFunctionDeprecationWrapper


# {{{ warnings

class LoopyWarningBase(UserWarning):
    pass


class LoopyWarning(LoopyWarningBase):
    pass


class LoopyAdvisory(LoopyWarningBase):
    pass


class ParameterFinderWarning(LoopyWarning):
    pass


class WriteRaceConditionWarning(LoopyWarning):
    pass

# }}}


def warn_with_kernel(kernel, id, text, type=LoopyWarning):
    from fnmatch import fnmatchcase
    for sw in kernel.silenced_warnings:
        if fnmatchcase(id, sw):
            return

    text += (" (add '%s' to silenced_warnings kernel argument to disable)"
            % id)

    from warnings import warn
    warn("in kernel %s: %s" % (kernel.name, text), type, stacklevel=2)


warn = MovedFunctionDeprecationWrapper(warn_with_kernel)


# {{{ errors

class LoopyError(RuntimeError):
    pass


class CannotBranchDomainTree(LoopyError):
    pass


class TypeInferenceFailure(LoopyError):
    pass


class AutomaticTestFailure(LoopyError):
    pass


class StaticValueFindingError(LoopyError):
    pass


class DependencyTypeInferenceFailure(TypeInferenceFailure):
    def __init__(self, message, symbol):
        TypeInferenceFailure.__init__(self, message)
        self.symbol = symbol

# }}}


# vim: foldmethod=marker

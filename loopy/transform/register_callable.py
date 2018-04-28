from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2018 Kaushik Kulkarni"

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

from loopy.kernel import LoopKernel
from loopy.kernel.function_interface import CallableKernel
from pytools import ImmutableRecord

__doc__ = """
.. currentmodule:: loopy

.. autofunction:: register_callable_kernel
"""


# {{{ register function lookup

def register_function_lookup(kernel, function_lookup):
    """
    Returns a copy of *kernel* with the *function_lookup* registered.

    :arg function_lookup: A function of signature ``(target, identifier)``
        returning a :class:`loopy.kernel.function_interface.InKernelCallable`.
    """

    # adding the function lookup to the set of function lookers in the kernel.
    new_function_scopers = kernel.function_scopers + [function_lookup]
    registered_kernel = kernel.copy(function_scopers=new_function_scopers)
    from loopy.kernel.creation import scope_functions

    # returning the scoped_version of the kernel, as new functions maybe
    # resolved.
    return scope_functions(registered_kernel)

# }}}


# {{{ register_callable_kernel

class RegisterCalleeKernel(ImmutableRecord):
    """
    Helper class to make the function scoper from
    :func:`loopy.transform.register_callable_kernel` picklable. As python
    cannot pickle lexical closures.
    """
    fields = set(['function_name', 'callable_kernel'])

    def __init__(self, function_name, callable_kernel):
        self.function_name = function_name
        self.callable_kernel = callable_kernel

    def __call__(self, target, identifier):
        if identifier == self.function_name:
            return self.callable_kernel
        return None


def register_callable_kernel(caller_kernel, function_name, callee_kernel):
    """Returns a copy of *caller_kernel* which identifies *function_name* in an
    expression as a call to *callee_kernel*.

    :arg caller_kernel: An instance of :class:`loopy.kernel.LoopKernel`.
    :arg function_name: An instance of :class:`str`.
    :arg callee_kernel: An instance of :class:`loopy.kernel.LoopKernel`.
    """

    # {{{ sanity checks

    assert isinstance(caller_kernel, LoopKernel)
    assert isinstance(callee_kernel, LoopKernel)
    assert isinstance(function_name, str)

    # }}}

    # making the target of the child kernel to be same as the target of parent
    # kernel.
    callable_kernel = CallableKernel(subkernel=callee_kernel.copy(
                        target=caller_kernel.target,
                        is_master_kernel=False))

    # disabling global barriers for callee kernel
    from loopy import set_options
    callee_kernel = set_options(callee_kernel, "disable_global_barriers")

    return register_function_lookup(caller_kernel,
            RegisterCalleeKernel(function_name, callable_kernel))

# }}}

# vim: foldmethod=marker

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
from loopy.diagnostic import LoopyError
from loopy.kernel.function_interface import CallableKernel

__doc__ = """
.. currentmodule:: loopy

.. autofunction:: register_callable_kernel
"""


# {{{ register_callable_kernel

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

    if function_name in caller_kernel.function_identifiers:
        raise LoopyError("%s is being used a default function "
                "identifier--maybe use a different function name in order to "
                "associate with a callable kernel." % function_name)

    # }}}

    # now we know some new functions, and hence scoping them.
    from loopy.kernel.creation import scope_functions

    # scoping the function corresponding to kernel call
    caller_kernel = scope_functions(caller_kernel, set([function_name]))
    updated_scoped_functions = caller_kernel.scoped_functions

    # making the target of the child kernel to be same as the target of parent
    # kernel.
    from pymbolic.primitives import Variable
    updated_scoped_functions[Variable(function_name)] = CallableKernel(
        subkernel=callee_kernel.copy(target=caller_kernel.target))

    # returning the parent kernel with the new scoped function dictionary
    return caller_kernel.copy(scoped_functions=updated_scoped_functions)

# }}}


# {{{ register scalar callable

def register_function_lookup(kernel, function_lookup):
    """
    Returns a copy of *kernel* with the *function_lookup* registered.

    :arg function_lookup: A function of signature ``(target, identifier)``
        returning a :class:`loopy.kernel.function_interface.InKernelCallable`.
    """

    # adding the function lookup to the set of function lookers in the kernel.
    new_function_scopers = kernel.function_scopers | frozenset([function_lookup])
    registered_kernel = kernel.copy(function_scopers=new_function_scopers)
    from loopy.kernel.creation import scope_functions

    # returning the scoped_version of the kernel, as new functions maybe
    # resolved.
    return scope_functions(registered_kernel)

# }}}

# vim: foldmethod=marker

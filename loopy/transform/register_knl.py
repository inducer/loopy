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
from loopy.kernel.creation import FunctionScoper
from loopy.diagnostic import LoopyError
from loopy.kernel.function_interface import CallableKernel

from loopy.kernel.instruction import (MultiAssignmentBase, CallInstruction,
        CInstruction, _DataObliviousInstruction)

__doc__ = """
.. currentmodule:: loopy

.. autofunction:: register_callable_kernel
"""


# {{{ main entrypoint

def register_callable_kernel(parent, function_name, child):
    """
    The purpose of this transformation is so that one can inoke the child
    kernel in the parent kernel.

    :arg parent

        This is the "main" kernel which will mostly remain unaltered and one
        can interpret it as stitching up the child kernel in the parent kernel.

    :arg function_name

        The name of the function call with which the child kernel must be
        associated in the parent kernel

    :arg child

        This is like a function in every other language and this might be
        invoked in one of the instructions of the parent kernel.

    ..note::

        One should note that the kernels would go under stringent compatibilty
        tests so that both of them can be confirmed to be made for each other.
    """

    # {{{ sanity checks

    assert isinstance(parent, LoopKernel)
    assert isinstance(child, LoopKernel)
    assert isinstance(function_name, str)

    # }}}

    # scoping the function
    function_scoper = FunctionScoper(set([function_name]))
    new_insns = []

    for insn in parent.instructions:
        if isinstance(insn, CallInstruction):
            new_insn = insn.copy(expression=function_scoper(insn.expression))
            new_insns.append(new_insn)
        elif isinstance(insn, (_DataObliviousInstruction, MultiAssignmentBase,
                CInstruction)):
            new_insns.append(insn)
        else:
            raise NotImplementedError("scope_functions not implemented for %s" %
                    type(insn))

    # adding the scoped function to the scoped function dict of the parent
    # kernel.

    scoped_functions = parent.scoped_functions.copy()

    if function_name in scoped_functions:
        raise LoopyError("%s is already being used as a funciton name -- maybe"
                "use a different name for registering the subkernel")

    scoped_functions[function_name] = CallableKernel(name=function_name,
        subkernel=child.copy(target=parent.target))

    # returning the parent kernel with the new scoped function dictionary
    return parent.copy(scoped_functions=scoped_functions,
            instructions=new_insns)

# }}}

# vim: foldmethod=marker

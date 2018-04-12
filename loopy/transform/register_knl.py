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


import six

from loopy.kernel import LoopKernel
from loopy.kernel.creation import FunctionScoper
from loopy.diagnostic import LoopyError
from loopy.kernel.function_interface import CallableKernel

from loopy.kernel.instruction import (MultiAssignmentBase, CallInstruction,
        CInstruction, _DataObliviousInstruction)

__doc__ = """
.. currentmodule:: loopy

.. autofunction:: register_callable_kernel

.. autofunction:: inline_kernel
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
        subkernel=child)

    # returning the parent kernel with the new scoped function dictionary
    return parent.copy(scoped_functions=scoped_functions,
            instructions=new_insns)

# }}}



def inline_kernel(kernel, function, arg_map=None):

    child = kernel.scoped_functions[function].subkernel
    vng = kernel.get_var_name_generator()

    # duplicate and rename inames

    import islpy as isl

    dim_type = isl.dim_type.set

    child_iname_map = {}
    for iname in child.all_inames():
        child_iname_map[iname] = vng(iname)

    new_domains = []
    for domain in child.domains:
        new_domain = domain.copy()
        n_dim = new_domain.n_dim()
        for i in range(n_dim):
            iname = new_domain.get_dim_name(dim_type, i)
            new_iname = child_iname_map[iname]
            new_domain = new_domain.set_dim_name(dim_type, i, new_iname)
        new_domains.append(new_domain)

    kernel = kernel.copy(domains= kernel.domains + new_domains)

    # rename temporaries

    child_temp_map = {}
    new_temps = kernel.temporary_variables.copy()
    for name, temp in six.iteritems(child.temporary_variables):
        new_name = vng(name)
        child_temp_map[name] = new_name
        new_temps[new_name] = temp.copy(name=new_name)

    kernel = kernel.copy(temporary_variables=new_temps)

    # rename arguments
    # TODO: put this in a loop
    calls = [insn for insn in kernel.instructions if isinstance(insn, CallInstruction) and insn.expression.function.name == function]
    assert len(calls) == 1
    call, = calls

    parameters = call.assignees + call.expression.parameters

    child_arg_map = {}  # arg -> SubArrayRef
    for inside, outside in six.iteritems(arg_map):
        child_arg_map[inside], = [p for p in parameters if p.subscript.aggregate.name == outside]


    # Rewrite instructions

    import pymbolic.primitives as p
    from pymbolic.mapper.substitutor import make_subst_func
    from loopy.symbolic import SubstitutionMapper

    class KernelInliner(SubstitutionMapper):
        def map_subscript(self, expr):
            if expr.aggregate.name in child_arg_map:
                aggregate = self.subst_func(expr.aggregate)
                indices = [self.subst_func(i) for i in expr.index_tuple]
                sar = child_arg_map[expr.aggregate.name]  # SubArrayRef
                # insert non-sweeping indices from outter kernel
                # TODO: sweeping indices might flip: [i,j]: A[j, i]
                for i, index in enumerate(sar.subscript.index_tuple):
                    if index not in sar.swept_inames:
                        indices.insert(i, index)
                return aggregate.index(tuple(indices))
            else:
                return super(KernelInliner, self).map_subscript(expr)

    var_map = dict((p.Variable(k), p.Variable(v)) for k, v in six.iteritems(child_iname_map))
    var_map.update(dict((p.Variable(k), p.Variable(v)) for k, v in six.iteritems(child_temp_map)))
    var_map.update(dict((p.Variable(k), p.Variable(v)) for k, v in six.iteritems(arg_map)))
    subst_mapper =  KernelInliner(make_subst_func(var_map))

    inner_insns = []
    for insn in child.instructions:
        new_insn = insn.with_transformed_expressions(subst_mapper)
        within_inames = [child_iname_map[iname] for iname in insn.within_inames]
        within_inames.extend(call.within_inames)
        new_insn = new_insn.copy(within_inames=frozenset(within_inames), priority=call.priority)
        # TODO: depends on?
        inner_insns.append(new_insn)

    new_insns = []
    for insn in kernel.instructions:
        if insn == call:
            new_insns.extend(inner_insns)
        else:
            new_insns.append(insn)

    kernel = kernel.copy(instructions=new_insns)
    return kernel


# vim: foldmethod=marker

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

import numpy as np

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
    """
    This transformation inlines a callable child kernel into the parent kernel.

    :arg: kernel

        The parent kernel.

    :arg: function

        The name of the function call to which the callable kernel is inlined.

    :arg: arg_map

        Dictionary which maps argument names in the child kernel to variables
        in the parnet kernel. If not provided, the arguments will be mapped
        according to their access and position, i.e. the first argument in the
        child kernel with write access will be mapped to the first assignee in
        the function call, and so on.

    """

    assert isinstance(kernel, LoopKernel)
    assert isinstance(function, str)
    if not arg_map:
        assert isinstance(arg_map, dict)

    if function not in kernel.scoped_functions:
        raise LoopyError("function: {0} does not exist".format(function))

    child = kernel.scoped_functions[function].subkernel

    for call in kernel.instructions:
        if not isinstance(call, CallInstruction):
            continue
        if call.expression.function.name != function:
            continue

        # {{{ duplicate and rename inames

        import islpy as isl

        vng = kernel.get_var_name_generator()
        dim_type = isl.dim_type.set

        child_iname_map = {}
        for iname in child.all_inames():
            child_iname_map[iname] = vng("child_"+iname)

        new_domains = []
        for domain in child.domains:
            new_domain = domain.copy()
            for i in range(new_domain.n_dim()):
                iname = new_domain.get_dim_name(dim_type, i)
                new_domain = new_domain.set_dim_name(
                    dim_type, i, child_iname_map[iname])
            new_domains.append(new_domain)

        kernel = kernel.copy(domains=kernel.domains + new_domains)

        # }}}

        # {{{ rename temporaries

        child_temp_map = {}
        new_temps = kernel.temporary_variables.copy()
        for name, temp in six.iteritems(child.temporary_variables):
            new_name = vng("child_"+name)
            child_temp_map[name] = new_name
            new_temps[new_name] = temp.copy(name=new_name)

        kernel = kernel.copy(temporary_variables=new_temps)

        # }}}

        # {{{ match kernel arguments

        child_arg_map = {}  # child arg name -> SubArrayRef

        # for kernel call: out1, out2 = func(in1, in2), we match out1, out2 to
        # the written arguments, and in1, in2 to the readonly arguments in
        # child kernel, according the order they appear in child.args
        writes = child.get_written_variables()
        reads = [arg.name for arg in child.args if arg.name not in writes]
        writes = [arg.name for arg in child.args if arg.name in writes]

        if arg_map:
            for inside, outside in six.iteritems(arg_map):
                if inside not in child.arg_dict:
                    raise LoopyError("arg named '{0}' not in the child "
                                     "kernel".format(inside))
                if inside in writes:
                    sar = [sar for sar in call.assignees
                           if sar.subscript.aggregate.name == outside]
                    if len(sar) != 1:
                        raise LoopyError("wrong number of variables "
                                         "named '{0}'".format(outside))
                    child_arg_map[inside], = sar
                else:
                    sar = [sar for sar in call.expression.parameters
                           if sar.subscript.aggregate.name == outside]
                    if len(sar) != 1:
                        raise LoopyError("wrong number of variables "
                                         "named '{0}'".format(outside))
                    child_arg_map[inside], = sar
        else:
            if len(call.assignees) != len(writes):
                raise LoopyError("expect {0} output variable(s), got {1}".format(
                    len(writes), len(call.assignees)))
            if len(call.expression.parameters) != len(reads):
                raise LoopyError("expect {0} input variable(s), got {1}".format(
                    len(reads), len(call.expression.parameters)))
            for arg_name, sar in zip(writes, call.assignees):
                child_arg_map[arg_name] = sar
            for arg_name, sar in zip(reads, call.expression.parameters):
                child_arg_map[arg_name] = sar

        # }}}

        # {{{ rewrite instructions

        import pymbolic.primitives as p
        from pymbolic.mapper.substitutor import make_subst_func
        from loopy.symbolic import SubstitutionMapper
        from loopy.isl_helpers import simplify_via_aff
        from functools import reduce

        class KernelInliner(SubstitutionMapper):
            """
            Mapper to replace variables (indices, temporaries, arguments) in
            the inner kernel.
            """
            def map_subscript(self, expr):
                if expr.aggregate.name in child_arg_map:
                    aggregate = self.subst_func(expr.aggregate)
                    sar = child_arg_map[expr.aggregate.name]  # SubArrayRef (parent)
                    arg_in = child.arg_dict[expr.aggregate.name]  # Arg (child)

                    # Firstly, map inner inames to outer inames.
                    outer_indices = self.map_tuple(expr.index_tuple)

                    # Next, reshape to match dimension of outer arrays.
                    # We can have e.g. A[3, 2] from outside and B[6] from inside
                    from numbers import Integral
                    if not all(isinstance(d, Integral) for d in arg_in.shape):
                        raise LoopyError(
                            "Argument: {0} in child kernel: {1} does not have "
                            "constant shape.".format(arg_in, child.name))
                    inner_sizes = [int(np.prod(arg_in.shape[i+1:]))
                                   for i in range(len(arg_in.shape))]
                    flatten_index = reduce(
                        lambda x, y: p.Sum((x, y)),
                        map(p.Product, zip(outer_indices, inner_sizes)))
                    flatten_index = simplify_via_aff(flatten_index)

                    from loopy.symbolic import pw_aff_to_expr
                    bounds = [kernel.get_iname_bounds(i.name)
                              for i in sar.swept_inames]
                    sizes = [pw_aff_to_expr(b.size) for b in bounds]
                    if not all(isinstance(d, Integral) for d in sizes):
                        raise LoopyError(
                            "SubArrayRef: {0} in parent kernel: {1} does not have "
                            "swept inames with constant size.".format(
                                sar, kernel.name))

                    sizes = [int(np.prod(sizes[i+1:])) for i in range(len(sizes))]

                    new_indices = []
                    for s in sizes:
                        ind = flatten_index // s
                        flatten_index = flatten_index - s * ind
                        new_indices.append(ind)

                    # Lastly, map sweeping indices to indices in Subscripts
                    # This takes care of cases such as [i, j]: A[i+j, i-j]
                    index_map = dict(zip(sar.swept_inames, new_indices))
                    index_mapper = SubstitutionMapper(make_subst_func(index_map))
                    new_indices = index_mapper.map_tuple(sar.subscript.index_tuple)
                    new_indices = tuple(simplify_via_aff(i) for i in new_indices)
                    return aggregate.index(tuple(new_indices))
                else:
                    return super(KernelInliner, self).map_subscript(expr)

        var_map = dict((p.Variable(k), p.Variable(v))
                       for k, v in six.iteritems(child_iname_map))
        var_map.update(dict((p.Variable(k), p.Variable(v))
                            for k, v in six.iteritems(child_temp_map)))
        var_map.update(dict((p.Variable(k), p.Variable(v.subscript.aggregate.name))
                            for k, v in six.iteritems(child_arg_map)))
        subst_mapper = KernelInliner(make_subst_func(var_map))

        ing = kernel.get_instruction_id_generator()
        insn_id = {}
        for insn in child.instructions:
            insn_id[insn.id] = ing("child_"+insn.id)

        # {{{ root and leave instructions in child kernel

        dep_map = child.recursive_insn_dep_map()
        # roots depend on nothing
        heads = set(insn for insn, deps in six.iteritems(dep_map) if not deps)
        # leaves have nothing that depends on them
        tails = set(dep_map.keys())
        for insn, deps in six.iteritems(dep_map):
            tails = tails - deps

        # }}}

        # {{{ use NoOp to mark the start and end of child kernel

        from loopy.kernel.instruction import NoOpInstruction

        noop_start = NoOpInstruction(
            id=ing("child_start"),
            within_inames=call.within_inames,
            depends_on=call.depends_on
        )
        noop_end = NoOpInstruction(
            id=call.id,
            within_inames=call.within_inames,
            depends_on=frozenset(insn_id[insn] for insn in tails)
        )
        # }}}

        inner_insns = [noop_start]

        for _insn in child.instructions:
            insn = _insn.with_transformed_expressions(subst_mapper)
            within_inames = frozenset(map(child_iname_map.get, insn.within_inames))
            within_inames = within_inames | call.within_inames
            depends_on = frozenset(map(insn_id.get, insn.depends_on))
            if insn.id in heads:
                depends_on = depends_on | set([noop_start.id])
            insn = insn.copy(
                id=insn_id[insn.id],
                within_inames=within_inames,
                # TODO: probaby need to keep priority in child kernel
                priority=call.priority,
                depends_on=depends_on
            )
            inner_insns.append(insn)

        inner_insns.append(noop_end)

        new_insns = []
        for insn in kernel.instructions:
            if insn == call:
                new_insns.extend(inner_insns)
            else:
                new_insns.append(insn)

        kernel = kernel.copy(instructions=new_insns)

        # }}}

    return kernel


# vim: foldmethod=marker

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
from loopy.diagnostic import LoopyError
from loopy.kernel.function_interface import CallableKernel
from loopy.program import ResolvedFunctionMarker

__doc__ = """
.. currentmodule:: loopy

.. autofunction:: register_function_id_to_in_knl_callable_mapper

"""


# {{{ register function lookup

def _resolved_callables_from_function_lookup(program,
        func_id_to_in_kernel_callable_mapper):
    """
    Returns a copy of *program* with the expression nodes marked "Resolved"
    if any match is found through the given
    *func_id_to_in_kernel_callable_mapper*.

    :arg func_id_to_in_kernel_callable_mapper: A function with signature
        ``(target, identifier)`` that returns either an instance of
        :class:`loopy.InKernelCallable` or *None*.
    """
    callables_table = program.callables_table

    callable_knls = dict(
            (func_id, in_knl_callable) for func_id, in_knl_callable in
            callables_table.items() if isinstance(in_knl_callable,
                CallableKernel))
    edited_callable_knls = {}

    for func_id, in_knl_callable in callable_knls.items():
        kernel = in_knl_callable.subkernel

        from loopy.symbolic import SubstitutionRuleMappingContext
        rule_mapping_context = SubstitutionRuleMappingContext(
                kernel.substitutions, kernel.get_var_name_generator())

        resolved_function_marker = ResolvedFunctionMarker(
                rule_mapping_context, kernel, callables_table,
                [func_id_to_in_kernel_callable_mapper])

        new_subkernel = rule_mapping_context.finish_kernel(
                resolved_function_marker.map_kernel(kernel))
        callables_table = resolved_function_marker.callables_table

        edited_callable_knls[func_id] = in_knl_callable.copy(
                subkernel=new_subkernel)

    new_resolved_functions = {}

    for func_id, in_knl_callable in callables_table.items():
        if func_id in edited_callable_knls:
            new_resolved_functions[func_id] = edited_callable_knls[func_id]
        else:
            new_resolved_functions[func_id] = in_knl_callable

    callables_table = callables_table.copy(
            resolved_functions=new_resolved_functions)

    return program.copy(callables_table=callables_table)


def register_function_id_to_in_knl_callable_mapper(program,
        func_id_to_in_knl_callable_mapper):
    """
    Returns a copy of *program* with the *function_lookup* registered.

    :arg func_id_to_in_knl_callable_mapper: A function of signature ``(target,
        identifier)`` returning a
        :class:`loopy.kernel.function_interface.InKernelCallable` or *None* if
        the *function_identifier* is not known.
    """

    # adding the function lookup to the set of function lookers in the kernel.
    if func_id_to_in_knl_callable_mapper not in (
            program.func_id_to_in_knl_callable_mappers):
        from loopy.tools import unpickles_equally
        if not unpickles_equally(func_id_to_in_knl_callable_mapper):
            raise LoopyError("function '%s' does not "
                    "compare equally after being upickled "
                    "and would disrupt loopy's caches"
                    % func_id_to_in_knl_callable_mapper)
        new_func_id_mappers = program.func_id_to_in_knl_callable_mappers + (
                [func_id_to_in_knl_callable_mapper])

    program = _resolved_callables_from_function_lookup(program,
            func_id_to_in_knl_callable_mapper)

    new_program = program.copy(
            func_id_to_in_knl_callable_mappers=new_func_id_mappers)

    return new_program

# }}}


# vim: foldmethod=marker

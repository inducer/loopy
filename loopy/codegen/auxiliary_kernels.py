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
import islpy as isl

from loopy.codegen import (
        ImplementedDataInfo,
        CodeGenerationState)
from loopy.diagnostic import LoopyError
from loopy.kernel.instruction import (
        Assignment, NoOpInstruction, BarrierInstruction, CallInstruction,
        CInstruction, _DataObliviousInstruction)
from cgen import Collection

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. currentmodule:: loopy

.. autofunction:: generate_auxiliary_kernel_device_code

"""


# {{{ code generation for the auxiliary kernel

def generate_auxiliary_kernel_device_code(kernel, target):
    """
    Generates device programs for the given auxiliary kernel, with the target
    specified by the parent kernel
    :returns: a :class:`CodeGenerationResult`
    """
    kernel = kernel.copy(target=target)

    from loopy.kernel import kernel_state
    if kernel.state == kernel_state.INITIAL:
        from loopy.preprocess import preprocess_kernel
        kernel = preprocess_kernel(kernel)

    if kernel.schedule is None:
        from loopy.schedule import get_one_scheduled_kernel
        kernel = get_one_scheduled_kernel(kernel)

    if kernel.state != kernel_state.SCHEDULED:
        raise LoopyError(
                "cannot generate code for a kernel that has not been "
                "scheduled")

    from loopy.type_inference import infer_unknown_types
    kernel = infer_unknown_types(kernel, expect_completion=True)

    from loopy.check import pre_codegen_checks
    pre_codegen_checks(kernel)

    logger.info("%s: generate Auxillary Kernel code: start" % kernel.name)

    # {{{ examine arg list

    from loopy.kernel.data import ValueArg
    from loopy.kernel.array import ArrayBase

    implemented_data_info = []

    for arg in kernel.args:
        is_written = arg.name in kernel.get_written_variables()
        if isinstance(arg, ArrayBase):
            implemented_data_info.extend(
                    arg.decl_info(
                        kernel.target,
                        is_written=is_written,
                        index_dtype=kernel.index_dtype))

        elif isinstance(arg, ValueArg):
            implemented_data_info.append(ImplementedDataInfo(
                target=kernel.target,
                name=arg.name,
                dtype=arg.dtype,
                arg_class=ValueArg,
                is_written=is_written))

        else:
            raise ValueError("argument type not understood: '%s'" % type(arg))

    allow_complex = False
    for var in kernel.args + list(six.itervalues(kernel.temporary_variables)):
        if var.dtype.involves_complex():
            allow_complex = True

    # }}}

    seen_dtypes = set()
    seen_functions = set()
    seen_atomic_dtypes = set()

    initial_implemented_domain = isl.BasicSet.from_params(kernel.assumptions)
    codegen_state = CodeGenerationState(
            kernel=kernel,
            implemented_data_info=implemented_data_info,
            implemented_domain=initial_implemented_domain,
            implemented_predicates=frozenset(),
            seen_dtypes=seen_dtypes,
            seen_functions=seen_functions,
            seen_atomic_dtypes=seen_atomic_dtypes,
            var_subst_map={},
            allow_complex=allow_complex,
            var_name_generator=kernel.get_var_name_generator(),
            is_generating_device_code=False,
            gen_program_name=kernel.name,
            schedule_index_end=len(kernel.schedule),
            is_generating_master_kernel=False)

    from loopy.codegen.result import generate_host_or_device_program

    # {{{ collecting ASTs of auxiliary kernels

    auxiliary_dev_progs = []

    from loopy.codegen.auxiliary_kernels import generate_auxiliary_kernel_device_code
    for insn in kernel.instructions:
        if isinstance(insn, CallInstruction):
            in_knl_callable = kernel.scoped_functions[insn.expression.function.name]
            if in_knl_callable.subkernel is not None:
                auxiliary_dev_prog = generate_auxiliary_kernel_device_code(
                        in_knl_callable.subkernel,
                        kernel.target).device_programs[0].ast
                auxiliary_dev_progs.append(auxiliary_dev_prog)
        elif isinstance(insn, (Assignment, NoOpInstruction, Assignment,
                               BarrierInstruction, CInstruction,
                               _DataObliviousInstruction)):
            pass
        else:
            raise NotImplementedError("register_knl not made for %s type of"
                    "instruciton" % (str(type(insn))))

    # }}}

    codegen_result = generate_host_or_device_program(
            codegen_state,
            schedule_index=0)

    # {{{ pasting the auxiliary functions code to the first device program

    new_dev_prog = codegen_result.device_programs[0]
    for auxiliary_dev_prog in auxiliary_dev_progs:
        new_dev_prog = new_dev_prog.copy(
                ast=Collection([auxiliary_dev_prog, new_dev_prog.ast]))
    new_device_programs = [new_dev_prog] + codegen_result.device_programs[1:]
    codegen_result = codegen_result.copy(device_programs=new_device_programs)

    # }}}

    # For faster unpickling in the common case when implemented_domains isn't needed.
    from loopy.tools import LazilyUnpicklingDict
    codegen_result = codegen_result.copy(
            implemented_domains=LazilyUnpicklingDict(
                    codegen_result.implemented_domains))

    logger.info("%s: generate code: done" % kernel.name)

    return codegen_result

# }}}

# vim: foldmethod=marker

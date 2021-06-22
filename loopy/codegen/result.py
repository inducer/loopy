__copyright__ = "Copyright (C) 2016 Andreas Kloeckner"

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

import pymbolic.primitives as prim
from loopy.codegen import VectorizationInfo
from loopy.schedule.tree import CombineMapper
from dataclasses import dataclass
from typing import Optional, Any, List, Union, Mapping
from pytools import ImmutableRecord


def process_preambles(preambles):
    seen_preamble_tags = set()
    dedup_preambles = []

    for tag, preamble in sorted(preambles, key=lambda tag_code: tag_code[0]):
        if tag in seen_preamble_tags:
            continue

        seen_preamble_tags.add(tag)
        dedup_preambles.append(preamble)

    from loopy.tools import remove_common_indentation
    return [
            remove_common_indentation(lines) + "\n"
            for lines in dedup_preambles]


__doc__ = """
.. currentmodule:: loopy.codegen.result

.. autoclass:: GeneratedProgram

.. autoclass:: CodeGenerationResult
"""


# {{{ code generation result

class GeneratedProgram(ImmutableRecord):
    """
    .. attribute:: name

    .. attribute:: is_device_program

    .. attribute:: ast

        Once generated, this captures the AST of the overall function
        definition, including the body.
    """


def prepend_code(prog, code, astb):
    """
    Prepends *code* to :attr:`GeneratedProgram.ast`.

    :arg astb: An instance of :class:`loopy.target.ASTBuilderBase`.
    :arg ast: AST within *astb*.
    """
    new_ast = astb.emit_collection(code + [prog.ast])
    return prog.copy(ast=new_ast)


class CodeGenerationResult(ImmutableRecord):
    """
    .. attribute:: host_program
    .. attribute:: device_programs

        A list of :class:`GeneratedProgram` instances
        intended to run on the compute device.

    .. attribute:: host_preambles
    .. attribute:: device_preambles

    .. automethod:: host_code
    .. automethod:: device_code
    .. automethod:: all_code

    .. attribute:: implemented_data_info

        a list of :class:`loopy.codegen.ImplementedDataInfo` objects.
        Only added at the very end of code generation.
    """
    def __init__(self, host_program, device_programs, implemented_data_info,
                 host_preambles=[], device_preambles=[]):
        super().__init__(host_program=host_program,
                         device_programs=device_programs,
                         implemented_data_info=implemented_data_info,
                         host_preambles=host_preambles,
                         device_preambles=device_preambles)

    def host_code(self):
        preamble_codes = process_preambles(getattr(self, "host_preambles", []))

        return (
                "".join(preamble_codes)
                +
                str(self.host_program.ast))

    def device_code(self):
        preamble_codes = process_preambles(getattr(self, "device_preambles", []))

        return (
                "".join(preamble_codes)
                + "\n"
                + "\n\n".join(str(dp.ast) for dp in self.device_programs))

    def all_code(self):
        preamble_codes = process_preambles(
                getattr(self, "host_preambles", [])
                +
                getattr(self, "device_preambles", [])
                )

        return (
                "".join(preamble_codes)
                + "\n"
                + "\n\n".join(str(dp.ast) for dp in self.device_programs)
                + "\n\n"
                + str(self.host_program.ast))

# }}}


def get_idis_for_kernel(kernel):
    """
    Returns a :class:`list` of :class:`~loopy.codegen.ImplementedDataInfo` for
    *kernel*.

    :arg kernel: An instance of :class:`loopy.LoopKernel`.
    """
    from loopy.kernel.data import ValueArg
    from loopy.kernel.array import ArrayBase
    from loopy.codegen import ImplementedDataInfo

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

    return implemented_data_info


# {{{ program generation top-level

@dataclass(frozen=True)
class CodeGenMapperAccumulator:
    host_ast: List[Union[Any]]
    device_ast: List[Union[GeneratedProgram, Any]]


@dataclass(frozen=True)
class CodeGenerationContext:
    """
    A context passed around while traversing the schedule tree to generate the
    target AST.
    """
    in_device: bool
    iname_exprs: Mapping[str, prim.Expression]
    vectorization_info: Optional[VectorizationInfo] = None

    def copy(self, *, in_device=None, iname_exprs=None,
             vectorization_info=None):
        if in_device is None:
            in_device = self.in_device

        if iname_exprs is None:
            iname_exprs = self.iname_exprs

        if vectorization_info is None:
            vectorization_info = self.vectorization_info

        return CodeGenerationContext(
            in_device=in_device,
            iname_exprs=iname_exprs,
            vectorization_info=vectorization_info)


class CodeGenMapper(CombineMapper):
    def __init__(self, kernel, callables_table, is_entrypoint,
                 unvectorizable_inames):
        self.kernel = kernel
        self.callables_table = callables_table
        self.is_entrypoint = is_entrypoint
        self.unvectorizable_inames = unvectorizable_inames
        self.host_ast_builder = kernel.target.get_host_ast_builder()
        self.device_ast_builder = kernel.target.get_device_ast_builder()

    def combine(self, accumulators):

        def _is_a_list_of_generated_program(ast):
            return (isinstance(ast, list) and all(isinstance(el, GeneratedProgram)
                                                for el in ast))

        def _is_a_list_of_ast_nodes(astb, ast):
            return (isinstance(ast, list) and all(isinstance(el, astb.ast_base_class)
                                                  for el in ast))

        # either all of them are programs or all of them are ASTs
        assert (all(_is_a_list_of_generated_program(acc.device_ast)
                    for acc in accumulators)
                or all(_is_a_list_of_ast_nodes(self.device_ast_builder,
                                               acc.device_ast)
                    for acc in accumulators))

        # for each accumulator
        assert all(_is_a_list_of_ast_nodes(self.host_ast_builder, acc.host_ast)
                   for acc in accumulators)

        host_components = []
        dev_components = []

        for acc in accumulators:
            if acc.host_ast is not None:
                host_components.extend(acc.host_ast)

            if acc.device_ast is not None:
                dev_components.extend(acc.device_ast)

        return CodeGenMapperAccumulator(host_components,
                                        dev_components)

    def map_schedule(self, expr):
        from loopy.kernel.data import AddressSpace

        downstream_asts = self.combine([self.rec(child,
                                                 CodeGenerationContext(False, {}))
                                        for child in expr.children])
        assert all(isinstance(el, GeneratedProgram)
                   for el in downstream_asts.device_ast)
        device_programs = downstream_asts.device_ast
        host_ast = downstream_asts.host_ast
        host_tgt_prelude = self.host_ast_builder.generate_top_of_body(self.kernel)

        # {{{ emit global temporary declarations/initializations

        tv_init_host_asts = self.host_ast_builder.get_temporary_decls(
            self.kernel, self.callables_table, subkernel_name=None)

        tv_init_device_asts = []
        for tv in sorted((tv for tv in self.kernel.temporary_variables.values()
                          if tv.address_space == AddressSpace.GLOBAL),
                         key=lambda tv: tv.name):
            if tv.initializer is not None:
                assert tv.read_only

                decl_info, = tv.decl_info(self.kernel.target,
                                          index_dtype=self.kernel.index_dtype)
                decl = self.device_ast_builder.wrap_global_constant(
                    self.device_ast_builder.get_temporary_decl(self.kernel,
                                                               self.callables_table,
                                                               tv,
                                                               decl_info))
                rhs = self.device_ast_builder.emit_array_literal(
                    self.kernel, self.callables_table, tv, tv.initializer)
                init_ast = self.device_ast_builder.emit_initializer(decl, rhs)

                tv_init_device_asts.append(init_ast)

        if tv_init_device_asts:
            tv_init_device_asts.append(self.device_ast_builder.emit_blank_line())

        # prepend the tv inits to the first device program
        device_programs = ([prepend_code(device_programs[0], tv_init_device_asts,
                                       self.device_ast_builder)]
                           + device_programs[1:])

        # }}}

        idis = get_idis_for_kernel(self.kernel)

        if self.is_entrypoint:
            host_fn_body_ast = self.host_ast_builder.ast_block_class(
                host_tgt_prelude + tv_init_host_asts + host_ast)

            host_fn_name = (self.kernel.target.host_program_name_prefix
                            + self.kernel.name
                            + self.kernel.target.host_program_name_suffix)
            host_fn_decl = (self
                            .host_ast_builder
                            .get_function_declaration(self.kernel,
                                                      self.callables_table,
                                                      host_fn_name, idis,
                                                      is_generating_device_code=True,
                                                      is_entrypoint=True)
                            )
            host_fn_ast = (self
                           .host_ast_builder
                           .get_function_definition(self.kernel, host_fn_name,
                                                    idis, host_fn_decl,
                                                    host_fn_body_ast))

            host_prog = GeneratedProgram(name=host_fn_name, is_device_program=False,
                                         ast=host_fn_ast)
            return CodeGenerationResult(host_prog, device_programs, idis)
        else:
            return CodeGenerationResult(host_program=None,
                                        device_programs=device_programs,
                                        implemented_data_info=idis)

    def map_function(self, expr, context):
        from loopy.codegen.control import synthesize_idis_for_extra_args
        assert not context.in_device

        # {{{ Host-side: call the kernel

        from loopy.schedule.tree import InstructionGatherer
        idis = (get_idis_for_kernel(self.kernel)
                + synthesize_idis_for_extra_args(self.kernel, expr))

        dev_fn_decl = (self
                       .device_ast_builder
                       .get_function_declaration(self.kernel,
                                                 self.callables_table,
                                                 expr.name, idis,
                                                 is_generating_device_code=True,
                                                 is_entrypoint=self.is_entrypoint))

        if self.is_entrypoint:
            host_ast = [self.host_ast_builder.get_kernel_call(self.kernel,
                                                              self.callables_table,
                                                              expr.name, idis,
                                                              # 'idis' include the
                                                              # "extra_args"
                                                              extra_args=[])]
        else:
            host_ast = []

        # }}}

        # {{{ Device side: Define the kernel

        # {{{ record the iname_exprs for downstream elements

        from functools import reduce
        from loopy.kernel.data import GroupInameTag, LocalInameTag
        from loopy.isl_helpers import static_min_of_pw_aff
        from loopy.symbolic import (GroupHardwareAxisIndex,
                                    LocalHardwareAxisIndex,
                                    pw_aff_to_expr)

        all_inames = reduce(frozenset.union,
                            (self.kernel.id_to_insn[id].within_inames
                             for id in InstructionGatherer()(expr)),
                            frozenset())

        def _hw_iname_expr(iname):
            tag, = self.kernel.iname_tags_of_type(iname, (GroupInameTag,
                                                          LocalInameTag))
            lbound = static_min_of_pw_aff(self
                                          .kernel.get_iname_bounds(iname)
                                          .lower_bound_pw_aff,
                                          constants_only=False)

            return pw_aff_to_expr(lbound) + (GroupHardwareAxisIndex(tag.axis)
                                             if isinstance(tag, GroupInameTag)
                                             else
                                             LocalHardwareAxisIndex(tag.axis))

        iname_exprs = {iname: _hw_iname_expr(iname)
                       for iname in all_inames
                       if self.kernel.iname_tags_of_type(iname,
                                                         (LocalInameTag,
                                                          GroupInameTag))}

        # }}}

        dwnstrm_ctx = context.copy(in_device=True, iname_exprs=iname_exprs)

        dev_fn_decl = (self
                       .device_ast_builder
                       .get_function_declaration(self.kernel,
                                                 self.callables_table,
                                                 expr.name, idis,
                                                 is_generating_device_code=True,
                                                 is_entrypoint=self.is_entrypoint))

        tgt_prelude = self.device_ast_builder.generate_top_of_body(self.kernel)
        temp_decls_asts = self.device_ast_builder.get_temporary_decls(
                self.kernel, self.callables_table, expr.name)
        children_res = self.combine([self.rec(child, dwnstrm_ctx)
                                     for child in expr.children])
        dev_fn_body_ast = self.device_ast_builder.ast_block_class(tgt_prelude
                                                                  + temp_decls_asts
                                                                  + (children_res
                                                                     .device_ast))
        assert children_res.host_ast == []

        dev_fn_ast = (self
                      .device_ast_builder
                      .get_function_definition(self.kernel, expr.name, idis,
                                               dev_fn_decl, dev_fn_body_ast))

        dev_prog = GeneratedProgram(name=expr.name, is_device_program=True,
                                    ast=dev_fn_ast)

        # }}}

        return CodeGenMapperAccumulator(host_ast, [dev_prog])

    # {{{ for loop

    def map_for(self, expr, context):
        from loopy.kernel.data import (UnrolledIlpTag, UnrollTag,
                                       VectorizeTag, LoopedIlpTag,
                                       ForceSequentialTag,
                                       InOrderSequentialSequentialTag)

        unr_tags = (UnrolledIlpTag, UnrollTag)
        vec_tags = (VectorizeTag, )
        seq_tags = (LoopedIlpTag, ForceSequentialTag,
                    InOrderSequentialSequentialTag)
        ast_builder = self.device_ast_builder if context.in_device else self.host_ast_builder  # noqa: E501

        if (self.kernel.iname_tags_of_type(expr.iname, vec_tags)
                and expr.iname not in self.unvectorizable_inames):
            assert isinstance(expr.lower_bound, int)
            assert isinstance(expr.upper_bound, int)
            assert expr.step == 1
            length = expr.upper_bound - expr.lower_bound + 1
            dwnstrm_ctx = context.copy(
                vectorization_info=VectorizationInfo(iname=expr.iname,
                                                     length=length))
            return self.combine([self.rec(child, dwnstrm_ctx)
                                 for child in expr.children])
        else:
            assert (len(self.kernel.inames[expr.iname].tags) == 0
                    or self.kernel.iname_tags_of_type(expr.iname,
                                                      seq_tags+unr_tags+vec_tags))
            assert expr.step == 1

            if expr.upper_bound != expr.lower_bound:
                dwnstrm_ctx = context.copy(vectorization_info=None)
            else:
                # special case: if ubound == lbound => unroll
                new_iname_exprs = context.iname_exprs.copy()
                new_iname_exprs[expr.iname] = expr.upper_bound
                dwnstrm_ctx = context.copy(vectorization_info=None,
                                           iname_exprs=new_iname_exprs)

            children_res = self.combine([self.rec(child, dwnstrm_ctx)
                                         for child in expr.children])
            body_ast = (children_res.device_ast
                        if context.in_device
                        else children_res.host_ast)

            if expr.upper_bound != expr.lower_bound:
                loop_body = ast_builder.ast_block_class(body_ast)
                loop_ast = [ast_builder.emit_sequential_loop(self.kernel,
                                                             self.callables_table,
                                                             expr.iname,
                                                             self.kernel.index_dtype,
                                                             expr.lower_bound,
                                                             expr.upper_bound,
                                                             loop_body,
                                                             context.iname_exprs)]
            else:
                # special case: if ubound == lbound => just have the body
                loop_ast = body_ast

            if context.in_device:
                return CodeGenMapperAccumulator(host_ast=children_res.host_ast,
                                                device_ast=loop_ast)
            else:
                return CodeGenMapperAccumulator(host_ast=loop_ast,
                                                device_ast=children_res.device_ast)

    # }}}

    # {{{ If

    def map_if(self, expr, context):
        ast_builder = self.device_ast_builder if context.in_device else self.host_ast_builder  # noqa: E501
        children_res = self.combine([self.rec(child, context)
                                     for child in expr.children])

        if_body = ast_builder.ast_block_class(children_res.device_ast
                                              if context.in_device
                                              else children_res.host_ast)

        if_ast = ast_builder.emit_if(self.kernel, self.callables_table,
                                     expr.condition, if_body,
                                     context.iname_exprs,
                                     context.vectorization_info)

        if context.in_device:
            return CodeGenMapperAccumulator(host_ast=children_res.host_ast,
                                            device_ast=[if_ast])
        else:
            return CodeGenMapperAccumulator(host_ast=[if_ast],
                                            device_ast=children_res.device_ast)

    # }}}

    def map_barrier(self, expr, context):
        if context.in_device:
            ast_builder = self.device_ast_builder
            barrier_ast = ast_builder.emit_barrier(expr.synchronization_kind,
                                                   expr.mem_kind, expr.comment)
            return CodeGenMapperAccumulator(device_ast=[barrier_ast],
                                            host_ast=[])
        else:
            if expr.synchronization_kind in ["global", "local"]:
                return CodeGenMapperAccumulator(host_ast=[self
                                                          .host_ast_builder
                                                          .emit_blank_line()],
                                                device_ast=[])
            else:
                raise NotImplementedError(f"Host barrier for {expr}.")

    # {{{ instruction

    def map_run_instruction(self, expr, context):
        from loopy.kernel.instruction import (CallInstruction, Assignment,
                                              CInstruction, NoOpInstruction)
        from loopy.codegen.instruction import (generate_assignment_instruction_code,
                                               generate_c_instruction_code,
                                               generate_nop_instruction_code,
                                               generate_call_code)

        ast_builder = self.device_ast_builder if context.in_device else self.host_ast_builder  # noqa: E501

        insn = self.kernel.id_to_insn[expr.insn_id]

        if isinstance(insn, CallInstruction):
            insn_ast = generate_call_code(self.kernel,
                                          self.callables_table,
                                          insn,
                                          ast_builder,
                                          context.iname_exprs,
                                          (context
                                           .vectorization_info))
        elif isinstance(insn, Assignment):
            insn_ast = generate_assignment_instruction_code(self.kernel,
                                                            self.callables_table,
                                                            insn,
                                                            ast_builder,
                                                            (context
                                                             .iname_exprs),
                                                            (context
                                                             .vectorization_info))
        elif isinstance(insn, CInstruction):
            insn_ast = generate_c_instruction_code(self.kernel,
                                                   self.callables_table,
                                                   insn,
                                                   ast_builder,
                                                   context.iname_exprs,
                                                   (context
                                                    .vectorization_info))
        elif isinstance(insn, NoOpInstruction):
            insn_ast = generate_nop_instruction_code(self.kernel,
                                                     self.callables_table,
                                                     insn,
                                                     ast_builder,
                                                     context.iname_exprs,
                                                     (context
                                                      .vectorization_info))
        else:
            raise NotImplementedError(type(insn))

        if context.in_device:
            return CodeGenMapperAccumulator(host_ast=[],
                                            device_ast=[insn_ast])
        else:
            return CodeGenMapperAccumulator(host_ast=[insn_ast],
                                            device_ast=[])

    # }}}

    def map_loop(self, expr, context):
        raise RuntimeError("Cannot handle loops. At this point every loop"
                           " should have been resolved as 'For' nodes.")

# }}}

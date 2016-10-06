from __future__ import division, absolute_import

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

import six
from pytools import Record


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


# {{{ code generation result

class GeneratedProgram(Record):
    """
    .. attribute:: name

    .. attribute:: is_device_program

    .. attribute:: ast

        Once generated, this captures the AST of the overall function
        definition, including the body.

    .. attribute:: body_ast

        Once generated, this captures the AST of the operative function
        body (including declaration of necessary temporaries), but not
        the overall function definition.
    """


class CodeGenerationResult(Record):
    """
    .. attribute:: host_program
    .. attribute:: device_programs

        A list of :class:`GeneratedProgram` instances
        intended to run on the compute device.

    .. attribute:: implemented_domains

        A mapping from instruction ID to a list of :class:`islpy.Set`
        objects.

    .. attribute:: host_preambles
    .. attribute:: device_preambles

    .. automethod:: host_code
    .. automethod:: device_code
    .. automethod:: all_code

    .. attribute:: implemented_data_info

        a list of :class:`loopy.codegen.ImplementedDataInfo` objects.
        Only added at the very end of code generation.
    """

    @staticmethod
    def new(codegen_state, insn_id, ast, implemented_domain):
        prg = GeneratedProgram(
                name=codegen_state.gen_program_name,
                is_device_program=codegen_state.is_generating_device_code,
                ast=ast)

        if codegen_state.is_generating_device_code:
            kwargs = {
                    "host_program": None,
                    "device_programs": [prg],
                    }
        else:
            kwargs = {
                    "host_program": prg,
                    "device_programs": [],
                    }

        return CodeGenerationResult(
                implemented_data_info=codegen_state.implemented_data_info,
                implemented_domains={insn_id: [implemented_domain]},
                **kwargs)

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

    def current_program(self, codegen_state):
        if codegen_state.is_generating_device_code:
            if self.device_programs:
                result = self.device_programs[-1]
            else:
                result = None
        else:
            result = self.host_program

        if result is None:
            ast = codegen_state.ast_builder.ast_block_class([])
            result = GeneratedProgram(
                    name=codegen_state.gen_program_name,
                    is_device_program=codegen_state.is_generating_device_code,
                    ast=ast)

        assert result.name == codegen_state.gen_program_name
        return result

    def with_new_program(self, codegen_state, program):
        if codegen_state.is_generating_device_code:
            assert program.name == codegen_state.gen_program_name
            assert program.is_device_program
            return self.copy(
                    device_programs=(
                        self.device_programs[:-1]
                        +
                        [program]))
        else:
            assert program.name == codegen_state.gen_program_name
            assert not program.is_device_program
            return self.copy(host_program=program)

    def current_ast(self, codegen_state):
        return self.current_program(codegen_state).ast

    def with_new_ast(self, codegen_state, new_ast):
        return self.with_new_program(
                codegen_state,
                self.current_program(codegen_state).copy(
                    ast=new_ast))

# }}}


# {{{ support code for AST merging

def merge_codegen_results(codegen_state, elements, collapse=True):
    elements = [el for el in elements if el is not None]

    if not elements:
        return CodeGenerationResult(
                host_program=None,
                device_programs=[],
                implemented_domains={},
                implemented_data_info=codegen_state.implemented_data_info)

    ast_els = []
    new_device_programs = []
    dev_program_names = set()
    implemented_domains = {}
    codegen_result = None

    block_cls = codegen_state.ast_builder.ast_block_class

    for el in elements:
        if isinstance(el, CodeGenerationResult):
            if codegen_result is None:
                codegen_result = el
            else:
                assert (
                        el.current_program(codegen_state).name
                        == codegen_result.current_program(codegen_state).name)

            for insn_id, idoms in six.iteritems(el.implemented_domains):
                implemented_domains.setdefault(insn_id, []).extend(idoms)

            if not codegen_state.is_generating_device_code:
                for dp in el.device_programs:
                    if dp.name not in dev_program_names:
                        new_device_programs.append(dp)
                        dev_program_names.add(dp.name)

            cur_ast = el.current_ast(codegen_state)
            if isinstance(cur_ast, block_cls):
                ast_els.extend(cur_ast.contents)
            else:
                ast_els.append(cur_ast)

        else:
            ast_els.append(el)

    if collapse and len(ast_els) == 1:
        ast, = ast_els
    else:
        ast = block_cls(ast_els)

    kwargs = {}
    if not codegen_state.is_generating_device_code:
        kwargs["device_programs"] = new_device_programs

    return (codegen_result
            .with_new_ast(codegen_state, ast)
            .copy(
                implemented_domains=implemented_domains,
                implemented_data_info=codegen_state.implemented_data_info,
                **kwargs))


def wrap_in_if(codegen_state, condition_exprs, inner):
    if condition_exprs:
        from pymbolic.primitives import LogicalAnd
        from pymbolic.mapper.stringifier import PREC_NONE
        cur_ast = inner.current_ast(codegen_state)
        return inner.with_new_ast(
                codegen_state,
                codegen_state.ast_builder.emit_if(
                    codegen_state.expression_to_code_mapper(
                        LogicalAnd(tuple(condition_exprs)), PREC_NONE),
                    cur_ast))

    return inner

# }}}


# {{{ program generation top-level

def generate_host_or_device_program(codegen_state, schedule_index):
    ast_builder = codegen_state.ast_builder
    temp_decls = ast_builder.get_temporary_decls(codegen_state, schedule_index)

    from functools import partial

    from loopy.codegen.control import build_loop_nest
    if codegen_state.is_generating_device_code:
        from loopy.schedule import CallKernel
        assert isinstance(codegen_state.kernel.schedule[schedule_index], CallKernel)

        from loopy.codegen.loop import set_up_hw_parallel_loops
        codegen_result = set_up_hw_parallel_loops(
                codegen_state, schedule_index,
                next_func=partial(build_loop_nest,
                    schedule_index=schedule_index + 1))
    else:
        codegen_result = build_loop_nest(codegen_state, schedule_index)

    codegen_result = merge_codegen_results(
            codegen_state,
            ast_builder.generate_top_of_body(codegen_state)
            + temp_decls
            + [codegen_result],
            collapse=False)

    cur_prog = codegen_result.current_program(codegen_state)
    body_ast = cur_prog.ast
    fdecl_ast = ast_builder.get_function_declaration(
            codegen_state, codegen_result, schedule_index)

    fdef_ast = ast_builder.get_function_definition(
            codegen_state, codegen_result,
            schedule_index, fdecl_ast, body_ast)

    codegen_result = codegen_result.with_new_program(
            codegen_state,
            cur_prog.copy(
                ast=fdef_ast,
                body_ast=body_ast))

    return codegen_result

# }}}

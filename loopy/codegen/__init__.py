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


from pytools import Record
import islpy as isl


# {{{ support code for AST wrapper objects

class GeneratedInstruction(Record):
    """Objects of this type are wrapped around ASTs upon
    return from generation calls to collect information about them.

    :ivar implemented_domains: A map from an insn id to a list of
        implemented domains, with the purpose of checking that
        each instruction's exact iteration space has been covered.
    """
    __slots__ = ["insn_id", "implemented_domain", "ast"]


class GeneratedCode(Record):
    """Objects of this type are wrapped around ASTs upon
    return from generation calls to collect information about them.

    :ivar implemented_domains: A map from an insn id to a list of
        implemented domains, with the purpose of checking that
        each instruction's exact iteration space has been covered.
    """
    __slots__ = ["ast", "implemented_domains"]


def gen_code_block(elements):
    from cgen import Block, Comment, Line, Initializer

    block_els = []
    implemented_domains = {}

    for el in elements:
        if isinstance(el, GeneratedCode):
            for insn_id, idoms in el.implemented_domains.iteritems():
                implemented_domains.setdefault(insn_id, []).extend(idoms)

            if isinstance(el.ast, Block):
                block_els.extend(el.ast.contents)
            else:
                block_els.append(el.ast)

        elif isinstance(el, Initializer):
            block_els.append(el)

        elif isinstance(el, Comment):
            block_els.append(el)

        elif isinstance(el, Line):
            assert not el.text
            block_els.append(el)

        elif isinstance(el, GeneratedInstruction):
            block_els.append(el.ast)
            if el.implemented_domain is not None:
                implemented_domains.setdefault(el.insn_id, []).append(
                        el.implemented_domain)

        else:
            raise ValueError("unrecognized object of type '%s' in block"
                    % type(el))

    if len(block_els) == 1:
        ast, = block_els
    else:
        ast = Block(block_els)

    return GeneratedCode(ast=ast, implemented_domains=implemented_domains)


def wrap_in(cls, *args):
    inner = args[-1]
    args = args[:-1]

    if not isinstance(inner, GeneratedCode):
        raise ValueError("unrecognized object of type '%s' in block"
                % type(inner))

    args = args + (inner.ast,)

    return GeneratedCode(ast=cls(*args),
            implemented_domains=inner.implemented_domains)


def wrap_in_if(condition_codelets, inner):
    from cgen import If

    if condition_codelets:
        return wrap_in(If,
                "\n&& ".join(condition_codelets),
                inner)

    return inner


def add_comment(cmt, code):
    if cmt is None:
        return code

    from cgen import add_comment
    assert isinstance(code, GeneratedCode)

    return GeneratedCode(
            ast=add_comment(cmt, code.ast),
            implemented_domains=code.implemented_domains)

# }}}


# {{{ code generation state

class CodeGenerationState(object):
    def __init__(self, implemented_domain, c_code_mapper):
        """
        :param implemented_domain: The entire implemented domain,
            i.e. all constraints that have been enforced so far.
        :param c_code_mapper: A C code mapper that does not take per-ILP
            assignments into account.
        """
        self.implemented_domain = implemented_domain

        self.c_code_mapper = c_code_mapper

    def intersect(self, other):
        new_impl, new_other = isl.align_two(self.implemented_domain, other)
        return CodeGenerationState(
                new_impl & new_other,
                self.c_code_mapper)

    def fix(self, iname, aff):
        from loopy.isl_helpers import iname_rel_aff
        iname_plus_lb_aff = iname_rel_aff(
                self.implemented_domain.get_space(), iname, "==", aff)

        from loopy.symbolic import pw_aff_to_expr
        cns = isl.Constraint.equality_from_aff(iname_plus_lb_aff)
        expr = pw_aff_to_expr(aff)

        return CodeGenerationState(
                self.implemented_domain.add_constraint(cns),
                self.c_code_mapper.copy_and_assign(iname, expr))

# }}}


# {{{ initial assignments

def make_initial_assignments(kernel):
    assignments = {}

    global_size, local_size = kernel.get_grid_sizes()

    from loopy.kernel.data import LocalIndexTag, GroupIndexTag
    from pymbolic import var

    for iname in kernel.all_inames():
        tag = kernel.iname_to_tag.get(iname)

        if isinstance(tag, LocalIndexTag):
            hw_axis_expr = var("lid")(tag.axis)

        elif isinstance(tag, GroupIndexTag):
            hw_axis_expr = var("gid")(tag.axis)

        else:
            continue

        bounds = kernel.get_iname_bounds(iname)

        from loopy.symbolic import pw_aff_to_expr
        assignments[iname] = pw_aff_to_expr(bounds.lower_bound_pw_aff) + hw_axis_expr

    return assignments

# }}}


# {{{ cgen overrides

from cgen import POD as PODBase


class POD(PODBase):
    def get_decl_pair(self):
        from pyopencl.tools import dtype_to_ctype
        return [dtype_to_ctype(self.dtype)], self.name

# }}}


class CLArgumentInfo(Record):
    """
    .. attribute:: name
    .. attribute:: base_name
    .. attribute:: dtype
    .. attribute:: shape
    .. attribute:: offset_for_name
    """


# {{{ main code generation entrypoint

def generate_code(kernel, with_annotation=False,
        allow_complex=None):
    from cgen import (FunctionBody, FunctionDeclaration,
            Value, Module, Block,
            Line, Const, LiteralLines, Initializer)

    from cgen.opencl import (CLKernel, CLRequiredWorkGroupSize)

    allow_complex = False
    for var in kernel.args + list(kernel.temporary_variables.itervalues()):
        if var.dtype.kind == "c":
            allow_complex = True

    seen_dtypes = set()
    seen_functions = set()

    from loopy.codegen.expression import LoopyCCodeMapper
    ccm = (LoopyCCodeMapper(kernel, seen_dtypes, seen_functions,
        with_annotation=with_annotation,
        allow_complex=allow_complex)
        .copy_and_assign_many(make_initial_assignments(kernel)))

    mod = []

    body = Block()

    # {{{ examine arg list

    from loopy.kernel.data import ImageArg, ValueArg
    from loopy.kernel.array import ArrayBase

    arg_decls = []
    cl_arg_info = []

    for arg in kernel.args:
        if isinstance(arg, ArrayBase):
            for cdecl, clai in arg.decl_info(
                    is_written=arg.name in kernel.get_written_variables(),
                    index_dtype=kernel.index_dtype):
                arg_decls.append(cdecl)
                cl_arg_info.append(clai)

        elif isinstance(arg, ValueArg):
            arg_decls.append(Const(POD(arg.dtype, arg.name)))
            cl_arg_info.append(CLArgumentInfo(
                name=arg.name,
                base_name=arg.name,
                dtype=arg.dtype,
                shape=None))

        else:
            raise ValueError("argument type not understood: '%s'" % type(arg))

    if any(isinstance(arg, ImageArg) for arg in kernel.args):
        body.append(Initializer(Const(Value("sampler_t", "loopy_sampler")),
            "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP "
                "| CLK_FILTER_NEAREST"))

    # }}}

    from pyopencl.tools import dtype_to_ctype
    mod.extend([
        LiteralLines(r"""
        #define lid(N) ((%(idx_ctype)s) get_local_id(N))
        #define gid(N) ((%(idx_ctype)s) get_group_id(N))
        """ % dict(idx_ctype=dtype_to_ctype(kernel.index_dtype))),
        Line()])

    # {{{ build lmem array declarators for temporary variables

    for tv in kernel.temporary_variables.itervalues():
        for cdecl, clai in tv.decl_info(
                is_written=True, index_dtype=kernel.index_dtype):
            body.append(cdecl)

    # }}}

    initial_implemented_domain = isl.BasicSet.from_params(kernel.assumptions)
    codegen_state = CodeGenerationState(
            initial_implemented_domain, c_code_mapper=ccm)

    from loopy.codegen.loop import set_up_hw_parallel_loops
    gen_code = set_up_hw_parallel_loops(kernel, 0, codegen_state)

    body.append(Line())

    if isinstance(gen_code.ast, Block):
        body.extend(gen_code.ast.contents)
    else:
        body.append(gen_code.ast)

    mod.append(
        FunctionBody(
            CLRequiredWorkGroupSize(
                kernel.get_grid_sizes_as_exprs()[1],
                CLKernel(FunctionDeclaration(
                    Value("void", kernel.name), arg_decls))),
            body))

    # {{{ handle preambles

    for arg in kernel.args:
        seen_dtypes.add(arg.dtype)
    for tv in kernel.temporary_variables.itervalues():
        seen_dtypes.add(tv.dtype)

    preambles = kernel.preambles[:]
    for prea_gen in kernel.preamble_generators:
        preambles.extend(prea_gen(seen_dtypes, seen_functions))

    seen_preamble_tags = set()
    dedup_preambles = []

    for tag, preamble in sorted(preambles, key=lambda tag_code: tag_code[0]):
        if tag in seen_preamble_tags:
            continue

        seen_preamble_tags.add(tag)
        dedup_preambles.append(preamble)

    mod = ([LiteralLines(lines) for lines in dedup_preambles]
            + [Line()] + mod)

    # }}}

    result = str(Module(mod))

    from loopy.check import check_implemented_domains
    assert check_implemented_domains(kernel, gen_code.implemented_domains,
            result)

    return result, cl_arg_info

# }}}





# vim: foldmethod=marker

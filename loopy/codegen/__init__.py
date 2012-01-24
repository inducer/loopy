from __future__ import division

from pytools import Record
import numpy as np
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

    def intersect(self, set):
        return CodeGenerationState(
                self.implemented_domain & set,
                self.c_code_mapper)

    def fix(self, iname, aff, space):
        from loopy.isl_helpers import iname_rel_aff
        iname_plus_lb_aff = iname_rel_aff(space, iname, "==", aff)

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

    from loopy.kernel import LocalIndexTag, GroupIndexTag
    from pymbolic import var

    for iname in kernel.all_inames():
        tag = kernel.iname_to_tag.get(iname)

        if isinstance(tag, LocalIndexTag):
            hw_axis_expr = var("lid")(tag.axis)
            hw_axis_size = local_size[tag.axis]

        elif isinstance(tag, GroupIndexTag):
            hw_axis_expr = var("gid")(tag.axis)
            hw_axis_size = global_size[tag.axis]

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

# {{{ main code generation entrypoint

def generate_code(kernel, with_annotation=False):
    from cgen import (FunctionBody, FunctionDeclaration,
            Value, ArrayOf, Module, Block,
            Line, Const, LiteralLines, Initializer)

    from cgen.opencl import (CLKernel, CLGlobal, CLRequiredWorkGroupSize,
            CLLocal, CLImage, CLConstant)

    from loopy.codegen.expression import LoopyCCodeMapper
    ccm = (LoopyCCodeMapper(kernel, with_annotation=with_annotation)
            .copy_and_assign_many(make_initial_assignments(kernel)))

    mod = Module()

    body = Block()

    # {{{ examine arg list

    def restrict_ptr_if_not_nvidia(arg):
        from cgen import Pointer, RestrictPointer

        if "nvidia" in kernel.device.platform.name.lower():
            return Pointer(arg)
        else:
            return RestrictPointer(arg)

    has_double = False
    has_image = False

    from loopy.kernel import ArrayArg, ConstantArrayArg, ImageArg, ScalarArg

    args = []
    for arg in kernel.args:
        if isinstance(arg, (ConstantArrayArg, ArrayArg)):
            arg_decl = restrict_ptr_if_not_nvidia(
                    POD(arg.dtype, arg.name))
            if arg_decl.name not in kernel.get_written_variables():
                if arg.constant_mem:
                    arg_decl = CLConstant(Const(arg_decl))
                else:
                    arg_decl = Const(arg_decl)
            if isinstance(arg, ConstantArrayArg):
                arg_decl = CLConstant(arg_decl)
            else:
                arg_decl = CLGlobal(arg_decl)
        elif isinstance(arg, ImageArg):
            if arg.name in kernel.get_written_variables():
                mode = "w"
            else:
                mode = "r"

            arg_decl = CLImage(arg.dimensions, mode, arg.name)

            has_image = True
        elif isinstance(arg, ScalarArg):
            arg_decl = Const(POD(arg.dtype, arg.name))
        else:
            raise ValueError("argument type not understood: '%s'" % type(arg))

        if arg.dtype in [np.float64, np.complex128]:
            has_double = True

        args.append(arg_decl)

    if has_double:
        mod.extend([
            Line("#pragma OPENCL EXTENSION cl_khr_fp64: enable"),
            Line()])

    if has_image:
        body.append(Initializer(Const(Value("sampler_t", "loopy_sampler")),
            "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP "
                "| CLK_FILTER_NEAREST"))

    # }}}

    if kernel.preamble is not None:
        mod.extend([LiteralLines(kernel.preamble), Line()])

    mod.extend([
        LiteralLines(r"""
        #define lid(N) ((int) get_local_id(N))
        #define gid(N) ((int) get_group_id(N))
        """),
        Line()])

    # {{{ build lmem array declarators for temporary variables

    for tv in kernel.temporary_variables.itervalues():
        temp_var_decl = POD(tv.dtype, tv.name)

        try:
            storage_shape = tv.storage_shape
        except AttributeError:
            storage_shape = tv.shape

        for l in storage_shape:
            temp_var_decl = ArrayOf(temp_var_decl, l)

        if tv.is_local:
            temp_var_decl = CLLocal(temp_var_decl)

        body.append(temp_var_decl)

    # }}}

    from islpy import align_spaces
    initial_implemented_domain = align_spaces(kernel.assumptions, kernel.domain)
    codegen_state = CodeGenerationState(initial_implemented_domain, c_code_mapper=ccm)

    from loopy.codegen.loop import set_up_hw_parallel_loops
    gen_code = set_up_hw_parallel_loops(kernel, 0, codegen_state)

    body.append(Line())

    if isinstance(gen_code.ast, Block):
        body.extend(gen_code.ast.contents)
    else:
        body.append(gen_code.ast)

    from loopy.symbolic import pw_aff_to_expr
    mod.append(
        FunctionBody(
            CLRequiredWorkGroupSize(
                tuple(pw_aff_to_expr(sz) for sz in kernel.get_grid_sizes()[1]),
                CLKernel(FunctionDeclaration(
                    Value("void", kernel.name), args))),
            body))

    gen_code_str = str(gen_code)

    from cgen import LiteralLines
    if "int_floor_div" in gen_code_str:
        mod.extend(LiteralLines("""
            #define int_floor_div(a,b) \
              (( (a) - \
                 ( ( (a)<0 ) != ( (b)<0 )) \
                  *( (b) + ( (b)<0 ) - ( (b)>=0 ) )) \
               / (b) )
            """))

    if "int_floor_div_pos_b" in gen_code_str:
        mod.extend(LiteralLines("""
            #define int_floor_div_pos_b(a,b) ( \
                ( (a) - ( ((a)<0) ? ((b)-1) : 0 )  ) / (b) \
                )
            """))

    from loopy.check import check_implemented_domains
    assert check_implemented_domains(kernel, gen_code.implemented_domains)

    return str(mod)

# }}}





# vim: foldmethod=marker

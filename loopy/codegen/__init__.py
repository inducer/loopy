from __future__ import division

from pytools import Record
import numpy as np




# {{{ support code for AST wrapper objects

class GeneratedCode(Record):
    """Objects of this type are wrapped around ASTs upon
    return from generation calls to collect information about them.
    """
    __slots__ = ["ast", "num_conditionals"]

def gen_code_block(elements, is_alternatives=False, denest=False):
    """
    :param is_alternatives: a :class:`bool` indicating that
        only one of the *elements* will effectively be executed.
    """

    from cgen import Generable, Block

    conditional_counts = []
    block_els = []
    for el in elements:
        if isinstance(el, GeneratedCode):
            conditional_counts.append(el.num_conditionals)
            if isinstance(el.ast, Block) and denest:
                block_els.extend(el.ast.contents)
            else:
                block_els.append(el.ast)
        elif isinstance(el, Generable):
            block_els.append(el)
        else:
            raise ValueError("unidentifiable object in block")

    if is_alternatives:
        num_conditionals = min(conditional_counts)
    else:
        num_conditionals = sum(conditional_counts)

    if len(block_els) == 1:
        ast, = block_els
    else:
        ast = Block(block_els)

    return GeneratedCode(ast=ast, num_conditionals=num_conditionals)

def wrap_in(cls, *args):
    inner = args[-1]
    args = args[:-1]

    from cgen import If, Generable

    if isinstance(inner, GeneratedCode):
        num_conditionals = inner.num_conditionals
        ast = inner.ast
    elif isinstance(inner, Generable):
        num_conditionals = 0
        ast = inner

    args = args + (ast,)
    ast = cls(*args)

    if isinstance(ast, If):
        import re
        cond_joiner_re = re.compile(r"\|\||\&\&")
        num_conditionals += len(cond_joiner_re.split(ast.condition))

    return GeneratedCode(ast=ast, num_conditionals=num_conditionals)

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

    from cgen import add_comment, Block
    block_with_comment = add_comment(cmt, code.ast)
    assert isinstance(block_with_comment, Block)
    return gen_code_block(block_with_comment.contents)

# }}}

# {{{ main code generation entrypoint

class ExecutionSubdomain(Record):
    __slots__ = ["implemented_domain", "c_code_mapper"]

    def __init__(self, implemented_domain, c_code_mapper):
        Record.__init__(self,
                implemented_domain=implemented_domain,
                c_code_mapper=c_code_mapper)

    def intersect(self, set):
        return ExecutionSubdomain(
                self.implemented_domain.intersect(set),
                self.c_code_mapper)

class ExecutionDomain(object):
    def __init__(self, implemented_domain, c_code_mapper, subdomains=None):
        """
        :param implemented_domain: The entire implemented domain,
            i.e. all constraints that have been enforced so far.
        :param subdomains: a list of :class:`ExecutionSubdomain`
            instances.

            The point of this being a list is the implementation of
            ILP, and each entry represents a 'fake-parallel' trip through the 
            ILP'd loop, with the requisite implemented_domain
            and a C code mapper that realizes the necessary assignments.
        :param c_code_mapper: A C code mapper that does not take per-ILP
            assignments into account.
        """
        self.implemented_domain = implemented_domain
        if subdomains is None:
            self.subdomains = [
                    ExecutionSubdomain(implemented_domain, c_code_mapper)]
        else:
            self.subdomains = subdomains

        self.c_code_mapper = c_code_mapper

    def intersect(self, set):
        return ExecutionDomain(
                self.implemented_domain.intersect(set),
                self.c_code_mapper,
                [subd.intersect(set) for subd in self.subdomains])

    def get_the_one_domain(self):
        assert len(self.subdomains) == 1
        return self.implemented_domain




def generate_code(kernel):
    from loopy.codegen.prefetch import preprocess_prefetch
    kernel = preprocess_prefetch(kernel)

    from cgen import (FunctionBody, FunctionDeclaration,
            POD, Value, ArrayOf, Module, Block,
            Define, Line, Const, LiteralLines, Initializer)

    from cgen.opencl import (CLKernel, CLGlobal, CLRequiredWorkGroupSize,
            CLLocal, CLImage, CLConstant)

    from loopy.symbolic import LoopyCCodeMapper
    ccm = LoopyCCodeMapper(kernel)

    # {{{ build top-level

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

    from loopy.kernel import ArrayArg, ImageArg

    args = []
    for arg in kernel.args:
        if isinstance(arg, ArrayArg):
            arg_decl = restrict_ptr_if_not_nvidia(
                    POD(arg.dtype, arg.name))
            if arg_decl.name in kernel.input_vectors():
                if arg.constant_mem:
                    arg_decl = CLConstant(Const(arg_decl))
                else:
                    arg_decl = Const(arg_decl)
            arg_decl = CLGlobal(arg_decl)
        elif isinstance(arg, ImageArg):
            if arg.name in kernel.input_vectors():
                mode = "r"
            else:
                mode = "w"

            arg_decl = CLImage(arg.dimensions, mode, arg.name)

            has_image = True
        else:
            arg_decl = Const(POD(arg.dtype, arg.name))

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
        #define int_floor_div(a,b) \
          (( (a) - \
             ( ( (a)<0 ) != ( (b)<0 )) \
              *( (b) + ( (b)<0 ) - ( (b)>=0 ) )) \
           / (b) )


        #define int_floor_div_pos_b(a,b) ( \
            ( (a) - ( ((a)<0) ? ((b)-1) : 0 )  ) / (b) \
            )

        """),
        Line()])

    # {{{ symbolic names for group and local indices

    from loopy.kernel import TAG_GROUP_IDX, TAG_WORK_ITEM_IDX
    for what_cls, func in [
            (TAG_GROUP_IDX, "get_group_id"),
            (TAG_WORK_ITEM_IDX, "get_local_id")]:
        for iname in kernel.ordered_inames_by_tag_type(what_cls):
            lower, upper, equality = kernel.get_bounds(iname, (iname,), allow_parameters=True)
            assert not equality
            mod.append(Define(iname, "(%s + (int) %s(%d)) /* [%s, %s) */"
                        % (ccm(lower),
                            func,
                            kernel.iname_to_tag[iname].axis,
                            ccm(lower),
                            ccm(upper))))

    mod.append(Line())

    # }}}

    # {{{ build lmem array declarators for prefetches

    for pf in kernel.prefetch.itervalues():
        smem_pf_array = POD(kernel.arg_dict[pf.input_vector].dtype, pf.name)
        for l in pf.dim_storage_lengths:
            smem_pf_array = ArrayOf(smem_pf_array, l)
        body.append(CLLocal(smem_pf_array))

    # }}}

    from loopy.codegen.dispatch import build_loop_nest

    gen_code = build_loop_nest(kernel, 0,
            ExecutionDomain( kernel.assumptions, c_code_mapper=ccm))
    body.extend([Line(), gen_code.ast])
    #print "# conditionals: %d" % gen_code.num_conditionals

    from loopy.kernel import TAG_WORK_ITEM_IDX
    mod.append(
        FunctionBody(
            CLRequiredWorkGroupSize(
                tuple(dim_length
                    for dim_length in kernel.tag_type_lengths(
                        TAG_WORK_ITEM_IDX,
                        allow_parameters=False)),
                CLKernel(FunctionDeclaration(
                    Value("void", kernel.name), args))),
            body))

    # }}}

    return str(mod)

# }}}




# vim: foldmethod=marker

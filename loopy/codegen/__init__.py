from __future__ import division

from pytools import Record
import numpy as np
from pymbolic.mapper.stringifier import PREC_NONE




# {{{ support code for AST wrapper objects

class GeneratedCode(Record):
    """Objects of this type are wrapped around ASTs upon
    return from generation calls to collect information about them.
    """
    __slots__ = ["ast", "num_conditionals"]

def gen_code_block(elements, is_alternatives=False):
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

class CodeGenerationState(Record):
    __slots__ = ["c_code_mapper", "try_slab_partition"]

class ExecutionDomain(object):
    def __init__(self, implemented_domain, assignments_and_impl_domains=None):
        """
        :param implemented_domain: The entire implemented domain,
            i.e. all constraints that have been enforced so far.
        :param assignments_and_impl_domains: a list of tuples 
            (assignments, implemented_domain), where *assignments*
            is a list of :class:`cgen.Initializer` instances
            and *implemented_domain* is the implemented domain to which
            the situation produced by the assignments corresponds.

            The point of this being is a list is the implementation of
            ILP, and each entry represents a 'fake-parallel' trip through the 
            ILP'd loop.
        """
        if assignments_and_impl_domains is None:
            assignments_and_impl_domains = [([], implemented_domain)]
        self.implemented_domain = implemented_domain
        self.assignments_and_impl_domains = assignments_and_impl_domains

    def __len__(self):
        return len(self.assignments_and_impl_domains)

    def __iter__(self):
        return iter(self.assignments_and_impl_domains)

    def intersect(self, set):
        return ExecutionDomain(
                self.implemented_domain.intersect(set),
                [(assignments, implemented_domain.intersect(set))
                for assignments, implemented_domain
                in self.assignments_and_impl_domains])

    def get_the_one_domain(self):
        assert len(self.assignments_and_impl_domains) == 1
        return self.implemented_domain





def generate_code(kernel):
    from loopy.codegen.prefetch import preprocess_prefetch
    kernel = preprocess_prefetch(kernel)

    from cgen import (FunctionBody, FunctionDeclaration,
            POD, Value, ArrayOf, Module, Block,
            Define, Line, Const, LiteralLines, Initializer)

    from cgen.opencl import (CLKernel, CLGlobal, CLRequiredWorkGroupSize,
            CLLocal, CLImage)

    from loopy.symbolic import LoopyCCodeMapper
    my_ccm = LoopyCCodeMapper(kernel)

    def ccm(expr, prec=PREC_NONE):
        return my_ccm(expr, prec)

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
            start, stop = kernel.get_projected_bounds(iname)
            mod.append(Define(iname, "(%s + (int) %s(%d)) /* [%s, %s) */"
                        % (ccm(start),
                            func,
                            kernel.iname_to_tag[iname].axis,
                            ccm(start),
                            ccm(stop))))

    mod.append(Line())

    # }}}

    # {{{ build lmem array declarators for prefetches

    for pf in kernel.prefetch.itervalues():
        smem_pf_array = POD(kernel.arg_dict[pf.input_vector].dtype, pf.name)
        for l in pf.dim_storage_lengths:
            smem_pf_array = ArrayOf(smem_pf_array, l)
        body.append(CLLocal(smem_pf_array))

    # }}}

    cgs = CodeGenerationState(c_code_mapper=ccm, try_slab_partition=True)

    from loopy.codegen.dispatch import build_loop_nest
    import islpy as isl

    gen_code = build_loop_nest(cgs, kernel, 0, 
            ExecutionDomain(isl.Set.universe(kernel.space)))
    body.extend([Line(), gen_code.ast])
    #print "# conditionals: %d" % gen_code.num_conditionals

    from loopy.kernel import TAG_WORK_ITEM_IDX
    mod.append(
        FunctionBody(
            CLRequiredWorkGroupSize(
                tuple(dim_length
                    for dim_length in kernel.tag_type_lengths(TAG_WORK_ITEM_IDX)),
                CLKernel(FunctionDeclaration(
                    Value("void", kernel.name), args))),
            body))

    # }}}

    return str(mod)

# }}}




# vim: foldmethod=marker

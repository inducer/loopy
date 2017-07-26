"""Plain C target and base for other C-family languages."""

from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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

import numpy as np  # noqa
from loopy.target import TargetBase, ASTBuilderBase, DummyHostASTBuilder
from loopy.diagnostic import LoopyError
from cgen import Pointer, NestedDeclarator, Block
from cgen.mapper import IdentityMapper as CASTIdentityMapperBase
from pymbolic.mapper.stringifier import PREC_NONE
from loopy.symbolic import IdentityMapper
import pymbolic.primitives as p

from pytools import memoize_method


# {{{ dtype registry wrapper

class DTypeRegistryWrapper(object):
    def __init__(self, wrapped_registry):
        self.wrapped_registry = wrapped_registry

    def get_or_register_dtype(self, names, dtype=None):
        if dtype is not None:
            from loopy.types import LoopyType, NumpyType
            assert isinstance(dtype, LoopyType)

            if isinstance(dtype, NumpyType):
                return self.wrapped_registry.get_or_register_dtype(
                        names, dtype.dtype)
            else:
                raise LoopyError(
                        "unable to get or register type '%s'"
                        % dtype)
        else:
            return self.wrapped_registry.get_or_register_dtype(names, dtype)

    def dtype_to_ctype(self, dtype):
        from loopy.types import LoopyType, NumpyType
        assert isinstance(dtype, LoopyType)

        if isinstance(dtype, NumpyType):
            return self.wrapped_registry.dtype_to_ctype(dtype)
        else:
            raise LoopyError(
                    "unable to convert type '%s' to C"
                    % dtype)

# }}}


# {{{ preamble generator

def _preamble_generator(preamble_info):
    c_funcs = set(func.c_name for func in preamble_info.seen_functions)
    if "int_floor_div" in c_funcs:
        yield ("05_int_floor_div", """
            #define int_floor_div(a,b) \
              (( (a) - \
                 ( ( (a)<0 ) != ( (b)<0 )) \
                  *( (b) + ( (b)<0 ) - ( (b)>=0 ) )) \
               / (b) )
            """)

    if "int_floor_div_pos_b" in c_funcs:
        yield ("05_int_floor_div_pos_b", """
            #define int_floor_div_pos_b(a,b) ( \
                ( (a) - ( ((a)<0) ? ((b)-1) : 0 )  ) / (b) \
                )
            """)

# }}}


# {{{ cgen overrides

from cgen import Declarator


class POD(Declarator):
    """A simple declarator: The type is given as a :class:`numpy.dtype`
    and the *name* is given as a string.
    """

    def __init__(self, ast_builder, dtype, name):
        from loopy.types import LoopyType
        assert isinstance(dtype, LoopyType)

        self.ast_builder = ast_builder
        self.ctype = ast_builder.target.dtype_to_typename(dtype)
        self.dtype = dtype
        self.name = name

    def get_decl_pair(self):
        return [self.ctype], self.name

    def struct_maker_code(self, name):
        return name

    def struct_format(self):
        return self.dtype.char

    def alignment_requirement(self):
        return self.ast_builder.target.alignment_requirement(self)

    def default_value(self):
        return 0

    mapper_method = "map_loopy_pod"


class ScopingBlock(Block):
    """A block that is mandatory for scoping and may not be simplified away
    by :func:`loopy.codegen.results.merge_codegen_results`.
    """


class FunctionDeclarationWrapper(NestedDeclarator):
    mapper_method = "map_function_decl_wrapper"

# }}}


# {{{ array literals

def generate_linearized_array(array, value):
    from pytools import product
    size = product(shape_ax for shape_ax in array.shape)

    if not isinstance(size, int):
        raise LoopyError("cannot produce literal for array '%s': "
                "shape is not a compile-time constant"
                % array.name)

    strides = []

    data = np.zeros(size, array.dtype.numpy_dtype)

    from loopy.kernel.array import FixedStrideArrayDimTag
    for i, dim_tag in enumerate(array.dim_tags):
        if isinstance(dim_tag, FixedStrideArrayDimTag):

            if not isinstance(dim_tag.stride, int):
                raise LoopyError("cannot produce literal for array '%s': "
                        "stride along axis %d (1-based) is not a "
                        "compile-time constant"
                        % (array.name, i+1))

            strides.append(dim_tag.stride)

        else:
            raise LoopyError("cannot produce literal for array '%s': "
                    "dim_tag type '%s' not supported"
                    % (array.name, type(dim_tag).__name__))

    assert array.offset == 0

    from pytools import indices_in_shape
    for ituple in indices_in_shape(value.shape):
        i = sum(i_ax * strd_ax for i_ax, strd_ax in zip(ituple, strides))
        data[i] = value[ituple]

    return data


def generate_array_literal(codegen_state, array, value):
    data = generate_linearized_array(array, value)

    ecm = codegen_state.expression_to_code_mapper

    from pymbolic.mapper.stringifier import PREC_NONE
    from loopy.expression import dtype_to_type_context
    from loopy.symbolic import ArrayLiteral

    type_context = dtype_to_type_context(codegen_state.kernel.target, array.dtype)
    return CExpression(
            codegen_state.ast_builder.get_c_expression_to_code_mapper(),
            ArrayLiteral(
                tuple(
                    ecm(d_i, PREC_NONE, type_context, array.dtype).expr
                    for d_i in data)))

# }}}


# {{{ subscript CSE

class CASTIdentityMapper(CASTIdentityMapperBase):
    def map_loopy_pod(self, node, *args, **kwargs):
        return type(node)(node.ast_builder, node.dtype, node.name)

    def map_function_decl_wrapper(self, node, *args, **kwargs):
        return FunctionDeclarationWrapper(
                self.rec(node.subdecl, *args, **kwargs))


class SubscriptSubsetCounter(IdentityMapper):
    def __init__(self, subset_counters):
        self.subset_counters = subset_counters


class ASTSubscriptCollector(CASTIdentityMapper):
    def __init__(self):
        self.subset_counters = {}

    def map_expression(self, expr):
        from pymbolic.primitives import is_constant
        if isinstance(expr, CExpression) or is_constant(expr):
            return expr
        elif isinstance(expr, str):
            return expr
        else:
            raise LoopyError(
                    "Unexpected expression type: %s" % type(expr).__name__)

# }}}


# {{{ lazy expression generation

class CExpression(object):
    def __init__(self, to_code_mapper, expr):
        self.to_code_mapper = to_code_mapper
        self.expr = expr

    def __str__(self):
        return self.to_code_mapper(self.expr, PREC_NONE)

# }}}


class CTarget(TargetBase):
    """A target for plain "C", without any parallel extensions.
    """

    hash_fields = TargetBase.hash_fields + ("fortran_abi",)
    comparison_fields = TargetBase.comparison_fields + ("fortran_abi",)

    def __init__(self, fortran_abi=False):
        self.fortran_abi = fortran_abi
        super(CTarget, self).__init__()

    def split_kernel_at_global_barriers(self):
        return False

    def get_host_ast_builder(self):
        return DummyHostASTBuilder(self)

    def get_device_ast_builder(self):
        return CASTBuilder(self)

    # {{{ types

    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c.compyte.dtypes import (
                DTypeRegistry, fill_registry_with_c_types)
        result = DTypeRegistry()
        fill_registry_with_c_types(result, respect_windows=False,
                include_bool=True)
        return DTypeRegistryWrapper(result)

    def is_vector_dtype(self, dtype):
        return False

    def get_vector_dtype(self, base, count):
        raise KeyError()

    def get_or_register_dtype(self, names, dtype=None):
        # These kind of shouldn't be here.
        return self.get_dtype_registry().get_or_register_dtype(names, dtype)

    def dtype_to_typename(self, dtype):
        # These kind of shouldn't be here.
        return self.get_dtype_registry().dtype_to_ctype(dtype)

    # }}}


class _ConstRestrictPointer(Pointer):
    def get_decl_pair(self):
        sub_tp, sub_decl = self.subdecl.get_decl_pair()
        return sub_tp, ("*const __restrict__ %s" % sub_decl)


class CASTBuilder(ASTBuilderBase):
    # {{{ library

    def preamble_generators(self):
        return (
                super(CASTBuilder, self).preamble_generators() + [
                    _preamble_generator,
                    ])

    # }}}

    # {{{ code generation

    def get_function_definition(self, codegen_state, codegen_result,
            schedule_index,
            function_decl, function_body):
        kernel = codegen_state.kernel

        from cgen import (
                FunctionBody,

                # Post-mid-2016 cgens have 'Collection', too.
                Module as Collection,
                Initializer,
                Line)

        result = []

        from loopy.kernel.data import temp_var_scope

        for tv in sorted(
                six.itervalues(kernel.temporary_variables),
                key=lambda tv: tv.name):

            if tv.scope == temp_var_scope.GLOBAL and tv.initializer is not None:
                assert tv.read_only

                decl_info, = tv.decl_info(self.target,
                                index_dtype=kernel.index_dtype)
                decl = self.wrap_global_constant(
                        self.get_temporary_decl(
                            codegen_state, schedule_index, tv,
                            decl_info))

                if tv.initializer is not None:
                    decl = Initializer(decl, generate_array_literal(
                        codegen_state, tv, tv.initializer))

                result.append(decl)

        fbody = FunctionBody(function_decl, function_body)
        if not result:
            return fbody
        else:
            return Collection(result+[Line(), fbody])

    def idi_to_cgen_declarator(self, kernel, idi):
        from loopy.kernel.data import InameArg
        if (idi.offset_for_name is not None
                or idi.stride_for_name_and_axis is not None):
            assert not idi.is_written
            from cgen import Const
            return Const(POD(self, idi.dtype, idi.name))
        elif issubclass(idi.arg_class, InameArg):
            return InameArg(idi.name, idi.dtype).get_arg_decl(self)
        else:
            name = idi.base_name or idi.name
            var_descr = kernel.get_var_descriptor(name)
            from loopy.kernel.data import ArrayBase
            if isinstance(var_descr, ArrayBase):
                return var_descr.get_arg_decl(
                        self,
                        idi.name[len(name):], idi.shape, idi.dtype,
                        idi.is_written)
            else:
                return var_descr.get_arg_decl(self)

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        from cgen import FunctionDeclaration, Value

        name = codegen_result.current_program(codegen_state).name
        if self.target.fortran_abi:
            name += "_"

        return FunctionDeclarationWrapper(
                FunctionDeclaration(
                    Value("void", name),
                    [self.idi_to_cgen_declarator(codegen_state.kernel, idi)
                        for idi in codegen_state.implemented_data_info]))

    def get_temporary_decls(self, codegen_state, schedule_index):
        from loopy.kernel.data import temp_var_scope

        kernel = codegen_state.kernel

        base_storage_decls = []
        temp_decls = []

        # {{{ declare temporaries

        base_storage_sizes = {}
        base_storage_to_scope = {}
        base_storage_to_align_bytes = {}

        from cgen import ArrayOf, Initializer, AlignedAttribute, Value, Line

        for tv in sorted(
                six.itervalues(kernel.temporary_variables),
                key=lambda tv: tv.name):
            decl_info = tv.decl_info(self.target, index_dtype=kernel.index_dtype)

            if not tv.base_storage:
                for idi in decl_info:
                    # global temp vars are mapped to arguments or global declarations
                    if tv.scope != temp_var_scope.GLOBAL:
                        decl = self.wrap_temporary_decl(
                                self.get_temporary_decl(
                                    codegen_state, schedule_index, tv, idi),
                                tv.scope)

                        if tv.initializer is not None:
                            assert tv.read_only
                            decl = Initializer(decl, generate_array_literal(
                                codegen_state, tv, tv.initializer))

                        temp_decls.append(decl)

            else:
                assert tv.initializer is None

                offset = 0
                base_storage_sizes.setdefault(tv.base_storage, []).append(
                        tv.nbytes)
                base_storage_to_scope.setdefault(tv.base_storage, []).append(
                        tv.scope)

                align_size = tv.dtype.itemsize

                from loopy.kernel.array import VectorArrayDimTag
                for dim_tag, axis_len in zip(tv.dim_tags, tv.shape):
                    if isinstance(dim_tag, VectorArrayDimTag):
                        align_size *= axis_len

                base_storage_to_align_bytes.setdefault(tv.base_storage, []).append(
                        align_size)

                for idi in decl_info:
                    cast_decl = POD(self, idi.dtype, "")
                    temp_var_decl = POD(self, idi.dtype, idi.name)

                    cast_decl = self.wrap_temporary_decl(cast_decl, tv.scope)
                    temp_var_decl = self.wrap_temporary_decl(
                            temp_var_decl, tv.scope)

                    # The 'restrict' part of this is a complete lie--of course
                    # all these temporaries are aliased. But we're promising to
                    # not use them to shovel data from one representation to the
                    # other. That counts, right?

                    cast_decl = _ConstRestrictPointer(cast_decl)
                    temp_var_decl = _ConstRestrictPointer(temp_var_decl)

                    cast_tp, cast_d = cast_decl.get_decl_pair()
                    temp_var_decl = Initializer(
                            temp_var_decl,
                            "(%s %s) (%s + %s)" % (
                                " ".join(cast_tp), cast_d,
                                tv.base_storage,
                                offset))

                    temp_decls.append(temp_var_decl)

                    from pytools import product
                    offset += (
                            idi.dtype.itemsize
                            * product(si for si in idi.shape))

        ecm = self.get_expression_to_code_mapper(codegen_state)

        for bs_name, bs_sizes in sorted(six.iteritems(base_storage_sizes)):
            bs_var_decl = Value("char", bs_name)
            from pytools import single_valued
            bs_var_decl = self.wrap_temporary_decl(
                    bs_var_decl, single_valued(base_storage_to_scope[bs_name]))

            # FIXME: Could try to use isl knowledge to simplify max.
            if all(isinstance(bs, int) for bs in bs_sizes):
                bs_size_max = max(bs_sizes)
            else:
                bs_size_max = p.Max(tuple(bs_sizes))

            bs_var_decl = ArrayOf(bs_var_decl, ecm(bs_size_max))

            alignment = max(base_storage_to_align_bytes[bs_name])
            bs_var_decl = AlignedAttribute(alignment, bs_var_decl)

            base_storage_decls.append(bs_var_decl)

        # }}}

        result = base_storage_decls + temp_decls

        if result:
            result.append(Line())

        return result

    @property
    def ast_block_class(self):
        from cgen import Block
        return Block

    @property
    def ast_block_scope_class(self):
        return ScopingBlock

    # }}}

    # {{{ code generation guts

    def get_expression_to_code_mapper(self, codegen_state):
        return self.get_expression_to_c_expression_mapper(codegen_state)

    def get_expression_to_c_expression_mapper(self, codegen_state):
        from loopy.target.c.codegen.expression import ExpressionToCExpressionMapper
        return ExpressionToCExpressionMapper(
                codegen_state, fortran_abi=self.target.fortran_abi)

    def get_c_expression_to_code_mapper(self):
        from loopy.target.c.codegen.expression import CExpressionToCodeMapper
        return CExpressionToCodeMapper()

    def get_temporary_decl(self, codegen_state, schedule_index, temp_var, decl_info):
        temp_var_decl = POD(self, decl_info.dtype, decl_info.name)

        if temp_var.read_only:
            from cgen import Const
            temp_var_decl = Const(temp_var_decl)

        if decl_info.shape:
            from cgen import ArrayOf
            ecm = self.get_expression_to_code_mapper(codegen_state)
            temp_var_decl = ArrayOf(temp_var_decl,
                    ecm(p.flattened_product(decl_info.shape),
                        prec=PREC_NONE, type_context="i"))

        return temp_var_decl

    def wrap_temporary_decl(self, decl, scope):
        return decl

    def wrap_global_constant(self, decl):
        return decl

    def get_value_arg_decl(self, name, shape, dtype, is_written):
        assert shape == ()

        result = POD(self, dtype, name)
        if not is_written:
            from cgen import Const
            result = Const(result)

        if self.target.fortran_abi:
            from cgen import Pointer
            result = Pointer(result)

        return result

    def get_global_arg_decl(self, name, shape, dtype, is_written):
        from cgen import RestrictPointer, Const

        arg_decl = RestrictPointer(POD(self, dtype, name))

        if not is_written:
            arg_decl = Const(arg_decl)

        return arg_decl

    def get_constant_arg_decl(self, name, shape, dtype, is_written):
        from loopy.target.c import POD  # uses the correct complex type
        from cgen import RestrictPointer, Const

        arg_decl = RestrictPointer(POD(self, dtype, name))

        if not is_written:
            arg_decl = Const(arg_decl)

        return arg_decl

    def emit_assignment(self, codegen_state, insn):
        kernel = codegen_state.kernel
        ecm = codegen_state.expression_to_code_mapper

        assignee_var_name, = insn.assignee_var_names()

        lhs_var = codegen_state.kernel.get_var_descriptor(assignee_var_name)
        lhs_dtype = lhs_var.dtype

        if insn.atomicity is not None:
            lhs_atomicity = [
                    a for a in insn.atomicity if a.var_name == assignee_var_name]
            assert len(lhs_atomicity) <= 1
            if lhs_atomicity:
                lhs_atomicity, = lhs_atomicity
            else:
                lhs_atomicity = None
        else:
            lhs_atomicity = None

        from loopy.kernel.data import AtomicInit, AtomicUpdate
        from loopy.expression import dtype_to_type_context

        lhs_code = ecm(insn.assignee, prec=PREC_NONE, type_context=None)
        rhs_type_context = dtype_to_type_context(kernel.target, lhs_dtype)
        if lhs_atomicity is None:
            from cgen import Assign
            return Assign(
                    lhs_code,
                    ecm(insn.expression, prec=PREC_NONE,
                        type_context=rhs_type_context,
                        needed_dtype=lhs_dtype))

        elif isinstance(lhs_atomicity, AtomicInit):
            raise NotImplementedError("atomic init")

        elif isinstance(lhs_atomicity, AtomicUpdate):
            codegen_state.seen_atomic_dtypes.add(lhs_dtype)
            return codegen_state.ast_builder.emit_atomic_update(
                    codegen_state, lhs_atomicity, lhs_var,
                    insn.assignee, insn.expression,
                    lhs_dtype, rhs_type_context)

        else:
            raise ValueError("unexpected lhs atomicity type: %s"
                    % type(lhs_atomicity).__name__)

    def emit_atomic_update(self, codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype):
        raise NotImplementedError("atomic updates in %s" % type(self).__name__)

    def emit_tuple_assignment(self, codegen_state, insn):
        ecm = codegen_state.expression_to_code_mapper

        from cgen import Assign, block_if_necessary
        assignments = []

        for i, (assignee, parameter) in enumerate(
                zip(insn.assignees, insn.expression.parameters)):
            lhs_code = ecm(assignee, prec=PREC_NONE, type_context=None)
            assignee_var_name = insn.assignee_var_names()[i]
            lhs_var = codegen_state.kernel.get_var_descriptor(assignee_var_name)
            lhs_dtype = lhs_var.dtype

            from loopy.expression import dtype_to_type_context
            rhs_type_context = dtype_to_type_context(
                    codegen_state.kernel.target, lhs_dtype)
            rhs_code = ecm(parameter, prec=PREC_NONE,
                    type_context=rhs_type_context, needed_dtype=lhs_dtype)

            assignments.append(Assign(lhs_code, rhs_code))

        return block_if_necessary(assignments)

    def emit_multiple_assignment(self, codegen_state, insn):
        ecm = codegen_state.expression_to_code_mapper

        from pymbolic.primitives import Variable
        from pymbolic.mapper.stringifier import PREC_NONE

        func_id = insn.expression.function
        parameters = insn.expression.parameters

        if isinstance(func_id, Variable):
            func_id = func_id.name

        assignee_var_descriptors = [
                codegen_state.kernel.get_var_descriptor(a)
                for a in insn.assignee_var_names()]

        par_dtypes = tuple(ecm.infer_type(par) for par in parameters)

        mangle_result = codegen_state.kernel.mangle_function(func_id, par_dtypes)
        if mangle_result is None:
            raise RuntimeError("function '%s' unknown--"
                    "maybe you need to register a function mangler?"
                    % func_id)

        assert mangle_result.arg_dtypes is not None

        if mangle_result.target_name == "loopy_make_tuple":
            # This shorcut avoids actually having to emit a 'make_tuple' function.
            return self.emit_tuple_assignment(codegen_state, insn)

        from loopy.expression import dtype_to_type_context
        c_parameters = [
                ecm(par, PREC_NONE,
                    dtype_to_type_context(self.target, tgt_dtype),
                    tgt_dtype).expr
                for par, par_dtype, tgt_dtype in zip(
                    parameters, par_dtypes, mangle_result.arg_dtypes)]

        from loopy.codegen import SeenFunction
        codegen_state.seen_functions.add(
                SeenFunction(func_id,
                    mangle_result.target_name,
                    mangle_result.arg_dtypes))

        from pymbolic import var
        for i, (a, tgt_dtype) in enumerate(
                zip(insn.assignees[1:], mangle_result.result_dtypes[1:])):
            if tgt_dtype != ecm.infer_type(a):
                raise LoopyError("type mismatch in %d'th (1-based) left-hand "
                        "side of instruction '%s'" % (i+1, insn.id))
            c_parameters.append(
                        # TODO Yuck: The "where-at function": &(...)
                        var("&")(
                            ecm(a, PREC_NONE,
                                dtype_to_type_context(self.target, tgt_dtype),
                                tgt_dtype).expr))

        from pymbolic import var
        result = var(mangle_result.target_name)(*c_parameters)

        # In case of no assignees, we are done
        if len(mangle_result.result_dtypes) == 0:
            from cgen import ExpressionStatement
            return ExpressionStatement(
                    CExpression(self.get_c_expression_to_code_mapper(), result))

        result = ecm.wrap_in_typecast(
                mangle_result.result_dtypes[0],
                assignee_var_descriptors[0].dtype,
                result)

        lhs_code = ecm(insn.assignees[0], prec=PREC_NONE, type_context=None)

        from cgen import Assign
        return Assign(
                lhs_code,
                CExpression(self.get_c_expression_to_code_mapper(), result))

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            lbound, ubound, inner):
        ecm = codegen_state.expression_to_code_mapper

        from pymbolic import var
        from pymbolic.primitives import Comparison
        from pymbolic.mapper.stringifier import PREC_NONE
        from cgen import For, InlineInitializer

        return For(
                InlineInitializer(
                    POD(self, iname_dtype, iname),
                    ecm(lbound, PREC_NONE, "i")),
                ecm(
                    Comparison(
                        var(iname),
                        "<=",
                        ubound),
                    PREC_NONE, "i"),
                "++%s" % iname,
                inner)

    def emit_initializer(self, codegen_state, dtype, name, val_str, is_const):
        decl = POD(self, dtype, name)

        from cgen import Initializer, Const

        if is_const:
            decl = Const(decl)

        return Initializer(decl, val_str)

    def emit_blank_line(self):
        from cgen import Line
        return Line()

    def emit_comment(self, s):
        from cgen import Comment
        return Comment(s)

    def emit_if(self, condition_str, ast):
        from cgen import If
        return If(condition_str, ast)

    # }}}

    def process_ast(self, node):
        sc = ASTSubscriptCollector()
        sc(node)
        return node


# {{{ header generation

class CFunctionDeclExtractor(CASTIdentityMapper):
    def __init__(self):
        self.decls = []

    def map_expression(self, expr):
        return expr

    def map_function_decl_wrapper(self, node):
        self.decls.append(node.subdecl)
        return super(CFunctionDeclExtractor, self)\
                .map_function_decl_wrapper(node)


def generate_header(kernel, codegen_result=None):
    """
    :arg kernel: a :class:`loopy.LoopKernel`
    :arg codegen_result: an instance of :class:`loopy.CodeGenerationResult`
    :returns: a list of AST nodes (which may have :func:`str`
        called on them to produce a string) representing
        function declarations for the generated device
        functions.
    """

    if not isinstance(kernel.target, CTarget):
        raise LoopyError(
                'Header generation for non C-based languages are not implemented')

    if codegen_result is None:
        from loopy.codegen import generate_code_v2
        codegen_result = generate_code_v2(kernel)

    fde = CFunctionDeclExtractor()
    for dev_prg in codegen_result.device_programs:
        fde(dev_prg.ast)

    return fde.decls

# }}}

# vim: foldmethod=marker

"""OpenCL target independent of PyOpenCL."""

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
from loopy.target import TargetBase
from loopy.diagnostic import LoopyError

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


class CTarget(TargetBase):
    """A target for plain "C", without any parallel extensions.
    """

    hash_fields = TargetBase.hash_fields + ("fortran_abi",)
    comparison_fields = TargetBase.comparison_fields + ("fortran_abi",)

    def __init__(self, fortran_abi=False):
        self.fortran_abi = fortran_abi
        super(CTarget, self).__init__()

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

    # {{{ library

    def preamble_generators(self):
        return (
                super(CTarget, self).preamble_generators() + [
                    _preamble_generator,
                    ])

    # }}}

    # {{{ code generation

    def generate_code(self, kernel, codegen_state, impl_arg_info):
        from cgen import FunctionBody, FunctionDeclaration, Value, Module

        body, implemented_domains = kernel.target.generate_body(
                kernel, codegen_state)

        name = kernel.name
        if self.fortran_abi:
            name += "_"

        mod = Module([
            FunctionBody(
                kernel.target.wrap_function_declaration(
                    kernel,
                    FunctionDeclaration(
                        Value("void", name),
                        [iai.cgen_declarator for iai in impl_arg_info])),
                body)
            ])

        return str(mod), implemented_domains

    def wrap_function_declaration(self, kernel, fdecl):
        return fdecl

    def generate_body(self, kernel, codegen_state):
        from cgen import Block
        body = Block()

        temp_decls = []

        # {{{ declare temporaries

        base_storage_sizes = {}
        base_storage_to_scope = {}
        base_storage_to_align_bytes = {}

        from cgen import ArrayOf, Pointer, Initializer, AlignedAttribute, Value
        from loopy.codegen import POD  # uses the correct complex type

        class ConstRestrictPointer(Pointer):
            def get_decl_pair(self):
                sub_tp, sub_decl = self.subdecl.get_decl_pair()
                return sub_tp, ("*const restrict %s" % sub_decl)

        for tv in sorted(
                six.itervalues(kernel.temporary_variables),
                key=lambda tv: tv.name):
            decl_info = tv.decl_info(self, index_dtype=kernel.index_dtype)

            if not tv.base_storage:
                for idi in decl_info:
                    temp_decls.append(
                            self.wrap_temporary_decl(
                                self.get_temporary_decl(kernel, tv, idi), tv.scope))

            else:
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

                    cast_decl = ConstRestrictPointer(cast_decl)
                    temp_var_decl = ConstRestrictPointer(temp_var_decl)

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

        for bs_name, bs_sizes in sorted(six.iteritems(base_storage_sizes)):
            bs_var_decl = Value("char", bs_name)
            from pytools import single_valued
            bs_var_decl = self.wrap_temporary_decl(
                    bs_var_decl, single_valued(base_storage_to_scope[bs_name]))
            bs_var_decl = ArrayOf(bs_var_decl, max(bs_sizes))

            alignment = max(base_storage_to_align_bytes[bs_name])
            bs_var_decl = AlignedAttribute(alignment, bs_var_decl)

            body.append(bs_var_decl)

        body.extend(temp_decls)

        # }}}

        from loopy.codegen.loop import set_up_hw_parallel_loops
        gen_code = set_up_hw_parallel_loops(kernel, 0, codegen_state)

        from cgen import Line
        body.append(Line())

        if isinstance(gen_code.ast, Block):
            body.extend(gen_code.ast.contents)
        else:
            body.append(gen_code.ast)

        return body, gen_code.implemented_domains

    # }}}

    # {{{ code generation guts

    def get_expression_to_code_mapper(self, codegen_state):
        from loopy.target.c.codegen.expression import LoopyCCodeMapper
        return LoopyCCodeMapper(codegen_state, fortran_abi=self.fortran_abi)

    def get_temporary_decl(self, knl, temp_var, decl_info):
        from loopy.codegen import POD  # uses the correct complex type
        temp_var_decl = POD(self, decl_info.dtype, decl_info.name)

        if decl_info.shape:
            from cgen import ArrayOf
            temp_var_decl = ArrayOf(temp_var_decl,
                    " * ".join(str(s) for s in decl_info.shape))

        return temp_var_decl

    def wrap_temporary_decl(self, decl, scope):
        return decl

    def get_value_arg_decl(self, name, shape, dtype, is_written):
        assert shape == ()

        from loopy.codegen import POD  # uses the correct complex type
        result = POD(self, dtype, name)
        if not is_written:
            from cgen import Const
            result = Const(result)

        if self.fortran_abi:
            from cgen import Pointer
            result = Pointer(result)

        return result

    def get_global_arg_decl(self, name, shape, dtype, is_written):
        from loopy.codegen import POD  # uses the correct complex type
        from cgen import RestrictPointer, Const

        arg_decl = RestrictPointer(POD(self, dtype, name))

        if not is_written:
            arg_decl = Const(arg_decl)

        return arg_decl

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            static_lbound, static_ubound, inner):
        ecm = codegen_state.expression_to_code_mapper

        from loopy.symbolic import aff_to_expr

        from loopy.codegen import wrap_in
        from pymbolic.mapper.stringifier import PREC_NONE
        from cgen import For

        return wrap_in(For,
                "%s %s = %s"
                % (self.dtype_to_typename(iname_dtype),
                    iname, ecm(aff_to_expr(static_lbound), PREC_NONE, "i")),
                "%s <= %s" % (
                    iname, ecm(aff_to_expr(static_ubound), PREC_NONE, "i")),
                "++%s" % iname,
                inner)

    def generate_multiple_assignment(self, codegen_state, insn):
        ecm = codegen_state.expression_to_code_mapper

        from pymbolic.primitives import Variable
        from pymbolic.mapper.stringifier import PREC_NONE

        func_id = insn.expression.function
        parameters = insn.expression.parameters

        if isinstance(func_id, Variable):
            func_id = func_id.name

        assignee_var_descriptors = [codegen_state.kernel.get_var_descriptor(a)
                for a, _ in insn.assignees_and_indices()]

        par_dtypes = tuple(ecm.infer_type(par) for par in parameters)

        str_parameters = None

        mangle_result = codegen_state.kernel.mangle_function(func_id, par_dtypes)
        if mangle_result is None:
            raise RuntimeError("function '%s' unknown--"
                    "maybe you need to register a function mangler?"
                    % func_id)

        assert mangle_result.arg_dtypes is not None

        from loopy.expression import dtype_to_type_context
        str_parameters = [
                ecm(par, PREC_NONE,
                    dtype_to_type_context(self, tgt_dtype),
                    tgt_dtype)
                for par, par_dtype, tgt_dtype in zip(
                    parameters, par_dtypes, mangle_result.arg_dtypes)]

        from loopy.codegen import SeenFunction
        codegen_state.seen_functions.add(
                SeenFunction(func_id,
                    mangle_result.target_name,
                    mangle_result.arg_dtypes))

        for a, tgt_dtype in zip(insn.assignees[1:], mangle_result.result_dtypes[1:]):
            if tgt_dtype != ecm.infer_type(a):
                raise LoopyError("type mismatch in %d'th (1-based) left-hand "
                        "side of instruction '%s'" % (insn.id))
            str_parameters.append(
                    "&(%s)" % ecm(a, PREC_NONE,
                        dtype_to_type_context(self, tgt_dtype),
                        tgt_dtype))

        result = "%s(%s)" % (mangle_result.target_name, ", ".join(str_parameters))

        result = ecm.wrap_in_typecast(
                mangle_result.result_dtypes[0],
                assignee_var_descriptors[0].dtype,
                result)

        lhs_code = ecm(insn.assignees[0], prec=PREC_NONE, type_context=None)

        from cgen import Assign
        return Assign(
                lhs_code,
                result)

    # }}}

# vim: foldmethod=marker

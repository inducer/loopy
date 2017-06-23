from __future__ import division, absolute_import

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


from six.moves import range, zip

import numpy as np

from pymbolic.mapper import RecursiveMapper, IdentityMapper
from pymbolic.mapper.stringifier import (PREC_NONE, PREC_CALL, PREC_PRODUCT,
        PREC_POWER,
        PREC_UNARY, PREC_LOGICAL_OR, PREC_LOGICAL_AND)
import islpy as isl
import pymbolic.primitives as p
from pymbolic import var


from loopy.expression import dtype_to_type_context
from loopy.type_inference import TypeInferenceMapper

from loopy.diagnostic import LoopyError, LoopyWarning
from loopy.tools import is_integer
from loopy.types import LoopyType


# {{{ Loopy expression to C expression mapper

class ExpressionToCExpressionMapper(IdentityMapper):
    def __init__(self, codegen_state, fortran_abi=False, type_inf_mapper=None):
        self.kernel = codegen_state.kernel
        self.codegen_state = codegen_state

        if type_inf_mapper is None:
            type_inf_mapper = TypeInferenceMapper(self.kernel)
        self.type_inf_mapper = type_inf_mapper

        self.allow_complex = codegen_state.allow_complex

        self.fortran_abi = fortran_abi

    # {{{ helpers

    def with_assignments(self, names_to_vars):
        type_inf_mapper = self.type_inf_mapper.with_assignments(names_to_vars)
        return type(self)(self.codegen_state, self.fortran_abi, type_inf_mapper)

    def infer_type(self, expr):
        result = self.type_inf_mapper(expr)
        assert isinstance(result, LoopyType)

        self.codegen_state.seen_dtypes.add(result)
        return result

    def find_array(self, expr):
        if expr.aggregate.name in self.kernel.arg_dict:
            ary = self.kernel.arg_dict[expr.aggregate.name]
        elif expr.aggregate.name in self.kernel.temporary_variables:
            ary = self.kernel.temporary_variables[expr.aggregate.name]
        else:
            raise RuntimeError("nothing known about subscripted variable '%s'"
                    % expr.aggregate.name)

        from loopy.kernel.array import ArrayBase
        if not isinstance(ary, ArrayBase):
            raise RuntimeError("subscripted variable '%s' is not an array"
                    % expr.aggregate.name)

        return ary

    def wrap_in_typecast(self, actual_type, needed_dtype, s):
        if (actual_type.is_complex() and needed_dtype.is_complex()
                and actual_type != needed_dtype):
            return var("%s_cast" % self.complex_type_name(needed_dtype))(s)
        elif not actual_type.is_complex() and needed_dtype.is_complex():
            return var("%s_fromreal" % self.complex_type_name(needed_dtype))(s)
        else:
            return s

    def rec(self, expr, type_context=None, needed_dtype=None):
        if needed_dtype is None:
            return RecursiveMapper.rec(self, expr, type_context)

        return self.wrap_in_typecast(
                self.infer_type(expr), needed_dtype,
                RecursiveMapper.rec(self, expr, type_context))

    def __call__(self, expr, prec=None, type_context=None, needed_dtype=None):
        if prec is None:
            prec = PREC_NONE

        assert prec == PREC_NONE
        from loopy.target.c import CExpression
        return CExpression(
                self.codegen_state.ast_builder.get_c_expression_to_code_mapper(),
                self.rec(expr, type_context, needed_dtype))

    # }}}

    def map_variable(self, expr, type_context):
        def postproc(x):
            return x

        if expr.name in self.codegen_state.var_subst_map:
            if self.kernel.options.annotate_inames:
                return var(
                        "/* %s */ %s" % (
                            expr.name,
                            self.rec(self.codegen_state.var_subst_map[expr.name],
                                type_context)))
            else:
                return self.rec(self.codegen_state.var_subst_map[expr.name],
                    type_context)
        elif expr.name in self.kernel.arg_dict:
            arg = self.kernel.arg_dict[expr.name]
            from loopy.kernel.array import ArrayBase
            if isinstance(arg, ArrayBase):
                if arg.shape == ():
                    if arg.offset:
                        # FIXME
                        raise NotImplementedError("in-memory scalar with offset")
                    else:
                        return var(expr.name)[0]
                else:
                    raise RuntimeError("unsubscripted reference to array '%s'"
                            % expr.name)

            from loopy.kernel.data import ValueArg
            if isinstance(arg, ValueArg) and self.fortran_abi:
                postproc = lambda x: x[0]  # noqa
        elif expr.name in self.kernel.temporary_variables:
            temporary = self.kernel.temporary_variables[expr.name]
            if temporary.base_storage:
                postproc = lambda x: x[0]  # noqa

        result = self.kernel.mangle_symbol(self.codegen_state.ast_builder, expr.name)
        if result is not None:
            _, c_name = result
            return postproc(var(c_name))

        return postproc(var(expr.name))

    def map_tagged_variable(self, expr, type_context):
        return var(expr.name)

    def map_subscript(self, expr, type_context):
        def base_impl(expr, type_context):
            return self.rec(expr.aggregate, type_context)[self.rec(expr.index, 'i')]

        def make_var(name):
            from loopy import TaggedVariable
            if isinstance(expr.aggregate, TaggedVariable):
                return TaggedVariable(name, expr.aggregate.tag)
            else:
                return var(name)

        from pymbolic.primitives import Variable
        if not isinstance(expr.aggregate, Variable):
            return base_impl(expr, type_context)

        ary = self.find_array(expr)

        from loopy.kernel.array import get_access_info
        from pymbolic import evaluate

        from loopy.symbolic import simplify_using_aff
        index_tuple = tuple(
                simplify_using_aff(self.kernel, idx) for idx in expr.index_tuple)

        access_info = get_access_info(self.kernel.target, ary, index_tuple,
                lambda expr: evaluate(expr, self.codegen_state.var_subst_map),
                self.codegen_state.vectorization_info)

        from loopy.kernel.data import (
                ImageArg, GlobalArg, TemporaryVariable, ConstantArg)

        if isinstance(ary, ImageArg):
            extra_axes = 0

            num_target_axes = ary.num_target_axes()
            if num_target_axes in [1, 2]:
                idx_vec_type = "float2"
                extra_axes = 2-num_target_axes
            elif num_target_axes == 3:
                idx_vec_type = "float4"
                extra_axes = 4-num_target_axes
            else:
                raise LoopyError("unsupported number (%d) of target axes in image"
                        % num_target_axes)

            idx_tuple = expr.index_tuple[::-1] + (0,) * extra_axes

            base_access = var("read_imagef")(
                    var(ary.name),
                    var("loopy_sampler"),
                    var("(%s)" % idx_vec_type)(*self.rec(idx_tuple, 'i')))

            if ary.dtype.numpy_dtype == np.float32:
                return base_access.attr("x")
            if self.kernel.target.is_vector_dtype(ary.dtype):
                return base_access
            elif ary.dtype.numpy_dtype == np.float64:
                return var("as_double")(base_access.attr("xy"))
            else:
                raise NotImplementedError(
                        "non-floating-point images not supported for now")

        elif isinstance(ary, (GlobalArg, TemporaryVariable, ConstantArg)):
            if len(access_info.subscripts) == 0:
                if (
                        (isinstance(ary, (ConstantArg, GlobalArg)) or
                         (isinstance(ary, TemporaryVariable) and ary.base_storage))):
                    # unsubscripted global args are pointers
                    result = make_var(access_info.array_name)[0]

                else:
                    # unsubscripted temp vars are scalars
                    # (unless they use base_storage)
                    result = make_var(access_info.array_name)

            else:
                subscript, = access_info.subscripts
                result = make_var(access_info.array_name)[self.rec(subscript, 'i')]

            if access_info.vector_index is not None:
                return self.codegen_state.ast_builder.add_vector_access(
                    result, access_info.vector_index)
            else:
                return result

        else:
            assert False

    def map_linear_subscript(self, expr, type_context):
        from pymbolic.primitives import Variable
        if not isinstance(expr.aggregate, Variable):
                raise RuntimeError("linear indexing on non-variable: %s"
                        % expr)

        if expr.aggregate.name in self.kernel.arg_dict:
            arg = self.kernel.arg_dict[expr.aggregate.name]

            from loopy.kernel.data import ImageArg
            if isinstance(arg, ImageArg):
                raise RuntimeError("linear indexing is not supported on images: %s"
                        % expr)

            else:
                # GlobalArg
                if arg.offset:
                    offset = Variable(arg.offset)
                else:
                    offset = 0

                return var(expr.aggregate.name)[
                        self.rec(offset + expr.index, 'i')]

        elif expr.aggregate.name in self.kernel.temporary_variables:
            raise RuntimeError("linear indexing is not supported on temporaries: %s"
                    % expr)

        else:
            raise RuntimeError(
                    "nothing known about variable '%s'" % expr.aggregate.name)

    def map_floor_div(self, expr, type_context):
        from loopy.symbolic import get_dependencies
        iname_deps = get_dependencies(expr) & self.kernel.all_inames()
        domain = self.kernel.get_inames_domain(iname_deps)

        assumption_non_param = isl.BasicSet.from_params(self.kernel.assumptions)
        assumptions, domain = isl.align_two(assumption_non_param, domain)
        domain = domain & assumptions

        from loopy.isl_helpers import is_nonnegative
        num_nonneg = is_nonnegative(expr.numerator, domain)
        den_nonneg = is_nonnegative(expr.denominator, domain)

        def seen_func(name):
            idt = self.kernel.index_dtype
            from loopy.codegen import SeenFunction
            self.codegen_state.seen_functions.add(
                    SeenFunction(name, name, (idt, idt)))

        if den_nonneg:
            if num_nonneg:
                # parenthesize to avoid negative signs being dragged in from the
                # outside by associativity
                return (
                        self.rec(expr.numerator, type_context)
                        //
                        self.rec(expr.denominator, type_context))
            else:
                seen_func("int_floor_div_pos_b")
                return var("int_floor_div_pos_b")(
                        self.rec(expr.numerator, 'i'),
                        self.rec(expr.denominator, 'i'))
        else:
            seen_func("int_floor_div")
            return var("int_floor_div")(
                    self.rec(expr.numerator, 'i'),
                    self.rec(expr.denominator, 'i'))

    def map_if(self, expr, type_context):
        return type(expr)(
                self.rec(expr.condition, "i"),
                self.rec(expr.then, type_context),
                self.rec(expr.else_, type_context),
                )

    def map_comparison(self, expr, type_context):
        inner_type_context = dtype_to_type_context(
                self.kernel.target,
                self.infer_type(expr.left - expr.right))

        return type(expr)(
                    self.rec(expr.left, inner_type_context),
                    expr.operator,
                    self.rec(expr.right, inner_type_context))

    def map_constant(self, expr, type_context):
        if isinstance(expr, (complex, np.complexfloating)):
            try:
                dtype = expr.dtype
            except AttributeError:
                # (COMPLEX_GUESS_LOGIC)
                # This made it through type 'guessing' above, and it
                # was concluded above (search for COMPLEX_GUESS_LOGIC),
                # that nothing was lost by using single precision.
                cast_type = "cfloat"
            else:
                if dtype == np.complex128:
                    cast_type = "cdouble"
                elif dtype == np.complex64:
                    cast_type = "cfloat"
                else:
                    raise RuntimeError("unsupported complex type in expression "
                            "generation: %s" % type(expr))

            return var("%s_new" % cast_type)(expr.real, expr.imag)
        else:
            from loopy.symbolic import Literal
            if type_context == "f":
                return Literal(repr(float(expr))+"f")
            elif type_context == "d":
                return Literal(repr(float(expr)))
            elif type_context == "i":
                return int(expr)
            else:
                if is_integer(expr):
                    return int(expr)

                raise RuntimeError("don't know how to generate code "
                        "for constant '%s'" % expr)

    def map_call(self, expr, type_context):
        from pymbolic.primitives import Variable, Subscript

        identifier = expr.function

        # {{{ implement indexof, indexof_vec

        if identifier.name in ["indexof", "indexof_vec"]:
            if len(expr.parameters) != 1:
                raise LoopyError("%s takes exactly one argument" % identifier.name)
            arg, = expr.parameters
            if not isinstance(arg, Subscript):
                raise LoopyError(
                        "argument to %s must be a subscript" % identifier.name)

            ary = self.find_array(arg)

            from loopy.kernel.array import get_access_info
            from pymbolic import evaluate
            access_info = get_access_info(self.kernel.target, ary, arg.index,
                    lambda expr: evaluate(expr, self.codegen_state.var_subst_map),
                    self.codegen_state.vectorization_info)

            from loopy.kernel.data import ImageArg
            if isinstance(ary, ImageArg):
                raise LoopyError("%s does not support images" % identifier.name)

            if identifier.name == "indexof":
                return access_info.subscripts[0]
            elif identifier.name == "indexof_vec":
                from loopy.kernel.array import VectorArrayDimTag
                ivec = None
                for iaxis, dim_tag in enumerate(ary.dim_tags):
                    if isinstance(dim_tag, VectorArrayDimTag):
                        ivec = iaxis

                if ivec is None:
                    return access_info.subscripts[0]
                else:
                    return (
                        access_info.subscripts[0]*ary.shape[ivec]
                        + access_info.vector_index)

            else:
                raise RuntimeError("should not get here")

        # }}}

        if isinstance(identifier, Variable):
            identifier = identifier.name

        par_dtypes = tuple(self.infer_type(par) for par in expr.parameters)

        processed_parameters = None

        mangle_result = self.kernel.mangle_function(
                identifier, par_dtypes,
                ast_builder=self.codegen_state.ast_builder)

        if mangle_result is None:
            raise RuntimeError("function '%s' unknown--"
                    "maybe you need to register a function mangler?"
                    % identifier)

        if len(mangle_result.result_dtypes) != 1:
            raise LoopyError("functions with more or fewer than one return value "
                    "may not be used in an expression")

        if mangle_result.arg_dtypes is not None:
            processed_parameters = tuple(
                    self.rec(par,
                        dtype_to_type_context(self.kernel.target, tgt_dtype),
                        tgt_dtype)
                    for par, par_dtype, tgt_dtype in zip(
                        expr.parameters, par_dtypes, mangle_result.arg_dtypes))

        else:
            # /!\ FIXME For some functions (e.g. 'sin'), it makes sense to
            # propagate the type context here. But for many others, it does
            # not. Using the inferred type as a stopgap for now.
            processed_parameters = tuple(
                    self.rec(par,
                        type_context=dtype_to_type_context(
                            self.kernel.target, par_dtype))
                    for par, par_dtype in zip(expr.parameters, par_dtypes))

            from warnings import warn
            warn("Calling function '%s' with unknown C signature--"
                    "return CallMangleInfo.arg_dtypes"
                    % identifier, LoopyWarning)

        from loopy.codegen import SeenFunction
        self.codegen_state.seen_functions.add(
                SeenFunction(identifier,
                    mangle_result.target_name,
                    mangle_result.arg_dtypes or par_dtypes))

        return var(mangle_result.target_name)(*processed_parameters)

    # {{{ deal with complex-valued variables

    def complex_type_name(self, dtype):
        from loopy.types import NumpyType
        if not isinstance(dtype, NumpyType):
            raise LoopyError("'%s' is not a complex type" % dtype)

        if dtype.dtype == np.complex64:
            return "cfloat"
        if dtype.dtype == np.complex128:
            return "cdouble"
        else:
            raise RuntimeError

    def map_sum(self, expr, type_context):
        def base_impl(expr, type_context):
            return super(ExpressionToCExpressionMapper, self).map_sum(
                    expr, type_context)

        # I've added 'type_context == "i"' because of the following
        # idiotic corner case: Code generation for subscripts comes
        # through here, and it may involve variables that we know
        # nothing about (offsets and such). If we fall into the allow_complex
        # branch, we'll try to do type inference on these variables,
        # and stuff breaks. This band-aid works around that. -AK
        if not self.allow_complex or type_context == "i":
            return base_impl(expr, type_context)

        tgt_dtype = self.infer_type(expr)
        is_complex = tgt_dtype.is_complex()

        if not is_complex:
            return base_impl(expr, type_context)
        else:
            tgt_name = self.complex_type_name(tgt_dtype)

            reals = []
            complexes = []
            for child in expr.children:
                if self.infer_type(child).is_complex():
                    complexes.append(child)
                else:
                    reals.append(child)

            real_sum = p.flattened_sum([self.rec(r, type_context) for r in reals])

            complex_sum = self.rec(complexes[0], type_context, tgt_dtype)
            for child in complexes[1:]:
                complex_sum = var("%s_add" % tgt_name)(
                        complex_sum,
                        self.rec(child, type_context, tgt_dtype))

            if real_sum:
                return var("%s_radd" % tgt_name)(real_sum, complex_sum)
            else:
                return complex_sum

    def map_product(self, expr, type_context):
        def base_impl(expr, type_context):
            return super(ExpressionToCExpressionMapper, self).map_product(
                    expr, type_context)

        # I've added 'type_context == "i"' because of the following
        # idiotic corner case: Code generation for subscripts comes
        # through here, and it may involve variables that we know
        # nothing about (offsets and such). If we fall into the allow_complex
        # branch, we'll try to do type inference on these variables,
        # and stuff breaks. This band-aid works around that. -AK
        if not self.allow_complex or type_context == "i":
            return base_impl(expr, type_context)

        tgt_dtype = self.infer_type(expr)
        is_complex = tgt_dtype.is_complex()

        if not is_complex:
            return base_impl(expr, type_context)
        else:
            tgt_name = self.complex_type_name(tgt_dtype)

            reals = []
            complexes = []
            for child in expr.children:
                if self.infer_type(child).is_complex():
                    complexes.append(child)
                else:
                    reals.append(child)

            real_prd = p.flattened_product(
                    [self.rec(r, type_context) for r in reals])

            complex_prd = self.rec(complexes[0], type_context, tgt_dtype)
            for child in complexes[1:]:
                complex_prd = var("%s_mul" % tgt_name)(
                        complex_prd,
                        self.rec(child, type_context, tgt_dtype))

            if real_prd:
                return var("%s_rmul" % tgt_name)(real_prd, complex_prd)
            else:
                return complex_prd

    def map_quotient(self, expr, type_context):
        def base_impl(expr, type_context, num_tgt_dtype=None):
            num = self.rec(expr.numerator, type_context, num_tgt_dtype)

            # analogous to ^{-1}
            denom = self.rec(expr.denominator, type_context)

            if (n_dtype.kind not in "fc"
                    and d_dtype.kind not in "fc"):
                # must both be integers
                if type_context == "f":
                    num = var("(float) ")(num)
                    denom = var("(float) ")(denom)
                elif type_context == "d":
                    num = var("(double) ")(num)
                    denom = var("(double) ")(denom)

            return type(expr)(num, denom)

        n_dtype = self.infer_type(expr.numerator).numpy_dtype
        d_dtype = self.infer_type(expr.denominator).numpy_dtype

        if not self.allow_complex:
            return base_impl(expr, type_context)

        n_complex = 'c' == n_dtype.kind
        d_complex = 'c' == d_dtype.kind

        tgt_dtype = self.infer_type(expr)

        if not (n_complex or d_complex):
            return base_impl(expr, type_context)
        elif n_complex and not d_complex:
            return var("%s_divider" % self.complex_type_name(tgt_dtype))(
                    self.rec(expr.numerator, type_context, tgt_dtype),
                    self.rec(expr.denominator, type_context))
        elif not n_complex and d_complex:
            return var("%s_rdivide" % self.complex_type_name(tgt_dtype))(
                    self.rec(expr.numerator, type_context),
                    self.rec(expr.denominator, type_context, tgt_dtype))
        else:
            return var("%s_divide" % self.complex_type_name(tgt_dtype))(
                    self.rec(expr.numerator, type_context, tgt_dtype),
                    self.rec(expr.denominator, type_context, tgt_dtype))

    def map_remainder(self, expr, type_context):
        tgt_dtype = self.infer_type(expr)
        if tgt_dtype.is_complex():
            raise RuntimeError("complex remainder not defined")

        return super(ExpressionToCExpressionMapper, self).map_remainder(
                expr, type_context)

    def map_power(self, expr, type_context):
        def base_impl(expr, type_context):
            from pymbolic.primitives import is_constant, is_zero
            if is_constant(expr.exponent):
                if is_zero(expr.exponent):
                    return 1
                elif is_zero(expr.exponent - 1):
                    return self.rec(expr.base, type_context)
                elif is_zero(expr.exponent - 2):
                    return self.rec(expr.base*expr.base, type_context)

            return type(expr)(
                    self.rec(expr.base, type_context),
                    self.rec(expr.exponent, type_context))

        if not self.allow_complex:
            return base_impl(expr, type_context)

        tgt_dtype = self.infer_type(expr)
        if tgt_dtype.is_complex():
            if expr.exponent in [2, 3, 4]:
                value = expr.base
                for i in range(expr.exponent-1):
                    value = value * expr.base
                return self.rec(value, type_context)
            else:
                b_complex = self.infer_type(expr.base).is_complex()
                e_complex = self.infer_type(expr.exponent).is_complex()

                if b_complex and not e_complex:
                    return var("%s_powr" % self.complex_type_name(tgt_dtype))(
                            self.rec(expr.base, type_context, tgt_dtype),
                            self.rec(expr.exponent, type_context))
                else:
                    return var("%s_pow" % self.complex_type_name(tgt_dtype))(
                            self.rec(expr.base, type_context, tgt_dtype),
                            self.rec(expr.exponent, type_context, tgt_dtype))

        return base_impl(expr, type_context)

    # }}}

    def map_group_hw_index(self, expr, type_context):
        raise LoopyError("plain C does not have group hw axes")

    def map_local_hw_index(self, expr, type_context):
        raise LoopyError("plain C does not have local hw axes")

# }}}


# {{{ C expression to code mapper

class CExpressionToCodeMapper(RecursiveMapper):
    # {{{ helpers

    def parenthesize_if_needed(self, s, enclosing_prec, my_prec):
        if enclosing_prec > my_prec:
            return "(%s)" % s
        else:
            return s

    def join_rec(self, joiner, iterable, prec):
        f = joiner.join("%s" for i in iterable)
        return f % tuple(
                self.rec(i, prec) for i in iterable)

    def join(self, joiner, iterable):
        f = joiner.join("%s" for i in iterable)
        return f % tuple(iterable)

    # }}}

    def map_constant(self, expr, prec):
        return repr(expr)

    def map_call(self, expr, enclosing_prec):
        from pymbolic.primitives import Variable
        from pymbolic.mapper.stringifier import PREC_NONE, PREC_CALL
        if isinstance(expr.function, Variable):
            func = expr.function.name
        else:
            func = self.rec(expr.function, PREC_CALL)

        return self.parenthesize_if_needed(
                "%s(%s)" % (
                    func,
                    self.join_rec(", ", expr.parameters, PREC_NONE)),
                enclosing_prec, PREC_CALL)

    def map_common_subexpression(self, expr, prec):
        raise RuntimeError("common subexpression should have been eliminated upon "
                "entry to loopy")

    def map_variable(self, expr, enclosing_prec):
        return expr.name

    map_tagged_variable = map_variable

    def map_lookup(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
                "%s.%s" % (
                    self.rec(expr.aggregate, PREC_CALL), expr.name),
                enclosing_prec, PREC_CALL)

    def map_subscript(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
                "%s[%s]" % (
                    self.rec(expr.aggregate, PREC_CALL),
                    self.rec(expr.index, PREC_NONE)),
                enclosing_prec, PREC_CALL)

    def map_floor_div(self, expr, enclosing_prec):
        # parenthesize to avoid negative signs being dragged in from the
        # outside by associativity
        return "(%s / %s)" % (
                    self.rec(expr.numerator, PREC_PRODUCT),
                    # analogous to ^{-1}
                    self.rec(expr.denominator, PREC_POWER))

    def map_min(self, expr, enclosing_prec):
        what = type(expr).__name__.lower()

        children = list(expr.children)

        result = self.rec(children.pop(), PREC_NONE)
        while children:
            result = "%s(%s, %s)" % (what,
                        self.rec(children.pop(), PREC_NONE),
                        result)

        return result

    map_max = map_min

    def map_if(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "(%s ? %s : %s)" % (
                self.rec(expr.condition, PREC_NONE),
                self.rec(expr.then, PREC_NONE),
                self.rec(expr.else_, PREC_NONE),
                )

    def map_comparison(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_COMPARISON

        return self.parenthesize_if_needed(
                "%s %s %s" % (
                    self.rec(expr.left, PREC_COMPARISON),
                    expr.operator,
                    self.rec(expr.right, PREC_COMPARISON)),
                enclosing_prec, PREC_COMPARISON)

    def map_literal(self, expr, enclosing_prec):
        return expr.s

    def map_logical_not(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
                "!" + self.rec(expr.child, PREC_UNARY),
                enclosing_prec, PREC_UNARY)

    def map_logical_and(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
                self.join_rec(" && ", expr.children, PREC_LOGICAL_AND),
                enclosing_prec, PREC_LOGICAL_AND)

    def map_logical_or(self, expr, enclosing_prec):
        mapped_children = []
        from pymbolic.primitives import LogicalAnd
        for child in expr.children:
            mapped_child = self.rec(child, PREC_LOGICAL_OR)
            # clang warns on unparenthesized && within ||
            if isinstance(child, LogicalAnd):
                mapped_child = "(%s)" % mapped_child
            mapped_children.append(mapped_child)

        result = self.join(" || ", mapped_children)
        if enclosing_prec > PREC_LOGICAL_OR:
            result = "(%s)" % result
        return result

    def map_sum(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_SUM

        return self.parenthesize_if_needed(
                self.join_rec(" + ", expr.children, PREC_SUM),
                enclosing_prec, PREC_SUM)

    def map_product(self, expr, enclosing_prec):
        # Spaces prevent '**z' (times dereference z), which
        # is hard to read.
        return self.parenthesize_if_needed(
                self.join_rec(" * ", expr.children, PREC_PRODUCT),
                enclosing_prec, PREC_PRODUCT)

    def map_quotient(self, expr, enclosing_prec):
        num = self.rec(expr.numerator, PREC_PRODUCT)

        # analogous to ^{-1}
        denom = self.rec(expr.denominator, PREC_POWER)

        return self.parenthesize_if_needed(
                "%s / %s" % (
                    # Space is necessary--otherwise '/*'
                    # (i.e. divide-dererference) becomes
                    # start-of-comment in C.
                    num,
                    denom),
                enclosing_prec, PREC_PRODUCT)

    def map_remainder(self, expr, enclosing_prec):
        return "(%s %% %s)" % (
                    self.rec(expr.numerator, PREC_PRODUCT),
                    # PREC_POWER analogous to ^{-1}
                    self.rec(expr.denominator, PREC_POWER))

    def map_power(self, expr, enclosing_prec):
        return "pow(%s, %s)" % (
                self.rec(expr.base, PREC_NONE),
                self.rec(expr.exponent, PREC_NONE))

    def map_array_literal(self, expr, enclosing_prec):
        return "{ %s }" % self.join_rec(", ", expr.children, PREC_NONE)

# }}}

# vim: fdm=marker

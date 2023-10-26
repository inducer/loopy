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


import numpy as np

from pymbolic.mapper import RecursiveMapper, IdentityMapper
from pymbolic.mapper.stringifier import (PREC_NONE, PREC_CALL, PREC_PRODUCT,
        PREC_SHIFT,
        PREC_UNARY, PREC_LOGICAL_OR, PREC_LOGICAL_AND,
        PREC_BITWISE_AND, PREC_BITWISE_OR, PREC_BITWISE_XOR)

import islpy as isl
import pymbolic.primitives as p
from pymbolic import var


from loopy.expression import dtype_to_type_context
from loopy.type_inference import TypeReader

from loopy.diagnostic import LoopyError
from loopy.tools import is_integer
from loopy.types import LoopyType
from loopy.target.c import CExpression
from loopy.typing import ExpressionT


__doc__ = """
.. currentmodule:: loopy.target.c.codegen.expression

.. autoclass:: ExpressionToCExpressionMapper
"""


# {{{ Loopy expression to C expression mapper

class ExpressionToCExpressionMapper(IdentityMapper):
    """
    Mapper that converts a loopy-semantic expression to a C-semantic expression
    with typecasts, appropriate arithmetic semantic mapping, etc.


    .. note::

        - All mapper methods take in an extra argument called *type_context*.
          The purpose of *type_context* is to inform the method about the
          expected type for untyped expressions such as python scalars. The
          type of the expressions takes precedence over *type_context*.
    """
    def __init__(self, codegen_state, fortran_abi=False, type_inf_mapper=None):
        self.kernel = codegen_state.kernel
        self.codegen_state = codegen_state

        if type_inf_mapper is None:
            type_inf_mapper = TypeReader(self.kernel,
                    self.codegen_state.callables_table)
        self.type_inf_mapper = type_inf_mapper

        self.allow_complex = codegen_state.allow_complex

        self.fortran_abi = fortran_abi

    # {{{ helpers

    def with_assignments(self, names_to_vars):
        type_inf_mapper = self.type_inf_mapper.with_assignments(names_to_vars)
        return type(self)(self.codegen_state, self.fortran_abi, type_inf_mapper)

    def infer_type(self, expr: ExpressionT) -> LoopyType:
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
        return s

    def wrap_in_typecast_lazy(self, actual_type_func, needed_dtype, s):
        """This is similar to *wrap_in_typecast*, but takes a function for
        the actual type argument instead of a type. This can be helpful
        when actual type argument is expensive to calculate and is not
        needed in some cases.
        """
        return s

    def rec(self, expr, type_context=None, needed_dtype=None):
        if needed_dtype is None:
            return RecursiveMapper.rec(self, expr, type_context)

        return self.wrap_in_typecast_lazy(
                lambda: self.infer_type(expr), needed_dtype,
                RecursiveMapper.rec(self, expr, type_context))

    def __call__(self, expr, prec=None, type_context=None, needed_dtype=None):
        if prec is None:
            prec = PREC_NONE

        assert prec == PREC_NONE
        return CExpression(
                self.codegen_state.ast_builder.get_c_expression_to_code_mapper(),
                self.rec(expr, type_context, needed_dtype))

    # }}}

    def map_variable(self, expr, type_context):
        from loopy.kernel.data import ValueArg, AddressSpace

        def postproc(x):
            return x

        if expr.name in self.codegen_state.var_subst_map:
            if self.kernel.options.annotate_inames:
                return var(
                        "/* {} */ {}".format(
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

                        from loopy.kernel.array import _apply_offset
                        from loopy.symbolic import simplify_using_aff

                        subscript = _apply_offset(0, arg)
                        result = self.make_subscript(
                                arg,
                                var(expr.name),
                                simplify_using_aff(
                                    self.kernel, self.rec(subscript, "i")))
                        return result
                    else:
                        return var(expr.name)[0]
                else:
                    raise RuntimeError("unsubscripted reference to array '%s'"
                            % expr.name)

            if isinstance(arg, ValueArg) and self.fortran_abi:
                postproc = lambda x: x[0]  # noqa
        elif expr.name in self.kernel.temporary_variables:
            temporary = self.kernel.temporary_variables[expr.name]
            if (temporary.base_storage
                    or temporary.address_space == AddressSpace.GLOBAL):
                postproc = lambda x: x[0]  # noqa

        result = self.kernel.mangle_symbol(self.codegen_state.ast_builder, expr.name)
        if result is not None:
            _, c_name = result
            return postproc(var(c_name))

        return postproc(var(expr.name))

    def map_tagged_variable(self, expr, type_context):
        return var(expr.name)

    def map_sub_array_ref(self, expr, type_context):
        from loopy.symbolic import get_start_subscript_from_sar
        return var("&")(self.rec(get_start_subscript_from_sar(expr, self.kernel),
            type_context))

    def map_subscript(self, expr, type_context):
        def base_impl(expr, type_context):
            return self.rec(expr.aggregate, type_context)[self.rec(expr.index, "i")]

        def make_var(name):
            from loopy import TaggedVariable
            if isinstance(expr.aggregate, TaggedVariable):
                return TaggedVariable(name, expr.aggregate.tags)
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

        access_info = get_access_info(self.kernel, ary, index_tuple,
                lambda expr: evaluate(expr, self.codegen_state.var_subst_map),
                self.codegen_state.vectorization_info)

        from loopy.kernel.data import (
                ImageArg, ArrayArg, TemporaryVariable, ConstantArg)

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
                    var("(%s)" % idx_vec_type)(*self.rec(idx_tuple, "i")))

            if ary.dtype.numpy_dtype == np.float32:
                return base_access.attr("x")
            if self.kernel.target.is_vector_dtype(ary.dtype):
                return base_access
            elif ary.dtype.numpy_dtype == np.float64:
                return var("as_double")(base_access.attr("xy"))
            else:
                raise NotImplementedError(
                        "non-floating-point images not supported for now")

        elif isinstance(ary, (ArrayArg, TemporaryVariable, ConstantArg)):
            if len(access_info.subscripts) == 0:
                if (
                        isinstance(ary, (ConstantArg, ArrayArg)) or
                        (isinstance(ary, TemporaryVariable) and ary.base_storage)):
                    # unsubscripted global args are pointers
                    result = self.make_subscript(
                            ary,
                            make_var(access_info.array_name),
                            (0,))

                else:
                    # unsubscripted temp vars are scalars
                    # (unless they use base_storage)
                    result = make_var(access_info.array_name)

            else:
                subscript, = access_info.subscripts
                result = self.make_subscript(
                        ary,
                        make_var(access_info.array_name),
                        simplify_using_aff(
                            self.kernel, self.rec(subscript, "i")))

            if access_info.vector_index is not None:
                return self.codegen_state.ast_builder.add_vector_access(
                    result, access_info.vector_index)
            else:
                return result

        else:
            raise AssertionError()

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

                return self.make_subscript(
                        arg,
                        var(expr.aggregate.name),
                        self.rec(offset + expr.index, "i"))

        elif expr.aggregate.name in self.kernel.temporary_variables:
            raise RuntimeError("linear indexing is not supported on temporaries: %s"
                    % expr)

        else:
            raise RuntimeError(
                    "nothing known about variable '%s'" % expr.aggregate.name)

    def make_subscript(self, array, base_expr, subscript):
        return base_expr[subscript]

    def _map_integer_div_operator(self, base_func_name, op_func, expr, type_context):
        from loopy.symbolic import get_dependencies
        iname_deps = get_dependencies(expr) & self.kernel.all_inames()
        domain = self.kernel.get_inames_domain(iname_deps)

        assumption_non_param = isl.BasicSet.from_params(self.kernel.assumptions)
        assumptions, domain = isl.align_two(assumption_non_param, domain)
        domain = domain & assumptions

        num_type = self.infer_type(expr.numerator)
        den_type = self.infer_type(expr.denominator)

        if not num_type.is_integral() or not den_type.is_integral():
            raise NotImplementedError("remainder and floordiv "
                                      "for floating-point types")

        from loopy.isl_helpers import is_nonnegative
        num_nonneg = is_nonnegative(expr.numerator, domain) \
            or num_type.numpy_dtype.kind == "u"
        den_nonneg = is_nonnegative(expr.denominator, domain) \
            or den_type.numpy_dtype.kind == "u"

        result_dtype = self.infer_type(expr)
        suffix = result_dtype.numpy_dtype.type.__name__

        def seen_func(name):
            from loopy.codegen import SeenFunction
            self.codegen_state.seen_functions.add(
                    SeenFunction(
                        name, f"{name}_{suffix}",
                        (result_dtype, result_dtype),
                        (result_dtype,)))

        if den_nonneg:
            if num_nonneg:
                return op_func(
                        self.rec(expr.numerator, "i"),
                        self.rec(expr.denominator, "i"))
            else:
                seen_func(f"{base_func_name}_pos_b")
                return var(f"{base_func_name}_pos_b_{suffix}")(
                        self.rec(expr.numerator, "i"),
                        self.rec(expr.denominator, "i"))
        else:
            seen_func(base_func_name)
            return var(f"{base_func_name}_{suffix}")(
                    self.rec(expr.numerator, "i"),
                    self.rec(expr.denominator, "i"))

    def map_floor_div(self, expr, type_context):
        import operator
        return self._map_integer_div_operator(
                "loopy_floor_div", operator.floordiv, expr, type_context)

    def map_remainder(self, expr, type_context):
        tgt_dtype = self.infer_type(expr)
        if tgt_dtype.is_complex():
            raise RuntimeError("complex remainder not defined")

        import operator
        return self._map_integer_div_operator(
                "loopy_mod", operator.mod, expr, type_context)

    def map_if(self, expr, type_context):
        from loopy.types import to_loopy_type
        result_type = self.infer_type(expr)
        return type(expr)(
                self.rec(expr.condition, type_context,
                         to_loopy_type(np.bool_)),
                self.rec(expr.then, type_context, result_type),
                self.rec(expr.else_, type_context, result_type),
                )

    def map_comparison(self, expr, type_context):
        inner_type_context = dtype_to_type_context(
                self.kernel.target,
                self.infer_type(expr.left - expr.right))

        return type(expr)(
                    self.rec(expr.left, inner_type_context),
                    expr.operator,
                    self.rec(expr.right, inner_type_context))

    def map_type_cast(self, expr, type_context):
        registry = self.codegen_state.ast_builder.target.get_dtype_registry()
        cast = var("(%s)" % registry.dtype_to_ctype(expr.type))
        return cast(self.rec(expr.child, type_context))

    def map_constant(self, expr, type_context):
        from loopy.symbolic import Literal

        if isinstance(expr, (complex, np.complexfloating)):
            real = self.rec(expr.real, type_context)
            imag = self.rec(expr.imag, type_context)
            iota = p.Variable("I" if "I" not in self.kernel.all_variable_names()
                    else "_Complex_I")
            return real + imag*iota
        elif np.isnan(expr):
            from warnings import warn
            warn("Encountered 'bare' floating point NaN value. Since NaN != NaN,"
                 " this leads to problems with cache retrieval."
                 " Consider using `pymbolic.primitives.NaN` instead of `math.nan`."
                 " The generated code will be equivalent with the added benefit"
                 " of sound pickling/unpickling of kernel objects.")
            from pymbolic.primitives import NaN
            data_type = expr.dtype.type if isinstance(expr, np.generic) else None
            return self.map_nan(NaN(data_type), type_context)
        elif np.isneginf(expr):
            return -p.Variable("INFINITY")
        elif np.isinf(expr):
            return p.Variable("INFINITY")
        elif isinstance(expr, np.generic):
            # Explicitly typed: Generated code must reflect type exactly.

            # FIXME: This assumes a 32-bit architecture.
            if isinstance(expr, np.float32):
                return Literal(repr(expr)+"f")

            elif isinstance(expr, np.float64):
                return Literal(repr(expr))

            # Disabled for now, possibly should be a subtarget.
            # elif isinstance(expr, np.float128):
            #     return Literal(repr(expr)+"l")

            elif isinstance(expr, np.integer):
                suffix = ""
                iinfo = np.iinfo(expr)
                if iinfo.min == 0:
                    suffix += "u"
                if iinfo.max > (2**31-1):
                    suffix += "l"
                return Literal(repr(expr)+suffix)
            elif isinstance(expr, np.bool_):
                return Literal("true") if expr else Literal("false")
            else:
                raise LoopyError("do not know how to generate code for "
                        "constant of numpy type '%s'" % type(expr).__name__)

        elif np.isfinite(expr):
            if type_context == "f":
                return Literal(repr(np.float32(expr))+"f")
            elif type_context == "d":
                return Literal(repr(float(expr)))
            elif type_context in ["i", "b"]:
                return int(expr)
            else:
                if is_integer(expr):
                    return int(expr)
                raise RuntimeError("don't know how to generate code "
                        "for constant '%s'" % expr)
        else:
            raise LoopyError("don't know how to generate code "
                             "for constant '%s'" % expr)

    def map_call(self, expr, type_context):
        return (
                self.codegen_state.callables_table[
                    expr.function.name].emit_call(
                        expression_to_code_mapper=self,
                    expression=expr,
                    target=self.kernel.target))

    # {{{ deal with complex-valued variables

    def map_quotient(self, expr, type_context):
        n_dtype = self.infer_type(expr.numerator).numpy_dtype
        d_dtype = self.infer_type(expr.denominator).numpy_dtype

        num = self.rec(expr.numerator, type_context)

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

    def map_power(self, expr, type_context):
        tgt_dtype = self.infer_type(expr)
        base_dtype = self.infer_type(expr.base)
        exponent_dtype = self.infer_type(expr.exponent)

        from pymbolic.primitives import is_constant, is_zero
        if is_constant(expr.exponent):
            if is_zero(expr.exponent):
                return 1
            elif is_zero(expr.exponent - 1):
                return self.rec(expr.base, type_context)
            elif is_zero(expr.exponent - 2):
                return self.rec(expr.base*expr.base, type_context)

        if exponent_dtype.is_integral():
            from loopy.codegen import SeenFunction
            func_name = ("loopy_pow_"
                    f"{tgt_dtype.numpy_dtype}_{exponent_dtype.numpy_dtype}")

            self.codegen_state.seen_functions.add(
                    SeenFunction(
                        "int_pow", func_name,
                        (tgt_dtype, exponent_dtype),
                        (tgt_dtype, )))
            # FIXME: This need some more callables to be registered.
            return var(func_name)(self.rec(expr.base, type_context),
                                  self.rec(expr.exponent, type_context))
        else:
            from loopy.codegen import SeenFunction
            clbl = self.codegen_state.ast_builder.known_callables["pow"]
            clbl = clbl.with_types({0: tgt_dtype, 1: exponent_dtype},
                    self.codegen_state.callables_table)[0]
            self.codegen_state.seen_functions.add(
                    SeenFunction(
                        clbl.name, clbl.name_in_target,
                        (base_dtype, exponent_dtype),
                        (tgt_dtype,)))
            return var(clbl.name_in_target)(self.rec(expr.base, type_context),
                    self.rec(expr.exponent, type_context))

    # }}}

    def map_group_hw_index(self, expr, type_context):
        raise LoopyError("plain C does not have group hw axes")

    def map_local_hw_index(self, expr, type_context):
        raise LoopyError("plain C does not have local hw axes")

    def map_nan(self, expr, type_context):
        from loopy.types import NumpyType
        if expr.data_type is None:
            if type_context == "f":
                return p.Variable("NAN")
            elif type_context == "d":
                registry = self.codegen_state.ast_builder.target.get_dtype_registry()
                lpy_type = NumpyType(np.dtype(np.float32))
                cast = var("(%s)" % registry.dtype_to_ctype(lpy_type))
                return cast(p.Variable("NAN"))
            else:
                raise NotImplementedError("lowering NaN with type context"
                                          f" '{type_context}'.")
        else:
            if isinstance(expr.data_type(float("nan")), np.float32):
                return p.Variable("NAN")
            elif isinstance(expr.data_type(float("nan")), np.floating):
                registry = self.codegen_state.ast_builder.target.get_dtype_registry()
                lpy_type = NumpyType(np.dtype(expr.data_type))
                cast = var("(%s)" % registry.dtype_to_ctype(lpy_type))
                return cast(p.Variable("NAN"))
            elif isinstance(expr.data_type(float("nan")), np.complexfloating):
                real_dtype = np.empty(0, dtype=expr.data_type).real.dtype.type
                return self.map_constant(real_dtype("nan") + expr.data_type(1j),
                                         type_context)
            else:
                raise NotImplementedError(f"{type(self.kernel.target)} does not"
                                          f" support NaNs of type {expr.data_type}.")

# }}}


# {{{ C expression to code mapper

class CExpressionToCodeMapper(RecursiveMapper):

    # {{{ helpers

    def parenthesize_if_needed(self, s, enclosing_prec, my_prec):
        if enclosing_prec > my_prec:
            return "(%s)" % s
        else:
            return s

    def join_rec(self, joiner, iterable, prec, force_parens_around=()):
        f = joiner.join("%s" for i in iterable)
        return f % tuple(
                self.rec_with_force_parens_around(
                    i, prec, force_parens_around=force_parens_around)
                for i in iterable)

    def rec_with_force_parens_around(
            self, expr, enclosing_prec, force_parens_around=()):
        result = self.rec(expr, enclosing_prec)

        if isinstance(expr, force_parens_around):
            result = "(%s)" % result

        return result

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
            func = self.rec(expr.function, PREC_CALL+1)

        return self.parenthesize_if_needed(
                "{}({})".format(
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
                "{}.{}".format(
                    self.rec(expr.aggregate, PREC_CALL), expr.name),
                enclosing_prec, PREC_CALL)

    def map_subscript(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
                "{}[{}]".format(
                    self.rec(expr.aggregate, PREC_CALL+1),
                    self.rec(expr.index, PREC_NONE)),
                enclosing_prec, PREC_CALL)

    def map_min(self, expr, enclosing_prec):
        what = type(expr).__name__.lower()

        children = list(expr.children)

        result = self.rec(children.pop(), PREC_NONE)
        while children:
            result = "{}({}, {})".format(what,
                        self.rec(children.pop(), PREC_NONE),
                        result)

        return result

    map_max = map_min

    def map_if(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE, PREC_CALL
        return "({} ? {} : {})".format(
                # Force parentheses around the condition to prevent compiler
                # warnings regarding precedence (e.g. with POCL 1.8/LLVM 12):
                # "warning: pocl-cache/tempfile_BYDWne.cl:96:2241: operator '?:'
                # has lower precedence than '*'; '*' will be evaluated first"
                self.rec(expr.condition, PREC_CALL),
                self.rec(expr.then, PREC_NONE),
                self.rec(expr.else_, PREC_NONE),
                )

    def map_comparison(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_COMPARISON

        return self.parenthesize_if_needed(
                "{} {} {}".format(
                    self.rec(expr.left, PREC_COMPARISON),
                    expr.operator,
                    self.rec(expr.right, PREC_COMPARISON)),
                enclosing_prec, PREC_COMPARISON)

    def map_literal(self, expr, enclosing_prec):
        return expr.s

    def map_left_shift(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
            self.join_rec(" << ", (expr.shiftee, expr.shift), PREC_SHIFT),
            enclosing_prec, PREC_SHIFT)

    def map_right_shift(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
            self.join_rec(" >> ", (expr.shiftee, expr.shift), PREC_SHIFT),
            enclosing_prec, PREC_SHIFT)

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

    def map_bitwise_not(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
                "~" + self.rec(expr.child, PREC_UNARY),
                enclosing_prec, PREC_UNARY)

    def map_bitwise_and(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
                self.join_rec(" & ", expr.children, PREC_BITWISE_AND),
                enclosing_prec, PREC_BITWISE_AND)

    def map_bitwise_or(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
                self.join_rec(" | ", expr.children, PREC_BITWISE_OR),
                enclosing_prec, PREC_BITWISE_OR)

    def map_bitwise_xor(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
                self.join_rec(" ^ ", expr.children, PREC_BITWISE_XOR),
                enclosing_prec, PREC_BITWISE_XOR)

    def map_sum(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_SUM

        return self.parenthesize_if_needed(
                self.join_rec(" + ", expr.children, PREC_SUM),
                enclosing_prec, PREC_SUM)

    multiplicative_primitives = (p.Product, p.Quotient, p.FloorDiv, p.Remainder)

    def map_product(self, expr, enclosing_prec):
        force_parens_around = (p.Quotient, p.FloorDiv, p.Remainder)

        # Spaces prevent '**z' (times dereference z), which is hard to read.
        return self.parenthesize_if_needed(
                self.join_rec(" * ", expr.children, PREC_PRODUCT,
                    force_parens_around=force_parens_around),
                enclosing_prec, PREC_PRODUCT)

    def _map_division_operator(self, operator, expr, enclosing_prec):
        num_s = self.rec_with_force_parens_around(expr.numerator, PREC_PRODUCT,
                force_parens_around=self.multiplicative_primitives)

        denom_s = self.rec_with_force_parens_around(expr.denominator, PREC_PRODUCT,
                force_parens_around=self.multiplicative_primitives)

        return self.parenthesize_if_needed(
                f"{num_s} {operator} {denom_s}",
                # Space is necessary--otherwise '/*'
                # (i.e. divide-dererference) becomes
                # start-of-comment in C.
                enclosing_prec, PREC_PRODUCT)

    def map_quotient(self, expr, enclosing_prec):
        return self._map_division_operator("/", expr, enclosing_prec)

    def map_floor_div(self, expr, enclosing_prec):
        return self._map_division_operator("/", expr, enclosing_prec)

    def map_remainder(self, expr, enclosing_prec):
        return self._map_division_operator("%", expr, enclosing_prec)

    def map_power(self, expr, enclosing_prec):
        raise RuntimeError(f"'{expr}' should have been transformed to 'Call'"
                           " expression node.")

    def map_array_literal(self, expr, enclosing_prec):
        return "{ %s }" % self.join_rec(", ", expr.children, PREC_NONE)

# }}}

# vim: fdm=marker

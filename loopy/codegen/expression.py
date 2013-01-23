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




import numpy as np

from pymbolic.mapper import RecursiveMapper
from pymbolic.mapper.stringifier import (PREC_NONE, PREC_CALL, PREC_PRODUCT,
        PREC_POWER)
from pymbolic.mapper import CombineMapper
import islpy as isl
import pyopencl as cl
import pyopencl.array
from pytools import memoize_method

# {{{ type inference

class TypeInferenceFailure(RuntimeError):
    pass

class DependencyTypeInferenceFailure(TypeInferenceFailure):
    pass

class TypeInferenceMapper(CombineMapper):
    def __init__(self, kernel, temporary_variables=None):
        self.kernel = kernel
        if temporary_variables is None:
            temporary_variables = kernel.temporary_variables

        self.temporary_variables = temporary_variables

    # /!\ Introduce caches with care--numpy.float32(x) and numpy.float64(x)
    # are Python-equal.

    def combine(self, dtypes):
        dtypes = list(dtypes)

        result = dtypes.pop()
        while dtypes:
            other = dtypes.pop()

            if result.isbuiltin and other.isbuiltin:
                result = (np.empty(0, dtype=result) + np.empty(0, dtype=other)).dtype
            elif result.isbuiltin and not other.isbuiltin:
                # assume the non-native type takes over
                result = other
            elif not result.isbuiltin and other.isbuiltin:
                # assume the non-native type takes over
                pass
            else:
                if not result is other:
                    raise TypeInferenceFailure("nothing known about result of operation on "
                            "'%s' and '%s'" % (result, other))

        return result

    def map_constant(self, expr):
        if isinstance(expr, int):
            for tp in [np.int8, np.int16, np.int32, np.int64]:
                iinfo = np.iinfo(tp)
                if iinfo.min <= expr <= iinfo.max:
                    return np.dtype(tp)

            else:
                raise TypeInferenceFailure("integer constant '%s' too large" % expr)

        dt = np.asarray(expr).dtype
        if hasattr(expr, "dtype"):
            return expr.dtype
        elif isinstance(expr, np.number):
            # Numpy types are sized
            return np.dtype(type(expr))
        elif dt.kind == "f":
            # deduce the smaller type by default
            return np.dtype(np.float32)
        elif dt.kind == "c":
            # Codegen for complex types depends on exactly correct types.
            # Refuse temptation to guess.
            raise TypeInferenceFailure("Complex constant '%s' needs to "
                    "be sized for type inference " % expr)
        else:
            raise TypeInferenceFailure("Cannot deduce type of constant '%s'" % expr)

    def map_subscript(self, expr):
        return self.rec(expr.aggregate)

    def map_linear_subscript(self, expr):
        return self.rec(expr.aggregate)

    def map_call(self, expr):
        from pymbolic.primitives import Variable

        identifier = expr.function
        if isinstance(identifier, Variable):
            identifier = identifier.name

        arg_dtypes = tuple(self.rec(par) for par in expr.parameters)

        mangle_result = self.kernel.mangle_function(identifier, arg_dtypes)
        if mangle_result is not None:
            return mangle_result[0]

        raise RuntimeError("no type inference information on "
                "function '%s'" % identifier)

    def map_variable(self, expr):
        try:
            return self.kernel.arg_dict[expr.name].dtype
        except KeyError:
            pass

        try:
            tv = self.temporary_variables[expr.name]
        except KeyError:
            # name is not a temporary variable, ok
            pass
        else:
            from loopy import infer_type
            if tv.dtype is infer_type:
                raise DependencyTypeInferenceFailure("attempted type inference on "
                        "variable requiring type inference")

            return tv.dtype

        if expr.name in self.kernel.all_inames():
            return self.kernel.index_dtype

        for mangler in self.kernel.symbol_manglers:
            result = mangler(expr.name)
            if result is not None:
                result_dtype, _ = result
                return result_dtype

        raise TypeInferenceFailure("nothing known about '%s'" % expr.name)

    map_tagged_variable = map_variable

    def map_lookup(self, expr):
        agg_result = self.rec(expr.aggregate)
        dtype, offset = agg_result.fields[expr.name]
        return dtype

    def map_reduction(self, expr):
        return expr.operation.result_dtype(self.rec(expr.expr), expr.inames)

# }}}

# {{{ C code mapper

# type_context may be:
# - 'i' for integer -
# - 'f' for single-precision floating point
# - 'd' for double-precision floating point
# or None for 'no known context'.

def dtype_to_type_context(dtype):
    dtype = np.dtype(dtype)

    if dtype.kind == 'i':
        return 'i'
    if dtype in [np.float64, np.complex128]:
        return 'd'
    if dtype in [np.float32, np.complex64]:
        return 'f'
    from pyopencl.array import vec
    if dtype in vec.types.values():
        return dtype_to_type_context(dtype.fields["x"][0])

    return None


class LoopyCCodeMapper(RecursiveMapper):
    def __init__(self, kernel, seen_dtypes, seen_functions, var_subst_map={},
            with_annotation=False, allow_complex=False):
        """
        :arg seen_dtypes: set of dtypes that were encountered
        :arg seen_functions: set of tuples (name, c_name, arg_types) indicating
            functions that were encountered.
        """

        self.kernel = kernel
        self.seen_dtypes = seen_dtypes
        self.seen_functions = seen_functions

        self.type_inf_mapper = TypeInferenceMapper(kernel)
        self.allow_complex = allow_complex

        self.with_annotation = with_annotation
        self.var_subst_map = var_subst_map.copy()

    # {{{ copy helpers

    def copy(self, var_subst_map=None):
        if var_subst_map is None:
            var_subst_map = self.var_subst_map
        return LoopyCCodeMapper(self.kernel, self.seen_dtypes, self.seen_functions,
                var_subst_map=var_subst_map,
                with_annotation=self.with_annotation,
                allow_complex=self.allow_complex)

    def copy_and_assign(self, name, value):
        """Make a copy of self with variable *name* fixed to *value*."""
        var_subst_map = self.var_subst_map.copy()
        var_subst_map[name] = value
        return self.copy(var_subst_map=var_subst_map)

    def copy_and_assign_many(self, assignments):
        """Make a copy of self with *assignments* included."""

        var_subst_map = self.var_subst_map.copy()
        var_subst_map.update(assignments)
        return self.copy(var_subst_map=var_subst_map)

    # }}}

    # {{{ helpers

    def infer_type(self, expr):
        result = self.type_inf_mapper(expr)
        self.seen_dtypes.add(result)
        return result

    def join_rec(self, joiner, iterable, prec, type_context, needed_dtype=None):
        f = joiner.join("%s" for i in iterable)
        return f % tuple(
                self.rec(i, prec, type_context, needed_dtype) for i in iterable)

    def parenthesize_if_needed(self, s, enclosing_prec, my_prec):
        if enclosing_prec > my_prec:
            return "(%s)" % s
        else:
            return s

    def rec(self, expr, prec, type_context=None, needed_dtype=None):
        if needed_dtype is None:
            return RecursiveMapper.rec(self, expr, prec, type_context)

        actual_type = self.infer_type(expr)

        if (actual_type.kind == "c" and needed_dtype.kind == "c"
                and actual_type != needed_dtype):
            result = RecursiveMapper.rec(self, expr, PREC_NONE, type_context)
            return "%s_cast(%s)" % (self.complex_type_name(needed_dtype), result)
        elif actual_type.kind != "c" and needed_dtype.kind == "c":
            result = RecursiveMapper.rec(self, expr, PREC_NONE, type_context)
            return "%s_fromreal(%s)" % (self.complex_type_name(needed_dtype), result)
        else:
            return RecursiveMapper.rec(self, expr, prec, type_context)

    __call__ = rec

    # }}}

    def map_common_subexpression(self, expr, prec, type_context):
        raise RuntimeError("common subexpression should have been eliminated upon "
                "entry to loopy")

    def map_variable(self, expr, enclosing_prec, type_context):
        if expr.name in self.var_subst_map:
            if self.with_annotation:
                return " /* %s */ %s" % (
                        expr.name,
                        self.rec(self.var_subst_map[expr.name],
                            enclosing_prec, type_context))
            else:
                return str(self.rec(self.var_subst_map[expr.name],
                    enclosing_prec, type_context))
        elif expr.name in self.kernel.arg_dict:
            arg = self.kernel.arg_dict[expr.name]
            from loopy.kernel import _ShapedArg
            if isinstance(arg, _ShapedArg) and arg.shape == ():
                return "*"+expr.name

        for mangler in self.kernel.symbol_manglers:
            result = mangler(expr.name)
            if result is not None:
                _, c_name = result
                return c_name

        return expr.name

    def map_tagged_variable(self, expr, enclosing_prec, type_context):
        return expr.name

    def map_lookup(self, expr, enclosing_prec, type_context):
        return self.parenthesize_if_needed(
                "%s.%s" %(self.rec(expr.aggregate, PREC_CALL, type_context), expr.name),
                enclosing_prec, PREC_CALL)

    def map_subscript(self, expr, enclosing_prec, type_context):
        def base_impl(expr, enclosing_prec, type_context):
            return self.parenthesize_if_needed(
                    "%s[%s]" % (
                        self.rec(expr.aggregate, PREC_CALL, type_context),
                        self.rec(expr.index, PREC_NONE, 'i')),
                    enclosing_prec, PREC_CALL)

        from pymbolic.primitives import Variable
        if not isinstance(expr.aggregate, Variable):
            return base_impl(expr, enclosing_prec, type_context)

        if expr.aggregate.name in self.kernel.arg_dict:
            arg = self.kernel.arg_dict[expr.aggregate.name]

            from loopy.kernel import ImageArg
            if isinstance(arg, ImageArg):
                assert isinstance(expr.index, tuple)

                base_access = ("read_imagef(%s, loopy_sampler, (float%d)(%s))"
                        % (arg.name, arg.dimensions,
                            ", ".join(self.rec(idx, PREC_NONE, 'i')
                                for idx in expr.index[::-1])))

                if arg.dtype == np.float32:
                    return base_access+".x"
                if arg.dtype in cl.array.vec.type_to_scalar_and_count:
                    return base_access
                elif arg.dtype == np.float64:
                    return "as_double(%s.xy)" % base_access
                else:
                    raise NotImplementedError(
                            "non-floating-point images not supported for now")

            else:
                # GlobalArg
                index_expr = expr.index
                if not isinstance(expr.index, tuple):
                    index_expr = (index_expr,)

                if arg.strides is not None:
                    ary_strides = arg.strides
                else:
                    ary_strides = (1,)

                if len(ary_strides) != len(index_expr):
                    raise RuntimeError("subscript to '%s' in '%s' has the wrong "
                            "number of indices (got: %d, expected: %d)" % (
                                expr.aggregate.name, expr,
                                len(index_expr), len(ary_strides)))

                if len(index_expr) == 0:
                    return "*" + expr.aggregate.name

                from pymbolic.primitives import Subscript
                return base_impl(
                        Subscript(expr.aggregate, arg.offset+sum(
                            stride*expr_i for stride, expr_i in zip(
                                ary_strides, index_expr))),
                        enclosing_prec, type_context)


        elif expr.aggregate.name in self.kernel.temporary_variables:
            temp_var = self.kernel.temporary_variables[expr.aggregate.name]
            if isinstance(expr.index, tuple):
                index = expr.index
            else:
                index = (expr.index,)

            return (temp_var.name + "".join("[%s]" % self.rec(idx, PREC_NONE, 'i')
                for idx in index))

        else:
            raise RuntimeError("nothing known about variable '%s'" % expr.aggregate.name)

    def map_linear_subscript(self, expr, enclosing_prec, type_context):
        def base_impl(expr, enclosing_prec, type_context):
            return self.parenthesize_if_needed(
                    "%s[%s]" % (
                        self.rec(expr.aggregate, PREC_CALL, type_context),
                        self.rec(expr.index, PREC_NONE, 'i')),
                    enclosing_prec, PREC_CALL)

        from pymbolic.primitives import Variable
        if not isinstance(expr.aggregate, Variable):
            return base_impl(expr, enclosing_prec, type_context)

        if expr.aggregate.name in self.kernel.arg_dict:
            arg = self.kernel.arg_dict[expr.aggregate.name]

            from loopy.kernel import ImageArg
            if isinstance(arg, ImageArg):
                raise RuntimeError("linear indexing doesn't work on images: %s"
                        % expr)

            else:
                # GlobalArg
                from pymbolic.primitives import Subscript
                return base_impl(
                        Subscript(expr.aggregate, arg.offset+expr.index),
                        enclosing_prec, type_context)

        elif expr.aggregate.name in self.kernel.temporary_variables:
            raise RuntimeError("linear indexing doesn't work on temporaries: %s"
                    % expr)

        else:
            raise RuntimeError("nothing known about variable '%s'" % expr.aggregate.name)

    def map_floor_div(self, expr, enclosing_prec, type_context):
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
            self.seen_functions.add((name, name, (idt, idt)))

        if den_nonneg:
            if num_nonneg:
                # parenthesize to avoid negative signs being dragged in from the
                # outside by associativity
                return "(%s / %s)" % (
                            self.rec(expr.numerator, PREC_PRODUCT, type_context),
                            # analogous to ^{-1}
                            self.rec(expr.denominator, PREC_POWER, type_context))
            else:
                seen_func("int_floor_div_pos_b")
                return ("int_floor_div_pos_b(%s, %s)"
                        % (self.rec(expr.numerator, PREC_NONE, 'i'),
                            self.rec(expr.denominator, PREC_NONE, 'i')))
        else:
            seen_func("int_floor_div")
            return ("int_floor_div(%s, %s)"
                    % (self.rec(expr.numerator, PREC_NONE, 'i'),
                        self.rec(expr.denominator, PREC_NONE, 'i')))

    def map_min(self, expr, prec, type_context):
        what = type(expr).__name__.lower()

        children = expr.children[:]

        result = self.rec(children.pop(), PREC_NONE, type_context)
        while children:
            result = "%s(%s, %s)" % (what,
                        self.rec(children.pop(), PREC_NONE, type_context),
                        result)

        return result

    map_max = map_min

    def map_if(self, expr, enclosing_prec, type_context):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "(%s ? %s : %s)" % (
                self.rec(expr.condition, PREC_NONE, "i"),
                self.rec(expr.then, PREC_NONE, type_context),
                self.rec(expr.else_, PREC_NONE, type_context),
                )

    def map_comparison(self, expr, enclosing_prec, type_context):
        from pymbolic.mapper.stringifier import PREC_COMPARISON

        return self.parenthesize_if_needed(
                "%s %s %s" % (
                    self.rec(expr.left, PREC_COMPARISON, None),
                    expr.operator,
                    self.rec(expr.right, PREC_COMPARISON, None)),
                enclosing_prec, PREC_COMPARISON)

    def map_constant(self, expr, enclosing_prec, type_context):
        if isinstance(expr, (complex, np.complexfloating)):
            if expr.dtype == np.complex128:
                cast_type = "cdouble_t"
            elif expr.dtype == np.complex64:
                cast_type = "cfloat_t"
            else:
                raise RuntimeError("unsupported complex type in expression "
                        "generation: %s" % type(expr))

            return "(%s) (%s, %s)" % (cast_type, repr(expr.real), repr(expr.imag))
        else:
            if type_context == "f":
                return repr(float(expr))+"f"
            elif type_context == "d":
                return repr(float(expr))
            elif type_context == "i":
                return str(int(expr))
            else:
                if isinstance(expr, int):
                    return str(int(expr))

                raise RuntimeError("don't know how to generated code "
                        "for constant '%s'" % expr)

    def map_call(self, expr, enclosing_prec, type_context):
        from pymbolic.primitives import Variable
        from pymbolic.mapper.stringifier import PREC_NONE

        identifier = expr.function

        c_name = None
        if isinstance(identifier, Variable):
            identifier = identifier.name
            c_name = identifier

        par_dtypes = tuple(self.infer_type(par) for par in expr.parameters)

        str_parameters = None

        mangle_result = self.kernel.mangle_function(identifier, par_dtypes)
        if mangle_result is not None:
            if len(mangle_result) == 2:
                result_dtype, c_name = mangle_result
            elif len(mangle_result) == 3:
                result_dtype, c_name, arg_tgt_dtypes = mangle_result

                str_parameters = [
                        self.rec(par, PREC_NONE, dtype_to_type_context(tgt_dtype),
                            tgt_dtype)
                        for par, par_dtype, tgt_dtype in zip(
                            expr.parameters, par_dtypes, arg_tgt_dtypes)]
            else:
                raise RuntimeError("result of function mangler "
                        "for function '%s' not understood"
                        % identifier)

        self.seen_functions.add((identifier, c_name, par_dtypes))
        if str_parameters is None:
            str_parameters = [
                    self.rec(par, PREC_NONE, type_context)
                    for par in expr.parameters]

        if c_name is None:
            raise RuntimeError("unable to find C name for function identifier '%s'"
                    % identifier)

        return "%s(%s)" % (c_name, ", ".join(str_parameters))

    # {{{ deal with complex-valued variables

    def complex_type_name(self, dtype):
        if dtype == np.complex64:
            return "cfloat"
        if dtype == np.complex128:
            return "cdouble"
        else:
            raise RuntimeError

    def map_sum(self, expr, enclosing_prec, type_context):
        from pymbolic.mapper.stringifier import PREC_SUM

        def base_impl(expr, enclosing_prec, type_context):
            return self.parenthesize_if_needed(
                    self.join_rec(" + ", expr.children, PREC_SUM, type_context),
                    enclosing_prec, PREC_SUM)

        if not self.allow_complex:
            return base_impl(expr, enclosing_prec, type_context)

        tgt_dtype = self.infer_type(expr)
        is_complex = tgt_dtype.kind == 'c'

        if not is_complex:
            return base_impl(expr, enclosing_prec, type_context)
        else:
            tgt_name = self.complex_type_name(tgt_dtype)

            reals = [child for child in expr.children
                    if 'c' != self.infer_type(child).kind]
            complexes = [child for child in expr.children
                    if 'c' == self.infer_type(child).kind]

            real_sum = self.join_rec(" + ", reals, PREC_SUM, type_context)
            complex_sum = self.join_rec(" + ", complexes, PREC_SUM, type_context, tgt_dtype)

            if real_sum:
                result = "%s_fromreal(%s) + %s" % (tgt_name, real_sum, complex_sum)
            else:
                result = complex_sum

            return self.parenthesize_if_needed(result, enclosing_prec, PREC_SUM)

    def map_product(self, expr, enclosing_prec, type_context):
        def base_impl(expr, enclosing_prec, type_context):
            # Spaces prevent '**z' (times dereference z), which
            # is hard to read.
            return self.parenthesize_if_needed(
                    self.join_rec(" * ", expr.children, PREC_PRODUCT, type_context),
                    enclosing_prec, PREC_PRODUCT)

        if not self.allow_complex:
            return base_impl(expr, enclosing_prec, type_context)

        tgt_dtype = self.infer_type(expr)
        is_complex = 'c' == tgt_dtype.kind

        if not is_complex:
            return base_impl(expr, enclosing_prec, type_context)
        else:
            tgt_name = self.complex_type_name(tgt_dtype)

            reals = [child for child in expr.children
                    if 'c' != self.infer_type(child).kind]
            complexes = [child for child in expr.children
                    if 'c' == self.infer_type(child).kind]

            real_prd = self.join_rec(" * ", reals, PREC_PRODUCT,
                    type_context)

            if len(complexes) == 1:
                myprec = PREC_PRODUCT
            else:
                myprec = PREC_NONE

            complex_prd = self.rec(complexes[0], myprec, type_context, tgt_dtype)
            for child in complexes[1:]:
                complex_prd = "%s_mul(%s, %s)" % (
                        tgt_name, complex_prd,
                        self.rec(child, PREC_NONE, type_context, tgt_dtype))

            if real_prd:
                # elementwise semantics are correct
                result = "%s*%s" % (real_prd, complex_prd)
            else:
                result = complex_prd

            return self.parenthesize_if_needed(result, enclosing_prec, PREC_PRODUCT)

    def map_quotient(self, expr, enclosing_prec, type_context):
        def base_impl(expr, enclosing_prec, type_context, num_tgt_dtype=None):
            return self.parenthesize_if_needed(
                    "%s / %s" % (
                        # Space is necessary--otherwise '/*'
                        # (i.e. divide-dererference) becomes
                        # start-of-comment in C.
                        self.rec(expr.numerator, PREC_PRODUCT, type_context, num_tgt_dtype),
                        # analogous to ^{-1}
                        self.rec(expr.denominator, PREC_POWER, type_context)),
                    enclosing_prec, PREC_PRODUCT)

        if not self.allow_complex:
            return base_impl(expr, enclosing_prec, type_context)

        n_complex = 'c' == self.infer_type(expr.numerator).kind
        d_complex = 'c' == self.infer_type(expr.denominator).kind

        tgt_dtype = self.infer_type(expr)

        if not (n_complex or d_complex):
            return base_impl(expr, enclosing_prec, type_context)
        elif n_complex and not d_complex:
            # elementwise semantics are correct
            return base_impl(expr, enclosing_prec, type_context,
                    num_tgt_dtype=tgt_dtype)
        elif not n_complex and d_complex:
            return "%s_rdivide(%s, %s)" % (
                    self.complex_type_name(tgt_dtype),
                    self.rec(expr.numerator, PREC_NONE, type_context),
                    self.rec(expr.denominator, PREC_NONE, type_context, tgt_dtype))
        else:
            return "%s_divide(%s, %s)" % (
                    self.complex_type_name(tgt_dtype),
                    self.rec(expr.numerator, PREC_NONE, type_context, tgt_dtype),
                    self.rec(expr.denominator, PREC_NONE, type_context, tgt_dtype))

    def map_remainder(self, expr, enclosing_prec, type_context):
        tgt_dtype = self.infer_type(expr)
        if 'c' == tgt_dtype.kind:
            raise RuntimeError("complex remainder not defined")

        return "(%s %% %s)" % (
                    self.rec(expr.numerator, PREC_PRODUCT, type_context),
                    self.rec(expr.denominator, PREC_POWER, type_context)) # analogous to ^{-1}

    def map_power(self, expr, enclosing_prec, type_context):
        def base_impl(expr, enclosing_prec, type_context):
            from pymbolic.mapper.stringifier import PREC_NONE
            from pymbolic.primitives import is_constant, is_zero
            if is_constant(expr.exponent):
                if is_zero(expr.exponent):
                    return "1"
                elif is_zero(expr.exponent - 1):
                    return self.rec(expr.base, enclosing_prec, type_context)
                elif is_zero(expr.exponent - 2):
                    return self.rec(expr.base*expr.base, enclosing_prec, type_context)

            return "pow(%s, %s)" % (
                    self.rec(expr.base, PREC_NONE, type_context),
                    self.rec(expr.exponent, PREC_NONE, type_context))

        if not self.allow_complex:
            return base_impl(expr, enclosing_prec, type_context)

        tgt_dtype = self.infer_type(expr)
        if 'c' == tgt_dtype.kind:
            if expr.exponent in [2, 3, 4]:
                value = expr.base
                for i in range(expr.exponent-1):
                    value = value * expr.base
                return self.rec(value, enclosing_prec, type_context)
            else:
                b_complex = 'c' == self.infer_type(expr.base).kind
                e_complex = 'c' == self.infer_type(expr.exponent).kind

                if b_complex and not e_complex:
                    return "%s_powr(%s, %s)" % (
                            self.complex_type_name(tgt_dtype),
                            self.rec(expr.base, PREC_NONE, type_context, tgt_dtype),
                            self.rec(expr.exponent, PREC_NONE, type_context))
                else:
                    return "%s_pow(%s, %s)" % (
                            self.complex_type_name(tgt_dtype),
                            self.rec(expr.base, PREC_NONE, type_context, tgt_dtype),
                            self.rec(expr.exponent, PREC_NONE, type_context, tgt_dtype))

        return base_impl(expr, enclosing_prec, type_context)

    # }}}

# }}}

# vim: fdm=marker

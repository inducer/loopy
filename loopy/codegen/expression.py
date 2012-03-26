from __future__ import division

import numpy as np

from pymbolic.mapper.c_code import CCodeMapper as CCodeMapper
from pymbolic.mapper.stringifier import PREC_NONE
from pymbolic.mapper import CombineMapper

# {{{ type inference

class TypeInferenceMapper(CombineMapper):
    def __init__(self, kernel):
        self.kernel = kernel

    def combine(self, dtypes):
        dtypes = list(dtypes)

        result = dtypes.pop()
        while dtypes:
            other = dtypes.pop()

            if result.isbuiltin and other.isbuiltin:
                result = (np.empty(0, dtype=result) + np.empty(0, dtype=other)).dtype
            elif result.isbuiltin and not other.isbuiltin:
                # assume the non-natiev type takes over
                result = other
            elif not result.isbuiltin and other.isbuiltin:
                # assume the non-natiev type takes over
                pass
            else:
                if not result is other:
                    raise TypeError("nothing known about result of operation on "
                            "'%s' and '%s'" % (result, other))

        return result

    def map_constant(self, expr):
        return np.asarray(expr).dtype

    def map_subscript(self, expr):
        return self.rec(expr.aggregate)

    def map_variable(self, expr):
        try:
            return self.kernel.arg_dict[expr.name].dtype
        except KeyError:
            pass

        try:
            return self.kernel.temporary_variables[expr.name].dtype
        except KeyError:
            pass

        if expr.name in self.kernel.all_inames():
            return np.dtype(np.int16) # don't force single-precision upcast

        raise RuntimeError("type inference: nothing known about '%s'" % expr.name)

# }}}

# {{{ C code mapper

class LoopyCCodeMapper(CCodeMapper):
    def __init__(self, kernel, cse_name_list=[], var_subst_map={},
            with_annotation=False, allow_complex=False):
        def constant_mapper(c):
            if isinstance(c, float):
                # FIXME: type-variable
                return "%sf" % repr(c)
            else:
                return repr(c)

        CCodeMapper.__init__(self, constant_mapper=constant_mapper,
                cse_name_list=cse_name_list)
        self.kernel = kernel
        self.infer_type = TypeInferenceMapper(kernel)
        self.allow_complex = allow_complex

        self.with_annotation = with_annotation
        self.var_subst_map = var_subst_map.copy()

    def copy(self, var_subst_map=None, cse_name_list=None):
        if var_subst_map is None:
            var_subst_map = self.var_subst_map
        if cse_name_list is None:
            cse_name_list = self.cse_name_list
        return LoopyCCodeMapper(self.kernel,
                cse_name_list=cse_name_list, var_subst_map=var_subst_map,
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

    def map_common_subexpression(self, expr, prec):
        raise RuntimeError("common subexpressions are not allowed in loopy")

    def map_variable(self, expr, prec):
        if expr.name in self.var_subst_map:
            if self.with_annotation:
                return " /* %s */ %s" % (
                        expr.name,
                        self.rec(self.var_subst_map[expr.name], prec))
            else:
                return str(self.rec(self.var_subst_map[expr.name], prec))
        else:
            return CCodeMapper.map_variable(self, expr, prec)

    def map_tagged_variable(self, expr, enclosing_prec):
        return expr.name

    def map_subscript(self, expr, enclosing_prec):
        from pymbolic.primitives import Variable
        if not isinstance(expr.aggregate, Variable):
            return CCodeMapper.map_subscript(self, expr, enclosing_prec)

        if expr.aggregate.name in self.kernel.arg_dict:
            arg = self.kernel.arg_dict[expr.aggregate.name]

            from loopy.kernel import ImageArg
            if isinstance(arg, ImageArg):
                assert isinstance(expr.index, tuple)

                base_access = ("read_imagef(%s, loopy_sampler, (float%d)(%s))"
                        % (arg.name, arg.dimensions,
                            ", ".join(self.rec(idx, PREC_NONE)
                                for idx in expr.index[::-1])))

                if arg.dtype == np.float32:
                    return base_access+".x"
                elif arg.dtype == np.float64:
                    return "as_double(%s.xy)" % base_access
                else:
                    raise NotImplementedError(
                            "non-floating-point images not supported for now")

            else:
                # ArrayArg
                index_expr = expr.index
                if isinstance(expr.index, tuple):
                    ary_strides = arg.strides
                    if ary_strides is None:
                        raise RuntimeError("tuple-indexed variable '%s' does not "
                                "have stride information" % expr.aggregate.name)
                else:
                    ary_strides = (1,)
                    index_expr = (index_expr,)

                from pymbolic.primitives import Subscript
                return CCodeMapper.map_subscript(self,
                        Subscript(expr.aggregate, arg.offset+sum(
                            stride*expr_i for stride, expr_i in zip(
                                ary_strides, index_expr))), enclosing_prec)


        elif expr.aggregate.name in self.kernel.temporary_variables:
            temp_var = self.kernel.temporary_variables[expr.aggregate.name]
            if isinstance(expr.index, tuple):
                index = expr.index
            else:
                index = (expr.index,)

            return (temp_var.name + "".join("[%s]" % self.rec(idx, PREC_NONE)
                for idx in index))

        else:
            raise RuntimeError("nothing known about variable '%s'" % expr.aggregate.name)

    def map_floor_div(self, expr, prec):
        from loopy.isl_helpers import is_nonnegative
        num_nonneg = is_nonnegative(expr.numerator, self.kernel.domain)
        den_nonneg = is_nonnegative(expr.denominator, self.kernel.domain)

        if den_nonneg:
            if num_nonneg:
                return CCodeMapper.map_floor_div(self, expr, prec)
            else:
                return ("int_floor_div_pos_b(%s, %s)"
                        % (self.rec(expr.numerator, PREC_NONE),
                            expr.denominator))
        else:
            return ("int_floor_div(%s, %s)"
                    % (self.rec(expr.numerator, PREC_NONE),
                        self.rec(expr.denominator, PREC_NONE)))

    def map_min(self, expr, prec):
        what = type(expr).__name__.lower()

        children = expr.children[:]

        result = self.rec(children.pop(), PREC_NONE)
        while children:
            result = "%s(%s, %s)" % (what,
                        self.rec(children.pop(), PREC_NONE),
                        result)

        return result

    map_max = map_min

    # {{{ deal with complex-valued variables

    def complex_type_name(self, dtype):
        if dtype == np.complex64:
            return "cfloat"
        if dtype == np.complex128:
            return "cdouble"
        else:
            raise RuntimeError

    def map_sum(self, expr, enclosing_prec):
        if not self.allow_complex:
            return CCodeMapper.map_sum(self, expr, enclosing_prec)

        tgt_dtype = self.infer_type(expr)
        is_complex = tgt_dtype.kind == 'c'

        if not is_complex:
            return CCodeMapper.map_sum(self, expr, enclosing_prec)
        else:
            tgt_name = self.complex_type_name(tgt_dtype)

            reals = [child for child in expr.children
                    if 'c' != self.infer_type(child).kind]
            complexes = [child for child in expr.children
                    if 'c' == self.infer_type(child).kind]

            from pymbolic.mapper.stringifier import PREC_SUM
            real_sum = self.join_rec(" + ", reals, PREC_SUM)
            complex_sum = self.join_rec(" + ", complexes, PREC_SUM)

            if real_sum:
                result = "%s_fromreal(%s) + %s" % (tgt_name, real_sum, complex_sum)
            else:
                result = complex_sum

            return self.parenthesize_if_needed(result, enclosing_prec, PREC_SUM)

    def map_product(self, expr, enclosing_prec):
        if not self.allow_complex:
            return CCodeMapper.map_product(self, expr, enclosing_prec)

        tgt_dtype = self.infer_type(expr)
        is_complex = 'c' == tgt_dtype.kind

        if not is_complex:
            return CCodeMapper.map_product(self, expr, enclosing_prec)
        else:
            tgt_name = self.complex_type_name(tgt_dtype)

            reals = [child for child in expr.children
                    if 'c' != self.infer_type(child).kind]
            complexes = [child for child in expr.children
                    if 'c' == self.infer_type(child).kind]

            from pymbolic.mapper.stringifier import PREC_PRODUCT
            real_prd = self.join_rec("*", reals, PREC_PRODUCT)

            complex_prd = self.rec(complexes[0], PREC_NONE)
            for child in complexes[1:]:
                complex_prd = "%s_mul(%s, %s)" % (
                        tgt_name, complex_prd,
                        self.rec(child, PREC_NONE))

            if real_prd:
                # elementwise semantics are correct
                result = "%s * %s" % (real_prd, complex_prd)
            else:
                result = complex_prd

            return self.parenthesize_if_needed(result, enclosing_prec, PREC_PRODUCT)

    def map_quotient(self, expr, enclosing_prec):
        if not self.allow_complex:
            return CCodeMapper.map_quotient(self, expr, enclosing_prec)

        n_complex = 'c' == self.infer_type(expr.numerator).kind
        d_complex = 'c' == self.infer_type(expr.denominator).kind

        tgt_dtype = self.infer_type(expr)

        if not (n_complex or d_complex):
            return CCodeMapper.map_quotient(self, expr, enclosing_prec)
        elif n_complex and not d_complex:
            # elementwise semnatics are correct
            return CCodeMapper.map_quotient(self, expr, enclosing_prec)
        elif not n_complex and d_complex:
            return "%s_rdivide(%s, %s)" % (
                    self.complex_type_name(tgt_dtype),
                    self.rec(expr.numerator, PREC_NONE),
                    self.rec(expr.denominator, PREC_NONE))
        else:
            return "%s_divide(%s, %s)" % (
                    self.complex_type_name(tgt_dtype),
                    self.rec(expr.numerator, PREC_NONE),
                    self.rec(expr.denominator, PREC_NONE))

    def map_remainder(self, expr, enclosing_prec):
        if not self.allow_complex:
            return CCodeMapper.map_remainder(self, expr, enclosing_prec)

        tgt_dtype = self.infer_type(expr)
        if 'c' == tgt_dtype.kind:
            raise RuntimeError("complex remainder not defined")

        return CCodeMapper.map_remainder(self, expr, enclosing_prec)

    def map_power(self, expr, enclosing_prec):
        if not self.allow_complex:
            return CCodeMapper.map_power(self, expr, enclosing_prec)

        from pymbolic.mapper.stringifier import PREC_NONE

        tgt_dtype = self.infer_type(expr)
        if 'c' == tgt_dtype.kind:
            if expr.exponent in [2, 3, 4]:
                value = expr.base
                for i in range(expr.exponent-1):
                    value = value * expr.base
                return self.rec(value, enclosing_prec)
            else:
                b_complex = 'c' == self.infer_type(expr.base).kind
                e_complex = 'c' == self.infer_type(expr.exponent).kind

                if b_complex and not e_complex:
                    return "%s_powr(%s, %s)" % (
                            self.complex_type_name(tgt_dtype),
                            self.rec(expr.base, PREC_NONE),
                            self.rec(expr.exponent, PREC_NONE))
                else:
                    return "%s_pow(%s, %s)" % (
                            self.complex_type_name(tgt_dtype),
                            self.rec(expr.base, PREC_NONE),
                            self.rec(expr.exponent, PREC_NONE))

        return CCodeMapper.map_power(self, expr, enclosing_prec)

    # }}}

# }}}

# vim: fdm=marker

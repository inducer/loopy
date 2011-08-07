"""Pymbolic mappers for loopy."""

from __future__ import division

from pymbolic.mapper import CombineMapper, RecursiveMapper
from pymbolic.mapper.c_code import CCodeMapper
from pymbolic.mapper.stringifier import PREC_NONE
import numpy as np




# {{{ subscript expression collector

class AllSubscriptExpressionCollector(CombineMapper):
    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, expr):
        return set()

    def map_algebraic_leaf(self, expr):
        return set()

    def map_subscript(self, expr):
        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        return set([expr])

# }}}

# {{{ coefficient collector

class CoefficientCollector(RecursiveMapper):
    def map_sum(self, expr):
        stride_dicts = [self.rec(ch) for ch in expr.children]

        result = {}
        for stride_dict in stride_dicts:
            for var, stride in stride_dict.iteritems():
                if var in result:
                    result[var] += stride
                else:
                    result[var] = stride

        return result

    def map_product(self, expr):
        result = {}

        children_coeffs = [self.rec(child) for child in expr.children]

        idx_of_child_with_vars = None
        for i, child_coeffs in enumerate(children_coeffs):
            for k in child_coeffs:
                if isinstance(k, str):
                    if (idx_of_child_with_vars is not None
                            and idx_of_child_with_vars != i):
                        raise RuntimeError(
                                "nonlinear expression")
                    idx_of_child_with_vars = i

        other_coeffs = 1
        for i, child_coeffs in enumerate(children_coeffs):
            if i != idx_of_child_with_vars:
                assert len(child_coeffs) == 1
                other_coeffs *= child_coeffs[1]

        if idx_of_child_with_vars is None:
            return {1: other_coeffs}
        else:
            return dict(
                    (var, other_coeffs*coeff)
                    for var, coeff in
                    children_coeffs[idx_of_child_with_vars].iteritems())

        return result

    def map_constant(self, expr):
        return {1: expr}

    def map_variable(self, expr):
        return {expr.name: 1}

    def map_subscript(self, expr):
        raise RuntimeError("cannot gather coefficients--indirect addressing in use")

# }}}

# {{{ variable index expression collector

class VariableIndexExpressionCollector(CombineMapper):
    def __init__(self, tgt_vector_name):
        self.tgt_vector_name = tgt_vector_name

    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, expr):
        return set()

    def map_algebraic_leaf(self, expr):
        return set()

    def map_subscript(self, expr):
        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        if expr.aggregate.name == self.tgt_vector_name:
            return set([expr.index])
        else:
            return CombineMapper.map_subscript(self, expr)

# }}}

# {{{ C code mapper

class LoopyCCodeMapper(CCodeMapper):
    def __init__(self, kernel, no_prefetch=False):
        def constant_mapper(c):
            if isinstance(c, float):
                # FIXME: type-variable
                return "%sf" % repr(c)
            else:
                return repr(c)

        CCodeMapper.__init__(self, constant_mapper=constant_mapper)
        self.kernel = kernel

        self.no_prefetch = no_prefetch

    def map_subscript(self, expr, enclosing_prec):
        from pymbolic.primitives import Variable
        if (not self.no_prefetch
                and isinstance(expr.aggregate, Variable)
                and expr.aggregate.name in self.kernel.input_vectors()):
            try:
                pf = self.kernel.prefetch[expr.aggregate.name, expr.index]
            except KeyError:
                pass
            else:
                from pymbolic.mapper.stringifier import PREC_SUM
                return pf.name+"".join(
                        "[%s - %s]" % (iname, self.rec(
                            self.kernel.get_projected_bounds(iname)[0],
                            PREC_SUM))
                        for iname in pf.inames)

        if isinstance(expr.aggregate, Variable):
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

        return CCodeMapper.map_subscript(self, expr, enclosing_prec)

    def map_floor_div(self, expr, prec):
        if isinstance(expr.denominator, int) and expr.denominator > 0:
            return ("int_floor_div_pos_b(%s, %s)"
                    % (self.rec(expr.numerator, PREC_NONE),
                        expr.denominator))
        else:
            return ("int_floor_div(%s, %s)"
                    % (self.rec(expr.numerator, PREC_NONE),
                        self.rec(expr.denominator, PREC_NONE)))

# }}}

# vim: foldmethod=marker

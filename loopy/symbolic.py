"""Pymbolic mappers for loopy."""

from __future__ import division

from pymbolic.primitives import AlgebraicLeaf
from pymbolic.mapper import (
        CombineMapper as CombineMapperBase,
        IdentityMapper as IdentityMapperBase,
        RecursiveMapper)
from pymbolic.mapper.c_code import CCodeMapper
from pymbolic.mapper.stringifier import PREC_NONE
from pymbolic.mapper.substitutor import \
        SubstitutionMapper as SubstitutionMapperBase
from pymbolic.mapper.stringifier import \
        StringifyMapper as StringifyMapperBase
from pymbolic.mapper.dependency import \
        DependencyMapper as DependencyMapperBase
from pymbolic.mapper.unifier import BidirectionalUnifier \
        as BidirectionalUnifierBase

import numpy as np
import islpy as isl
from islpy import dim_type




# {{{ loopy-specific primitives

class Reduction(AlgebraicLeaf):
    def __init__(self, operation, inames, expr):
        assert isinstance(inames, tuple)

        if isinstance(operation, str):
            from loopy.kernel import parse_reduction_op
            operation = parse_reduction_op(operation)

        self.operation = operation
        self.inames = inames
        self.expr = expr

    def __getinitargs__(self):
        return (self.operation, self.inames, self.expr)

    def get_hash(self):
        return hash((self.__class__, self.operation, self.inames,
            self.expr))

    def is_equal(self, other):
        return (other.__class__ == self.__class__
                and other.operation == self.operation
                and other.inames == self.inames
                and other.expr == self.expr)

    def stringifier(self):
        return StringifyMapper

    mapper_method = intern("map_reduction")

# }}}

# {{{ mappers with support for loopy-specific primitives

class IdentityMapperMixin(object):
    def map_reduction(self, expr):
        return Reduction(expr.operation, expr.inames, self.rec(expr.expr))

class IdentityMapper(IdentityMapperBase, IdentityMapperMixin):
    pass

class CombineMapper(CombineMapperBase):
    def map_reduction(self, expr):
        return self.rec(expr.expr)

class SubstitutionMapper(SubstitutionMapperBase, IdentityMapperMixin):
    pass

class StringifyMapper(StringifyMapperBase):
    def map_reduction(self, expr, prec):
        return "reduce(%s, [%s], %s)" % (
                expr.operation, ", ".join(expr.inames), expr.expr)

class DependencyMapper(DependencyMapperBase):
    def map_reduction(self, expr):
        return self.rec(expr.expr)

class BidirectionalUnifier(BidirectionalUnifierBase):
    def map_reduction(self, expr, other, unis):
        if not isinstance(other, type(expr)):
            return self.treat_mismatch(expr, other, unis)
        if (expr.inames != other.inames
                or type(expr.operation) != type(other.operation)):
            return []

        return self.rec(expr.expr, other.expr, unis)

# }}}

# {{{ functions to primitives

class FunctionToPrimitiveMapper(IdentityMapper):
    """Looks for invocations of a function called 'cse' or 'reduce' and
    turns those into the actual pymbolic primitives used for that.
    """

    def map_call(self, expr):
        from pymbolic.primitives import Variable
        if isinstance(expr.function, Variable) and expr.function.name == "cse":
            from pymbolic.primitives import CommonSubexpression
            if len(expr.parameters) in [1, 2]:
                if len(expr.parameters) == 2:
                    if not isinstance(expr.parameters[1], Variable):
                        raise TypeError("second argument to cse() must be a symbol")
                    tag = expr.parameters[1].name
                else:
                    tag = None

                return CommonSubexpression(
                        self.rec(expr.parameters[0]), tag)
            else:
                raise TypeError("cse takes two arguments")

        elif isinstance(expr.function, Variable):
            if expr.function.name == "reduce":
                if len(expr.parameters) == 3:
                    operation, inames, red_expr = expr.parameters
                else:
                    raise TypeError("invalid 'reduce' calling sequence")
            else:
                from loopy.kernel import parse_reduction_op
                if parse_reduction_op(expr.function.name):
                    if len(expr.parameters) != 2:
                        raise RuntimeError("invalid invocation of "
                                "reduction operation '%s'" % expr.function.name)

                    operation = expr.function
                    inames, red_expr = expr.parameters
                else:
                    return IdentityMapper.map_call(self, expr)

            red_expr = self.rec(red_expr)

            if not isinstance(operation, Variable):
                raise TypeError("operation argument to reduce() must be a symbol")
            operation = operation.name
            if isinstance(inames, Variable):
                inames = (inames,)

            if not isinstance(inames, (tuple)):
                raise TypeError("iname argument to reduce() must be a symbol "
                        "or a tuple of symbols")

            processed_inames = []
            for iname in inames:
                if not isinstance(iname, Variable):
                    raise TypeError("iname argument to reduce() must be a symbol "
                            "or a tuple or a tuple of symbols")

                processed_inames.append(iname.name)

            return Reduction(operation, tuple(processed_inames), red_expr)
        else:
            return IdentityMapper.map_call(self, expr)

# }}}

# {{{ reduction loop splitter

class ReductionLoopSplitter(IdentityMapper):
    def __init__(self, old_iname, outer_iname, inner_iname):
        self.old_iname = old_iname
        self.outer_iname = outer_iname
        self.inner_iname = inner_iname

    def map_reduction(self, expr):
        if self.old_iname in expr.inames:
            new_inames = list(expr.inames)
            new_inames.remove(self.old_iname)
            new_inames.extend([self.outer_iname, self.inner_iname])
            return Reduction(expr.operation, tuple(new_inames),
                        expr.expr)
        else:
            return IdentityMapper.map_reduction(self, expr)

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

class ArrayAccessFinder(CombineMapper):
    def __init__(self, tgt_vector_name=None):
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

        if self.tgt_vector_name is None or expr.aggregate.name == self.tgt_vector_name:
            return set([expr]) | self.rec(expr.index)
        else:
            return CombineMapper.map_subscript(self, expr)

# }}}

# {{{ C code mapper

class LoopyCCodeMapper(CCodeMapper):
    def __init__(self, kernel, cse_name_list=[], var_subst_map={},
            with_annotation=False):
        def constant_mapper(c):
            if isinstance(c, float):
                # FIXME: type-variable
                return "%sf" % repr(c)
            else:
                return repr(c)

        CCodeMapper.__init__(self, constant_mapper=constant_mapper,
                cse_name_list=cse_name_list)
        self.kernel = kernel

        self.with_annotation = with_annotation
        self.var_subst_map = var_subst_map.copy()

    def copy(self, var_subst_map=None, cse_name_list=None):
        if var_subst_map is None:
            var_subst_map = self.var_subst_map
        if cse_name_list is None:
            cse_name_list = self.cse_name_list
        return LoopyCCodeMapper(self.kernel,
                cse_name_list=cse_name_list, var_subst_map=var_subst_map)

    def copy_and_assign(self, name, value):
        var_subst_map = self.var_subst_map.copy()
        var_subst_map[name] = value
        return self.copy(var_subst_map=var_subst_map)

    def copy_and_assign_many(self, assignments):
        var_subst_map = self.var_subst_map.copy()
        var_subst_map.update(assignments)
        return self.copy(var_subst_map=var_subst_map)

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

# }}}

# {{{ aff <-> expr conversion

def aff_to_expr(aff, except_name=None, error_on_name=None):
    if except_name is not None and error_on_name is not None:
        raise ValueError("except_name and error_on_name may not be specified "
                "at the same time")
    from pymbolic import var

    except_coeff = 0

    result = int(aff.get_constant())
    for dt in [dim_type.in_, dim_type.param]:
        for i in xrange(aff.dim(dt)):
            coeff = int(aff.get_coefficient(dt, i))
            if coeff:
                dim_name = aff.get_dim_name(dt, i)
                if dim_name == except_name:
                    except_coeff += coeff
                elif dim_name == error_on_name:
                    raise RuntimeError("'%s' occurred in this subexpression--"
                            "this is not allowed" % dim_name)
                else:
                    result += coeff*var(dim_name)

    error_on_name = error_on_name or except_name

    for i in xrange(aff.dim(dim_type.div)):
        coeff = int(aff.get_coefficient(dim_type.div, i))
        if coeff:
            result += coeff*aff_to_expr(aff.get_div(i), error_on_name=error_on_name)

    denom = int(aff.get_denominator())
    if except_name is not None:
        if except_coeff % denom != 0:
            raise RuntimeError("coefficient of '%s' is not divisible by "
                    "aff denominator" % except_name)

        return result // denom, except_coeff // denom
    else:
        return result // denom




def pw_aff_to_expr(pw_aff):
    if isinstance(pw_aff, int):
        from warnings import warn
        warn("expected PwAff, got int", stacklevel=2)
        return pw_aff

    pieces = pw_aff.get_pieces()

    if len(pieces) != 1:
        raise NotImplementedError("pw_aff_to_expr for multi-piece PwAff instances")

    (set, aff), = pieces
    return aff_to_expr(aff)

def aff_from_expr(space, expr):
    zero = isl.Aff.zero_on_domain(isl.LocalSpace.from_space(space))
    context = {}
    for name, (dt, pos) in space.get_var_dict().iteritems():
        if dt == dim_type.set:
            dt = dim_type.in_

        context[name] = zero.set_coefficient(dt, pos, 1)

    from pymbolic import evaluate
    return zero + evaluate(expr, context)

# }}}

# {{{ expression <-> constraint conversion

def eq_constraint_from_expr(space, expr):
    return isl.Constraint.equality_from_aff(aff_from_expr(space,expr))

def ineq_constraint_from_expr(space, expr):
    return isl.Constraint.inequality_from_aff(aff_from_expr(space,expr))

def constraint_to_expr(cns, except_name=None):
    return aff_to_expr(cns.get_aff(), except_name=except_name)

# }}}

# {{{ CSE callback mapper

class CSECallbackMapper(IdentityMapper):
    def __init__(self, callback):
        self.callback = callback

    def map_common_subexpression(self, expr):
        result = self.callback(expr, self.rec)
        if result is None:
            return IdentityMapper.map_common_subexpression(self, expr)
        return result

# }}}

# {{{ Reduction callback mapper

class ReductionCallbackMapper(IdentityMapper):
    def __init__(self, callback):
        self.callback = callback

    def map_reduction(self, expr):
        result = self.callback(expr, self.rec)
        if result is None:
            return IdentityMapper.map_reduction(self, expr)
        return result

# }}}

# {{{ index dependency finding

class IndexVariableFinder(CombineMapper):
    def __init__(self, include_reduction_inames):
        self.include_reduction_inames = include_reduction_inames

    def combine(self, values):
        import operator
        return reduce(operator.or_, values, set())

    def map_constant(self, expr):
        return set()

    def map_algebraic_leaf(self, expr):
        return set()

    def map_subscript(self, expr):
        idx_vars = DependencyMapper()(expr.index)

        from pymbolic.primitives import Variable
        result = set()
        for idx_var in idx_vars:
            if isinstance(idx_var, Variable):
                result.add(idx_var.name)
            else:
                raise RuntimeError("index variable not understood: %s" % idx_var)
        return result

    def map_reduction(self, expr):
        result = self.rec(expr.expr)

        real_inames = set(iname.lstrip("@") for iname in expr.inames)
        if not (real_inames & result):
            raise RuntimeError("reduction '%s' does not depend on "
                    "reduction inames (%s)" % (expr, ",".join(expr.inames)))
        if self.include_reduction_inames:
            return result
        else:
            return result - real_inames

# }}}

# {{{ variable-fetch CSE mapper

class VariableFetchCSEMapper(IdentityMapper):
    """Turns fetches of a given variable names into CSEs."""
    def __init__(self, var_name, cse_tag_getter):
        self.var_name = var_name
        self.cse_tag_getter = cse_tag_getter

    def map_variable(self, expr):
        from pymbolic.primitives import CommonSubexpression
        if expr.name == self.var_name:
            return CommonSubexpression(expr, self.cse_tag_getter())
        else:
            return IdentityMapper.map_variable(self, expr)

    def map_subscript(self, expr):
        from pymbolic.primitives import CommonSubexpression, Variable, Subscript
        if (isinstance(expr.aggregate, Variable)
                and expr.aggregate.name == self.var_name):
            return CommonSubexpression(
                    Subscript(expr.aggregate, self.rec(expr.index)), self.cse_tag_getter())
        else:
            return IdentityMapper.map_subscript(self, expr)

# }}}

# {{{ CSE substitutor

class CSESubstitutor(IdentityMapper):
    def __init__(self, cses):
        """
        :arg cses: a mapping from CSE names to tuples (arg_names, expr).
        """
        self.cses = cses

    def map_variable(self, expr):
        if expr.name not in self.cses:
            return IdentityMapper.map_variable(self, expr)

        arg_names, cse_expr = self.cse[expr.name]
        if len(arg_names) != 0:
            raise RuntimeError("CSE '%s' must be invoked with %d arguments"
                    % (expr.name, len(arg_names)))

        from pymbolic.primitives import CommonSubexpression
        return CommonSubexpression(cse_expr, expr.name)

    def map_call(self, expr):
        from pymbolic.primitives import Variable, CommonSubexpression
        if (not isinstance(expr.function, Variable)
                or expr.function.name not in self.cses):
            return IdentityMapper.map_variable(self, expr)

        cse_name = expr.function.name
        arg_names, cse_expr = self.cses[cse_name]
        if len(arg_names) != len(expr.parameters):
            raise RuntimeError("CSE '%s' invoked with %d arguments (needs %d)"
                    % (cse_name, len(arg_names), len(expr.parameters)))

        from pymbolic.mapper.substitutor import make_subst_func
        subst_map = SubstitutionMapper(make_subst_func(
            dict(zip(arg_names, expr.parameters))))

        return CommonSubexpression(subst_map(cse_expr), cse_name)

# }}}

# {{{ prime-adder

class PrimeAdder(IdentityMapper):
    def __init__(self, which_vars):
        self.which_vars = which_vars

    def map_variable(self, expr):
        from pymbolic import var
        if expr.name in self.which_vars:
            return var(expr.name+"'")
        else:
            return expr

# }}}

def get_dependencies(expr):
    from loopy.symbolic import DependencyMapper
    dep_mapper = DependencyMapper(composite_leaves=False)

    return set(dep.name for dep in dep_mapper(expr))



# vim: foldmethod=marker

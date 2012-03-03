"""Pymbolic mappers for loopy."""

from __future__ import division

from pytools import memoize, memoize_method

from pymbolic.primitives import AlgebraicLeaf
from pymbolic.mapper import (
        CombineMapper as CombineMapperBase,
        IdentityMapper as IdentityMapperBase,
        RecursiveMapper,
        WalkMapper as WalkMapperBase,
        CallbackMapper as CallbackMapperBase,
        )
from pymbolic.mapper.substitutor import \
        SubstitutionMapper as SubstitutionMapperBase
from pymbolic.mapper.stringifier import \
        StringifyMapper as StringifyMapperBase
from pymbolic.mapper.dependency import \
        DependencyMapper as DependencyMapperBase
from pymbolic.mapper.unifier import UnidirectionalUnifier \
        as UnidirectionalUnifierBase

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

    @property
    @memoize_method
    def untagged_inames(self):
        return tuple(iname.lstrip("@") for iname in self.inames)

    @property
    @memoize_method
    def untagged_inames_set(self):
        return set(self.untagged_inames)

    mapper_method = intern("map_reduction")

# }}}

# {{{ mappers with support for loopy-specific primitives

class IdentityMapperMixin(object):
    def map_reduction(self, expr):
        return Reduction(expr.operation, expr.inames, self.rec(expr.expr))

class IdentityMapper(IdentityMapperBase, IdentityMapperMixin):
    pass

class WalkMapper(WalkMapperBase):
    def map_reduction(self, expr):
        self.rec(expr.expr)

class CallbackMapper(CallbackMapperBase, IdentityMapper):
    map_reduction = CallbackMapperBase.map_constant

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
        from pymbolic.primitives import Variable
        return (self.rec(expr.expr)
                - set(Variable(iname) for iname in expr.untagged_inames))

class UnidirectionalUnifier(UnidirectionalUnifierBase):
    def map_reduction(self, expr, other, unis):
        if not isinstance(other, type(expr)):
            return self.treat_mismatch(expr, other, unis)
        if (expr.inames != other.inames
                or type(expr.operation) != type(other.operation)):
            return []

        return self.rec(expr.expr, other.expr, unis)

# }}}

# {{{ functions to primitives, parsing

class FunctionToPrimitiveMapper(IdentityMapper):
    """Looks for invocations of a function called 'cse' or 'reduce' and
    turns those into the actual pymbolic primitives used for that.
    """

    def map_call(self, expr):
        from pymbolic.primitives import Variable
        if not isinstance(expr.function, Variable):
            return IdentityMapper.map_call(self, expr)

        name = expr.function.name
        if name == "cse":
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

        elif name == "reduce":
            if len(expr.parameters) == 3:
                operation, inames, red_expr = expr.parameters
            else:
                raise TypeError("invalid 'reduce' calling sequence")

        elif name == "if":
            if len(expr.parameters) in [2, 3]:
                from pymbolic.primitives import If
                return If(*expr.parameters)
            else:
                raise TypeError("if takes two or three arguments")

        else:
            # see if 'name' is an existing reduction op

            from loopy.kernel import parse_reduction_op
            if parse_reduction_op(name):
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

def parse(expr_str):
    from pymbolic import parse
    return FunctionToPrimitiveMapper()(parse(expr_str))

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

def aff_from_expr(space, expr, vars_to_zero=set()):
    zero = isl.Aff.zero_on_domain(isl.LocalSpace.from_space(space))
    context = {}
    for name, (dt, pos) in space.get_var_dict().iteritems():
        if dt == dim_type.set:
            dt = dim_type.in_

        context[name] = zero.set_coefficient(dt, pos, 1)

    for name in vars_to_zero:
        context[name] = zero

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

        if not (expr.untagged_inames_set & result):
            raise RuntimeError("reduction '%s' does not depend on "
                    "reduction inames (%s)" % (expr, ",".join(expr.inames)))
        if self.include_reduction_inames:
            return result
        else:
            return result - expr.untagged_inames_set

# }}}

# {{{ substitution callback mapper

class SubstitutionCallbackMapper(IdentityMapper):
    @staticmethod
    def parse_filter(filt):
        if not isinstance(filt, tuple):
            dotted_components = filt.split(".")
            if len(dotted_components) == 1:
                return (dotted_components[0], None)
            elif len(dotted_components) == 2:
                return tuple(dotted_components)
            else:
                raise RuntimeError("too many dotted components in '%s'" % filt)
        else:
            if len(filt) != 2:
                raise RuntimeError("substitution name filters "
                        "may have at most two components")

            return filt

    def __init__(self, names_filter, func):
        if names_filter is not None:
            new_names_filter = []
            for filt in names_filter:
                new_names_filter.append(self.parse_filter(filt))

            self.names_filter = new_names_filter
        else:
            self.names_filter = names_filter

        self.func = func

    def parse_name(self, expr):
        from pymbolic.primitives import Variable, Lookup
        if isinstance(expr, Variable):
            e_name, e_instance = expr.name, None
        elif isinstance(expr, Lookup):
            if not isinstance(expr.aggregate, Variable):
                return None
            e_name, e_instance = expr.aggregate.name, expr.name
        else:
            return None

        if self.names_filter is not None:
            for filt_name, filt_instance in self.names_filter:
                if e_name == filt_name:
                    if filt_instance is None or filt_instance == e_instance:
                        return e_name, e_instance
        else:
            return e_name, e_instance

        return None

    def map_variable(self, expr):
        parsed_name = self.parse_name(expr)
        if parsed_name is None:
            return IdentityMapper.map_variable(self, expr)

        name, instance = parsed_name

        result = self.func(expr, name, instance, (), self.rec)
        if result is None:
            return IdentityMapper.map_variable(self, expr)
        else:
            return result

    def map_lookup(self, expr):
        parsed_name = self.parse_name(expr)
        if parsed_name is None:
            return IdentityMapper.map_lookup(self, expr)

        name, instance = parsed_name

        result = self.func(expr, name, instance, (), self.rec)
        if result is None:
            return IdentityMapper.map_lookup(self, expr)
        else:
            return result

    def map_call(self, expr):
        parsed_name = self.parse_name(expr.function)
        if parsed_name is None:
            return IdentityMapper.map_call(self, expr)

        name, instance = parsed_name

        result = self.func(expr, name, instance, expr.parameters, self.rec)
        if result is None:
            return IdentityMapper.map_call(self, expr)
        else:
            return result

# }}}

# {{{ parametrized substitutor

class ParametrizedSubstitutor(object):
    def __init__(self, rules, one_level=False):
        self.rules = rules
        self.one_level = one_level

    def __call__(self, expr):
        level = [0]

        def expand_if_known(expr, name, instance, args, rec):
            if self.one_level and level[0] > 0:
                return None

            rule = self.rules[name]
            if len(rule.arguments) != len(args):
                raise RuntimeError("Rule '%s' invoked with %d arguments (needs %d)"
                        % (name, len(args), len(rule.arguments), ))

            from pymbolic.mapper.substitutor import make_subst_func
            subst_map = SubstitutionMapper(make_subst_func(
                dict(zip(rule.arguments, args))))

            level[0] += 1
            result = rec(subst_map(rule.expression))
            level[0] -= 1

            return result

        scm = SubstitutionCallbackMapper(self.rules.keys(), expand_if_known)
        return scm(expr)

# }}}

# {{{ wildcard -> unique variable mapper

class WildcardToUniqueVariableMapper(IdentityMapper):
    def __init__(self, unique_var_name_factory):
        self.unique_var_name_factory = unique_var_name_factory

    def map_wildcard(self, expr):
        from pymbolic import var
        return var(self.unique_var_name_factory())

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

@memoize
def get_dependencies(expr):
    from loopy.symbolic import DependencyMapper
    dep_mapper = DependencyMapper(composite_leaves=False)

    return set(dep.name for dep in dep_mapper(expr))



# vim: foldmethod=marker

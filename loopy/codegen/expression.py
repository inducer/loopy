from __future__ import division
from __future__ import absolute_import
from six.moves import range
from six.moves import zip

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
from pytools import Record

from loopy.tools import is_integer
from loopy.diagnostic import TypeInferenceFailure, DependencyTypeInferenceFailure


# {{{ type inference

class TypeInferenceMapper(CombineMapper):
    def __init__(self, kernel, new_assignments=None):
        """
        :arg new_assignments: mapping from names to either
            :class:`loopy.kernel.data.TemporaryVariable`
            or
            :class:`loopy.kernel.data.KernelArgument`
            instances
        """
        self.kernel = kernel
        if new_assignments is None:
            new_assignments = {}
        self.new_assignments = new_assignments

    # /!\ Introduce caches with care--numpy.float32(x) and numpy.float64(x)
    # are Python-equal (for many common constants such as integers).

    @staticmethod
    def combine(dtypes):
        dtypes = list(dtypes)

        result = dtypes.pop()
        while dtypes:
            other = dtypes.pop()

            if result.isbuiltin and other.isbuiltin:
                if (result, other) in [
                        (np.int32, np.float32), (np.int32, np.float32)]:
                    # numpy makes this a double. I disagree.
                    result = np.dtype(np.float32)
                else:
                    result = (
                            np.empty(0, dtype=result)
                            + np.empty(0, dtype=other)
                            ).dtype
            elif result.isbuiltin and not other.isbuiltin:
                # assume the non-native type takes over
                result = other
            elif not result.isbuiltin and other.isbuiltin:
                # assume the non-native type takes over
                pass
            else:
                if result is not other:
                    raise TypeInferenceFailure(
                            "nothing known about result of operation on "
                            "'%s' and '%s'" % (result, other))

        return result

    def map_sum(self, expr):
        dtypes = []
        small_integer_dtypes = []
        for child in expr.children:
            dtype = self.rec(child)
            if is_integer(child) and abs(child) < 1024:
                small_integer_dtypes.append(dtype)
            else:
                dtypes.append(dtype)

        from pytools import all
        if all(dtype.kind == "i" for dtype in dtypes):
            dtypes.extend(small_integer_dtypes)

        return self.combine(dtypes)

    map_product = map_sum

    def map_quotient(self, expr):
        n_dtype = self.rec(expr.numerator)
        d_dtype = self.rec(expr.denominator)

        if n_dtype.kind in "iu" and d_dtype.kind in "iu":
            # both integers
            return np.dtype(np.float64)

        else:
            return self.combine([n_dtype, d_dtype])

    def map_constant(self, expr):
        if is_integer(expr):
            for tp in [np.int32, np.int64]:
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
            if np.complex64(expr) == np.complex128(expr):
                # (COMPLEX_GUESS_LOGIC)
                # No precision is lost by 'guessing' single precision, use that.
                # This at least covers simple cases like '1j'.
                return np.dtype(np.complex64)

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
        if expr.name in self.kernel.all_inames():
            return self.kernel.index_dtype

        result = self.kernel.mangle_symbol(expr.name)
        if result is not None:
            result_dtype, _ = result
            return result_dtype

        obj = self.new_assignments.get(expr.name)

        if obj is None:
            obj = self.kernel.arg_dict.get(expr.name)

        if obj is None:
            obj = self.kernel.temporary_variables.get(expr.name)

        if obj is None:
            raise TypeInferenceFailure("name not known in type inference: %s"
                    % expr.name)

        from loopy.kernel.data import TemporaryVariable, KernelArgument
        import loopy as lp
        if isinstance(obj, TemporaryVariable):
            result = obj.dtype
            if result is lp.auto:
                raise DependencyTypeInferenceFailure(
                        "temporary variable '%s'" % expr.name,
                        expr.name)
            else:
                return result

        elif isinstance(obj, KernelArgument):
            result = obj.dtype
            if result is None:
                raise DependencyTypeInferenceFailure(
                        "argument '%s'" % expr.name,
                        expr.name)
            else:
                return result

        else:
            raise RuntimeError("unexpected type inference "
                    "object type for '%s'" % expr.name)

    map_tagged_variable = map_variable

    def map_lookup(self, expr):
        agg_result = self.rec(expr.aggregate)
        dtype, offset = agg_result.fields[expr.name]
        return dtype

    def map_comparison(self, expr):
        # "bool" is unusable because OpenCL's bool has indeterminate memory
        # format.
        return np.dtype(np.int32)

    map_logical_not = map_comparison
    map_logical_and = map_comparison
    map_logical_or = map_comparison

    def map_reduction(self, expr):
        return expr.operation.result_dtype(
                self.kernel.target, self.rec(expr.expr), expr.inames)

# }}}


# {{{ C code mapper

# type_context may be:
# - 'i' for integer -
# - 'f' for single-precision floating point
# - 'd' for double-precision floating point
# or None for 'no known context'.

def dtype_to_type_context(target, dtype):
    dtype = np.dtype(dtype)

    if dtype.kind == 'i':
        return 'i'
    if dtype in [np.float64, np.complex128]:
        return 'd'
    if dtype in [np.float32, np.complex64]:
        return 'f'
    if target.is_vector_dtype(dtype):
        return dtype_to_type_context(target, dtype.fields["x"][0])

    return None


def get_opencl_vec_member(idx):
    if idx is None:
        return idx

    # The 'int' avoids an 'L' suffix for long ints.
    return "s%s" % hex(int(idx))[2:]


class SeenFunction(Record):
    """
    .. attribute:: name
    .. attribute:: c_name
    .. attribute:: arg_dtypes

        a tuple of arg dtypes
    """

    def __init__(self, name, c_name, arg_dtypes):
        Record.__init__(self,
                name=name,
                c_name=c_name,
                arg_dtypes=arg_dtypes)

    def __hash__(self):
        return hash((type(self),)
                + tuple((f, getattr(self, f)) for f in type(self).fields))


class LoopyCCodeMapper(RecursiveMapper):
    def __init__(self, kernel, seen_dtypes, seen_functions, var_subst_map={},
            allow_complex=False):
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

        self.var_subst_map = var_subst_map.copy()

    # {{{ copy helpers

    def copy(self, var_subst_map=None):
        if var_subst_map is None:
            var_subst_map = self.var_subst_map
        return LoopyCCodeMapper(self.kernel, self.seen_dtypes, self.seen_functions,
                var_subst_map=var_subst_map,
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
            if self.kernel.options.annotate_inames:
                return " /* %s */ %s" % (
                        expr.name,
                        self.rec(self.var_subst_map[expr.name],
                            enclosing_prec, type_context))
            else:
                return str(self.rec(self.var_subst_map[expr.name],
                    enclosing_prec, type_context))
        elif expr.name in self.kernel.arg_dict:
            arg = self.kernel.arg_dict[expr.name]
            from loopy.kernel.array import ArrayBase
            if isinstance(arg, ArrayBase):
                if arg.shape == ():
                    if arg.offset:
                        # FIXME
                        raise NotImplementedError("in-memory scalar with offset")
                    else:
                        return "*"+expr.name
                else:
                    raise RuntimeError("unsubscripted reference to array '%s'"
                            % expr.name)

        result = self.kernel.mangle_symbol(expr.name)
        if result is not None:
            _, c_name = result
            return c_name

        return expr.name

    def map_tagged_variable(self, expr, enclosing_prec, type_context):
        return expr.name

    def map_lookup(self, expr, enclosing_prec, type_context):
        return self.parenthesize_if_needed(
                "%s.%s" % (
                    self.rec(expr.aggregate, PREC_CALL, type_context), expr.name),
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

        from loopy.kernel.array import get_access_info
        from pymbolic import evaluate

        access_info = get_access_info(self.kernel.target, ary, expr.index,
                lambda expr: evaluate(expr, self.var_subst_map))

        vec_member = get_opencl_vec_member(access_info.vector_index)

        from loopy.kernel.data import ImageArg, GlobalArg, TemporaryVariable

        if isinstance(ary, ImageArg):
            base_access = ("read_imagef(%s, loopy_sampler, (float%d)(%s))"
                    % (ary.name, ary.dimensions,
                        ", ".join(self.rec(idx, PREC_NONE, 'i')
                            for idx in expr.index[::-1])))

            if ary.dtype == np.float32:
                return base_access+".x"
            if self.kernel.target.is_vector_dtype(ary.dtype):
                return base_access
            elif ary.dtype == np.float64:
                return "as_double(%s.xy)" % base_access
            else:
                raise NotImplementedError(
                        "non-floating-point images not supported for now")

        elif isinstance(ary, (GlobalArg, TemporaryVariable)):
            if len(access_info.subscripts) == 0:
                if isinstance(ary, GlobalArg):
                    # unsubscripted global args are pointers
                    if vec_member is not None:
                        return "%s->%s" % (
                                access_info.array_name,
                                vec_member)
                    else:
                        return "*" + access_info.array_name

                else:
                    # unsubscripted temp vars are scalars
                    if vec_member is not None:
                        return "%s.%s" % (
                                access_info.array_name,
                                vec_member)
                    else:
                        return access_info.array_name

            else:
                subscript, = access_info.subscripts
                result = self.parenthesize_if_needed(
                        "%s[%s]" % (
                            access_info.array_name,
                            self.rec(subscript, PREC_NONE, 'i')),
                        enclosing_prec, PREC_CALL)

                if vec_member:
                    result += "."+vec_member

                return result

        else:
            assert False

    def map_linear_subscript(self, expr, enclosing_prec, type_context):
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

                return self.parenthesize_if_needed(
                        "%s[%s]" % (
                            expr.aggregate.name,
                            self.rec(offset + expr.index, PREC_NONE, 'i')),
                        enclosing_prec, PREC_CALL)

        elif expr.aggregate.name in self.kernel.temporary_variables:
            raise RuntimeError("linear indexing is not supported on temporaries: %s"
                    % expr)

        else:
            raise RuntimeError(
                    "nothing known about variable '%s'" % expr.aggregate.name)

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
            self.seen_functions.add(SeenFunction(name, name, (idt, idt)))

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
            try:
                dtype = expr.dtype
            except AttributeError:
                # (COMPLEX_GUESS_LOGIC)
                # This made it through type 'guessing' above, and it
                # was concluded above (search for COMPLEX_GUESS_LOGIC),
                # that nothing was lost by using single precision.
                cast_type = "cfloat_t"
            else:
                if dtype == np.complex128:
                    cast_type = "cdouble_t"
                elif dtype == np.complex64:
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
                if is_integer(expr):
                    return str(expr)

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
                        self.rec(par, PREC_NONE,
                            dtype_to_type_context(self.kernel.target, tgt_dtype),
                            tgt_dtype)
                        for par, par_dtype, tgt_dtype in zip(
                            expr.parameters, par_dtypes, arg_tgt_dtypes)]
            else:
                raise RuntimeError("result of function mangler "
                        "for function '%s' not understood"
                        % identifier)

        self.seen_functions.add(SeenFunction(identifier, c_name, par_dtypes))
        if str_parameters is None:
            # /!\ FIXME For some functions (e.g. 'sin'), it makes sense to
            # propagate the type context here. But for many others, it does
            # not. Using the inferred type as a stopgap for now.
            str_parameters = [
                    self.rec(par, PREC_NONE,
                        type_context=dtype_to_type_context(
                            self.kernel.target, par_dtype))
                    for par, par_dtype in zip(expr.parameters, par_dtypes)]

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

        # I've added 'type_context == "i"' because of the following
        # idiotic corner case: Code generation for subscripts comes
        # through here, and it may involve variables that we know
        # nothing about (offsets and such). If we fall into the allow_complex
        # branch, we'll try to do type inference on these variables,
        # and stuff breaks. This band-aid works around that. -AK
        if not self.allow_complex or type_context == "i":
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
            complex_sum = self.join_rec(
                    " + ", complexes, PREC_SUM, type_context, tgt_dtype)

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

        # I've added 'type_context == "i"' because of the following
        # idiotic corner case: Code generation for subscripts comes
        # through here, and it may involve variables that we know
        # nothing about (offsets and such). If we fall into the allow_complex
        # branch, we'll try to do type inference on these variables,
        # and stuff breaks. This band-aid works around that. -AK
        if not self.allow_complex or type_context == "i":
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
            num = self.rec(expr.numerator, PREC_PRODUCT, type_context, num_tgt_dtype)

            # analogous to ^{-1}
            denom = self.rec(expr.denominator, PREC_POWER, type_context)

            if n_dtype.kind not in "fc" and d_dtype.kind not in "fc":
                # must both be integers
                if type_context == "f":
                    num = "((float) (%s))" % num
                    denom = "((float) (%s))" % denom
                elif type_context == "f":
                    num = "((double) (%s))" % num
                    denom = "((double) (%s))" % denom

            return self.parenthesize_if_needed(
                    "%s / %s" % (
                        # Space is necessary--otherwise '/*'
                        # (i.e. divide-dererference) becomes
                        # start-of-comment in C.
                        num,
                        denom),
                    enclosing_prec, PREC_PRODUCT)

        n_dtype = self.infer_type(expr.numerator)
        d_dtype = self.infer_type(expr.denominator)

        if not self.allow_complex:
            return base_impl(expr, enclosing_prec, type_context)

        n_complex = 'c' == n_dtype.kind
        d_complex = 'c' == d_dtype.kind

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
                    # PREC_POWER analogous to ^{-1}
                    self.rec(expr.denominator, PREC_POWER, type_context))

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
                    return self.rec(
                            expr.base*expr.base, enclosing_prec, type_context)

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
                            self.rec(expr.exponent, PREC_NONE,
                                type_context, tgt_dtype))

        return base_impl(expr, enclosing_prec, type_context)

    # }}}

# }}}

# vim: fdm=marker

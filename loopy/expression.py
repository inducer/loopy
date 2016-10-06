from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2012-15 Andreas Kloeckner"

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

from pymbolic.mapper import CombineMapper, RecursiveMapper

from loopy.tools import is_integer
from loopy.types import NumpyType
from loopy.codegen import Unvectorizable
from loopy.diagnostic import (
        LoopyError,
        TypeInferenceFailure, DependencyTypeInferenceFailure)


# type_context may be:
# - 'i' for integer -
# - 'f' for single-precision floating point
# - 'd' for double-precision floating point
# or None for 'no known context'.

def dtype_to_type_context(target, dtype):
    from loopy.types import NumpyType

    if dtype.is_integral():
        return 'i'
    if isinstance(dtype, NumpyType) and dtype.dtype in [np.float64, np.complex128]:
        return 'd'
    if isinstance(dtype, NumpyType) and dtype.dtype in [np.float32, np.complex64]:
        return 'f'
    if target.is_vector_dtype(dtype):
        return dtype_to_type_context(
                target, NumpyType(dtype.numpy_dtype.fields["x"][0]))

    return None


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

    def with_assignments(self, names_to_vars):
        new_ass = self.new_assignments.copy()
        new_ass.update(names_to_vars)
        return type(self)(self.kernel, new_ass)

    @staticmethod
    def combine(dtypes):
        # dtypes may just be a generator expr
        dtypes = list(dtypes)

        from loopy.types import LoopyType, NumpyType
        assert all(isinstance(dtype, LoopyType) for dtype in dtypes)

        if not all(isinstance(dtype, NumpyType) for dtype in dtypes):
            from pytools import is_single_valued, single_valued
            if not is_single_valued(dtypes):
                raise TypeInferenceFailure(
                        "Nothing known about operations between '%s'"
                        % ", ".join(str(dt) for dt in dtypes))

            return single_valued(dtypes)

        dtypes = [dtype.dtype for dtype in dtypes]

        result = dtypes.pop()
        while dtypes:
            other = dtypes.pop()

            if result.fields is None and other.fields is None:
                if (result, other) in [
                        (np.int32, np.float32), (np.float32, np.int32)]:
                    # numpy makes this a double. I disagree.
                    result = np.dtype(np.float32)
                else:
                    result = (
                            np.empty(0, dtype=result)
                            + np.empty(0, dtype=other)
                            ).dtype

            elif result.fields is None and other.fields is not None:
                # assume the non-native type takes over
                # (This is used for vector types.)
                result = other
            elif result.fields is not None and other.fields is None:
                # assume the non-native type takes over
                # (This is used for vector types.)
                pass
            else:
                if result is not other:
                    raise TypeInferenceFailure(
                            "nothing known about result of operation on "
                            "'%s' and '%s'" % (result, other))

        return NumpyType(result)

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
        if all(dtype.is_integral() for dtype in dtypes):
            dtypes.extend(small_integer_dtypes)

        return self.combine(dtypes)

    map_product = map_sum

    def map_quotient(self, expr):
        n_dtype = self.rec(expr.numerator)
        d_dtype = self.rec(expr.denominator)

        if n_dtype.is_integral() and d_dtype.is_integral():
            # both integers
            return NumpyType(np.dtype(np.float64))

        else:
            return self.combine([n_dtype, d_dtype])

    def map_constant(self, expr):
        if is_integer(expr):
            for tp in [np.int32, np.int64]:
                iinfo = np.iinfo(tp)
                if iinfo.min <= expr <= iinfo.max:
                    return NumpyType(np.dtype(tp))

            else:
                raise TypeInferenceFailure("integer constant '%s' too large" % expr)

        dt = np.asarray(expr).dtype
        if hasattr(expr, "dtype"):
            return NumpyType(expr.dtype)
        elif isinstance(expr, np.number):
            # Numpy types are sized
            return NumpyType(np.dtype(type(expr)))
        elif dt.kind == "f":
            # deduce the smaller type by default
            return NumpyType(np.dtype(np.float32))
        elif dt.kind == "c":
            if np.complex64(expr) == np.complex128(expr):
                # (COMPLEX_GUESS_LOGIC)
                # No precision is lost by 'guessing' single precision, use that.
                # This at least covers simple cases like '1j'.
                return NumpyType(np.dtype(np.complex64))

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

    def map_call(self, expr, multiple_types_ok=False):
        from pymbolic.primitives import Variable

        identifier = expr.function
        if isinstance(identifier, Variable):
            identifier = identifier.name

        if identifier in ["indexof", "indexof_vec"]:
            return self.kernel.index_dtype

        arg_dtypes = tuple(self.rec(par) for par in expr.parameters)

        mangle_result = self.kernel.mangle_function(identifier, arg_dtypes)
        if multiple_types_ok:
            if mangle_result is not None:
                return mangle_result.result_dtypes
        else:
            if mangle_result is not None:
                if len(mangle_result.result_dtypes) != 1 and not multiple_types_ok:
                    raise LoopyError("functions with more or fewer than one "
                            "return value may only be used in direct assignments")

                return mangle_result.result_dtypes[0]

        raise RuntimeError("unable to resolve "
                "function '%s' with %d given arguments"
                % (identifier, len(arg_dtypes)))

    def map_variable(self, expr):
        if expr.name in self.kernel.all_inames():
            return self.kernel.index_dtype

        result = self.kernel.mangle_symbol(
                self.kernel.target.get_device_ast_builder(),
                expr.name)

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
        field = agg_result.numpy_dtype.fields[expr.name]
        dtype = field[0]
        return NumpyType(dtype)

    def map_comparison(self, expr):
        # "bool" is unusable because OpenCL's bool has indeterminate memory
        # format.
        return NumpyType(np.dtype(np.int32))

    map_logical_not = map_comparison
    map_logical_and = map_comparison
    map_logical_or = map_comparison

    def map_group_hw_index(self, expr, *args):
        return self.kernel.index_dtype

    def map_local_hw_index(self, expr, *args):
        return self.kernel.index_dtype

    def map_reduction(self, expr, multiple_types_ok=False):
        result = expr.operation.result_dtypes(
                self.kernel, self.rec(expr.expr), expr.inames)

        if multiple_types_ok:
            return result

        else:
            if len(result) != 1 and not multiple_types_ok:
                raise LoopyError("reductions with more or fewer than one "
                        "return value may only be used in direct assignments")

            return result[0]

# }}}


# {{{ vetorizability checker

class VectorizabilityChecker(RecursiveMapper):
    """The return value from this mapper is a :class:`bool` indicating whether
    the result of the expression is vectorized along :attr:`vec_iname`.
    If the expression is not vectorizable, the mapper raises :class:`Unvectorizable`.

    .. attribute:: vec_iname
    """

    def __init__(self, kernel, vec_iname, vec_iname_length):
        self.kernel = kernel
        self.vec_iname = vec_iname
        self.vec_iname_length = vec_iname_length

    @staticmethod
    def combine(vectorizabilities):
        from functools import reduce
        from operator import and_
        return reduce(and_, vectorizabilities)

    def map_sum(self, expr):
        return any(self.rec(child) for child in expr.children)

    map_product = map_sum

    def map_quotient(self, expr):
        return (self.rec(expr.numerator)
                or
                self.rec(expr.denominator))

    def map_linear_subscript(self, expr):
        return False

    def map_call(self, expr):
        # FIXME: Should implement better vectorization check for function calls

        rec_pars = [
                self.rec(child) for child in expr.parameters]
        if any(rec_pars):
            raise Unvectorizable("fucntion calls cannot yet be vectorized")

        return False

    def map_subscript(self, expr):
        name = expr.aggregate.name

        var = self.kernel.arg_dict.get(name)
        if var is None:
            var = self.kernel.temporary_variables.get(name)

        if var is None:
            raise LoopyError("unknown array variable in subscript: %s"
                    % name)

        from loopy.kernel.array import ArrayBase
        if not isinstance(var, ArrayBase):
            raise LoopyError("non-array subscript '%s'" % expr)

        index = expr.index_tuple

        from loopy.symbolic import get_dependencies
        from loopy.kernel.array import VectorArrayDimTag
        from pymbolic.primitives import Variable

        possible = None
        for i in range(len(var.shape)):
            if (
                    isinstance(var.dim_tags[i], VectorArrayDimTag)
                    and isinstance(index[i], Variable)
                    and index[i].name == self.vec_iname):
                if var.shape[i] != self.vec_iname_length:
                    raise Unvectorizable("vector length was mismatched")

                if possible is None:
                    possible = True

            else:
                if self.vec_iname in get_dependencies(index[i]):
                    raise Unvectorizable("vectorizing iname '%s' occurs in "
                            "unvectorized subscript axis %d (1-based) of "
                            "expression '%s'"
                            % (self.vec_iname, i+1, expr))
                    break

        return bool(possible)

    def map_constant(self, expr):
        return False

    def map_variable(self, expr):
        if expr.name == self.vec_iname:
            # Technically, this is doable. But we're not going there.
            raise Unvectorizable()

        # A single variable is always a scalar.
        return False

    map_tagged_variable = map_variable

    def map_lookup(self, expr):
        if self.rec(expr.aggregate):
            raise Unvectorizable()

        return False

    def map_comparison(self, expr):
        # FIXME: These actually can be vectorized:
        # https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/relationalFunctions.html

        raise Unvectorizable()

    def map_logical_not(self, expr):
        raise Unvectorizable()

    map_logical_and = map_logical_not
    map_logical_or = map_logical_not

    def map_reduction(self, expr):
        # FIXME: Do this more carefully
        raise Unvectorizable()

# }}}

# vim: fdm=marker

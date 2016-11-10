from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2012-16 Andreas Kloeckner"

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

from pymbolic.mapper import CombineMapper
import numpy as np

from loopy.tools import is_integer
from loopy.types import NumpyType

from loopy.diagnostic import (
        LoopyError,
        TypeInferenceFailure, DependencyTypeInferenceFailure)

import logging
logger = logging.getLogger(__name__)


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


# {{{ infer types

def _infer_var_type(kernel, var_name, type_inf_mapper, subst_expander):
    if var_name in kernel.all_params():
        return kernel.index_dtype, []

    def debug(s):
        logger.debug("%s: %s" % (kernel.name, s))

    dtypes = []

    import loopy as lp

    symbols_with_unavailable_types = []

    from loopy.diagnostic import DependencyTypeInferenceFailure
    for writer_insn_id in kernel.writer_map().get(var_name, []):
        writer_insn = kernel.id_to_insn[writer_insn_id]
        if not isinstance(writer_insn, lp.MultiAssignmentBase):
            continue

        expr = subst_expander(writer_insn.expression)

        try:
            debug("             via expr %s" % expr)
            if isinstance(writer_insn, lp.Assignment):
                result = type_inf_mapper(expr)
            elif isinstance(writer_insn, lp.CallInstruction):
                result_dtypes = type_inf_mapper(expr, multiple_types_ok=True)

                result = None
                for assignee, comp_dtype in zip(
                        writer_insn.assignee_var_names(), result_dtypes):
                    if assignee == var_name:
                        result = comp_dtype
                        break

                assert result is not None

            debug("             result: %s" % result)

            dtypes.append(result)

        except DependencyTypeInferenceFailure as e:
            debug("             failed: %s" % e)
            symbols_with_unavailable_types.append(e.symbol)
            #dtypes = None
            #break

    if not dtypes:
        return None, symbols_with_unavailable_types

    result = type_inf_mapper.combine(dtypes)

    return result, []


class _DictUnionView:
    def __init__(self, children):
        self.children = children

    def get(self, key):
        try:
            return self[key]
        except KeyError:
            return None

    def __getitem__(self, key):
        for ch in self.children:
            try:
                return ch[key]
            except KeyError:
                pass

        raise KeyError(key)


def infer_unknown_types(kernel, expect_completion=False):
    """Infer types on temporaries and arguments."""

    logger.debug("%s: infer types" % kernel.name)

    def debug(s):
        logger.debug("%s: %s" % (kernel.name, s))

    unexpanded_kernel = kernel
    if kernel.substitutions:
        from loopy.transform.subst import expand_subst
        kernel = expand_subst(kernel)

    new_temp_vars = kernel.temporary_variables.copy()
    new_arg_dict = kernel.arg_dict.copy()

    # {{{ fill queue

    # queue contains temporary variables
    queue = []

    import loopy as lp
    for tv in six.itervalues(kernel.temporary_variables):
        if tv.dtype is lp.auto:
            queue.append(tv)

    for arg in kernel.args:
        if arg.dtype is None:
            queue.append(arg)

    # }}}

    type_inf_mapper = TypeInferenceMapper(kernel,
            _DictUnionView([
                new_temp_vars,
                new_arg_dict
                ]))

    from loopy.symbolic import SubstitutionRuleExpander
    subst_expander = SubstitutionRuleExpander(kernel.substitutions)

    # {{{ work on type inference queue

    from loopy.kernel.data import TemporaryVariable, KernelArgument

    failed_names = set()
    while queue:
        item = queue.pop(0)

        debug("inferring type for %s %s" % (type(item).__name__, item.name))

        result, symbols_with_unavailable_types = \
                _infer_var_type(kernel, item.name, type_inf_mapper, subst_expander)

        failed = result is None
        if not failed:
            debug("     success: %s" % result)
            if isinstance(item, TemporaryVariable):
                new_temp_vars[item.name] = item.copy(dtype=result)
            elif isinstance(item, KernelArgument):
                new_arg_dict[item.name] = item.copy(dtype=result)
            else:
                raise LoopyError("unexpected item type in type inference")
        else:
            debug("     failure")

        if failed:
            if item.name in failed_names:
                # this item has failed before, give up.
                advice = ""
                if symbols_with_unavailable_types:
                    advice += (
                            " (need type of '%s'--check for missing arguments)"
                            % ", ".join(symbols_with_unavailable_types))

                if expect_completion:
                    raise LoopyError(
                            "could not determine type of '%s'%s"
                            % (item.name, advice))

                else:
                    # We're done here.
                    break

            # remember that this item failed
            failed_names.add(item.name)

            queue_names = set(qi.name for qi in queue)

            if queue_names == failed_names:
                # We did what we could...
                print(queue_names, failed_names, item.name)
                assert not expect_completion
                break

            # can't infer type yet, put back into queue
            queue.append(item)
        else:
            # we've made progress, reset failure markers
            failed_names = set()

    # }}}

    return unexpanded_kernel.copy(
            temporary_variables=new_temp_vars,
            args=[new_arg_dict[arg.name] for arg in kernel.args],
            )

# }}}

# vim: foldmethod=marker

"""Data used by the kernel object."""

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
from pytools import Record, memoize_method




# {{{ index tags

class IndexTag(Record):
    __slots__ = []

    def __hash__(self):
        raise RuntimeError("use .key to hash index tags")




class ParallelTag(IndexTag):
    pass

class HardwareParallelTag(ParallelTag):
    pass

class UniqueTag(IndexTag):
    @property
    def key(self):
        return type(self)

class AxisTag(UniqueTag):
    __slots__ = ["axis"]

    def __init__(self, axis):
        Record.__init__(self,
                axis=axis)

    @property
    def key(self):
        return (type(self), self.axis)

    def __str__(self):
        return "%s.%d" % (
                self.print_name, self.axis)

class GroupIndexTag(HardwareParallelTag, AxisTag):
    print_name = "g"

class LocalIndexTagBase(HardwareParallelTag):
    pass

class LocalIndexTag(LocalIndexTagBase, AxisTag):
    print_name = "l"

class AutoLocalIndexTagBase(LocalIndexTagBase):
    pass

class AutoFitLocalIndexTag(AutoLocalIndexTagBase):
    def __str__(self):
        return "l.auto"

class IlpBaseTag(ParallelTag):
    pass

class UnrolledIlpTag(IlpBaseTag):
    def __str__(self):
        return "ilp.unr"

class LoopedIlpTag(IlpBaseTag):
    def __str__(self):
        return "ilp.seq"

class UnrollTag(IndexTag):
    def __str__(self):
        return "unr"

class ForceSequentialTag(IndexTag):
    def __str__(self):
        return "forceseq"

def parse_tag(tag):
    if tag is None:
        return tag

    if isinstance(tag, IndexTag):
        return tag

    if not isinstance(tag, str):
        raise ValueError("cannot parse tag: %s" % tag)

    if tag == "for":
        return None
    elif tag in ["unr"]:
        return UnrollTag()
    elif tag in ["ilp", "ilp.unr"]:
        return UnrolledIlpTag()
    elif tag == "ilp.seq":
        return LoopedIlpTag()
    elif tag.startswith("g."):
        return GroupIndexTag(int(tag[2:]))
    elif tag.startswith("l."):
        axis = tag[2:]
        if axis == "auto":
            return AutoFitLocalIndexTag()
        else:
            return LocalIndexTag(int(axis))
    else:
        raise ValueError("cannot parse tag: %s" % tag)

# }}}

# {{{ arguments

class auto_shape:
    pass

class auto_strides:
    pass

def make_strides(shape, order):
    from pyopencl.compyte.array import (
            f_contiguous_strides,
            c_contiguous_strides)

    if order == "F":
        return f_contiguous_strides(1, shape)
    elif order == "C":
        return c_contiguous_strides(1, shape)
    else:
        raise ValueError("invalid order: %s" % order)

class ShapedArg(Record):
    def __init__(self, name, dtype=None, shape=None, strides=None, order=None,
            offset=0):
        """
        All of the following are optional. Specify either strides or shape.

        :arg shape:
        :arg strides: like numpy strides, but in multiples of
            data type size
        :arg order:
        :arg offset: Offset from the beginning of the vector from which
            the strides are counted.
        """
        if dtype is not None:
            dtype = np.dtype(dtype)

        def parse_if_necessary(x):
            if isinstance(x, str):
                from pymbolic import parse
                return parse(x)
            else:
                return x

        def process_tuple(x):
            x = parse_if_necessary(x)
            if not isinstance(x, tuple):
                x = (x,)

            return tuple(parse_if_necessary(xi) for xi in x)

        if strides == "auto":
            strides = auto_strides
        if shape == "auto":
            shape = auto_shape

        strides_known = strides is not None and strides is not auto_strides
        shape_known = shape is not None and shape is not auto_shape

        if strides_known:
            strides = process_tuple(strides)

        if shape_known:
            shape = process_tuple(shape)

        if not strides_known and shape_known:
            if len(shape) == 1:
                # don't need order to know that
                strides = (1,)
            elif order is not None:
                strides = make_strides(shape, order)

        Record.__init__(self,
                name=name,
                dtype=dtype,
                strides=strides,
                offset=offset,
                shape=shape)

    @property
    @memoize_method
    def numpy_strides(self):
        return tuple(self.dtype.itemsize*s for s in self.strides)

    @property
    def dimensions(self):
        return len(self.strides)

    def __str__(self):
        if self.shape is None:
            shape = "unknown"
        elif self.shape is auto_shape:
            shape = "auto"
        else:
            shape = ",".join(str(i) for i in self.shape)

        if self.strides is None:
            strides = "unknown"
        elif self.strides is auto_strides:
            strides = "auto"
        else:
            strides = ",".join(str(i) for i in self.strides)

        return "%s: %s, type: %s, shape: (%s), strides: (%s)" % (
                self.name, type(self).__name__, self.dtype, shape,
                strides)

    def __repr__(self):
        return "<%s>" % self.__str__()

class GlobalArg(ShapedArg):
    pass

class ConstantArg(ShapedArg):
    pass

class ArrayArg(GlobalArg):
    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("ArrayArg is a deprecated name of GlobalArg", DeprecationWarning,
                stacklevel=2)
        GlobalArg.__init__(self, *args, **kwargs)

class ImageArg(Record):
    def __init__(self, name, dtype=None, dimensions=None, shape=None):
        dtype = np.dtype(dtype)
        if shape is not None:
            if dimensions is not None and dimensions != len(shape):
                raise RuntimeError("cannot specify both shape and "
                        "disagreeing dimensions in ImageArg")
            dimensions = len(shape)
        else:
            if not isinstance(dimensions, int):
                raise RuntimeError("ImageArg: dimensions must be an integer")

        Record.__init__(self,
                dimensions=dimensions,
                shape=shape,
                dtype=dtype,
                name=name)

    def __str__(self):
        return "%s: ImageArg, type %s" % (self.name, self.dtype)

    def __repr__(self):
        return "<%s>" % self.__str__()

class ValueArg(Record):
    def __init__(self, name, dtype=None, approximately=1000):
        if dtype is not None:
            dtype = np.dtype(dtype)

        Record.__init__(self, name=name, dtype=dtype,
                approximately=approximately)

    def __str__(self):
        return "%s: ValueArg, type %s" % (self.name, self.dtype)

    def __repr__(self):
        return "<%s>" % self.__str__()

class ScalarArg(ValueArg):
    def __init__(self, name, dtype=None, approximately=1000):
        from warnings import warn
        warn("ScalarArg is a deprecated name of ValueArg",
                DeprecationWarning, stacklevel=2)

        ValueArg.__init__(self, name, dtype, approximately)

# }}}

# {{{ temporary variable

class TemporaryVariable(Record):
    """
    :ivar name:
    :ivar dtype:
    :ivar shape:
    :ivar storage_shape:
    :ivar base_indices:
    :ivar is_local:
    """

    def __init__(self, name, dtype, shape, is_local, base_indices=None,
            storage_shape=None):
        if base_indices is None:
            base_indices = (0,) * len(shape)

        if shape is not None and not isinstance(shape, tuple):
            shape = tuple(shape)

        Record.__init__(self, name=name, dtype=dtype, shape=shape, is_local=is_local,
                base_indices=base_indices,
                storage_shape=storage_shape)

    @property
    def nbytes(self):
        from pytools import product
        return product(si for si in self.shape)*self.dtype.itemsize

# }}}

# {{{ subsitution rule

class SubstitutionRule(Record):
    """
    :ivar name:
    :ivar arguments:
    :ivar expression:
    """

    def __init__(self, name, arguments, expression):
        assert isinstance(arguments, tuple)

        Record.__init__(self,
                name=name, arguments=arguments, expression=expression)

    def __str__(self):
        return "%s(%s) := %s" % (
                self.name, ", ".join(self.arguments), self.expression)

# }}}

# {{{ instruction

class Instruction(Record):
    """
    :ivar id: An (otherwise meaningless) identifier that is unique within
        a :class:`LoopKernel`.
    :ivar assignee:
    :ivar expression:
    :ivar forced_iname_deps: a set of inames that are added to the list of iname
        dependencies
    :ivar insn_deps: a list of ids of :class:`Instruction` instances that
        *must* be executed before this one. Note that loop scheduling augments this
        by adding dependencies on any writes to temporaries read by this instruction.
    :ivar boostable: Whether the instruction may safely be executed
        inside more loops than advertised without changing the meaning
        of the program. Allowed values are *None* (for unknown), *True*, and *False*.
    :ivar boostable_into: a set of inames into which the instruction
        may need to be boosted, as a heuristic help for the scheduler.
    :ivar priority: scheduling priority

    The following two instance variables are only used until :func:`loopy.make_kernel` is
    finished:

    :ivar temp_var_type: if not None, a type that will be assigned to the new temporary variable
        created from the assignee
    """
    def __init__(self,
            id, assignee, expression,
            forced_iname_deps=frozenset(), insn_deps=set(), boostable=None,
            boostable_into=None,
            temp_var_type=None, priority=0):

        from loopy.symbolic import parse
        if isinstance(assignee, str):
            assignee = parse(assignee)
        if isinstance(expression, str):
            assignee = parse(expression)

        assert isinstance(forced_iname_deps, frozenset)
        assert isinstance(insn_deps, set)

        Record.__init__(self,
                id=id, assignee=assignee, expression=expression,
                forced_iname_deps=forced_iname_deps,
                insn_deps=insn_deps, boostable=boostable,
                boostable_into=boostable_into,
                temp_var_type=temp_var_type,
                priority=priority)

    @memoize_method
    def reduction_inames(self):
        def map_reduction(expr, rec):
            rec(expr.expr)
            for iname in expr.inames:
                result.add(iname)

        from loopy.symbolic import ReductionCallbackMapper
        cb_mapper = ReductionCallbackMapper(map_reduction)

        result = set()
        cb_mapper(self.expression)

        return result

    def __str__(self):
        result = "%s: %s <- %s" % (self.id,
                self.assignee, self.expression)

        if self.boostable == True:
            if self.boostable_into:
                result += " (boostable into '%s')" % ",".join(self.boostable_into)
            else:
                result += " (boostable)"
        elif self.boostable == False:
            result += " (not boostable)"
        elif self.boostable is None:
            pass
        else:
            raise RuntimeError("unexpected value for Instruction.boostable")

        options = []

        if self.insn_deps:
            options.append("deps="+":".join(self.insn_deps))
        if self.priority:
            options.append("priority=%d" % self.priority)

        return result

    @memoize_method
    def get_assignee_var_name(self):
        from pymbolic.primitives import Variable, Subscript

        if isinstance(self.assignee, Variable):
            var_name = self.assignee.name
        elif isinstance(self.assignee, Subscript):
            agg = self.assignee.aggregate
            assert isinstance(agg, Variable)
            var_name = agg.name
        else:
            raise RuntimeError("invalid lvalue '%s'" % self.assignee)

        return var_name

    @memoize_method
    def get_assignee_indices(self):
        from pymbolic.primitives import Variable, Subscript

        if isinstance(self.assignee, Variable):
            return ()
        elif isinstance(self.assignee, Subscript):
            result = self.assignee.index
            if not isinstance(result, tuple):
                result = (result,)
            return result
        else:
            raise RuntimeError("invalid lvalue '%s'" % self.assignee)

    @memoize_method
    def get_read_var_names(self):
        from loopy.symbolic import get_dependencies
        return get_dependencies(self.expression)

# }}}

# {{{ function manglers / dtype getters

def default_function_mangler(name, arg_dtypes):
    from loopy.reduction import reduction_function_mangler

    manglers = [reduction_function_mangler]
    for mangler in manglers:
        result = mangler(name, arg_dtypes)
        if result is not None:
            return result

    return None

def opencl_function_mangler(name, arg_dtypes):
    if name == "atan2" and len(arg_dtypes) == 2:
        return arg_dtypes[0], name

    if len(arg_dtypes) == 1:
        arg_dtype, = arg_dtypes

        if arg_dtype.kind == "c":
            if arg_dtype == np.complex64:
                tpname = "cfloat"
            elif arg_dtype == np.complex128:
                tpname = "cdouble"
            else:
                raise RuntimeError("unexpected complex type '%s'" % arg_dtype)

            if name in ["sqrt", "exp", "log",
                    "sin", "cos", "tan",
                    "sinh", "cosh", "tanh"]:
                return arg_dtype, "%s_%s" % (tpname, name)

            if name in ["real", "imag"]:
                return np.dtype(arg_dtype.type(0).real), "%s_%s" % (tpname, name)

    if name == "dot":
        scalar_dtype, offset, field_name = arg_dtypes[0].fields["s0"]
        return scalar_dtype, name

    return None

def single_arg_function_mangler(name, arg_dtypes):
    if len(arg_dtypes) == 1:
        dtype, = arg_dtypes
        return dtype, name

    return None

def opencl_symbol_mangler(name):
    # FIXME: should be more picky about exact names
    if name.startswith("FLT_"):
        return np.dtype(np.float32), name
    elif name.startswith("DBL_"):
        return np.dtype(np.float64), name
    elif name.startswith("M_"):
        if name.endswith("_F"):
            return np.dtype(np.float32), name
        else:
            return np.dtype(np.float64), name
    else:
        return None

# }}}

# {{{ preamble generators

def default_preamble_generator(seen_dtypes, seen_functions):
    from loopy.reduction import reduction_preamble_generator

    for result in reduction_preamble_generator(seen_dtypes, seen_functions):
        yield result

    has_double = False
    has_complex = False

    for dtype in seen_dtypes:
        if dtype in [np.float64, np.complex128]:
            has_double = True
        if dtype.kind == "c":
            has_complex = True

    if has_double:
        yield ("00_enable_double", """
            #pragma OPENCL EXTENSION cl_khr_fp64: enable
            """)

    if has_complex:
        if has_double:
            yield ("10_include_complex_header", """
                #define PYOPENCL_DEFINE_CDOUBLE

                #include <pyopencl-complex.h>
                """)
        else:
            yield ("10_include_complex_header", """
                #include <pyopencl-complex.h>
                """)

    c_funcs = set(c_name for name, c_name, arg_dtypes in seen_functions)
    if "int_floor_div" in c_funcs:
        yield ("05_int_floor_div", """
            #define int_floor_div(a,b) \
              (( (a) - \
                 ( ( (a)<0 ) != ( (b)<0 )) \
                  *( (b) + ( (b)<0 ) - ( (b)>=0 ) )) \
               / (b) )
            """)

    if "int_floor_div_pos_b" in c_funcs:
        yield ("05_int_floor_div_pos_b", """
            #define int_floor_div_pos_b(a,b) ( \
                ( (a) - ( ((a)<0) ? ((b)-1) : 0 )  ) / (b) \
                )
            """)


# }}}

# vim: foldmethod=marker

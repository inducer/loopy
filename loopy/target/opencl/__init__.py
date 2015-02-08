"""OpenCL target independent of PyOpenCL."""

from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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

from loopy.target import TargetBase


# {{{ type registry

def _register_types():
    from loopy.target.opencl.compyte.dtypes import (
            _fill_dtype_registry, get_or_register_dtype)
    import struct

    _fill_dtype_registry(respect_windows=False, include_bool=False)

    # complex number support left out

    is_64_bit = struct.calcsize('@P') * 8 == 64
    if not is_64_bit:
        get_or_register_dtype(
                ["unsigned long", "unsigned long int"], np.uint64)
        get_or_register_dtype(
                ["signed long", "signed long int", "long int"], np.int64)

_register_types()

# }}}


# {{{ vector types

class vec:
    pass


def _create_vector_types():
    field_names = ["x", "y", "z", "w"]

    from loopy.target.opencl.compyte.dtypes import get_or_register_dtype

    vec.types = {}
    vec.type_to_scalar_and_count = {}

    counts = [2, 3, 4, 8, 16]

    for base_name, base_type in [
            ('char', np.int8),
            ('uchar', np.uint8),
            ('short', np.int16),
            ('ushort', np.uint16),
            ('int', np.int32),
            ('uint', np.uint32),
            ('long', np.int64),
            ('ulong', np.uint64),
            ('float', np.float32),
            ('double', np.float64),
            ]:
        for count in counts:
            name = "%s%d" % (base_name, count)

            titles = field_names[:count]

            padded_count = count
            if count == 3:
                padded_count = 4

            names = ["s%d" % i for i in range(count)]
            while len(names) < padded_count:
                names.append("padding%d" % (len(names)-count))

            if len(titles) < len(names):
                titles.extend((len(names)-len(titles))*[None])

            try:
                dtype = np.dtype(dict(
                    names=names,
                    formats=[base_type]*padded_count,
                    titles=titles))
            except NotImplementedError:
                try:
                    dtype = np.dtype([((n, title), base_type)
                                      for (n, title) in zip(names, titles)])
                except TypeError:
                    dtype = np.dtype([(n, base_type) for (n, title)
                                      in zip(names, titles)])

            get_or_register_dtype(name, dtype)

            setattr(vec, name, dtype)

            def create_array(dtype, count, padded_count, *args, **kwargs):
                if len(args) < count:
                    from warnings import warn
                    warn("default values for make_xxx are deprecated;"
                            " instead specify all parameters or use"
                            " array.vec.zeros_xxx", DeprecationWarning)
                padded_args = tuple(list(args)+[0]*(padded_count-len(args)))
                array = eval("array(padded_args, dtype=dtype)",
                        dict(array=np.array, padded_args=padded_args,
                        dtype=dtype))
                for key, val in kwargs.items():
                    array[key] = val
                return array

            setattr(vec, "make_"+name, staticmethod(eval(
                    "lambda *args, **kwargs: create_array(dtype, %i, %i, "
                    "*args, **kwargs)" % (count, padded_count),
                    dict(create_array=create_array, dtype=dtype))))
            setattr(vec, "filled_"+name, staticmethod(eval(
                    "lambda val: vec.make_%s(*[val]*%i)" % (name, count))))
            setattr(vec, "zeros_"+name,
                    staticmethod(eval("lambda: vec.filled_%s(0)" % (name))))
            setattr(vec, "ones_"+name,
                    staticmethod(eval("lambda: vec.filled_%s(1)" % (name))))

            vec.types[np.dtype(base_type), count] = dtype
            vec.type_to_scalar_and_count[dtype] = np.dtype(base_type), count

_create_vector_types()

# }}}


# {{{ function mangler

def opencl_function_mangler(target, name, arg_dtypes):
    if not isinstance(name, str):
        return None

    if name in ["max", "min"] and len(arg_dtypes) == 2:
        dtype = np.find_common_type([], arg_dtypes)

        if dtype.kind == "c":
            raise RuntimeError("min/max do not support complex numbers")

        if dtype.kind == "f":
            name = "f" + name

        return dtype, name

    if name in "atan2" and len(arg_dtypes) == 2:
        return arg_dtypes[0], name

    if name == "dot":
        scalar_dtype, offset, field_name = arg_dtypes[0].fields["s0"]
        return scalar_dtype, name

    return None

# }}}


# {{{ symbol mangler

def opencl_symbol_mangler(target, name):
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
    elif name == "INFINITY":
        return np.dtype(np.float32), name
    else:
        return None

# }}}


# {{{ preamble generator

def opencl_preamble_generator(target, seen_dtypes, seen_functions):
    has_double = False

    for dtype in seen_dtypes:
        if dtype in [np.float64, np.complex128]:
            has_double = True

    if has_double:
        yield ("00_enable_double", """
            #if __OPENCL_C_VERSION__ < 120
            #pragma OPENCL EXTENSION cl_khr_fp64: enable
            #endif
            """)

    c_funcs = set(func.c_name for func in seen_functions)
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


# {{{ target

class OpenCLTarget(TargetBase):
    def function_manglers(self):
        return (
                super(OpenCLTarget, self).function_manglers() + [
                    opencl_function_mangler
                    ])

    def symbol_manglers(self):
        return (
                super(OpenCLTarget, self).symbol_manglers() + [
                    opencl_symbol_mangler
                    ])

    def preamble_generators(self):
        from loopy.library.reduction import reduction_preamble_generator
        return (
                super(OpenCLTarget, self).preamble_generators() + [
                    opencl_preamble_generator,
                    reduction_preamble_generator
                    ])

    def get_or_register_dtype(self, names, dtype=None):
        from loopy.target.opencl.compyte.dtypes import get_or_register_dtype
        return get_or_register_dtype(names, dtype)

    def dtype_to_typename(self, dtype):
        from loopy.target.opencl.compyte.dtypes import dtype_to_ctype
        return dtype_to_ctype(dtype)

    def is_vector_dtype(self, dtype):
        return list(vec.types.values())

    def get_vector_dtype(self, base, count):
        return vec.types[base, count]

# }}}

# vim: foldmethod=marker

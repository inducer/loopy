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


def default_preamble_generator(seen_dtypes, seen_functions):
    from loopy.library.reduction import reduction_preamble_generator

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
            #if __OPENCL_C_VERSION__ < 120
            #pragma OPENCL EXTENSION cl_khr_fp64: enable
            #endif
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

# vim: foldmethod=marker

from __future__ import absolute_import, division, print_function

__copyright__ = "Copyright (C) 2017 Matt Wala"

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

from loopy.diagnostic import LoopyError


def tuple_function_mangler(kernel, name, arg_dtypes):
    if name == "make_tuple":
        from loopy.kernel.data import CallMangleInfo
        return CallMangleInfo(
                target_name=tuple_function_name(*arg_dtypes),
                result_dtypes=arg_dtypes,
                arg_dtypes=arg_dtypes)

    return None


def tuple_function_name(dtype0, dtype1):
    return "loopy_tuple_%s_%s" % (
            dtype0.numpy_dtype.type.__name__, dtype1.numpy_dtype.type.__name__)


def get_tuple_preamble(kernel, func_id, arg_dtypes):
    print("arg dtypes are", arg_dtypes)
    name = tuple_function_name(*arg_dtypes)
    return (name, """
    inline %(t0)s %(name)s(%(t0)s i0, %(t1)s i1, %(t1)s *o1)
    {
      *o1 = i1;
      return i0;
    }
    """ % dict(name=name,
            t0=kernel.target.dtype_to_typename(arg_dtypes[0]),
            t1=kernel.target.dtype_to_typename(arg_dtypes[1])))


def tuple_preamble_generator(preamble_info):
    from loopy.target.opencl import OpenCLTarget

    for func in preamble_info.seen_functions:
        if func.name == "make_tuple":
            if not isinstance(preamble_info.kernel.target, OpenCLTarget):
                raise LoopyError("only OpenCL supported for now")

            yield get_tuple_preamble(preamble_info.kernel, func.name,
                    func.arg_dtypes)

# vim: fdm=marker

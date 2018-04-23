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

from loopy.kernel.function_interface import ScalarCallable


def default_function_mangler(kernel, name, arg_dtypes):
    from loopy.library.reduction import reduction_function_mangler

    manglers = [reduction_function_mangler, tuple_function_mangler]
    for mangler in manglers:
        result = mangler(kernel, name, arg_dtypes)
        if result is not None:
            return result

    return None


def single_arg_function_mangler(kernel, name, arg_dtypes):
    if len(arg_dtypes) == 1:
        dtype, = arg_dtypes

        from loopy.kernel.data import CallMangleInfo
        return CallMangleInfo(name, (dtype,), (dtype,))

    return None


def tuple_function_mangler(kernel, name, arg_dtypes):
    if name == "make_tuple":
        from loopy.kernel.data import CallMangleInfo
        return CallMangleInfo(
                target_name="loopy_make_tuple",
                result_dtypes=arg_dtypes,
                arg_dtypes=arg_dtypes)

    return None


class MakeTupleCallable(ScalarCallable):
    def with_types(self, arg_id_to_dtype, kernel):
        new_arg_id_to_dtype = arg_id_to_dtype.copy()
        for i in range(len(arg_id_to_dtype)):
            if i in arg_id_to_dtype and arg_id_to_dtype[i] is not None:
                new_arg_id_to_dtype[-i-1] = new_arg_id_to_dtype[i]

        return self.copy(arg_id_to_dtype=new_arg_id_to_dtype,
            name_in_target="loopy_make_tuple")

    def with_descrs(self, arg_id_to_descr):
        from loopy.kernel.function_interface import ValueArgDescriptor
        new_arg_id_to_descr = dict(((id, ValueArgDescriptor()),
            (-id-1, ValueArgDescriptor())) for id in arg_id_to_descr.keys())

        return self.copy(arg_id_to_descr=new_arg_id_to_descr)


class IndexOfCallable(ScalarCallable):
    def with_types(self, arg_id_to_dtype, kernel):
        new_arg_id_to_dtype = arg_id_to_dtype.copy()
        new_arg_id_to_dtype[-1] = kernel.index_dtype

        return self.copy(arg_id_to_dtype=new_arg_id_to_dtype)


def loopy_specific_callable_scopers(target, identifier):
    if identifier == "make_tuple":
        return MakeTupleCallable(name="make_tuple")

    if identifier in ["indexof", "indexof_vec"]:
        return IndexOfCallable(name=identifier)

    from loopy.library.reduction import reduction_scoper
    return reduction_scoper(target, identifier)


# vim: foldmethod=marker

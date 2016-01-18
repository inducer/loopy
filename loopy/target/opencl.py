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

from loopy.target.c import CTarget
from loopy.target.c.codegen.expression import LoopyCCodeMapper
from pytools import memoize_method
from loopy.diagnostic import LoopyError


# {{{ vector types

class vec:  # noqa
    pass


def _create_vector_types():
    field_names = ["x", "y", "z", "w"]

    vec.types = {}
    vec.names_and_dtypes = []
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

            setattr(vec, name, dtype)

            vec.names_and_dtypes.append((name, dtype))

            vec.types[np.dtype(base_type), count] = dtype
            vec.type_to_scalar_and_count[dtype] = np.dtype(base_type), count

_create_vector_types()


def _register_vector_types(dtype_registry):
    for name, dtype in vec.names_and_dtypes:
        dtype_registry.get_or_register_dtype(name, dtype)

# }}}


# {{{ function mangler

def opencl_function_mangler(kernel, name, arg_dtypes):
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

def opencl_symbol_mangler(kernel, name):
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

def opencl_preamble_generator(kernel, seen_dtypes, seen_functions):
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

# }}}


# {{{ expression mapper

class LoopyOpenCLCCodeMapper(LoopyCCodeMapper):
    def map_group_hw_index(self, expr, enclosing_prec, type_context):
        return "gid(%d)" % expr.axis

    def map_local_hw_index(self, expr, enclosing_prec, type_context):
        return "lid(%d)" % expr.axis

# }}}


# {{{ target

class OpenCLTarget(CTarget):
    # {{{ library

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

    # }}}

    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c.compyte.dtypes import (DTypeRegistry,
                fill_registry_with_opencl_c_types)

        result = DTypeRegistry()
        fill_registry_with_opencl_c_types(result)

        # no complex number support--needs PyOpenCLTarget

        _register_vector_types(result)

        return result

    def is_vector_dtype(self, dtype):
        return list(vec.types.values())

    def vector_dtype(self, base, count):
        return vec.types[base, count]

    # }}}

    # {{{ top-level codegen

    def wrap_function_declaration(self, kernel, fdecl):
        from cgen.opencl import CLKernel, CLRequiredWorkGroupSize
        fdecl = CLKernel(fdecl)

        _, local_sizes = kernel.get_grid_sizes_as_exprs()

        from loopy.symbolic import get_dependencies
        if not get_dependencies(local_sizes):
            # sizes can't have parameter dependencies if they are
            # to be used in static WG size.

            fdecl = CLRequiredWorkGroupSize(local_sizes, fdecl)

        return fdecl

    def generate_code(self, kernel, codegen_state, impl_arg_info):
        code, implemented_domains = (
                super(OpenCLTarget, self).generate_code(
                    kernel, codegen_state, impl_arg_info))

        from loopy.tools import remove_common_indentation
        code = (
                remove_common_indentation("""
                    #define lid(N) ((%(idx_ctype)s) get_local_id(N))
                    #define gid(N) ((%(idx_ctype)s) get_group_id(N))
                    """ % dict(idx_ctype=self.dtype_to_typename(kernel.index_dtype)))
                + "\n\n"
                + code)

        return code, implemented_domains

    def generate_body(self, kernel, codegen_state):
        body, implemented_domains = (
                super(OpenCLTarget, self).generate_body(kernel, codegen_state))

        from loopy.kernel.data import ImageArg

        if any(isinstance(arg, ImageArg) for arg in kernel.args):
            from cgen import Value, Const, Initializer
            body.contents.insert(0,
                    Initializer(Const(Value("sampler_t", "loopy_sampler")),
                        "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP "
                        "| CLK_FILTER_NEAREST"))

        return body, implemented_domains

    # }}}

    # {{{ code generation guts

    def get_expression_to_code_mapper(self, codegen_state):
        return LoopyOpenCLCCodeMapper(codegen_state)

    def add_vector_access(self, access_str, index):
        # The 'int' avoids an 'L' suffix for long ints.
        return "(%s).s%s" % (access_str, hex(int(index))[2:])

    def emit_barrier(self, kind, comment):
        """
        :arg kind: ``"local"`` or ``"global"``
        :return: a :class:`loopy.codegen.GeneratedInstruction`.
        """
        if kind == "local":
            if comment:
                comment = " /* %s */" % comment

            from loopy.codegen import GeneratedInstruction
            from cgen import Statement
            return GeneratedInstruction(
                    ast=Statement("barrier(CLK_LOCAL_MEM_FENCE)%s" % comment),
                    implemented_domain=None)
        elif kind == "global":
            raise LoopyError("OpenCL does not have global barriers")
        else:
            raise LoopyError("unknown barrier kind")

    def wrap_temporary_decl(self, decl, is_local):
        if is_local:
            from cgen.opencl import CLLocal
            return CLLocal(decl)
        else:
            return decl

    def get_global_arg_decl(self, name, shape, dtype, is_written):
        from cgen.opencl import CLGlobal

        return CLGlobal(super(OpenCLTarget, self).get_global_arg_decl(
            name, shape, dtype, is_written))

    def get_image_arg_decl(self, name, shape, num_target_axes, dtype, is_written):
        if is_written:
            mode = "w"
        else:
            mode = "r"

        from cgen.opencl import CLImage
        return CLImage(num_target_axes, mode, name)

    def get_constant_arg_decl(self, name, shape, dtype, is_written):
        from loopy.codegen import POD  # uses the correct complex type
        from cgen import RestrictPointer, Const
        from cgen.opencl import CLConstant

        arg_decl = RestrictPointer(POD(dtype, name))

        if not is_written:
            arg_decl = Const(arg_decl)

        return CLConstant(arg_decl)

    # }}}

# }}}

# vim: foldmethod=marker

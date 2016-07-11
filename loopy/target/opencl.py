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

from loopy.target.c import CTarget, CASTBuilder
from loopy.target.c.codegen.expression import ExpressionToCMapper
from pytools import memoize_method
from loopy.diagnostic import LoopyError
from loopy.types import NumpyType
from loopy.target.c import DTypeRegistryWrapper
from loopy.kernel.data import temp_var_scope, CallMangleInfo


# {{{ dtype registry wrappers

class DTypeRegistryWrapperWithAtomics(DTypeRegistryWrapper):
    def get_or_register_dtype(self, names, dtype=None):
        if dtype is not None:
            from loopy.types import AtomicNumpyType, NumpyType
            if isinstance(dtype, AtomicNumpyType):
                return super(self.wrapped_registry.get_or_register_dtype(
                        names, NumpyType(dtype.dtype)))

        return super(DTypeRegistryWrapperWithAtomics, self).get_or_register_dtype(
                names, dtype)


class DTypeRegistryWrapperWithCL1Atomics(DTypeRegistryWrapperWithAtomics):
    def dtype_to_ctype(self, dtype):
        from loopy.types import AtomicNumpyType

        if isinstance(dtype, AtomicNumpyType):
            return "volatile " + self.wrapped_registry.dtype_to_ctype(dtype)
        else:
            return super(DTypeRegistryWrapperWithCL1Atomics, self).dtype_to_ctype(
                    dtype)

# }}}


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

_CL_SIMPLE_MULTI_ARG_FUNCTIONS = {
        "clamp": 3,
        "atan2": 2,
        }


VECTOR_LITERAL_FUNCS = dict(
        ("make_%s%d" % (name, count), (name, dtype, count))
        for name, dtype in [
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
            ]
        for count in [2, 3, 4, 8, 16]
        )


def opencl_function_mangler(kernel, name, arg_dtypes):
    if not isinstance(name, str):
        return None

    if (name == "abs"
            and len(arg_dtypes) == 1
            and arg_dtypes[0].numpy_dtype.kind == "f"):
        return CallMangleInfo(
                target_name="fabs",
                result_dtypes=arg_dtypes,
                arg_dtypes=arg_dtypes)

    if name in ["max", "min"] and len(arg_dtypes) == 2:
        dtype = np.find_common_type(
                [], [dtype.numpy_dtype for dtype in arg_dtypes])

        if dtype.kind == "c":
            raise RuntimeError("min/max do not support complex numbers")

        if dtype.kind == "f":
            name = "f" + name

        result_dtype = NumpyType(dtype)
        return CallMangleInfo(
                target_name=name,
                result_dtypes=(result_dtype,),
                arg_dtypes=2*(result_dtype,))

    if name == "dot":
        scalar_dtype, offset, field_name = arg_dtypes[0].numpy_dtype.fields["s0"]
        return CallMangleInfo(
                target_name=name,
                result_dtypes=(NumpyType(scalar_dtype),),
                arg_dtypes=(arg_dtypes[0],)*2)

    if name in _CL_SIMPLE_MULTI_ARG_FUNCTIONS:
        num_args = _CL_SIMPLE_MULTI_ARG_FUNCTIONS[name]
        if len(arg_dtypes) != num_args:
            raise LoopyError("%s takes %d arguments (%d received)"
                    % (name, num_args, len(arg_dtypes)))

        dtype = np.find_common_type(
                [], [dtype.numpy_dtype for dtype in arg_dtypes])

        if dtype.kind == "c":
            raise LoopyError("%s does not support complex numbers"
                    % name)

        result_dtype = NumpyType(dtype)
        return CallMangleInfo(
                target_name=name,
                result_dtypes=(result_dtype,),
                arg_dtypes=(result_dtype,)*num_args)

    if name in VECTOR_LITERAL_FUNCS:
        base_tp_name, dtype, count = VECTOR_LITERAL_FUNCS[name]

        if count != len(arg_dtypes):
            return None

        return CallMangleInfo(
                target_name="(%s%d) " % (base_tp_name, count),
                result_dtypes=(kernel.target.vector_dtype(
                    NumpyType(dtype), count),),
                arg_dtypes=(NumpyType(dtype),)*count)

    return None

# }}}


# {{{ symbol mangler

def opencl_symbol_mangler(kernel, name):
    # FIXME: should be more picky about exact names
    if name.startswith("FLT_"):
        return NumpyType(np.dtype(np.float32)), name
    elif name.startswith("DBL_"):
        return NumpyType(np.dtype(np.float64)), name
    elif name.startswith("M_"):
        if name.endswith("_F"):
            return NumpyType(np.dtype(np.float32)), name
        else:
            return NumpyType(np.dtype(np.float64)), name
    elif name == "INFINITY":
        return NumpyType(np.dtype(np.float32)), name
    else:
        return None

# }}}


# {{{ preamble generator

def opencl_preamble_generator(preamble_info):
    has_double = False

    for dtype in preamble_info.seen_dtypes:
        if (isinstance(dtype, NumpyType)
                and dtype.numpy_dtype in [np.float64, np.complex128]):
            has_double = True

    if has_double:
        yield ("00_enable_double", """
            #if __OPENCL_C_VERSION__ < 120
            #pragma OPENCL EXTENSION cl_khr_fp64: enable
            #endif
            """)

    from loopy.types import AtomicNumpyType
    seen_64_bit_atomics = any(
            isinstance(dtype, AtomicNumpyType) and dtype.numpy_dtype.itemsize == 8
            for dtype in preamble_info.seen_atomic_dtypes)

    if seen_64_bit_atomics:
        # FIXME: Should gate on "CL1" atomics style
        yield ("00_enable_64bit_atomics", """
            #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
            """)

    from loopy.tools import remove_common_indentation
    kernel = preamble_info.kernel
    yield ("00_declare_gid_lid",
            remove_common_indentation("""
                #define lid(N) ((%(idx_ctype)s) get_local_id(N))
                #define gid(N) ((%(idx_ctype)s) get_group_id(N))
                """ % dict(idx_ctype=kernel.target.dtype_to_typename(
                    kernel.index_dtype))))

# }}}


# {{{ expression mapper

class ExpressionToOpenCLCMapper(ExpressionToCMapper):
    def map_group_hw_index(self, expr, enclosing_prec, type_context):
        return "gid(%d)" % expr.axis

    def map_local_hw_index(self, expr, enclosing_prec, type_context):
        return "lid(%d)" % expr.axis

# }}}


# {{{ target

class OpenCLTarget(CTarget):
    """A target for the OpenCL C heterogeneous compute programming language.
    """

    def __init__(self, atomics_flavor="cl1"):
        """
        :arg atomics_flavor: one of ``"cl1"`` (C11-style atomics from OpenCL 2.0),
            ``"cl1"`` (OpenCL 1.1 atomics, using bit-for-bit compare-and-swap
            for floating point), ``"cl1-exch"`` (OpenCL 1.1 atomics, using
            double-exchange for floating point--not yet supported).
        """
        super(OpenCLTarget, self).__init__()

        if atomics_flavor not in ["cl1", "cl2"]:
            raise ValueError("unsupported atomics flavor: %s" % atomics_flavor)

        self.atomics_flavor = atomics_flavor

    def split_kernel_at_global_barriers(self):
        return True

    def get_device_ast_builder(self):
        return OpenCLCASTBuilder(self)

    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c.compyte.dtypes import (DTypeRegistry,
                fill_registry_with_opencl_c_types)

        result = DTypeRegistry()
        fill_registry_with_opencl_c_types(result)

        # no complex number support--needs PyOpenCLTarget

        _register_vector_types(result)

        if self.atomics_flavor == "cl1":
            return DTypeRegistryWrapperWithCL1Atomics(result)
        else:
            raise NotImplementedError("atomics flavor: %s" % self.atomics_flavor)

    def is_vector_dtype(self, dtype):
        return (isinstance(dtype, NumpyType)
                and dtype.numpy_dtype in list(vec.types.values()))

    def vector_dtype(self, base, count):
        return NumpyType(
                vec.types[base.numpy_dtype, count],
                target=self)

    # }}}

# }}}


# {{{ ast builder

class OpenCLCASTBuilder(CASTBuilder):
    # {{{ library

    def function_manglers(self):
        return (
                super(OpenCLCASTBuilder, self).function_manglers() + [
                    opencl_function_mangler
                    ])

    def symbol_manglers(self):
        return (
                super(OpenCLCASTBuilder, self).symbol_manglers() + [
                    opencl_symbol_mangler
                    ])

    def preamble_generators(self):
        from loopy.library.reduction import reduction_preamble_generator
        return (
                super(OpenCLCASTBuilder, self).preamble_generators() + [
                    opencl_preamble_generator,
                    reduction_preamble_generator
                    ])

    # }}}

    # {{{ top-level codegen

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        fdecl = super(OpenCLCASTBuilder, self).get_function_declaration(
                codegen_state, codegen_result, schedule_index)

        from cgen.opencl import CLKernel, CLRequiredWorkGroupSize
        fdecl = CLKernel(fdecl)

        from loopy.schedule import get_insn_ids_for_block_at
        _, local_sizes = codegen_state.kernel.get_grid_sizes_for_insn_ids_as_exprs(
                get_insn_ids_for_block_at(
                    codegen_state.kernel.schedule, schedule_index))

        from loopy.symbolic import get_dependencies
        if not get_dependencies(local_sizes):
            # sizes can't have parameter dependencies if they are
            # to be used in static WG size.

            fdecl = CLRequiredWorkGroupSize(local_sizes, fdecl)

        return fdecl

    def generate_top_of_body(self, codegen_state):
        from loopy.kernel.data import ImageArg
        if any(isinstance(arg, ImageArg) for arg in codegen_state.kernel.args):
            from cgen import Value, Const, Initializer
            return [
                    Initializer(Const(Value("sampler_t", "loopy_sampler")),
                        "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP "
                        "| CLK_FILTER_NEAREST")
                    ]

        return []

    # }}}

    # {{{ code generation guts

    def get_expression_to_code_mapper(self, codegen_state):
        return ExpressionToOpenCLCMapper(codegen_state)

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

            from cgen import Statement
            return Statement("barrier(CLK_LOCAL_MEM_FENCE)%s" % comment)
        elif kind == "global":
            raise LoopyError("OpenCL does not have global barriers")
        else:
            raise LoopyError("unknown barrier kind")

    def wrap_temporary_decl(self, decl, scope):
        if scope == temp_var_scope.LOCAL:
            from cgen.opencl import CLLocal
            return CLLocal(decl)
        elif scope == temp_var_scope.PRIVATE:
            return decl
        else:
            raise ValueError("unexpected temporary variable scope: %s"
                    % scope)

    def wrap_global_constant(self, decl):
        from cgen.opencl import CLConstant
        return CLConstant(decl)

    def get_global_arg_decl(self, name, shape, dtype, is_written):
        from cgen.opencl import CLGlobal

        return CLGlobal(super(OpenCLCASTBuilder, self).get_global_arg_decl(
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

    # {{{ code generation for atomic update

    def generate_atomic_update(self, kernel, codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype, rhs_type_context):
        from pymbolic.mapper.stringifier import PREC_NONE

        # FIXME: Could detect operations, generate atomic_{add,...} when
        # appropriate.

        if isinstance(lhs_dtype, NumpyType) and lhs_dtype.numpy_dtype in [
                np.int32, np.int64, np.float32, np.float64]:
            from cgen import Block, DoWhile, Assign
            from loopy.target.c import POD
            old_val_var = codegen_state.var_name_generator("loopy_old_val")
            new_val_var = codegen_state.var_name_generator("loopy_new_val")

            from loopy.kernel.data import TemporaryVariable, temp_var_scope
            ecm = codegen_state.expression_to_code_mapper.with_assignments(
                    {
                        old_val_var: TemporaryVariable(old_val_var, lhs_dtype),
                        new_val_var: TemporaryVariable(new_val_var, lhs_dtype),
                        })

            lhs_expr_code = ecm(lhs_expr, prec=PREC_NONE, type_context=None)

            from pymbolic.mapper.substitutor import make_subst_func
            from pymbolic import var
            from loopy.symbolic import SubstitutionMapper

            subst = SubstitutionMapper(
                    make_subst_func({lhs_expr: var(old_val_var)}))
            rhs_expr_code = ecm(subst(rhs_expr), prec=PREC_NONE,
                    type_context=rhs_type_context,
                    needed_dtype=lhs_dtype)

            if lhs_dtype.numpy_dtype.itemsize == 4:
                func_name = "atomic_cmpxchg"
            elif lhs_dtype.numpy_dtype.itemsize == 8:
                func_name = "atom_cmpxchg"
            else:
                raise LoopyError("unexpected atomic size")

            cast_str = ""
            old_val = old_val_var
            new_val = new_val_var

            if lhs_dtype.numpy_dtype.kind == "f":
                if lhs_dtype.numpy_dtype == np.float32:
                    ctype = "int"
                elif lhs_dtype.numpy_dtype == np.float64:
                    ctype = "long"
                else:
                    assert False

                from loopy.kernel.data import TemporaryVariable, GlobalArg
                if isinstance(lhs_var, GlobalArg):
                    var_kind = "__global"
                elif (
                        isinstance(lhs_var, TemporaryVariable)
                        and lhs_var.scope == temp_var_scope.LOCAL):
                    var_kind = "__local"
                elif (
                        isinstance(lhs_var, TemporaryVariable)
                        and lhs_var.scope == temp_var_scope.GLOBAL):
                    var_kind = "__global"
                else:
                    raise LoopyError("unexpected kind of variable '%s' in "
                            "atomic operation: "
                            % (lhs_var.name, type(lhs_var).__name__))

                old_val = "*(%s *) &" % ctype + old_val
                new_val = "*(%s *) &" % ctype + new_val
                cast_str = "(%s %s *) " % (var_kind, ctype)

            return Block([
                POD(self, NumpyType(lhs_dtype.dtype, target=self.target),
                    old_val_var),
                POD(self, NumpyType(lhs_dtype.dtype, target=self.target),
                    new_val_var),
                DoWhile(
                    "%(func_name)s("
                    "%(cast_str)s&(%(lhs_expr)s), "
                    "%(old_val)s, "
                    "%(new_val)s"
                    ") != %(old_val)s"
                    % {
                        "func_name": func_name,
                        "cast_str": cast_str,
                        "lhs_expr": lhs_expr_code,
                        "old_val": old_val,
                        "new_val": new_val,
                        },
                    Block([
                        Assign(old_val_var, lhs_expr_code),
                        Assign(new_val_var, rhs_expr_code),
                        ])
                    )
                ])
        else:
            raise NotImplementedError("atomic update for '%s'" % lhs_dtype)

    # }}}

    # }}}

# }}}

# vim: foldmethod=marker

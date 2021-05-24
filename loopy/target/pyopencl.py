"""OpenCL target integrated with PyOpenCL."""

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
import pymbolic.primitives as p

from loopy.target.opencl import (OpenCLTarget, OpenCLCASTBuilder,
        ExpressionToOpenCLCExpressionMapper)
from loopy.target.python import PythonASTBuilderBase
from loopy.types import NumpyType
from loopy.diagnostic import LoopyError, LoopyTypeError
from warnings import warn
from loopy.kernel.function_interface import ScalarCallable

import logging
logger = logging.getLogger(__name__)


# {{{ pyopencl function scopers

class PyOpenCLCallable(ScalarCallable):
    """
    Records information about the callables which are not covered by
    :class:`loopy.target.opencl.OpenCLCallable`
    """
    def with_types(self, arg_id_to_dtype, callables_table):

        name = self.name

        for id in arg_id_to_dtype:
            # since all the below functions are single arg.
            if not -1 <= id <= 0:
                raise LoopyError(f"{name} can only take one argument")

        if 0 not in arg_id_to_dtype or arg_id_to_dtype[0] is None:
            # the types provided aren't mature enough to specialize the
            # callable
            return (
                    self.copy(arg_id_to_dtype=arg_id_to_dtype),
                    callables_table)

        dtype = arg_id_to_dtype[0]

        if name in ["real", "imag", "abs"]:
            if dtype.is_complex():
                if dtype.numpy_dtype == np.complex64:
                    tpname = "cfloat"
                elif dtype.numpy_dtype == np.complex128:
                    tpname = "cdouble"
                else:
                    raise LoopyTypeError(f"unexpected complex type '{dtype}'")

                return (
                        self.copy(name_in_target=f"{tpname}_{name}",
                            arg_id_to_dtype={0: dtype, -1: NumpyType(
                                np.dtype(dtype.numpy_dtype.type(0).real))}),
                        callables_table)

        if name in ["real", "imag"]:
            if not dtype.is_complex():
                tpname = dtype.numpy_dtype.type.__name__
                return (
                        self.copy(
                            name_in_target=f"lpy_{name}_{tpname}",
                            arg_id_to_dtype={0: dtype, -1: dtype}),
                        callables_table)

        if name in ["sqrt", "exp", "log",
                "sin", "cos", "tan",
                "sinh", "cosh", "tanh",
                "conj", "abs"]:
            if dtype.is_complex():
                # function parameters are complex.
                if dtype.numpy_dtype == np.complex64:
                    tpname = "cfloat"
                elif dtype.numpy_dtype == np.complex128:
                    tpname = "cdouble"
                else:
                    raise LoopyTypeError("unexpected complex type '%s'" % dtype)

                return (
                        self.copy(name_in_target=f"{tpname}_{name}",
                            arg_id_to_dtype={0: dtype, -1: dtype}),
                        callables_table)
            else:
                # function calls for floating-point parameters.
                numpy_dtype = dtype.numpy_dtype
                if numpy_dtype.kind in ("u", "i"):
                    dtype = NumpyType(np.float32)
                if name == "abs":
                    name = "fabs"
                return (
                        self.copy(name_in_target=name,
                            arg_id_to_dtype={0: dtype, -1: dtype}),
                        callables_table)

        return (
                self.copy(arg_id_to_dtype=arg_id_to_dtype),
                callables_table)

    def generate_preambles(self, target):
        name = self.name_in_target
        if name.startswith("lpy_real") or name.startswith("lpy_imag"):
            if name.startswith("lpy_real"):
                ret = "x"
            else:
                ret = "0"

            dtype = self.arg_id_to_dtype[-1]
            ctype = target.dtype_to_typename(dtype)

            yield(f"40_{name}", f"""
            static inline {ctype} {name}({ctype} x) {{
                return {ret};
            }}
            """)


def get_pyopencl_callables():
    pyopencl_ids = ["sqrt", "exp", "log", "sin", "cos", "tan", "sinh", "cosh",
            "tanh", "conj", "real", "imag", "abs"]
    return {id_: PyOpenCLCallable(name=id_) for id_ in pyopencl_ids}

# }}}


# {{{ preamble generator

def pyopencl_preamble_generator(preamble_info):
    has_double = False
    has_complex = False

    from loopy.types import NumpyType
    for dtype in preamble_info.seen_dtypes:
        if (isinstance(dtype, NumpyType)
                and dtype.dtype in [np.float64, np.complex128]):
            has_double = True
        if dtype.involves_complex():
            has_complex = True

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

# }}}


# {{{ expression mapper

class ExpressionToPyOpenCLCExpressionMapper(ExpressionToOpenCLCExpressionMapper):
    def complex_type_name(self, dtype):
        from loopy.types import NumpyType
        if not isinstance(dtype, NumpyType):
            raise LoopyError("'%s' is not a complex type" % dtype)

        if dtype.dtype == np.complex64:
            return "cfloat"
        if dtype.dtype == np.complex128:
            return "cdouble"
        else:
            raise RuntimeError

    @property
    def allow_complex(self):
        from loopy.kernel.tools import has_complex_dtyped_var
        return has_complex_dtyped_var(self.kernel)

    def wrap_in_typecast_lazy(self, actual_type_func, needed_dtype, s):
        if needed_dtype.is_complex():
            return self.wrap_in_typecast(actual_type_func(), needed_dtype, s)
        else:
            return super().wrap_in_typecast_lazy(actual_type_func,
                                                 needed_dtype, s)

    def wrap_in_typecast(self, actual_type, needed_dtype, s):
        if (actual_type.is_complex() and needed_dtype.is_complex()
                and actual_type != needed_dtype):
            return p.Variable("%s_cast" % self.complex_type_name(needed_dtype))(s)
        elif not actual_type.is_complex() and needed_dtype.is_complex():
            return p.Variable("%s_fromreal" % self.complex_type_name(needed_dtype))(
                    s)
        else:
            return super().wrap_in_typecast_lazy(actual_type,
                                                 needed_dtype, s)

    def map_sum(self, expr, type_context):
        # I've added 'type_context == "i"' because of the following
        # idiotic corner case: Code generation for subscripts comes
        # through here, and it may involve variables that we know
        # nothing about (offsets and such). If we fall into the allow_complex
        # branch, we'll try to do type inference on these variables,
        # and stuff breaks. This band-aid works around that. -AK
        if not self.allow_complex or type_context == "i":
            return super().map_sum(expr, type_context)

        tgt_dtype = self.infer_type(expr)
        is_complex = tgt_dtype.is_complex()

        if not is_complex:
            return super().map_sum(expr, type_context)
        else:
            tgt_name = self.complex_type_name(tgt_dtype)

            reals = []
            complexes = []
            for child in expr.children:
                if self.infer_type(child).is_complex():
                    complexes.append(child)
                else:
                    reals.append(child)

            real_sum = p.flattened_sum([self.rec(r, type_context) for r in reals])

            c_applied = [self.rec(c, type_context, tgt_dtype) for c in complexes]

            def binary_tree_add(start, end):
                if start + 1 == end:
                    return c_applied[start]
                mid = (start + end)//2
                lsum = binary_tree_add(start, mid)
                rsum = binary_tree_add(mid, end)
                return p.Variable("%s_add" % tgt_name)(lsum, rsum)

            complex_sum = binary_tree_add(0, len(c_applied))

            if reals:
                return p.Variable("%s_radd" % tgt_name)(real_sum, complex_sum)
            else:
                return complex_sum

    def map_product(self, expr, type_context):
        # I've added 'type_context == "i"' because of the following
        # idiotic corner case: Code generation for subscripts comes
        # through here, and it may involve variables that we know
        # nothing about (offsets and such). If we fall into the allow_complex
        # branch, we'll try to do type inference on these variables,
        # and stuff breaks. This band-aid works around that. -AK
        if not self.allow_complex or type_context == "i":
            return super().map_product(expr, type_context)

        tgt_dtype = self.infer_type(expr)
        is_complex = tgt_dtype.is_complex()

        if not is_complex:
            return super().map_product(expr, type_context)
        else:
            tgt_name = self.complex_type_name(tgt_dtype)

            reals = []
            complexes = []
            for child in expr.children:
                if self.infer_type(child).is_complex():
                    complexes.append(child)
                else:
                    reals.append(child)

            real_prd = p.flattened_product(
                    [self.rec(r, type_context) for r in reals])

            c_applied = [self.rec(c, type_context, tgt_dtype) for c in complexes]

            def binary_tree_mul(start, end):
                if start + 1 == end:
                    return c_applied[start]
                mid = (start + end)//2
                lsum = binary_tree_mul(start, mid)
                rsum = binary_tree_mul(mid, end)
                return p.Variable("%s_mul" % tgt_name)(lsum, rsum)

            complex_prd = binary_tree_mul(0, len(complexes))

            if reals:
                return p.Variable("%s_rmul" % tgt_name)(real_prd, complex_prd)
            else:
                return complex_prd

    def map_quotient(self, expr, type_context):
        n_dtype = self.infer_type(expr.numerator).numpy_dtype
        d_dtype = self.infer_type(expr.denominator).numpy_dtype
        tgt_dtype = self.infer_type(expr)
        n_complex = "c" == n_dtype.kind
        d_complex = "c" == d_dtype.kind

        if not self.allow_complex or (not (n_complex or d_complex)):
            return super().map_quotient(expr, type_context)

        if n_complex and not d_complex:
            return p.Variable("%s_divider" % self.complex_type_name(tgt_dtype))(
                    self.rec(expr.numerator, type_context, tgt_dtype),
                    self.rec(expr.denominator, type_context))
        elif not n_complex and d_complex:
            return p.Variable("%s_rdivide" % self.complex_type_name(tgt_dtype))(
                    self.rec(expr.numerator, type_context),
                    self.rec(expr.denominator, type_context, tgt_dtype))
        else:
            return p.Variable("%s_divide" % self.complex_type_name(tgt_dtype))(
                    self.rec(expr.numerator, type_context, tgt_dtype),
                    self.rec(expr.denominator, type_context, tgt_dtype))

    def map_constant(self, expr, type_context):
        if isinstance(expr, (complex, np.complexfloating)):
            try:
                dtype = expr.dtype
            except AttributeError:
                # (COMPLEX_GUESS_LOGIC) This made it through type 'guessing' in
                # type inference, and it was concluded there (search for
                # COMPLEX_GUESS_LOGIC in loopy.type_inference), that no
                # accuracy was lost by using single precision.
                cast_type = "cfloat"
            else:
                if dtype == np.complex128:
                    cast_type = "cdouble"
                elif dtype == np.complex64:
                    cast_type = "cfloat"
                else:
                    raise RuntimeError("unsupported complex type in expression "
                            "generation: %s" % type(expr))

            return p.Variable("%s_new" % cast_type)(expr.real, expr.imag)

        return super().map_constant(expr, type_context)

    def map_power(self, expr, type_context):
        tgt_dtype = self.infer_type(expr)
        base_dtype = self.infer_type(expr.base)
        exponent_dtype = self.infer_type(expr.exponent)

        if not self.allow_complex or (not tgt_dtype.is_complex()):
            return super().map_power(expr, type_context)

        if expr.exponent in [2, 3, 4]:
            value = expr.base
            for _i in range(expr.exponent-1):
                value = value * expr.base
            return self.rec(value, type_context)
        else:
            b_complex = base_dtype.is_complex()
            e_complex = exponent_dtype.is_complex()

            if b_complex and not e_complex:
                return p.Variable("%s_powr" % self.complex_type_name(tgt_dtype))(
                        self.rec(expr.base, type_context, tgt_dtype),
                        self.rec(expr.exponent, type_context))
            else:
                return p.Variable("%s_pow" % self.complex_type_name(tgt_dtype))(
                        self.rec(expr.base, type_context, tgt_dtype),
                        self.rec(expr.exponent, type_context, tgt_dtype))

# }}}


# {{{ target

class PyOpenCLTarget(OpenCLTarget):
    """A code generation target that takes special advantage of :mod:`pyopencl`
    features such as run-time knowledge of the target device (to generate
    warnings) and support for complex numbers.
    """

    # FIXME make prefixes conform to naming rules
    # (see Reference: Loopy’s Model of a Kernel)

    host_program_name_prefix = "_lpy_host_"
    host_program_name_suffix = ""

    def __init__(self, device=None, pyopencl_module_name="_lpy_cl",
                 atomics_flavor=None, use_int8_for_bool=True):
        # This ensures the dtype registry is populated.
        import pyopencl.tools  # noqa

        super().__init__(
            atomics_flavor=atomics_flavor,
            use_int8_for_bool=use_int8_for_bool)

        import pyopencl.version
        if pyopencl.version.VERSION < (2021, 2):
            raise RuntimeError("The version of loopy you have installed "
                    "generates invoker code that requires PyOpenCL 2021.2 "
                    "or newer.")

        if device is not None:
            warn("Passing device is deprecated, it will stop working in 2022.",
                    DeprecationWarning, stacklevel=2)

        self.pyopencl_module_name = pyopencl_module_name

    @property
    def device(self):
        warn("PyOpenCLTarget.device is deprecated, it will stop working in 2022.",
                DeprecationWarning, stacklevel=2)
        return None

    # NB: Not including 'device', as that is handled specially here.
    hash_fields = OpenCLTarget.hash_fields + (
            "pyopencl_module_name",)
    comparison_fields = OpenCLTarget.comparison_fields + (
            "pyopencl_module_name",)

    def get_host_ast_builder(self):
        return PyOpenCLPythonASTBuilder(self)

    def get_device_ast_builder(self):
        return PyOpenCLCASTBuilder(self)

    # {{{ types

    def get_dtype_registry(self):
        from pyopencl.compyte.dtypes import TYPE_REGISTRY
        result = TYPE_REGISTRY

        from loopy.target.opencl import (
                DTypeRegistryWrapperWithCL1Atomics,
                DTypeRegistryWrapperWithInt8ForBool)

        result = DTypeRegistryWrapperWithInt8ForBool(result)
        if self.atomics_flavor == "cl1":
            result = DTypeRegistryWrapperWithCL1Atomics(result)
        else:
            raise NotImplementedError("atomics flavor: %s" % self.atomics_flavor)

        return result

    def is_vector_dtype(self, dtype):
        try:
            import pyopencl.cltypes as cltypes
            vec_types = cltypes.vec_types
        except ImportError:
            from pyopencl.array import vec
            vec_types = vec.types

        return (isinstance(dtype, NumpyType)
                and dtype.numpy_dtype in list(vec_types.values()))

    def vector_dtype(self, base, count):
        try:
            import pyopencl.cltypes as cltypes
            vec_types = cltypes.vec_types
        except ImportError:
            from pyopencl.array import vec
            vec_types = vec.types

        return NumpyType(
                vec_types[base.numpy_dtype, count],
                target=self)

    def alignment_requirement(self, type_decl):
        import struct

        fmt = (type_decl.struct_format()
                .replace("F", "ff")
                .replace("D", "dd"))

        return struct.calcsize(fmt)

    # }}}

    def get_kernel_executor_cache_key(self, queue, **kwargs):
        return queue.context

    def preprocess_translation_unit_for_passed_args(self, t_unit, epoint,
                                                   passed_args_dict):

        # {{{ ValueArgs -> GlobalArgs if passed as array shapes

        from loopy.kernel.data import ValueArg, GlobalArg
        import pyopencl.array as cla

        knl = t_unit[epoint]
        new_args = []

        for arg in knl.args:
            if isinstance(arg, ValueArg):
                if (arg.name in passed_args_dict
                        and isinstance(passed_args_dict[arg.name], cla.Array)
                        and passed_args_dict[arg.name].shape == ()):
                    arg = GlobalArg(name=arg.name, dtype=arg.dtype, shape=(),
                                    is_output=False, is_input=True)

            new_args.append(arg)

        knl = knl.copy(args=new_args)

        t_unit = t_unit.with_kernel(knl)

        # }}}

        return t_unit

    def get_kernel_executor(self, program, queue, **kwargs):
        from loopy.target.pyopencl_execution import PyOpenCLKernelExecutor

        epoint = kwargs.pop("entrypoint")
        program = self.preprocess_translation_unit_for_passed_args(program,
                                                                   epoint,
                                                                   kwargs)

        return PyOpenCLKernelExecutor(queue.context, program,
                                      entrypoint=epoint)

    def with_device(self, device):
        from warnings import warn
        warn("PyOpenCLTarget.with_device is deprecated, it will "
                "stop working in 2022.", DeprecationWarning, stacklevel=2)
        return self

# }}}


# {{{ host code: value arg setup

def generate_value_arg_setup(kernel, implemented_data_info):
    options = kernel.options

    import loopy as lp
    from loopy.kernel.array import ArrayBase

    cl_arg_idx = 0
    arg_idx_to_cl_arg_idx = {}

    fp_arg_count = 0

    from genpy import If, Raise, Statement as S, Suite

    result = []
    gen = result.append

    buf_indices_and_args = []
    buf_pack_indices_and_args = []

    from pyopencl.invoker import BUF_PACK_TYPECHARS

    def add_buf_arg(arg_idx, typechar, expr_str):
        if typechar in BUF_PACK_TYPECHARS:
            buf_pack_indices_and_args.append(arg_idx)
            buf_pack_indices_and_args.append(repr(typechar.encode()))
            buf_pack_indices_and_args.append(expr_str)
        else:
            buf_indices_and_args.append(arg_idx)
            buf_indices_and_args.append(f"pack('{typechar}', {expr_str})")

    for arg_idx, idi in enumerate(implemented_data_info):
        arg_idx_to_cl_arg_idx[arg_idx] = cl_arg_idx

        if not issubclass(idi.arg_class, lp.ValueArg):
            assert issubclass(idi.arg_class, ArrayBase)

            # assume each of those generates exactly one...
            cl_arg_idx += 1

            continue

        if not options.skip_arg_checks:
            gen(If("%s is None" % idi.name,
                Raise('RuntimeError("input argument \'{name}\' '
                        'must be supplied")'.format(name=idi.name))))

        if idi.dtype.is_composite():
            buf_indices_and_args.append(cl_arg_idx)
            buf_indices_and_args.append(f"{idi.name}")

            cl_arg_idx += 1

        elif idi.dtype.is_complex():
            assert isinstance(idi.dtype, NumpyType)

            dtype = idi.dtype

            if dtype.numpy_dtype == np.complex64:
                arg_char = "f"
            elif dtype.numpy_dtype == np.complex128:
                arg_char = "d"
            else:
                raise TypeError("unexpected complex type: %s" % dtype)

            buf_indices_and_args.append(cl_arg_idx)
            buf_indices_and_args.append(
                f"_lpy_pack('{arg_char}{arg_char}', "
                f"{idi.name}.real, {idi.name}.imag)")
            cl_arg_idx += 1

            fp_arg_count += 2

        elif isinstance(idi.dtype, NumpyType):
            if idi.dtype.dtype.kind == "f":
                fp_arg_count += 1

            add_buf_arg(cl_arg_idx, idi.dtype.dtype.char, idi.name)
            cl_arg_idx += 1

        else:
            raise LoopyError("do not know how to pass argument of type '%s'"
                    % idi.dtype)

    for arg_kind, args_and_indices, entry_length in [
            ("_buf", buf_indices_and_args, 2),
            ("_buf_pack", buf_pack_indices_and_args, 3),
            ]:
        assert len(args_and_indices) % entry_length == 0
        if args_and_indices:
            gen(S(f"_lpy_knl._set_arg{arg_kind}_multi("
                    f"({', '.join(str(i) for i in args_and_indices)},), "
                    ")"))

    return Suite(result), arg_idx_to_cl_arg_idx, cl_arg_idx

# }}}


def generate_array_arg_setup(kernel, implemented_data_info, arg_idx_to_cl_arg_idx):
    from loopy.kernel.array import ArrayBase
    from genpy import Statement as S, Suite

    result = []
    gen = result.append

    cl_indices_and_args = []
    for arg_idx, arg in enumerate(implemented_data_info):
        if issubclass(arg.arg_class, ArrayBase):
            cl_indices_and_args.append(arg_idx_to_cl_arg_idx[arg_idx])
            cl_indices_and_args.append(arg.name)

    if cl_indices_and_args:
        assert len(cl_indices_and_args) % 2 == 0

        gen(S(f"_lpy_knl._set_arg_multi("
            f"({', '.join(str(i) for i in cl_indices_and_args)},)"
            ")"))

    return Suite(result)


# {{{ host ast builder

class PyOpenCLPythonASTBuilder(PythonASTBuilderBase):
    """A Python host AST builder for integration with PyOpenCL.
    """

    # {{{ code generation guts

    def get_function_definition(self, kernel, name, implemented_data_info,
                                function_decl, function_body):
        from loopy.kernel.data import TemporaryVariable
        args = (
                ["_lpy_cl_kernels", "queue"]
                + [idi.name for idi in implemented_data_info
                    if not issubclass(idi.arg_class, TemporaryVariable)]
                + ["wait_for=None", "allocator=None"])

        from genpy import (For, Function, Suite, Return, Line, Statement as S)
        return Function(
                name,
                args,
                Suite([
                    Line(),
                    ] + [
                    Line(),
                    function_body,
                    Line(),
                    ] + ([
                        For("_tv", "_global_temporaries",
                            # free global temporaries
                            S("_tv.release()"))
                        ] if self._get_global_temporaries(kernel) else []
                    ) + [
                    Line(),
                    Return("_lpy_evt"),
                    ]))

    def get_function_declaration(self, kernel, name, implemented_data_info,
                                 is_generating_device_code):
        # no such thing in Python
        return None

    def _get_global_temporaries(self, kernel):
        from loopy.kernel.data import AddressSpace

        return sorted(
            (tv for tv in kernel.temporary_variables.values()
            if tv.address_space == AddressSpace.GLOBAL),
            key=lambda tv: tv.name)

    def get_temporary_decls(self, kernel, subkernel_name):
        from genpy import Assign, Comment, Line
        from collections import defaultdict
        from numbers import Number
        import pymbolic.primitives as prim

        def alloc_nbytes(tv):
            from functools import reduce
            from operator import mul
            return tv.dtype.numpy_dtype.itemsize * reduce(mul, tv.shape, 1)

        from pymbolic.mapper.stringifier import PREC_NONE
        ecm = self.get_expression_to_code_mapper(kernel, var_subst_map={})

        global_temporaries = self._get_global_temporaries(kernel)
        if not global_temporaries:
            return []

        # {{{ allocate space for the base_storage

        base_storage_sizes = defaultdict(set)

        for tv in global_temporaries:
            if tv.base_storage:
                base_storage_sizes[tv.base_storage].add(tv.nbytes)

        # }}}

        allocated_var_names = []
        code_lines = []
        code_lines.append(Line())
        code_lines.append(Comment("{{{ allocate global temporaries"))
        code_lines.append(Line())

        for name, sizes in base_storage_sizes.items():
            if all(isinstance(s, Number) for s in sizes):
                size = max(sizes)
            else:
                size = prim.Max(tuple(sizes))

            allocated_var_names.append(name)
            code_lines.append(Assign(name,
                                     f"allocator({ecm(size, PREC_NONE, 'i')})"))

        for tv in global_temporaries:
            if tv.base_storage:
                assert tv.base_storage in base_storage_sizes
                code_lines.append(Assign(tv.name, tv.base_storage))
            else:
                nbytes_str = ecm(tv.nbytes, PREC_NONE, "i")
                allocated_var_names.append(tv.name)
                code_lines.append(Assign(tv.name,
                                         f"allocator({nbytes_str})"))

        code_lines.append(Assign("_global_temporaries", "[{tvs}]".format(
            tvs=", ".join(tv for tv in allocated_var_names))))

        code_lines.append(Line())
        code_lines.append(Comment("}}}"))
        code_lines.append(Line())

        return code_lines

    def get_kernel_call(self, kernel, name, implemented_data_info, extra_args):
        ecm = self.get_expression_to_code_mapper(kernel, var_subst_map={},
                                                 vectorization_info=None)

        from loopy.schedule.tree import get_insns_in_function
        gsize, lsize = kernel.get_grid_sizes_for_insn_ids_as_exprs(
            get_insns_in_function(kernel, name))

        if not gsize:
            gsize = (1,)
        if not lsize:
            lsize = (1,)

        all_args = implemented_data_info + extra_args

        value_arg_code, arg_idx_to_cl_arg_idx, cl_arg_count = \
            generate_value_arg_setup(
                    kernel,
                    all_args)
        arry_arg_code = generate_array_arg_setup(
            kernel,
            all_args,
            arg_idx_to_cl_arg_idx)

        from genpy import Suite, Assign, Assert, Line, Comment
        from pymbolic.mapper.stringifier import PREC_NONE

        import pyopencl.version as cl_ver
        if cl_ver.VERSION < (2020, 2):
            from warnings import warn
            warn("Your kernel invocation will likely fail because your "
                    "version of PyOpenCL does not support allow_empty_ndrange. "
                    "Please upgrade to version 2020.2 or newer.")

        # TODO: Generate finer-grained dependency structure
        return Suite([
            Comment("{{{ enqueue %s" % name),
            Line(),
            Assign("_lpy_knl", "_lpy_cl_kernels."+name),
            Assert("_lpy_knl.num_args == %d" % cl_arg_count),
            Line(),
            value_arg_code,
            arry_arg_code,
            Assign("_lpy_evt", "%(pyopencl_module_name)s.enqueue_nd_range_kernel("
                "queue, _lpy_knl, "
                "%(gsize)s, %(lsize)s, "
                # using positional args because pybind is slow with kwargs
                "None, "  # offset
                "wait_for, "
                "True, "  # g_times_l
                "True, "  # allow_empty_ndrange
                ")"
                % dict(
                    pyopencl_module_name=self.target.pyopencl_module_name,
                    gsize=ecm(gsize, prec=PREC_NONE, type_context="i"),
                    lsize=ecm(lsize, prec=PREC_NONE, type_context="i"))),
            Assign("wait_for", "[_lpy_evt]"),
            Line(),
            Comment("}}}"),
            Line(),
            ])

    # }}}

# }}}


# {{{ device ast builder

class PyOpenCLCASTBuilder(OpenCLCASTBuilder):
    """A C device AST builder for integration with PyOpenCL.
    """

    # {{{ library

    @property
    def known_callables(self):
        from loopy.library.random123 import get_random123_callables

        # order matters: e.g. prefer our abs() over that of the
        # superclass
        callables = super().known_callables
        callables.update(get_pyopencl_callables())
        callables.update(get_random123_callables(self.target))
        return callables

    def preamble_generators(self):
        return ([
            pyopencl_preamble_generator,
            ] + super().preamble_generators())

    # }}}

    def get_expression_to_c_expression_mapper(self, kernel, var_subst_map,
                                              vectorization_info):
        return ExpressionToPyOpenCLCExpressionMapper(kernel, self, var_subst_map,
                                                     vectorization_info)
# }}}


# {{{ volatile mem acccess target

class VolatileMemPyOpenCLCASTBuilder(PyOpenCLCASTBuilder):
    def get_expression_to_c_expression_mapper(self, kernel, var_subst_map,
                                              vectorization_info):
        from loopy.target.opencl import \
                VolatileMemExpressionToOpenCLCExpressionMapper
        return VolatileMemExpressionToOpenCLCExpressionMapper(kernel, self,
                                                              var_subst_map,
                                                              vectorization_info)


class VolatileMemPyOpenCLTarget(PyOpenCLTarget):
    def get_device_ast_builder(self):
        return VolatileMemPyOpenCLCASTBuilder(self)

# }}}

# vim: foldmethod=marker

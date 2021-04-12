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

from loopy.kernel.data import CallMangleInfo
from loopy.target.opencl import (OpenCLTarget, OpenCLCASTBuilder,
        ExpressionToOpenCLCExpressionMapper)
from loopy.target.python import PythonASTBuilderBase
from loopy.types import NumpyType
from loopy.diagnostic import LoopyError, warn_with_kernel
from warnings import warn

import logging
logger = logging.getLogger(__name__)


# {{{ temp storage adjust for bank conflict

def adjust_local_temp_var_storage(kernel, device):
    import pyopencl as cl
    import pyopencl.characterize as cl_char

    logger.debug("%s: adjust temp var storage" % kernel.name)

    new_temp_vars = {}

    from loopy.kernel.data import AddressSpace

    lmem_size = cl_char.usable_local_mem_size(device)
    for temp_var in kernel.temporary_variables.values():
        if temp_var.address_space != AddressSpace.LOCAL:
            new_temp_vars[temp_var.name] = \
                    temp_var.copy(storage_shape=temp_var.shape)
            continue

        if not temp_var.shape:
            # scalar, no need to mess with storage shape
            new_temp_vars[temp_var.name] = temp_var
            continue

        other_loctemp_nbytes = [
                tv.nbytes
                for tv in kernel.temporary_variables.values()
                if tv.address_space == AddressSpace.LOCAL
                and tv.name != temp_var.name]

        storage_shape = temp_var.storage_shape

        if storage_shape is None:
            storage_shape = temp_var.shape

        storage_shape = list(storage_shape)

        # sizes of all dims except the last one, which we may change
        # below to avoid bank conflicts
        from pytools import product

        if device.local_mem_type == cl.device_local_mem_type.GLOBAL:
            # FIXME: could try to avoid cache associativity disasters
            new_storage_shape = storage_shape

        elif device.local_mem_type == cl.device_local_mem_type.LOCAL:
            min_mult = cl_char.local_memory_bank_count(device)
            good_incr = None
            new_storage_shape = storage_shape
            min_why_not = None

            for increment in range(storage_shape[-1]//2):

                test_storage_shape = storage_shape[:]
                test_storage_shape[-1] = test_storage_shape[-1] + increment
                new_mult, why_not = cl_char.why_not_local_access_conflict_free(
                        device, temp_var.dtype.itemsize,
                        temp_var.shape, test_storage_shape)

                # will choose smallest increment 'automatically'
                if new_mult < min_mult:
                    new_lmem_use = (sum(other_loctemp_nbytes)
                            + temp_var.dtype.itemsize*product(test_storage_shape))
                    if new_lmem_use < lmem_size:
                        new_storage_shape = test_storage_shape
                        min_mult = new_mult
                        min_why_not = why_not
                        good_incr = increment

            if min_mult != 1:
                from warnings import warn
                from loopy.diagnostic import LoopyAdvisory
                warn("could not find a conflict-free mem layout "
                        "for local variable '%s' "
                        "(currently: %dx conflict, increment: %s, reason: %s)"
                        % (temp_var.name, min_mult, good_incr, min_why_not),
                        LoopyAdvisory)
        else:
            from warnings import warn
            warn("unknown type of local memory")

            new_storage_shape = storage_shape

        new_temp_vars[temp_var.name] = temp_var.copy(
                storage_shape=tuple(new_storage_shape))

    return kernel.copy(temporary_variables=new_temp_vars)

# }}}


# {{{ check sizes against device properties

def check_sizes(kernel, device):
    import loopy as lp

    from loopy.diagnostic import LoopyAdvisory, LoopyError

    if device is None:
        warn_with_kernel(kernel, "no_device_in_pre_codegen_checks",
                "No device parameter was passed to the PyOpenCLTarget. "
                "Perhaps you want to pass a device to benefit from "
                "additional checking.", LoopyAdvisory)
        return

    parameters = {}
    for arg in kernel.args:
        if isinstance(arg, lp.ValueArg) and arg.approximately is not None:
            parameters[arg.name] = arg.approximately

    glens, llens = kernel.get_grid_size_upper_bounds_as_exprs()

    if (max(len(glens), len(llens))
            > device.max_work_item_dimensions):
        raise LoopyError("too many work item dimensions")

    from pymbolic import evaluate
    from pymbolic.mapper.evaluator import UnknownVariableError
    try:
        glens = evaluate(glens, parameters)
        llens = evaluate(llens, parameters)
    except UnknownVariableError as name:
        from warnings import warn
        warn("could not check axis bounds because no value "
                "for variable '%s' was passed to check_kernels()"
                % name, LoopyAdvisory)
    else:
        for i in range(len(llens)):
            if llens[i] > device.max_work_item_sizes[i]:
                raise LoopyError("group axis %d too big" % i)

        from pytools import product
        if product(llens) > device.max_work_group_size:
            raise LoopyError("work group too big")

    local_mem_use = kernel.local_mem_use()

    from pyopencl.characterize import usable_local_mem_size
    import numbers
    if isinstance(local_mem_use, numbers.Integral):
        if local_mem_use > usable_local_mem_size(device):
            raise LoopyError("using too much local memory")
    else:
        warn_with_kernel(kernel, "non_constant_local_mem",
                "The amount of local memory used by the kernel "
                "is not a constant. This will likely cause problems.")

    from loopy.kernel.data import ConstantArg
    const_arg_count = sum(
            1 for arg in kernel.args
            if isinstance(arg, ConstantArg))

    if const_arg_count > device.max_constant_args:
        raise LoopyError("too many constant arguments")

# }}}


def pyopencl_function_mangler(target, name, arg_dtypes):
    if len(arg_dtypes) == 1 and isinstance(name, str):
        arg_dtype, = arg_dtypes

        if arg_dtype.is_complex():
            if arg_dtype.numpy_dtype == np.complex64:
                tpname = "cfloat"
            elif arg_dtype.numpy_dtype == np.complex128:
                tpname = "cdouble"
            else:
                raise RuntimeError("unexpected complex type '%s'" % arg_dtype)

            if name in ["sqrt", "exp", "log",
                    "sin", "cos", "tan",
                    "sinh", "cosh", "tanh",
                    "conj"]:
                return CallMangleInfo(
                        target_name=f"{tpname}_{name}",
                        result_dtypes=(arg_dtype,),
                        arg_dtypes=(arg_dtype,))

            if name in ["real", "imag", "abs"]:
                return CallMangleInfo(
                        target_name=f"{tpname}_{name}",
                        result_dtypes=(NumpyType(
                            np.dtype(arg_dtype.numpy_dtype.type(0).real)),
                            ),
                        arg_dtypes=(arg_dtype,))

    return None


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


# {{{ pyopencl tools

class _LegacyTypeRegistryStub:
    """Adapts legacy PyOpenCL type registry to be usable with PyOpenCLTarget."""

    def get_or_register_dtype(self, names, dtype=None):
        from pyopencl.compyte.dtypes import get_or_register_dtype
        return get_or_register_dtype(names, dtype)

    def dtype_to_ctype(self, dtype):
        from pyopencl.compyte.dtypes import dtype_to_ctype
        return dtype_to_ctype(dtype)

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

    def wrap_in_typecast_lazy(self, actual_type_func, needed_dtype, s):
        if needed_dtype.is_complex():
            return self.wrap_in_typecase(self, actual_type_func(), needed_dtype, s)
        else:
            return s

    def wrap_in_typecast(self, actual_type, needed_dtype, s):
        if (actual_type.is_complex() and needed_dtype.is_complex()
                and actual_type != needed_dtype):
            return p.Variable("%s_cast" % self.complex_type_name(needed_dtype))(s)
        elif not actual_type.is_complex() and needed_dtype.is_complex():
            return p.Variable("%s_fromreal" % self.complex_type_name(needed_dtype))(
                    s)
        else:
            return s

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

            if real_sum:
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

            if real_prd:
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
            for i in range(expr.exponent-1):
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
        if pyopencl.version.VERSION < (2021, 1):
            raise RuntimeError("The version of loopy you have installed "
                    "generates invoker code that requires PyOpenCL 2021.1 "
                    "or newer.")

        self.device = device
        self.pyopencl_module_name = pyopencl_module_name

    # NB: Not including 'device', as that is handled specially here.
    hash_fields = OpenCLTarget.hash_fields + (
            "pyopencl_module_name",)
    comparison_fields = OpenCLTarget.comparison_fields + (
            "pyopencl_module_name",)

    def __eq__(self, other):
        if not super().__eq__(other):
            return False

        if (self.device is None) != (other.device is None):
            return False

        if self.device is not None:
            assert other.device is not None
            return (self.device.hashable_model_and_version_identifier
                    == other.device.hashable_model_and_version_identifier)
        else:
            assert other.device is None
            return True

    def update_persistent_hash(self, key_hash, key_builder):
        super().update_persistent_hash(key_hash, key_builder)
        key_builder.rec(key_hash, getattr(
            self.device, "hashable_model_and_version_identifier", None))

    def __getstate__(self):
        dev_id = None
        if self.device is not None:
            dev_id = self.device.hashable_model_and_version_identifier

        return {
                "device_id": dev_id,
                "atomics_flavor": self.atomics_flavor,
                "use_int8_for_bool": self.use_int8_for_bool,
                "fortran_abi": self.fortran_abi,
                "pyopencl_module_name": self.pyopencl_module_name,
                }

    def __setstate__(self, state):
        self.atomics_flavor = state["atomics_flavor"]
        self.use_int8_for_bool = state["use_int8_for_bool"]
        self.fortran_abi = state["fortran_abi"]
        self.pyopencl_module_name = state["pyopencl_module_name"]

        dev_id = state["device_id"]
        if dev_id is None:
            self.device = None
        else:
            import pyopencl as cl
            matches = [
                dev
                for plat in cl.get_platforms()
                for dev in plat.get_devices()
                if dev.hashable_model_and_version_identifier == dev_id]

            if matches:
                self.device = matches[0]
            else:
                raise LoopyError(
                        "cannot unpickle device '%s': not found"
                        % dev_id)

    def preprocess(self, kernel):
        if self.device is not None:
            kernel = adjust_local_temp_var_storage(kernel, self.device)
        return kernel

    def pre_codegen_check(self, kernel):
        check_sizes(kernel, self.device)

    def get_host_ast_builder(self):
        return PyOpenCLPythonASTBuilder(self)

    def get_device_ast_builder(self):
        return PyOpenCLCASTBuilder(self)

    # {{{ types

    def get_dtype_registry(self):
        try:
            from pyopencl.compyte.dtypes import TYPE_REGISTRY
        except ImportError:
            result = _LegacyTypeRegistryStub()
        else:
            result = TYPE_REGISTRY

        from loopy.target.opencl import (DTypeRegistryWrapperWithCL1Atomics,
                                         DTypeRegistryWrapperWithInt8ForBool)

        if self.atomics_flavor == "cl1":
            result = DTypeRegistryWrapperWithCL1Atomics(result)
        else:
            raise NotImplementedError("atomics flavor: %s" % self.atomics_flavor)

        if self.use_int8_for_bool:
            result = DTypeRegistryWrapperWithInt8ForBool(result)

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

    def get_kernel_executor(self, kernel, queue, **kwargs):
        from loopy.target.pyopencl_execution import PyOpenCLKernelExecutor
        return PyOpenCLKernelExecutor(queue.context, kernel)

    def with_device(self, device):
        return type(self)(device)

# }}}


# {{{ host code: value arg setup

def generate_value_arg_setup(kernel, devices, implemented_data_info):
    options = kernel.options

    import loopy as lp
    from loopy.kernel.array import ArrayBase

    # {{{ arg counting bug handling

    # For example:
    # https://github.com/pocl/pocl/issues/197
    # (but Apple CPU has a similar bug)

    work_around_arg_count_bug = False
    warn_about_arg_count_bug = False

    try:
        from pyopencl.characterize import has_struct_arg_count_bug

    except ImportError:
        count_bug_per_dev = [False]*len(devices)

    else:
        count_bug_per_dev = [
                has_struct_arg_count_bug(dev)
                if dev is not None else False
                for dev in devices]

    if any(dev is None for dev in devices):
        warn("{knl_name}: device not supplied to PyOpenCLTarget--"
                "workarounds for broken OpenCL implementations "
                "(such as those relating to complex numbers) "
                "may not be enabled when needed. To avoid this, "
                "pass target=lp.PyOpenCLTarget(dev) when creating "
                "the kernel."
                .format(knl_name=kernel.name))

    if any(count_bug_per_dev):
        if all(count_bug_per_dev):
            work_around_arg_count_bug = True
        else:
            warn_about_arg_count_bug = True

    # }}}

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

            if warn_about_arg_count_bug:
                warn("{knl_name}: arguments include complex numbers, and "
                        "some (but not all) of the target devices mishandle "
                        "struct kernel arguments (hence the workaround is "
                        "disabled".format(
                            knl_name=kernel.name))

            if dtype.numpy_dtype == np.complex64:
                arg_char = "f"
            elif dtype.numpy_dtype == np.complex128:
                arg_char = "d"
            else:
                raise TypeError("unexpected complex type: %s" % dtype)

            if (work_around_arg_count_bug
                    and dtype.numpy_dtype == np.complex128
                    and fp_arg_count + 2 <= 8):
                add_buf_arg(cl_arg_idx, arg_char, f"{idi.name}.real")
                cl_arg_idx += 1

                add_buf_arg(cl_arg_idx, arg_char, f"{idi.name}.imag")
                cl_arg_idx += 1
            else:
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

    def get_function_definition(self, codegen_state, codegen_result,
            schedule_index, function_decl, function_body):
        from loopy.kernel.data import TemporaryVariable
        args = (
                ["_lpy_cl_kernels", "queue"]
                + [idi.name for idi in codegen_state.implemented_data_info
                    if not issubclass(idi.arg_class, TemporaryVariable)]
                + ["wait_for=None", "allocator=None"])

        from genpy import (For, Function, Suite, Return, Line, Statement as S)
        return Function(
                codegen_result.current_program(codegen_state).name,
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
                        ] if self._get_global_temporaries(codegen_state) else []
                    ) + [
                    Line(),
                    Return("_lpy_evt"),
                    ]))

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        # no such thing in Python
        return None

    def _get_global_temporaries(self, codegen_state):
        from loopy.kernel.data import AddressSpace

        return sorted(
            (tv for tv in codegen_state.kernel.temporary_variables.values()
            if tv.address_space == AddressSpace.GLOBAL),
            key=lambda tv: tv.name)

    def get_temporary_decls(self, codegen_state, schedule_state):
        from genpy import Assign, Comment, Line

        def alloc_nbytes(tv):
            from functools import reduce
            from operator import mul
            return tv.dtype.numpy_dtype.itemsize * reduce(mul, tv.shape, 1)

        from pymbolic.mapper.stringifier import PREC_NONE
        ecm = self.get_expression_to_code_mapper(codegen_state)

        global_temporaries = self._get_global_temporaries(codegen_state)
        if not global_temporaries:
            return []

        return [
            Comment("{{{ allocate global temporaries"),
            Line()] + [
            Assign(tv.name, "allocator(%s)" %
                ecm(alloc_nbytes(tv), PREC_NONE, "i"))
                for tv in global_temporaries] + [
            Assign("_global_temporaries", "[{tvs}]".format(tvs=", ".join(
                tv.name for tv in global_temporaries)))] + [
            Line(),
            Comment("}}}"),
            Line()]

    def get_kernel_call(self, codegen_state, name, gsize, lsize, extra_args):
        ecm = self.get_expression_to_code_mapper(codegen_state)

        if not gsize:
            gsize = (1,)
        if not lsize:
            lsize = (1,)

        all_args = codegen_state.implemented_data_info + extra_args

        value_arg_code, arg_idx_to_cl_arg_idx, cl_arg_count = \
            generate_value_arg_setup(
                    codegen_state.kernel,
                    [self.target.device],
                    all_args)
        arry_arg_code = generate_array_arg_setup(
            codegen_state.kernel,
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

    def function_manglers(self):
        from loopy.library.random123 import random123_function_mangler
        return (
                [
                    pyopencl_function_mangler,
                    random123_function_mangler
                    # order matters: e.g. prefer our abs() over that of the
                    # superclass
                    ] + super().function_manglers())

    def preamble_generators(self):
        from loopy.library.random123 import random123_preamble_generator
        return ([
            pyopencl_preamble_generator,
            random123_preamble_generator,
            ] + super().preamble_generators())

    # }}}

    def get_expression_to_c_expression_mapper(self, codegen_state):
        return ExpressionToPyOpenCLCExpressionMapper(codegen_state)


# }}}


# {{{ volatile mem acccess target

class VolatileMemPyOpenCLCASTBuilder(PyOpenCLCASTBuilder):
    def get_expression_to_c_expression_mapper(self, codegen_state):
        from loopy.target.opencl import \
                VolatileMemExpressionToOpenCLCExpressionMapper
        return VolatileMemExpressionToOpenCLCExpressionMapper(codegen_state)


class VolatileMemPyOpenCLTarget(PyOpenCLTarget):
    def get_device_ast_builder(self):
        return VolatileMemPyOpenCLCASTBuilder(self)

# }}}

# vim: foldmethod=marker

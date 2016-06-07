"""OpenCL target integrated with PyOpenCL."""

from __future__ import division, absolute_import

import sys

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

import six
from six.moves import range

import numpy as np

from loopy.kernel.data import CallMangleInfo
from loopy.target.opencl import OpenCLTarget, OpenCLCASTBuilder
from loopy.target.python import PythonASTBuilderBase
from loopy.types import NumpyType
from loopy.diagnostic import LoopyError
from warnings import warn

import logging
logger = logging.getLogger(__name__)


# {{{ temp storage adjust for bank conflict

def adjust_local_temp_var_storage(kernel, device):
    import pyopencl as cl
    import pyopencl.characterize as cl_char

    logger.debug("%s: adjust temp var storage" % kernel.name)

    new_temp_vars = {}

    from loopy.kernel.data import temp_var_scope

    lmem_size = cl_char.usable_local_mem_size(device)
    for temp_var in six.itervalues(kernel.temporary_variables):
        if temp_var.scope != temp_var_scope.LOCAL:
            new_temp_vars[temp_var.name] = \
                    temp_var.copy(storage_shape=temp_var.shape)
            continue

        other_loctemp_nbytes = [
                tv.nbytes
                for tv in six.itervalues(kernel.temporary_variables)
                if tv.scope == temp_var_scope.LOCAL
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

        new_temp_vars[temp_var.name] = temp_var.copy(storage_shape=new_storage_shape)

    return kernel.copy(temporary_variables=new_temp_vars)

# }}}


# {{{ check sizes against device properties

def check_sizes(kernel, device):
    import loopy as lp

    from loopy.diagnostic import LoopyAdvisory, LoopyError

    if device is None:
        from loopy.diagnostic import warn
        warn(kernel, "no_device_in_pre_codegen_checks",
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

    from pyopencl.characterize import usable_local_mem_size
    if kernel.local_mem_use() > usable_local_mem_size(device):
        raise LoopyError("using too much local memory")

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
                        target_name="%s_%s" % (tpname, name),
                        result_dtypes=(arg_dtype,),
                        arg_dtypes=(arg_dtype,))

            if name in ["real", "imag", "abs"]:
                return CallMangleInfo(
                        target_name="%s_%s" % (tpname, name),
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

class _LegacyTypeRegistryStub(object):
    """Adapts legacy PyOpenCL type registry to be usable with PyOpenCLTarget."""

    def get_or_register_dtype(self, names, dtype=None):
        from pyopencl.compyte.dtypes import get_or_register_dtype
        return get_or_register_dtype(names, dtype)

    def dtype_to_ctype(self, dtype):
        from pyopencl.compyte.dtypes import dtype_to_ctype
        return dtype_to_ctype(dtype)

# }}}


# {{{ target

class PyOpenCLTarget(OpenCLTarget):
    """A code generation target that takes special advantage of :mod:`pyopencl`
    features such as run-time knowledge of the target device (to generate
    warnings) and support for complex numbers.
    """

    host_program_name_prefix = "_lpy_host_"
    host_program_name_suffix = ""

    def __init__(self, device=None, pyopencl_module_name="_lpy_cl"):
        # This ensures the dtype registry is populated.
        import pyopencl.tools  # noqa

        super(PyOpenCLTarget, self).__init__()

        self.device = device
        self.pyopencl_module_name = pyopencl_module_name

    comparison_fields = ["device"]

    def update_persistent_hash(self, key_hash, key_builder):
        super(PyOpenCLTarget, self).update_persistent_hash(key_hash, key_builder)
        key_builder.rec(key_hash, getattr(self.device, "persistent_unique_id", None))

    def __getstate__(self):
        dev_id = None
        if self.device is not None:
            dev_id = self.device.persistent_unique_id

        return {
                "device_id": dev_id,
                "atomics_flavor": self.atomics_flavor,
                "fortran_abi": self.fortran_abi,
                "pyopencl_module_name": self.pyopencl_module_name,
                }

    def __setstate__(self, state):
        self.atomics_flavor = state["atomics_flavor"]
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
                if dev.persistent_unique_id == dev_id]

            if matches:
                self.device = matches[0]
            else:
                raise LoopyError(
                        "cannot unpickle device '%s': not found"
                        % dev_id)

    def preprocess(self, kernel):
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

        from loopy.target.opencl import DTypeRegistryWrapperWithCL1Atomics
        if self.atomics_flavor == "cl1":
            return DTypeRegistryWrapperWithCL1Atomics(result)
        else:
            raise NotImplementedError("atomics flavor: %s" % self.atomics_flavor)

    def is_vector_dtype(self, dtype):
        from pyopencl.array import vec
        return (isinstance(dtype, NumpyType)
                and dtype.numpy_dtype in list(vec.types.values()))

    def vector_dtype(self, base, count):
        from pyopencl.array import vec
        return NumpyType(
                vec.types[base.numpy_dtype, count],
                target=self)

    def alignment_requirement(self, type_decl):
        import struct

        fmt = (type_decl.struct_format()
                .replace("F", "ff")
                .replace("D", "dd"))

        return struct.calcsize(fmt)

    # }}}

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
                "may not be enabled when needed"
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

    from genpy import (
            Comment, Line, If, Raise, Assign, Statement as S, Suite)

    result = []
    gen = result.append

    for arg_idx, idi in enumerate(implemented_data_info):
        arg_idx_to_cl_arg_idx[arg_idx] = cl_arg_idx

        if not issubclass(idi.arg_class, lp.ValueArg):
            assert issubclass(idi.arg_class, ArrayBase)

            # assume each of those generates exactly one...
            cl_arg_idx += 1

            continue

        gen(Comment("{{{ process %s" % idi.name))
        gen(Line())

        if not options.skip_arg_checks:
            gen(If("%s is None" % idi.name,
                Raise('RuntimeError("input argument \'{name}\' '
                        'must be supplied")'.format(name=idi.name))))

        if idi.dtype.is_integral():
            gen(Comment("cast to Python int to avoid trouble "
                "with struct packing or Boost.Python"))
            if sys.version_info < (3,):
                py_type = "long"
            else:
                py_type = "int"

            gen(Assign(idi.name, "%s(%s)" % (py_type, idi.name)))
            gen(Line())

        if idi.dtype.is_composite():
            gen(S("_lpy_knl.set_arg(%d, %s)" % (cl_arg_idx, idi.name)))
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
                gen(Assign(
                    "_lpy_buf",
                    "_lpy_pack('{arg_char}', {arg_var}.real)"
                    .format(arg_char=arg_char, arg_var=idi.name)))
                gen(S(
                    "_lpy_knl.set_arg({cl_arg_idx}, _lpy_buf)"
                    .format(cl_arg_idx=cl_arg_idx)))
                cl_arg_idx += 1

                gen(Assign(
                    "_lpy_buf",
                    "_lpy_pack('{arg_char}', {arg_var}.imag)"
                    .format(arg_char=arg_char, arg_var=idi.name)))
                gen(S(
                        "_lpy_knl.set_arg({cl_arg_idx}, _lpy_buf)"
                        .format(cl_arg_idx=cl_arg_idx)))
                cl_arg_idx += 1
            else:
                gen(Assign(
                    "_lpy_buf",
                    "_lpy_pack('{arg_char}{arg_char}', "
                    "{arg_var}.real, {arg_var}.imag)"
                    .format(arg_char=arg_char, arg_var=idi.name)))
                gen(S(
                    "_lpy_knl.set_arg({cl_arg_idx}, _lpy_buf)"
                    .format(cl_arg_idx=cl_arg_idx)))
                cl_arg_idx += 1

            fp_arg_count += 2

        elif isinstance(idi.dtype, NumpyType):
            if idi.dtype.dtype.kind == "f":
                fp_arg_count += 1

            gen(S(
                "_lpy_knl.set_arg(%d, _lpy_pack('%s', %s))"
                % (cl_arg_idx, idi.dtype.dtype.char, idi.name)))

            cl_arg_idx += 1

        else:
            raise LoopyError("do not know how to pass argument of type '%s'"
                    % idi.dtype)

        gen(Line())

        gen(Comment("}}}"))
        gen(Line())

    return Suite(result), arg_idx_to_cl_arg_idx, cl_arg_idx

# }}}


def generate_array_arg_setup(kernel, implemented_data_info, arg_idx_to_cl_arg_idx):
    from loopy.kernel.array import ArrayBase
    from genpy import Statement as S, Suite

    result = []
    gen = result.append

    for arg_idx, arg in enumerate(implemented_data_info):
        if not issubclass(arg.arg_class, ArrayBase):
            continue

        cl_arg_idx = arg_idx_to_cl_arg_idx[arg_idx]

        gen(S("_lpy_knl.set_arg(%d, %s)" % (cl_arg_idx, arg.name)))

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

        from genpy import (For, Function, Suite, Import, ImportAs, Return,
                FromImport, If, Assign, Line, Statement as S)
        return Function(
                codegen_result.current_program(codegen_state).name,
                args,
                Suite([
                    FromImport("struct", ["pack as _lpy_pack"]),
                    ImportAs("pyopencl", "_lpy_cl"),
                    Import("pyopencl.tools"),
                    Line(),
                    If("allocator is None",
                        Assign(
                            "allocator",
                            "_lpy_cl_tools.DeferredAllocator(queue.context)")),
                    Line(),
                    ] + [
                    Line(),
                    function_body,
                    Line(),
                    ] + [
                    For("_tv", "_global_temporaries",
                        # free global temporaries
                        S("_tv.release()"))
                    ] + [
                    Line(),
                    Return("_lpy_evt"),
                    ]))

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        # no such thing in Python
        return None

    def get_temporary_decls(self, codegen_state, schedule_state):
        from genpy import Assign, Comment, Line

        def alloc_nbytes(tv):
            return tv.dtype.numpy_dtype.itemsize

        from loopy.kernel.data import temp_var_scope

        global_temporaries = sorted(
            (tv for tv in six.itervalues(codegen_state.kernel.temporary_variables)
            if tv.scope == temp_var_scope.GLOBAL),
            key=lambda tv: tv.name)

        from pymbolic.mapper.stringifier import PREC_NONE
        ecm = self.get_expression_to_code_mapper(codegen_state)

        if not global_temporaries:
            return [Assign("_global_temporaries", "[]"), Line()]

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
                "%(gsize)s, %(lsize)s,  wait_for=wait_for, g_times_l=True)"
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
                super(PyOpenCLCASTBuilder, self).function_manglers() + [
                    pyopencl_function_mangler,
                    random123_function_mangler
                    ])

    def preamble_generators(self):
        from loopy.library.random123 import random123_preamble_generator
        return ([
            pyopencl_preamble_generator,
            random123_preamble_generator,
            ] + super(PyOpenCLCASTBuilder, self).preamble_generators())

    # }}}

# }}}


# vim: foldmethod=marker

__copyright__ = "Copyright (C) 2017 Nick Curtis"

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

from typing import Callable, Any, Union, Tuple, Sequence, Optional
import tempfile
import os
import ctypes
from dataclasses import dataclass

from immutables import Map
from pytools import memoize_method
from pytools.codegen import Indentation, CodeGenerator
from pytools.prefork import ExecError
from codepy.toolchain import guess_toolchain, ToolchainGuessError, GCCToolchain
from codepy.jit import compile_from_string

import numpy as np

from loopy.typing import ExpressionT
from loopy.types import LoopyType
from loopy.kernel import LoopKernel
from loopy.kernel.array import ArrayBase
from loopy.kernel.data import ArrayArg
from loopy.schedule.tools import KernelArgInfo
from loopy.codegen.result import GeneratedProgram
from loopy.translation_unit import TranslationUnit
from loopy.target.execution import (ExecutorBase,
                             ExecutionWrapperGeneratorBase, get_highlighted_code)

import logging
logger = logging.getLogger(__name__)

DEF_EVEN_DIV_FUNCTION = """
def _lpy_even_div(a, b):
    result, remdr = divmod(a, b)
    if remdr != 0:
        # FIXME: This error message is kind of crummy.
        raise ValueError("expected even division")
    return result


def _lpy_even_div_none(a, b):
    if a is None:
        return None

    result, remdr = divmod(a, b)
    if remdr != 0:
        # FIXME: This error message is kind of crummy.
        raise ValueError("expected even division")
    return result
"""


# {{{ CExecutionWrapperGenerator

class CExecutionWrapperGenerator(ExecutionWrapperGeneratorBase):
    """
    Specialized form of the :class:`ExecutionWrapperGeneratorBase` for
    pyopencl execution
    """

    def __init__(self):
        system_args = ["_lpy_c_kernels"]
        super().__init__(system_args)

    def python_dtype_str_inner(self, dtype):
        if np.dtype(str(dtype)).isbuiltin:
            name = dtype.name
            if dtype.name == "bool":
                name = "bool_"
            return f"_lpy_np.dtype(_lpy_np.{name})"
        raise Exception(f"dtype: {dtype} not recognized")

    # {{{ handle non numpy arguements

    def handle_non_numpy_arg(self, gen, arg):
        pass

    # }}}

    # {{{ handle allocation of unspecified arguements

    def handle_alloc(
            self, gen: CodeGenerator, arg: ArrayArg,
            strify: Callable[[Union[ExpressionT, Tuple[ExpressionT]]], str],
            skip_arg_checks: bool) -> None:
        """
        Handle allocation of non-specified arguements for C-execution
        """
        from pymbolic import var

        assert isinstance(arg.shape, tuple)
        assert isinstance(arg.dtype, LoopyType)
        num_axes = len(arg.shape)
        for i in range(num_axes):
            gen("_lpy_shape_%d = %s" % (i, strify(arg.shape[i])))

        from loopy.kernel.array import get_strides
        strides = get_strides(arg)
        num_axes = len(strides)

        itemsize = arg.dtype.numpy_dtype.itemsize
        for i in range(num_axes):
            gen("_lpy_strides_%d = %s" % (i, strify(
                itemsize*strides[i])))

        if not skip_arg_checks:
            for i in range(num_axes):
                gen("assert _lpy_strides_%d > 0, "
                        "\"'%s' has negative stride in axis %d\""
                        % (i, arg.name, i))

        sym_strides = tuple(
                var("_lpy_strides_%d" % i)
                for i in range(num_axes))

        sym_shape = tuple(
                var("_lpy_shape_%d" % i)
                for i in range(num_axes))

        # find order of array
        from loopy.kernel.array import get_strides
        strides = get_strides(arg)
        order = "'C'" if (arg.shape == () or strides[-1] == 1) else "'F'"

        gen(f"{arg.name} = _lpy_np.empty({strify(sym_shape)}, "
                f"{self.python_dtype_str(gen, arg.dtype.numpy_dtype)}, "
                f"order={order})")

        expected_strides = tuple(
                var("_lpy_expected_strides_%s" % i)
                for i in range(num_axes))

        gen("{} = {}.strides".format(strify(expected_strides), arg.name))

        #check strides
        if not skip_arg_checks:
            strides_check_expr = self.get_strides_check_expr(
                    [strify(s) for s in sym_shape],
                    [strify(s) for s in sym_strides],
                    [strify(s) for s in expected_strides])
            gen(f"assert {strides_check_expr}, "
                    f"'Strides of loopy created array {arg.name}, "
                    "do not match expected.'")
            for i in range(num_axes):
                gen("del _lpy_shape_%d" % i)
                gen("del _lpy_strides_%d" % i)
            gen("")

    # }}}

    def target_specific_preamble(self, gen):
        """
        Add default C-imports to preamble
        """
        gen.add_to_preamble("import numpy as _lpy_np")
        gen.add_to_preamble(DEF_EVEN_DIV_FUNCTION)

    def initialize_system_args(self, gen):
        """
        Initializes possibly empty system arguements
        """
        pass

    # {{{ generate invocation

    def generate_invocation(self, gen: CodeGenerator, kernel: LoopKernel,
            kai: KernelArgInfo, host_program_name: str, args: Sequence[str]) -> None:
        gen("for knl in _lpy_c_kernels:")
        with Indentation(gen):
            gen(f"knl({', '.join(args)})")

    # }}}

    # {{{

    def generate_output_handler(self, gen: CodeGenerator,
            kernel: LoopKernel, kai: KernelArgInfo) -> None:
        options = kernel.options

        if options.return_dict:
            gen("return None, {%s}"
                    % ", ".join(f'"{arg_name}": {arg_name}'
                        for arg_name in kai.passed_arg_names
                        if kernel.arg_dict[arg_name].is_output))
        else:
            out_names = [arg_name for arg_name in kai.passed_arg_names
                    if kernel.arg_dict[arg_name].is_output]
            if out_names:
                gen(f"return None, ({', '.join(out_names)},)")
            else:
                gen("return None, ()")

    # }}}

    def generate_host_code(self, gen, codegen_result):
        # "host" code for C is embedded in the same file as the "device" code
        # this will enable a logical jumping off point for global barriers for
        # OpenMP, etc.
        pass

    def get_arg_pass(self, arg):
        return arg.name

# }}}


# {{{ CCompiler

class CCompiler:
    """
    The compiler module handles invocation of compilers to generate a shared lib
    using codepy, which can subsequently be loaded via ctypes.

    The general strategy here is as follows:

    1.  A :class:`codepy.Toolchain` is guessed from distutils.
        The user may override any flags obtained therein by passing in arguements
        to cc, cflags, etc.

    2.  The kernel source is built into and object first, then made into a shared
        library using :meth:`codepy.jit.compile_from_string`, which additionally
        handles caching

    3.  The resulting shared library is turned into a :class:`ctypes.CDLL`
        to enable calling by the invoker generated by, e.g.,
        :class:`CExecutionWrapperGenerator`
    """

    def __init__(self, toolchain=None,
                 cc="gcc", cflags=None,
                 ldflags=None, libraries=None,
                 include_dirs=None, library_dirs=None, defines=None,
                 source_suffix="c"):
        if cflags is None:
            cflags = "-std=c99 -O3 -fPIC".split()
        if ldflags is None:
            ldflags = "-shared".split()
        if libraries is None:
            libraries = []
        if include_dirs is None:
            include_dirs = []
        if library_dirs is None:
            library_dirs = []
        if defines is None:
            defines = []

        # try to get a default toolchain
        # or subclass supplied version if available
        self.toolchain = toolchain
        if toolchain is None:
            try:
                self.toolchain = guess_toolchain()
            except (ToolchainGuessError, ExecError):
                # missing compiler python was built with (likely, Conda)
                # use a default GCCToolchain
                logger = logging.getLogger(__name__)
                logger.warn("Default toolchain guessed from python config "
                            "not found, replacing with default GCCToolchain.")
                # this is ugly, but I'm not sure there's a clean way to copy the
                # default args
                self.toolchain = GCCToolchain(
                    cc="gcc",
                    cflags="-std=c99 -O3 -fPIC".split(),
                    ldflags="-shared".split(),
                    libraries=[],
                    library_dirs=[],
                    defines=[],
                    undefines=[],
                    source_suffix="c",
                    so_ext=".so",
                    o_ext=".o",
                    include_dirs=[])

        if toolchain is None:
            # copy in all differing values
            diff = {"cc": cc,
                    "cflags": cflags,
                    "ldflags": ldflags,
                    "libraries": libraries,
                    "include_dirs": include_dirs,
                    "library_dirs": library_dirs,
                    "defines": defines}
            # filter empty and those equal to toolchain defaults
            diff = {k: v for k, v in diff.items()
                    if v and (not hasattr(self.toolchain, k) or
                              getattr(self.toolchain, k) != v)}
            self.toolchain = self.toolchain.copy(**diff)
        self.tempdir = tempfile.mkdtemp(prefix="tmp_loopy")
        self.source_suffix = source_suffix

    def _tempname(self, name):
        """Build temporary filename path in tempdir."""
        return os.path.join(self.tempdir, name)

    def build(self, name, code, debug=False, wait_on_error=None,
                     debug_recompile=True):
        """Compile code, build and load shared library."""
        logger.debug(code)
        c_fname = self._tempname("code." + self.source_suffix)

        # build object
        _, mod_name, ext_file, recompiled = \
            compile_from_string(self.toolchain, name, code, c_fname,
                                self.tempdir, debug, wait_on_error,
                                debug_recompile, False)

        if recompiled:
            logger.debug(f"Kernel {name} compiled from source")
        else:
            logger.debug(f"Kernel {name} retrieved from cache")

        # and return compiled
        return ctypes.CDLL(ext_file)

# }}}


# {{{ CPlusPlusCompiler

class CPlusPlusCompiler(CCompiler):
    """Subclass of CCompiler to invoke a C++ compiler."""

    def __init__(self, toolchain=None,
                 cc="g++", cflags=None,
                 ldflags=None, libraries=None,
                 include_dirs=None, library_dirs=None, defines=None,
                 source_suffix="cpp"):

        super().__init__(
            toolchain=toolchain, cc=cc, cflags=cflags, ldflags=ldflags,
            libraries=libraries, include_dirs=include_dirs,
            library_dirs=library_dirs, defines=defines, source_suffix=source_suffix)

# }}}


# {{{ placeholder till ctypes fixes: https://github.com/python/cpython/issues/61103

class Complex64(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class Complex128(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


class Complex256(ctypes.Structure):
    _fields_ = [("real", ctypes.c_longdouble), ("imag", ctypes.c_longdouble)]


_NUMPY_COMPLEX_TYPE_TO_CTYPE = {
        np.complex64: Complex64,
        np.complex128: Complex128,
        }
if hasattr(np, "complex256"):
    _NUMPY_COMPLEX_TYPE_TO_CTYPE[np.complex256] = Complex256

# }}}


# {{{ _args_to_ctypes

def _args_to_ctypes(kernel: LoopKernel, passed_names: Sequence[str]):
    def _dtype_to_ctype(dtype):
        """Map NumPy dtype to equivalent ctypes type."""
        if dtype.is_complex():
            # complex ctypes aren't exposed
            np_dtype = dtype.numpy_dtype.type
            basetype = _NUMPY_COMPLEX_TYPE_TO_CTYPE[np_dtype]
        else:
            basetype = np.ctypeslib.as_ctypes_type(dtype)
        return basetype

    arg_info = []
    for arg_name in passed_names:
        arg = kernel.arg_dict[arg_name]

        ctype = _dtype_to_ctype(arg.dtype)
        if isinstance(arg, ArrayBase):
            ctype = ctypes.POINTER(ctype)
        arg_info.append(ctype)

    return arg_info

# }}}


# {{{ CompiledCKernel

class CompiledCKernel:
    """
    A CompiledCKernel wraps a loopy kernel, compiling it and loading the
    result as a shared library, and provides access to the kernel as a
    ctypes function object, wrapped by the __call__ method, which attempts
    to automatically map argument types.
    """

    def __init__(self, kernel: LoopKernel, devprog: GeneratedProgram,
            passed_names: Sequence[str], dev_code: str,
            comp: Optional["CCompiler"] = None):
        # get code and build
        self.code = dev_code
        self.comp = comp if comp is not None else CCompiler()
        self.dll = self.comp.build(devprog.name, self.code)

        # get the function declaration for interface with ctypes
        self._fn = getattr(self.dll, devprog.name)
        # kernels are void by defn.
        self._fn.restype = None
        self._fn.argtypes = _args_to_ctypes(kernel, passed_names)

    def __call__(self, *args):
        """Execute kernel with given args mapped to ctypes equivalents."""
        args_ = []
        for arg, arg_t in zip(args, self._fn.argtypes):
            if hasattr(arg, "ctypes"):
                if arg.size == 0:
                    # TODO eliminate unused arguments from kernel
                    arg_ = arg_t(0.0)
                else:
                    arg_ = arg.ctypes.data_as(arg_t)
            else:
                arg_ = arg_t(arg)
            args_.append(arg_)
        self._fn(*args_)

# }}}


@dataclass(frozen=True)
class _KernelInfo:
    t_unit: TranslationUnit
    c_kernels: Sequence[CompiledCKernel]
    invoker: Callable[..., Any]


# {{{ CExecutor

class CExecutor(ExecutorBase):
    """An object connecting a kernel to a :class:`CompiledKernel`
    for execution.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, program, entrypoint, compiler: Optional["CCompiler"] = None):
        """
        :arg kernel: may be a loopy.LoopKernel, a generator returning kernels
            (a warning will be issued if more than one is returned). If the
            kernel has not yet been loop-scheduled, that is done, too, with no
            specific arguments.
        """

        self.compiler = compiler if compiler else CCompiler()
        super().__init__(program, entrypoint)

    def get_invoker_uncached(self, kernel, entrypoint, codegen_result):
        generator = CExecutionWrapperGenerator()
        return generator(kernel, entrypoint, codegen_result)

    def get_wrapper_generator(self):
        return CExecutionWrapperGenerator()

    @memoize_method
    def translation_unit_info(self,
            arg_to_dtype: Optional[Map[str, LoopyType]] = None) -> _KernelInfo:
        t_unit = self.get_typed_and_scheduled_translation_unit(arg_to_dtype)

        from loopy.codegen import generate_code_v2
        codegen_result = generate_code_v2(t_unit)

        dev_code = codegen_result.device_code()
        host_code = codegen_result.host_code()
        all_code = "\n".join([dev_code, "", host_code])

        if t_unit[self.entrypoint].options.write_code:
            output = all_code
            if t_unit[self.entrypoint].options.allow_terminal_colors:
                output = get_highlighted_code(output)

            if t_unit[self.entrypoint].options.write_code is True:
                print(output)
            else:
                with open(t_unit[self.entrypoint].options.write_code, "w") as outf:
                    outf.write(output)

        if t_unit[self.entrypoint].options.edit_code:
            from pytools import invoke_editor
            dev_code = invoke_editor(dev_code, "code.c")
            # update code from editor
            all_code = "\n".join([dev_code, "", host_code])

        c_kernels = []

        from loopy.schedule.tools import get_kernel_arg_info
        kai = get_kernel_arg_info(t_unit[self.entrypoint])
        for dp in codegen_result.device_programs:
            c_kernels.append(CompiledCKernel(
                t_unit[self.entrypoint], dp, kai.passed_names, all_code,
                self.compiler))

        return _KernelInfo(
                t_unit=t_unit,
                c_kernels=c_kernels,
                invoker=self.get_invoker(t_unit, self.entrypoint, codegen_result))

    def __call__(self, *args, **kwargs):
        """
        :returns: ``(None, output)`` the output is a tuple of output arguments
            (arguments that are written as part of the kernel). The order is given
            by the order of kernel arguments. If this order is unspecified
            (such as when kernel arguments are inferred automatically),
            enable :attr:`loopy.Options.return_dict` to make *output* a
            :class:`dict` instead, with keys of argument names and values
            of the returned arrays.
        """
        if __debug__:
            self.check_for_required_array_arguments(kwargs.keys())

        if self.packing_controller is not None:
            kwargs = self.packing_controller(kwargs)

        program_info = self.translation_unit_info(self.arg_to_dtype(kwargs))

        return program_info.invoker(
                program_info.c_kernels, *args, **kwargs)

# }}}

# vim: foldmethod=marker

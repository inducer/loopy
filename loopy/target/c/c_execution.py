from __future__ import division, with_statement, absolute_import

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

import tempfile
import os

from loopy.target.execution import (KernelExecutorBase, _KernelInfo,
                             ExecutionWrapperGeneratorBase, get_highlighted_code)
from pytools import memoize_method
from pytools.py_codegen import (Indentation)
from pytools.prefork import ExecError
from codepy.toolchain import guess_toolchain, ToolchainGuessError, GCCToolchain
from codepy.jit import compile_from_string
import six
import ctypes

import numpy as np

import logging
logger = logging.getLogger(__name__)


class CExecutionWrapperGenerator(ExecutionWrapperGeneratorBase):
    """
    Specialized form of the :class:`ExecutionWrapperGeneratorBase` for
    pyopencl execution
    """

    def __init__(self):
        system_args = ["_lpy_c_kernels"]
        super(CExecutionWrapperGenerator, self).__init__(system_args)

    def python_dtype_str(self, dtype):
        if np.dtype(str(dtype)).isbuiltin:
            return "_lpy_np."+dtype.name
        raise Exception('dtype: {0} not recognized'.format(dtype))

    # {{{ handle non numpy arguements

    def handle_non_numpy_arg(self, gen, arg):
        pass

    # }}}

    # {{{ handle allocation of unspecified arguements

    def handle_alloc(self, gen, arg, kernel_arg, strify, skip_arg_checks):
        """
        Handle allocation of non-specified arguements for C-execution
        """
        from pymbolic import var

        num_axes = len(arg.unvec_shape)
        for i in range(num_axes):
            gen("_lpy_shape_%d = %s" % (i, strify(arg.unvec_shape[i])))

        itemsize = kernel_arg.dtype.numpy_dtype.itemsize
        for i in range(num_axes):
            gen("_lpy_strides_%d = %s" % (i, strify(
                itemsize*arg.unvec_strides[i])))

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
        order = "'C'" if arg.unvec_strides[-1] == 1 else "'F'"

        gen("%(name)s = _lpy_np.empty(%(shape)s, "
                "%(dtype)s, order=%(order)s)"
                % dict(
                    name=arg.name,
                    shape=strify(sym_shape),
                    dtype=self.python_dtype_str(
                        kernel_arg.dtype.numpy_dtype),
                    order=order))

        expected_strides = tuple(
                var("_lpy_expected_strides_%s" % i)
                for i in range(num_axes))

        gen("%s = %s.strides" % (strify(expected_strides), arg.name))

        #check strides
        if not skip_arg_checks:
            strides_check_expr = self.get_strides_check_expr(
                    (strify(s) for s in sym_shape),
                    (strify(s) for s in sym_strides),
                    (strify(s) for s in expected_strides))
            gen("assert %(strides_check)s, "
                    "'Strides of loopy created array %(name)s, "
                    "do not match expected.'" %
                    dict(strides_check=strides_check_expr,
                         name=arg.name,
                         strides=strify(sym_strides)))
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

    def initialize_system_args(self, gen):
        """
        Initializes possibly empty system arguements
        """
        pass

    # {{{ generate invocation

    def generate_invocation(self, gen, kernel_name, args,
            kernel, implemented_data_info):
        gen("for knl in _lpy_c_kernels:")
        with Indentation(gen):
            gen('knl({args})'.format(
                args=", ".join(args)))

    # }}}

    # {{{

    def generate_output_handler(
            self, gen, options, kernel, implemented_data_info):

        from loopy.kernel.data import KernelArgument

        if options.return_dict:
            gen("return None, {%s}"
                    % ", ".join("\"%s\": %s" % (arg.name, arg.name)
                        for arg in implemented_data_info
                        if issubclass(arg.arg_class, KernelArgument)
                        if arg.base_name in kernel.get_written_variables()))
        else:
            out_args = [arg
                    for arg in implemented_data_info
                        if issubclass(arg.arg_class, KernelArgument)
                    if arg.base_name in kernel.get_written_variables()]
            if out_args:
                gen("return None, (%s,)"
                        % ", ".join(arg.name for arg in out_args))
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


class CCompiler(object):
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
                 cc='gcc', cflags='-std=c99 -O3 -fPIC'.split(),
                 ldflags='-shared'.split(), libraries=[],
                 include_dirs=[], library_dirs=[], defines=[],
                 source_suffix='c'):
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
                logger.warn('Default toolchain guessed from python config '
                            'not found, replacing with default GCCToolchain.')
                # this is ugly, but I'm not sure there's a clean way to copy the
                # default args
                self.toolchain = GCCToolchain(
                    cc='gcc',
                    cflags='-std=c99 -O3 -fPIC'.split(),
                    ldflags='-shared'.split(),
                    libraries=[],
                    library_dirs=[],
                    defines=[],
                    undefines=[],
                    source_suffix='c',
                    so_ext='.so',
                    o_ext='.o',
                    include_dirs=[])

        if toolchain is None:
            # copy in all differing values
            diff = {'cc': cc,
                    'cflags': cflags,
                    'ldflags': ldflags,
                    'libraries': libraries,
                    'include_dirs': include_dirs,
                    'library_dirs': library_dirs,
                    'defines': defines}
            # filter empty and those equal to toolchain defaults
            diff = dict((k, v) for k, v in six.iteritems(diff)
                    if v and (not hasattr(self.toolchain, k) or
                              getattr(self.toolchain, k) != v))
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
        c_fname = self._tempname('code.' + self.source_suffix)

        # build object
        _, mod_name, ext_file, recompiled = \
            compile_from_string(self.toolchain, name, code, c_fname,
                                self.tempdir, debug, wait_on_error,
                                debug_recompile, False)

        if recompiled:
            logger.debug('Kernel {0} compiled from source'.format(name))
        else:
            logger.debug('Kernel {0} retrieved from cache'.format(name))

        # and return compiled
        return ctypes.CDLL(ext_file)


class CPlusPlusCompiler(CCompiler):
    """Subclass of CCompiler to invoke a C++ compiler."""

    def __init__(self, toolchain=None,
                 cc='g++', cflags='-std=c++98 -O3 -fPIC'.split(),
                 ldflags=[], libraries=[],
                 include_dirs=[], library_dirs=[], defines=[],
                 source_suffix='cpp'):

        super(CPlusPlusCompiler, self).__init__(
            toolchain=toolchain, cc=cc, cflags=cflags, ldflags=ldflags,
            libraries=libraries, include_dirs=include_dirs,
            library_dirs=library_dirs, defines=defines, source_suffix=source_suffix)


class IDIToCDLL(object):
    """
    A utility class that extracts arguement and return type info from a
    :class:`ImplementedDataInfo` in order to create a :class:`ctype.CDLL`
    """
    def __init__(self, target):
        self.target = target
        self.registry = target.get_dtype_registry().wrapped_registry

    def __call__(self, knl, idi):
        # next loop through the implemented data info to get the arg data
        arg_info = []
        for arg in idi:
            # check if pointer
            pointer = arg.shape
            arg_info.append(self._dtype_to_ctype(arg.dtype, pointer))

        return arg_info

    def _dtype_to_ctype(self, dtype, pointer=False):
        """Map NumPy dtype to equivalent ctypes type."""
        typename = self.registry.dtype_to_ctype(dtype)
        typename = {'unsigned': 'uint'}.get(typename, typename)
        basetype = getattr(ctypes, 'c_' + typename)
        if pointer:
            return ctypes.POINTER(basetype)
        return basetype


class CompiledCKernel(object):
    """
    A CompiledCKernel wraps a loopy kernel, compiling it and loading the
    result as a shared library, and provides access to the kernel as a
    ctypes function object, wrapped by the __call__ method, which attempts
    to automatically map argument types.
    """

    def __init__(self, knl, idi, dev_code, target, comp=None):
        from loopy.target.c import ExecutableCTarget
        assert isinstance(target, ExecutableCTarget)
        self.target = target
        self.name = knl.name
        # get code and build
        self.code = dev_code
        self.comp = comp if comp is not None else CCompiler()
        self.dll = self.comp.build(self.name, self.code)

        # get the function declaration for interface with ctypes
        func_decl = IDIToCDLL(self.target)
        arg_info = func_decl(knl, idi)
        self._fn = getattr(self.dll, self.name)
        # kernels are void by defn.
        self._fn.restype = None
        self._fn.argtypes = [ctype for ctype in arg_info]

    def __call__(self, *args):
        """Execute kernel with given args mapped to ctypes equivalents."""
        args_ = []
        for arg, arg_t in zip(args, self._fn.argtypes):
            if hasattr(arg, 'ctypes'):
                if arg.size == 0:
                    # TODO eliminate unused arguments from kernel
                    arg_ = arg_t(0.0)
                else:
                    arg_ = arg.ctypes.data_as(arg_t)
            else:
                arg_ = arg_t(arg)
            args_.append(arg_)
        self._fn(*args_)


class CKernelExecutor(KernelExecutorBase):
    """An object connecting a kernel to a :class:`CompiledKernel`
    for execution.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, kernel, compiler=None):
        """
        :arg kernel: may be a loopy.LoopKernel, a generator returning kernels
            (a warning will be issued if more than one is returned). If the
            kernel has not yet been loop-scheduled, that is done, too, with no
            specific arguments.
        """

        self.compiler = compiler if compiler else CCompiler()
        super(CKernelExecutor, self).__init__(kernel)

    def get_invoker_uncached(self, kernel, codegen_result):
        generator = CExecutionWrapperGenerator()
        return generator(kernel, codegen_result)

    @memoize_method
    def kernel_info(self, arg_to_dtype_set=frozenset(), all_kwargs=None):
        kernel = self.get_typed_and_scheduled_kernel(arg_to_dtype_set)

        from loopy.codegen import generate_code_v2
        codegen_result = generate_code_v2(kernel)

        dev_code = codegen_result.device_code()
        host_code = codegen_result.host_code()
        all_code = '\n'.join([dev_code, '', host_code])

        if self.kernel.options.write_cl:
            output = all_code
            if self.kernel.options.highlight_cl:
                output = get_highlighted_code(output)

            if self.kernel.options.write_cl is True:
                print(output)
            else:
                with open(self.kernel.options.write_cl, "w") as outf:
                    outf.write(output)

        if self.kernel.options.edit_cl:
            from pytools import invoke_editor
            dev_code = invoke_editor(dev_code, "code.c")
            # update code from editor
            all_code = '\n'.join([dev_code, '', host_code])

        c_kernels = []
        for dp in codegen_result.device_programs:
            c_kernels.append(CompiledCKernel(dp,
                codegen_result.implemented_data_info, all_code, self.kernel.target,
                self.compiler))

        return _KernelInfo(
                kernel=kernel,
                c_kernels=c_kernels,
                implemented_data_info=codegen_result.implemented_data_info,
                invoker=self.get_invoker(kernel, codegen_result))

    # }}}

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

        kwargs = self.packing_controller.unpack(kwargs)

        kernel_info = self.kernel_info(self.arg_to_dtype_set(kwargs))

        return kernel_info.invoker(
                kernel_info.c_kernels, *args, **kwargs)

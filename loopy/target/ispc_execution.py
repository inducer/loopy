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

from loopy.target.c.c_execution import (CKernelExecutor, CCompiler,
    CExecutionWrapperGenerator)
from codepy.toolchain import (GCCLikeToolchain, Toolchain,
                              _guess_toolchain_kwargs_from_python_config)


class ISPCToolchain(GCCLikeToolchain):
    def get_version_tuple(self):
        ver = self.get_version()
        words = ver.split()
        numbers = words[4].split(".")

        result = []
        for n in numbers:
            try:
                result.append(int(n))
            except ValueError:
                # not an integer? too bad.
                break

        return tuple(result)

    def _cmdline(self, files, object=False):
        if object:
            ld_options = []
            link = []
        else:
            ld_options = self.ldflags
            link = ["-L%s" % ldir for ldir in self.library_dirs]
            link.extend(["-l%s" % lib for lib in self.libraries])
        return (
            [self.cc]
            + self.cflags
            + ld_options
            + ["-D%s" % define for define in self.defines]
            + ["-U%s" % undefine for undefine in self.undefines]
            + ["-I%s" % idir for idir in self.include_dirs]
            + files
            + link
            )

    def abi_id(self):
        return Toolchain.abi_id(self) + [self._cmdline([])]

    def with_optimization_level(self, level, debug=False, **extra):
        def remove_prefix(l, prefix):
            return [f for f in l if not f.startswith(prefix)]

        cflags = self.cflags
        for pfx in ["-O", "-g", "-DNDEBUG"]:
            cflags = remove_prefix(cflags, pfx)

        if level == "debug":
            oflags = ["-g"]
        else:
            oflags = ["-O%d" % level, "-DNDEBUG"]

        return self.copy(cflags=cflags + oflags)


class ISPCCompiler(CCompiler):
    """Subclass of Compiler to invoke the ispc compiler."""

    def __init__(self, *args, **kwargs):
        toolchain_defaults = _guess_toolchain_kwargs_from_python_config()
        toolchain_kwargs = dict(
            cc='ispc',
            ldflags=[],
            libraries=toolchain_defaults["libraries"],
            cflags=toolchain_defaults['cflags'],
            include_dirs=toolchain_defaults["include_dirs"],
            library_dirs=toolchain_defaults["library_dirs"],
            so_ext=toolchain_defaults["so_ext"],
            o_ext=toolchain_defaults["o_ext"],
            defines=toolchain_defaults["defines"],
            undefines=toolchain_defaults["undefines"],
            )
        defaults = {'toolchain': ISPCToolchain(**toolchain_kwargs),
                    'source_suffix': 'ispc',
                    'cc': 'ispc',
                    'cflags': '-O3'
                    }

        # update to use any user specified info
        defaults.update(kwargs)

        # and create
        super(ISPCCompiler, self).__init__(
            *args, requires_separate_linkage=True, **defaults)


class ISPCKernelExecutor(CKernelExecutor):
    """An object connecting a kernel to a :class:`CompiledKernel`
    for execution.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, kernel, invoker=CExecutionWrapperGenerator(),
                 compiler=None):
        """
        :arg kernel: may be a loopy.LoopKernel, a generator returning kernels
            (a warning will be issued if more than one is returned). If the
            kernel has not yet been loop-scheduled, that is done, too, with no
            specific arguments.
        """

        self.compiler = compiler if compiler else ISPCCompiler()
        super(ISPCKernelExecutor, self).__init__(
            kernel, invoker=invoker, compiler=self.compiler)

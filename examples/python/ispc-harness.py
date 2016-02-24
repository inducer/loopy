import loopy as lp
import numpy as np
import numpy.linalg as la
import ctypes
import os
from time import time


# {{{ temporary directory

class TemporaryDirectory(object):
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.
    """

    # Yanked from
    # https://hg.python.org/cpython/file/3.3/Lib/tempfile.py

    # Handle mkdtemp raising an exception
    name = None
    _closed = False

    def __init__(self, suffix="", prefix="tmp", dir=None):
        from tempfile import mkdtemp
        self.name = mkdtemp(suffix, prefix, dir)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def cleanup(self, _warn=False):
        import warnings
        if self.name and not self._closed:
            from shutil import rmtree
            try:
                rmtree(self.name)
            except (TypeError, AttributeError) as ex:
                if "None" not in '%s' % (ex,):
                    raise
                self._rmtree(self.name)
            self._closed = True
            if _warn and warnings.warn:
                warnings.warn("Implicitly cleaning up {!r}".format(self))

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def __del__(self):
        # Issue a ResourceWarning if implicit cleanup needed
        self.cleanup(_warn=True)

# }}}


# {{{ build_ispc_shared_lib

def build_ispc_shared_lib(
        cwd, ispc_sources, cxx_sources,
        ispc_options=[], cxx_options=[]):
    from os.path import join

    ispc_source_names = []
    for name, contents in ispc_sources:
        ispc_source_names.append(name)

        with open(join(cwd, name), "w") as srcf:
            srcf.write(contents)

    cxx_source_names = []
    for name, contents in cxx_sources:
        cxx_source_names.append(name)

        with open(join(cwd, name), "w") as srcf:
            srcf.write(contents)

    from subprocess import check_call

    check_call(
            ["ispc",
                "--pic",
                "--opt=force-aligned-memory",
                "--target=avx2-i32x8",
                "-o", "ispc.o"]
            + ispc_options
            + list(ispc_source_names),
            cwd=cwd)

    check_call(
            [
                "g++",
                "-shared", "-fopenmp", "-Wl,--export-dynamic",
                "-fPIC",
                "-oshared.so",
                "ispc.o",
                ]
            + cxx_options
            + list(cxx_source_names),
            cwd=cwd)

# }}}


def cptr_from_numpy(obj):
    ary_intf = getattr(obj, "__array_interface__", None)
    if ary_intf is None:
        raise RuntimeError("no array interface")

    buf_base, is_read_only = ary_intf["data"]
    return ctypes.c_void_p(buf_base + ary_intf.get("offset", 0))


def main():
    with open("tasksys.cpp", "r") as ts_file:
        tasksys_source = ts_file.read()

    from loopy.target.ispc import ISPCTarget
    stream_knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            "z[i] = a*x[i] + y[i]",
            target=ISPCTarget())

    stream_dtype = np.float64
    stream_ctype = ctypes.c_double

    stream_knl = lp.add_and_infer_dtypes(stream_knl, {
        "a": stream_dtype,
        "x": stream_dtype,
        "y": stream_dtype
        })

    stream_knl = lp.assume(stream_knl, "n>0")
    stream_knl = lp.split_iname(stream_knl, "i", 8, inner_tag="l.0")
    stream_knl = lp.split_iname(stream_knl,
            "i_outer", 2**22, outer_tag="g.0")
    stream_knl = lp.preprocess_kernel(stream_knl)
    stream_knl = lp.get_one_scheduled_kernel(stream_knl)
    stream_knl = lp.set_argument_order(stream_knl, "n,a,x,y,z")
    ispc_code, arg_info = lp.generate_code(stream_knl)

    with TemporaryDirectory() as tmpdir:
        build_ispc_shared_lib(
                tmpdir,
                [("stream.ispc", ispc_code)],
                [("tasksys.cpp", tasksys_source)])

        print(ispc_code)
        knl_lib = ctypes.cdll.LoadLibrary(os.path.join(tmpdir, "shared.so"))

        n = 2**28
        a = 5
        x = np.empty(n, dtype=stream_dtype)
        y = np.empty(n, dtype=stream_dtype)
        z = np.empty(n, dtype=stream_dtype)

        nruns = 30
        start_time = time()
        for irun in range(nruns):
            knl_lib.loopy_kernel(
                    ctypes.c_int(n), stream_ctype(a),
                    cptr_from_numpy(x),
                    cptr_from_numpy(y),
                    cptr_from_numpy(z))
        elapsed = time() - start_time

        print(1e-9*3*x.nbytes*nruns/elapsed, "GB/s")

        print(la.norm(z-a*x+y))



if __name__ == "__main__":
    main()

# vim: foldmethod=marker

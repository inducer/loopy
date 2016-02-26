import loopy as lp
import numpy as np
import numpy.linalg as la
import ctypes
import ctypes.util
import os
from time import time
from tempfile import TemporaryDirectory


# {{{ build_ispc_shared_lib

def build_ispc_shared_lib(
        cwd, ispc_sources, cxx_sources,
        ispc_options=[], cxx_options=[],
        ispc_bin="ispc",
        cxx_bin="g++",
        quiet=True):
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

    ispc_cmd = ([ispc_bin,
                "--pic",
                "-o", "ispc.o"]
            + ispc_options
            + list(ispc_source_names))
    if not quiet:
        print(" ".join(ispc_cmd))

    check_call(ispc_cmd, cwd=cwd)

    cxx_cmd = ([
                cxx_bin,
                "-shared", "-Wl,--export-dynamic",
                "-fPIC",
                "-oshared.so",
                "ispc.o",
                ]
            + cxx_options
            + list(cxx_source_names))

    check_call(cxx_cmd, cwd=cwd)

    if not quiet:
        print(" ".join(cxx_cmd))

# }}}


def address_from_numpy(obj):
    ary_intf = getattr(obj, "__array_interface__", None)
    if ary_intf is None:
        raise RuntimeError("no array interface")

    buf_base, is_read_only = ary_intf["data"]
    return buf_base + ary_intf.get("offset", 0)


def cptr_from_numpy(obj):
    return ctypes.c_void_p(address_from_numpy(obj))


# https://github.com/hgomersall/pyFFTW/blob/master/pyfftw/utils.pxi#L172
def empty_aligned(shape, dtype, order='C', n=64):
    '''empty_aligned(shape, dtype='float64', order='C', n=None)
    Function that returns an empty numpy array that is n-byte aligned,
    where ``n`` is determined by inspecting the CPU if it is not
    provided.
    The alignment is given by the final optional argument, ``n``. If
    ``n`` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.empty`.
    '''
    itemsize = np.dtype(dtype).itemsize

    # Apparently there is an issue with numpy.prod wrapping around on 32-bits
    # on Windows 64-bit. This shouldn't happen, but the following code
    # alleviates the problem.
    if not isinstance(shape, (int, np.integer)):
        array_length = 1
        for each_dimension in shape:
            array_length *= each_dimension

    else:
        array_length = shape

    base_ary = np.empty(array_length*itemsize+n, dtype=np.int8)

    # We now need to know how to offset base_ary
    # so it is correctly aligned
    _array_aligned_offset = (n-address_from_numpy(base_ary)) % n

    array = np.frombuffer(
            base_ary[_array_aligned_offset:_array_aligned_offset-n].data,
            dtype=dtype).reshape(shape, order=order)

    return array


def main():
    with open("tasksys.cpp", "r") as ts_file:
        tasksys_source = ts_file.read()

    stream_dtype = np.float64
    stream_ctype = ctypes.c_double
    index_dtype = np.int32

    from loopy.target.ispc import ISPCTarget
    stream_knl = lp.make_kernel(
            "{[i]: 0<=i<n}",
            "z[i] = x[i] + a*y[i]",
            target=ISPCTarget(),
            index_dtype=index_dtype)

    stream_knl = lp.add_and_infer_dtypes(stream_knl, {
        "a": stream_dtype,
        "x": stream_dtype,
        "y": stream_dtype
        })

    stream_knl = lp.assume(stream_knl, "n>0")
    stream_knl = lp.split_iname(stream_knl,
            "i", 2**18, outer_tag="g.0", slabs=(0, 1))
    stream_knl = lp.split_iname(stream_knl, "i_inner", 8, inner_tag="l.0")
    stream_knl = lp.preprocess_kernel(stream_knl)
    stream_knl = lp.get_one_scheduled_kernel(stream_knl)
    stream_knl = lp.set_argument_order(stream_knl, "n,a,x,y,z")
    ispc_code, arg_info = lp.generate_code(stream_knl)

    with TemporaryDirectory() as tmpdir:
        print(ispc_code)

        build_ispc_shared_lib(
                tmpdir,
                [("stream.ispc", ispc_code)],
                [("tasksys.cpp", tasksys_source)],
                cxx_options=["-g", "-fopenmp", "-DISPC_USE_OMP"],
                ispc_options=([
                    #"-g", "--no-omit-frame-pointer",
                    "--target=avx2-i32x8",
                    "--opt=force-aligned-memory",
                    #"--opt=fast-math",
                    #"--opt=disable-fma",
                    ]
                    + (["--addressing=64"] if index_dtype == np.int64 else [])
                    ),
                ispc_bin="/home/andreask/pack/ispc-v1.9.0-linux/ispc",
                quiet=False,
                )

        knl_lib = ctypes.cdll.LoadLibrary(os.path.join(tmpdir, "shared.so"))

        n = 2**28
        a = 5

        align_to = 64
        x = empty_aligned(n, dtype=stream_dtype, n=align_to)
        y = empty_aligned(n, dtype=stream_dtype, n=align_to)
        z = empty_aligned(n, dtype=stream_dtype, n=align_to)

        print(
                hex(address_from_numpy(x)),
                hex(address_from_numpy(y)),
                hex(address_from_numpy(z)))
        assert address_from_numpy(x) % align_to == 0
        assert address_from_numpy(y) % align_to == 0
        assert address_from_numpy(z) % align_to == 0

        nruns = 10

        def call_kernel():
            knl_lib.loopy_kernel(
                    ctypes.c_int(n), stream_ctype(a),
                    cptr_from_numpy(x),
                    cptr_from_numpy(y),
                    cptr_from_numpy(z))

        call_kernel()
        call_kernel()

        start_time = time()

        for irun in range(nruns):
            call_kernel()

        elapsed = time() - start_time

        print(elapsed/nruns)

        print(1e-9 * 3 * x.nbytes * nruns / elapsed, "GB/s")

        assert la.norm(z-a*x+y) < 1e-10


if __name__ == "__main__":
    main()

# vim: foldmethod=marker

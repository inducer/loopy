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


# {{{ numpy address munging

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

# }}}


def transform(knl, vars, stream_dtype):
    vars = [v.strip() for v in vars.split(",")]
    knl = lp.assume(knl, "n>0")
    knl = lp.split_iname(
        knl, "i", 2**18, outer_tag="g.0", slabs=(0, 1))
    knl = lp.split_iname(knl, "i_inner", 8, inner_tag="l.0")

    knl = lp.add_and_infer_dtypes(knl, {
        var: stream_dtype
        for var in vars
        })

    knl = lp.set_argument_order(knl, vars + ["n"])

    return knl


def gen_code(knl):
    knl = lp.preprocess_kernel(knl)
    knl = lp.get_one_scheduled_kernel(knl)
    ispc_code, arg_info = lp.generate_code(knl)

    return ispc_code


NRUNS = 10
ALIGN_TO = 4096
ARRAY_SIZE = 2**28

if 0:
    STREAM_DTYPE = np.float64
    STREAM_CTYPE = ctypes.c_double
else:
    STREAM_DTYPE = np.float32
    STREAM_CTYPE = ctypes.c_float

if 1:
    INDEX_DTYPE = np.int32
    INDEX_CTYPE = ctypes.c_int
else:
    INDEX_DTYPE = np.int64
    INDEX_CTYPE = ctypes.c_longlong


def main():
    with open("tasksys.cpp", "r") as ts_file:
        tasksys_source = ts_file.read()

    def make_knl(name, insn, vars):
        knl = lp.make_kernel(
                "{[i]: 0<=i<n}",
                insn,
                target=lp.ISPCTarget(), index_dtype=INDEX_DTYPE,
                name="stream_"+name+"_tasks")

        knl = transform(knl, vars, STREAM_DTYPE)
        return knl

    init_knl = make_knl("init", """
                a[i] = 1
                b[i] = 2
                c[i] = 0
                """, "a,b,c")
    triad_knl = make_knl("triad", """
            a[i] = b[i] + scalar * c[i]
            """, "a,b,c,scalar")

    with TemporaryDirectory() as tmpdir:
        ispc_code = gen_code(init_knl) + gen_code(triad_knl)
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
                    + (["--addressing=64"] if INDEX_DTYPE == np.int64 else [])
                    ),
                ispc_bin="/home/andreask/pack/ispc-v1.9.0-linux/ispc",
                quiet=False,
                )

        knl_lib = ctypes.cdll.LoadLibrary(os.path.join(tmpdir, "shared.so"))

        scalar = 5

        a = empty_aligned(ARRAY_SIZE, dtype=STREAM_DTYPE, n=ALIGN_TO)
        b = empty_aligned(ARRAY_SIZE, dtype=STREAM_DTYPE, n=ALIGN_TO)
        c = empty_aligned(ARRAY_SIZE, dtype=STREAM_DTYPE, n=ALIGN_TO)

        print(
                hex(address_from_numpy(a)),
                hex(address_from_numpy(b)),
                hex(address_from_numpy(c)))
        assert address_from_numpy(a) % ALIGN_TO == 0
        assert address_from_numpy(b) % ALIGN_TO == 0
        assert address_from_numpy(c) % ALIGN_TO == 0

        knl_lib.stream_init_tasks(
                cptr_from_numpy(a),
                cptr_from_numpy(b),
                cptr_from_numpy(c),
                INDEX_CTYPE(ARRAY_SIZE),
                )

        def call_kernel():
            knl_lib.stream_triad_tasks(
                    cptr_from_numpy(a),
                    cptr_from_numpy(b),
                    cptr_from_numpy(c),
                    STREAM_CTYPE(scalar),
                    INDEX_CTYPE(ARRAY_SIZE),
                    )

        call_kernel()
        call_kernel()

        start_time = time()

        for irun in range(NRUNS):
            call_kernel()

        elapsed = time() - start_time

        print(elapsed/NRUNS)

        print(1e-9*3*a.nbytes*NRUNS/elapsed, "GB/s")

        assert la.norm(a-b+scalar*c, np.inf) < np.finfo(STREAM_DTYPE).eps * 10


if __name__ == "__main__":
    main()

# vim: foldmethod=marker

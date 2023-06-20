import loopy as lp
import numpy as np
import numpy.linalg as la
import ctypes
import ctypes.util
import os
from time import time
from tempfile import TemporaryDirectory

from loopy.tools import (empty_aligned, address_from_numpy,
        build_ispc_shared_lib, cptr_from_numpy)
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


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
    codegen_result = lp.generate_code_v2(knl)

    return codegen_result.device_code() + "\n" + codegen_result.host_code()


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
    this_dir = os.path.dirname(__file__)
    with open(os.path.join(this_dir, "tasksys.cpp")) as ts_file:
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
                    "--opt=disable-loop-unroll",
                    #"--opt=fast-math",
                    #"--opt=disable-fma",
                    ]
                    + (["--addressing=64"] if INDEX_DTYPE == np.int64 else [])
                    ),
                #ispc_bin="/home/andreask/pack/ispc-v1.9.0-linux/ispc",
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

        for _irun in range(NRUNS):
            call_kernel()

        elapsed = time() - start_time

        print(elapsed/NRUNS)

        print(1e-9*3*a.nbytes*NRUNS/elapsed, "GB/s")

        assert la.norm(a-b+scalar*c, np.inf) < np.finfo(STREAM_DTYPE).eps * 10


if __name__ == "__main__":
    main()

# vim: foldmethod=marker

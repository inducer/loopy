from __future__ import division

import numpy as np
import loopy as lp

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests




def test_owed_barriers(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            "{[i]: 0<=i<100}",
            [
                "[i:l.0] <float32> z[i] = a[i]"
                ],
            [lp.ArrayArg("a", np.float32, shape=(100,))]
            )

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen)

    for gen_knl in kernel_gen:
        compiled = lp.CompiledKernel(ctx, gen_knl)
        print compiled.code




def test_wg_too_small(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            "{[i]: 0<=i<100}",
            [
                "[i:l.0] <float32> z[i] = a[i]"
                ],
            [lp.ArrayArg("a", np.float32, shape=(100,))],
            local_sizes={0: 16})

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen)

    for gen_knl in kernel_gen:
        try:
            compiled = lp.CompiledKernel(ctx, gen_knl)
        except RuntimeError, e:
            assert "implemented and desired" in str(e)
            pass # expected!
        else:
            assert False # expecting an error




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

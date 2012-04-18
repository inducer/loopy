from __future__ import division

import numpy as np
import loopy as lp
import pyopencl as cl

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

__all__ = ["pytest_generate_tests",
    "cl" # 'cl.create_some_context'
    ]




def test_owed_barriers(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            "{[i]: 0<=i<100}",
            [
                "[i:l.0] <float32> z[i] = a[i]"
                ],
            [lp.GlobalArg("a", np.float32, shape=(100,))]
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
            [lp.GlobalArg("a", np.float32, shape=(100,))],
            local_sizes={0: 16})

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen)

    for gen_knl in kernel_gen:
        try:
            lp.CompiledKernel(ctx, gen_knl)
        except RuntimeError, e:
            assert "implemented and desired" in str(e)
            pass # expected!
        else:
            assert False # expecting an error




def test_multi_cse(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            "{[i]: 0<=i<100}",
            [
                "[i] <float32> z[i] = a[i] + a[i]**2"
                ],
            [lp.GlobalArg("a", np.float32, shape=(100,))],
            local_sizes={0: 16})

    knl = lp.split_dimension(knl, "i", 16, inner_tag="l.0")
    knl = lp.add_prefetch(knl, "a", [])

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen)

    for gen_knl in kernel_gen:
        compiled = lp.CompiledKernel(ctx, gen_knl)
        print compiled.code





def test_stencil(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j]: 0<= i,j < 32}",
            [
                "[i] <float32> z[i,j] = -2*a[i,j]"
                    " + a[i,j-1]"
                    " + a[i,j+1]"
                    " + a[i-1,j]"
                    " + a[i+1,j]"
                ],
            [
                lp.GlobalArg("a", np.float32, shape=(32,32,))
                ])


    def variant_1(knl):
        knl = lp.add_prefetch(knl, "a", [0, 1])
        return knl

    def variant_2(knl):
        knl = lp.split_dimension(knl, "i", 16, outer_tag="g.1", inner_tag="l.1")
        knl = lp.split_dimension(knl, "j", 16, outer_tag="g.0", inner_tag="l.0")
        knl = lp.add_prefetch(knl, "a", ["i_inner", "j_inner"])
        return knl

    #for variant in [variant_1, variant_2]:
    for variant in [variant_2]:
        kernel_gen = lp.generate_loop_schedules(variant(knl),
                loop_priority=["i_outer", "i_inner_0", "j_0"])
        kernel_gen = lp.check_kernels(kernel_gen)

        for knl in kernel_gen:
            print lp.generate_code(knl)




def test_eq_constraint(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j]: 0<= i,j < 32}",
            [
                "a[i] = b[i]"
                ],
            [
                lp.GlobalArg("a", np.float32, shape=(1000,)),
                lp.GlobalArg("b", np.float32, shape=(1000,))
                ])

    knl = lp.split_dimension(knl, "i", 16, outer_tag="g.0")
    knl = lp.split_dimension(knl, "i_inner", 16, outer_tag=None, inner_tag="l.0")

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen)

    for knl in kernel_gen:
        print lp.generate_code(knl)




def test_argmax(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    order = "C"

    n = 10000

    knl = lp.make_kernel(ctx.devices[0],
            "{[i]: 0<=i<%d}" % n,
            [
                "<> result = argmax_float32(i, fabs(a[i]))",
                "max_idx = result.index",
                "max_val = result.value",
                ],
            [
                lp.GlobalArg("a", dtype, shape=(n,), order=order),
                lp.GlobalArg("max_idx", np.int32, shape=(), order=order),
                lp.GlobalArg("max_val", dtype, shape=(), order=order),
                ])

    a = np.random.randn(10000).astype(dtype)
    cknl = lp.CompiledKernel(ctx, knl)
    evt, (max_idx, max_val) = cknl(queue, a=a)
    assert max_val == np.max(np.abs(a))
    assert max_idx == np.where(np.abs(a)==max_val)[-1]





if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

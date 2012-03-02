
from __future__ import division

import numpy as np
import pyopencl as cl
import loopy as lp

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests



def test_nbody(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            "[N] -> {[i,j,k]: 0<=i,j<N and 0<=k<3 }",
           [
            "axdist(k) := x[i,k]-x[j,k]",
            "invdist := rsqrt(sum_float32(k, axdist(k)**2))",
            "pot[i] = sum_float32(j, if(i != j, invdist, 0))",
            ],
            [
            lp.ArrayArg("x", dtype, shape="N,3", order="C"),
            lp.ArrayArg("pot", dtype, shape="N", order="C"),
            lp.ScalarArg("N", np.int32),
            ],
             name="nbody", assumptions="N>=1")

    seq_knl = knl

    def variant_1(knl):
        knl = lp.split_dimension(knl, "i", 256,
                outer_tag="g.0", inner_tag="l.0",
                slabs=(0,1))
        knl = lp.split_dimension(knl, "j", 256, slabs=(0,1))
        return knl, []

    def variant_cpu(knl):
        knl = lp.split_dimension(knl, "i", 1024,
                outer_tag="g.0", slabs=(0,1))
        return knl, []

    def variant_gpu(knl):
        knl = lp.split_dimension(knl, "i", 256,
                outer_tag="g.0", inner_tag="l.0", slabs=(0,1))
        knl = lp.split_dimension(knl, "j", 256, slabs=(0,1))
        knl = lp.add_prefetch(knl, "x[i,k]", ["k"], default_tag=None)
        knl = lp.add_prefetch(knl, "x[j,k]", ["j_inner", "k"])
        return knl, ["j_outer", "j_inner"]

    n = 100

    for variant in [variant_gpu]:
        variant_knl, loop_prio = variant(knl)
        kernel_gen = lp.generate_loop_schedules(variant_knl,
                loop_priority=loop_prio)
        kernel_gen = lp.check_kernels(kernel_gen, dict(N=n))

        lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
                op_count=4*n**2*1e-9, op_label="GOps/s",
                parameters={"N": n}, print_ref_code=True)




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

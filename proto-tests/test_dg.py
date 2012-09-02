from __future__ import division

import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as cl_random
import loopy as lp

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests




1/0 # unfinished



def test_dg_matrix_mul(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    Np = 84
    Np_padded = 96
    K = get_suitable_size(ctx)*4
    dim = 3
    num_flds = 2
    use_images = False

    from pymbolic import var
    fld = var("fld")
    matrix_names = ["d%d" % i for i in range(dim)]
    i, j, k = [var(s) for s in "i j k".split()]

    fld_strides = (1, Np_padded)

    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j< %d and 0<=k<%d}" % (Np, K),
            [
                (var(mn+"fld%d" % ifld)[i, k], 
                    var(mn)[i, j]*var("fld%d" % ifld)[j, k])
                for mn in matrix_names
                for ifld in range(num_flds)
                ],
            ([lp.ImageArg(mn, dtype, 2) for mn in matrix_names]
            if use_images else
            [lp.ArrayArg(mn, dtype, shape=(Np, Np), order="C") for mn in matrix_names])
            + [lp.ArrayArg("fld%d" % ifld, dtype,
                strides=fld_strides)
                for ifld in range(num_flds)
                ]
            + [lp.ArrayArg(mn+"fld%d" % ifld, dtype,
                strides=fld_strides)
                for ifld in range(num_flds)
                for mn in matrix_names
                ],
            name="dg_matmul")

    #ilp = 4
    knl = lp.split_iname(knl, "i", 30, 32, outer_tag="g.0", inner_tag="l.0")
    knl = lp.split_iname(knl, "k", 16, outer_tag="g.1", inner_tag="l.1")
    #knl = lp.split_iname(knl, "k_inner", 16, outer_tag="ilp", inner_tag="l.1")

    assert Np % 2 == 0
    #knl = lp.split_iname(knl, "j", Np//2)
    #knl = lp.split_iname(knl, "k", 32)

    #for mn in matrix_names:
        #knl = lp.add_prefetch(knl, mn, ["j", "i_inner"])
    for ifld in range(num_flds):
        knl = lp.add_prefetch(knl, 'fld%d' % ifld,
                #["k_inner_outer", "k_inner_inner", "j"])
                ["k_inner", "j"])
    assert knl.get_problems({})[0] <= 2

    kernel_gen = list(lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl))[:1]

    matrices = [
            make_well_conditioned_dev_matrix(queue, Np, dtype=dtype, order="C",
                ran_factor=0)
            for mn in matrix_names]
    flds = [
            make_well_conditioned_dev_matrix(queue, (Np_padded, K), dtype=dtype, order="F")
            for ifld in range(num_flds)]
    outputs = [cl_array.empty_like(flds[0])
            for ifld in range(num_flds)
            for mn in matrix_names]

    ref_soln = [np.dot(mat.get(), fld.get()[:Np]) 
            for fld in flds
            for mat in matrices]

    if use_images:
        mat_images = [
                cl.image_from_array(ctx, mat.get(), 1) for mat in matrices]

    def launcher(kernel, gsize, lsize, check):
        if use_images:
            args = mat_images
        else:
            args = [mat.data for mat in matrices]

        args = args + [fld.data for fld in flds] + [out.data for out in outputs]
        kwargs = dict(g_times_l=True)
        evt = kernel(queue, gsize(), lsize(), *args, g_times_l=True)

        if check:
            for out, ref in zip(outputs, ref_soln):
                check_error(ref, out.get()[:Np])

        return evt

    lp.drive_timing_run(kernel_gen, queue, launcher, num_flds*dim*2*(Np**2)*K)





if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

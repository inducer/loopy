from __future__ import division

import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests




def test_laplacian_stiffness(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    # FIXME: make dim-independent

    dim = 2

    Nq = 40 # num. quadrature points
    Nc = 1000 # num. cells
    Nb = 20 # num. basis functions

    # K - run-time symbolic

    from pymbolic import var
    Nc_sym = var("Nc")

    knl = lp.make_kernel(ctx.devices[0],
            "[Nc] -> {[K,i,j,q]: 0<=K<Nc and 0<=i,j<%(Nb)d and 0<=q<%(Nq)d}" 
            % dict(Nb=Nb, Nq=Nq),
            [
                "SUBST: dPsi(a,dxi) = jacInv[K,q,0,dxi] * DPsi[a,q,0] "
                    "+ jacInv[K,q,1,dxi] * DPsi[a,q,1]",
                "A[K, i, j] = sum_float32(q, w[q] * jacDet[K,q] * ("
                    "dPsi(0,0)*dPsi(1,0) + dPsi(0,1)*dPsi(1,1)))"

                ],
            [
            lp.ArrayArg("jacInv", dtype, shape=(Nc_sym, Nq, dim, dim), order=order),
            lp.ConstantArrayArg("DPsi", dtype, shape=(Nb, Nq, dim), order=order),
            lp.ArrayArg("jacDet", dtype, shape=(Nc_sym, Nq), order=order),
            lp.ConstantArrayArg("w", dtype, shape=(Nq, dim), order=order),
            lp.ArrayArg("A", dtype, shape=(Nc_sym, Nb, Nb), order=order),
            lp.ScalarArg("Nc",  np.int32, approximately=1000),
            ],
            name="semlap", assumptions="Nc>=1")

    seq_knl = knl

    knl = lp.split_dimension(knl, "K", 16, outer_tag="g.0", slabs=(0,1))
    knl = lp.split_dimension(knl, "K_inner", 4, inner_tag="ilp")
    knl = lp.tag_dimensions(knl, {"i": "l.0", "j": "l.1"})
    knl = lp.add_prefetch(knl, 'jacInv', ["Kii", "Kio", "q", "x", "y"],
            uni_template="jacInv[Kii + 4*Kio +16*Ko,q,x,y]")

    kernel_gen = lp.generate_loop_schedules(knl,
            loop_priority=["K", "i", "j"])
    kernel_gen = lp.check_kernels(kernel_gen, dict(Nc=Nc))

    lp.auto_test_vs_seq(seq_knl, ctx, kernel_gen,
            op_count=0, op_label="GFlops",
            parameters={"Nc": Nc}, print_seq_code=True,
            timing_rounds=30)



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])


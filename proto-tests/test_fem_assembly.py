import numpy as np
import pyopencl as cl  # noqa
import loopy as lp

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests  # noqa


def test_laplacian_stiffness(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    dim = 2  # (baked into code)

    Nq = 40  # num. quadrature points (baked into code)  # noqa
    Nb = 20  # num. basis functions (baked into code)  # noqa
    Nc = 100  # num. cells (run-time symbolic)  # noqa

    from pymbolic import var
    Nc_sym = var("Nc")  # noqa

    knl = lp.make_kernel(ctx.devices[0],
            "[Nc] -> {[K,i,j,q, dx_axis, ax_b]: 0<=K<Nc and 0<=i,j<%(Nb)d and 0<=q<%(Nq)d "  # noqa
            "and 0<= dx_axis, ax_b < %(dim)d}"
            % dict(Nb=Nb, Nq=Nq, dim=dim),
            [
                "dPsi(ij, dxi) := sum_float32(@ax_b,"
                    "  jacInv[ax_b,dxi,K,q] * DPsi[ax_b,ij,q])",  # noqa
                "A[K, i, j] = sum_float32(q, w[q] * jacDet[K,q] * ("
                    "sum_float32(dx_axis, dPsi$one(i,dx_axis)*dPsi$two(j,dx_axis))))"
                ],
            [
            lp.GlobalArg("jacInv", dtype, shape=(dim, dim, Nc_sym, Nq), order=order),
            lp.ConstantArg("DPsi", dtype, shape=(dim, Nb, Nq), order=order),
            lp.GlobalArg("jacDet", dtype, shape=(Nc_sym, Nq), order=order),
            lp.ConstantArg("w", dtype, shape=(Nq,), order=order),
            lp.GlobalArg("A", dtype, shape=(Nc_sym, Nb, Nb), order=order),
            lp.ValueArg("Nc",  np.int32, approximately=1000),
            ],
            name="lapquad", assumptions="Nc>=1")

    knl = lp.tag_inames(knl, dict(ax_b="unr"))
    seq_knl = knl

    def variant_fig31(knl):
        # This (mostly) reproduces Figure 3.1.

        knl = lp.tag_inames(knl, {"dx_axis": "unr"})
        return knl, ["K", "i", "j", "q", "ax_b_insn"]

    def variant_pg4(knl):
        # This (mostly) reproduces the unlabeled code snippet on pg. 4.

        knl = lp.tag_inames(knl, {"dx_axis": "unr"})
        Ncloc = 16  # noqa
        knl = lp.split_iname(knl, "K", Ncloc,
                outer_iname="Ko", inner_iname="Kloc")
        return knl, ["Ko", "Kloc", "i", "j", "q", "ax_b_insn"]

    def variant_fig32(knl):
        # This (mostly) reproduces Figure 3.2.

        Ncloc = 16  # noqa
        knl = lp.split_iname(knl, "K", Ncloc,
                outer_iname="Ko", inner_iname="Kloc")
        knl = lp.precompute(knl, "dPsi", np.float32, ["i", "q", "dx_axis"],
                default_tag=None)
        knl = lp.tag_inames(knl, {"dx_axis": "unr", "dxi": "unr"})
        return knl, ["Ko", "Kloc", "dPsi_q", "ij", "i", "j", "q", "ax_b_insn"]

    def variant_fig33(knl):
        # This is meant to (mostly) reproduce Figure 3.3.

        Ncloc = 16  # noqa
        knl = lp.split_iname(knl, "K", Ncloc,
                outer_iname="Ko", inner_iname="Kloc")
        knl = lp.precompute(knl, "dPsi$one", np.float32, ["dx_axis"], default_tag=None)  # noqa
        knl = lp.tag_inames(knl, {"j": "ilp.seq"})

        return knl, ["Ko", "Kloc"]

    def variant_simple_gpu(knl):
        # This is a simple GPU-ish variant.

        # It's not the same thing as Matt's code, but I'll need some more time
        # to reverse-engineer what is going on there. Some discussion might
        # help, too. :)

        knl = lp.tag_inames(knl, {"dx_axis": "unr"})
        Ncloc = 16  # noqa
        knl = lp.split_iname(knl, "K", Ncloc,
                outer_iname="Ko", inner_iname="Kloc",
                outer_tag="g.0")
        knl = lp.tag_inames(knl, {"i": "l.1", "j": "l.0"})
        return knl, ["K", "i", "j", "q", "ax_b_insn"]

    def variant_simple_gpu_prefetch(knl):
        # This adds prefetching to the GPU variant above.

        # In this variant (on my machine), loopy makes a silly choice
        # for the upper bound of Kloc (it uses Nc). I'll investigate and
        # fix that. (FIXME)

        knl = lp.tag_inames(knl, {"dx_axis": "unr"})
        Ncloc = 16  # noqa
        knl = lp.split_iname(knl, "K", Ncloc,
                outer_iname="Ko", inner_iname="Kloc",
                outer_tag="g.0")
        knl = lp.tag_inames(knl, {"i": "l.1", "j": "l.0"})
        knl = lp.add_prefetch(knl, "w", ["q"], default_tag="l.auto")
        knl = lp.add_prefetch(knl, "DPsi", [0, 1, 2], default_tag="l.auto")
        knl = lp.add_prefetch(knl, "jacInv", [0, 1, 3], default_tag="l.auto")
        knl = lp.add_prefetch(knl, "jacDet", [1], default_tag="l.auto")
        return knl, ["K", "i", "j", "q", "ax_b_insn"]

    # Plug in variant name here
    #                        |
    #                        v
    for variant in [variant_fig33]:
        var_knl, loop_prio = variant(knl)
        kernel_gen = lp.generate_loop_schedules(var_knl,
                loop_priority=loop_prio)
        kernel_gen = lp.check_kernels(kernel_gen, dict(Nc=Nc))

        #print lp.preprocess_kernel(var_knl)

        lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
                op_count=0, op_label="GFlops",
                parameters={"Nc": Nc}, print_ref_code=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

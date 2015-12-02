"""gNUMA differentiation kernel, wrapped up as a test."""

from __future__ import division

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import pytest
import loopy as lp
import pyopencl as cl  # noqa
import sys

pytestmark = pytest.mark.importorskip("fparser")


@pytest.mark.parametrize("Nq", [7])
@pytest.mark.parametrize("ilp_multiple", [1, 2])
def test_gnuma_horiz_kernel(ctx_factory, ilp_multiple, Nq):
    ctx = ctx_factory()

    filename = "strongVolumeKernels.f90"
    with open(filename, "r") as sourcef:
        source = sourcef.read()

    source = source.replace("datafloat", "real*4")

    hsv_r, hsv_s = [
           knl for knl in lp.parse_fortran(source, filename, auto_dependencies=False)
           if "KernelR" in knl.name or "KernelS" in knl.name
           ]
    hsv_r = lp.tag_instructions(hsv_r, "rknl")
    hsv_s = lp.tag_instructions(hsv_s, "sknl")
    hsv = lp.fuse_kernels([hsv_r, hsv_s], ["_r", "_s"])
    #hsv = hsv_s

    from gnuma_loopy_transforms import (
          fix_euler_parameters,
          set_q_storage_format,
          prefetch_and_set_D_storage_format)

    hsv = lp.fix_parameters(hsv, Nq=Nq)
    hsv = lp.set_loop_priority(hsv, "e,k,j,i")
    hsv = lp.tag_inames(hsv, dict(e="g.0", j="l.1", i="l.0"))
    hsv = lp.assume(hsv, "elements >= 1")

    hsv = fix_euler_parameters(hsv, p_p0=1, p_Gamma=1.4, p_R=1)
    for name in ["Q", "rhsQ"]:
        hsv = set_q_storage_format(hsv, name)

    hsv = prefetch_and_set_D_storage_format(hsv)
    #hsv = lp.add_prefetch(hsv, "volumeGeometricFactors")

    ref_hsv = hsv

    # turn the first reads into subst rules
    local_prep_var_names = set()
    for insn in lp.find_instructions(hsv, "tag:local_prep"):
        (assignee, _), = insn.assignees_and_indices()
        local_prep_var_names.add(assignee)
        hsv = lp.assignment_to_subst(hsv, assignee)

    # precompute fluxes
    hsv = lp.assignment_to_subst(hsv, "JinvD_r")
    hsv = lp.assignment_to_subst(hsv, "JinvD_s")

    r_fluxes = lp.find_instructions(hsv, "tag:compute_fluxes and tag:rknl")
    s_fluxes = lp.find_instructions(hsv, "tag:compute_fluxes and tag:sknl")

    ilp_multiple = 1
    if ilp_multiple > 1:
        hsv = lp.split_iname(hsv, "k", 2, inner_tag="ilp")
        ilp_inames = ("k_inner",)
        flux_ilp_inames = ("kk",)
    else:
        ilp_inames = ()
        flux_ilp_inames = ()

    rtmps = []
    stmps = []

    flux_store_idx = 0

    for rflux_insn, sflux_insn in zip(r_fluxes, s_fluxes):
        for knl_tag, insn, flux_inames, tmps, flux_precomp_inames in [
                  ("rknl", rflux_insn, ("j", "n",), rtmps, ("jj", "ii",)),
                  ("sknl", sflux_insn, ("i", "n",), stmps, ("ii", "jj",)),
                  ]:
            (flux_var, _), = insn.assignees_and_indices()
            print(insn)

            reader, = lp.find_instructions(hsv,
                  "tag:{knl_tag} and reads:{flux_var}"
                  .format(knl_tag=knl_tag, flux_var=flux_var))

            hsv = lp.assignment_to_subst(hsv, flux_var)

            flux_store_name = "flux_store_%d" % flux_store_idx
            flux_store_idx += 1
            tmps.append(flux_store_name)

            hsv = lp.precompute(hsv, flux_var+"_subst", flux_inames + ilp_inames,
                temporary_name=flux_store_name,
                precompute_inames=flux_precomp_inames + flux_ilp_inames,
                default_tag=None)
            if flux_var.endswith("_s"):
                hsv = lp.tag_data_axes(hsv, flux_store_name, "N0,N1,N2?")
            else:
                hsv = lp.tag_data_axes(hsv, flux_store_name, "N1,N0,N2?")

            n_iname = "n_"+flux_var.replace("_r", "").replace("_s", "")
            if n_iname.endswith("_0"):
                n_iname = n_iname[:-2]
            hsv = lp.rename_iname(hsv, "n", n_iname, within="id:"+reader.id,
                  existing_ok=True)

    hsv = lp.tag_inames(hsv, dict(ii="l.0", jj="l.1"))
    for iname in flux_ilp_inames:
        hsv = lp.tag_inames(hsv, {iname: "ilp"})

    hsv = lp.alias_temporaries(hsv, rtmps)
    hsv = lp.alias_temporaries(hsv, stmps)

    for prep_var_name in local_prep_var_names:
        if prep_var_name.startswith("Jinv") or "_s" in prep_var_name:
            continue
        hsv = lp.precompute(hsv,
            lp.find_one_rule_matching(hsv, prep_var_name+"_*subst*"))

    hsv = lp.add_prefetch(hsv, "Q[ii,jj,k,:,:,e]", sweep_inames=ilp_inames)
    hsv = lp.buffer_array(hsv, "rhsQ", ilp_inames,
          fetch_bounding_box=True, default_tag="for",
          init_expression="0")

    # buffer axes need to be vectorized in order for this to work
    hsv = lp.tag_data_axes(hsv, "rhsQ_buf", "c?,vec,c")
    hsv = lp.tag_data_axes(hsv, "Q_fetch", "c?,vec,c")
    hsv = lp.tag_data_axes(hsv, "D_fetch", "f,f")
    hsv = lp.tag_inames(hsv,
            {"Q_dim_k": "unr", "rhsQ_init_k": "unr", "rhsQ_store_k": "unr"},
            ignore_nonexistent=True)
    hsv = lp.tag_inames(hsv, dict(
          rhsQ_init_field_inner="vec", rhsQ_store_field_inner="vec",
          rhsQ_init_field_outer="unr", rhsQ_store_field_outer="unr",
          Q_dim_field_inner="vec",
          Q_dim_field_outer="unr"))
    hsv = lp.collect_common_factors_on_increment(hsv, "rhsQ_buf",
          vary_by_axes=(0,) if ilp_multiple > 1 else ())

    if 0:
        print("OPS")
        op_poly = lp.get_op_poly(hsv)
        print(lp.stringify_stats_mapping(op_poly))

        print("MEM")
        gmem_poly = lp.sum_mem_access_to_bytes(lp.get_gmem_access_poly(hsv))
        print(lp.stringify_stats_mapping(gmem_poly))

    # FIXME
    if 0:
        hsv = lp.set_options(hsv, cl_build_options=[
            "-cl-no-signed-zeros",
            "-cl-fast-relaxed-math",
            "-cl-mad-enable",
            "-cl-uniform-work-group-size",
            ])

    hsv = hsv.copy(name="horizontalStrongVolumeKernel")

    lp.auto_test_vs_ref(ref_hsv, ctx, hsv, parameters=dict(elements=300),
            do_check=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: foldmethod=marker
